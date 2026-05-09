"""Stage 4F-2 manually gated real IBKR paper connection preflight.

This module accepts injected factories and clients only. It does not import
IBKR runtime libraries, read config.py, mutate state, write ledgers, submit or
cancel orders, request market data, or qualify contracts.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from algo_trader_unified.core.ibkr_paper_order_mapper import (
    IbkrPaperConfig,
    validate_ibkr_paper_readonly_config,
)
from algo_trader_unified.core.paper_broker_adapter import sanitize_json_safe


CLIENT_ID_RECOMMENDATION = (
    "If connection failed, check whether this client_id is already in use by a "
    "zombie process or another active daemon."
)
ORDERED_NEXT_STEPS = [
    "Run this preflight manually against paper Gateway before Stage 4F-3.",
    CLIENT_ID_RECOMMENDATION,
    "Keep submission, cancellation, market data, and contract qualification out of Stage 4F-2.",
]
DO_NOT_DO_YET = [
    "Do not enable paper order submission.",
    "Do not submit or cancel orders.",
    "Do not request market data or qualify contracts.",
    "Do not wire this preflight into scheduler, daemon, lifecycle jobs, or live trading.",
]


def build_ibkr_paper_connection_preflight_report(
    *,
    config: dict[str, Any] | IbkrPaperConfig,
    ib_factory: Callable[[], Any],
    client_factory: Callable[[Any, IbkrPaperConfig], Any],
    allow_real_ibkr: bool = False,
    connect_timeout_seconds: float = 10.0,
    now_provider: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe Stage 4F-2 real-paper connection preflight report."""

    generated_at = _iso_now(now_provider)
    errors: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []
    validated_config: IbkrPaperConfig | None = None
    validation_reason = "valid"

    raw_config = _config_input_dict(config)
    try:
        validated_config = validate_ibkr_paper_readonly_config(raw_config)
        config_payload = _config_payload(validated_config, True, validation_reason)
    except Exception as exc:  # noqa: BLE001
        validation_reason = _exception_reason(exc)
        errors.append(f"validate_ibkr_paper_readonly_config failed: {validation_reason}")
        blockers.append("IBKR paper connection config is not valid.")
        config_payload = _config_payload(raw_config, False, validation_reason)

    connection = {
        "allow_real_ibkr": allow_real_ibkr is True,
        "attempted": False,
        "connected": False,
        "paper_mode": False,
        "disconnected": False,
        "timeout_seconds": connect_timeout_seconds,
        "reason": None,
    }
    readonly_checks = {
        "current_time_ok": False,
        "account_snapshot_ok": False,
        "open_orders_ok": False,
        "positions_ok": False,
    }
    account_snapshot = {
        "available": False,
        "net_liquidation": None,
        "available_funds": None,
        "buying_power": None,
        "timestamp": None,
        "failure_reason": None,
    }
    open_orders = {"available": False, "count": 0, "failure_reason": None}
    positions = {"available": False, "count": 0, "failure_reason": None}

    client = None
    connect_attempted = False
    reads_allowed = False

    if connect_timeout_seconds <= 0:
        errors.append("connect_timeout_seconds must be positive")
        blockers.append("Connect timeout must be positive.")
    elif not allow_real_ibkr:
        reason = "real IBKR connection not attempted because allow_real_ibkr is False"
        connection["reason"] = reason
        blockers.append("Real IBKR connection was not explicitly allowed.")
    elif validated_config is None:
        connection["reason"] = "config validation failed before real IBKR connection"
    else:
        try:
            ib = ib_factory()
        except Exception as exc:  # noqa: BLE001
            reason = _exception_reason(exc)
            errors.append(f"ib_factory failed: {reason}")
            blockers.append("IBKR paper IB factory failed.")
            connection["reason"] = reason
        else:
            try:
                client = client_factory(ib, validated_config)
            except Exception as exc:  # noqa: BLE001
                reason = _exception_reason(exc)
                errors.append(f"client_factory failed: {reason}")
                blockers.append("IBKR paper read-only client factory failed.")
                connection["reason"] = reason
            else:
                connect_attempted = True
                connection["attempted"] = True
                try:
                    connect_status = _call_with_timeout(
                        client.connect,
                        connect_timeout_seconds,
                    )
                    _merge_connection_status(connection, connect_status)
                    if not connection["connected"]:
                        reason = connection.get("reason") or "connect did not establish a connection"
                        connection["reason"] = reason
                        errors.append(f"connect failed: {reason}")
                        blockers.append("IBKR paper connection did not establish.")
                    else:
                        status = client.get_connection_status()
                        _merge_connection_status(connection, status)
                        if not connection["connected"] or not connection["paper_mode"]:
                            reason = (
                                connection.get("reason")
                                or "client did not confirm connected paper mode"
                            )
                            connection["reason"] = reason
                            errors.append(f"connection status failed: {reason}")
                            blockers.append("IBKR paper connection status was not confirmed.")
                        else:
                            reads_allowed = True
                except Exception as exc:  # noqa: BLE001
                    reason = _exception_reason(exc)
                    errors.append(f"connect failed: {reason}")
                    blockers.append("IBKR paper connection failed.")
                    connection["reason"] = reason

                if reads_allowed:
                    _run_readonly_checks(
                        client=client,
                        readonly_checks=readonly_checks,
                        account_snapshot=account_snapshot,
                        open_orders=open_orders,
                        positions=positions,
                        errors=errors,
                        blockers=blockers,
                    )

    if client is not None and connect_attempted:
        try:
            disconnect_status = client.disconnect()
            connection["disconnected"] = True
            reason = _status_reason(disconnect_status)
            if reason:
                warnings.append(f"disconnect reported: {reason}")
        except Exception as exc:  # noqa: BLE001
            reason = _exception_reason(exc)
            errors.append(f"disconnect failed: {reason}")
            blockers.append("IBKR paper disconnect failed.")
            connection["disconnected"] = False

    safety = {
        "real_ibkr_connection_enabled": allow_real_ibkr is True,
        "paper_order_submission_enabled": False,
        "cancel_enabled": False,
        "live_orders_enabled": False,
        "market_data_enabled": False,
        "contract_qualification_enabled": False,
        "scheduler_changes_enabled": False,
        "lifecycle_wiring_enabled": False,
    }
    ready = _ready_for_stage4f3(
        config_valid=validated_config is not None,
        connection=connection,
        readonly_checks=readonly_checks,
        account_snapshot=account_snapshot,
        open_orders=open_orders,
        positions=positions,
        safety=safety,
        blockers=blockers,
    )

    report = {
        "dry_run": True,
        "ibkr_paper_connection_preflight": True,
        "generated_at": generated_at,
        "config": config_payload,
        "connection": connection,
        "readonly_checks": readonly_checks,
        "account_snapshot": account_snapshot,
        "open_orders": open_orders,
        "positions": positions,
        "readiness_for_stage4f3": {
            "ready_to_build_manual_real_paper_submit_command": ready,
            "blockers": sorted(blockers),
            "warnings": sorted(warnings),
        },
        "safety": safety,
        "recommendations": {
            "ordered_next_steps": list(ORDERED_NEXT_STEPS),
            "do_not_do_yet": list(DO_NOT_DO_YET),
        },
        "success": True,
        "errors": list(errors),
        "warnings": list(warnings),
    }
    safe = sanitize_json_safe(report)
    return safe if isinstance(safe, dict) else {"success": False, "errors": ["report serialization failed"]}


def _run_readonly_checks(
    *,
    client: Any,
    readonly_checks: dict[str, bool],
    account_snapshot: dict[str, Any],
    open_orders: dict[str, Any],
    positions: dict[str, Any],
    errors: list[str],
    blockers: list[str],
) -> None:
    try:
        client.get_current_time()
        readonly_checks["current_time_ok"] = True
    except Exception as exc:  # noqa: BLE001
        reason = _exception_reason(exc)
        errors.append(f"current_time failed: {reason}")
        blockers.append("IBKR paper current time read failed.")

    try:
        snapshot = client.get_account_snapshot()
        account_snapshot.update(_account_snapshot_payload(snapshot))
        readonly_checks["account_snapshot_ok"] = bool(account_snapshot["available"])
        if not account_snapshot["available"]:
            reason = _snapshot_failure_reason(snapshot) or "account snapshot unavailable"
            account_snapshot["failure_reason"] = reason
            errors.append(f"account_snapshot failed: {reason}")
            blockers.append("IBKR paper account snapshot read failed.")
    except Exception as exc:  # noqa: BLE001
        reason = _exception_reason(exc)
        account_snapshot["failure_reason"] = reason
        errors.append(f"account_snapshot failed: {reason}")
        blockers.append("IBKR paper account snapshot read failed.")

    try:
        orders = client.list_open_orders()
        open_orders["count"] = len(_as_list(orders))
        open_orders["available"] = True
        readonly_checks["open_orders_ok"] = True
    except Exception as exc:  # noqa: BLE001
        reason = _exception_reason(exc)
        open_orders["failure_reason"] = reason
        errors.append(f"open_orders failed: {reason}")
        blockers.append("IBKR paper open orders read failed.")

    try:
        position_rows = client.list_positions()
        positions["count"] = len(_as_list(position_rows))
        positions["available"] = True
        readonly_checks["positions_ok"] = True
    except Exception as exc:  # noqa: BLE001
        reason = _exception_reason(exc)
        positions["failure_reason"] = reason
        errors.append(f"positions failed: {reason}")
        blockers.append("IBKR paper positions read failed.")


def _ready_for_stage4f3(
    *,
    config_valid: bool,
    connection: dict[str, Any],
    readonly_checks: dict[str, bool],
    account_snapshot: dict[str, Any],
    open_orders: dict[str, Any],
    positions: dict[str, Any],
    safety: dict[str, bool],
    blockers: list[str],
) -> bool:
    checks = [
        connection["allow_real_ibkr"] is True,
        config_valid,
        connection["attempted"] is True,
        connection["connected"] is True,
        connection["paper_mode"] is True,
        connection["disconnected"] is True,
        readonly_checks["current_time_ok"] is True,
        readonly_checks["account_snapshot_ok"] is True,
        readonly_checks["open_orders_ok"] is True,
        readonly_checks["positions_ok"] is True,
        account_snapshot["available"] is True,
        open_orders["available"] is True,
        positions["available"] is True,
        safety["paper_order_submission_enabled"] is False,
        safety["cancel_enabled"] is False,
        safety["live_orders_enabled"] is False,
        safety["market_data_enabled"] is False,
        safety["contract_qualification_enabled"] is False,
        safety["scheduler_changes_enabled"] is False,
        safety["lifecycle_wiring_enabled"] is False,
    ]
    if not all(checks) and "Stage 4F-2 readiness checks are incomplete." not in blockers:
        blockers.append("Stage 4F-2 readiness checks are incomplete.")
    return all(checks) and not blockers


def _call_with_timeout(callback: Callable[[], Any], timeout_seconds: float) -> Any:
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put(("result", callback()))
        except BaseException as exc:  # noqa: BLE001
            result_queue.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        kind, value = result_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError(
            f"connect did not complete within {timeout_seconds} seconds"
        ) from exc
    if kind == "error":
        raise value
    return value


def _config_input_dict(config: dict[str, Any] | IbkrPaperConfig) -> dict[str, Any]:
    if isinstance(config, IbkrPaperConfig):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise ValueError("config must be a dict or IbkrPaperConfig")


def _config_payload(
    config: dict[str, Any] | IbkrPaperConfig,
    paper_config_valid: bool,
    validation_reason: str,
) -> dict[str, Any]:
    payload = asdict(config) if isinstance(config, IbkrPaperConfig) else dict(config)
    return {
        "trading_mode": payload.get("trading_mode"),
        "host": payload.get("host"),
        "port": payload.get("port"),
        "client_id": payload.get("client_id"),
        "account_id": payload.get("account_id"),
        "paper_config_valid": paper_config_valid,
        "validation_reason": validation_reason,
    }


def _merge_connection_status(connection: dict[str, Any], status: Any) -> None:
    payload = _to_dict(status)
    if "connected" in payload:
        connection["connected"] = bool(payload.get("connected"))
    if "paper_mode" in payload:
        connection["paper_mode"] = bool(payload.get("paper_mode"))
    if payload.get("reason") is not None:
        connection["reason"] = str(payload.get("reason"))
    elif connection.get("connected"):
        connection["reason"] = None


def _account_snapshot_payload(snapshot: Any) -> dict[str, Any]:
    payload = _to_dict(snapshot)
    result = {
        "available": False,
        "net_liquidation": payload.get("net_liquidation"),
        "available_funds": payload.get("available_funds"),
        "buying_power": payload.get("buying_power"),
        "timestamp": payload.get("timestamp"),
        "failure_reason": _snapshot_failure_reason(snapshot),
    }
    has_value = any(
        result.get(key) is not None
        for key in ("net_liquidation", "available_funds", "buying_power", "timestamp")
    )
    result["available"] = bool(has_value and result["failure_reason"] is None)
    return result


def _snapshot_failure_reason(snapshot: Any) -> str | None:
    payload = _to_dict(snapshot)
    raw = payload.get("raw")
    if isinstance(raw, dict) and raw.get("error") is not None:
        return str(raw.get("error"))
    return None


def _status_reason(status: Any) -> str | None:
    payload = _to_dict(status)
    reason = payload.get("reason")
    return str(reason) if reason is not None else None


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        return converted if isinstance(converted, dict) else {}
    if is_dataclass(value):
        converted = asdict(value)
        return converted if isinstance(converted, dict) else {}
    return {
        name: getattr(value, name)
        for name in (
            "trading_mode",
            "host",
            "port",
            "client_id",
            "account_id",
            "readonly",
            "connected",
            "paper_mode",
            "reason",
            "net_liquidation",
            "available_funds",
            "buying_power",
            "timestamp",
            "raw",
        )
        if hasattr(value, name)
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _iso_now(now_provider: Callable[[], datetime] | None) -> str:
    now = now_provider() if now_provider is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.isoformat()


def _exception_reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"
