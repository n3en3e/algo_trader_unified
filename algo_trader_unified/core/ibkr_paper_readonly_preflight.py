"""Stage 4E-2 read-only IBKR paper operator preflight report."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from algo_trader_unified.core.paper_broker_adapter import sanitize_json_safe


ORDERED_SUCCESS_STEPS = [
    "Paper Gateway read-only connectivity is available; next phase may add paper submit client behind explicit gates.",
]
ORDERED_FAILURE_STEPS = [
    "Fix paper Gateway connectivity before paper submit work.",
]
DO_NOT_DO_YET = [
    "Do not enable paper order submission until explicit 4E submit phase.",
    "Do not wire client into scheduler/lifecycle yet.",
]


def build_ibkr_paper_readonly_preflight_report(
    *,
    client: Any,
    now_provider: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe read-only IBKR paper preflight report.

    The client is expected to expose only the Stage 4E-1 read-only methods.
    This function does not import IBKR libraries, read config, instantiate
    broker clients, mutate state, or write files.
    """

    generated_at = _iso_now(now_provider)
    errors: list[str] = []
    warnings: list[str] = []
    config = _config_from_client(client)
    connection = {
        "attempted": False,
        "connected": False,
        "paper_mode": False,
        "reason": None,
    }
    readonly_checks = {
        "current_time_ok": False,
        "account_snapshot_ok": False,
        "open_orders_ok": False,
        "positions_ok": False,
        "disconnect_ok": False,
    }
    account_snapshot = {
        "net_liquidation": None,
        "available_funds": None,
        "buying_power": None,
        "timestamp": None,
        "available": False,
    }
    open_orders = {"count": 0, "available": False, "failure_reason": None}
    positions = {"count": 0, "available": False, "failure_reason": None}

    connect_attempted = False
    read_failure_seen = False

    try:
        connect_attempted = True
        connection["attempted"] = True
        connect_status = _call_client(client, "connect")
        _merge_connection_status(connection, connect_status)
        if not connection["connected"]:
            reason = connection.get("reason") or "connect did not establish a connection"
            connection["reason"] = reason
            errors.append(f"connect failed: {reason}")
        else:
            status = _call_client(client, "get_connection_status")
            _merge_connection_status(connection, status)
            if not connection["connected"]:
                reason = connection.get("reason") or "client reported disconnected after connect"
                connection["reason"] = reason
                errors.append(f"connection status failed: {reason}")
            else:
                try:
                    _call_client(client, "get_current_time")
                    readonly_checks["current_time_ok"] = True
                except Exception as exc:  # noqa: BLE001
                    read_failure_seen = True
                    reason = _exception_reason(exc)
                    errors.append(f"current_time failed: {reason}")

                try:
                    snapshot = _call_client(client, "get_account_snapshot")
                    account_snapshot.update(_account_snapshot_payload(snapshot))
                    readonly_checks["account_snapshot_ok"] = bool(account_snapshot["available"])
                    if not account_snapshot["available"]:
                        read_failure_seen = True
                        reason = _snapshot_failure_reason(snapshot) or "account snapshot unavailable"
                        errors.append(f"account_snapshot failed: {reason}")
                except Exception as exc:  # noqa: BLE001
                    read_failure_seen = True
                    reason = _exception_reason(exc)
                    account_snapshot["available"] = False
                    errors.append(f"account_snapshot failed: {reason}")

                try:
                    orders = _call_client(client, "list_open_orders")
                    open_orders["count"] = len(_as_list(orders))
                    open_orders["available"] = True
                    readonly_checks["open_orders_ok"] = True
                except Exception as exc:  # noqa: BLE001
                    read_failure_seen = True
                    reason = _exception_reason(exc)
                    open_orders["failure_reason"] = reason
                    errors.append(f"open_orders failed: {reason}")

                try:
                    position_rows = _call_client(client, "list_positions")
                    positions["count"] = len(_as_list(position_rows))
                    positions["available"] = True
                    readonly_checks["positions_ok"] = True
                except Exception as exc:  # noqa: BLE001
                    read_failure_seen = True
                    reason = _exception_reason(exc)
                    positions["failure_reason"] = reason
                    errors.append(f"positions failed: {reason}")

    except Exception as exc:  # noqa: BLE001
        read_failure_seen = True
        reason = _exception_reason(exc)
        if not connection["connected"]:
            connection["reason"] = reason
        errors.append(f"preflight failed: {reason}")
    finally:
        if connect_attempted:
            try:
                disconnect_status = _call_client(client, "disconnect")
                readonly_checks["disconnect_ok"] = True
                if _status_reason(disconnect_status):
                    warnings.append(f"disconnect reported: {_status_reason(disconnect_status)}")
            except Exception as exc:  # noqa: BLE001
                reason = _exception_reason(exc)
                readonly_checks["disconnect_ok"] = False
                message = f"disconnect failed: {reason}"
                errors.append(message)

    return _final_report(
        generated_at=generated_at,
        config=config,
        connection=connection,
        readonly_checks=readonly_checks,
        account_snapshot=account_snapshot,
        open_orders=open_orders,
        positions=positions,
        errors=errors,
        warnings=warnings,
    )


def _final_report(
    *,
    generated_at: str,
    config: dict[str, Any],
    connection: dict[str, Any],
    readonly_checks: dict[str, bool],
    account_snapshot: dict[str, Any],
    open_orders: dict[str, Any],
    positions: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    success = (
        connection["attempted"]
        and connection["connected"]
        and connection["paper_mode"]
        and all(readonly_checks.values())
        and not errors
    )
    steps = ORDERED_SUCCESS_STEPS if success else ORDERED_FAILURE_STEPS
    report = {
        "dry_run": True,
        "ibkr_paper_readonly_preflight": True,
        "generated_at": generated_at,
        "config": config,
        "connection": connection,
        "readonly_checks": readonly_checks,
        "account_snapshot": account_snapshot,
        "open_orders": open_orders,
        "positions": positions,
        "safety": {
            "broker_calls_enabled": True,
            "order_submission_enabled": False,
            "cancel_enabled": False,
            "market_data_enabled": False,
            "contract_qualification_enabled": False,
            "live_orders_enabled": False,
            "scheduler_changes_enabled": False,
        },
        "recommendations": {
            "ordered_next_steps": list(steps),
            "do_not_do_yet": list(DO_NOT_DO_YET),
        },
        "success": success,
        "errors": list(errors),
        "warnings": list(warnings),
    }
    safe = sanitize_json_safe(report)
    return safe if isinstance(safe, dict) else {"success": False, "errors": ["report serialization failed"]}


def _call_client(client: Any, method_name: str) -> Any:
    method = getattr(client, method_name)
    return method()


def _config_from_client(client: Any) -> dict[str, Any]:
    raw_config = getattr(client, "config", None)
    config = _to_dict(raw_config)
    return {
        "trading_mode": config.get("trading_mode"),
        "host": config.get("host"),
        "port": config.get("port"),
        "client_id": config.get("client_id"),
        "account_id": config.get("account_id"),
        "readonly": config.get("readonly"),
        "paper_config_valid": _paper_config_valid(config),
    }


def _paper_config_valid(config: dict[str, Any]) -> bool:
    return (
        config.get("trading_mode") == "PAPER"
        and config.get("port") == 4004
        and isinstance(config.get("host"), str)
        and bool(str(config.get("host")).strip())
        and isinstance(config.get("client_id"), int)
        and not isinstance(config.get("client_id"), bool)
        and config.get("client_id") > 0
        and config.get("readonly") is True
    )


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
        "net_liquidation": payload.get("net_liquidation"),
        "available_funds": payload.get("available_funds"),
        "buying_power": payload.get("buying_power"),
        "timestamp": payload.get("timestamp"),
        "available": False,
    }
    has_value = any(
        result.get(key) is not None
        for key in ("net_liquidation", "available_funds", "buying_power", "timestamp")
    )
    result["available"] = bool(has_value and not _snapshot_failure_reason(snapshot))
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
