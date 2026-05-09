"""Stage 4F-3 manual real IBKR paper submit command boundary.

This module coordinates one already-approved paper ticket, one successful
read-only preflight report, and injected real-paper factories. It does not
import IBKR runtime libraries, read config.py, read ledgers, write state, fetch
market data, qualify contracts, or wire execution into lifecycle jobs.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal
import math
from typing import Any, Callable

from algo_trader_unified.core.ibkr_paper_order_mapper import (
    IBKR_PAPER_PORT,
    IbkrPaperConfig,
    validate_ibkr_paper_config,
)


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
REQUIRED_ACKNOWLEDGEMENTS = [
    "I understand this is IBKR PAPER only.",
    "I understand this will submit one real paper order.",
    "I understand no live orders are allowed.",
    "I understand scheduler/lifecycle automation remains disabled.",
    "I reviewed the ticket and preflight report.",
]
ORDERED_NEXT_STEPS = [
    "Review this manual real-paper submit report with the operator.",
    "Record the submit result alongside the Stage 4E-4 ticket and Stage 4F-2 preflight.",
    "Keep any future paper submit attempts single-ticket and manually gated.",
]
DO_NOT_DO_YET = [
    "Do not enable live orders.",
    "Do not wire paper submission into scheduler, daemon, or lifecycle jobs.",
    "Do not request market data or qualify contracts.",
    "Do not change strategy thresholds, sizing, or execution cadence.",
]


def build_manual_real_paper_submit_report(
    *,
    ticket_report: dict,
    connection_preflight_report: dict,
    execution_client_factory: Callable[[Any, IbkrPaperConfig], Any],
    ib_factory: Callable[[], Any],
    config: dict[str, Any] | IbkrPaperConfig,
    operator_acknowledgements: list[str],
    allow_real_ibkr: bool = False,
    allow_real_paper_submit: bool = False,
    now_provider: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Build and optionally execute one manually gated real-paper submit report."""

    generated_at = _generated_at(now_provider)
    errors: list[str] = []
    warnings: list[str] = []
    reasons: list[str] = []
    validated_config: IbkrPaperConfig | None = None

    ticket_source = ticket_report if isinstance(ticket_report, dict) else {}
    preflight_source = (
        connection_preflight_report
        if isinstance(connection_preflight_report, dict)
        else {}
    )
    plan = _mapping(ticket_source.get("ibkr_order_plan"))
    ticket_safety = _mapping(ticket_source.get("safety"))
    preflight_safety = _mapping(preflight_source.get("safety"))
    readonly_checks = _mapping(preflight_source.get("readonly_checks"))
    connection = _mapping(preflight_source.get("connection"))
    readiness = _mapping(preflight_source.get("readiness_for_stage4f3"))

    config_valid = False
    port_explicit = _config_has_explicit_port(config)
    port_is_paper = _config_port(config) == IBKR_PAPER_PORT
    try:
        if not port_explicit:
            raise ValueError("config must explicitly include port 4004")
        validated_config = validate_ibkr_paper_config(_config_input_dict(config))
        config_valid = True
    except Exception as exc:  # noqa: BLE001 - report boundary must not crash.
        reason = _exception_reason(exc)
        errors.append(f"validate_ibkr_paper_config failed: {reason}")
        reasons.append(f"config validation failed: {reason}")

    acknowledgements = _acknowledgement_report(operator_acknowledgements)
    ticket_eligible = _ticket_eligible(ticket_source, plan, ticket_safety, reasons)
    preflight_ready = _preflight_ready(
        preflight_source,
        connection,
        readonly_checks,
        readiness,
        preflight_safety,
        reasons,
    )

    if allow_real_ibkr is not True:
        reasons.append("allow_real_ibkr must be True")
    if allow_real_paper_submit is not True:
        reasons.append("allow_real_paper_submit must be True")
    if not port_explicit and "config must explicitly include port 4004" not in reasons:
        reasons.append("config must explicitly include port 4004")
    if port_explicit and not port_is_paper:
        reasons.append("config port must be exactly 4004 for IBKR paper")
    if not acknowledgements["exact_match"]:
        for missing in acknowledgements["missing"]:
            reasons.append(f"missing required operator acknowledgement: {missing}")
    _append_override_reasons(ticket_source, preflight_source, reasons)

    gates = {
        "allow_real_ibkr": allow_real_ibkr is True,
        "allow_real_paper_submit": allow_real_paper_submit is True,
        "ticket_eligible": ticket_eligible,
        "preflight_ready": preflight_ready,
        "acknowledgements_ok": acknowledgements["exact_match"],
        "config_valid": config_valid,
        "port_explicit": port_explicit,
        "port_is_paper": port_is_paper,
        "passed": False,
        "reasons": [],
    }
    deduped_reasons = _dedupe(reasons)
    gates["passed"] = not deduped_reasons
    gates["reasons"] = deduped_reasons

    submission = {
        "attempted": False,
        "submitted": False,
        "broker_order_id": None,
        "client_order_id": plan.get("client_order_id"),
        "reason": None if gates["passed"] else "; ".join(deduped_reasons),
        "raw": None,
    }
    cleanup = {
        "disconnect_attempted": False,
        "disconnect_ok": False,
        "disconnect_reason": None,
    }

    if gates["passed"]:
        ib = None
        execution_client = None
        try:
            ib = ib_factory()
        except Exception as exc:  # noqa: BLE001
            reason = _exception_reason(exc)
            errors.append(f"ib_factory failed: {reason}")
            submission["reason"] = reason
        else:
            try:
                execution_client = execution_client_factory(ib, validated_config)
            except Exception as exc:  # noqa: BLE001
                reason = _exception_reason(exc)
                errors.append(f"execution_client_factory failed: {reason}")
                submission["reason"] = reason
            else:
                submission["attempted"] = True
                try:
                    broker_result = execution_client.submit_order_plan(plan)
                    submission.update(_submission_from_broker_result(broker_result, plan))
                except Exception as exc:  # noqa: BLE001
                    reason = _exception_reason(exc)
                    errors.append(f"submit_order_plan failed: {reason}")
                    submission.update(
                        {
                            "submitted": False,
                            "broker_order_id": None,
                            "client_order_id": plan.get("client_order_id"),
                            "reason": reason,
                            "raw": {
                                "operation": "submit_order_plan",
                                "error": reason,
                            },
                        }
                    )
                finally:
                    cleanup["disconnect_attempted"] = True
                    try:
                        disconnect_result = execution_client.disconnect()
                        cleanup.update(_cleanup_from_disconnect(disconnect_result))
                    except Exception as exc:  # noqa: BLE001
                        reason = _exception_reason(exc)
                        warnings.append(f"disconnect failed: {reason}")
                        cleanup.update(
                            {
                                "disconnect_ok": False,
                                "disconnect_reason": reason,
                            }
                        )

    report = {
        "dry_run": True,
        "manual_real_paper_submit": True,
        "generated_at": generated_at,
        "gates": gates,
        "ticket": _ticket_summary(ticket_source, plan),
        "preflight": {
            "connected": connection.get("connected") is True,
            "current_time_ok": readonly_checks.get("current_time_ok") is True,
            "account_snapshot_ok": readonly_checks.get("account_snapshot_ok") is True,
            "open_orders_ok": readonly_checks.get("open_orders_ok") is True,
            "positions_ok": readonly_checks.get("positions_ok") is True,
        },
        "acknowledgements": acknowledgements,
        "submission": submission,
        "cleanup": cleanup,
        "safety": {
            "real_ibkr_enabled": allow_real_ibkr is True,
            "real_paper_submit_enabled": allow_real_paper_submit is True,
            "live_orders_enabled": False,
            "market_data_enabled": False,
            "contract_qualification_enabled": False,
            "scheduler_changes_enabled": False,
            "lifecycle_wiring_enabled": False,
        },
        "recommendations": {
            "ordered_next_steps": list(ORDERED_NEXT_STEPS),
            "do_not_do_yet": list(DO_NOT_DO_YET),
        },
        "success": True,
        "errors": list(errors),
        "warnings": list(warnings),
    }
    return _json_safe(report)


def _generated_at(now_provider: Callable[[], Any] | None) -> str:
    if now_provider is None:
        return DEFAULT_GENERATED_AT
    value = now_provider()
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _config_input_dict(config: dict[str, Any] | IbkrPaperConfig) -> dict[str, Any]:
    if isinstance(config, IbkrPaperConfig):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise ValueError("config must be a dict or IbkrPaperConfig")


def _config_has_explicit_port(config: Any) -> bool:
    if isinstance(config, IbkrPaperConfig):
        return True
    return isinstance(config, dict) and "port" in config


def _config_port(config: Any) -> Any:
    if isinstance(config, IbkrPaperConfig):
        return config.port
    if isinstance(config, dict):
        return config.get("port")
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _acknowledgement_report(provided: list[str]) -> dict[str, Any]:
    normalized = [item.strip() for item in provided if isinstance(item, str)]
    missing = [item for item in REQUIRED_ACKNOWLEDGEMENTS if item not in normalized]
    return {
        "required": list(REQUIRED_ACKNOWLEDGEMENTS),
        "provided": normalized,
        "missing": missing,
        "exact_match": not missing,
    }


def _ticket_eligible(
    report: dict[str, Any],
    plan: dict[str, Any],
    safety: dict[str, Any],
    reasons: list[str],
) -> bool:
    before = len(reasons)
    submit_gate = _mapping(report.get("submit_gate"))
    if report.get("paper_order_ticket_report") is not True:
        reasons.append("ticket_report.paper_order_ticket_report must be True")
    if submit_gate.get("eligible_for_future_manual_submit") is not True:
        reasons.append("ticket submit gate must be eligible for future manual submit")
    if plan.get("ready_for_submission") is not True:
        reasons.append("IBKR paper order plan must be ready for submission")
    if plan.get("paper_only") is not True:
        reasons.append("IBKR paper order plan must be paper_only")
    if plan.get("dry_run") is not True:
        reasons.append("IBKR paper order plan must be dry_run")
    if plan.get("blockers"):
        reasons.append("IBKR paper order plan blockers must be empty")
    if safety.get("live_orders_enabled") is not False:
        reasons.append("ticket safety live_orders_enabled must be False")
    if safety.get("scheduler_changes_enabled") is not False:
        reasons.append("ticket safety scheduler_changes_enabled must be False")
    return len(reasons) == before


def _preflight_ready(
    report: dict[str, Any],
    connection: dict[str, Any],
    readonly_checks: dict[str, Any],
    readiness: dict[str, Any],
    safety: dict[str, Any],
    reasons: list[str],
) -> bool:
    before = len(reasons)
    if report.get("ibkr_paper_connection_preflight") is not True:
        reasons.append("connection_preflight_report.ibkr_paper_connection_preflight must be True")
    if connection.get("connected") is not True:
        reasons.append("connection preflight must report connected True")
    for key in (
        "current_time_ok",
        "account_snapshot_ok",
        "open_orders_ok",
        "positions_ok",
    ):
        if readonly_checks.get(key) is not True:
            reasons.append(f"connection preflight readonly_checks.{key} must be True")
    if (
        readiness.get("ready_to_build_manual_real_paper_submit_command")
        is not True
    ):
        reasons.append("Stage 4F-2 readiness for manual real-paper submit must be True")
    if safety.get("live_orders_enabled") is not False:
        reasons.append("preflight safety live_orders_enabled must be False")
    if safety.get("scheduler_changes_enabled") is not False:
        reasons.append("preflight safety scheduler_changes_enabled must be False")
    return len(reasons) == before


def _append_override_reasons(
    ticket_report: dict[str, Any],
    preflight_report: dict[str, Any],
    reasons: list[str],
) -> None:
    for label, report in (
        ("ticket", ticket_report),
        ("preflight", preflight_report),
    ):
        safety = _mapping(report.get("safety"))
        for key in ("allow_live", "live_override", "live_orders_override_enabled"):
            if report.get(key) is True or safety.get(key) is True:
                reasons.append(f"{label} live override flag must not be enabled")
        for key in (
            "scheduler_override",
            "lifecycle_override",
            "scheduler_lifecycle_override_enabled",
        ):
            if report.get(key) is True or safety.get(key) is True:
                reasons.append(
                    f"{label} scheduler/lifecycle override flag must not be enabled"
                )


def _ticket_summary(
    report: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    intent = _mapping(report.get("intent"))
    request = _mapping(report.get("broker_order_request"))
    return {
        "client_order_id": plan.get("client_order_id") or request.get("client_order_id"),
        "strategy_id": request.get("strategy_id") or intent.get("strategy_id"),
        "symbol": request.get("symbol") or intent.get("symbol"),
        "action": plan.get("action"),
        "quantity": plan.get("quantity") or request.get("quantity"),
        "order_type": plan.get("order_type") or request.get("order_type"),
        "paper_only": plan.get("paper_only"),
        "ready_for_submission": plan.get("ready_for_submission"),
    }


def _submission_from_broker_result(
    broker_result: Any,
    plan: dict[str, Any],
) -> dict[str, Any]:
    if hasattr(broker_result, "to_dict"):
        source = broker_result.to_dict()
    elif isinstance(broker_result, dict):
        source = dict(broker_result)
    else:
        source = {
            "accepted": getattr(broker_result, "accepted", False),
            "broker_order_id": getattr(broker_result, "broker_order_id", None),
            "client_order_id": getattr(broker_result, "client_order_id", None),
            "reason": getattr(broker_result, "reason", None),
            "raw": getattr(broker_result, "raw", None),
        }
    return {
        "submitted": source.get("accepted") is True,
        "broker_order_id": source.get("broker_order_id"),
        "client_order_id": source.get("client_order_id") or plan.get("client_order_id"),
        "reason": source.get("reason"),
        "raw": _raw_json_safe(source.get("raw")),
    }


def _cleanup_from_disconnect(disconnect_result: Any) -> dict[str, Any]:
    if hasattr(disconnect_result, "to_dict"):
        source = disconnect_result.to_dict()
    elif isinstance(disconnect_result, dict):
        source = disconnect_result
    else:
        source = {
            "connected": getattr(disconnect_result, "connected", None),
            "reason": getattr(disconnect_result, "reason", None),
        }
    reason = source.get("reason")
    return {
        "disconnect_ok": reason in (None, ""),
        "disconnect_reason": reason,
    }


def _exception_reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _json_safe(report: dict[str, Any]) -> dict[str, Any]:
    safe = _raw_json_safe(report)
    if isinstance(safe, dict):
        return safe
    return {"success": False, "errors": ["manual real-paper submit report was not JSON-safe"]}


def _raw_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Decimal):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if isinstance(value, dict):
        return {str(key): _raw_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_raw_json_safe(item) for item in value]
    return "<object>"
