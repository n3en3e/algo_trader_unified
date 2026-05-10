"""Pure Stage 4F manual real-paper execution acceptance reporting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable


PHASE_KEYS = (
    "stage4f1_factory_preflight",
    "stage4f2_connection_preflight",
    "stage4f3_manual_submit",
    "stage4f4_status_cancel",
    "stage4f5_smoke_test",
)
MODULE_CHECK_KEYS = (
    "ibkr_paper_factory_present",
    "ibkr_paper_connection_preflight_present",
    "manual_real_paper_submit_present",
    "manual_real_paper_order_control_present",
    "stage4f5_smoke_test_report_present",
)
SAFETY_CHECK_KEYS = (
    "no_live_path",
    "no_market_data_path",
    "no_contract_qualification_path",
    "no_daemon_wiring",
    "no_scheduler_wiring",
    "no_lifecycle_wiring",
    "no_automated_paper_trading",
    "no_system" + "d_changes",
    "no_external_fetch",
)
UNKNOWN_EXPOSURE_KEYS = (
    "unknown_broker_exposure_count",
    "unknown_broker_exposures",
    "broker_exposure_unknown_count",
    "unreconciled_broker_positions_count",
)
SMOKE_SAFETY_TRUE_KEYS = (
    "no_live_orders",
    "no_market_data",
    "no_contract_qualification",
    "no_scheduler_changes",
    "no_lifecycle_wiring",
)
SMOKE_SAFETY_FALSE_KEYS = (
    "live_orders_enabled",
    "market_data_enabled",
    "contract_qualification_enabled",
    "scheduler_changes_enabled",
    "lifecycle_wiring_enabled",
)
ORDERED_NEXT_STEPS = [
    "Begin Stage 4G manual paper lifecycle validation behind explicit operator gates.",
    "Keep scheduler/lifecycle paper execution disabled.",
    "Use only manual paper status/reconciliation checks in 4G.",
]
DO_NOT_DO_YET = [
    "Do not enable automated paper trading yet.",
    "Do not begin live trading.",
]


def build_stage4f_acceptance_report(
    *,
    reports: dict[str, dict] | None = None,
    module_checks: dict[str, bool] | None = None,
    safety_checks: dict[str, bool] | None = None,
    smoke_test_report: dict | None = None,
    state_snapshot: dict | None = None,
    now_provider: Callable[[], datetime] | None = None,
) -> dict:
    """Build a JSON-safe Stage 4F acceptance report from supplied dictionaries."""

    try:
        return _build_report(
            reports=reports,
            module_checks=module_checks,
            safety_checks=safety_checks,
            smoke_test_report=smoke_test_report,
            state_snapshot=state_snapshot,
            now_provider=now_provider,
        )
    except Exception as exc:  # noqa: BLE001 - acceptance boundary returns data.
        return _json_safe(
            {
                "dry_run": True,
                "stage4f_acceptance_report": True,
                "generated_at": _current_time(now_provider).isoformat(),
                "phase_status": {key: False for key in PHASE_KEYS},
                "module_checks": {key: False for key in MODULE_CHECK_KEYS},
                "safety_checks": {key: False for key in SAFETY_CHECK_KEYS},
                "smoke_test": _empty_smoke_test(),
                "state_safety": _empty_state_safety(),
                "readiness_for_stage4g": {
                    "ready_to_begin_manual_paper_lifecycle_validation": False,
                    "blockers": ["unexpected report failure"],
                    "warnings": [],
                },
                "recommendations": {
                    "ordered_next_steps": list(ORDERED_NEXT_STEPS),
                    "do_not_do_yet": list(DO_NOT_DO_YET),
                },
                "safety": _disabled_safety(False),
                "success": False,
                "errors": [f"unexpected report failure: {type(exc).__name__}: {exc}"],
                "warnings": [],
            }
        )


def _build_report(
    *,
    reports: dict[str, dict] | None,
    module_checks: dict[str, bool] | None,
    safety_checks: dict[str, bool] | None,
    smoke_test_report: dict | None,
    state_snapshot: dict | None,
    now_provider: Callable[[], datetime] | None,
) -> dict:
    now = _current_time(now_provider)
    reports = reports if isinstance(reports, dict) else {}
    module_checks = module_checks if isinstance(module_checks, dict) else {}
    safety_checks = safety_checks if isinstance(safety_checks, dict) else {}
    smoke_report = smoke_test_report if isinstance(smoke_test_report, dict) else None
    snapshot = state_snapshot if isinstance(state_snapshot, dict) else None

    errors: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []

    if smoke_test_report is not None and smoke_report is None:
        errors.append("smoke_test_report must be a dict")
        blockers.append("smoke_test_report malformed")
    if state_snapshot is not None and snapshot is None:
        warnings.append("state_snapshot ignored because it is not a dict")

    phase_status = {key: _phase_passed(reports.get(key)) for key in PHASE_KEYS}
    for key, passed in phase_status.items():
        if not passed:
            blockers.append(f"{key} missing or not passed")

    normalized_module_checks = {
        key: bool(module_checks.get(key)) for key in MODULE_CHECK_KEYS
    }
    for key, passed in normalized_module_checks.items():
        if not passed:
            blockers.append(f"module check failed: {key}")

    normalized_safety_checks = {
        key: bool(safety_checks.get(key)) for key in SAFETY_CHECK_KEYS
    }
    for key, passed in normalized_safety_checks.items():
        if not passed:
            blockers.append(f"safety check failed: {key}")

    smoke_summary, smoke_blockers, smoke_warnings, smoke_errors = _smoke_test_summary(
        smoke_report
    )
    blockers.extend(smoke_blockers)
    warnings.extend(smoke_warnings)
    errors.extend(smoke_errors)

    state_safety, state_blockers, state_warnings = _state_safety(snapshot)
    blockers.extend(state_blockers)
    warnings.extend(state_warnings)

    blocker_list = sorted(set(blockers))
    warning_list = sorted(set(warnings))
    error_list = sorted(set(errors))
    ready = not blocker_list and not error_list

    report = {
        "dry_run": True,
        "stage4f_acceptance_report": True,
        "generated_at": now.isoformat(),
        "phase_status": phase_status,
        "module_checks": normalized_module_checks,
        "safety_checks": normalized_safety_checks,
        "smoke_test": smoke_summary,
        "state_safety": state_safety,
        "readiness_for_stage4g": {
            "ready_to_begin_manual_paper_lifecycle_validation": ready,
            "blockers": blocker_list,
            "warnings": warning_list,
        },
        "recommendations": {
            "ordered_next_steps": list(ORDERED_NEXT_STEPS),
            "do_not_do_yet": list(DO_NOT_DO_YET),
        },
        "safety": _disabled_safety(smoke_summary["accepted"] and ready),
        "success": True,
        "errors": error_list,
        "warnings": warning_list,
    }
    return _json_safe(report)


def _phase_passed(value: Any) -> bool:
    if value is True:
        return True
    if not isinstance(value, dict):
        return False
    if value.get("success") is False:
        return False
    if value.get("passed") is True or value.get("available") is True:
        return True
    return value.get("success") is True


def _smoke_test_summary(
    report: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    if report is None:
        blockers.append("smoke_test_report missing")
        return _empty_smoke_test(), blockers, warnings, errors

    smoke = _mapping(report.get("smoke_test"))
    if not smoke:
        blockers.append("smoke_test_report.smoke_test missing or malformed")
        errors.append("smoke_test_report.smoke_test missing or malformed")

    summary = {
        "present": True,
        "accepted": smoke.get("accepted") is True,
        "one_order_only": smoke.get("one_order_only") is True,
        "broker_order_id": smoke.get("broker_order_id"),
        "client_order_id": smoke.get("client_order_id"),
        "submitted": smoke.get("submitted") is True,
        "status_seen": smoke.get("status_seen") is True,
        "cancel_seen": smoke.get("cancel_seen") is True,
        "terminal_or_safe_state_seen": (
            smoke.get("terminal_or_safe_state_seen") is True
        ),
    }
    checks = {
        "smoke_test_report.stage4f5_smoke_test_report must be True": (
            report.get("stage4f5_smoke_test_report") is True
        ),
        "smoke_test.accepted must be True": summary["accepted"],
        "smoke_test.one_order_only must be True": summary["one_order_only"],
        "smoke_test.submitted must be True": summary["submitted"],
        "smoke_test.status_seen must be True": summary["status_seen"],
        "smoke_test.terminal_or_safe_state_seen must be True": summary[
            "terminal_or_safe_state_seen"
        ],
    }
    for reason, passed in checks.items():
        if not passed:
            blockers.append(reason)

    safety = _mapping(report.get("safety_checks"))
    if not safety:
        blockers.append("smoke_test_report.safety_checks missing or malformed")
        errors.append("smoke_test_report.safety_checks missing or malformed")
    for key in SMOKE_SAFETY_TRUE_KEYS:
        if safety.get(key) is not True:
            blockers.append(f"smoke_test_report safety check failed: {key}")
    for key in SMOKE_SAFETY_FALSE_KEYS:
        if key in safety and safety.get(key) is not False:
            blockers.append(f"smoke_test_report unsafe flag enabled: {key}")
    return summary, blockers, warnings, errors


def _state_safety(
    snapshot: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    if snapshot is None:
        return _empty_state_safety(), [], []

    blockers: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {}
    result["unresolved_needs_reconciliation_count"] = _first_int(
        snapshot,
        ("unresolved_needs_reconciliation_count", "needs_reconciliation_count"),
        default=_count_needs_reconciliation(snapshot),
    )
    result["active_intents_count"] = _first_int(
        snapshot,
        ("active_intents_count", "open_intents_count"),
    )
    result["open_positions_count"] = _first_int(
        snapshot,
        ("open_positions_count", "positions_open_count"),
    )
    result["active_halt"] = _active_halt(snapshot)

    unresolved = result.get("unresolved_needs_reconciliation_count")
    if isinstance(unresolved, int) and unresolved > 0:
        blockers.append("unresolved NEEDS_RECONCILIATION records exist")
    active_halt = result.get("active_halt")
    if active_halt is True:
        blockers.append("active halt is present")

    unknown = _unknown_broker_exposure_count(snapshot)
    if unknown is None:
        warnings.append("unknown broker exposure not tracked in state_snapshot")
    else:
        result["unknown_broker_exposure_count"] = unknown
        if unknown > 0:
            blockers.append("unknown broker exposure exists")
    return result, blockers, warnings


def _empty_smoke_test() -> dict[str, Any]:
    return {
        "present": False,
        "accepted": False,
        "one_order_only": False,
        "broker_order_id": None,
        "client_order_id": None,
        "submitted": False,
        "status_seen": False,
        "cancel_seen": False,
        "terminal_or_safe_state_seen": False,
    }


def _empty_state_safety() -> dict[str, Any]:
    return {
        "unresolved_needs_reconciliation_count": None,
        "active_intents_count": None,
        "open_positions_count": None,
        "active_halt": None,
    }


def _disabled_safety(real_paper_proven: bool) -> dict[str, bool]:
    return {
        "real_ibkr_paper_manual_execution_proven": bool(real_paper_proven),
        "automated_paper_trading_enabled": False,
        "live_orders_enabled": False,
        "market_data_enabled": False,
        "contract_qualification_enabled": False,
        "scheduler_changes_enabled": False,
        "lifecycle_wiring_enabled": False,
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_int(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    *,
    default: int | None = None,
) -> int | None:
    for key in keys:
        value = payload.get(key)
        count = _count_value(value)
        if count is not None:
            return count
    return default


def _count_needs_reconciliation(payload: dict[str, Any]) -> int | None:
    positions = payload.get("positions")
    if not isinstance(positions, (dict, list, tuple)):
        return None
    values = positions.values() if isinstance(positions, dict) else positions
    return sum(
        1
        for item in values
        if isinstance(item, dict) and item.get("status") == "NEEDS_RECONCILIATION"
    )


def _active_halt(payload: dict[str, Any]) -> bool | None:
    for key in ("active_halt", "halt_active"):
        if key in payload:
            return bool(payload.get(key))
    return None


def _unknown_broker_exposure_count(payload: dict[str, Any]) -> int | None:
    for key in UNKNOWN_EXPOSURE_KEYS:
        if key in payload:
            return _count_value(payload.get(key)) or 0
    return None


def _count_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return None


def _current_time(now_provider: Callable[[], datetime] | None) -> datetime:
    value = now_provider() if now_provider is not None else datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return f"<{type(value).__name__}>"
    return value
