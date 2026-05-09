"""Pure Stage 4E paper-execution acceptance reporting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable


PHASE_KEYS = (
    "stage4e1_readonly_client",
    "stage4e2_readonly_preflight",
    "stage4e3_fake_execution_client",
    "stage4e4_ticket_gate",
    "stage4e5_manual_submit_gate",
)
MODULE_CHECK_KEYS = (
    "ibkr_paper_client_present",
    "ibkr_paper_readonly_preflight_present",
    "ibkr_paper_execution_client_present",
    "paper_order_ticket_report_present",
    "manual_paper_submit_gate_present",
)
SAFETY_CHECK_KEYS = (
    "no_real_ibkr_required",
    "no_ib_" + "insync_runtime_requirement",
    "no_place_order_path",
    "no_cancel_order_path_outside_fake_client",
    "no_market_data_path",
    "no_contract_qualification_path",
    "no_daemon_wiring",
    "no_scheduler_wiring",
    "no_lifecycle_wiring",
    "no_live_path",
    "no_system" + "d_changes",
    "no_external_fetch",
)
UNKNOWN_EXPOSURE_KEYS = (
    "unknown_broker_exposure_count",
    "unknown_broker_exposures",
    "broker_exposure_unknown_count",
    "unreconciled_broker_positions_count",
)
ORDERED_NEXT_STEPS = [
    "Begin Stage 4F real IBKR paper submit planning behind explicit manual gates.",
    "Keep scheduler/lifecycle paper execution disabled.",
    "Run read-only paper preflight before any real paper submit test.",
]
DO_NOT_DO_YET = [
    "Do not enable automated paper execution yet.",
    "Do not begin live trading.",
    "Do not bypass readiness gates.",
]


def build_stage4e_acceptance_report(
    *,
    reports: dict[str, dict] | None = None,
    module_checks: dict[str, bool] | None = None,
    safety_checks: dict[str, bool] | None = None,
    state_snapshot: dict | None = None,
    now_provider: Callable[[], datetime] | None = None,
) -> dict:
    """Build a JSON-safe acceptance report from already-collected dictionaries."""

    now = _current_time(now_provider)
    reports = reports or {}
    module_checks = module_checks or {}
    safety_checks = safety_checks or {}
    snapshot = state_snapshot if isinstance(state_snapshot, dict) else None

    errors: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []

    phase_status = {
        key: _phase_passed(reports.get(key)) for key in PHASE_KEYS
    }
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

    state_safety, state_warnings = _state_safety(snapshot)
    warnings.extend(state_warnings)
    if snapshot is None:
        blockers.append("state_snapshot unavailable")
    else:
        unresolved = state_safety.get("unresolved_needs_reconciliation_count")
        if isinstance(unresolved, int) and unresolved > 0:
            blockers.append("unresolved NEEDS_RECONCILIATION records exist")
        active_halt = state_safety.get("active_halt")
        if active_halt is True:
            blockers.append("active halt is present")
        elif active_halt is None:
            warnings.append("active halt not tracked in state_snapshot")
        unknown_exposure = state_safety.get("unknown_broker_exposure_count")
        if isinstance(unknown_exposure, int) and unknown_exposure > 0:
            blockers.append("unknown broker exposure exists")

    blocker_list = sorted(set(blockers))
    warning_list = sorted(set(warnings))
    ready = not blocker_list

    report = {
        "dry_run": True,
        "stage4e_acceptance_report": True,
        "generated_at": now.isoformat(),
        "phase_status": phase_status,
        "module_checks": normalized_module_checks,
        "safety_checks": normalized_safety_checks,
        "state_safety": state_safety,
        "readiness_for_stage4f": {
            "ready_to_begin_real_ibkr_paper_submit_planning": ready,
            "blockers": blocker_list,
            "warnings": warning_list,
        },
        "recommendations": {
            "ordered_next_steps": ORDERED_NEXT_STEPS[:],
            "do_not_do_yet": DO_NOT_DO_YET[:],
        },
        "safety": {
            "real_ibkr_enabled": False,
            "paper_order_submission_enabled": False,
            "live_orders_enabled": False,
            "market_data_enabled": False,
            "contract_qualification_enabled": False,
            "scheduler_changes_enabled": False,
            "lifecycle_wiring_enabled": False,
        },
        "success": True,
        "errors": errors,
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


def _state_safety(snapshot: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    if snapshot is None:
        state = {
            "unresolved_needs_reconciliation_count": None,
            "active_intents_count": None,
            "open_positions_count": None,
            "active_halt": None,
        }
        return state, []

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

    unknown = _unknown_broker_exposure_count(snapshot)
    if unknown is None:
        warnings.append("unknown broker exposure not tracked in state_snapshot")
    else:
        result["unknown_broker_exposure_count"] = unknown
    return result, warnings


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
    broker_count = _count_value(payload.get("broker_open_positions_count"))
    internal_count = _count_value(payload.get("internal_open_positions_count"))
    if broker_count is not None and internal_count is not None:
        return max(0, broker_count - internal_count)
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
        return str(value)
    return value
