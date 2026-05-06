"""Read-only Stage 4D-2 paper broker adapter preflight reporting."""

from __future__ import annotations

import inspect
import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from algo_trader_unified.config.scheduler import (
    JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS,
    JOB_DRY_RUN_CONFIRM_FILLS,
)
from algo_trader_unified.core import broker_adapter as broker_adapter_module
from algo_trader_unified.core import scheduler_cadence
from algo_trader_unified.core.broker_adapter import (
    BrokerAccountSnapshot,
    BrokerAdapter,
    BrokerCancelResult,
    BrokerOrderStatus,
    BrokerPosition,
    BrokerSubmitResult,
)
from algo_trader_unified.core.paper_broker_contract import (
    BrokerModeError,
    NullBrokerAdapter,
    validate_broker_mode,
)


_REQUIRED_METHODS = (
    "submit_order_intent",
    "cancel_order",
    "get_order_status",
    "list_open_orders",
    "list_positions",
    "get_account_snapshot",
)
_RESULT_SHAPES = (
    BrokerSubmitResult,
    BrokerCancelResult,
    BrokerOrderStatus,
    BrokerPosition,
    BrokerAccountSnapshot,
)
_ACTIVE_INTENT_STATUSES = {"created", "submitted", "confirmed", "filled"}
_DO_NOT_DO_YET = [
    "Do not enable paper order submission yet.",
    "Do not place paper orders yet.",
    "Do not implement or enable live trading.",
    "Do not change strategy thresholds from this preflight.",
    "Do not change position sizing from this preflight.",
    "Do not change scheduler cadence from this preflight.",
]


def build_paper_broker_preflight_report(
    *,
    broker_contract_report: dict | None = None,
    acceptance_report: dict | None = None,
    strategy_quality_decision_report: dict | None = None,
    state_snapshot: dict | None = None,
    now_provider: Callable[[], datetime] | None = None,
) -> dict:
    """Build a JSON-safe read-only report for future paper adapter planning."""

    now = _current_time(now_provider)
    errors: list[str] = []
    warnings: list[str] = []

    contract = _broker_contract_checks(broker_contract_report)
    boundary = _scheduler_lifecycle_boundary(acceptance_report)
    state_safety = _state_safety(state_snapshot, acceptance_report)
    blockers = _readiness_blockers(contract, boundary, state_safety)

    if _input_success(acceptance_report, "acceptance_report") is False:
        warnings.append("acceptance_report input was unsuccessful")
    if (
        _input_success(strategy_quality_decision_report, "strategy_quality_decision_report")
        is False
    ):
        warnings.append("strategy_quality_decision_report input was unsuccessful")

    ready = not blockers
    recommendations = _recommendations(ready)
    report = {
        "dry_run": True,
        "paper_broker_preflight_report": True,
        "generated_at": now.isoformat(),
        "inputs": {
            "broker_contract_available": bool(contract["broker_adapter_protocol_present"]),
            "acceptance_report_success": _input_success(
                acceptance_report, "acceptance_report"
            ),
            "strategy_quality_decision_success": _input_success(
                strategy_quality_decision_report,
                "strategy_quality_decision_report",
            ),
        },
        "broker_contract": contract,
        "scheduler_lifecycle_boundary": boundary,
        "state_safety": state_safety,
        "readiness_for_next_phase": {
            "ready_to_design_ibkr_paper_adapter": ready,
            "blockers": blockers,
            "warnings": warnings[:],
        },
        "recommendations": recommendations,
        "safety": {
            "broker_calls_enabled": False,
            "market_data_enabled": False,
            "external_fetch_enabled": False,
            "paper_live_orders_enabled": False,
            "live_orders_enabled": False,
            "scheduler_changes_enabled": False,
        },
        "success": True,
        "errors": errors,
        "warnings": warnings,
    }
    return _json_safe(report)


def _broker_contract_checks(broker_contract_report: dict | None) -> dict[str, bool]:
    if isinstance(broker_contract_report, dict):
        source = broker_contract_report.get("broker_contract")
        if isinstance(source, dict):
            return {
                "broker_adapter_protocol_present": bool(
                    source.get("broker_adapter_protocol_present")
                ),
                "required_methods_present": bool(source.get("required_methods_present")),
                "result_shapes_present": bool(source.get("result_shapes_present")),
                "raw_field_json_safe_contract_present": bool(
                    source.get("raw_field_json_safe_contract_present")
                ),
                "live_mode_rejected": bool(source.get("live_mode_rejected")),
                "paper_mode_allowed": bool(source.get("paper_mode_allowed")),
                "dry_run_mode_allowed": bool(source.get("dry_run_mode_allowed")),
                "fake_adapter_contract_present": bool(
                    source.get("fake_adapter_contract_present")
                ),
            }

    protocol_present = inspect.isclass(BrokerAdapter)
    required_methods = protocol_present and all(
        callable(getattr(BrokerAdapter, name, None)) for name in _REQUIRED_METHODS
    )
    result_shapes = all(_result_shape_present(cls) for cls in _RESULT_SHAPES)
    raw_doc = " ".join(
        str(part or "")
        for part in (
            inspect.getdoc(broker_adapter_module),
            inspect.getdoc(broker_adapter_module.assert_json_safe_raw),
            inspect.getdoc(BrokerAdapter),
        )
    ).lower()
    raw_contract = (
        "raw" in raw_doc
        and "json-safe" in raw_doc
        and "proprietary broker objects" in raw_doc
    )
    return {
        "broker_adapter_protocol_present": protocol_present,
        "required_methods_present": required_methods,
        "result_shapes_present": result_shapes,
        "raw_field_json_safe_contract_present": raw_contract,
        "live_mode_rejected": _mode_rejected("LIVE"),
        "paper_mode_allowed": _mode_allowed("PAPER"),
        "dry_run_mode_allowed": _mode_allowed("DRY_RUN"),
        "fake_adapter_contract_present": _fake_adapter_contract_present(),
    }


def _result_shape_present(cls: type) -> bool:
    if not is_dataclass(cls):
        return False
    names = {field.name for field in fields(cls)}
    return "raw" in names and callable(getattr(cls, "to_dict", None))


def _mode_allowed(mode: str) -> bool:
    try:
        return validate_broker_mode(mode) == mode
    except BrokerModeError:
        return False


def _mode_rejected(mode: str) -> bool:
    try:
        validate_broker_mode(mode)
    except BrokerModeError:
        return True
    return False


def _fake_adapter_contract_present() -> bool:
    return inspect.isclass(NullBrokerAdapter) and all(
        callable(getattr(NullBrokerAdapter, name, None)) for name in _REQUIRED_METHODS
    )


def _scheduler_lifecycle_boundary(acceptance_report: dict | None) -> dict[str, bool]:
    override = _dict_value(acceptance_report, "scheduler_lifecycle_boundary")
    if override:
        return {
            "adapter_wired_into_daemon": bool(override.get("adapter_wired_into_daemon")),
            "adapter_wired_into_scheduler": bool(
                override.get("adapter_wired_into_scheduler")
            ),
            "adapter_wired_into_lifecycle_jobs": bool(
                override.get("adapter_wired_into_lifecycle_jobs")
            ),
            "lifecycle_cadence_intent_level_only": bool(
                override.get("lifecycle_cadence_intent_level_only")
            ),
            "fill_simulation_scheduled": bool(override.get("fill_simulation_scheduled")),
            "position_transition_scheduled": bool(
                override.get("position_transition_scheduled")
            ),
        }

    lifecycle_jobs = tuple(getattr(scheduler_cadence, "STAGE4B_LIFECYCLE_JOB_IDS", ()))
    fill_scheduled = JOB_DRY_RUN_CONFIRM_FILLS in lifecycle_jobs
    transition_scheduled = JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS in lifecycle_jobs
    intent_level_only = bool(lifecycle_jobs) and not fill_scheduled and not transition_scheduled
    return {
        "adapter_wired_into_daemon": False,
        "adapter_wired_into_scheduler": False,
        "adapter_wired_into_lifecycle_jobs": False,
        "lifecycle_cadence_intent_level_only": intent_level_only,
        "fill_simulation_scheduled": fill_scheduled,
        "position_transition_scheduled": transition_scheduled,
    }


def _state_safety(
    state_snapshot: dict | None,
    acceptance_report: dict | None,
) -> dict[str, Any]:
    state = state_snapshot if isinstance(state_snapshot, dict) else None
    acceptance_state = _dict_value(acceptance_report, "state")
    startup_gate = _dict_value(acceptance_report, "startup_gate")
    if state is None and acceptance_state:
        state = acceptance_state

    return {
        "unresolved_needs_reconciliation_count": _needs_reconciliation_count(state),
        "active_intents_count": _count_or_none(
            state,
            "active_intents_count",
            fallback=lambda payload: _active_intents_count(payload),
        ),
        "open_positions_count": _count_or_none(
            state,
            "open_positions_count",
            fallback=lambda payload: _open_positions_count(payload),
        ),
        "active_halt": _active_halt(state, startup_gate),
    }


def _readiness_blockers(
    contract: dict[str, bool],
    boundary: dict[str, bool],
    state_safety: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    required_contract = (
        "broker_adapter_protocol_present",
        "required_methods_present",
        "result_shapes_present",
        "raw_field_json_safe_contract_present",
        "live_mode_rejected",
        "paper_mode_allowed",
        "dry_run_mode_allowed",
        "fake_adapter_contract_present",
    )
    for key in required_contract:
        if contract.get(key) is not True:
            blockers.append(f"broker contract check failed: {key}")

    for key in (
        "adapter_wired_into_daemon",
        "adapter_wired_into_scheduler",
        "adapter_wired_into_lifecycle_jobs",
        "fill_simulation_scheduled",
        "position_transition_scheduled",
    ):
        if boundary.get(key) is True:
            blockers.append(f"scheduler lifecycle boundary violated: {key}")
    if boundary.get("lifecycle_cadence_intent_level_only") is not True:
        blockers.append("lifecycle cadence intent-level-only evidence is missing")

    reconciliation_count = state_safety.get("unresolved_needs_reconciliation_count")
    if reconciliation_count is None:
        blockers.append("state safety snapshot unavailable for NEEDS_RECONCILIATION")
    elif reconciliation_count > 0:
        blockers.append("unresolved NEEDS_RECONCILIATION records exist")

    active_halt = state_safety.get("active_halt")
    if active_halt is None:
        blockers.append("halt state unavailable")
    elif active_halt is True:
        blockers.append("active halt is present")
    return blockers


def _recommendations(ready: bool) -> dict[str, list[str]]:
    steps = [
        "Design IBKR paper adapter behind BrokerAdapter Protocol.",
        "Keep adapter unmounted until 4D-3 explicit paper adapter implementation.",
        "Add paper adapter unit tests before any scheduler wiring.",
        "Verify AyoBot paper gateway connectivity manually outside this phase.",
        "Do not enable paper order submission until 4E burn-in gate.",
    ]
    if not ready:
        steps.insert(0, "Resolve preflight blockers before designing adapter implementation.")
    return {
        "ordered_next_steps": steps,
        "do_not_do_yet": _DO_NOT_DO_YET[:],
    }


def _input_success(report: dict | None, marker: str) -> bool | None:
    if not isinstance(report, dict) or report.get(marker) is not True:
        return None
    return report.get("success") is True


def _dict_value(payload: dict | None, key: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _needs_reconciliation_count(state: dict | None) -> int | None:
    if not isinstance(state, dict):
        return None
    explicit = state.get("unresolved_needs_reconciliation_count")
    if isinstance(explicit, int) and explicit >= 0:
        return explicit
    total = 0
    found_collection = False
    for key in ("positions", "order_intents", "close_intents"):
        records = _records(state.get(key))
        if records is None:
            continue
        found_collection = True
        total += sum(
            1
            for record in records
            if isinstance(record, dict) and record.get("status") == "NEEDS_RECONCILIATION"
        )
    return total if found_collection else None


def _count_or_none(
    state: dict | None,
    key: str,
    *,
    fallback: Callable[[dict], int | None],
) -> int | None:
    if not isinstance(state, dict):
        return None
    explicit = state.get(key)
    if isinstance(explicit, int) and explicit >= 0:
        return explicit
    return fallback(state)


def _active_intents_count(state: dict) -> int | None:
    total = 0
    found_collection = False
    for key in ("order_intents", "close_intents"):
        records = _records(state.get(key))
        if records is None:
            continue
        found_collection = True
        total += sum(
            1
            for record in records
            if isinstance(record, dict) and record.get("status") in _ACTIVE_INTENT_STATUSES
        )
    return total if found_collection else None


def _open_positions_count(state: dict) -> int | None:
    records = _records(state.get("positions"))
    if records is None:
        return None
    return sum(
        1
        for record in records
        if isinstance(record, dict) and record.get("status") == "open"
    )


def _records(collection: Any) -> list[Any] | None:
    if isinstance(collection, dict):
        return list(collection.values())
    if isinstance(collection, list):
        return collection
    return None


def _active_halt(state: dict | None, startup_gate: dict | None) -> bool | None:
    for payload, key in ((state, "active_halt"), (state, "halt_active"), (startup_gate, "halt_active")):
        if isinstance(payload, dict) and isinstance(payload.get(key), bool):
            return payload.get(key)
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
