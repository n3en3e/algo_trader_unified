"""Pure Stage 4G-6 manual paper lifecycle write acceptance report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
REQUIRED_OPERATION_KEYS = (
    "sequence_number",
    "target",
    "operation",
    "client_order_id",
    "broker_order_id",
    "result",
)
STATE_TARGET = "StateStore"
LEDGER_TARGET = "Ledger"
LIFECYCLE_TARGET = "Lifecycle"
ORDERED_NEXT_STEPS = [
    "Begin Stage 4H controlled automated paper trading launch.",
    "Start Stage 4H with read-only automation readiness checks.",
    "Keep scheduler/lifecycle automation disabled until 4H explicitly enables one strategy.",
]
DO_NOT_DO_YET = [
    "Do not enable live trading.",
    "Do not enable all strategies at once.",
    "Do not change strategy thresholds.",
    "Do not change position sizing.",
    "Do not change scheduler cadence without a 4H gate.",
    "Do not bypass readiness gates.",
]
SAFETY_FALSE_FLAGS = (
    "live_orders_enabled",
    "market_data_enabled",
    "contract_qualification_enabled",
    "scheduler_changes_enabled",
    "daemon_wiring_enabled",
    "scheduler_wiring_enabled",
    "lifecycle_wiring_enabled",
    "automated_paper_trading_enabled",
    "lifecycle_transition_enabled",
)
WRITE_PLAN_FALSE_FLAGS = (
    "lifecycle_transition_enabled",
    "daemon_wiring_enabled",
    "scheduler_wiring_enabled",
)
PAPER_EVENT_MARKERS = ("paper", "lifecycle", "state_write", "write")


def build_stage4g6_lifecycle_write_acceptance_report(
    *,
    state_write_executor_report: dict | None,
    existing_state_snapshot: dict | None = None,
    ledger_snapshot: dict | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Evaluate a Stage 4G-5 executor report and optional manual snapshots."""

    try:
        return _json_safe(
            _build_report(
                state_write_executor_report=state_write_executor_report,
                existing_state_snapshot=existing_state_snapshot,
                ledger_snapshot=ledger_snapshot,
                now_provider=now_provider,
            )
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must stay data-only.
        generated_at = _generated_at(now_provider)
        message = f"unexpected report failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=generated_at,
                artifact_checks=_default_artifact_checks(),
                operation_checks=_default_operation_checks(),
                consistency_checks=_default_consistency_checks(),
                post_write_snapshot_checks=_default_snapshot_checks(),
                safety_checks=_default_safety_checks(),
                blockers=[message],
                warnings=[],
                errors=[message],
            )
        )


def _build_report(
    *,
    state_write_executor_report: Any,
    existing_state_snapshot: Any,
    ledger_snapshot: Any,
    now_provider: Callable[[], Any] | None,
) -> dict:
    generated_at = _generated_at(now_provider)
    report = state_write_executor_report if isinstance(state_write_executor_report, dict) else None
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if state_write_executor_report is None:
        blockers.append("state_write_executor_report missing")
    elif report is None:
        blockers.append("state_write_executor_report must be a dict")
        errors.append("state_write_executor_report must be a dict")

    artifact_checks, artifact_blockers = _artifact_checks(report)
    blockers.extend(artifact_blockers)
    errors.extend(_as_string_list(_mapping(report).get("errors")))

    operation_checks, ids, wrote_position, operation_blockers = _operation_checks(report)
    blockers.extend(operation_blockers)

    safety_checks, safety_blockers = _safety_checks(report)
    blockers.extend(safety_blockers)

    consistency_checks = {
        "client_order_id": ids.get("client_order_id"),
        "broker_order_id": ids.get("broker_order_id"),
        "client_order_id_consistent": operation_checks["applied_operation_ids_consistent"]
        and _present(ids.get("client_order_id")),
        "broker_order_id_consistent": operation_checks["applied_operation_ids_consistent"]
        and _present(ids.get("broker_order_id")),
        "state_snapshot_matches_when_provided": True,
        "ledger_snapshot_matches_when_provided": True,
    }
    if not consistency_checks["client_order_id_consistent"]:
        blockers.append("client_order_id missing or inconsistent")
    if not consistency_checks["broker_order_id_consistent"]:
        blockers.append("broker_order_id missing or inconsistent")

    state_checks, state_blockers, state_warnings = _state_snapshot_checks(
        existing_state_snapshot,
        client_order_id=ids.get("client_order_id"),
        broker_order_id=ids.get("broker_order_id"),
        wrote_position=wrote_position,
    )
    ledger_checks, ledger_blockers, ledger_warnings = _ledger_snapshot_checks(
        ledger_snapshot,
        client_order_id=ids.get("client_order_id"),
        broker_order_id=ids.get("broker_order_id"),
    )
    blockers.extend(state_blockers)
    blockers.extend(ledger_blockers)
    warnings.extend(state_warnings)
    warnings.extend(ledger_warnings)
    consistency_checks["state_snapshot_matches_when_provided"] = not state_blockers
    consistency_checks["ledger_snapshot_matches_when_provided"] = not ledger_blockers

    post_write_snapshot_checks = {
        **state_checks,
        **ledger_checks,
    }
    ready = (
        all(artifact_checks.values())
        and all(operation_checks.values())
        and all(consistency_checks[key] is True for key in (
            "client_order_id_consistent",
            "broker_order_id_consistent",
            "state_snapshot_matches_when_provided",
            "ledger_snapshot_matches_when_provided",
        ))
        and all(safety_checks.values())
        and not blockers
        and not errors
    )
    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        operation_checks=operation_checks,
        consistency_checks=consistency_checks,
        post_write_snapshot_checks=post_write_snapshot_checks,
        safety_checks=safety_checks,
        blockers=_dedupe(blockers),
        warnings=_dedupe(warnings),
        errors=_dedupe(errors),
        ready=ready,
    )


def _artifact_checks(report: dict[str, Any] | None) -> tuple[dict[str, bool], list[str]]:
    data = _mapping(report)
    execution = _mapping(data.get("execution"))
    rollback = _mapping(data.get("rollback"))
    readiness = _mapping(data.get("readiness_for_stage4g6"))
    gates = _mapping(data.get("gates"))
    checks = {
        "executor_report_present": report is not None,
        "executor_report_ready": (
            data.get("stage4g5_state_write_executor") is True
            and readiness.get("ready_to_build_manual_lifecycle_write_acceptance_report") is True
            and gates.get("passed") is True
        ),
        "executor_completed": execution.get("attempted") is True
        and execution.get("completed") is True,
        "state_store_write_succeeded": execution.get("state_store_write_succeeded") is True,
        "ledger_write_succeeded": execution.get("ledger_write_succeeded") is True,
        "rollback_not_required": rollback.get("rollback_required") is False
        and rollback.get("rollback_attempted") is False,
    }
    reasons = {
        "executor_report_present": "Stage 4G-5 executor report is missing",
        "executor_report_ready": "Stage 4G-5 executor report is not ready for Stage 4G-6",
        "executor_completed": "Stage 4G-5 executor did not complete",
        "state_store_write_succeeded": "StateStore write did not succeed",
        "ledger_write_succeeded": "ledger write did not succeed",
        "rollback_not_required": "rollback is required or was attempted",
    }
    return checks, [reasons[key] for key, value in checks.items() if value is not True]


def _operation_checks(
    report: dict[str, Any] | None,
) -> tuple[dict[str, bool], dict[str, Any], bool, list[str]]:
    applied = _as_list(_mapping(report).get("applied_operations"))
    skipped = _as_list(_mapping(report).get("skipped_operations"))
    execution = _mapping(_mapping(report).get("execution"))
    blockers: list[str] = []
    ids: dict[str, Any] = {"client_order_id": None, "broker_order_id": None}
    schema_valid = bool(applied)
    ids_consistent = bool(applied)
    lifecycle_clean = execution.get("lifecycle_transition_executed") is False
    phases: list[int] = []
    state_seen = False
    ledger_seen = False
    wrote_position = False

    for index, item in enumerate(applied):
        if not isinstance(item, dict):
            schema_valid = False
            ids_consistent = False
            blockers.append(f"applied_operations[{index}] must be a dict")
            continue
        missing = [key for key in REQUIRED_OPERATION_KEYS if key not in item]
        if missing:
            schema_valid = False
            blockers.append(
                f"applied_operations[{index}] missing required fields: {', '.join(missing)}"
            )
        target = item.get("target")
        operation = item.get("operation")
        client_order_id = item.get("client_order_id")
        broker_order_id = item.get("broker_order_id")
        if not _present(client_order_id) or not _present(broker_order_id):
            ids_consistent = False
            blockers.append(f"applied_operations[{index}] missing written IDs")
        elif ids["client_order_id"] is None and ids["broker_order_id"] is None:
            ids["client_order_id"] = client_order_id
            ids["broker_order_id"] = broker_order_id
        elif (
            str(client_order_id) != str(ids["client_order_id"])
            or str(broker_order_id) != str(ids["broker_order_id"])
        ):
            ids_consistent = False
            blockers.append(f"applied_operations[{index}] references different written IDs")

        phase = _operation_phase(target, operation)
        if phase is None:
            schema_valid = False
            blockers.append(f"applied_operations[{index}] has unrecognized target or operation")
        else:
            phases.append(phase)
        if target == STATE_TARGET:
            state_seen = True
            if operation == "upsert_position":
                wrote_position = True
        if target == LEDGER_TARGET:
            ledger_seen = True
        if target == LIFECYCLE_TARGET or _suggests_transition(operation):
            lifecycle_clean = False
            blockers.append(f"applied_operations[{index}] indicates lifecycle transition execution")

    deterministic_order = bool(phases) and phases == sorted(phases)
    checks = {
        "applied_operations_present": bool(applied),
        "skipped_operations_empty": len(skipped) == 0,
        "state_store_operation_present": state_seen,
        "ledger_operation_present": ledger_seen,
        "lifecycle_transition_not_executed": lifecycle_clean,
        "deterministic_operation_order": deterministic_order,
        "applied_operation_schema_valid": schema_valid,
        "applied_operation_ids_consistent": ids_consistent,
    }
    reasons = {
        "applied_operations_present": "applied_operations are required",
        "skipped_operations_empty": "skipped_operations must be empty",
        "state_store_operation_present": "at least one StateStore applied operation is required",
        "ledger_operation_present": "at least one Ledger applied operation is required",
        "lifecycle_transition_not_executed": "lifecycle transition execution is forbidden",
        "deterministic_operation_order": "applied operations are not in deterministic order",
        "applied_operation_schema_valid": "applied operation schema is invalid",
        "applied_operation_ids_consistent": "applied operation IDs are inconsistent",
    }
    blockers.extend([reasons[key] for key, value in checks.items() if value is not True])
    return checks, ids, wrote_position, _dedupe(blockers)


def _operation_phase(target: Any, operation: Any) -> int | None:
    if target == STATE_TARGET and operation == "upsert_order":
        return 0
    if target == STATE_TARGET and operation == "upsert_position":
        return 1
    if target == LEDGER_TARGET:
        return 2
    if target == LIFECYCLE_TARGET:
        return 3
    return None


def _suggests_transition(operation: Any) -> bool:
    text = str(operation or "").lower()
    return "transition" in text or text.startswith("record_lifecycle")


def _state_snapshot_checks(
    snapshot: Any,
    *,
    client_order_id: Any,
    broker_order_id: Any,
    wrote_position: bool,
) -> tuple[dict[str, Any], list[str], list[str]]:
    present = isinstance(snapshot, dict)
    data = snapshot if present else {}
    blockers: list[str] = []
    warnings: list[str] = []
    state_order_seen = False
    state_position_seen = False
    unresolved = data.get("unresolved_needs_reconciliation_count")
    if unresolved is None:
        unresolved = data.get("needs_reconciliation_count")
    active_halt = data.get("active_halt")

    if not present:
        warnings.append("state snapshot missing; manual post-write state verification is still required")
    else:
        order_sources = [
            data.get("order_records"),
            data.get("paper_orders"),
            data.get("open_orders"),
            {
                "client_order_ids": data.get("client_order_ids"),
                "broker_order_ids": data.get("broker_order_ids"),
            },
        ]
        if any(_has_explicit_records(source) for source in order_sources):
            state_order_seen = any(
                _contains_id_pair(source, client_order_id, broker_order_id)
                for source in order_sources
            )
            if not state_order_seen:
                blockers.append("state snapshot explicitly contradicts written order IDs")
        else:
            warnings.append("state snapshot has no explicit order records to verify")

        position_sources = [
            data.get("position_records"),
            data.get("paper_positions"),
            data.get("open_positions"),
        ]
        if wrote_position and any(_has_explicit_records(source) for source in position_sources):
            state_position_seen = any(
                _contains_id_pair(source, client_order_id, broker_order_id)
                for source in position_sources
            )
            if not state_position_seen:
                blockers.append("state snapshot explicitly contradicts written position IDs")
        elif wrote_position:
            warnings.append("state snapshot has no explicit position records to verify")

    if _positive_int(unresolved):
        blockers.append("unresolved NEEDS_RECONCILIATION is present after write")
    if active_halt:
        blockers.append("active halt is present after write")

    return (
        {
            "state_snapshot_present": present,
            "state_order_seen": state_order_seen,
            "state_position_seen": state_position_seen,
            "unresolved_needs_reconciliation_count": unresolved,
            "active_halt": active_halt,
        },
        blockers,
        warnings,
    )


def _ledger_snapshot_checks(
    snapshot: Any,
    *,
    client_order_id: Any,
    broker_order_id: Any,
) -> tuple[dict[str, Any], list[str], list[str]]:
    present = isinstance(snapshot, dict)
    data = snapshot if present else {}
    blockers: list[str] = []
    warnings: list[str] = []
    event_seen = False
    if not present:
        warnings.append("ledger snapshot missing; manual ledger verification is still required")
    else:
        sources = [
            data.get("events"),
            data.get("paper_lifecycle_events"),
            {
                "client_order_ids": data.get("client_order_ids"),
                "broker_order_ids": data.get("broker_order_ids"),
            },
        ]
        if any(_has_explicit_records(source) for source in sources):
            event_seen = any(
                _contains_id_pair(source, client_order_id, broker_order_id)
                for source in sources
            )
            if not event_seen:
                blockers.append("ledger snapshot explicitly contradicts written IDs")
        else:
            warnings.append("ledger snapshot has no explicit events to verify")

        event_types = _as_list(data.get("event_types"))
        if event_types and not any(_is_paper_write_event_type(value) for value in event_types):
            blockers.append("ledger snapshot event types are not paper lifecycle write related")
    return (
        {
            "ledger_snapshot_present": present,
            "ledger_event_seen": event_seen,
        },
        blockers,
        warnings,
    )


def _safety_checks(report: dict[str, Any] | None) -> tuple[dict[str, bool], list[str]]:
    data = _mapping(report)
    safety = _mapping(data.get("safety"))
    write_plan = _mapping(data.get("write_plan"))
    checks = {
        "no_live_orders": safety.get("live_orders_enabled") is False,
        "no_market_data": safety.get("market_data_enabled") is False,
        "no_contract_qualification": safety.get("contract_qualification_enabled") is False,
        "no_scheduler_changes": (
            safety.get("scheduler_changes_enabled") is False
            and write_plan.get("scheduler_wiring_enabled") is False
        ),
        "no_lifecycle_wiring": safety.get("lifecycle_wiring_enabled") is False
        and write_plan.get("lifecycle_transition_enabled") is False,
        "no_automated_paper_trading": safety.get("automated_paper_trading_enabled") is False,
        "no_lifecycle_transition_execution": _mapping(data.get("execution")).get(
            "lifecycle_transition_executed"
        )
        is False,
    }
    for key in SAFETY_FALSE_FLAGS:
        if _contains_true_flag(data, key):
            if key == "live_orders_enabled":
                checks["no_live_orders"] = False
            elif key == "market_data_enabled":
                checks["no_market_data"] = False
            elif key == "contract_qualification_enabled":
                checks["no_contract_qualification"] = False
            elif key in ("scheduler_changes_enabled", "daemon_wiring_enabled", "scheduler_wiring_enabled"):
                checks["no_scheduler_changes"] = False
            elif key in ("lifecycle_wiring_enabled", "lifecycle_transition_enabled"):
                checks["no_lifecycle_wiring"] = False
            elif key == "automated_paper_trading_enabled":
                checks["no_automated_paper_trading"] = False
    for key in WRITE_PLAN_FALSE_FLAGS:
        if write_plan.get(key) is not False:
            if key in ("daemon_wiring_enabled", "scheduler_wiring_enabled"):
                checks["no_scheduler_changes"] = False
            else:
                checks["no_lifecycle_wiring"] = False
    reasons = {
        "no_live_orders": "live order flag is enabled",
        "no_market_data": "market-data flag is enabled",
        "no_contract_qualification": "contract qualification flag is enabled",
        "no_scheduler_changes": "scheduler or daemon change flag is enabled",
        "no_lifecycle_wiring": "lifecycle wiring flag is enabled",
        "no_automated_paper_trading": "automated paper trading flag is enabled",
        "no_lifecycle_transition_execution": "lifecycle transition execution is enabled",
    }
    return checks, [reasons[key] for key, value in checks.items() if value is not True]


def _contains_id_pair(value: Any, client_order_id: Any, broker_order_id: Any) -> bool:
    if not _present(client_order_id) or not _present(broker_order_id):
        return False
    if isinstance(value, dict):
        if _ids_match(value.get("client_order_id"), client_order_id) and _ids_match(
            value.get("broker_order_id"), broker_order_id
        ):
            return True
        client_ids = _as_list(value.get("client_order_ids"))
        broker_ids = _as_list(value.get("broker_order_ids"))
        if any(_ids_match(item, client_order_id) for item in client_ids) and any(
            _ids_match(item, broker_order_id) for item in broker_ids
        ):
            return True
        return any(_contains_id_pair(item, client_order_id, broker_order_id) for item in value.values())
    if isinstance(value, list):
        return any(_contains_id_pair(item, client_order_id, broker_order_id) for item in value)
    return False


def _has_explicit_records(value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    if isinstance(value, dict):
        if value.get("client_order_id") is not None or value.get("broker_order_id") is not None:
            return True
        if value.get("client_order_ids") is not None or value.get("broker_order_ids") is not None:
            return True
        return any(_has_explicit_records(item) for item in value.values())
    if isinstance(value, list):
        return bool(value)
    return False


def _is_paper_write_event_type(value: Any) -> bool:
    text = str(value or "").lower()
    return "paper" in text and any(marker in text for marker in PAPER_EVENT_MARKERS)


def _ids_match(actual: Any, expected: Any) -> bool:
    return _present(actual) and _present(expected) and str(actual) == str(expected)


def _positive_int(value: Any) -> bool:
    if isinstance(value, bool):
        return value is True
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    operation_checks: dict[str, Any],
    consistency_checks: dict[str, Any],
    post_write_snapshot_checks: dict[str, Any],
    safety_checks: dict[str, Any],
    blockers: list[str],
    warnings: list[str],
    errors: list[str],
    ready: bool = False,
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "stage4g6_lifecycle_write_acceptance_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "operation_checks": operation_checks,
        "consistency_checks": consistency_checks,
        "post_write_snapshot_checks": post_write_snapshot_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4h": {
            "ready_to_begin_controlled_automated_paper_trading_launch": ready,
            "blockers": list(blockers),
            "warnings": list(warnings),
        },
        "recommendations": {
            "ordered_next_steps": list(ORDERED_NEXT_STEPS),
            "do_not_do_yet": list(DO_NOT_DO_YET),
        },
        "success": ready,
        "errors": list(errors),
        "warnings": list(warnings),
    }


def _default_artifact_checks() -> dict[str, bool]:
    return {
        "executor_report_present": False,
        "executor_report_ready": False,
        "executor_completed": False,
        "state_store_write_succeeded": False,
        "ledger_write_succeeded": False,
        "rollback_not_required": False,
    }


def _default_operation_checks() -> dict[str, bool]:
    return {
        "applied_operations_present": False,
        "skipped_operations_empty": False,
        "state_store_operation_present": False,
        "ledger_operation_present": False,
        "lifecycle_transition_not_executed": False,
        "deterministic_operation_order": False,
        "applied_operation_schema_valid": False,
        "applied_operation_ids_consistent": False,
    }


def _default_consistency_checks() -> dict[str, Any]:
    return {
        "client_order_id": None,
        "broker_order_id": None,
        "client_order_id_consistent": False,
        "broker_order_id_consistent": False,
        "state_snapshot_matches_when_provided": False,
        "ledger_snapshot_matches_when_provided": False,
    }


def _default_snapshot_checks() -> dict[str, Any]:
    return {
        "state_snapshot_present": False,
        "state_order_seen": False,
        "state_position_seen": False,
        "ledger_snapshot_present": False,
        "ledger_event_seen": False,
        "unresolved_needs_reconciliation_count": None,
        "active_halt": None,
    }


def _default_safety_checks() -> dict[str, bool]:
    return {
        "no_live_orders": False,
        "no_market_data": False,
        "no_contract_qualification": False,
        "no_scheduler_changes": False,
        "no_lifecycle_wiring": False,
        "no_automated_paper_trading": False,
        "no_lifecycle_transition_execution": False,
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _as_string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _present(value: Any) -> bool:
    return value not in (None, "")


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _contains_true_flag(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return any(
            (item_key == key and item is True) or _contains_true_flag(item, key)
            for item_key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_true_flag(item, key) for item in value)
    return False


def _generated_at(now_provider: Callable[[], Any] | None) -> str:
    if now_provider is None:
        return DEFAULT_GENERATED_AT
    value = now_provider()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _json_safe(value: Any) -> Any:
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
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return "<object>"
