"""Pure Stage 4G-4 manual paper lifecycle state write dry-run reporting."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
RECOGNIZED_OPERATIONS = ("upsert_order", "upsert_position")
WRITE_PLAN_KEYS = (
    "state_store_write_enabled",
    "ledger_write_enabled",
    "lifecycle_transition_enabled",
    "daemon_wiring_enabled",
    "scheduler_wiring_enabled",
)
SAFETY_FLAG_KEYS = {
    "live_orders_enabled": "no_live_orders",
    "market_data_enabled": "no_market_data",
    "contract_qualification_enabled": "no_contract_qualification",
    "scheduler_changes_enabled": "no_scheduler_changes",
    "daemon_wiring_enabled": "no_scheduler_changes",
    "scheduler_wiring_enabled": "no_scheduler_changes",
    "lifecycle_wiring_enabled": "no_lifecycle_wiring",
    "state_mutation_enabled": "no_state_mutation",
    "state_store_write_enabled": "no_state_mutation",
    "ledger_writes_enabled": "no_ledger_writes",
    "ledger_write_enabled": "no_ledger_writes",
    "lifecycle_transition_enabled": "no_lifecycle_wiring",
}
SAFETY_RESULT_KEYS = (
    "no_live_orders",
    "no_market_data",
    "no_contract_qualification",
    "no_scheduler_changes",
    "no_lifecycle_wiring",
    "no_state_mutation",
    "no_ledger_writes",
)
REQUIRED_ACKNOWLEDGEMENTS = [
    "I understand this will write paper lifecycle state.",
    "I understand this will write ledger events.",
    "I understand this is still PAPER only.",
    "I understand this does not enable scheduler automation.",
    "I reviewed the proposed StateStore and ledger payloads.",
]
ORDERED_NEXT_STEPS = [
    "Build Stage 4G-5 manual state write executor behind explicit operator gates.",
    "Keep scheduler/lifecycle automation disabled.",
    "Do not write StateStore until the 4G-5 executor is separately reviewed and explicitly approved.",
    "Do not append ledger events until explicitly approved.",
    "Do not begin live trading.",
]
DO_NOT_DO_YET = [
    "Do not write StateStore in Stage 4G-4.",
    "Do not append ledger events in Stage 4G-4.",
    "Do not execute lifecycle transitions.",
    "Do not enable scheduler or lifecycle automation.",
    "Do not begin live trading.",
]


def build_stage4g4_state_write_dry_run(
    *,
    state_write_proposal_report: dict | None,
    existing_state_snapshot: dict | None = None,
    operator_acknowledgements: list[str] | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a read-only dry-run packet from a Stage 4G-3 proposal report."""

    try:
        return _build_report(
            state_write_proposal_report=state_write_proposal_report,
            existing_state_snapshot=existing_state_snapshot,
            operator_acknowledgements=operator_acknowledgements,
            now_provider=now_provider,
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must stay data-only.
        generated_at = _generated_at(now_provider)
        return _json_safe(
            {
                "dry_run": True,
                "stage4g4_state_write_dry_run": True,
                "generated_at": generated_at,
                "artifact_checks": _artifact_checks(None),
                "operation_schema_checks": _operation_schema_checks([]),
                "acknowledgement_checks": _acknowledgement_checks(None),
                "dry_run_packet": _dry_run_packet(
                    available=False,
                    dry_run_operations=[],
                    final_gate=_final_write_gate(
                        allowed=False,
                        missing_acknowledgements=list(REQUIRED_ACKNOWLEDGEMENTS),
                        blockers=["unexpected report failure"],
                        warnings=[],
                    ),
                ),
                "write_plan": _write_plan(),
                "conflict_checks": _empty_conflict_checks(),
                "safety_checks": _default_safety_checks(),
                "state_snapshot_checks": _empty_state_snapshot_checks(),
                "readiness_for_stage4g5": {
                    "ready_to_build_manual_state_write_executor": False,
                    "blockers": ["unexpected report failure"],
                    "warnings": [],
                },
                "recommendations": _recommendations(),
                "success": False,
                "errors": [f"unexpected report failure: {type(exc).__name__}: {exc}"],
                "warnings": [],
            }
        )


def _build_report(
    *,
    state_write_proposal_report: dict | None,
    existing_state_snapshot: dict | None,
    operator_acknowledgements: list[str] | None,
    now_provider: Callable[[], Any] | None,
) -> dict:
    generated_at = _generated_at(now_provider)
    proposal_report = (
        state_write_proposal_report
        if isinstance(state_write_proposal_report, dict)
        else None
    )
    snapshot = existing_state_snapshot if isinstance(existing_state_snapshot, dict) else None
    proposal = _mapping(_mapping(proposal_report).get("proposal"))
    state_operations = _as_list(proposal.get("proposed_state_store_operations"))
    ledger_events = _as_list(proposal.get("proposed_ledger_events"))
    lifecycle_transition = proposal.get("proposed_lifecycle_transition")

    errors: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []
    if state_write_proposal_report is not None and proposal_report is None:
        errors.append("state_write_proposal_report must be a dict")
        blockers.append("state_write_proposal_report malformed")
    if existing_state_snapshot is not None and snapshot is None:
        warnings.append("existing_state_snapshot ignored because it is not a dict")
    if operator_acknowledgements is not None and not isinstance(
        operator_acknowledgements, list
    ):
        warnings.append("operator_acknowledgements ignored because it is not a list")

    artifact_checks = _artifact_checks(proposal_report)
    if not artifact_checks["state_write_proposal_present"]:
        blockers.append("state_write_proposal_report missing")
    if not artifact_checks["state_write_proposal_ready"]:
        blockers.append("Stage 4G-3 proposal is not ready for Stage 4G-4")
    if not artifact_checks["proposal_available"]:
        blockers.append("Stage 4G-3 proposal is not available")
    if not artifact_checks["state_store_operations_present"]:
        blockers.append("proposed_state_store_operations are required")
    if not artifact_checks["ledger_events_present"]:
        blockers.append("proposed_ledger_events are required")
    if not artifact_checks["lifecycle_transition_present"]:
        blockers.append("proposed_lifecycle_transition is required")

    write_plan_blockers = _validate_write_plan(proposal_report)
    blockers.extend(write_plan_blockers)
    safety_checks = _safety_checks([proposal_report], blockers)
    state_checks, state_blockers = _state_snapshot_checks(snapshot)
    blockers.extend(state_blockers)

    operation_schema_checks = _operation_schema_checks(state_operations)
    blockers.extend(_operation_schema_blockers(state_operations, operation_schema_checks))
    transition_supported = isinstance(lifecycle_transition, dict)
    if not transition_supported:
        blockers.append("proposed_lifecycle_transition must be a dict")

    conflict_checks = _conflict_checks(snapshot, state_operations)
    blockers.extend(conflict_checks["conflict_reasons"])

    acknowledgement_checks = _acknowledgement_checks(operator_acknowledgements)
    if acknowledgement_checks["missing"]:
        blockers.append("required operator acknowledgements missing")

    dry_run_operations = _dry_run_operations(
        state_operations=state_operations,
        ledger_events=ledger_events,
        lifecycle_transition=lifecycle_transition,
    )
    rollback = _rollback_simulation()
    dry_run_available = (
        artifact_checks["state_write_proposal_present"]
        and artifact_checks["state_write_proposal_ready"]
        and artifact_checks["proposal_available"]
        and artifact_checks["state_store_operations_present"]
        and artifact_checks["ledger_events_present"]
        and artifact_checks["lifecycle_transition_present"]
        and operation_schema_checks["operations_structured"]
        and operation_schema_checks["recognized_operations"]
        and operation_schema_checks["deterministic_operation_order"]
        and transition_supported
        and bool(dry_run_operations)
    )

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    output_write_plan = _write_plan()
    ready = (
        dry_run_available
        and not blocker_list
        and not error_list
        and all(output_write_plan[key] is False for key in WRITE_PLAN_KEYS)
        and all(safety_checks.values())
        and not conflict_checks["duplicate_client_order_id"]
        and not conflict_checks["duplicate_broker_order_id"]
        and not conflict_checks["duplicate_position_key"]
        and state_checks.get("unresolved_needs_reconciliation_count") in (None, 0)
        and state_checks.get("active_halt") in (None, False)
        and acknowledgement_checks["exact_match"] is True
        and rollback["available"] is True
        and "no automated rollback is supported in this phase"
        in rollback["rollback_limitations"]
    )
    final_gate = _final_write_gate(
        allowed=ready,
        missing_acknowledgements=acknowledgement_checks["missing"],
        blockers=blocker_list,
        warnings=warning_list,
    )

    return _json_safe(
        {
            "dry_run": True,
            "stage4g4_state_write_dry_run": True,
            "generated_at": generated_at,
            "artifact_checks": artifact_checks,
            "operation_schema_checks": operation_schema_checks,
            "acknowledgement_checks": acknowledgement_checks,
            "dry_run_packet": _dry_run_packet(
                available=dry_run_available,
                dry_run_operations=dry_run_operations,
                final_gate=final_gate,
            ),
            "write_plan": output_write_plan,
            "conflict_checks": conflict_checks,
            "safety_checks": safety_checks,
            "state_snapshot_checks": state_checks,
            "readiness_for_stage4g5": {
                "ready_to_build_manual_state_write_executor": ready,
                "blockers": blocker_list,
                "warnings": warning_list,
            },
            "recommendations": _recommendations(),
            "success": True,
            "errors": error_list,
            "warnings": warning_list,
        }
    )


def _artifact_checks(report: dict[str, Any] | None) -> dict[str, bool]:
    proposal = _mapping(_mapping(report).get("proposal"))
    readiness = _mapping(_mapping(report).get("readiness_for_stage4g4"))
    return {
        "state_write_proposal_present": report is not None
        and _mapping(report).get("stage4g3_state_write_proposal") is True,
        "state_write_proposal_ready": readiness.get(
            "ready_to_build_manual_state_write_dry_run"
        )
        is True,
        "proposal_available": proposal.get("available") is True,
        "state_store_operations_present": bool(
            _as_list(proposal.get("proposed_state_store_operations"))
        ),
        "ledger_events_present": bool(_as_list(proposal.get("proposed_ledger_events"))),
        "lifecycle_transition_present": proposal.get("proposed_lifecycle_transition")
        is not None,
    }


def _operation_schema_checks(operations: list[Any]) -> dict[str, bool]:
    structured = bool(operations) and all(
        isinstance(item, dict)
        and "operation" in item
        and "payload" in item
        and isinstance(item.get("payload"), dict)
        for item in operations
    )
    recognized = structured and all(
        item.get("operation") in RECOGNIZED_OPERATIONS
        for item in operations
        if isinstance(item, dict)
    )
    return {
        "operations_structured": structured,
        "recognized_operations": recognized,
        "deterministic_operation_order": _has_deterministic_operation_order(operations),
    }


def _operation_schema_blockers(
    operations: list[Any],
    checks: dict[str, bool],
) -> list[str]:
    blockers: list[str] = []
    if not checks["operations_structured"]:
        blockers.append("proposed_state_store_operations must be structured")
    if not checks["recognized_operations"]:
        blockers.append("unrecognized proposed StateStore operation")
    if not checks["deterministic_operation_order"]:
        blockers.append("proposed StateStore operations are not deterministically ordered")
    for index, item in enumerate(operations):
        if not isinstance(item, dict):
            blockers.append(f"proposed StateStore operation must be a dict at index {index}")
            continue
        if "operation" not in item:
            blockers.append(f"proposed StateStore operation missing operation at index {index}")
        if "payload" not in item:
            blockers.append(f"proposed StateStore operation missing payload at index {index}")
        elif not isinstance(item.get("payload"), dict):
            blockers.append(
                f"proposed StateStore operation payload must be a JSON-safe dict at index {index}"
            )
    return blockers


def _has_deterministic_operation_order(operations: list[Any]) -> bool:
    seen_position = False
    for item in operations:
        if not isinstance(item, dict):
            return False
        operation = item.get("operation")
        if operation == "upsert_position":
            seen_position = True
        elif operation == "upsert_order" and seen_position:
            return False
        elif operation not in RECOGNIZED_OPERATIONS:
            return False
    return bool(operations)


def _dry_run_operations(
    *,
    state_operations: list[Any],
    ledger_events: list[Any],
    lifecycle_transition: Any,
) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    ordered_state_operations = [
        item
        for operation_name in RECOGNIZED_OPERATIONS
        for item in state_operations
        if isinstance(item, dict) and item.get("operation") == operation_name
    ]
    for item in ordered_state_operations:
        operations.append(
            {
                "sequence_number": len(operations) + 1,
                "operation_type": "state_store_operation",
                "operation": item.get("operation"),
                "payload": _json_safe(item.get("payload")),
                "would_execute": False,
                "target": "StateStore",
            }
        )
    for event in ledger_events:
        operations.append(
            {
                "sequence_number": len(operations) + 1,
                "operation_type": "ledger_event",
                "operation": "append_ledger_event",
                "payload": _json_safe(_mapping(event)),
                "would_execute": False,
                "target": "Ledger",
            }
        )
    if isinstance(lifecycle_transition, dict):
        operations.append(
            {
                "sequence_number": len(operations) + 1,
                "operation_type": "lifecycle_transition",
                "operation": "record_lifecycle_transition",
                "payload": _json_safe(lifecycle_transition),
                "would_execute": False,
                "target": "Lifecycle",
            }
        )
    return operations


def _dry_run_packet(
    *,
    available: bool,
    dry_run_operations: list[dict[str, Any]],
    final_gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "available": available,
        "validation_steps": [
            "validate stage4g3 proposal identity",
            "validate write_plan disabled",
            "validate safety flags",
            "validate no duplicate IDs",
            "validate operator acknowledgements",
            "validate operation schema",
            "validate rollback plan",
        ],
        "dry_run_operations": dry_run_operations,
        "rollback_simulation": _rollback_simulation(),
        "final_write_gate": final_gate,
    }


def _rollback_simulation() -> dict[str, Any]:
    return {
        "available": True,
        "rollback_steps": [
            "Rollback requires manual StateStore/ledger file reversion using standard system backups."
        ],
        "rollback_limitations": [
            "no automated rollback is supported in this phase",
        ],
    }


def _final_write_gate(
    *,
    allowed: bool,
    missing_acknowledgements: list[str],
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "would_be_allowed_if_explicit_write_phase": allowed,
        "missing_acknowledgements": list(missing_acknowledgements),
        "blockers": list(blockers),
        "warnings": list(warnings),
    }


def _acknowledgement_checks(values: list[str] | None) -> dict[str, Any]:
    provided = [
        value.strip()
        for value in values
        if isinstance(value, str)
    ] if isinstance(values, list) else []
    missing = [value for value in REQUIRED_ACKNOWLEDGEMENTS if value not in provided]
    return {
        "required": list(REQUIRED_ACKNOWLEDGEMENTS),
        "provided": provided,
        "missing": missing,
        "exact_match": not missing,
    }


def _validate_write_plan(report: dict[str, Any] | None) -> list[str]:
    write_plan = _mapping(_mapping(report).get("write_plan"))
    blockers: list[str] = []
    for key in WRITE_PLAN_KEYS:
        if write_plan.get(key) is not False:
            blockers.append(f"Stage 4G-3 write_plan flag must be False: {key}")
    return blockers


def _write_plan() -> dict[str, bool]:
    return {key: False for key in WRITE_PLAN_KEYS}


def _conflict_checks(
    snapshot: dict[str, Any] | None,
    operations: list[Any],
) -> dict[str, Any]:
    proposed_client_ids = _payload_values(operations, "client_order_id")
    proposed_broker_ids = _payload_values(operations, "broker_order_id")
    proposed_position_keys = _payload_values(operations, "position_key")
    reasons: list[str] = []
    duplicate_client = False
    duplicate_broker = False
    duplicate_position = False
    if snapshot is not None:
        duplicate_client = any(
            _value_in_snapshot(
                snapshot,
                ("existing_client_order_ids", "open_order_client_ids"),
                value,
            )
            for value in proposed_client_ids
        )
        duplicate_broker = any(
            _value_in_snapshot(
                snapshot,
                ("existing_broker_order_ids", "open_order_broker_ids"),
                value,
            )
            for value in proposed_broker_ids
        )
        duplicate_position = any(
            _value_in_snapshot(snapshot, ("existing_position_keys",), value)
            for value in proposed_position_keys
        )
    if duplicate_client:
        reasons.append("duplicate client_order_id exists in state snapshot")
    if duplicate_broker:
        reasons.append("duplicate broker_order_id exists in state snapshot")
    if duplicate_position:
        reasons.append("duplicate position_key exists in state snapshot")
    return {
        "duplicate_client_order_id": duplicate_client,
        "duplicate_broker_order_id": duplicate_broker,
        "duplicate_position_key": duplicate_position,
        "conflict_reasons": reasons,
    }


def _empty_conflict_checks() -> dict[str, Any]:
    return {
        "duplicate_client_order_id": False,
        "duplicate_broker_order_id": False,
        "duplicate_position_key": False,
        "conflict_reasons": [],
    }


def _payload_values(operations: list[Any], key: str) -> list[Any]:
    values: list[Any] = []
    for item in operations:
        payload = _mapping(_mapping(item).get("payload"))
        value = payload.get(key)
        if _present(value):
            values.append(value)
    return values


def _safety_checks(reports: list[Any], blockers: list[str]) -> dict[str, bool]:
    checks = _default_safety_checks()
    for index, report in enumerate(reports):
        if not isinstance(report, dict):
            continue
        for source_key, result_key in SAFETY_FLAG_KEYS.items():
            if _contains_true_flag(report, source_key):
                checks[result_key] = False
                blockers.append(f"unsafe flag enabled: {source_key} in supplied report {index}")
        for result_key in SAFETY_RESULT_KEYS:
            if _contains_false_flag(report, result_key):
                checks[result_key] = False
                blockers.append(f"unsafe safety check failed: {result_key} in supplied report {index}")
    return checks


def _default_safety_checks() -> dict[str, bool]:
    return {
        "no_live_orders": True,
        "no_market_data": True,
        "no_contract_qualification": True,
        "no_scheduler_changes": True,
        "no_lifecycle_wiring": True,
        "no_state_mutation": True,
        "no_ledger_writes": True,
    }


def _state_snapshot_checks(snapshot: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    if snapshot is None:
        return _empty_state_snapshot_checks(), []
    checks = {
        "unresolved_needs_reconciliation_count": _first_int(
            snapshot,
            ("unresolved_needs_reconciliation_count", "needs_reconciliation_count"),
            default=_count_needs_reconciliation(snapshot),
        ),
        "active_halt": _active_halt(snapshot),
        "active_intents_count": _first_int(
            snapshot,
            ("active_intents_count", "open_intents_count"),
        ),
        "open_positions_count": _first_int(
            snapshot,
            ("open_positions_count", "positions_open_count"),
        ),
    }
    blockers: list[str] = []
    unresolved = checks.get("unresolved_needs_reconciliation_count")
    if isinstance(unresolved, int) and unresolved > 0:
        blockers.append("unresolved NEEDS_RECONCILIATION records exist")
    if checks.get("active_halt") is True:
        blockers.append("active halt is present")
    return checks, blockers


def _empty_state_snapshot_checks() -> dict[str, Any]:
    return {
        "unresolved_needs_reconciliation_count": None,
        "active_halt": None,
        "active_intents_count": None,
        "open_positions_count": None,
    }


def _value_in_snapshot(
    snapshot: dict[str, Any],
    keys: tuple[str, ...],
    value: Any,
) -> bool:
    if not _present(value):
        return False
    needle = str(value)
    for key in keys:
        values = snapshot.get(key)
        if isinstance(values, dict):
            haystack = list(values.keys()) + list(values.values())
        elif isinstance(values, (list, tuple, set)):
            haystack = list(values)
        else:
            haystack = []
        if needle in {str(item) for item in haystack if _present(item)}:
            return True
    return False


def _contains_true_flag(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return any(
            (item_key == key and item is True) or _contains_true_flag(item, key)
            for item_key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_true_flag(item, key) for item in value)
    return False


def _contains_false_flag(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return any(
            (item_key == key and item is False) or _contains_false_flag(item, key)
            for item_key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_false_flag(item, key) for item in value)
    return False


def _first_int(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    *,
    default: int | None = None,
) -> int | None:
    for key in keys:
        count = _count_value(payload.get(key))
        if count is not None:
            return count
    return default


def _count_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return None


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


def _present(value: Any) -> bool:
    return value not in (None, "")


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _recommendations() -> dict[str, list[str]]:
    return {
        "ordered_next_steps": list(ORDERED_NEXT_STEPS),
        "do_not_do_yet": list(DO_NOT_DO_YET),
    }


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
