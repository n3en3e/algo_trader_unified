"""Pure Stage 4G-3 manual paper lifecycle state write proposal reporting."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
SUPPORTED_LIFECYCLE_STATES = {
    "paper_order_submitted",
    "paper_order_filled",
    "paper_order_cancelled",
    "paper_order_submitted_unverified",
    "paper_cancel_requested_unverified",
}
UNSUPPORTED_LIFECYCLE_STATES = {
    "paper_order_partially_filled_review",
    "paper_order_rejected_or_inactive_review",
    "paper_order_unknown_status_review",
    "needs_reconciliation",
    "unsafe_artifact",
}
RECOGNIZED_OPERATIONS = {"upsert_order", "upsert_position"}
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
SAFETY_RESULT_KEYS = {
    "no_live_orders",
    "no_market_data",
    "no_contract_qualification",
    "no_scheduler_changes",
    "no_lifecycle_wiring",
    "no_state_mutation",
    "no_ledger_writes",
}
ORDERED_NEXT_STEPS = [
    "Build Stage 4G-4 manual state write dry run behind explicit operator gates.",
    "Keep scheduler/lifecycle automation disabled.",
    "Do not write StateStore until the 4G-4 dry run and later explicit write gate are reviewed.",
    "Do not append ledger events until explicitly approved.",
    "Do not begin live trading.",
]
DO_NOT_DO_YET = [
    "Do not write StateStore in Stage 4G-3.",
    "Do not append ledger events in Stage 4G-3.",
    "Do not execute lifecycle transitions.",
    "Do not enable scheduler or lifecycle automation.",
    "Do not begin live trading.",
]
OPERATOR_ACKNOWLEDGEMENTS = [
    "I understand this will write paper lifecycle state.",
    "I understand this will write ledger events.",
    "I understand this is still PAPER only.",
    "I understand this does not enable scheduler automation.",
    "I reviewed the proposed StateStore and ledger payloads.",
]


def build_stage4g3_state_write_proposal(
    *,
    lifecycle_state_preview_report: dict | None,
    existing_state_snapshot: dict | None = None,
    operator_notes: dict | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a read-only proposal packet from a Stage 4G-2 preview report."""

    try:
        return _build_report(
            lifecycle_state_preview_report=lifecycle_state_preview_report,
            existing_state_snapshot=existing_state_snapshot,
            operator_notes=operator_notes,
            now_provider=now_provider,
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must stay data-only.
        generated_at = _generated_at(now_provider)
        return _json_safe(
            {
                "dry_run": True,
                "stage4g3_state_write_proposal": True,
                "generated_at": generated_at,
                "artifact_checks": _artifact_checks(None),
                "input_summary": _empty_input_summary(),
                "operator_notes": {},
                "proposal": _empty_proposal(),
                "write_plan": _write_plan(),
                "conflict_checks": _empty_conflict_checks(),
                "safety_checks": _default_safety_checks(),
                "state_snapshot_checks": _empty_state_snapshot_checks(),
                "readiness_for_stage4g4": {
                    "ready_to_build_manual_state_write_dry_run": False,
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
    lifecycle_state_preview_report: dict | None,
    existing_state_snapshot: dict | None,
    operator_notes: dict | None,
    now_provider: Callable[[], Any] | None,
) -> dict:
    generated_at = _generated_at(now_provider)
    preview_report = (
        lifecycle_state_preview_report
        if isinstance(lifecycle_state_preview_report, dict)
        else None
    )
    snapshot = existing_state_snapshot if isinstance(existing_state_snapshot, dict) else None
    notes = operator_notes if isinstance(operator_notes, dict) else None
    preview = _mapping(_mapping(preview_report).get("preview"))
    order_record = _mapping(preview.get("proposed_order_record"))
    position_record = _optional_mapping(preview.get("proposed_position_record"))

    errors: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []
    if lifecycle_state_preview_report is not None and preview_report is None:
        errors.append("lifecycle_state_preview_report must be a dict")
        blockers.append("lifecycle_state_preview_report malformed")
    if existing_state_snapshot is not None and snapshot is None:
        warnings.append("existing_state_snapshot ignored because it is not a dict")
    if operator_notes is not None and notes is None:
        warnings.append("operator_notes ignored because it is not a dict")

    artifact_checks = _artifact_checks(preview_report)
    if not artifact_checks["lifecycle_state_preview_present"]:
        blockers.append("lifecycle_state_preview_report missing")
    if not artifact_checks["lifecycle_state_preview_ready"]:
        blockers.append("Stage 4G-2 preview is not ready for Stage 4G-3")
    if not artifact_checks["preview_available"]:
        blockers.append("Stage 4G-2 preview is not available")
    if not artifact_checks["proposed_order_record_present"]:
        blockers.append("proposed_order_record is required")

    input_summary = _input_summary(preview, order_record)
    if not _present(input_summary["broker_order_id"]):
        blockers.append("broker_order_id is required")
    if not _present(input_summary["client_order_id"]):
        blockers.append("client_order_id is required")

    proposed_state = input_summary["proposed_lifecycle_state"]
    if proposed_state in UNSUPPORTED_LIFECYCLE_STATES:
        blockers.append(f"unsupported lifecycle state for 4G-3 proposal: {proposed_state}")
    if proposed_state not in SUPPORTED_LIFECYCLE_STATES:
        blockers.append(f"proposed lifecycle state is not eligible for 4G-4: {proposed_state}")
    if proposed_state in {
        "paper_order_submitted_unverified",
        "paper_cancel_requested_unverified",
    }:
        warnings.append(f"manual follow-up required before writing {proposed_state}")

    write_plan_blockers = _validate_stage4g2_write_plan(preview_report)
    blockers.extend(write_plan_blockers)
    safety_checks = _safety_checks([preview_report], blockers)
    state_checks, state_blockers = _state_snapshot_checks(snapshot)
    blockers.extend(state_blockers)
    conflict_checks = _conflict_checks(snapshot, order_record, position_record)
    blockers.extend(conflict_checks["conflict_reasons"])

    notes_payload = _operator_notes(notes)
    proposed_operations = _proposed_state_store_operations(order_record, position_record)
    operations_blockers = _validate_operations(proposed_operations)
    blockers.extend(operations_blockers)

    source_events = _as_list(preview.get("proposed_ledger_events"))
    if not source_events:
        blockers.append("proposed_ledger_events missing from Stage 4G-2 preview")
    proposed_events = _proposed_ledger_events(
        source_events=source_events,
        generated_at=generated_at,
        input_summary=input_summary,
    )
    if not proposed_events:
        blockers.append("proposed_ledger_events are required")

    transition = _proposed_lifecycle_transition(proposed_state)
    if transition is None:
        blockers.append("supported proposed_lifecycle_transition is required")

    proposal = {
        "available": bool(proposed_operations and proposed_events and transition is not None),
        "proposed_state_store_operations": proposed_operations,
        "proposed_ledger_events": proposed_events,
        "proposed_lifecycle_transition": transition,
        "proposed_validation_checks": _proposed_validation_checks(),
        "proposed_rollback_plan": _proposed_rollback_plan(),
        "operator_approval_requirements": list(OPERATOR_ACKNOWLEDGEMENTS),
    }
    output_write_plan = _write_plan()

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    ready = (
        not blocker_list
        and not error_list
        and proposal["available"] is True
        and bool(proposed_operations)
        and _operations_valid(proposed_operations)
        and bool(proposed_events)
        and transition is not None
        and _present(input_summary["broker_order_id"])
        and _present(input_summary["client_order_id"])
        and all(safety_checks.values())
        and all(value is False for value in output_write_plan.values())
        and not conflict_checks["duplicate_client_order_id"]
        and not conflict_checks["duplicate_broker_order_id"]
        and not conflict_checks["duplicate_position_key"]
        and state_checks.get("unresolved_needs_reconciliation_count") in (None, 0)
        and state_checks.get("active_halt") in (None, False)
        and proposed_state in SUPPORTED_LIFECYCLE_STATES
    )

    return _json_safe(
        {
            "dry_run": True,
            "stage4g3_state_write_proposal": True,
            "generated_at": generated_at,
            "artifact_checks": artifact_checks,
            "input_summary": input_summary,
            "operator_notes": notes_payload,
            "proposal": proposal,
            "write_plan": output_write_plan,
            "conflict_checks": conflict_checks,
            "safety_checks": safety_checks,
            "state_snapshot_checks": state_checks,
            "readiness_for_stage4g4": {
                "ready_to_build_manual_state_write_dry_run": ready,
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
    readiness = _mapping(_mapping(report).get("readiness_for_stage4g3"))
    preview = _mapping(_mapping(report).get("preview"))
    return {
        "lifecycle_state_preview_present": report is not None
        and _mapping(report).get("stage4g2_lifecycle_state_preview") is True,
        "lifecycle_state_preview_ready": readiness.get(
            "ready_to_build_manual_state_write_proposal"
        )
        is True,
        "preview_available": preview.get("available") is True,
        "proposed_order_record_present": isinstance(
            preview.get("proposed_order_record"), dict
        ),
    }


def _input_summary(
    preview: dict[str, Any],
    order_record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "broker_order_id": order_record.get("broker_order_id"),
        "client_order_id": order_record.get("client_order_id"),
        "proposed_lifecycle_state": preview.get("proposed_lifecycle_state"),
        "proposed_order_status": order_record.get("status"),
    }


def _empty_input_summary() -> dict[str, Any]:
    return {
        "broker_order_id": None,
        "client_order_id": None,
        "proposed_lifecycle_state": None,
        "proposed_order_status": None,
    }


def _proposed_state_store_operations(
    order_record: dict[str, Any],
    position_record: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    if order_record:
        operations.append(
            {
                "operation": "upsert_order",
                "payload": _json_safe(
                    {
                        "client_order_id": order_record.get("client_order_id"),
                        "broker_order_id": order_record.get("broker_order_id"),
                        "strategy_id": order_record.get("strategy_id"),
                        "symbol": order_record.get("symbol"),
                        "action": order_record.get("action"),
                        "quantity": order_record.get("quantity"),
                        "order_type": order_record.get("order_type"),
                        "status": order_record.get("status"),
                        "paper_only": True,
                        "source_stage": "4G-3",
                        "derived_from_stage4g2": True,
                    }
                ),
            }
        )
    if position_record is not None:
        operations.append(
            {
                "operation": "upsert_position",
                "payload": _json_safe(
                    {
                        "broker_order_id": position_record.get("broker_order_id"),
                        "client_order_id": position_record.get("client_order_id"),
                        "symbol": position_record.get("symbol"),
                        "quantity": position_record.get("quantity"),
                        "avg_fill_price": position_record.get("avg_fill_price"),
                        "paper_only": True,
                        "source_stage": "4G-3",
                        "derived_from_stage4g2": True,
                    }
                ),
            }
        )
    return operations


def _proposed_ledger_events(
    *,
    source_events: list[Any],
    generated_at: str,
    input_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    if not source_events:
        return [
            {
                "event_type": "PAPER_LIFECYCLE_STATE_PROPOSED",
                "proposal_only": True,
                "timestamp": generated_at,
                "generated_at": generated_at,
                "client_order_id": input_summary.get("client_order_id"),
                "broker_order_id": input_summary.get("broker_order_id"),
                "proposed_lifecycle_state": input_summary.get("proposed_lifecycle_state"),
            }
        ]

    proposed: list[dict[str, Any]] = []
    for event in source_events:
        item = _mapping(event)
        event_type = item.get("event_type") or item.get("type")
        proposed.append(
            {
                "event_type": event_type,
                "proposal_only": True,
                "timestamp": item.get("timestamp"),
                "generated_at": generated_at,
                "client_order_id": item.get(
                    "client_order_id", input_summary.get("client_order_id")
                ),
                "broker_order_id": item.get(
                    "broker_order_id", input_summary.get("broker_order_id")
                ),
                "proposed_lifecycle_state": item.get(
                    "proposed_lifecycle_state",
                    input_summary.get("proposed_lifecycle_state"),
                ),
                "payload": _json_safe(item),
            }
        )
    return proposed


def _proposed_lifecycle_transition(state: Any) -> dict[str, Any] | None:
    if state not in SUPPORTED_LIFECYCLE_STATES:
        return None
    return {
        "transition_to": state,
        "proposal_only": True,
        "enabled": False,
        "source_stage": "4G-3",
    }


def _proposed_validation_checks() -> list[dict[str, Any]]:
    return [
        {"check": "confirm_state_store_payloads", "required_before_future_write": True},
        {"check": "confirm_ledger_event_payloads", "required_before_future_write": True},
        {"check": "confirm_paper_only_scope", "required_before_future_write": True},
        {"check": "confirm_scheduler_automation_disabled", "required_before_future_write": True},
    ]


def _proposed_rollback_plan() -> dict[str, Any]:
    return {
        "proposal_only": True,
        "future_write_phase_required": True,
        "steps": [
            "Review generated StateStore payloads before any future write.",
            "Review generated ledger event payloads before any future append.",
            "If a future write is approved and later found incorrect, use a separately approved manual reconciliation plan.",
        ],
    }


def _empty_proposal() -> dict[str, Any]:
    return {
        "available": False,
        "proposed_state_store_operations": [],
        "proposed_ledger_events": [],
        "proposed_lifecycle_transition": None,
        "proposed_validation_checks": _proposed_validation_checks(),
        "proposed_rollback_plan": _proposed_rollback_plan(),
        "operator_approval_requirements": list(OPERATOR_ACKNOWLEDGEMENTS),
    }


def _validate_operations(operations: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    if not operations:
        blockers.append("proposed_state_store_operations are required")
    for index, operation in enumerate(operations):
        name = operation.get("operation")
        payload = operation.get("payload")
        if name not in RECOGNIZED_OPERATIONS:
            blockers.append(f"unrecognized proposed StateStore operation at index {index}")
        if not isinstance(payload, dict):
            blockers.append(f"proposed StateStore operation payload must be a dict at index {index}")
    return blockers


def _operations_valid(operations: list[dict[str, Any]]) -> bool:
    return not _validate_operations(operations)


def _validate_stage4g2_write_plan(report: dict[str, Any] | None) -> list[str]:
    write_plan = _mapping(_mapping(report).get("write_plan"))
    blockers: list[str] = []
    for key in (
        "state_store_write_enabled",
        "ledger_write_enabled",
        "lifecycle_transition_enabled",
        "daemon_wiring_enabled",
        "scheduler_wiring_enabled",
    ):
        if write_plan.get(key) is not False:
            blockers.append(f"Stage 4G-2 write_plan flag must be False: {key}")
    return blockers


def _write_plan() -> dict[str, bool]:
    return {
        "state_store_write_enabled": False,
        "ledger_write_enabled": False,
        "lifecycle_transition_enabled": False,
        "daemon_wiring_enabled": False,
        "scheduler_wiring_enabled": False,
    }


def _conflict_checks(
    snapshot: dict[str, Any] | None,
    order_record: dict[str, Any],
    position_record: dict[str, Any] | None,
) -> dict[str, Any]:
    broker_order_id = order_record.get("broker_order_id")
    client_order_id = order_record.get("client_order_id")
    position_key = _position_key(position_record)
    reasons: list[str] = []
    duplicate_client = False
    duplicate_broker = False
    duplicate_position = False
    if snapshot is not None:
        duplicate_client = _value_in_snapshot(
            snapshot,
            ("existing_client_order_ids", "open_order_client_ids"),
            client_order_id,
        )
        duplicate_broker = _value_in_snapshot(
            snapshot,
            ("existing_broker_order_ids", "open_order_broker_ids"),
            broker_order_id,
        )
        duplicate_position = _value_in_snapshot(
            snapshot,
            ("existing_position_keys",),
            position_key,
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


def _operator_notes(notes: dict[str, Any] | None) -> dict[str, Any]:
    if notes is None:
        return {}
    return _json_safe(
        {
            "manual_observation": notes.get("manual_observation"),
            "cleanup_ticket": notes.get("cleanup_ticket"),
            "operator_initials": notes.get("operator_initials"),
            "follow_up_required": notes.get("follow_up_required") is True,
        }
    )


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


def _position_key(position_record: dict[str, Any] | None) -> str | None:
    position = _mapping(position_record)
    explicit = position.get("position_key")
    if _present(explicit):
        return str(explicit)
    symbol = position.get("symbol")
    strategy_id = position.get("strategy_id")
    client_order_id = position.get("client_order_id")
    if not (_present(symbol) and _present(strategy_id) and _present(client_order_id)):
        return None
    return f"{strategy_id}:{symbol}:{client_order_id}"


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


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return value if isinstance(value, dict) else None


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
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


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
