"""Pure Stage 4G-2 manual paper lifecycle state preview reporting."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
ORDERED_NEXT_STEPS = [
    "Build Stage 4G-3 manual state write proposal behind explicit operator gates.",
    "Keep scheduler/lifecycle automation disabled.",
    "Do not write StateStore until 4G-3 proposal is reviewed.",
    "Do not write ledger events until explicitly approved.",
    "Do not begin live trading.",
]
DO_NOT_DO_YET = [
    "Do not write persistent state in Stage 4G-2.",
    "Do not write ledger events in Stage 4G-2.",
    "Do not execute lifecycle transitions.",
    "Do not enable scheduler or daemon lifecycle automation.",
    "Do not begin live trading.",
]
SAFETY_FLAG_KEYS = {
    "live_orders_enabled": "no_live_orders",
    "market_data_enabled": "no_market_data",
    "contract_qualification_enabled": "no_contract_qualification",
    "scheduler_changes_enabled": "no_scheduler_changes",
    "daemon_wiring_enabled": "no_scheduler_changes",
    "scheduler_wiring_enabled": "no_scheduler_changes",
    "lifecycle_wiring_enabled": "no_lifecycle_wiring",
    "state_mutation_enabled": "no_state_mutation",
    "ledger_writes_enabled": "no_ledger_writes",
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
STATE_MAP = {
    "broker_submitted": ("paper_order_submitted", "submitted"),
    "broker_filled": ("paper_order_filled", "filled"),
    "broker_partially_filled": (
        "paper_order_partially_filled_review",
        "partially_filled",
    ),
    "broker_cancelled": ("paper_order_cancelled", "cancelled"),
    "broker_rejected_or_inactive": (
        "paper_order_rejected_or_inactive_review",
        "rejected_or_inactive",
    ),
    "submitted_unverified": ("paper_order_submitted_unverified", "submitted_unverified"),
    "cancel_requested_unverified": (
        "paper_cancel_requested_unverified",
        "cancel_requested_unverified",
    ),
    "unknown_broker_status": ("paper_order_unknown_status_review", "unknown"),
    "needs_reconciliation": ("needs_reconciliation", "needs_reconciliation"),
    "unsafe_artifact": ("unsafe_artifact", "unsafe_artifact"),
}
READY_STATES = {
    "paper_order_submitted",
    "paper_order_filled",
    "paper_order_cancelled",
    "paper_order_submitted_unverified",
    "paper_cancel_requested_unverified",
}
REVIEW_STATES = {
    "paper_order_partially_filled_review",
    "paper_order_rejected_or_inactive_review",
    "paper_order_unknown_status_review",
    "needs_reconciliation",
    "unsafe_artifact",
}


def build_stage4g2_lifecycle_state_preview(
    *,
    lifecycle_intake_report: dict | None,
    existing_state_snapshot: dict | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a read-only preview of lifecycle/state records from a 4G-1 report."""

    try:
        return _build_report(
            lifecycle_intake_report=lifecycle_intake_report,
            existing_state_snapshot=existing_state_snapshot,
            now_provider=now_provider,
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must remain data-only.
        generated_at = _generated_at(now_provider)
        return _json_safe(
            {
                "dry_run": True,
                "stage4g2_lifecycle_state_preview": True,
                "generated_at": generated_at,
                "artifact_checks": _artifact_checks(None),
                "input_summary": _empty_input_summary(),
                "preview": _empty_preview(),
                "write_plan": _write_plan(),
                "conflict_checks": _empty_conflict_checks(),
                "safety_checks": _default_safety_checks(),
                "state_snapshot_checks": _empty_state_snapshot_checks(),
                "readiness_for_stage4g3": {
                    "ready_to_build_manual_state_write_proposal": False,
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
    lifecycle_intake_report: dict | None,
    existing_state_snapshot: dict | None,
    now_provider: Callable[[], Any] | None,
) -> dict:
    generated_at = _generated_at(now_provider)
    intake = lifecycle_intake_report if isinstance(lifecycle_intake_report, dict) else None
    snapshot = existing_state_snapshot if isinstance(existing_state_snapshot, dict) else None
    candidate = _mapping(_mapping(intake).get("lifecycle_intake_candidate"))
    readiness_4g2 = _mapping(_mapping(intake).get("readiness_for_stage4g2"))

    errors: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []

    if lifecycle_intake_report is not None and intake is None:
        errors.append("lifecycle_intake_report must be a dict")
        blockers.append("lifecycle_intake_report malformed")
    if existing_state_snapshot is not None and snapshot is None:
        warnings.append("existing_state_snapshot ignored because it is not a dict")

    artifact_checks = _artifact_checks(intake)
    if not artifact_checks["lifecycle_intake_present"]:
        blockers.append("lifecycle_intake_report missing")
    if not artifact_checks["lifecycle_intake_ready"]:
        blockers.append("Stage 4G-1 intake is not ready for Stage 4G-2")
    if not artifact_checks["lifecycle_candidate_available"]:
        blockers.append("lifecycle_intake_candidate is not available")

    input_summary = _input_summary(candidate)
    if not _present(input_summary["broker_order_id"]):
        blockers.append("broker_order_id is required")
    if not _present(input_summary["client_order_id"]):
        blockers.append("client_order_id is required")
    if not _present(input_summary["suggested_internal_lifecycle_state"]):
        blockers.append("suggested_internal_lifecycle_state is required")

    safety_checks = _safety_checks([intake], blockers)
    state_checks, state_blockers = _state_snapshot_checks(snapshot)
    blockers.extend(state_blockers)
    conflict_checks = _conflict_checks(snapshot, candidate)
    blockers.extend(conflict_checks["conflict_reasons"])

    preview, preview_blockers, preview_warnings = _preview(candidate, generated_at)
    blockers.extend(preview_blockers)
    warnings.extend(preview_warnings)

    proposed_state = preview.get("proposed_lifecycle_state")
    if proposed_state in REVIEW_STATES:
        blockers.append(f"proposed lifecycle state requires manual review: {proposed_state}")
    if proposed_state not in READY_STATES:
        blockers.append(f"proposed lifecycle state is not eligible for 4G-3: {proposed_state}")

    if proposed_state in {
        "paper_order_submitted_unverified",
        "paper_cancel_requested_unverified",
    }:
        warnings.append(f"manual follow-up required before relying on {proposed_state}")

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    ready = (
        not blocker_list
        and not error_list
        and preview.get("available") is True
        and bool(input_summary["broker_order_id"])
        and bool(input_summary["client_order_id"])
        and all(safety_checks.values())
        and not conflict_checks["duplicate_client_order_id"]
        and not conflict_checks["duplicate_broker_order_id"]
        and not conflict_checks["duplicate_position_key"]
        and proposed_state in READY_STATES
    )

    return _json_safe(
        {
            "dry_run": True,
            "stage4g2_lifecycle_state_preview": True,
            "generated_at": generated_at,
            "artifact_checks": artifact_checks,
            "input_summary": input_summary,
            "preview": preview,
            "write_plan": _write_plan(),
            "conflict_checks": conflict_checks,
            "safety_checks": safety_checks,
            "state_snapshot_checks": state_checks,
            "readiness_for_stage4g3": {
                "ready_to_build_manual_state_write_proposal": ready,
                "blockers": blocker_list,
                "warnings": warning_list,
            },
            "recommendations": _recommendations(),
            "success": True,
            "errors": error_list,
            "warnings": warning_list,
        }
    )


def _artifact_checks(intake: dict[str, Any] | None) -> dict[str, bool]:
    readiness = _mapping(_mapping(intake).get("readiness_for_stage4g2"))
    candidate = _mapping(_mapping(intake).get("lifecycle_intake_candidate"))
    return {
        "lifecycle_intake_present": intake is not None
        and _mapping(intake).get("stage4g1_lifecycle_intake_report") is True,
        "lifecycle_intake_ready": readiness.get(
            "ready_to_build_manual_lifecycle_state_preview"
        )
        is True,
        "lifecycle_candidate_available": candidate.get("available") is True,
    }


def _input_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "broker_order_id": candidate.get("broker_order_id"),
        "client_order_id": candidate.get("client_order_id"),
        "suggested_internal_lifecycle_state": candidate.get(
            "suggested_internal_lifecycle_state"
        ),
        "reconciliation_required_from_intake": candidate.get("reconciliation_required")
        is True,
    }


def _empty_input_summary() -> dict[str, Any]:
    return {
        "broker_order_id": None,
        "client_order_id": None,
        "suggested_internal_lifecycle_state": None,
        "reconciliation_required_from_intake": False,
    }


def _preview(candidate: dict[str, Any], generated_at: str) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    state = candidate.get("suggested_internal_lifecycle_state")
    proposed_state, proposed_order_status = STATE_MAP.get(
        state,
        ("paper_order_unknown_status_review", "unknown"),
    )
    if state not in STATE_MAP:
        blockers.append(f"unrecognized suggested_internal_lifecycle_state: {state}")

    order_record = _order_record(candidate, proposed_order_status, proposed_state)
    position_record, position_blockers = _position_record(candidate, proposed_state)
    blockers.extend(position_blockers)

    reconciliation_required = _reconciliation_required(candidate, proposed_state)
    reconciliation_reasons = list(_as_list(candidate.get("reconciliation_reasons")))
    if reconciliation_required and not reconciliation_reasons:
        reconciliation_reasons.append(f"{proposed_state} requires manual review")

    operator_actions = _operator_actions(
        proposed_state=proposed_state,
        reconciliation_required=reconciliation_required,
    )
    ledger_events = _ledger_events(
        generated_at=generated_at,
        candidate=candidate,
        proposed_state=proposed_state,
        proposed_order_status=proposed_order_status,
        position_record=position_record,
    )
    return (
        {
            "available": bool(candidate.get("available") is True and order_record),
            "proposed_lifecycle_state": proposed_state,
            "proposed_order_record": order_record,
            "proposed_position_record": position_record,
            "proposed_reconciliation_flags": {
                "reconciliation_required": reconciliation_required,
                "reconciliation_reasons": _dedupe(reconciliation_reasons),
            },
            "proposed_operator_actions": operator_actions,
            "proposed_ledger_events": ledger_events,
        },
        blockers,
        warnings,
    )


def _order_record(
    candidate: dict[str, Any],
    status: str,
    lifecycle_state: str,
) -> dict[str, Any]:
    return {
        "record_type": "paper_order",
        "preview_only": True,
        "broker_order_id": candidate.get("broker_order_id"),
        "client_order_id": candidate.get("client_order_id"),
        "strategy_id": candidate.get("strategy_id"),
        "symbol": candidate.get("symbol"),
        "action": candidate.get("action"),
        "quantity": _safe_float(candidate.get("quantity")),
        "filled_quantity": _safe_float(candidate.get("filled_quantity")),
        "remaining_quantity": _safe_float(candidate.get("remaining_quantity")),
        "avg_fill_price": _safe_float(candidate.get("avg_fill_price")),
        "order_type": candidate.get("order_type"),
        "status": status,
        "lifecycle_state": lifecycle_state,
    }


def _position_record(
    candidate: dict[str, Any],
    proposed_state: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    blockers: list[str] = []
    filled_quantity = _safe_float(candidate.get("filled_quantity"))
    order_quantity = _safe_float(candidate.get("quantity"))
    position_quantity: float | None = None

    if proposed_state == "paper_order_filled":
        if filled_quantity is not None and filled_quantity > 0:
            position_quantity = filled_quantity
        elif order_quantity is not None and order_quantity > 0:
            position_quantity = order_quantity
        else:
            blockers.append("filled candidate lacks usable position quantity")
    elif proposed_state in {
        "paper_order_partially_filled_review",
        "paper_cancel_requested_unverified",
        "paper_order_unknown_status_review",
    }:
        if filled_quantity is not None and filled_quantity > 0:
            position_quantity = filled_quantity

    if position_quantity is None:
        return None, blockers

    return (
        {
            "record_type": "paper_position",
            "preview_only": True,
            "position_key": _position_key(candidate),
            "broker_order_id": candidate.get("broker_order_id"),
            "client_order_id": candidate.get("client_order_id"),
            "strategy_id": candidate.get("strategy_id"),
            "symbol": candidate.get("symbol"),
            "quantity": position_quantity,
            "avg_fill_price": _safe_float(candidate.get("avg_fill_price")),
            "source_lifecycle_state": proposed_state,
        },
        blockers,
    )


def _reconciliation_required(candidate: dict[str, Any], proposed_state: str) -> bool:
    return candidate.get("reconciliation_required") is True or proposed_state in {
        "paper_order_partially_filled_review",
        "paper_order_rejected_or_inactive_review",
        "paper_order_unknown_status_review",
        "paper_order_submitted_unverified",
        "paper_cancel_requested_unverified",
        "needs_reconciliation",
        "unsafe_artifact",
    }


def _operator_actions(*, proposed_state: str, reconciliation_required: bool) -> list[str]:
    actions = ["review preview-only proposed records before Stage 4G-3"]
    if reconciliation_required:
        actions.append("perform manual broker/state reconciliation before approval")
    if proposed_state in {
        "paper_order_submitted_unverified",
        "paper_cancel_requested_unverified",
    }:
        actions.append("capture manual follow-up evidence before any future write proposal")
    if proposed_state in REVIEW_STATES:
        actions.append("do not advance without explicit operator reconciliation")
    return actions


def _ledger_events(
    *,
    generated_at: str,
    candidate: dict[str, Any],
    proposed_state: str,
    proposed_order_status: str,
    position_record: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    events = [
        {
            "event_type": "paper_order_lifecycle_state_preview",
            "preview_only": True,
            "timestamp": generated_at,
            "client_order_id": candidate.get("client_order_id"),
            "broker_order_id": candidate.get("broker_order_id"),
            "proposed_lifecycle_state": proposed_state,
            "proposed_order_status": proposed_order_status,
        }
    ]
    if position_record is not None:
        events.append(
            {
                "event_type": "paper_position_state_preview",
                "preview_only": True,
                "timestamp": generated_at,
                "client_order_id": candidate.get("client_order_id"),
                "broker_order_id": candidate.get("broker_order_id"),
                "position_key": position_record.get("position_key"),
                "quantity": position_record.get("quantity"),
            }
        )
    return events


def _empty_preview() -> dict[str, Any]:
    return {
        "available": False,
        "proposed_lifecycle_state": None,
        "proposed_order_record": None,
        "proposed_position_record": None,
        "proposed_reconciliation_flags": {
            "reconciliation_required": True,
            "reconciliation_reasons": [],
        },
        "proposed_operator_actions": [],
        "proposed_ledger_events": [],
    }


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
    candidate: dict[str, Any],
) -> dict[str, Any]:
    broker_order_id = candidate.get("broker_order_id")
    client_order_id = candidate.get("client_order_id")
    position_key = _position_key(candidate)
    reasons: list[str] = []
    same_target = _same_preview_target(snapshot, broker_order_id, client_order_id)
    duplicate_client = False
    duplicate_broker = False
    duplicate_position = False
    if snapshot is not None and not same_target:
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


def _same_preview_target(
    snapshot: dict[str, Any] | None,
    broker_order_id: Any,
    client_order_id: Any,
) -> bool:
    target = _mapping(_mapping(snapshot).get("same_preview_target"))
    return (
        _present(broker_order_id)
        and _present(client_order_id)
        and str(target.get("broker_order_id")) == str(broker_order_id)
        and str(target.get("client_order_id")) == str(client_order_id)
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


def _position_key(candidate: dict[str, Any]) -> str | None:
    explicit = candidate.get("position_key")
    if _present(explicit):
        return str(explicit)
    symbol = candidate.get("symbol")
    strategy_id = candidate.get("strategy_id")
    client_order_id = candidate.get("client_order_id")
    if not (_present(symbol) and _present(strategy_id) and _present(client_order_id)):
        return None
    return f"{strategy_id}:{symbol}:{client_order_id}"


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            decimal_value = Decimal(text)
        except (InvalidOperation, ValueError):
            return None
        as_float = float(decimal_value)
        return as_float if math.isfinite(as_float) else None
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
