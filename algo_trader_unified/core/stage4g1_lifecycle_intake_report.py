"""Pure Stage 4G-1 manual paper lifecycle intake reporting."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
ORDERED_NEXT_STEPS = [
    "Review the Stage 4G-1 lifecycle intake candidate with the operator.",
    "Build Stage 4G-2 as a manual lifecycle state preview only.",
    "Keep paper lifecycle automation disabled until a later approved stage.",
]
DO_NOT_DO_YET = [
    "Do not persist lifecycle intake records yet.",
    "Do not enable scheduler or daemon paper lifecycle automation.",
    "Do not begin live trading.",
    "Do not add any new broker action path.",
]
BENIGN_RECONCILIATION_STATES = {
    "submitted_unverified",
    "cancel_requested_unverified",
}
BLOCKING_RECONCILIATION_STATES = {
    "broker_partially_filled",
    "broker_rejected_or_inactive",
    "unknown_broker_status",
    "needs_reconciliation",
    "unsafe_artifact",
}
SAFETY_FLAG_KEYS = {
    "live_orders_enabled": "no_live_orders",
    "market_data_enabled": "no_market_data",
    "contract_qualification_enabled": "no_contract_qualification",
    "scheduler_changes_enabled": "no_scheduler_changes",
    "daemon_wiring_enabled": "no_scheduler_changes",
    "lifecycle_wiring_enabled": "no_lifecycle_wiring",
    "state_mutation_enabled": "no_state_mutation",
    "ledger_writes_enabled": "no_ledger_writes",
}
STATUS_SUBMITTED = {"submitted", "presubmitted", "pendingsubmit"}
STATUS_FILLED = {"filled"}
STATUS_PARTIAL = {"partiallyfilled", "partialfilled", "partially filled"}
STATUS_CANCELLED = {"cancelled", "canceled", "apicancelled", "api cancelled"}
STATUS_REJECTED = {"inactive", "rejected"}


def build_stage4g1_lifecycle_intake_report(
    *,
    stage4f_acceptance_report: dict | None,
    smoke_test_report: dict | None,
    submit_report: dict | None,
    order_control_reports: list[dict] | None = None,
    existing_state_snapshot: dict | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a read-only lifecycle intake candidate report from supplied artifacts."""

    try:
        return _build_report(
            stage4f_acceptance_report=stage4f_acceptance_report,
            smoke_test_report=smoke_test_report,
            submit_report=submit_report,
            order_control_reports=order_control_reports,
            existing_state_snapshot=existing_state_snapshot,
            now_provider=now_provider,
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must remain data-only.
        return _json_safe(
            {
                "dry_run": True,
                "stage4g1_lifecycle_intake_report": True,
                "generated_at": _generated_at(now_provider),
                "artifact_checks": _artifact_checks(None, None, None, []),
                "consistency_checks": _empty_consistency_checks(),
                "lifecycle_intake_candidate": _empty_candidate(
                    "unknown_broker_status",
                    ["unexpected report failure"],
                ),
                "safety_checks": _default_safety_checks(),
                "state_snapshot_checks": _empty_state_snapshot_checks(),
                "readiness_for_stage4g2": {
                    "ready_to_build_manual_lifecycle_state_preview": False,
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
    stage4f_acceptance_report: dict | None,
    smoke_test_report: dict | None,
    submit_report: dict | None,
    order_control_reports: list[dict] | None,
    existing_state_snapshot: dict | None,
    now_provider: Callable[[], Any] | None,
) -> dict:
    generated_at = _generated_at(now_provider)
    stage4f = stage4f_acceptance_report if isinstance(stage4f_acceptance_report, dict) else None
    smoke_report = smoke_test_report if isinstance(smoke_test_report, dict) else None
    submit = submit_report if isinstance(submit_report, dict) else None
    controls = [item for item in (order_control_reports or []) if isinstance(item, dict)]
    snapshot = existing_state_snapshot if isinstance(existing_state_snapshot, dict) else None

    errors: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []

    if stage4f_acceptance_report is not None and stage4f is None:
        errors.append("stage4f_acceptance_report must be a dict")
        blockers.append("stage4f_acceptance_report malformed")
    if smoke_test_report is not None and smoke_report is None:
        errors.append("smoke_test_report must be a dict")
        blockers.append("smoke_test_report malformed")
    if submit_report is not None and submit is None:
        errors.append("submit_report must be a dict")
        blockers.append("submit_report malformed")
    if existing_state_snapshot is not None and snapshot is None:
        warnings.append("existing_state_snapshot ignored because it is not a dict")

    stage4f_ready = _stage4f_ready(stage4f)
    smoke_summary = _smoke_summary(smoke_report)
    submit_summary = _submit_summary(submit)
    order_summary = _order_control_summary(controls)
    ids = _id_summary(smoke_summary, submit_summary, controls)

    consistency_checks = {
        "stage4f_ready": stage4f_ready,
        "smoke_test_accepted": smoke_summary["accepted"],
        "submit_succeeded": submit_summary["submitted"],
        "broker_order_id_present": bool(ids["broker_order_id"]),
        "client_order_id_present": bool(ids["client_order_id"]),
        "broker_order_id_consistent": ids["broker_order_id_consistent"],
        "client_order_id_consistent": ids["client_order_id_consistent"],
        "status_report_present": order_summary["status_report_present"],
    }
    _add_consistency_blockers(
        stage4f=stage4f,
        smoke_report=smoke_report,
        submit=submit,
        checks=consistency_checks,
        blockers=blockers,
        warnings=warnings,
    )

    if not order_summary["status_report_present"] and _operator_left_open(smoke_report):
        warnings.append("operator notes document intentional open follow-up")
    elif not order_summary["status_report_present"]:
        blockers.append("at least one status report is required")

    safety_checks = _safety_checks([stage4f, smoke_report, submit, *controls], blockers)
    state_checks, state_blockers = _state_snapshot_checks(snapshot)
    blockers.extend(state_blockers)

    candidate = _candidate(
        ids=ids,
        submit_summary=submit_summary,
        smoke_summary=smoke_summary,
        order_summary=order_summary,
        consistency_checks=consistency_checks,
        safety_checks=safety_checks,
    )

    candidate_state = candidate["suggested_internal_lifecycle_state"]
    if candidate_state in BLOCKING_RECONCILIATION_STATES:
        blockers.append(f"candidate state requires manual review: {candidate_state}")
    elif (
        candidate["reconciliation_required"] is True
        and candidate_state not in BENIGN_RECONCILIATION_STATES
    ):
        blockers.append("candidate reconciliation is not benign manual follow-up")
    if candidate["reconciliation_required"] is True and candidate_state in BENIGN_RECONCILIATION_STATES:
        warnings.append(f"candidate requires manual follow-up: {candidate_state}")

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    ready = (
        not blocker_list
        and not error_list
        and stage4f_ready
        and smoke_summary["accepted"]
        and submit_summary["submitted"]
        and bool(ids["broker_order_id"])
        and bool(ids["client_order_id"])
        and ids["broker_order_id_consistent"]
        and ids["client_order_id_consistent"]
        and all(safety_checks.values())
        and candidate["available"] is True
        and candidate_state != "unsafe_artifact"
    )

    report = {
        "dry_run": True,
        "stage4g1_lifecycle_intake_report": True,
        "generated_at": generated_at,
        "artifact_checks": _artifact_checks(stage4f, smoke_report, submit, controls),
        "consistency_checks": consistency_checks,
        "lifecycle_intake_candidate": candidate,
        "safety_checks": safety_checks,
        "state_snapshot_checks": state_checks,
        "readiness_for_stage4g2": {
            "ready_to_build_manual_lifecycle_state_preview": ready,
            "blockers": blocker_list,
            "warnings": warning_list,
        },
        "recommendations": _recommendations(),
        "success": True,
        "errors": error_list,
        "warnings": warning_list,
    }
    return _json_safe(report)


def _artifact_checks(
    stage4f: dict[str, Any] | None,
    smoke: dict[str, Any] | None,
    submit: dict[str, Any] | None,
    controls: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "stage4f_acceptance_present": stage4f is not None,
        "smoke_test_report_present": smoke is not None,
        "submit_report_present": submit is not None,
        "order_control_report_count": len(controls),
    }


def _empty_consistency_checks() -> dict[str, bool]:
    return {
        "stage4f_ready": False,
        "smoke_test_accepted": False,
        "submit_succeeded": False,
        "broker_order_id_present": False,
        "client_order_id_present": False,
        "broker_order_id_consistent": False,
        "client_order_id_consistent": False,
        "status_report_present": False,
    }


def _add_consistency_blockers(
    *,
    stage4f: dict[str, Any] | None,
    smoke_report: dict[str, Any] | None,
    submit: dict[str, Any] | None,
    checks: dict[str, bool],
    blockers: list[str],
    warnings: list[str],
) -> None:
    if stage4f is None:
        blockers.append("stage4f_acceptance_report missing")
    elif not checks["stage4f_ready"]:
        blockers.append("Stage 4F acceptance is not ready for Stage 4G")
    if smoke_report is None:
        blockers.append("smoke_test_report missing")
    elif not checks["smoke_test_accepted"]:
        blockers.append("smoke_test.accepted must be True")
    if submit is None:
        blockers.append("submit_report missing")
    elif not checks["submit_succeeded"]:
        blockers.append("submission.submitted must be True")
    if not checks["broker_order_id_present"]:
        blockers.append("broker_order_id is required")
    if not checks["client_order_id_present"]:
        blockers.append("client_order_id is required")
    if not checks["broker_order_id_consistent"]:
        blockers.append("broker_order_id mismatch across artifacts")
    if not checks["client_order_id_consistent"]:
        blockers.append("client_order_id mismatch across artifacts")
    if not checks["status_report_present"]:
        warnings.append("status report was not supplied")


def _stage4f_ready(report: dict[str, Any] | None) -> bool:
    readiness = _mapping(_mapping(report).get("readiness_for_stage4g"))
    return readiness.get("ready_to_begin_manual_paper_lifecycle_validation") is True


def _smoke_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    smoke = _mapping(_mapping(report).get("smoke_test"))
    order_control = _mapping(_mapping(report).get("order_control_summary"))
    return {
        "accepted": smoke.get("accepted") is True,
        "broker_order_id": smoke.get("broker_order_id"),
        "client_order_id": smoke.get("client_order_id"),
        "strategy_id": _first_non_empty(smoke.get("strategy_id"), report and report.get("strategy_id")),
        "symbol": _first_non_empty(smoke.get("symbol"), report and report.get("symbol")),
        "action": _first_non_empty(smoke.get("action"), smoke.get("side"), report and report.get("action")),
        "quantity": _first_non_empty(smoke.get("quantity"), report and report.get("quantity")),
        "order_type": _first_non_empty(smoke.get("order_type"), report and report.get("order_type")),
        "submitted": smoke.get("submitted") is True,
        "terminal_or_safe_state_seen": smoke.get("terminal_or_safe_state_seen") is True,
        "last_known_status": order_control.get("last_known_status"),
        "cancel_attempted": order_control.get("cancel_attempted") is True,
        "cancel_succeeded": _bool_or_none(order_control.get("cancel_succeeded")),
    }


def _submit_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    root = _mapping(report)
    submission = _mapping(root.get("submission"))
    plan = _mapping(root.get("ibkr_order_plan"))
    ticket = _mapping(root.get("ticket"))
    return {
        "submitted": submission.get("submitted") is True,
        "broker_order_id": _first_non_empty(submission.get("broker_order_id"), root.get("broker_order_id")),
        "client_order_id": _first_non_empty(
            submission.get("client_order_id"),
            plan.get("client_order_id"),
            ticket.get("client_order_id"),
            root.get("client_order_id"),
        ),
        "strategy_id": _first_non_empty(
            submission.get("strategy_id"),
            plan.get("strategy_id"),
            root.get("strategy_id"),
        ),
        "symbol": _first_non_empty(submission.get("symbol"), plan.get("symbol"), root.get("symbol")),
        "action": _first_non_empty(
            submission.get("action"),
            submission.get("side"),
            plan.get("action"),
            plan.get("side"),
            root.get("action"),
            root.get("side"),
        ),
        "quantity": _first_non_empty(submission.get("quantity"), plan.get("quantity"), root.get("quantity")),
        "order_type": _first_non_empty(
            submission.get("order_type"),
            plan.get("order_type"),
            root.get("order_type"),
        ),
    }


def _order_control_summary(controls: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: list[str] = []
    status_report_present = False
    last_known_status: str | None = None
    filled_quantity: Any = None
    remaining_quantity: Any = None
    avg_fill_price: Any = None
    cancel_attempted = False
    cancel_succeeded: bool | None = None

    for report in controls:
        action = report.get("action")
        status = _status_value(report)
        if action == "status" or status is not None:
            status_report_present = True
        if status is not None:
            statuses.append(status)
            last_known_status = status
            status_payload = _mapping(report.get("status"))
            order_payload = _mapping(report.get("order"))
            filled_quantity = _first_non_empty(
                status_payload.get("filled_quantity"),
                status_payload.get("filled"),
                order_payload.get("filled_quantity"),
                order_payload.get("filled"),
                report.get("filled_quantity"),
            )
            remaining_quantity = _first_non_empty(
                status_payload.get("remaining_quantity"),
                status_payload.get("remaining"),
                order_payload.get("remaining_quantity"),
                order_payload.get("remaining"),
                report.get("remaining_quantity"),
            )
            avg_fill_price = _first_non_empty(
                status_payload.get("avg_fill_price"),
                status_payload.get("avgFillPrice"),
                order_payload.get("avg_fill_price"),
                order_payload.get("avgFillPrice"),
                report.get("avg_fill_price"),
            )
        if action == "cancel" or "cancel" in report:
            cancel_attempted = True
            cancel_succeeded = _cancel_succeeded(report)

    return {
        "statuses": statuses,
        "status_report_present": status_report_present,
        "last_known_status": last_known_status,
        "filled_quantity": filled_quantity,
        "remaining_quantity": remaining_quantity,
        "avg_fill_price": avg_fill_price,
        "cancel_attempted": cancel_attempted,
        "cancel_succeeded": cancel_succeeded,
    }


def _id_summary(
    smoke_summary: dict[str, Any],
    submit_summary: dict[str, Any],
    controls: list[dict[str, Any]],
) -> dict[str, Any]:
    broker_values = [
        value
        for value in (submit_summary["broker_order_id"], smoke_summary["broker_order_id"])
        if _present(value)
    ]
    client_values = [
        value
        for value in (submit_summary["client_order_id"], smoke_summary["client_order_id"])
        if _present(value)
    ]
    for report in controls:
        order = _mapping(report.get("order"))
        status = _mapping(report.get("status"))
        cancel = _mapping(report.get("cancel"))
        broker = _first_non_empty(
            order.get("broker_order_id"),
            status.get("broker_order_id"),
            cancel.get("broker_order_id"),
            report.get("broker_order_id"),
        )
        client = _first_non_empty(
            order.get("client_order_id"),
            status.get("client_order_id"),
            cancel.get("client_order_id"),
            report.get("client_order_id"),
        )
        if _present(broker):
            broker_values.append(broker)
        if _present(client):
            client_values.append(client)
    broker_id = _first_non_empty(submit_summary["broker_order_id"], smoke_summary["broker_order_id"])
    client_id = _first_non_empty(submit_summary["client_order_id"], smoke_summary["client_order_id"])
    return {
        "broker_order_id": broker_id,
        "client_order_id": client_id,
        "broker_order_id_consistent": _all_same_present(broker_values),
        "client_order_id_consistent": _all_same_present(client_values),
    }


def _candidate(
    *,
    ids: dict[str, Any],
    submit_summary: dict[str, Any],
    smoke_summary: dict[str, Any],
    order_summary: dict[str, Any],
    consistency_checks: dict[str, bool],
    safety_checks: dict[str, bool],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not consistency_checks["broker_order_id_consistent"] or not consistency_checks["client_order_id_consistent"]:
        reasons.append("broker_order_id or client_order_id mismatch across artifacts")
    if not all(safety_checks.values()):
        reasons.append("unsafe artifact flag detected")
    if not consistency_checks["status_report_present"]:
        reasons.append("broker status is not available")

    state, state_reasons = _suggest_state(
        submitted=submit_summary["submitted"],
        status=order_summary["last_known_status"],
        remaining_quantity=order_summary["remaining_quantity"],
        cancel_attempted=order_summary["cancel_attempted"] or smoke_summary["cancel_attempted"],
        cancel_succeeded=order_summary["cancel_succeeded"],
        id_consistent=(
            consistency_checks["broker_order_id_consistent"]
            and consistency_checks["client_order_id_consistent"]
        ),
        safety_clean=all(safety_checks.values()),
    )
    reasons.extend(state_reasons)
    reconciliation_required = bool(reasons) or state in {
        "submitted_unverified",
        "broker_partially_filled",
        "broker_rejected_or_inactive",
        "cancel_requested_unverified",
        "needs_reconciliation",
        "unsafe_artifact",
        "unknown_broker_status",
    }
    if reconciliation_required and not reasons:
        reasons.append(f"{state} requires manual reconciliation")

    broker_order_id = ids["broker_order_id"]
    client_order_id = ids["client_order_id"]
    return {
        "available": bool(submit_summary["submitted"] and broker_order_id and client_order_id),
        "broker_order_id": broker_order_id,
        "client_order_id": client_order_id,
        "strategy_id": _first_non_empty(submit_summary["strategy_id"], smoke_summary["strategy_id"]),
        "symbol": _first_non_empty(submit_summary["symbol"], smoke_summary["symbol"]),
        "action": _first_non_empty(submit_summary["action"], smoke_summary["action"]),
        "quantity": _first_non_empty(submit_summary["quantity"], smoke_summary["quantity"]),
        "order_type": _first_non_empty(submit_summary["order_type"], smoke_summary["order_type"]),
        "submitted": submit_summary["submitted"],
        "last_known_broker_status": order_summary["last_known_status"],
        "filled_quantity": order_summary["filled_quantity"],
        "remaining_quantity": order_summary["remaining_quantity"],
        "avg_fill_price": order_summary["avg_fill_price"],
        "cancel_attempted": order_summary["cancel_attempted"] or smoke_summary["cancel_attempted"],
        "cancel_succeeded": _first_non_empty(order_summary["cancel_succeeded"], smoke_summary["cancel_succeeded"]),
        "terminal_or_safe_state_seen": _terminal_or_safe_state_seen(
            order_summary["last_known_status"],
            smoke_summary["terminal_or_safe_state_seen"],
            order_summary["cancel_succeeded"],
        ),
        "suggested_internal_lifecycle_state": state,
        "reconciliation_required": reconciliation_required,
        "reconciliation_reasons": _dedupe(reasons),
    }


def _empty_candidate(state: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "available": False,
        "broker_order_id": None,
        "client_order_id": None,
        "strategy_id": None,
        "symbol": None,
        "action": None,
        "quantity": None,
        "order_type": None,
        "submitted": False,
        "last_known_broker_status": None,
        "filled_quantity": None,
        "remaining_quantity": None,
        "avg_fill_price": None,
        "cancel_attempted": False,
        "cancel_succeeded": None,
        "terminal_or_safe_state_seen": False,
        "suggested_internal_lifecycle_state": state,
        "reconciliation_required": True,
        "reconciliation_reasons": list(reasons),
    }


def _suggest_state(
    *,
    submitted: bool,
    status: Any,
    remaining_quantity: Any,
    cancel_attempted: bool,
    cancel_succeeded: bool | None,
    id_consistent: bool,
    safety_clean: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not safety_clean:
        return "unsafe_artifact", ["unsafe safety flag found in supplied artifacts"]
    if not id_consistent:
        return "needs_reconciliation", ["broker_order_id or client_order_id mismatch"]
    if not submitted:
        return "unknown_broker_status", ["submit report does not confirm submission"]

    normalized = _normalize_status(status)
    if normalized is None:
        if cancel_attempted:
            return "cancel_requested_unverified", ["cancel was attempted but final broker status is unknown"]
        return "submitted_unverified", ["submitted order has no broker status report"]

    if normalized in STATUS_SUBMITTED:
        return "broker_submitted", []
    if normalized in STATUS_PARTIAL:
        return "broker_partially_filled", ["partial fill requires manual operator review"]
    if normalized in STATUS_FILLED:
        remaining = _safe_float(remaining_quantity)
        if remaining == 0.0:
            return "broker_filled", []
        if remaining is not None and remaining > 0:
            return "broker_partially_filled", ["Filled status still has non-zero remaining_quantity"]
        return "unknown_broker_status", ["remaining_quantity could not be parsed for Filled status"]
    if normalized in STATUS_CANCELLED:
        return "broker_cancelled", []
    if normalized in STATUS_REJECTED:
        return "broker_rejected_or_inactive", ["broker status is rejected or inactive"]
    if cancel_attempted and cancel_succeeded is not True:
        return "cancel_requested_unverified", ["cancel was attempted but final broker status is unknown"]
    return "unknown_broker_status", [f"unrecognized broker status: {status}"]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
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


def _safety_checks(reports: list[Any], blockers: list[str]) -> dict[str, bool]:
    checks = _default_safety_checks()
    for index, report in enumerate(reports):
        if not isinstance(report, dict):
            continue
        for source_key, result_key in SAFETY_FLAG_KEYS.items():
            if _contains_true_flag(report, source_key):
                checks[result_key] = False
                blockers.append(f"unsafe flag enabled: {source_key} in supplied report {index}")
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
        value = payload.get(key)
        count = _count_value(value)
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


def _status_value(report: dict[str, Any]) -> str | None:
    for source in (report, _mapping(report.get("status")), _mapping(report.get("order"))):
        value = source.get("status") or source.get("state") or source.get("broker_status")
        if value not in (None, "") and not isinstance(value, (dict, list, tuple)):
            return str(value)
    return None


def _cancel_succeeded(report: dict[str, Any]) -> bool | None:
    for source in (report, _mapping(report.get("cancel")), _mapping(report.get("order"))):
        if "cancelled" in source:
            return source.get("cancelled") is True
        if "canceled" in source:
            return source.get("canceled") is True
        if "accepted" in source:
            return source.get("accepted") is True
    return None


def _terminal_or_safe_state_seen(
    status: Any,
    smoke_terminal_or_safe: bool,
    cancel_succeeded: bool | None,
) -> bool:
    normalized = _normalize_status(status)
    return (
        smoke_terminal_or_safe
        or cancel_succeeded is True
        or normalized in STATUS_FILLED
        or normalized in STATUS_CANCELLED
        or normalized in STATUS_REJECTED
    )


def _operator_left_open(report: dict[str, Any] | None) -> bool:
    root = _mapping(report)
    notes = _mapping(root.get("operator_notes"))
    smoke = _mapping(root.get("smoke_test"))
    return (
        notes.get("order_intentionally_left_open") is True
        and bool(str(notes.get("manual_observation") or "").strip())
    ) or smoke.get("order_intentionally_left_open") is True


def _normalize_status(value: Any) -> str | None:
    if value in (None, "") or isinstance(value, (dict, list, tuple)):
        return None
    return str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _present(value: Any) -> bool:
    return value not in (None, "")


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _all_same_present(values: list[Any]) -> bool:
    if not values:
        return False
    normalized = [str(value) for value in values if _present(value)]
    return bool(normalized) and len(set(normalized)) == 1


def _bool_or_none(value: Any) -> bool | None:
    if value is True:
        return True
    if value is False:
        return False
    return None


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
