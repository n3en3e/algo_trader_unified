"""Pure Stage 4F-5 manual one-order smoke-test acceptance reporting."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
TERMINAL_OR_SAFE_STATUSES = {
    "filled",
    "cancelled",
    "canceled",
    "inactive",
    "api cancelled",
    "closed",
    "done",
}
ORDERED_NEXT_STEPS = [
    "Archive the Stage 4F-2 preflight, Stage 4E-4 ticket, Stage 4F-3 submit report, and Stage 4F-4 follow-up report together.",
    "Review the smoke-test evidence before building the final Stage 4F acceptance report.",
    "Keep any further paper-order checks manual and single-order until Stage 4F-6 is complete.",
]
DO_NOT_DO_YET = [
    "Do not enable automated paper trading.",
    "Do not wire paper execution into daemon, scheduler, or lifecycle jobs.",
    "Do not enable live trading.",
    "Do not add market-data or contract-qualification behavior.",
]
SAFETY_FLAG_KEYS = {
    "live_orders_enabled": "no_live_orders",
    "market_data_enabled": "no_market_data",
    "contract_qualification_enabled": "no_contract_qualification",
    "scheduler_changes_enabled": "no_scheduler_changes",
    "lifecycle_wiring_enabled": "no_lifecycle_wiring",
}
UNSAFE_ERROR_HINTS = (
    "live",
    "market data",
    "contract qualification",
    "scheduler",
    "lifecycle",
    "automation",
)


def build_stage4f5_smoke_test_report(
    *,
    connection_preflight_report: dict | None,
    ticket_report: dict | None,
    submit_report: dict | None,
    order_control_reports: list[dict] | None = None,
    operator_notes: dict | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Evaluate supplied Stage 4F smoke-test artifacts without side effects."""

    try:
        return _build_report(
            connection_preflight_report=connection_preflight_report,
            ticket_report=ticket_report,
            submit_report=submit_report,
            order_control_reports=order_control_reports,
            operator_notes=operator_notes,
            now_provider=now_provider,
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must stay JSON-safe.
        return _json_safe(
            {
                "dry_run": True,
                "stage4f5_smoke_test_report": True,
                "generated_at": _generated_at(now_provider),
                "success": False,
                "errors": [f"unexpected report failure: {type(exc).__name__}: {exc}"],
                "warnings": [],
            }
        )


def _build_report(
    *,
    connection_preflight_report: dict | None,
    ticket_report: dict | None,
    submit_report: dict | None,
    order_control_reports: list[dict] | None,
    operator_notes: dict | None,
    now_provider: Callable[[], Any] | None,
) -> dict:
    generated_at = _generated_at(now_provider)
    preflight = connection_preflight_report if isinstance(connection_preflight_report, dict) else None
    ticket = ticket_report if isinstance(ticket_report, dict) else None
    submit = submit_report if isinstance(submit_report, dict) else None
    controls = [item for item in (order_control_reports or []) if isinstance(item, dict)]
    notes = operator_notes if isinstance(operator_notes, dict) else {}

    errors: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []

    if connection_preflight_report is not None and preflight is None:
        blockers.append("connection preflight report must be a dict")
    if ticket_report is not None and ticket is None:
        blockers.append("ticket report must be a dict")
    if submit_report is not None and submit is None:
        blockers.append("submit report must be a dict")
    if operator_notes is not None and notes is not operator_notes:
        warnings.append("operator_notes ignored because it is not a dict")

    connection_ok = _check_connection_preflight(preflight, blockers)
    ticket_ok, ticket_client_order_id = _check_ticket(ticket, blockers)
    submit_ok, submit_summary = _check_submit(submit, blockers)

    broker_order_id = submit_summary["broker_order_id"]
    submit_client_order_id = submit_summary["client_order_id"]
    client_order_id = submit_client_order_id or ticket_client_order_id
    client_match = bool(ticket_client_order_id and submit_client_order_id and ticket_client_order_id == submit_client_order_id)
    if ticket_ok and submit_ok and not client_match:
        blockers.append("submit client_order_id must match ticket client_order_id")

    submitted_ids = _submitted_broker_order_ids(submit)
    no_extra_submissions = len(set(submitted_ids)) <= 1
    if not no_extra_submissions:
        blockers.append("multiple submitted broker_order_ids detected")

    order_control = _summarize_order_controls(
        controls=controls,
        submitted_broker_order_id=broker_order_id,
        blockers=blockers,
        warnings=warnings,
    )

    status_seen = bool(order_control["status_seen"])
    cancel_seen = bool(order_control["cancel_seen"])
    terminal_or_safe_state_seen = bool(order_control["terminal_or_safe_state_seen"])
    if broker_order_id and not status_seen:
        blockers.append("at least one matching status report is required")
    if broker_order_id and not terminal_or_safe_state_seen:
        if _operator_left_open(notes):
            warnings.append("operator notes document that the paper order was intentionally left open for follow-up")
            if status_seen:
                terminal_or_safe_state_seen = True
        else:
            blockers.append("terminal or operator-documented safe follow-up state was not seen")
            warnings.append("paper order may still be open because no terminal status or accepted open follow-up note was supplied")
    if broker_order_id and not cancel_seen and not terminal_or_safe_state_seen:
        warnings.append("no cancel report was supplied; verify the paper order is filled, closed, cancelled, or intentionally tracked")

    broker_match = bool(broker_order_id) and not order_control["mismatched_broker_order_id"]
    safety_checks = _safety_checks(
        [preflight, ticket, submit, *controls],
        blockers=blockers,
    )
    _unsafe_errors([preflight, ticket, submit, *controls], blockers)

    one_order_only = (
        submit_ok
        and bool(broker_order_id)
        and no_extra_submissions
        and broker_match
    )
    accepted = (
        connection_ok
        and ticket_ok
        and submit_ok
        and one_order_only
        and client_match
        and status_seen
        and terminal_or_safe_state_seen
        and all(safety_checks.values())
        and not blockers
    )

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    ready = (
        accepted
        and not blocker_list
        and one_order_only
        and status_seen
        and broker_match
        and client_match
        and all(safety_checks.values())
    )

    report = {
        "dry_run": True,
        "stage4f5_smoke_test_report": True,
        "generated_at": generated_at,
        "smoke_test": {
            "accepted": accepted,
            "one_order_only": one_order_only,
            "broker_order_id": broker_order_id,
            "client_order_id": client_order_id,
            "submitted": submit_summary["submitted"],
            "status_seen": status_seen,
            "cancel_seen": cancel_seen,
            "terminal_or_safe_state_seen": terminal_or_safe_state_seen,
        },
        "artifact_checks": {
            "connection_preflight_present": preflight is not None,
            "ticket_report_present": ticket is not None,
            "submit_report_present": submit is not None,
            "order_control_report_count": len(controls),
        },
        "sequence_checks": {
            "connection_preflight_ok": connection_ok,
            "ticket_ok": ticket_ok,
            "submit_ok": submit_ok,
            "client_order_id_matches": client_match,
            "broker_order_id_matches": broker_match,
            "status_report_present": status_seen,
            "cancel_report_present": cancel_seen,
        },
        "safety_checks": {
            **safety_checks,
            "no_extra_submissions_detected": no_extra_submissions,
        },
        "order_control_summary": {
            "statuses": order_control["statuses"],
            "cancel_attempted": order_control["cancel_attempted"],
            "cancel_succeeded": order_control["cancel_succeeded"],
            "last_known_status": order_control["last_known_status"],
        },
        "operator_notes": _json_safe(notes),
        "readiness_for_stage4f6": {
            "ready_for_stage4f_acceptance_report": ready,
            "blockers": blocker_list,
            "warnings": warning_list,
        },
        "recommendations": {
            "ordered_next_steps": list(ORDERED_NEXT_STEPS),
            "do_not_do_yet": list(DO_NOT_DO_YET),
        },
        "success": True,
        "errors": _dedupe(errors),
        "warnings": warning_list,
    }
    return _json_safe(report)


def _check_connection_preflight(report: dict[str, Any] | None, blockers: list[str]) -> bool:
    if report is None:
        blockers.append("connection preflight report is required")
        return False
    ok = True
    connection = _mapping(report.get("connection"))
    readonly_checks = _mapping(report.get("readonly_checks"))
    safety = _mapping(report.get("safety"))
    checks = {
        "ibkr_paper_connection_preflight must be True": report.get("ibkr_paper_connection_preflight") is True,
        "connection allow_real_ibkr must be True": connection.get("allow_real_ibkr") is True,
        "connection.connected must be True": connection.get("connected") is True,
        "readonly current_time_ok must be True": readonly_checks.get("current_time_ok") is True,
        "readonly account_snapshot_ok must be True": readonly_checks.get("account_snapshot_ok") is True,
        "readonly open_orders_ok must be True": readonly_checks.get("open_orders_ok") is True,
        "readonly positions_ok must be True": readonly_checks.get("positions_ok") is True,
        "preflight paper_order_submission_enabled must be False": safety.get("paper_order_submission_enabled") is False,
        "preflight live_orders_enabled must be False": safety.get("live_orders_enabled") is False,
        "preflight market_data_enabled must be False": safety.get("market_data_enabled") is False,
        "preflight contract_qualification_enabled must be False": safety.get("contract_qualification_enabled") is False,
        "preflight scheduler_changes_enabled must be False": safety.get("scheduler_changes_enabled") is False,
        "preflight lifecycle_wiring_enabled must be False": safety.get("lifecycle_wiring_enabled") is False,
    }
    for reason, passed in checks.items():
        if not passed:
            blockers.append(reason)
            ok = False
    return ok


def _check_ticket(report: dict[str, Any] | None, blockers: list[str]) -> tuple[bool, Any]:
    if report is None:
        blockers.append("ticket report is required")
        return False, None
    submit_gate = _mapping(report.get("submit_gate"))
    plan = _mapping(report.get("ibkr_order_plan"))
    client_order_id = plan.get("client_order_id")
    checks = {
        "ticket paper_order_ticket_report must be True": report.get("paper_order_ticket_report") is True,
        "ticket submit gate must be eligible": submit_gate.get("eligible_for_future_manual_submit") is True,
        "ticket plan ready_for_submission must be True": plan.get("ready_for_submission") is True,
        "ticket plan paper_only must be True": plan.get("paper_only") is True,
        "ticket plan dry_run must be True": plan.get("dry_run") is True,
        "ticket plan blockers must be empty": not bool(plan.get("blockers")),
        "ticket client_order_id is required": bool(client_order_id),
    }
    ok = True
    for reason, passed in checks.items():
        if not passed:
            blockers.append(reason)
            ok = False
    return ok, client_order_id


def _check_submit(report: dict[str, Any] | None, blockers: list[str]) -> tuple[bool, dict[str, Any]]:
    submission = _mapping(report.get("submission")) if isinstance(report, dict) else {}
    summary = {
        "submitted": submission.get("submitted") is True,
        "broker_order_id": submission.get("broker_order_id"),
        "client_order_id": submission.get("client_order_id"),
    }
    if report is None:
        blockers.append("submit report is required")
        return False, summary
    gates = _mapping(report.get("gates"))
    safety = _mapping(report.get("safety"))
    checks = {
        "submit manual_real_paper_submit must be True": report.get("manual_real_paper_submit") is True,
        "submit gates.passed must be True": gates.get("passed") is True,
        "submit submission.attempted must be True": submission.get("attempted") is True,
        "submit submission.submitted must be True": submission.get("submitted") is True,
        "submit broker_order_id is required": bool(summary["broker_order_id"]),
        "submit client_order_id is required": bool(summary["client_order_id"]),
        "submit live_orders_enabled must be False": safety.get("live_orders_enabled") is False,
        "submit scheduler_changes_enabled must be False": safety.get("scheduler_changes_enabled") is False,
        "submit market_data_enabled must be False": safety.get("market_data_enabled") is False,
        "submit contract_qualification_enabled must be False": safety.get("contract_qualification_enabled") is False,
    }
    ok = True
    for reason, passed in checks.items():
        if not passed:
            blockers.append(reason)
            ok = False
    return ok, summary


def _summarize_order_controls(
    *,
    controls: list[dict[str, Any]],
    submitted_broker_order_id: Any,
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    statuses: list[str] = []
    status_seen = False
    cancel_seen = False
    cancel_attempted = False
    cancel_succeeded: bool | None = None
    last_known_status: str | None = None
    terminal_or_safe = False
    mismatched = False

    for index, report in enumerate(controls):
        action = report.get("action")
        order = _mapping(report.get("order"))
        safety = _mapping(report.get("safety"))
        broker_order_id = order.get("broker_order_id")
        prefix = f"order_control[{index}]"
        if report.get("manual_real_paper_order_control") is not True:
            blockers.append(f"{prefix} manual_real_paper_order_control must be True")
        if action not in {"status", "cancel"}:
            blockers.append(f"{prefix} action must be status or cancel")
        if submitted_broker_order_id and broker_order_id != submitted_broker_order_id:
            blockers.append(f"{prefix} broker_order_id must match submitted broker_order_id")
            mismatched = True
        for key in (
            "live_orders_enabled",
            "paper_order_submission_enabled",
            "market_data_enabled",
            "contract_qualification_enabled",
            "scheduler_changes_enabled",
            "lifecycle_wiring_enabled",
        ):
            if safety.get(key) is not False:
                blockers.append(f"{prefix} safety.{key} must be False")
        if action == "status" and broker_order_id == submitted_broker_order_id:
            status_seen = True
            status = _status_value(report)
            if status:
                statuses.append(status)
                last_known_status = status
                if _is_terminal_or_safe_status(status):
                    terminal_or_safe = True
        if action == "cancel" and broker_order_id == submitted_broker_order_id:
            cancel_seen = True
            cancel_attempted = True
            cancel_succeeded = _cancel_succeeded(report)
            if cancel_succeeded is True:
                terminal_or_safe = True
            elif cancel_succeeded is False:
                reason = _cancel_reason(report)
                if reason:
                    warnings.append(f"cancel did not succeed: {reason}")
                else:
                    warnings.append("cancel did not succeed and no reason was supplied")

    return {
        "statuses": statuses,
        "status_seen": status_seen,
        "cancel_seen": cancel_seen,
        "cancel_attempted": cancel_attempted,
        "cancel_succeeded": cancel_succeeded,
        "last_known_status": last_known_status,
        "terminal_or_safe_state_seen": terminal_or_safe,
        "mismatched_broker_order_id": mismatched,
    }


def _safety_checks(reports: list[Any], *, blockers: list[str]) -> dict[str, bool]:
    checks = {value: True for value in SAFETY_FLAG_KEYS.values()}
    for label, report in enumerate(reports):
        if not isinstance(report, dict):
            continue
        for key, result_key in SAFETY_FLAG_KEYS.items():
            if _contains_true_flag(report, key):
                checks[result_key] = False
                blockers.append(f"unsafe flag enabled: {key} in supplied report {label}")
    return checks


def _contains_true_flag(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return any((item_key == key and item is True) or _contains_true_flag(item, key) for item_key, item in value.items())
    if isinstance(value, list):
        return any(_contains_true_flag(item, key) for item in value)
    return False


def _unsafe_errors(reports: list[Any], blockers: list[str]) -> None:
    for index, report in enumerate(reports):
        if not isinstance(report, dict):
            continue
        for error in report.get("errors") or []:
            text = str(error).lower()
            if any(hint in text for hint in UNSAFE_ERROR_HINTS):
                blockers.append(f"unsafe error in supplied report {index}: {error}")


def _submitted_broker_order_ids(report: dict[str, Any] | None) -> list[Any]:
    if not isinstance(report, dict):
        return []
    ids: list[Any] = []
    submission = _mapping(report.get("submission"))
    if submission.get("submitted") is True and submission.get("broker_order_id"):
        ids.append(submission.get("broker_order_id"))
    for item in report.get("submissions") or []:
        if isinstance(item, dict) and item.get("submitted") is True and item.get("broker_order_id"):
            ids.append(item.get("broker_order_id"))
    return ids


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


def _cancel_reason(report: dict[str, Any]) -> str | None:
    for source in (report, _mapping(report.get("cancel")), _mapping(report.get("order"))):
        value = source.get("reason") or source.get("failure_reason")
        if value not in (None, ""):
            return str(value)
    return None


def _is_terminal_or_safe_status(status: str) -> bool:
    return status.strip().lower() in TERMINAL_OR_SAFE_STATUSES


def _operator_left_open(notes: dict[str, Any]) -> bool:
    return (
        notes.get("order_intentionally_left_open") is True
        and bool(str(notes.get("manual_observation") or "").strip())
    )


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _generated_at(now_provider: Callable[[], Any] | None) -> str:
    if now_provider is None:
        return DEFAULT_GENERATED_AT
    value = now_provider()
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


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
