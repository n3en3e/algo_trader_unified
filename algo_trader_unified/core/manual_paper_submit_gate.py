"""Stage 4E-5 manual paper submit gate.

This module is a pure manual gate around a pre-built Stage 4E-4 ticket report.
It never constructs broker clients, fetches market data, qualifies contracts, or
wires scheduler/lifecycle work. The only possible submit call is made against
an injected fake-client-like object after all explicit operator gates pass.
"""

from __future__ import annotations

from typing import Any, Callable

from algo_trader_unified.core.paper_broker_adapter import sanitize_json_safe


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"


def build_manual_paper_submit_result(
    *,
    ticket_report: dict,
    execution_client: Any,
    operator_acknowledgements: list[str],
    now_provider: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    generated_at = _generated_at(now_provider)

    report = ticket_report if isinstance(ticket_report, dict) else {}
    plan = _mapping(report.get("ibkr_order_plan"))
    ticket_submit_gate = _mapping(report.get("submit_gate"))
    source_safety = _mapping(report.get("safety"))
    acknowledgements = _acknowledgement_report(
        required=_string_list(ticket_submit_gate.get("required_operator_acknowledgements")),
        provided=operator_acknowledgements,
    )
    reasons = _gate_reasons(
        report=report,
        plan=plan,
        ticket_submit_gate=ticket_submit_gate,
        safety=source_safety,
        acknowledgements=acknowledgements,
    )
    passed = not reasons

    submission = {
        "attempted": False,
        "submitted": False,
        "broker_order_id": None,
        "client_order_id": plan.get("client_order_id"),
        "reason": None if passed else "; ".join(reasons),
        "raw": None,
    }

    if passed:
        submission["attempted"] = True
        try:
            broker_result = execution_client.submit_order_plan(plan)
            mapped = _submission_from_broker_result(broker_result)
            submission.update(mapped)
        except Exception as exc:  # noqa: BLE001 - report boundary must not crash.
            reason = f"{type(exc).__name__}: {exc}"
            errors.append(reason)
            submission.update(
                {
                    "submitted": False,
                    "broker_order_id": None,
                    "client_order_id": plan.get("client_order_id"),
                    "reason": reason,
                    "raw": {"operation": "fake_submit_order_plan", "error": reason},
                }
            )

    result = {
        "dry_run": True,
        "manual_paper_submit_gate": True,
        "generated_at": generated_at,
        "ticket": _ticket_summary(report, plan, ticket_submit_gate),
        "acknowledgements": acknowledgements,
        "submit_gate": {"passed": passed, "reasons": reasons},
        "submission": submission,
        "safety": {
            "fake_client_only": True,
            "real_ibkr_enabled": False,
            "order_submission_enabled": bool(submission["attempted"]),
            "live_orders_enabled": False,
            "market_data_enabled": False,
            "contract_qualification_enabled": False,
            "scheduler_changes_enabled": False,
            "lifecycle_wiring_enabled": False,
        },
        "recommendations": _recommendations(passed),
        "success": True,
        "errors": errors,
        "warnings": warnings,
    }
    return _json_safe(result)


def _generated_at(now_provider: Callable[[], Any] | None) -> str:
    if now_provider is None:
        return DEFAULT_GENERATED_AT
    value = now_provider()
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _normal_acknowledgements(values: list[str]) -> list[str]:
    return [item.strip() for item in values if isinstance(item, str)]


def _acknowledgement_report(*, required: list[str], provided: list[str]) -> dict[str, Any]:
    normalized_provided = _normal_acknowledgements(provided)
    missing = [item for item in required if item not in normalized_provided]
    return {
        "required": list(required),
        "provided": normalized_provided,
        "missing": missing,
        "exact_match": not missing,
    }


def _gate_reasons(
    *,
    report: dict[str, Any],
    plan: dict[str, Any],
    ticket_submit_gate: dict[str, Any],
    safety: dict[str, Any],
    acknowledgements: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if report.get("paper_order_ticket_report") is not True:
        reasons.append("ticket_report.paper_order_ticket_report must be True")
    if ticket_submit_gate.get("eligible_for_future_manual_submit") is not True:
        reasons.append("ticket is not eligible for future manual submit")
    if plan.get("ready_for_submission") is not True:
        reasons.append("IBKR paper order plan is not ready for submission")
    if plan.get("paper_only") is not True:
        reasons.append("IBKR paper order plan is not marked paper_only")
    if plan.get("dry_run") is not True:
        reasons.append("IBKR paper order plan is not marked dry_run")
    if plan.get("blockers"):
        reasons.append("IBKR paper order plan has blockers")
    if safety.get("order_submission_enabled") is not False:
        reasons.append("ticket safety order_submission_enabled must be False")
    if safety.get("live_orders_enabled") is not False:
        reasons.append("ticket safety live_orders_enabled must be False")
    if safety.get("scheduler_changes_enabled") is not False:
        reasons.append("ticket safety scheduler_changes_enabled must be False")
    for missing in acknowledgements["missing"]:
        reasons.append(f"missing required operator acknowledgement: {missing}")
    return reasons


def _ticket_summary(
    report: dict[str, Any],
    plan: dict[str, Any],
    ticket_submit_gate: dict[str, Any],
) -> dict[str, Any]:
    intent = _mapping(report.get("intent"))
    request = _mapping(report.get("broker_order_request"))
    return {
        "eligible_for_future_manual_submit": ticket_submit_gate.get(
            "eligible_for_future_manual_submit"
        ),
        "client_order_id": plan.get("client_order_id") or request.get("client_order_id"),
        "strategy_id": request.get("strategy_id") or intent.get("strategy_id"),
        "symbol": request.get("symbol") or intent.get("symbol"),
        "side": request.get("side") or intent.get("side"),
        "action": plan.get("action"),
        "quantity": plan.get("quantity") or request.get("quantity"),
        "order_type": plan.get("order_type") or request.get("order_type"),
        "paper_only": plan.get("paper_only"),
        "ready_for_submission": plan.get("ready_for_submission"),
    }


def _submission_from_broker_result(broker_result: Any) -> dict[str, Any]:
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
        "submitted": bool(source.get("accepted")),
        "broker_order_id": source.get("broker_order_id"),
        "client_order_id": source.get("client_order_id"),
        "reason": source.get("reason"),
        "raw": _json_safe_value(source.get("raw")),
    }


def _json_safe_value(value: Any) -> Any:
    return sanitize_json_safe(value)


def _recommendations(passed: bool) -> dict[str, list[str]]:
    ordered_next_steps = [
        "Review the manual gate result with an operator.",
        "Record the fake-client result alongside the Stage 4E-4 ticket report.",
    ]
    if not passed:
        ordered_next_steps.insert(0, "Resolve all manual submit gate refusal reasons.")
    return {
        "ordered_next_steps": ordered_next_steps,
        "do_not_do_yet": [
            "Do not connect this command to real IBKR.",
            "Do not enable live orders.",
            "Do not wire paper execution into scheduler or lifecycle jobs.",
            "Do not request market data or qualify contracts.",
        ],
    }


def _json_safe(result: dict[str, Any]) -> dict[str, Any]:
    safe = sanitize_json_safe(result)
    if isinstance(safe, dict):
        return safe
    return {"success": False, "errors": ["manual gate result was not JSON-safe"], "value": safe}
