"""Stage 4E-4 manual paper order ticket readiness report.

This module is a pure reporting boundary. It translates one injected order
intent into a broker request and IBKR paper order plan, but it never submits,
cancels, fetches market data, qualifies contracts, or wires lifecycle jobs.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from algo_trader_unified.core.ibkr_paper_order_mapper import (
    IbkrPaperConfig,
    build_ibkr_paper_order_plan,
    validate_ibkr_paper_config,
)
from algo_trader_unified.core.paper_broker_adapter import (
    BrokerOrderRequest,
    build_broker_order_request,
    sanitize_json_safe,
)


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"


def build_paper_order_ticket_report(
    *,
    intent: dict,
    ibkr_config: dict | IbkrPaperConfig,
    now_provider: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    generated_at = _generated_at(now_provider)

    request: BrokerOrderRequest | None = None
    config: IbkrPaperConfig | None = None
    plan: Any | None = None

    intent_report = _intent_stub(intent)
    request_report = _request_report(None)
    config_report = _config_report(ibkr_config, valid=False, reason=None)
    plan_report = _plan_report(None)

    try:
        request = build_broker_order_request(intent)
        intent_report.update(
            {
                "valid": True,
                "validation_reason": None,
            }
        )
        request_report = _request_report(request)
    except Exception as exc:  # noqa: BLE001 - report boundary must not crash.
        reason = _exception_reason("build_broker_order_request", exc)
        errors.append(reason)
        intent_report.update({"valid": False, "validation_reason": reason})

    try:
        config_input = _config_input_dict(ibkr_config)
        config = validate_ibkr_paper_config(config_input)
        config_report = _config_report(config, valid=True, reason=None)
    except Exception as exc:  # noqa: BLE001
        reason = _exception_reason("validate_ibkr_paper_config", exc)
        errors.append(reason)
        config_report = _config_report(ibkr_config, valid=False, reason=reason)

    if request is not None and config is not None:
        try:
            plan = build_ibkr_paper_order_plan(request, config=config)
            plan_report = _plan_report(plan)
            plan_warnings = getattr(plan, "warnings", [])
            if isinstance(plan_warnings, list):
                warnings.extend(str(item) for item in plan_warnings)
        except Exception as exc:  # noqa: BLE001
            reason = _exception_reason("build_ibkr_paper_order_plan", exc)
            errors.append(reason)
            plan_report = _plan_report(None)
    else:
        if request is None:
            warnings.append("IBKR paper order plan was not built because intent validation failed.")
        if config is None:
            warnings.append("IBKR paper order plan was not built because IBKR paper config failed.")

    safety = _safety()
    submit_gate = _submit_gate(
        intent_valid=bool(intent_report["valid"]),
        config_valid=bool(config_report["valid"]),
        plan_report=plan_report,
        safety=safety,
    )
    recommendations = _recommendations(submit_gate["eligible_for_future_manual_submit"])

    report = {
        "dry_run": True,
        "paper_order_ticket_report": True,
        "generated_at": generated_at,
        "intent": intent_report,
        "broker_order_request": request_report,
        "ibkr_config": config_report,
        "ibkr_order_plan": plan_report,
        "submit_gate": submit_gate,
        "safety": safety,
        "recommendations": recommendations,
        "success": True,
        "errors": errors,
        "warnings": warnings,
    }
    return _json_safe(report)


def _generated_at(now_provider: Callable[[], Any] | None) -> str:
    if now_provider is None:
        return DEFAULT_GENERATED_AT
    value = now_provider()
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _intent_stub(intent: Any) -> dict[str, Any]:
    source = intent if isinstance(intent, dict) else {}
    return {
        "intent_id": source.get("intent_id"),
        "strategy_id": source.get("strategy_id"),
        "symbol": source.get("symbol") or source.get("underlying"),
        "side": source.get("side"),
        "quantity": source.get("quantity"),
        "order_type": source.get("order_type"),
        "valid": False,
        "validation_reason": None,
    }


def _request_report(request: BrokerOrderRequest | None) -> dict[str, Any]:
    if request is None:
        return {
            "available": False,
            "client_order_id": None,
            "strategy_id": None,
            "symbol": None,
            "asset_type": None,
            "side": None,
            "quantity": None,
            "order_type": None,
            "limit_price": None,
            "time_in_force": None,
            "intent_id": None,
        }
    return {
        "available": True,
        "client_order_id": request.client_order_id,
        "strategy_id": request.strategy_id,
        "symbol": request.symbol,
        "asset_type": request.asset_type,
        "side": request.side,
        "quantity": request.quantity,
        "order_type": request.order_type,
        "limit_price": request.limit_price,
        "time_in_force": request.time_in_force,
        "intent_id": request.intent_id,
    }


def _config_input_dict(ibkr_config: dict | IbkrPaperConfig) -> dict[str, Any]:
    if isinstance(ibkr_config, IbkrPaperConfig):
        return asdict(ibkr_config)
    if isinstance(ibkr_config, dict):
        return dict(ibkr_config)
    raise ValueError("ibkr_config must be a dict or IbkrPaperConfig")


def _config_report(
    ibkr_config: Any,
    *,
    valid: bool,
    reason: str | None,
) -> dict[str, Any]:
    if isinstance(ibkr_config, IbkrPaperConfig):
        source = asdict(ibkr_config)
    elif isinstance(ibkr_config, dict):
        source = ibkr_config
    else:
        source = {}
    return {
        "trading_mode": source.get("trading_mode"),
        "host": source.get("host"),
        "port": source.get("port"),
        "client_id": source.get("client_id"),
        "account_id": source.get("account_id"),
        "valid": valid,
        "validation_reason": reason,
    }


def _plan_report(plan: Any | None) -> dict[str, Any]:
    if plan is None:
        return {
            "available": False,
            "ready_for_submission": False,
            "client_order_id": None,
            "action": None,
            "quantity": None,
            "order_type": None,
            "limit_price": None,
            "time_in_force": None,
            "paper_only": False,
            "dry_run": True,
            "blockers": [],
            "warnings": [],
            "ibkr_contract_hint": {},
            "ibkr_order_hint": {},
        }
    return {
        "available": True,
        "ready_for_submission": bool(getattr(plan, "ready_for_submission", False)),
        "client_order_id": getattr(plan, "client_order_id", None),
        "action": getattr(plan, "action", None),
        "quantity": getattr(plan, "quantity", None),
        "order_type": getattr(plan, "order_type", None),
        "limit_price": getattr(plan, "limit_price", None),
        "time_in_force": getattr(plan, "time_in_force", None),
        "paper_only": bool(getattr(plan, "paper_only", False)),
        "dry_run": bool(getattr(plan, "dry_run", False)),
        "blockers": list(getattr(plan, "blockers", []) or []),
        "warnings": list(getattr(plan, "warnings", []) or []),
        "ibkr_contract_hint": _plain_dict(getattr(plan, "ibkr_contract_hint", {})),
        "ibkr_order_hint": _plain_dict(getattr(plan, "ibkr_order_hint", {})),
    }


def _plain_dict(value: Any) -> dict[str, Any]:
    safe = sanitize_json_safe(value)
    if type(safe) is dict:
        return safe
    return {"value": safe}


def _submit_gate(
    *,
    intent_valid: bool,
    config_valid: bool,
    plan_report: dict[str, Any],
    safety: dict[str, bool],
) -> dict[str, Any]:
    reasons: list[str] = []

    if not intent_valid:
        reasons.append("intent does not validate into BrokerOrderRequest")
    if not config_valid:
        reasons.append("IBKR config does not validate as PAPER on port 4004")
    if not plan_report["available"]:
        reasons.append("IBKR paper order plan is not available")
    if plan_report["available"] and not plan_report["ready_for_submission"]:
        reasons.append("IBKR paper order plan is not ready for submission")
    if not plan_report["paper_only"]:
        reasons.append("IBKR paper order plan is not marked paper_only")
    if not plan_report["dry_run"]:
        reasons.append("IBKR paper order plan is not marked dry_run")
    if plan_report["blockers"]:
        reasons.append("IBKR paper order plan has blockers")
    if not _non_empty_string(plan_report["client_order_id"]):
        reasons.append("client_order_id is missing")

    enabled_flags = [key for key, value in safety.items() if value is True]
    if enabled_flags:
        reasons.append(f"safety flags are enabled: {', '.join(sorted(enabled_flags))}")

    eligible = not reasons
    return {
        "eligible_for_future_manual_submit": eligible,
        "reasons": reasons,
        "required_operator_acknowledgements": _operator_acknowledgements()
        if eligible
        else [],
    }


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _operator_acknowledgements() -> list[str]:
    return [
        "I understand this is PAPER only.",
        "I understand no live orders are allowed.",
        "I understand this ticket is not yet submitted.",
        "I understand scheduler/lifecycle wiring remains disabled.",
    ]


def _recommendations(eligible: bool) -> dict[str, list[str]]:
    steps = ["Review this ticket manually before any future paper submit phase."]
    if not eligible:
        steps.extend(
            [
                "Fix intent validation errors before paper submit.",
                "Fix IBKR paper config before paper submit.",
                "Resolve order plan blockers before paper submit.",
            ]
        )
    return {
        "ordered_next_steps": steps,
        "do_not_do_yet": [
            "Do not wire paper execution into scheduler/lifecycle yet.",
            "Do not enable live trading.",
            "Do not change strategy thresholds or sizing for this ticket.",
            "Do not bypass readiness gates.",
        ],
    }


def _safety() -> dict[str, bool]:
    return {
        "broker_calls_enabled": False,
        "order_submission_enabled": False,
        "cancel_enabled": False,
        "market_data_enabled": False,
        "contract_qualification_enabled": False,
        "live_orders_enabled": False,
        "scheduler_changes_enabled": False,
    }


def _exception_reason(stage: str, exc: BaseException) -> str:
    return f"{stage} failed: {type(exc).__name__}: {exc}"


def _json_safe(report: dict[str, Any]) -> dict[str, Any]:
    safe = sanitize_json_safe(report)
    if is_dataclass(safe):
        safe = asdict(safe)
    if isinstance(safe, dict):
        return safe
    return {"success": False, "errors": ["report was not JSON-safe"], "value": safe}
