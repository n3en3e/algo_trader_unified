"""Pure Stage 4K-5 controlled market data and contract qualification executor report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
import re
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding"
)
PAPER_IBKR_PORTS = {4004}
PAYLOAD_FALSE_FLAGS = (
    "allow_live_trading",
    "allow_broker_submission",
    "allow_order_submission",
    "allow_state_write",
    "allow_ledger_write",
)
PROPOSED_FALSE_FLAGS = (
    "allow_direct_reqMktData",
    "allow_direct_qualifyContracts",
    "allow_direct_reqContractDetails",
    "allow_strategy_scan",
    "allow_intent_creation",
    "allow_ticket_creation",
    "allow_order_submission",
    "allow_broker_submission",
    "allow_state_write",
    "allow_ledger_write",
    "live_trading_enabled",
    "all_strategies_enabled",
)
STALE_4J_KEYS = (
    "proposed_execution_permissions_for_4J5",
    "may_call_strategy_next_phase",
    "may_build_executor_next_phase",
    "may_fetch_market_data_next_phase",
)
STAGE4K4_BOUNDARY_FLAGS = (
    "no_market_data_fetched_now",
    "no_contracts_qualified_now",
    "no_provider_called_now",
    "no_direct_ib_call",
    "no_reqMktData",
    "no_qualifyContracts",
    "no_reqContractDetails",
    "no_strategy_scan",
    "no_intents_created",
    "no_tickets_created",
    "no_orders_submitted",
    "no_state_written",
    "no_ledger_written",
    "no_broker_submission",
    "no_live_trading",
    "no_all_strategy_enablement",
    "no_scheduler_registration",
    "no_lifecycle_execution",
)
SAFETY_FLAGS = (
    "no_live_trading",
    "no_all_strategy_enablement",
    "no_broker_submission_enabled",
    "no_order_submission",
    "no_intent_creation",
    "no_ticket_creation",
    "no_state_write",
    "no_ledger_write",
)
RESULT_UNSAFE_TRUE_FLAGS = (
    "live_trading_enabled",
    "broker_submission_enabled",
    "order_submitted",
    "order_submission_enabled",
    "state_written",
    "state_write_enabled",
    "ledger_written",
    "ledger_write_enabled",
    "direct_ib_call_made",
    "reqMktData_called",
    "qualifyContracts_called",
    "reqContractDetails_called",
)
RESULT_DISABLED_FLAGS = (
    "live_trading_enabled",
    "broker_submission_enabled",
    "order_submission_enabled",
    "state_write_enabled",
    "ledger_write_enabled",
    "direct_ib_call_made",
    "reqMktData_called",
    "qualifyContracts_called",
    "reqContractDetails_called",
)
ORDERED_NEXT_STEPS = [
    "Build Stage 4K-6 market data and contract qualification acceptance.",
    "Keep live trading disabled.",
    "Keep all other strategies disabled.",
    "Keep broker submission separately gated until the explicit broker-submission phase.",
    "Keep intents, tickets, state writes, and ledger writes separately gated until their explicit phases.",
    "Use the controlled provider results only as read-only inputs for future signal planning.",
]
DO_NOT_DO_YET = [
    "Do not enable live trading.",
    "Do not enable all strategies.",
    "Do not place orders now.",
    "Do not enable broker submission now.",
    "Do not create intents or tickets now.",
    "Do not write state or ledger now.",
    "Do not change strategy thresholds.",
    "Do not change position sizing.",
    "Do not bypass risk controls.",
    "Do not call direct market data methods.",
    "Do not call direct contract qualification methods.",
    "Do not run strategy scans now.",
]


def build_stage4k5_market_data_contract_executor_report(
    *,
    stage4k4_execution_gate_report: dict | None,
    controlled_market_data_provider: Any = None,
    controlled_contract_qualification_provider: Any = None,
    scheduler_activation_snapshot: dict | None = None,
    lifecycle_activation_snapshot: dict | None = None,
    activation_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    market_window_snapshot: dict | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a Stage 4K-5 report, executing only injected controlled providers when allowed."""

    try:
        return _json_safe(
            _build_report(
                stage4k4_execution_gate_report=stage4k4_execution_gate_report,
                controlled_market_data_provider=controlled_market_data_provider,
                controlled_contract_qualification_provider=controlled_contract_qualification_provider,
                scheduler_activation_snapshot=scheduler_activation_snapshot,
                lifecycle_activation_snapshot=lifecycle_activation_snapshot,
                activation_snapshot=activation_snapshot,
                state_snapshot=state_snapshot,
                risk_snapshot=risk_snapshot,
                scheduler_snapshot=scheduler_snapshot,
                lifecycle_snapshot=lifecycle_snapshot,
                paper_broker_snapshot=paper_broker_snapshot,
                market_window_snapshot=market_window_snapshot,
                now_provider=now_provider,
            )
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must stay JSON-safe.
        message = f"unexpected Stage 4K-5 executor failure: {_flat_exception(exc)}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_default_selected_strategy(),
                operation=_default_operation(),
                stage4k4_gate_checks={},
                execution_payload_checks={},
                provider_availability_checks={},
                market_data_execution_results=_market_data_results([], 0, 0, 0, False),
                contract_qualification_execution_results=_contract_results([], 0, 0, 0, False),
                provider_call_trace=[],
                applied_operations=[],
                failed_operations=[],
                skipped_operations=[],
                boundary_checks=_default_boundary_checks(),
                required_inputs_for_4k6=_required_inputs_for_4k6(),
                activation_snapshot_checks={},
                state_checks=_default_state_checks(),
                risk_checks=_default_risk_checks(),
                scheduler_checks=_default_scheduler_checks(),
                lifecycle_checks=_default_lifecycle_checks(),
                paper_broker_checks=_default_paper_broker_checks(),
                market_window_checks=_default_market_window_checks(),
                safety_checks=_default_safety_checks(),
                ready=False,
                blockers=[message],
                warnings=[],
                errors=[message],
            )
        )


def _build_report(
    *,
    stage4k4_execution_gate_report: Any,
    controlled_market_data_provider: Any,
    controlled_contract_qualification_provider: Any,
    scheduler_activation_snapshot: Any,
    lifecycle_activation_snapshot: Any,
    activation_snapshot: Any,
    state_snapshot: Any,
    risk_snapshot: Any,
    scheduler_snapshot: Any,
    lifecycle_snapshot: Any,
    paper_broker_snapshot: Any,
    market_window_snapshot: Any,
    now_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    generated_at = _generated_at(now_provider)
    data = stage4k4_execution_gate_report if isinstance(stage4k4_execution_gate_report, dict) else {}
    blockers: list[str] = []
    warnings: list[str] = []
    errors = _as_string_list(data.get("errors"))

    if stage4k4_execution_gate_report is None:
        blockers.append("Stage 4K-4 execution gate report is missing")
    elif not isinstance(stage4k4_execution_gate_report, dict):
        blockers.append("Stage 4K-4 execution gate report must be a dict")
        errors.append("Stage 4K-4 execution gate report must be a dict")

    selected_strategy_id = _selected_strategy_id(data)
    operation_id = _operation_id(data)
    proposed_payload = _mapping(data.get("proposed_4k5_execution_payload"))
    artifact_checks = _artifact_checks(stage4k4_execution_gate_report, data, selected_strategy_id, operation_id, proposed_payload)
    blockers.extend(_artifact_blockers(artifact_checks, data))

    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    operation, operation_blockers = _operation_checks(data, operation_id)
    payload_checks, payloads, payload_blockers = _execution_payload_checks(proposed_payload, selected_strategy_id, operation_id)
    stage4k4_checks, gate_blockers = _stage4k4_gate_checks(data)
    boundary_checks, boundary_blockers = _boundary_checks(data)
    safety_checks, safety_blockers = _safety_checks(data)
    provider_checks, provider_blockers = _provider_availability_checks(
        proposed_payload,
        payloads,
        controlled_market_data_provider,
        controlled_contract_qualification_provider,
    )
    activation_checks, activation_blockers, activation_warnings = _activation_snapshot_group_checks(
        scheduler_activation_snapshot=scheduler_activation_snapshot,
        lifecycle_activation_snapshot=lifecycle_activation_snapshot,
        activation_snapshot=activation_snapshot,
        selected_strategy_id=selected_strategy_id,
    )
    state_checks, state_blockers, state_warnings = _state_checks(state_snapshot)
    risk_checks, risk_blockers, risk_warnings = _risk_checks(risk_snapshot)
    scheduler_checks, scheduler_blockers, scheduler_warnings = _scheduler_checks(scheduler_snapshot, selected_strategy_id)
    lifecycle_checks, lifecycle_blockers, lifecycle_warnings = _lifecycle_checks(lifecycle_snapshot, selected_strategy_id)
    paper_broker_checks, broker_blockers, broker_warnings = _paper_broker_checks(paper_broker_snapshot)
    market_window_checks, market_blockers, market_warnings = _market_window_checks(market_window_snapshot)

    for group in (
        selected_blockers,
        operation_blockers,
        payload_blockers,
        gate_blockers,
        boundary_blockers,
        safety_blockers,
        provider_blockers,
        activation_blockers,
        state_blockers,
        risk_blockers,
        scheduler_blockers,
        lifecycle_blockers,
        broker_blockers,
        market_blockers,
    ):
        blockers.extend(group)
    for group in (
        activation_warnings,
        state_warnings,
        risk_warnings,
        scheduler_warnings,
        lifecycle_warnings,
        broker_warnings,
        market_warnings,
    ):
        warnings.extend(group)

    preflight_blockers = _dedupe(blockers)
    applied: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    md_counts = {"attempted": 0, "succeeded": 0, "failed": 0, "skipped": 0, "called": False}
    cq_counts = {"attempted": 0, "succeeded": 0, "failed": 0, "skipped": 0, "called": False}

    if preflight_blockers or errors:
        _mark_permission_skips(
            payloads,
            proposed_payload,
            selected_strategy_id,
            operation_id,
            trace,
            skipped,
            md_counts,
            cq_counts,
            "preflight validation blocked provider execution",
        )
    else:
        _execute_provider_category(
            provider_type="market_data",
            method_name="request_controlled_market_data",
            provider=controlled_market_data_provider,
            payloads=payloads["market_data_provider_payloads"],
            selected_strategy_id=selected_strategy_id,
            operation_id=operation_id,
            trace=trace,
            applied=applied,
            failed=failed,
            skipped=skipped,
            counts=md_counts,
            stop_state={"stop": False},
        )
        stop_state = {"stop": bool(failed)}
        _execute_provider_category(
            provider_type="contract_qualification",
            method_name="qualify_controlled_contracts",
            provider=controlled_contract_qualification_provider,
            payloads=payloads["contract_qualification_provider_payloads"],
            selected_strategy_id=selected_strategy_id,
            operation_id=operation_id,
            trace=trace,
            applied=applied,
            failed=failed,
            skipped=skipped,
            counts=cq_counts,
            stop_state=stop_state,
        )

    operation_blockers_after = []
    if failed:
        operation_blockers_after.append("failed provider operations are present")
    unexpected_skips = [item for item in skipped if isinstance(item, dict) and item.get("status") == "skipped"]
    if unexpected_skips:
        operation_blockers_after.append("unexpected skipped provider operations are present")
    result_blockers = _provider_result_blockers(trace)
    all_blockers = _dedupe(preflight_blockers + operation_blockers_after + result_blockers)
    error_list = _dedupe(errors)
    warning_list = _dedupe(warnings)
    ready = not all_blockers and not error_list

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
        operation=operation,
        stage4k4_gate_checks=stage4k4_checks,
        execution_payload_checks=payload_checks,
        provider_availability_checks=provider_checks,
        market_data_execution_results=_market_data_results(
            [item.get("result", {}) for item in trace if isinstance(item, dict) and item.get("provider_type") == "market_data"],
            md_counts["attempted"],
            md_counts["succeeded"],
            md_counts["failed"],
            md_counts["skipped"],
            md_counts["called"],
        ),
        contract_qualification_execution_results=_contract_results(
            [item.get("result", {}) for item in trace if isinstance(item, dict) and item.get("provider_type") == "contract_qualification"],
            cq_counts["attempted"],
            cq_counts["succeeded"],
            cq_counts["failed"],
            cq_counts["skipped"],
            cq_counts["called"],
        ),
        provider_call_trace=trace,
        applied_operations=applied,
        failed_operations=failed,
        skipped_operations=skipped,
        boundary_checks=boundary_checks,
        required_inputs_for_4k6=_required_inputs_for_4k6(),
        activation_snapshot_checks=activation_checks,
        state_checks=state_checks,
        risk_checks=risk_checks,
        scheduler_checks=scheduler_checks,
        lifecycle_checks=lifecycle_checks,
        paper_broker_checks=paper_broker_checks,
        market_window_checks=market_window_checks,
        safety_checks=safety_checks,
        ready=ready,
        blockers=all_blockers if not ready else [],
        warnings=warning_list,
        errors=error_list,
    )


def _artifact_checks(report: Any, data: dict[str, Any], selected_strategy_id: str | None, operation_id: str | None, proposed_payload: dict[str, Any]) -> dict[str, bool]:
    readiness = _mapping(data.get("readiness_for_stage4k5"))
    return {
        "stage4k4_report_present": isinstance(report, dict),
        "stage4k4_report_ready": (
            data.get("stage4k4_market_data_contract_execution_gate_report") is True
            and readiness.get("ready_to_execute_controlled_market_data_contract_providers") is True
            and data.get("success") is True
        ),
        "selected_strategy_present": isinstance(selected_strategy_id, str) and bool(selected_strategy_id),
        "operation_id_present": isinstance(operation_id, str) and bool(operation_id),
        "proposed_4k5_execution_payload_present": isinstance(data.get("proposed_4k5_execution_payload"), dict) and bool(proposed_payload),
    }


def _artifact_blockers(checks: dict[str, bool], data: dict[str, Any]) -> list[str]:
    labels = {
        "stage4k4_report_present": "Stage 4K-4 execution gate report is missing",
        "stage4k4_report_ready": "Stage 4K-4 execution gate report is not ready for Stage 4K-5",
        "selected_strategy_present": "selected strategy is missing from Stage 4K-4 report",
        "operation_id_present": "operation_id is missing from Stage 4K-4 report",
        "proposed_4k5_execution_payload_present": "proposed_4k5_execution_payload is missing or malformed",
    }
    blockers = [label for key, label in labels.items() if checks.get(key) is not True]
    if _as_list(data.get("errors")):
        blockers.append("Stage 4K-4 report contains errors")
    return blockers


def _selected_strategy_checks(report: dict[str, Any], selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str]]:
    selected = _mapping(report.get("selected_strategy"))
    blockers: list[str] = []
    if not selected_strategy_id:
        blockers.append("selected_strategy.selected_strategy_id is missing or invalid")
    if selected.get("paper_only") is not True:
        blockers.append("selected strategy must be paper_only true")
    if selected.get("one_strategy_only") is not True:
        blockers.append("selected strategy must be one_strategy_only true")
    return {"selected_strategy_id": selected_strategy_id, "paper_only": True, "one_strategy_only": True}, blockers


def _operation_checks(report: dict[str, Any], operation_id: str | None) -> tuple[dict[str, Any], list[str]]:
    operation = _mapping(report.get("operation"))
    blockers: list[str] = []
    if not operation_id:
        blockers.append("operation.operation_id is missing or invalid")
    if operation.get("live_trading_enabled") is not False:
        blockers.append("Stage 4K-4 operation must keep live trading disabled")
    if operation.get("broker_submission_enabled") is not False:
        blockers.append("Stage 4K-4 operation must keep broker submission disabled")
    return {
        "operation_id": operation_id,
        "operation_scope": operation.get("operation_scope") or "single_strategy_controlled_market_data_contract_execution",
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }, blockers


def _stage4k4_gate_checks(report: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    readiness = _mapping(report.get("readiness_for_stage4k5"))
    checks = {
        "stage4k4_report_true": report.get("stage4k4_market_data_contract_execution_gate_report") is True,
        "stage4k4_success": report.get("success") is True,
        "ready_to_execute_controlled_market_data_contract_providers": readiness.get("ready_to_execute_controlled_market_data_contract_providers") is True,
    }
    blockers = [f"Stage 4K-4 gate check {key} must be strict native boolean true" for key, value in checks.items() if value is not True]
    return checks, blockers


def _execution_payload_checks(payload: dict[str, Any], selected_strategy_id: str | None, operation_id: str | None) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[str]]:
    blockers: list[str] = []
    checks: dict[str, Any] = {
        "payload_present": bool(payload),
        "payload_is_dict": isinstance(payload, dict) and bool(payload),
        "payload_json_safe": _is_json_native(payload),
    }
    if not payload:
        return checks, {"market_data_provider_payloads": [], "contract_qualification_provider_payloads": []}, ["proposed_4k5_execution_payload is missing or malformed"]
    if not _is_json_native(payload):
        blockers.append("proposed_4k5_execution_payload must be JSON-safe")
    if payload.get("selected_strategy_id") != selected_strategy_id:
        blockers.append("proposed_4k5_execution_payload.selected_strategy_id does not match selected strategy")
    if payload.get("operation_id") != operation_id:
        blockers.append("proposed_4k5_execution_payload.operation_id does not match operation")
    if payload.get("paper_only") is not True:
        blockers.append("proposed_4k5_execution_payload.paper_only must be true")
    if payload.get("one_strategy_only") is not True:
        blockers.append("proposed_4k5_execution_payload.one_strategy_only must be true")
    allow_md = payload.get("allow_controlled_market_data_provider_call")
    allow_cq = payload.get("allow_controlled_contract_qualification_provider_call")
    if not isinstance(allow_md, bool):
        blockers.append("proposed_4k5_execution_payload.allow_controlled_market_data_provider_call must be native bool")
    if not isinstance(allow_cq, bool):
        blockers.append("proposed_4k5_execution_payload.allow_controlled_contract_qualification_provider_call must be native bool")
    if allow_md is False and allow_cq is False:
        blockers.append("at least one controlled provider call permission must be true")
    for key in PROPOSED_FALSE_FLAGS:
        if payload.get(key) is not False:
            blockers.append(f"proposed_4k5_execution_payload.{key} must be strict native boolean false")
    for key in STALE_4J_KEYS:
        if key in payload:
            blockers.append(f"proposed_4k5_execution_payload contains stale Stage 4J key {key}")

    md_raw = payload.get("market_data_provider_payloads", [])
    cq_raw = payload.get("contract_qualification_provider_payloads", [])
    if md_raw is None:
        md_raw = []
    if cq_raw is None:
        cq_raw = []
    if not isinstance(md_raw, list):
        blockers.append("market_data_provider_payloads must be a list")
        md_raw = []
    if not isinstance(cq_raw, list):
        blockers.append("contract_qualification_provider_payloads must be a list")
        cq_raw = []
    if allow_md is True and len(md_raw or []) == 0:
        blockers.append("market data provider permission is true but payload list is empty")
    if allow_cq is True and len(cq_raw or []) == 0:
        blockers.append("contract qualification provider permission is true but payload list is empty")
    if allow_md is False and len(md_raw or []) > 0:
        blockers.append("market data provider payloads present while permission is false")
    if allow_cq is False and len(cq_raw or []) > 0:
        blockers.append("contract qualification provider payloads present while permission is false")
    md_payloads, md_blockers = _sanitize_payload_list(md_raw or [], "market_data", selected_strategy_id, operation_id)
    cq_payloads, cq_blockers = _sanitize_payload_list(cq_raw or [], "contract_qualification", selected_strategy_id, operation_id)
    blockers.extend(md_blockers + cq_blockers)
    for key, value in {
        "selected_strategy_matches": payload.get("selected_strategy_id") == selected_strategy_id,
        "operation_id_matches": payload.get("operation_id") == operation_id,
        "paper_only": payload.get("paper_only") is True,
        "one_strategy_only": payload.get("one_strategy_only") is True,
        "allow_controlled_market_data_provider_call": allow_md,
        "allow_controlled_contract_qualification_provider_call": allow_cq,
        "market_data_provider_payload_count": len(md_payloads),
        "contract_qualification_provider_payload_count": len(cq_payloads),
        "stale_4j_keys_absent": not any(key in payload for key in STALE_4J_KEYS),
    }.items():
        checks[key] = value
    return checks, {"market_data_provider_payloads": md_payloads, "contract_qualification_provider_payloads": cq_payloads}, _dedupe(blockers)


def _sanitize_payload_list(raw_items: list[Any], payload_name: str, selected_strategy_id: str | None, operation_id: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    payloads: list[dict[str, Any]] = []
    blockers: list[str] = []
    for item in raw_items or []:
        if not isinstance(item, dict):
            blockers.append(f"{payload_name} provider payload contains a non-dict item")
            continue
        if not _is_json_native(item):
            blockers.append(f"{payload_name} provider payload must be JSON-safe")
        if item.get("selected_strategy_id") != selected_strategy_id:
            blockers.append(f"{payload_name} provider payload selected_strategy_id does not match")
        if item.get("operation_id") != operation_id:
            blockers.append(f"{payload_name} provider payload operation_id does not match")
        for key in PAYLOAD_FALSE_FLAGS:
            if item.get(key) is not False:
                blockers.append(f"{payload_name} provider payload {key} must be strict native boolean false")
        payloads.append(_json_safe(dict(item)))
    return payloads, blockers


def _provider_availability_checks(payload: dict[str, Any], payloads: dict[str, list[dict[str, Any]]], md_provider: Any, cq_provider: Any) -> tuple[dict[str, Any], list[str]]:
    allow_md = payload.get("allow_controlled_market_data_provider_call") is True
    allow_cq = payload.get("allow_controlled_contract_qualification_provider_call") is True
    md_method = _provider_method(md_provider, "request_controlled_market_data")
    cq_method = _provider_method(cq_provider, "qualify_controlled_contracts")
    blockers: list[str] = []
    if allow_md and len(payloads.get("market_data_provider_payloads", []) or []) > 0 and md_method is None:
        blockers.append("controlled market data provider is missing required request_controlled_market_data method")
    if allow_cq and len(payloads.get("contract_qualification_provider_payloads", []) or []) > 0 and cq_method is None:
        blockers.append("controlled contract qualification provider is missing required qualify_controlled_contracts method")
    return {
        "controlled_market_data_provider_required": allow_md,
        "controlled_market_data_provider_available": md_method is not None,
        "controlled_market_data_provider_method": "request_controlled_market_data",
        "controlled_contract_qualification_provider_required": allow_cq,
        "controlled_contract_qualification_provider_available": cq_method is not None,
        "controlled_contract_qualification_provider_method": "qualify_controlled_contracts",
    }, blockers


def _execute_provider_category(
    *,
    provider_type: str,
    method_name: str,
    provider: Any,
    payloads: list[dict[str, Any]],
    selected_strategy_id: str | None,
    operation_id: str | None,
    trace: list[dict[str, Any]],
    applied: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    counts: dict[str, Any],
    stop_state: dict[str, bool],
) -> None:
    method = _provider_method(provider, method_name)
    for index, payload in enumerate(payloads or [], start=1):
        payload_id = _payload_id(payload, index, provider_type)
        if stop_state.get("stop") is True or method is None:
            reason = "prior provider failure blocked subsequent provider execution" if stop_state.get("stop") is True else "required provider method unavailable"
            trace.append(_trace_item(len(trace) + 1, provider_type, method_name, payload_id, selected_strategy_id, operation_id, payload, False, False, {}, None, reason))
            skipped.append(_operation_item(provider_type, method_name, payload_id, selected_strategy_id, operation_id, payload, "skipped", skipped_reason=reason))
            counts["skipped"] += 1
            continue
        counts["attempted"] += 1
        try:
            raw_result = method(dict(payload))
        except Exception as exc:  # noqa: BLE001 - provider failure is report data.
            reason = _flat_exception(exc)
            trace.append(_trace_item(len(trace) + 1, provider_type, method_name, payload_id, selected_strategy_id, operation_id, payload, True, False, {}, reason, None))
            failed.append(_operation_item(provider_type, method_name, payload_id, selected_strategy_id, operation_id, payload, "failed", failure_reason=reason))
            counts["called"] = True
            counts["failed"] += 1
            stop_state["stop"] = True
            continue
        if raw_result is None:
            result = {}
            reason = "provider returned None"
            success = False
        elif not isinstance(raw_result, dict):
            result = {}
            reason = "provider returned non-dict result"
            success = False
        else:
            result_blockers = _provider_single_result_blockers(raw_result)
            result = _json_safe(dict(raw_result))
            reason = "; ".join(result_blockers) if result_blockers else None
            success = reason is None
        trace.append(_trace_item(len(trace) + 1, provider_type, method_name, payload_id, selected_strategy_id, operation_id, payload, True, success, result, reason, None))
        counts["called"] = True
        if success:
            applied.append(_operation_item(provider_type, method_name, payload_id, selected_strategy_id, operation_id, payload, "applied"))
            counts["succeeded"] += 1
        else:
            failed.append(_operation_item(provider_type, method_name, payload_id, selected_strategy_id, operation_id, payload, "failed", failure_reason=reason))
            counts["failed"] += 1
            stop_state["stop"] = True


def _mark_permission_skips(
    payloads: dict[str, list[dict[str, Any]]],
    proposed_payload: dict[str, Any],
    selected_strategy_id: str | None,
    operation_id: str | None,
    trace: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    md_counts: dict[str, Any],
    cq_counts: dict[str, Any],
    reason: str,
) -> None:
    for provider_type, method_name, key, counts in (
        ("market_data", "request_controlled_market_data", "market_data_provider_payloads", md_counts),
        ("contract_qualification", "qualify_controlled_contracts", "contract_qualification_provider_payloads", cq_counts),
    ):
        allow_key = "allow_controlled_market_data_provider_call" if provider_type == "market_data" else "allow_controlled_contract_qualification_provider_call"
        if proposed_payload.get(allow_key) is not True:
            continue
        for index, payload in enumerate(payloads.get(key, []) or [], start=1):
            payload_id = _payload_id(payload, index, provider_type)
            trace.append(_trace_item(len(trace) + 1, provider_type, method_name, payload_id, selected_strategy_id, operation_id, payload, False, False, {}, None, reason))
            skipped.append(_operation_item(provider_type, method_name, payload_id, selected_strategy_id, operation_id, payload, "skipped", skipped_reason=reason))
            counts["skipped"] += 1


def _provider_single_result_blockers(result: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not _is_json_native(result):
        blockers.append("provider result must be JSON-safe")
    for key in RESULT_UNSAFE_TRUE_FLAGS:
        if result.get(key) is True:
            blockers.append(f"provider result {key} must not be true")
    for key in RESULT_DISABLED_FLAGS:
        if result.get(key) == "False":
            blockers.append(f"provider result {key} must be native bool false, not string")
    return blockers


def _provider_result_blockers(trace: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for item in trace:
        if not isinstance(item, dict):
            blockers.append("provider_call_trace contains malformed item")
            continue
        if item.get("direct_ib_call_made") is not False:
            blockers.append("provider_call_trace direct_ib_call_made must remain false")
        for key in ("failure_reason", "skipped_reason"):
            value = item.get(key)
            if value is not None and (_has_raw_newline(value) or _has_memory_repr(value)):
                blockers.append(f"provider_call_trace {key} must be flat JSON-safe string")
    return _dedupe(blockers)


def _boundary_checks(report: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    source = _mapping(report.get("boundary_checks"))
    checks = {
        "no_direct_ib_call": source.get("no_direct_ib_call") is True,
        "no_direct_reqMktData": source.get("no_reqMktData") is True,
        "no_direct_qualifyContracts": source.get("no_qualifyContracts") is True,
        "no_direct_reqContractDetails": source.get("no_reqContractDetails") is True,
        "no_strategy_scan": source.get("no_strategy_scan") is True,
        "no_intents_created": source.get("no_intents_created") is True,
        "no_tickets_created": source.get("no_tickets_created") is True,
        "no_orders_submitted": source.get("no_orders_submitted") is True,
        "no_broker_submission": source.get("no_broker_submission") is True,
        "no_state_written": source.get("no_state_written") is True,
        "no_ledger_written": source.get("no_ledger_written") is True,
        "no_live_trading": source.get("no_live_trading") is True,
        "no_all_strategy_enablement": source.get("no_all_strategy_enablement") is True,
        "no_scheduler_registration": source.get("no_scheduler_registration") is True,
        "no_lifecycle_execution": source.get("no_lifecycle_execution") is True,
        "controlled_market_data_provider_call_only_if_allowed": True,
        "controlled_contract_qualification_provider_call_only_if_allowed": True,
    }
    blockers = [f"Stage 4K-4 boundary check {key} must be strict native boolean true" for key in STAGE4K4_BOUNDARY_FLAGS if source.get(key) is not True]
    return checks, blockers


def _safety_checks(report: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    source = _mapping(report.get("safety_checks"))
    checks = {
        "no_live_trading": source.get("no_live_trading") is True,
        "no_all_strategy_enablement": source.get("no_all_strategy_enablement") is True,
        "no_broker_submission_enabled": source.get("no_broker_submission_enabled") is True,
        "no_direct_market_data": source.get("no_direct_market_data") is True,
        "no_direct_contract_qualification": source.get("no_direct_contract_qualification") is True,
        "no_order_submission": source.get("no_order_submission") is True,
        "no_intent_creation": source.get("no_intent_creation") is True,
        "no_ticket_creation": source.get("no_ticket_creation") is True,
        "no_state_write": source.get("no_state_write") is True,
        "no_ledger_write": source.get("no_ledger_write") is True,
    }
    blockers = [f"Stage 4K-4 safety check {key} must be strict native boolean true" for key in SAFETY_FLAGS if source.get(key) is not True]
    if checks["no_direct_market_data"] is not True:
        blockers.append("Stage 4K-4 safety check no direct market data must be true")
    if checks["no_direct_contract_qualification"] is not True:
        blockers.append("Stage 4K-4 safety check no direct contract qualification must be true")
    return checks, blockers


def _activation_snapshot_group_checks(
    *,
    scheduler_activation_snapshot: Any,
    lifecycle_activation_snapshot: Any,
    activation_snapshot: Any,
    selected_strategy_id: str | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    warnings: list[str] = []
    for prefix, snapshot, record_key, list_key, execution_key in (
        ("scheduler_activation", scheduler_activation_snapshot, "scheduler_activation_record", "scheduler_activations", "strategy_scan_execution_enabled"),
        ("lifecycle_activation", lifecycle_activation_snapshot, "lifecycle_activation_record", "lifecycle_activations", "lifecycle_transition_execution_enabled"),
        ("activation", activation_snapshot, "activation_record", "activations", None),
    ):
        current_checks, current_blockers, current_warnings = _activation_snapshot_checks(snapshot, selected_strategy_id, prefix, record_key, list_key, execution_key)
        for key, value in current_checks.items():
            checks[key] = value
        blockers.extend(current_blockers)
        warnings.extend(current_warnings)
    return checks, blockers, warnings


def _activation_snapshot_checks(snapshot: Any, selected_strategy_id: str | None, prefix: str, record_key: str, list_key: str, execution_key: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append(f"{prefix} snapshot missing; verify activation artifact before Stage 4K-6")
        return {f"{prefix}_snapshot_present": False, f"{prefix}_snapshot_matches": True}, blockers, warnings
    for record in _explicit_records(data, record_key, list_key):
        if not isinstance(record, dict):
            warnings.append(f"malformed {prefix} snapshot entry ignored")
            continue
        if record.get("selected_strategy_id") not in (None, selected_strategy_id):
            blockers.append(f"{prefix} snapshot selected_strategy_id does not match")
        for key in ("live_trading_enabled", "all_strategies_enabled", "broker_submission_enabled"):
            if record.get(key) is True:
                blockers.append(f"{prefix} snapshot {key} contradicts Stage 4K-5 safety")
        if execution_key and record.get(execution_key) is True:
            blockers.append(f"{prefix} snapshot {execution_key} contradicts Stage 4K-5 safety")
    return {f"{prefix}_snapshot_present": True, f"{prefix}_snapshot_matches": not blockers}, blockers, warnings


def _state_checks(snapshot: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    active_intents_count = _safe_int(data.get("active_intents_count"))
    active_intents_safe = data.get("active_intents_safe_for_enablement") is True
    unresolved_count = _safe_int(_first_present(data.get("unresolved_needs_reconciliation_count"), data.get("needs_reconciliation_count"), default=0))
    active_halt = bool(data.get("active_halt")) if present else False
    clean = not active_halt and unresolved_count == 0 and (active_intents_count == 0 or active_intents_safe)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("state snapshot missing; verify halt, reconciliation, intents, and positions before Stage 4K-6")
    if active_halt:
        blockers.append("active halt is present")
    if unresolved_count > 0:
        blockers.append("unresolved NEEDS_RECONCILIATION is present")
    if active_intents_count > 0 and not active_intents_safe:
        blockers.append("unsafe active intents are present")
    if active_intents_count > 0 and active_intents_safe:
        warnings.append("active intents present but explicitly marked safe for enablement")
    return {
        "state_snapshot_present": present,
        "active_halt": active_halt,
        "unresolved_needs_reconciliation_count": unresolved_count,
        "active_intents_count": active_intents_count,
        "active_intents_safe_for_enablement": active_intents_safe,
        "open_positions_count": _safe_int(data.get("open_positions_count")),
        "state_snapshot_clean": clean,
    }, blockers, warnings


def _risk_checks(snapshot: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    checks = {
        "risk_snapshot_present": present,
        "kill_switch_available": data.get("kill_switch_available") is True if present else None,
        "hard_halt_available": data.get("hard_halt_available") is True if present else None,
        "daily_loss_limit_available": data.get("daily_loss_limit_available") is True if present else None,
        "max_position_limit_available": data.get("max_position_limit_available") is not False if present else None,
        "risk_bypass_enabled": data.get("risk_bypass_enabled") is True if present else False,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("risk snapshot missing; verify kill switch, hard halt, and daily loss controls before Stage 4K-6")
        return checks, blockers, warnings
    for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
        if checks[key] is not True:
            blockers.append(f"risk snapshot {key} must be true")
    if data.get("max_position_limit_available") is False:
        blockers.append("risk snapshot max_position_limit_available must not be false when supplied")
    if checks["risk_bypass_enabled"] is True:
        blockers.append("risk bypass is enabled")
    return checks, blockers, warnings


def _scheduler_checks(snapshot: Any, selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    jobs = _as_list(data.get("jobs")) + _as_list(data.get("scheduled_jobs"))
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("scheduler snapshot missing; verify scheduler state before Stage 4K-6")
    broad_enabled = any(data.get(key) is True for key in ("all_strategy_scheduler_enabled", "all_strategies_enabled", "broad_scheduler_enabled"))
    scan_enabled = data.get("strategy_scan_execution_enabled") is True
    selected_job_matches = True
    for job in jobs:
        if not isinstance(job, dict):
            continue
        job_strategy = _first_present(job.get("selected_strategy_id"), job.get("strategy_id"), default=None)
        if job_strategy == selected_strategy_id:
            selected_job_matches = selected_job_matches and _scheduler_job_safe(job, selected_strategy_id)
        elif job.get("active") is True or job.get("enabled") is True:
            blockers.append("scheduler snapshot contains active job for another strategy")
    if broad_enabled:
        blockers.append("broad/all-strategy scheduler automation enabled")
    if scan_enabled:
        blockers.append("scheduler strategy scan execution is enabled")
    if jobs and not selected_job_matches:
        blockers.append("scheduler selected strategy job is not within Stage 4K-5 safety constraints")
    return {
        "scheduler_snapshot_present": present,
        "all_strategy_scheduler_enabled": broad_enabled,
        "selected_strategy_job_matches": selected_job_matches,
        "strategy_scan_execution_enabled": scan_enabled,
    }, blockers, warnings


def _scheduler_job_safe(job: dict[str, Any], selected_strategy_id: str | None) -> bool:
    return (
        _first_present(job.get("selected_strategy_id"), job.get("strategy_id"), default=None) == selected_strategy_id
        and job.get("broker_submission_enabled") is False
        and job.get("live_trading_enabled") is False
        and job.get("all_strategies_enabled") is False
        and job.get("strategy_scan_execution_enabled") is False
    )


def _lifecycle_checks(snapshot: Any, selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; verify lifecycle state before Stage 4K-6")
    broad_enabled = any(data.get(key) is True for key in ("all_strategy_lifecycle_enabled", "all_strategies_enabled", "broad_lifecycle_enabled"))
    transition_enabled = data.get("lifecycle_transition_execution_enabled") is True
    lifecycle_enabled = data.get("lifecycle_automation_enabled") is True
    matches = True
    if present and lifecycle_enabled:
        matches = (
            data.get("selected_strategy_id") == selected_strategy_id
            and data.get("broker_submission_enabled") is False
            and data.get("live_trading_enabled") is False
            and data.get("all_strategies_enabled") is False
            and transition_enabled is False
            and broad_enabled is False
        )
        if not matches:
            blockers.append("lifecycle snapshot automation does not match selected strategy safety constraints")
    if broad_enabled:
        blockers.append("broad/all-strategy lifecycle automation enabled")
    if transition_enabled:
        blockers.append("lifecycle transition execution is enabled")
    return {
        "lifecycle_snapshot_present": present,
        "lifecycle_already_enabled": lifecycle_enabled,
        "lifecycle_matches_selected_strategy": matches,
        "lifecycle_transition_execution_enabled": transition_enabled,
    }, blockers, warnings


def _paper_broker_checks(snapshot: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    mode = data.get("mode")
    paper_trading = data.get("paper_trading")
    ibkr_port = data.get("ibkr_port")
    live_enabled = data.get("live_trading_enabled") is True
    broker_enabled = data.get("broker_submission_enabled") is True
    valid = True
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("paper broker snapshot missing; verify PAPER broker config before Stage 4K-6")
    if present and mode is not None and str(mode).upper() != "PAPER":
        valid = False
        blockers.append("paper broker snapshot mode must be PAPER")
    if present and paper_trading is False:
        valid = False
        blockers.append("paper broker snapshot paper_trading must not be false")
    if present and ibkr_port is not None and ibkr_port not in PAPER_IBKR_PORTS:
        valid = False
        blockers.append("paper broker snapshot ibkr_port must use the project PAPER port")
    if live_enabled:
        valid = False
        blockers.append("paper broker snapshot live_trading_enabled must be false")
    if broker_enabled:
        valid = False
        blockers.append("paper broker snapshot broker_submission_enabled must remain false")
    return {
        "paper_broker_snapshot_present": present,
        "mode": mode,
        "paper_trading": paper_trading,
        "ibkr_port": ibkr_port,
        "paper_config_valid": valid,
        "live_trading_enabled": live_enabled,
        "broker_submission_enabled": broker_enabled,
    }, blockers, warnings


def _market_window_checks(snapshot: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    allowed = data.get("allowed_to_schedule_paper_run") if present else None
    is_trading_day = data.get("is_trading_day") if present else None
    market_open = data.get("market_open") if present else None
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append(MARKET_WINDOW_MANUAL_WARNING)
    if allowed is False:
        blockers.append("market window snapshot explicitly blocks execution validation")
    if market_open is False:
        warnings.append("market is currently closed; controlled provider execution validation remains allowed")
    if is_trading_day is False:
        warnings.append("snapshot is not a trading day; controlled provider execution validation remains allowed")
    return {
        "market_window_snapshot_present": present,
        "allowed_to_schedule_paper_run": allowed,
        "is_trading_day": is_trading_day,
        "market_open": market_open,
        "reason": data.get("reason") if present else None,
    }, blockers, warnings


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    operation: dict[str, Any],
    stage4k4_gate_checks: dict[str, Any],
    execution_payload_checks: dict[str, Any],
    provider_availability_checks: dict[str, Any],
    market_data_execution_results: dict[str, Any],
    contract_qualification_execution_results: dict[str, Any],
    provider_call_trace: list[dict[str, Any]],
    applied_operations: list[dict[str, Any]],
    failed_operations: list[dict[str, Any]],
    skipped_operations: list[dict[str, Any]],
    boundary_checks: dict[str, bool],
    required_inputs_for_4k6: list[str],
    activation_snapshot_checks: dict[str, Any],
    state_checks: dict[str, Any],
    risk_checks: dict[str, Any],
    scheduler_checks: dict[str, Any],
    lifecycle_checks: dict[str, Any],
    paper_broker_checks: dict[str, Any],
    market_window_checks: dict[str, Any],
    safety_checks: dict[str, bool],
    ready: bool,
    blockers: list[str],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "dry_run": False,
        "stage4k5_market_data_contract_executor_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "operation": operation,
        "stage4k4_gate_checks": stage4k4_gate_checks,
        "execution_payload_checks": execution_payload_checks,
        "provider_availability_checks": provider_availability_checks,
        "market_data_execution_results": market_data_execution_results,
        "contract_qualification_execution_results": contract_qualification_execution_results,
        "provider_call_trace": provider_call_trace,
        "applied_operations": applied_operations,
        "failed_operations": failed_operations,
        "skipped_operations": skipped_operations,
        "boundary_checks": boundary_checks,
        "required_inputs_for_4k6": required_inputs_for_4k6,
        "activation_snapshot_checks": activation_snapshot_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4k6": {
            "ready_to_build_market_data_contract_acceptance": ready,
            "blockers": list(blockers if not ready else []),
            "warnings": list(warnings),
        },
        "recommendations": {"ordered_next_steps": list(ORDERED_NEXT_STEPS), "do_not_do_yet": list(DO_NOT_DO_YET)},
        "success": ready,
        "errors": list(errors),
        "warnings": list(warnings),
    }


def _market_data_results(results: list[Any], attempted: int, succeeded: int, failed: int, skipped: int, called: bool) -> dict[str, Any]:
    return {
        "attempted": attempted > 0,
        "provider_called": bool(called),
        "direct_ib_call_made": False,
        "reqMktData_called": False,
        "request_count_attempted": attempted,
        "request_count_succeeded": succeeded,
        "request_count_failed": failed,
        "request_count_skipped": skipped,
        "provider_method_name": "request_controlled_market_data",
        "provider_results": [_json_safe(item) if isinstance(item, dict) else {} for item in results],
        "success": attempted > 0 and failed == 0 and skipped == 0,
    }


def _contract_results(results: list[Any], attempted: int, succeeded: int, failed: int, skipped: int, called: bool) -> dict[str, Any]:
    return {
        "attempted": attempted > 0,
        "provider_called": bool(called),
        "direct_ib_call_made": False,
        "qualifyContracts_called": False,
        "reqContractDetails_called": False,
        "qualification_count_attempted": attempted,
        "qualification_count_succeeded": succeeded,
        "qualification_count_failed": failed,
        "qualification_count_skipped": skipped,
        "provider_method_name": "qualify_controlled_contracts",
        "provider_results": [_json_safe(item) if isinstance(item, dict) else {} for item in results],
        "success": attempted > 0 and failed == 0 and skipped == 0,
    }


def _trace_item(
    sequence_number: int,
    provider_type: str,
    provider_method: str,
    payload_id: str,
    selected_strategy_id: str | None,
    operation_id: str | None,
    input_payload: dict[str, Any],
    provider_called: bool,
    success: bool,
    result: dict[str, Any],
    failure_reason: str | None,
    skipped_reason: str | None,
) -> dict[str, Any]:
    return {
        "sequence_number": sequence_number,
        "provider_type": provider_type,
        "provider_method": provider_method,
        "payload_id": payload_id,
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "input_payload": _json_safe(dict(input_payload)),
        "provider_called": bool(provider_called),
        "direct_ib_call_made": False,
        "success": bool(success),
        "result": _json_safe(dict(result)),
        "failure_reason": _flat_text(failure_reason) if failure_reason else None,
        "skipped_reason": _flat_text(skipped_reason) if skipped_reason else None,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
        "order_submission_enabled": False,
        "state_write_enabled": False,
        "ledger_write_enabled": False,
    }


def _operation_item(
    provider_type: str,
    provider_method: str,
    payload_id: str,
    selected_strategy_id: str | None,
    operation_id: str | None,
    payload: dict[str, Any],
    status: str,
    failure_reason: str | None = None,
    skipped_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "provider_type": provider_type,
        "provider_method": provider_method,
        "payload_id": payload_id,
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "target": _sort_text(payload.get("symbol")) or payload_id,
        "status": status,
        "failure_reason": _flat_text(failure_reason) if failure_reason else None,
        "skipped_reason": _flat_text(skipped_reason) if skipped_reason else None,
    }


def _required_inputs_for_4k6() -> list[str]:
    return [
        "accepted Stage 4K-5 controlled market data and contract qualification executor report",
        "same selected_strategy_id and operation_id from Stage 4K-4",
        "JSON-safe controlled market data provider results when market data was permitted",
        "JSON-safe controlled contract qualification provider results when qualification was permitted",
        "fresh state, risk, scheduler, lifecycle, paper broker, and market window snapshots",
    ]


def _provider_method(provider: Any, method_name: str) -> Callable[[dict[str, Any]], Any] | None:
    if provider is None:
        return None
    if callable(provider):
        return provider
    method = getattr(provider, method_name, None)
    return method if callable(method) else None


def _selected_strategy_id(report: dict[str, Any]) -> str | None:
    value = _mapping(report.get("selected_strategy")).get("selected_strategy_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _operation_id(report: dict[str, Any]) -> str | None:
    value = _mapping(report.get("operation")).get("operation_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _payload_id(payload: dict[str, Any], index: int, provider_type: str) -> str:
    preferred = "request_id" if provider_type == "market_data" else "qualification_id"
    value = payload.get(preferred)
    if isinstance(value, str) and value:
        return value
    symbol = _sort_text(payload.get("symbol")) or "payload"
    prefix = "md" if provider_type == "market_data" else "cq"
    return f"{prefix}-{index:03d}-{symbol}"


def _explicit_records(data: dict[str, Any], record_key: str, list_key: str) -> list[Any]:
    records: list[Any] = []
    if record_key in data:
        records.append(data.get(record_key))
    records.extend(_as_list(data.get(list_key)))
    if any(key in data for key in ("selected_strategy_id", "live_trading_enabled", "all_strategies_enabled", "broker_submission_enabled")):
        records.append(data)
    return records


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_mapping(value: Any) -> tuple[bool, dict[str, Any]]:
    return isinstance(value, dict), value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_string_list(value: Any) -> list[str]:
    return [item.strip() for item in _as_list(value) if isinstance(item, str) and item.strip()]


def _safe_int(value: Any) -> int:
    try:
        if isinstance(value, bool) or value is None:
            return 0
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _first_present(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _sort_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ("" if value is None else str(value))


def _generated_at(now_provider: Callable[[], Any] | None) -> str:
    if now_provider is None:
        return DEFAULT_GENERATED_AT
    value = now_provider()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc).isoformat()
    if isinstance(value, str):
        return value
    return DEFAULT_GENERATED_AT


def _is_json_native(value: Any) -> bool:
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_native(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_is_json_native(item) for item in value)
    if isinstance(value, (str, int, bool)) or value is None:
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _flat_exception(exc: Exception) -> str:
    return _flat_text(f"{type(exc).__name__}: {exc}")


def _flat_text(value: Any) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"<[^<>]* at 0x[0-9A-Fa-f]+>", "<object>", text)
    text = re.sub(r"0x[0-9A-Fa-f]+", "0xADDR", text)
    return " ".join(text.split())


def _has_raw_newline(value: Any) -> bool:
    return isinstance(value, str) and ("\n" in value or "\r" in value)


def _has_memory_repr(value: Any) -> bool:
    return isinstance(value, str) and re.search(r"<[^<>]* at 0x[0-9A-Fa-f]+>", value) is not None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _default_artifact_checks() -> dict[str, bool]:
    return {
        "stage4k4_report_present": False,
        "stage4k4_report_ready": False,
        "selected_strategy_present": False,
        "operation_id_present": False,
        "proposed_4k5_execution_payload_present": False,
    }


def _default_selected_strategy() -> dict[str, Any]:
    return {"selected_strategy_id": None, "paper_only": True, "one_strategy_only": True}


def _default_operation() -> dict[str, Any]:
    return {
        "operation_id": None,
        "operation_scope": "single_strategy_controlled_market_data_contract_execution",
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }


def _default_boundary_checks() -> dict[str, bool]:
    return {
        "no_direct_ib_call": False,
        "no_direct_reqMktData": False,
        "no_direct_qualifyContracts": False,
        "no_direct_reqContractDetails": False,
        "no_strategy_scan": False,
        "no_intents_created": False,
        "no_tickets_created": False,
        "no_orders_submitted": False,
        "no_broker_submission": False,
        "no_state_written": False,
        "no_ledger_written": False,
        "no_live_trading": False,
        "no_all_strategy_enablement": False,
        "no_scheduler_registration": False,
        "no_lifecycle_execution": False,
        "controlled_market_data_provider_call_only_if_allowed": False,
        "controlled_contract_qualification_provider_call_only_if_allowed": False,
    }


def _default_state_checks() -> dict[str, Any]:
    return {
        "state_snapshot_present": False,
        "active_halt": False,
        "unresolved_needs_reconciliation_count": 0,
        "active_intents_count": 0,
        "active_intents_safe_for_enablement": False,
        "open_positions_count": 0,
        "state_snapshot_clean": True,
    }


def _default_risk_checks() -> dict[str, Any]:
    return {
        "risk_snapshot_present": False,
        "kill_switch_available": None,
        "hard_halt_available": None,
        "daily_loss_limit_available": None,
        "max_position_limit_available": None,
        "risk_bypass_enabled": False,
    }


def _default_scheduler_checks() -> dict[str, Any]:
    return {
        "scheduler_snapshot_present": False,
        "all_strategy_scheduler_enabled": False,
        "selected_strategy_job_matches": True,
        "strategy_scan_execution_enabled": False,
    }


def _default_lifecycle_checks() -> dict[str, Any]:
    return {
        "lifecycle_snapshot_present": False,
        "lifecycle_already_enabled": False,
        "lifecycle_matches_selected_strategy": True,
        "lifecycle_transition_execution_enabled": False,
    }


def _default_paper_broker_checks() -> dict[str, Any]:
    return {
        "paper_broker_snapshot_present": False,
        "mode": None,
        "paper_trading": None,
        "ibkr_port": None,
        "paper_config_valid": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }


def _default_market_window_checks() -> dict[str, Any]:
    return {
        "market_window_snapshot_present": False,
        "allowed_to_schedule_paper_run": None,
        "is_trading_day": None,
        "market_open": None,
        "reason": None,
    }


def _default_safety_checks() -> dict[str, bool]:
    return {
        "no_live_trading": False,
        "no_all_strategy_enablement": False,
        "no_broker_submission_enabled": False,
        "no_direct_market_data": False,
        "no_direct_contract_qualification": False,
        "no_order_submission": False,
        "no_intent_creation": False,
        "no_ticket_creation": False,
        "no_state_write": False,
        "no_ledger_write": False,
    }
