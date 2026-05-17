"""Pure Stage 4K-4 market data and contract qualification execution-gate report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding"
)
PAPER_IBKR_PORTS = {4004}
REQUIRED_OPERATOR_ACKNOWLEDGEMENTS = [
    "ACK_4K4_MARKET_DATA_AND_CONTRACT_GATE_ONLY",
    "ACK_NO_ORDER_SUBMISSION",
    "ACK_NO_BROKER_SUBMISSION",
    "ACK_NO_STATE_OR_LEDGER_WRITES",
    "ACK_LIVE_TRADING_DISABLED",
    "ACK_SINGLE_STRATEGY_ONLY",
]
TRACE_FALSE_FLAGS = (
    "would_execute_now",
    "would_fetch_market_data_now",
    "would_qualify_contracts_now",
    "would_call_strategy_now",
    "would_create_intents_now",
    "would_create_tickets_now",
    "would_submit_orders_now",
    "would_write_state_now",
    "would_write_ledger_now",
    "live_trading_enabled",
    "broker_submission_enabled",
)
PAYLOAD_FALSE_FLAGS = (
    "allow_live_trading",
    "allow_broker_submission",
    "allow_order_submission",
    "allow_state_write",
    "allow_ledger_write",
)
BOUNDARY_FLAGS = (
    "no_market_data_fetched",
    "no_contracts_qualified",
    "no_provider_called",
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
SAFETY_SOURCE_FLAGS = (
    "no_live_trading",
    "no_all_strategy_enablement",
    "no_broker_submission_enabled",
    "no_order_submission",
    "no_intent_creation",
    "no_ticket_creation",
    "no_state_write",
    "no_ledger_write",
)
ORDERED_NEXT_STEPS = [
    "Build Stage 4K-5 controlled market data and contract qualification executor.",
    "Keep live trading disabled.",
    "Keep all other strategies disabled.",
    "Keep broker submission separately gated until the explicit broker-submission phase.",
    "Keep intents, tickets, state writes, and ledger writes separately gated until their explicit phases.",
    "Use only injected controlled provider abstractions in 4K-5.",
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
BLOCKED_ACTIONS = [
    "direct_reqMktData",
    "direct_qualifyContracts",
    "direct_reqContractDetails",
    "strategy_scan",
    "intent_creation",
    "ticket_creation",
    "order_submission",
    "broker_submission",
    "state_write",
    "ledger_write",
    "live_trading",
    "all_strategy_automation",
]


def build_stage4k4_market_data_contract_execution_gate_report(
    *,
    stage4k3_dry_run_report: dict | None,
    operator_acknowledgements: list[str] | None = None,
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
    """Build a read-only execution gate report from an accepted Stage 4K-3 dry run."""

    acknowledgements = [] if operator_acknowledgements is None else operator_acknowledgements
    try:
        return _json_safe(
            _build_report(
                stage4k3_dry_run_report=stage4k3_dry_run_report,
                operator_acknowledgements=acknowledgements,
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
        message = f"unexpected Stage 4K-4 execution-gate failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_default_selected_strategy(),
                operation=_default_operation(),
                stage4k3_dry_run_checks={},
                operator_acknowledgement_checks=_acknowledgement_checks(acknowledgements),
                market_data_execution_permissions=_market_data_permissions([], False),
                contract_qualification_execution_permissions=_contract_permissions([], False),
                execution_gate=_execution_gate(None, None, False, [message], [], False),
                proposed_4k5_execution_payload=_proposed_payload(None, None, [], [], False, False, _generated_at(now_provider)),
                boundary_checks=_default_boundary_checks(),
                required_inputs_for_4k5=_required_inputs_for_4k5(),
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
    stage4k3_dry_run_report: Any,
    operator_acknowledgements: Any,
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
    data = stage4k3_dry_run_report if isinstance(stage4k3_dry_run_report, dict) else {}
    blockers: list[str] = []
    warnings: list[str] = []
    errors = _as_string_list(data.get("errors"))

    if stage4k3_dry_run_report is None:
        blockers.append("Stage 4K-3 dry-run report is missing")
    elif not isinstance(stage4k3_dry_run_report, dict):
        blockers.append("Stage 4K-3 dry-run report must be a dict")
        errors.append("Stage 4K-3 dry-run report must be a dict")

    selected_strategy_id = _selected_strategy_id(data)
    operation_id = _operation_id(data)
    artifact_checks = _artifact_checks(stage4k3_dry_run_report, data, selected_strategy_id, operation_id)
    blockers.extend(_artifact_blockers(artifact_checks, data))

    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    operation, operation_blockers = _operation_checks(data, operation_id)
    trace_checks, trace_blockers = _dry_run_trace_checks(data.get("dry_run_trace"))
    payloads, payload_blockers = _provider_payload_checks(data, selected_strategy_id, operation_id)
    md_results_checks, md_result_blockers = _market_data_result_checks(data.get("market_data_dry_run_results"))
    cq_results_checks, cq_result_blockers = _contract_result_checks(data.get("contract_qualification_dry_run_results"))
    boundary_checks, boundary_blockers = _boundary_checks(data)
    safety_checks, safety_blockers = _safety_checks(data)
    ack_checks = _acknowledgement_checks(operator_acknowledgements)
    if ack_checks.get("valid_input") is not True:
        blockers.append("operator_acknowledgements must be a list")
    if ack_checks.get("complete") is not True:
        blockers.append("required operator acknowledgements are missing")
    if ack_checks.get("extra"):
        warnings.append("extra operator acknowledgements ignored")

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
        trace_blockers,
        payload_blockers,
        md_result_blockers,
        cq_result_blockers,
        boundary_blockers,
        safety_blockers,
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

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    prelim_ready = not blocker_list and not error_list
    md_payloads = payloads["market_data_provider_payloads"]
    cq_payloads = payloads["contract_qualification_provider_payloads"]
    allow_md = prelim_ready and bool(md_payloads)
    allow_cq = prelim_ready and bool(cq_payloads)
    proposed_payload = _proposed_payload(selected_strategy_id, operation_id, md_payloads, cq_payloads, allow_md, allow_cq, generated_at)
    proposed_blockers = _proposed_payload_blockers(proposed_payload)
    blocker_list = _dedupe(blocker_list + proposed_blockers)
    ready = not blocker_list and not error_list and (allow_md or allow_cq)
    if not ready:
        allow_md = False
        allow_cq = False
        proposed_payload = _proposed_payload(selected_strategy_id, operation_id, md_payloads, cq_payloads, False, False, generated_at)
        if md_payloads or cq_payloads:
            blocker_list = _dedupe(blocker_list)
        elif "at least one controlled provider payload list must be non-empty" not in blocker_list:
            blocker_list.append("at least one controlled provider payload list must be non-empty")
    if not allow_md and not allow_cq and not any("controlled provider call" in item for item in blocker_list):
        blocker_list = _dedupe(blocker_list + ["at least one controlled provider call must be permitted for Stage 4K-5"])
        ready = False

    market_data_permissions = _market_data_permissions(md_payloads, allow_md)
    contract_permissions = _contract_permissions(cq_payloads, allow_cq)
    stage4k3_checks = {
        "stage4k3_report_true": data.get("stage4k3_market_data_contract_dry_run_report") is True,
        "stage4k3_success": data.get("success") is True,
        "ready_to_build_market_data_contract_execution_gate": _mapping(data.get("readiness_for_stage4k4")).get("ready_to_build_market_data_contract_execution_gate") is True,
        "dry_run_trace_valid": not trace_blockers,
        "dry_run_provider_payloads_valid": not payload_blockers,
        "market_data_dry_run_results_clean": not md_result_blockers and md_results_checks.get("present") is True,
        "contract_qualification_dry_run_results_clean": not cq_result_blockers and cq_results_checks.get("present") is True,
    }
    execution_gate = _execution_gate(selected_strategy_id, operation_id, ready, blocker_list if not ready else [], warning_list, ready)

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
        operation=operation,
        stage4k3_dry_run_checks=stage4k3_checks,
        operator_acknowledgement_checks=ack_checks,
        market_data_execution_permissions=market_data_permissions,
        contract_qualification_execution_permissions=contract_permissions,
        execution_gate=execution_gate,
        proposed_4k5_execution_payload=proposed_payload,
        boundary_checks=boundary_checks,
        required_inputs_for_4k5=_required_inputs_for_4k5(),
        activation_snapshot_checks=activation_checks,
        state_checks=state_checks,
        risk_checks=risk_checks,
        scheduler_checks=scheduler_checks,
        lifecycle_checks=lifecycle_checks,
        paper_broker_checks=paper_broker_checks,
        market_window_checks=market_window_checks,
        safety_checks=safety_checks,
        ready=ready,
        blockers=blocker_list if not ready else [],
        warnings=warning_list,
        errors=error_list,
    )


def _artifact_checks(report: Any, data: dict[str, Any], selected_strategy_id: str | None, operation_id: str | None) -> dict[str, bool]:
    readiness = _mapping(data.get("readiness_for_stage4k4"))
    return {
        "stage4k3_report_present": isinstance(report, dict),
        "stage4k3_report_ready": (
            data.get("stage4k3_market_data_contract_dry_run_report") is True
            and readiness.get("ready_to_build_market_data_contract_execution_gate") is True
            and data.get("success") is True
        ),
        "selected_strategy_present": isinstance(selected_strategy_id, str) and bool(selected_strategy_id),
        "operation_id_present": isinstance(operation_id, str) and bool(operation_id),
        "dry_run_trace_present": isinstance(data.get("dry_run_trace"), list),
        "dry_run_provider_payloads_present": isinstance(data.get("dry_run_provider_payloads"), dict),
    }


def _artifact_blockers(checks: dict[str, bool], data: dict[str, Any]) -> list[str]:
    labels = {
        "stage4k3_report_present": "Stage 4K-3 dry-run report is missing",
        "stage4k3_report_ready": "Stage 4K-3 dry-run report is not ready for Stage 4K-4",
        "selected_strategy_present": "selected strategy is missing from Stage 4K-3 report",
        "operation_id_present": "operation_id is missing from Stage 4K-3 report",
        "dry_run_trace_present": "dry_run_trace is missing or malformed",
        "dry_run_provider_payloads_present": "dry_run_provider_payloads is missing or malformed",
    }
    blockers = [label for key, label in labels.items() if checks.get(key) is not True]
    if _as_list(data.get("errors")):
        blockers.append("Stage 4K-3 report contains errors")
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
        blockers.append("Stage 4K-3 operation must keep live trading disabled")
    if operation.get("broker_submission_enabled") is not False:
        blockers.append("Stage 4K-3 operation must keep broker submission disabled")
    return {
        "operation_id": operation_id,
        "operation_scope": operation.get("operation_scope") or "single_strategy_market_data_contract_execution_gate",
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }, blockers


def _dry_run_trace_checks(raw: Any) -> tuple[dict[str, Any], list[str]]:
    trace = raw if isinstance(raw, list) else []
    blockers: list[str] = []
    checks = {"present": isinstance(raw, list) and bool(raw), "items_are_dicts": True, "all_simulated": bool(trace)}
    if not isinstance(raw, list) or not trace:
        blockers.append("dry_run_trace must be a non-empty list")
    for item in trace:
        if not isinstance(item, dict):
            checks["items_are_dicts"] = False
            checks["all_simulated"] = False
            blockers.append("dry_run_trace contains non-dict item")
            continue
        if item.get("status") != "simulated":
            checks["all_simulated"] = False
            blockers.append("every dry_run_trace item status must be simulated")
        for key in TRACE_FALSE_FLAGS:
            if item.get(key) is not False:
                blockers.append(f"dry_run_trace item {key} must be strict native boolean false")
    return checks, _dedupe(blockers)


def _provider_payload_checks(report: dict[str, Any], selected_strategy_id: str | None, operation_id: str | None) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    parent = _mapping(report.get("dry_run_provider_payloads"))
    blockers: list[str] = []
    if not parent:
        blockers.append("dry_run_provider_payloads is missing or malformed")
    md_raw = parent.get("market_data_provider_payloads", [])
    cq_raw = parent.get("contract_qualification_provider_payloads", [])
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
    if not md_raw and not cq_raw:
        blockers.append("at least one controlled provider payload list must be non-empty")
    md_payloads, md_blockers = _sanitize_payload_list(md_raw or [], "market_data", selected_strategy_id, operation_id)
    cq_payloads, cq_blockers = _sanitize_payload_list(cq_raw or [], "contract_qualification", selected_strategy_id, operation_id)
    blockers.extend(md_blockers + cq_blockers)
    return {
        "market_data_provider_payloads": md_payloads,
        "contract_qualification_provider_payloads": cq_payloads,
    }, blockers


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


def _market_data_result_checks(raw: Any) -> tuple[dict[str, Any], list[str]]:
    data = _mapping(raw)
    blockers: list[str] = []
    if not data:
        blockers.append("market_data_dry_run_results is missing or malformed")
    for key in ("attempted", "provider_called", "direct_ib_call_made", "reqMktData_called"):
        if data.get(key) is not False:
            blockers.append(f"market_data_dry_run_results.{key} must be strict native boolean false")
    return {"present": bool(data), "clean": not blockers}, blockers


def _contract_result_checks(raw: Any) -> tuple[dict[str, Any], list[str]]:
    data = _mapping(raw)
    blockers: list[str] = []
    if not data:
        blockers.append("contract_qualification_dry_run_results is missing or malformed")
    for key in ("attempted", "provider_called", "direct_ib_call_made", "qualifyContracts_called", "reqContractDetails_called"):
        if data.get(key) is not False:
            blockers.append(f"contract_qualification_dry_run_results.{key} must be strict native boolean false")
    return {"present": bool(data), "clean": not blockers}, blockers


def _boundary_checks(report: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    source = _mapping(report.get("boundary_checks"))
    checks = {
        "no_market_data_fetched_now": source.get("no_market_data_fetched") is True,
        "no_contracts_qualified_now": source.get("no_contracts_qualified") is True,
        "no_provider_called_now": source.get("no_provider_called") is True,
        "no_direct_ib_call": source.get("no_direct_ib_call") is True,
        "no_reqMktData": source.get("no_reqMktData") is True,
        "no_qualifyContracts": source.get("no_qualifyContracts") is True,
        "no_reqContractDetails": source.get("no_reqContractDetails") is True,
        "no_strategy_scan": source.get("no_strategy_scan") is True,
        "no_intents_created": source.get("no_intents_created") is True,
        "no_tickets_created": source.get("no_tickets_created") is True,
        "no_orders_submitted": source.get("no_orders_submitted") is True,
        "no_state_written": source.get("no_state_written") is True,
        "no_ledger_written": source.get("no_ledger_written") is True,
        "no_broker_submission": source.get("no_broker_submission") is True,
        "no_live_trading": source.get("no_live_trading") is True,
        "no_all_strategy_enablement": source.get("no_all_strategy_enablement") is True,
        "no_scheduler_registration": source.get("no_scheduler_registration") is True,
        "no_lifecycle_execution": source.get("no_lifecycle_execution") is True,
    }
    blockers = [f"Stage 4K-3 boundary check {key} must be strict native boolean true" for key in BOUNDARY_FLAGS if source.get(key) is not True]
    return checks, blockers


def _safety_checks(report: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    source = _mapping(report.get("safety_checks"))
    checks = {
        "no_live_trading": source.get("no_live_trading") is True,
        "no_all_strategy_enablement": source.get("no_all_strategy_enablement") is True,
        "no_broker_submission_enabled": source.get("no_broker_submission_enabled") is True,
        "no_direct_market_data": source.get("no_direct_market_data") is True or source.get("no_market_data") is True,
        "no_direct_contract_qualification": source.get("no_direct_contract_qualification") is True or source.get("no_contract_qualification") is True,
        "no_order_submission": source.get("no_order_submission") is True,
        "no_intent_creation": source.get("no_intent_creation") is True,
        "no_ticket_creation": source.get("no_ticket_creation") is True,
        "no_state_write": source.get("no_state_write") is True,
        "no_ledger_write": source.get("no_ledger_write") is True,
    }
    blockers = [f"Stage 4K-3 safety check {key} must be strict native boolean true" for key in SAFETY_SOURCE_FLAGS if source.get(key) is not True]
    if checks["no_direct_market_data"] is not True:
        blockers.append("Stage 4K-3 safety check no direct market data must be true")
    if checks["no_direct_contract_qualification"] is not True:
        blockers.append("Stage 4K-3 safety check no direct contract qualification must be true")
    return checks, blockers


def _acknowledgement_checks(values: Any) -> dict[str, Any]:
    if not isinstance(values, list):
        return {
            "required": list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS),
            "provided": [],
            "missing": list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS),
            "extra": [],
            "valid_input": False,
            "complete": False,
            "exact_match_required": True,
        }
    provided = [item for item in values if isinstance(item, str)]
    missing = [item for item in REQUIRED_OPERATOR_ACKNOWLEDGEMENTS if item not in provided]
    extra = [item for item in provided if item not in REQUIRED_OPERATOR_ACKNOWLEDGEMENTS]
    return {
        "required": list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS),
        "provided": provided,
        "missing": missing,
        "extra": extra,
        "valid_input": True,
        "complete": not missing,
        "exact_match_required": True,
    }


def _market_data_permissions(payloads: list[dict[str, Any]], allow: bool) -> dict[str, Any]:
    return {
        "may_call_controlled_market_data_provider_in_4k5": bool(allow),
        "may_call_reqMktData_directly": False,
        "may_call_yfinance": False,
        "may_call_requests": False,
        "may_call_urllib": False,
        "selected_strategy_only": True,
        "paper_only": True,
        "provider_method_name": "request_controlled_market_data",
        "provider_payload_count": len(payloads),
        "provider_payload_ids": [_payload_id(payload, "request_id", index, "md") for index, payload in enumerate(payloads, start=1)],
        "permission_scope": "controlled_market_data_provider_only",
    }


def _contract_permissions(payloads: list[dict[str, Any]], allow: bool) -> dict[str, Any]:
    return {
        "may_call_controlled_contract_qualification_provider_in_4k5": bool(allow),
        "may_call_qualifyContracts_directly": False,
        "may_call_reqContractDetails_directly": False,
        "selected_strategy_only": True,
        "paper_only": True,
        "provider_method_name": "qualify_controlled_contracts",
        "provider_payload_count": len(payloads),
        "provider_payload_ids": [_payload_id(payload, "qualification_id", index, "cq") for index, payload in enumerate(payloads, start=1)],
        "permission_scope": "controlled_contract_qualification_provider_only",
    }


def _proposed_payload(
    selected_strategy_id: str | None,
    operation_id: str | None,
    md_payloads: list[dict[str, Any]],
    cq_payloads: list[dict[str, Any]],
    allow_md: bool,
    allow_cq: bool,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "source_stage": "4K-4",
        "permission_source_stage": "4K-3",
        "execution_scope": "single_strategy_controlled_market_data_contract_execution",
        "paper_only": True,
        "one_strategy_only": True,
        "allow_controlled_market_data_provider_call": bool(allow_md),
        "allow_controlled_contract_qualification_provider_call": bool(allow_cq),
        "allow_direct_reqMktData": False,
        "allow_direct_qualifyContracts": False,
        "allow_direct_reqContractDetails": False,
        "allow_strategy_scan": False,
        "allow_intent_creation": False,
        "allow_ticket_creation": False,
        "allow_order_submission": False,
        "allow_broker_submission": False,
        "allow_state_write": False,
        "allow_ledger_write": False,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "market_data_provider_payloads": [_json_safe(dict(payload)) for payload in md_payloads],
        "contract_qualification_provider_payloads": [_json_safe(dict(payload)) for payload in cq_payloads],
        "generated_at": generated_at,
    }


def _proposed_payload_blockers(payload: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not isinstance(payload, dict) or not _is_json_native(payload):
        blockers.append("proposed_4k5_execution_payload must be a native JSON-safe dict")
        return blockers
    for key in (
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
    ):
        if payload.get(key) is not False:
            blockers.append(f"proposed_4k5_execution_payload.{key} must be strict native boolean false")
    return blockers


def _execution_gate(selected_strategy_id: str | None, operation_id: str | None, accepted: bool, blockers: list[str], warnings: list[str], ready: bool) -> dict[str, Any]:
    return {
        "ready_for_4k5_controlled_provider_execution": ready,
        "gate_status": "passed" if ready else "blocked",
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "allowed_next_phase": "stage4k5_market_data_contract_execution",
        "blocked_actions": list(BLOCKED_ACTIONS),
        "reasons": ["accepted Stage 4K-3 dry-run report permits controlled provider gate evaluation"] if accepted else [],
        "blockers": list(blockers),
        "warnings": list(warnings),
    }


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
        warnings.append(f"{prefix} snapshot missing; verify activation artifact before Stage 4K-5")
        return {f"{prefix}_snapshot_present": False, f"{prefix}_snapshot_matches": True}, blockers, warnings
    for record in _explicit_records(data, record_key, list_key):
        if not isinstance(record, dict):
            warnings.append(f"malformed {prefix} snapshot entry ignored")
            continue
        if record.get("selected_strategy_id") not in (None, selected_strategy_id):
            blockers.append(f"{prefix} snapshot selected_strategy_id does not match")
        for key in ("live_trading_enabled", "all_strategies_enabled", "broker_submission_enabled"):
            if record.get(key) is True:
                blockers.append(f"{prefix} snapshot {key} contradicts Stage 4K-4 safety")
        if execution_key and record.get(execution_key) is True:
            blockers.append(f"{prefix} snapshot {execution_key} contradicts Stage 4K-4 safety")
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
        warnings.append("state snapshot missing; verify halt, reconciliation, intents, and positions before Stage 4K-5")
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
        warnings.append("risk snapshot missing; verify kill switch, hard halt, and daily loss controls before Stage 4K-5")
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
        warnings.append("scheduler snapshot missing; verify scheduler state before Stage 4K-5")
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
        blockers.append("scheduler selected strategy job is not within Stage 4K-4 safety constraints")
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
        warnings.append("lifecycle snapshot missing; verify lifecycle state before Stage 4K-5")
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
        warnings.append("paper broker snapshot missing; verify PAPER broker config before Stage 4K-5")
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
        blockers.append("market window snapshot explicitly blocks gate validation")
    if market_open is False:
        warnings.append("market is currently closed; execution-gate validation remains report-only")
    if is_trading_day is False:
        warnings.append("snapshot is not a trading day; execution-gate validation remains report-only")
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
    stage4k3_dry_run_checks: dict[str, Any],
    operator_acknowledgement_checks: dict[str, Any],
    market_data_execution_permissions: dict[str, Any],
    contract_qualification_execution_permissions: dict[str, Any],
    execution_gate: dict[str, Any],
    proposed_4k5_execution_payload: dict[str, Any],
    boundary_checks: dict[str, bool],
    required_inputs_for_4k5: list[str],
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
        "dry_run": True,
        "stage4k4_market_data_contract_execution_gate_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "operation": operation,
        "stage4k3_dry_run_checks": stage4k3_dry_run_checks,
        "operator_acknowledgement_checks": operator_acknowledgement_checks,
        "market_data_execution_permissions": market_data_execution_permissions,
        "contract_qualification_execution_permissions": contract_qualification_execution_permissions,
        "execution_gate": execution_gate,
        "proposed_4k5_execution_payload": proposed_4k5_execution_payload,
        "boundary_checks": boundary_checks,
        "required_inputs_for_4k5": required_inputs_for_4k5,
        "activation_snapshot_checks": activation_snapshot_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4k5": {
            "ready_to_execute_controlled_market_data_contract_providers": ready,
            "blockers": list(blockers if not ready else []),
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


def _required_inputs_for_4k5() -> list[str]:
    return [
        "accepted Stage 4K-4 market data and contract qualification execution-gate report",
        "accepted Stage 4K-3 market data and contract qualification dry-run report",
        "same selected_strategy_id and operation_id from Stage 4K-3",
        "native JSON-safe controlled provider payloads for the selected strategy only",
        "fresh state, risk, scheduler, lifecycle, paper broker, and market window snapshots",
        "exact operator acknowledgements",
    ]


def _selected_strategy_id(report: dict[str, Any]) -> str | None:
    value = _mapping(report.get("selected_strategy")).get("selected_strategy_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _operation_id(report: dict[str, Any]) -> str | None:
    value = _mapping(report.get("operation")).get("operation_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _payload_id(payload: dict[str, Any], preferred_key: str, index: int, prefix: str) -> str:
    value = payload.get(preferred_key)
    if isinstance(value, str) and value:
        return value
    symbol = _sort_text(payload.get("symbol")) or "payload"
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
        "stage4k3_report_present": False,
        "stage4k3_report_ready": False,
        "selected_strategy_present": False,
        "operation_id_present": False,
        "dry_run_trace_present": False,
        "dry_run_provider_payloads_present": False,
    }


def _default_selected_strategy() -> dict[str, Any]:
    return {"selected_strategy_id": None, "paper_only": True, "one_strategy_only": True}


def _default_operation() -> dict[str, Any]:
    return {
        "operation_id": None,
        "operation_scope": "single_strategy_market_data_contract_execution_gate",
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }


def _default_boundary_checks() -> dict[str, bool]:
    return {
        "no_market_data_fetched_now": False,
        "no_contracts_qualified_now": False,
        "no_provider_called_now": False,
        "no_direct_ib_call": False,
        "no_reqMktData": False,
        "no_qualifyContracts": False,
        "no_reqContractDetails": False,
        "no_strategy_scan": False,
        "no_intents_created": False,
        "no_tickets_created": False,
        "no_orders_submitted": False,
        "no_state_written": False,
        "no_ledger_written": False,
        "no_broker_submission": False,
        "no_live_trading": False,
        "no_all_strategy_enablement": False,
        "no_scheduler_registration": False,
        "no_lifecycle_execution": False,
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
