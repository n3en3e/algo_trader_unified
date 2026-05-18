"""Pure Stage 4L-2 signal readiness gate report."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import math
import re
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding"
)
NEXT_RECOMMENDED_PHASE = "Stage 4L-3 controlled signal readiness validator"
PAPER_IBKR_PORTS = {4004}
REQUIRED_ACKNOWLEDGEMENTS = (
    "ACK_4L2_SIGNAL_READINESS_GATE_ONLY",
    "ACK_NO_STRATEGY_SCAN",
    "ACK_NO_SIGNAL_EXECUTION",
    "ACK_NO_INTENT_OR_TICKET_CREATION",
    "ACK_NO_ORDER_SUBMISSION",
    "ACK_NO_BROKER_SUBMISSION",
    "ACK_NO_STATE_OR_LEDGER_WRITES",
    "ACK_LIVE_TRADING_DISABLED",
    "ACK_SINGLE_STRATEGY_ONLY",
)
EXPECTED_4L1_FLOW_STEPS = (
    "validate_stage4k_acceptance",
    "validate_selected_strategy_scope",
    "validate_accepted_market_data_inputs",
    "validate_accepted_contract_qualification_inputs",
    "validate_signal_schema_requirements",
    "validate_no_execution_permissions",
    "prepare_stage4l2_signal_readiness_gate_inputs",
)
EXPECTED_4L3_FLOW_STEPS = (
    "validate_stage4l1_plan",
    "validate_operator_acknowledgements",
    "validate_signal_readiness_inputs",
    "validate_strategy_registry_scope",
    "validate_signal_schema_scope",
    "validate_no_execution_permissions",
    "prepare_stage4l3_controlled_validator_payload",
)
STALE_4J_KEYS = (
    "proposed_execution_permissions_for_" + "4J5",
    "may_call_" + "strategy_next_phase",
    "may_build_" + "executor_next_phase",
    "may_fetch_" + "market_data_next_phase",
)
UNSAFE_TRUE_FLAGS = (
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
DISABLED_INPUT_FLAGS = (
    "allow_strategy_scan",
    "allow_signal_execution",
    "allow_intent_creation",
    "allow_ticket_creation",
    "allow_order_submission",
    "allow_broker_submission",
    "allow_state_write",
    "allow_ledger_write",
    "live_trading_enabled",
    "all_strategies_enabled",
)
DISABLED_PAYLOAD_FLAGS = DISABLED_INPUT_FLAGS + (
    "allow_direct_reqMktData",
    "allow_direct_qualifyContracts",
    "allow_direct_reqContractDetails",
    "allow_provider_execution",
)
BLOCKED_ACTIONS = [
    "strategy_scan",
    "signal_execution",
    "intent_creation",
    "ticket_creation",
    "order_submission",
    "broker_submission",
    "state_write",
    "ledger_write",
    "direct_reqMktData",
    "direct_qualifyContracts",
    "direct_reqContractDetails",
    "provider_execution",
    "live_trading",
    "all_strategy_automation",
]
ORDERED_NEXT_STEPS = [
    "Build Stage 4L-3 controlled signal readiness validator.",
    "Keep live trading disabled.",
    "Keep all other strategies disabled until explicitly enabled.",
    "Keep broker submission separately gated until the explicit broker-submission phase.",
    "Keep intents, tickets, state writes, and ledger writes separately gated until their explicit phases.",
    "Use proposed 4L-3 payload only for controlled signal-readiness validation.",
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
    "Do not calculate signals now.",
    "Do not call validators now.",
]


def build_stage4l2_signal_readiness_gate_report(
    *,
    stage4l1_signal_readiness_plan_report: dict | None,
    operator_acknowledgements: list[str] | None = None,
    strategy_registry_snapshot: dict | list | None = None,
    signal_schema_snapshot: dict | None = None,
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
    """Build a read-only gate report for Stage 4L-3 controlled validation."""

    acknowledgements: Any = [] if operator_acknowledgements is None else operator_acknowledgements
    try:
        return _json_safe(
            _build_report(
                stage4l1_signal_readiness_plan_report=stage4l1_signal_readiness_plan_report,
                operator_acknowledgements=acknowledgements,
                strategy_registry_snapshot=strategy_registry_snapshot,
                signal_schema_snapshot=signal_schema_snapshot,
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
        generated_at = _generated_at(now_provider)
        message = f"unexpected Stage 4L-2 signal readiness gate failure: {type(exc).__name__}: {exc}"
        payload = _proposed_4l3_payload(None, None, {}, generated_at, False)
        flow = _proposed_4l3_flow()
        return _json_safe(
            _base_report(
                generated_at=generated_at,
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_selected_strategy(None, False, False),
                operation=_operation(None),
                stage4l1_plan_checks=_default_stage4l1_plan_checks(),
                operator_acknowledgement_checks=_acknowledgement_checks(acknowledgements),
                signal_input_checks=_default_signal_input_checks(),
                strategy_registry_checks=_default_strategy_registry_checks(),
                signal_schema_checks=_default_signal_schema_checks(),
                signal_readiness_gate=_signal_readiness_gate(None, None, False, [message], [message], []),
                proposed_4l3_signal_readiness_payload=payload,
                proposed_4l3_validation_flow=flow,
                boundary_checks=_boundary_checks(),
                required_inputs_for_4l3=_required_inputs_for_4l3(),
                activation_snapshot_checks=_default_activation_snapshot_checks(),
                state_checks=_default_state_checks(),
                risk_checks=_default_risk_checks(),
                scheduler_checks=_default_scheduler_checks(),
                lifecycle_checks=_default_lifecycle_checks(),
                paper_broker_checks=_default_paper_broker_checks(),
                market_window_checks=_default_market_window_checks(),
                safety_checks=_safety_checks(_boundary_checks(), payload),
                ready=False,
                blockers=[message],
                warnings=[],
                errors=[message],
            )
        )


def _build_report(
    *,
    stage4l1_signal_readiness_plan_report: Any,
    operator_acknowledgements: Any,
    strategy_registry_snapshot: Any,
    signal_schema_snapshot: Any,
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
    report = stage4l1_signal_readiness_plan_report if isinstance(stage4l1_signal_readiness_plan_report, dict) else None
    data = _mapping(report)
    blockers: list[str] = []
    warnings: list[str] = []
    errors = _as_string_list(data.get("errors"))

    if stage4l1_signal_readiness_plan_report is None:
        blockers.append("Stage 4L-1 signal readiness plan report is missing")
    elif report is None:
        blockers.append("Stage 4L-1 signal readiness plan report must be a dict")
        errors.append("Stage 4L-1 signal readiness plan report must be a dict")

    selected_strategy_id = _selected_strategy_id(data)
    operation_id = _operation_id(data)
    inputs = _mapping(data.get("proposed_signal_readiness_inputs"))
    sanitized_inputs = _copy_dict(inputs) if isinstance(inputs, dict) else {}

    artifact_checks = _artifact_checks(stage4l1_signal_readiness_plan_report, data)
    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    operation, operation_blockers = _operation_checks(data, operation_id)
    stage4l1_checks, stage4l1_blockers = _stage4l1_plan_checks(data)
    ack_checks, ack_blockers, ack_warnings = _acknowledgement_checks(operator_acknowledgements)
    signal_input_checks, signal_input_blockers = _signal_input_checks(data, selected_strategy_id, operation_id)
    flow_checks, flow_blockers = _inherited_flow_checks(data.get("proposed_4l2_validation_flow"))
    registry_checks, registry_blockers, registry_warnings = _strategy_registry_checks(strategy_registry_snapshot, selected_strategy_id)
    schema_checks, schema_blockers, schema_warnings = _signal_schema_checks(signal_schema_snapshot, selected_strategy_id, signal_input_checks, sanitized_inputs)
    boundary_checks, boundary_blockers = _stage4l1_boundary_checks(data)
    stage4l1_safety_blockers = _stage4l1_safety_blockers(data)
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
    stale_blockers = _stale_key_blockers(
        {
            "stage4l1_plan_checks": data,
            "proposed_4l3_signal_readiness_payload": sanitized_inputs,
        }
    )

    preliminary_blockers = (
        selected_blockers
        + operation_blockers
        + _artifact_blockers(artifact_checks)
        + stage4l1_blockers
        + ack_blockers
        + signal_input_blockers
        + flow_blockers
        + registry_blockers
        + schema_blockers
        + boundary_blockers
        + stage4l1_safety_blockers
        + stale_blockers
        + activation_blockers
        + state_blockers
        + risk_blockers
        + scheduler_blockers
        + lifecycle_blockers
        + broker_blockers
        + market_blockers
    )
    warnings.extend(
        ack_warnings
        + registry_warnings
        + schema_warnings
        + activation_warnings
        + state_warnings
        + risk_warnings
        + scheduler_warnings
        + lifecycle_warnings
        + broker_warnings
        + market_warnings
    )
    if errors:
        preliminary_blockers.append("Stage 4L-1 report contains errors")

    proposed_flow = _proposed_4l3_flow()
    payload_ready = not preliminary_blockers and not errors
    payload = _proposed_4l3_payload(selected_strategy_id, operation_id, sanitized_inputs, generated_at, payload_ready)
    payload_blockers = _payload_blockers(payload)
    proposed_flow_checks, proposed_flow_blockers = _proposed_flow_checks(proposed_flow)
    safety_checks = _safety_checks(boundary_checks, payload)
    safety_blockers = _safety_blockers(safety_checks)

    blockers.extend(preliminary_blockers + payload_blockers + proposed_flow_blockers + safety_blockers)
    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    ready = (
        not blocker_list
        and not error_list
        and stage4l1_checks.get("stage4l1_ready_for_stage4l2") is True
        and ack_checks.get("acknowledgements_complete") is True
        and signal_input_checks.get("signal_readiness_inputs_valid") is True
        and flow_checks.get("flow_ready") is True
        and proposed_flow_checks.get("flow_ready") is True
        and payload.get("allow_controlled_signal_readiness_validator_call") is True
        and all(safety_checks.values())
    )
    if not ready and payload.get("allow_controlled_signal_readiness_validator_call") is True:
        payload["allow_controlled_signal_readiness_validator_call"] = False

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
        operation=operation,
        stage4l1_plan_checks={**stage4l1_checks, "inherited_flow_checks": flow_checks},
        operator_acknowledgement_checks=ack_checks,
        signal_input_checks=signal_input_checks,
        strategy_registry_checks=registry_checks,
        signal_schema_checks=schema_checks,
        signal_readiness_gate=_signal_readiness_gate(selected_strategy_id, operation_id, ready, ["all Stage 4L-2 gates passed"] if ready else [], blocker_list, warning_list),
        proposed_4l3_signal_readiness_payload=payload,
        proposed_4l3_validation_flow=proposed_flow,
        boundary_checks=boundary_checks,
        required_inputs_for_4l3=_required_inputs_for_4l3(),
        activation_snapshot_checks=activation_checks,
        state_checks=state_checks,
        risk_checks=risk_checks,
        scheduler_checks=scheduler_checks,
        lifecycle_checks=lifecycle_checks,
        paper_broker_checks=paper_broker_checks,
        market_window_checks=market_window_checks,
        safety_checks=safety_checks,
        ready=ready,
        blockers=[] if ready else blocker_list,
        warnings=warning_list,
        errors=error_list,
    )


def _artifact_checks(report: Any, data: dict[str, Any]) -> dict[str, bool]:
    readiness = _mapping(data.get("readiness_for_stage4l2"))
    return {
        "stage4l1_report_present": isinstance(report, dict),
        "stage4l1_report_ready": data.get("stage4l1_signal_readiness_plan_report") is True
        and readiness.get("ready_to_build_signal_readiness_gate") is True
        and data.get("success") is True,
        "selected_strategy_present": isinstance(_selected_strategy_id(data), str),
        "operation_id_present": isinstance(_operation_id(data), str),
        "signal_readiness_inputs_present": isinstance(data.get("proposed_signal_readiness_inputs"), dict),
        "proposed_4l2_validation_flow_present": isinstance(data.get("proposed_4l2_validation_flow"), list),
    }


def _artifact_blockers(checks: dict[str, bool]) -> list[str]:
    labels = {
        "stage4l1_report_present": "Stage 4L-1 signal readiness plan report is missing",
        "stage4l1_report_ready": "Stage 4L-1 signal readiness plan report is not ready for Stage 4L-2",
        "selected_strategy_present": "selected strategy is missing from Stage 4L-1 report",
        "operation_id_present": "operation_id is missing from Stage 4L-1 report",
        "signal_readiness_inputs_present": "proposed_signal_readiness_inputs must be a native dict",
        "proposed_4l2_validation_flow_present": "proposed_4l2_validation_flow must be present",
    }
    return [label for key, label in labels.items() if checks.get(key) is not True]


def _selected_strategy_checks(report: dict[str, Any], selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str]]:
    selected = _mapping(report.get("selected_strategy"))
    paper_only = selected.get("paper_only") is True
    one_strategy_only = selected.get("one_strategy_only") is True
    blockers: list[str] = []
    if not selected_strategy_id:
        blockers.append("selected strategy id is missing or invalid")
    if not paper_only:
        blockers.append("selected strategy must be paper_only true")
    if not one_strategy_only:
        blockers.append("selected strategy must be one_strategy_only true")
    return _selected_strategy(selected_strategy_id, paper_only, one_strategy_only), blockers


def _operation_checks(report: dict[str, Any], operation_id: str | None) -> tuple[dict[str, Any], list[str]]:
    operation = _mapping(report.get("operation"))
    blockers: list[str] = []
    if not operation_id:
        blockers.append("operation_id is missing from Stage 4L-1 operation")
    if operation.get("live_trading_enabled") is True:
        blockers.append("Stage 4L-1 operation shows live trading enabled")
    if operation.get("broker_submission_enabled") is True:
        blockers.append("Stage 4L-1 operation shows broker submission enabled")
    return _operation(operation_id, operation.get("operation_scope") or "single_strategy_signal_readiness_gate"), blockers


def _stage4l1_plan_checks(report: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    readiness = _mapping(report.get("readiness_for_stage4l2"))
    checks = {
        "stage4l1_signal_readiness_plan_report": report.get("stage4l1_signal_readiness_plan_report") is True,
        "stage4l1_success": report.get("success") is True,
        "stage4l1_ready_for_stage4l2": readiness.get("ready_to_build_signal_readiness_gate") is True,
    }
    blockers = [f"Stage 4L-1 plan check {key} failed" for key, value in checks.items() if value is not True]
    return checks, blockers


def _acknowledgement_checks(acks: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if not isinstance(acks, list):
        return {
            "operator_acknowledgements_is_list": False,
            "required_acknowledgements": list(REQUIRED_ACKNOWLEDGEMENTS),
            "provided_acknowledgements": [],
            "missing_acknowledgements": list(REQUIRED_ACKNOWLEDGEMENTS),
            "extra_acknowledgements": [],
            "acknowledgements_complete": False,
        }, ["operator_acknowledgements must be a list"], warnings
    provided = [item for item in acks if isinstance(item, str)]
    malformed_count = len(acks) - len(provided)
    missing = [item for item in REQUIRED_ACKNOWLEDGEMENTS if item not in provided]
    extra = [item for item in provided if item not in REQUIRED_ACKNOWLEDGEMENTS]
    if malformed_count:
        blockers.append("operator_acknowledgements entries must be strings")
    if missing:
        blockers.append("required operator acknowledgements are missing")
    if extra:
        warnings.append("extra operator acknowledgement strings were ignored")
    return {
        "operator_acknowledgements_is_list": True,
        "required_acknowledgements": list(REQUIRED_ACKNOWLEDGEMENTS),
        "provided_acknowledgements": provided,
        "missing_acknowledgements": missing,
        "extra_acknowledgements": extra,
        "acknowledgements_complete": not missing and malformed_count == 0,
    }, blockers, warnings


def _signal_input_checks(report: dict[str, Any], selected_strategy_id: str | None, operation_id: str | None) -> tuple[dict[str, Any], list[str]]:
    raw = report.get("proposed_signal_readiness_inputs")
    inputs = _mapping(raw)
    blockers: list[str] = []
    if not isinstance(raw, dict):
        return _default_signal_input_checks(), ["proposed_signal_readiness_inputs must be a native dict"]
    if not _primitive_json_safe(inputs):
        blockers.append("proposed_signal_readiness_inputs must be JSON-safe")
    exact_checks = {
        "selected_strategy_id_matches": inputs.get("selected_strategy_id") == selected_strategy_id,
        "operation_id_matches": inputs.get("operation_id") == operation_id,
        "paper_only": inputs.get("paper_only") is True,
        "one_strategy_only": inputs.get("one_strategy_only") is True,
        "read_only": inputs.get("read_only") is True,
        "market_data_inputs_available_bool": isinstance(inputs.get("market_data_inputs_available"), bool),
        "contract_qualification_inputs_available_bool": isinstance(inputs.get("contract_qualification_inputs_available"), bool),
    }
    for key, value in exact_checks.items():
        if value is not True:
            blockers.append(f"proposed_signal_readiness_inputs {key} failed")
    md_available = inputs.get("market_data_inputs_available") is True
    cq_available = inputs.get("contract_qualification_inputs_available") is True
    if not (md_available or cq_available):
        blockers.append("at least one signal readiness input category must be available")
    md_results, md_blockers = _accepted_result_list_checks(inputs.get("accepted_market_data_results"), "accepted_market_data_results")
    cq_results, cq_blockers = _accepted_result_list_checks(inputs.get("accepted_contract_qualification_results"), "accepted_contract_qualification_results")
    blockers.extend(md_blockers + cq_blockers)
    for key in DISABLED_INPUT_FLAGS:
        if inputs.get(key) is not False:
            blockers.append(f"proposed_signal_readiness_inputs.{key} must be strict native boolean false")
        if inputs.get(key) == "False":
            blockers.append(f"proposed_signal_readiness_inputs.{key} must not be string False")
    checks = {
        "signal_readiness_inputs_valid": not blockers,
        **exact_checks,
        "market_data_inputs_available": md_available,
        "contract_qualification_inputs_available": cq_available,
        "at_least_one_input_category_available": md_available or cq_available,
        "accepted_market_data_results_is_list": isinstance(inputs.get("accepted_market_data_results"), list),
        "accepted_contract_qualification_results_is_list": isinstance(inputs.get("accepted_contract_qualification_results"), list),
        "accepted_market_data_result_count": len(md_results),
        "accepted_contract_qualification_result_count": len(cq_results),
        "accepted_results_json_safe": not any("JSON-safe" in item for item in blockers),
        "accepted_results_have_no_unsafe_flags": not any("must not be true" in item or "native bool false" in item for item in blockers),
    }
    return checks, _dedupe(blockers)


def _accepted_result_list_checks(raw: Any, label: str) -> tuple[list[dict[str, Any]], list[str]]:
    blockers: list[str] = []
    if raw is None:
        return [], [f"proposed_signal_readiness_inputs.{label} must be a native list"]
    if not isinstance(raw, list):
        return [], [f"proposed_signal_readiness_inputs.{label} must be a native list"]
    results: list[dict[str, Any]] = []
    for index, item in enumerate(raw or []):
        item_label = f"proposed_signal_readiness_inputs.{label}[{index}]"
        if not isinstance(item, dict):
            blockers.append(f"{item_label} must be a native dict")
            continue
        if not _primitive_json_safe(item):
            blockers.append(f"{item_label} must be JSON-safe")
        blockers.extend(_accepted_result_blockers(item, item_label))
        for key in ("failure_reason", "skipped_reason"):
            value = item.get(key)
            if isinstance(value, str) and (_has_raw_newline(value) or _has_memory_repr(value)):
                blockers.append(f"{item_label}.{key} must be flat JSON-safe string")
        results.append(_copy_dict(item))
    return results, blockers


def _accepted_result_blockers(result: dict[str, Any], label: str) -> list[str]:
    blockers: list[str] = []
    for key in UNSAFE_TRUE_FLAGS:
        if result.get(key) is True:
            blockers.append(f"{label}.{key} must not be true")
        if result.get(key) == "False":
            blockers.append(f"{label}.{key} must be native bool false, not string")
    return blockers


def _inherited_flow_checks(flow: Any) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if not isinstance(flow, list):
        return {"flow_ready": False, "step_count": 0, "sequence_numbers": [], "step_names": []}, ["proposed_4l2_validation_flow must be a native list"]
    if len(flow) != len(EXPECTED_4L1_FLOW_STEPS):
        blockers.append("proposed_4l2_validation_flow must contain exactly seven expected Stage 4L-1 steps")
    sequence_numbers: list[Any] = []
    step_names: list[Any] = []
    for index, item in enumerate(flow or []):
        if not isinstance(item, dict):
            blockers.append("proposed_4l2_validation_flow contains malformed step")
            continue
        sequence_numbers.append(item.get("sequence_number"))
        step_names.append(item.get("step_name"))
        if item.get("sequence_number") != index + 1:
            blockers.append("proposed_4l2_validation_flow sequence_number values must be strictly sequential integers 1 through 7")
        if index < len(EXPECTED_4L1_FLOW_STEPS) and item.get("step_name") != EXPECTED_4L1_FLOW_STEPS[index]:
            blockers.append("proposed_4l2_validation_flow step_name values must exactly match the Stage 4L-1 contract")
        for key in (
            "would_execute_strategy_now",
            "would_calculate_signal_now",
            "would_create_intent_now",
            "would_create_ticket_now",
            "would_submit_order_now",
            "would_write_state_now",
            "would_write_ledger_now",
            "live_trading_enabled",
            "broker_submission_enabled",
        ):
            if item.get(key) is not False:
                blockers.append(f"proposed_4l2_validation_flow {key} must be false")
        if item.get("paper_only") is not True:
            blockers.append("proposed_4l2_validation_flow paper_only must be true")
    return {"flow_ready": not blockers, "step_count": len(flow), "sequence_numbers": sequence_numbers, "step_names": step_names}, _dedupe(blockers)


def _strategy_registry_checks(snapshot: Any, selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if snapshot is None:
        warnings.append("strategy registry snapshot missing; verify selected strategy eligibility before Stage 4L-3")
        return _default_strategy_registry_checks(), blockers, warnings
    candidates: set[str] = set()
    has_candidate_set = False
    selected_explicit_paper = None
    malformed = False
    all_enabled = False
    if isinstance(snapshot, dict):
        all_enabled = snapshot.get("all_strategies_enabled") is True
        for key in ("candidate_strategy_ids", "paper_eligible_strategy_ids"):
            values = snapshot.get(key)
            if isinstance(values, list):
                has_candidate_set = True
                candidates.update(item for item in values if isinstance(item, str) and item)
            elif values is not None:
                malformed = True
        strategies = snapshot.get("strategies")
        if isinstance(strategies, list):
            has_candidate_set = True
            _collect_strategy_records(strategies, candidates, selected_strategy_id, blockers)
            selected_explicit_paper = _selected_paper_from_records(strategies, selected_strategy_id)
        elif strategies is not None:
            malformed = True
    elif isinstance(snapshot, list):
        has_candidate_set = True
        if all(isinstance(item, str) for item in snapshot):
            candidates.update(item for item in snapshot if item)
        elif all(isinstance(item, dict) for item in snapshot):
            _collect_strategy_records(snapshot, candidates, selected_strategy_id, blockers)
            selected_explicit_paper = _selected_paper_from_records(snapshot, selected_strategy_id)
        else:
            malformed = True
    else:
        malformed = True
    if all_enabled:
        blockers.append("strategy registry snapshot explicitly enables all strategies")
    if has_candidate_set and selected_strategy_id not in candidates:
        blockers.append("selected strategy is not present in supplied strategy registry snapshot")
    if selected_explicit_paper is False:
        blockers.append("strategy registry snapshot marks selected strategy paper_eligible false")
    if malformed:
        warnings.append("strategy registry snapshot malformed; ignored malformed entries without reselecting strategy")
    return {
        "strategy_registry_snapshot_present": True,
        "candidate_strategy_ids": sorted(candidates),
        "selected_strategy_present": (selected_strategy_id in candidates) if has_candidate_set else None,
        "selected_strategy_paper_eligible": selected_explicit_paper,
        "all_strategies_enabled": all_enabled,
        "registry_parse_warning": malformed,
    }, blockers, warnings


def _collect_strategy_records(records: list[Any], candidates: set[str], selected_strategy_id: str | None, blockers: list[str]) -> None:
    for record in records:
        if not isinstance(record, dict):
            continue
        strategy_id = record.get("strategy_id")
        if isinstance(strategy_id, str) and strategy_id:
            if record.get("paper_eligible") is not False:
                candidates.add(strategy_id)
            if strategy_id == selected_strategy_id and record.get("paper_eligible") is False:
                blockers.append("strategy registry snapshot marks selected strategy paper_eligible false")


def _selected_paper_from_records(records: list[Any], selected_strategy_id: str | None) -> bool | None:
    for record in records:
        if isinstance(record, dict) and record.get("strategy_id") == selected_strategy_id and "paper_eligible" in record:
            return record.get("paper_eligible") is True
    return None


def _signal_schema_checks(snapshot: Any, selected_strategy_id: str | None, input_checks: dict[str, Any], proposed_inputs: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if snapshot is None:
        warnings.append("signal schema snapshot missing; verify Stage 4L-3 controlled validator input contract before proceeding")
        return _default_signal_schema_checks(), blockers, warnings
    if not isinstance(snapshot, dict):
        return _default_signal_schema_checks(present=True, valid=False), ["signal_schema_snapshot must be a native dict"], warnings
    if snapshot.get("selected_strategy_id") not in (None, selected_strategy_id):
        blockers.append("signal schema selected_strategy_id does not match selected strategy")
    expected_sections = snapshot.get("expected_input_sections")
    expected_sections_valid = True
    missing_sections: list[str] = []
    if expected_sections is not None:
        if not isinstance(expected_sections, list):
            expected_sections_valid = False
            blockers.append("signal_schema_snapshot.expected_input_sections must be a list")
        else:
            for item in expected_sections:
                if not isinstance(item, str):
                    expected_sections_valid = False
                    blockers.append("signal_schema_snapshot.expected_input_sections entries must be strings")
                    continue
                if item not in proposed_inputs:
                    missing_sections.append(item)
            if missing_sections:
                blockers.append("proposed_signal_readiness_inputs missing expected signal schema sections")
    if snapshot.get("requires_market_data") is True and input_checks.get("market_data_inputs_available") is not True:
        blockers.append("signal schema requires market data but market data inputs are unavailable")
    if snapshot.get("requires_contract_qualification") is True and input_checks.get("contract_qualification_inputs_available") is not True:
        blockers.append("signal schema requires contract qualification but contract qualification inputs are unavailable")
    for key in (
        "allow_strategy_scan",
        "allow_signal_execution",
        "allow_intent_creation",
        "allow_order_submission",
        "allow_state_write",
        "allow_ledger_write",
        "live_trading_enabled",
        "broker_submission_enabled",
        "all_strategies_enabled",
    ):
        if snapshot.get(key) is True:
            blockers.append(f"signal schema {key} must not be true")
    return {
        "signal_schema_snapshot_present": True,
        "signal_schema_snapshot_valid": not blockers,
        "selected_strategy_matches": snapshot.get("selected_strategy_id") in (None, selected_strategy_id),
        "expected_input_sections_valid": expected_sections_valid,
        "missing_input_sections": sorted(missing_sections),
        "requires_market_data": snapshot.get("requires_market_data") is True,
        "requires_contract_qualification": snapshot.get("requires_contract_qualification") is True,
        "execution_permissions_disabled": not any(snapshot.get(key) is True for key in DISABLED_INPUT_FLAGS + ("broker_submission_enabled",)),
    }, blockers, warnings


def _proposed_4l3_payload(
    selected_strategy_id: str | None,
    operation_id: str | None,
    signal_readiness_inputs: dict[str, Any],
    generated_at: str,
    allow_validator: bool,
) -> dict[str, Any]:
    return {
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "source_stage": "4L-2",
        "source_plan_stage": "4L-1",
        "input_scope": "single_strategy_controlled_signal_readiness_validation",
        "paper_only": True,
        "one_strategy_only": True,
        "read_only": True,
        "allow_controlled_signal_readiness_validator_call": allow_validator,
        "allow_strategy_scan": False,
        "allow_signal_execution": False,
        "allow_intent_creation": False,
        "allow_ticket_creation": False,
        "allow_order_submission": False,
        "allow_broker_submission": False,
        "allow_state_write": False,
        "allow_ledger_write": False,
        "allow_direct_reqMktData": False,
        "allow_direct_qualifyContracts": False,
        "allow_direct_reqContractDetails": False,
        "allow_provider_execution": False,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "signal_readiness_inputs": _copy_dict(signal_readiness_inputs),
        "generated_at": generated_at,
    }


def _payload_blockers(payload: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not isinstance(payload, dict) or not _primitive_json_safe(payload):
        blockers.append("proposed_4l3_signal_readiness_payload must be a native JSON-safe dict")
    for key in DISABLED_PAYLOAD_FLAGS:
        if payload.get(key) is not False:
            blockers.append(f"proposed_4l3_signal_readiness_payload.{key} must be strict native boolean false")
        if payload.get(key) == "False":
            blockers.append(f"proposed_4l3_signal_readiness_payload.{key} must not be string False")
    return blockers


def _proposed_4l3_flow() -> list[dict[str, Any]]:
    return [
        {
            "sequence_number": index + 1,
            "step_name": step_name,
            "target_component": "stage4l3_controlled_signal_readiness_validator",
            "input_sections": _flow_input_sections(step_name),
            "would_call_signal_readiness_validator_now": False,
            "would_execute_strategy_now": False,
            "would_calculate_signal_now": False,
            "would_create_intent_now": False,
            "would_create_ticket_now": False,
            "would_submit_order_now": False,
            "would_write_state_now": False,
            "would_write_ledger_now": False,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
            "paper_only": True,
        }
        for index, step_name in enumerate(EXPECTED_4L3_FLOW_STEPS)
    ]


def _flow_input_sections(step_name: str) -> list[str]:
    sections = {
        "validate_stage4l1_plan": ["stage4l1_plan_checks", "artifact_checks"],
        "validate_operator_acknowledgements": ["operator_acknowledgement_checks"],
        "validate_signal_readiness_inputs": ["signal_input_checks", "proposed_4l3_signal_readiness_payload.signal_readiness_inputs"],
        "validate_strategy_registry_scope": ["strategy_registry_checks"],
        "validate_signal_schema_scope": ["signal_schema_checks"],
        "validate_no_execution_permissions": ["boundary_checks", "safety_checks"],
        "prepare_stage4l3_controlled_validator_payload": ["proposed_4l3_signal_readiness_payload", "required_inputs_for_4l3"],
    }
    return sections.get(step_name, [])


def _proposed_flow_checks(flow: Any) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if not isinstance(flow, list):
        return {"flow_ready": False}, ["proposed_4l3_validation_flow must be a native list"]
    if len(flow) != len(EXPECTED_4L3_FLOW_STEPS):
        blockers.append("proposed_4l3_validation_flow must contain exactly seven steps")
    for index, item in enumerate(flow or []):
        if not isinstance(item, dict):
            blockers.append("proposed_4l3_validation_flow contains malformed step")
            continue
        if item.get("sequence_number") != index + 1:
            blockers.append("proposed_4l3_validation_flow sequence_number values must be strictly sequential 1 through 7")
        if index < len(EXPECTED_4L3_FLOW_STEPS) and item.get("step_name") != EXPECTED_4L3_FLOW_STEPS[index]:
            blockers.append("proposed_4l3_validation_flow step order is not deterministic")
        for key in (
            "would_call_signal_readiness_validator_now",
            "would_execute_strategy_now",
            "would_calculate_signal_now",
            "would_create_intent_now",
            "would_create_ticket_now",
            "would_submit_order_now",
            "would_write_state_now",
            "would_write_ledger_now",
            "live_trading_enabled",
            "broker_submission_enabled",
        ):
            if item.get(key) is not False:
                blockers.append(f"proposed_4l3_validation_flow {key} must be false")
        if item.get("paper_only") is not True:
            blockers.append("proposed_4l3_validation_flow paper_only must be true")
    return {"flow_ready": not blockers, "step_count": len(flow), "sequence_numbers": [item.get("sequence_number") for item in flow if isinstance(item, dict)]}, _dedupe(blockers)


def _stage4l1_boundary_checks(report: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    source = _mapping(report.get("boundary_checks"))
    checks = _boundary_checks()
    required = {
        "no_provider_called_in_4l1": source.get("no_provider_called_in_4l1") is True,
        "no_market_data_fetched": source.get("no_market_data_fetched") is True,
        "no_contracts_qualified": source.get("no_contracts_qualified") is True,
        "no_direct_ib_call": source.get("no_direct_ib_call") is True,
        "no_direct_reqMktData": source.get("no_direct_reqMktData") is True,
        "no_direct_qualifyContracts": source.get("no_direct_qualifyContracts") is True,
        "no_direct_reqContractDetails": source.get("no_direct_reqContractDetails") is True,
        "no_strategy_scan": source.get("no_strategy_scan") is True,
        "no_signal_execution": source.get("no_signal_execution") is True,
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
    }
    checks.update(required)
    checks["no_signal_readiness_validator_called"] = True
    checks["no_provider_called"] = True
    blockers = [f"Stage 4L-1 boundary check {key} must be strict native boolean true" for key, value in required.items() if value is not True]
    return checks, blockers


def _stage4l1_safety_blockers(report: dict[str, Any]) -> list[str]:
    source = _mapping(report.get("safety_checks"))
    required = (
        "no_live_trading",
        "no_all_strategy_enablement",
        "no_broker_submission_enabled",
        "no_direct_market_data",
        "no_direct_contract_qualification",
        "no_strategy_scan",
        "no_signal_execution",
        "no_order_submission",
        "no_intent_creation",
        "no_ticket_creation",
        "no_state_write",
        "no_ledger_write",
    )
    return [f"Stage 4L-1 safety check {key} must be strict native boolean true" for key in required if source.get(key) is not True]


def _boundary_checks() -> dict[str, bool]:
    return {
        "no_signal_readiness_validator_called": True,
        "no_provider_called": True,
        "no_market_data_fetched": True,
        "no_contracts_qualified": True,
        "no_direct_ib_call": True,
        "no_direct_reqMktData": True,
        "no_direct_qualifyContracts": True,
        "no_direct_reqContractDetails": True,
        "no_strategy_scan": True,
        "no_signal_execution": True,
        "no_intents_created": True,
        "no_tickets_created": True,
        "no_orders_submitted": True,
        "no_broker_submission": True,
        "no_state_written": True,
        "no_ledger_written": True,
        "no_live_trading": True,
        "no_all_strategy_enablement": True,
        "no_scheduler_registration": True,
        "no_lifecycle_execution": True,
    }


def _safety_checks(boundary_checks: dict[str, bool], payload: dict[str, Any]) -> dict[str, bool]:
    return {
        "no_live_trading": boundary_checks.get("no_live_trading") is True and payload.get("live_trading_enabled") is False,
        "no_all_strategy_enablement": boundary_checks.get("no_all_strategy_enablement") is True and payload.get("all_strategies_enabled") is False,
        "no_broker_submission_enabled": boundary_checks.get("no_broker_submission") is True and payload.get("allow_broker_submission") is False,
        "no_direct_market_data": boundary_checks.get("no_market_data_fetched") is True and boundary_checks.get("no_direct_reqMktData") is True and payload.get("allow_direct_reqMktData") is False,
        "no_direct_contract_qualification": boundary_checks.get("no_contracts_qualified") is True
        and boundary_checks.get("no_direct_qualifyContracts") is True
        and boundary_checks.get("no_direct_reqContractDetails") is True
        and payload.get("allow_direct_qualifyContracts") is False
        and payload.get("allow_direct_reqContractDetails") is False,
        "no_strategy_scan": boundary_checks.get("no_strategy_scan") is True and payload.get("allow_strategy_scan") is False,
        "no_signal_execution": boundary_checks.get("no_signal_execution") is True and payload.get("allow_signal_execution") is False,
        "no_order_submission": boundary_checks.get("no_orders_submitted") is True and payload.get("allow_order_submission") is False,
        "no_intent_creation": boundary_checks.get("no_intents_created") is True and payload.get("allow_intent_creation") is False,
        "no_ticket_creation": boundary_checks.get("no_tickets_created") is True and payload.get("allow_ticket_creation") is False,
        "no_state_write": boundary_checks.get("no_state_written") is True and payload.get("allow_state_write") is False,
        "no_ledger_write": boundary_checks.get("no_ledger_written") is True and payload.get("allow_ledger_write") is False,
    }


def _safety_blockers(checks: dict[str, bool]) -> list[str]:
    return [f"safety check {key} failed" for key, value in checks.items() if value is not True]


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
    for prefix, snapshot in (
        ("scheduler_activation", scheduler_activation_snapshot),
        ("lifecycle_activation", lifecycle_activation_snapshot),
        ("activation", activation_snapshot),
    ):
        current, current_blockers, current_warnings = _activation_snapshot_checks(snapshot, selected_strategy_id, prefix)
        checks.update(current)
        blockers.extend(current_blockers)
        warnings.extend(current_warnings)
    return checks, blockers, warnings


def _activation_snapshot_checks(snapshot: Any, selected_strategy_id: str | None, prefix: str) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append(f"{prefix} snapshot missing; verify activation artifact before Stage 4L-3")
        return {f"{prefix}_snapshot_present": False, f"{prefix}_snapshot_matches": True}, blockers, warnings
    records = _explicit_records(data, f"{prefix}_record", f"{prefix}s")
    for record in records:
        if not isinstance(record, dict):
            warnings.append(f"malformed {prefix} snapshot entry ignored")
            continue
        if record.get("selected_strategy_id") not in (None, selected_strategy_id):
            blockers.append(f"{prefix} snapshot selected_strategy_id does not match")
        for key in ("live_trading_enabled", "all_strategies_enabled", "broker_submission_enabled", "strategy_scan_execution_enabled", "signal_execution_enabled", "lifecycle_transition_execution_enabled"):
            if record.get(key) is True:
                blockers.append(f"{prefix} snapshot {key} contradicts Stage 4L-2 safety")
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
        warnings.append("state snapshot missing; verify halt, reconciliation, intents, and positions before Stage 4L-3")
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
        warnings.append("risk snapshot missing; verify kill switch, hard halt, and daily loss controls before Stage 4L-3")
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
    matching_jobs = [job for job in jobs if isinstance(job, dict) and _job_strategy_id(job) == selected_strategy_id]
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("scheduler snapshot missing; verify scheduler state before Stage 4L-3")
    all_strategy_enabled = any(data.get(key) is True for key in ("all_strategy_scheduler_enabled", "all_strategies_enabled", "broad_scheduler_enabled"))
    scan_enabled = data.get("strategy_scan_execution_enabled") is True
    signal_enabled = data.get("signal_execution_enabled") is True
    selected_job_matches = True
    for job in matching_jobs:
        safe = (
            _job_strategy_id(job) == selected_strategy_id
            and job.get("broker_submission_enabled") is False
            and job.get("live_trading_enabled") is False
            and job.get("all_strategies_enabled") is False
            and job.get("strategy_scan_execution_enabled") is False
            and job.get("signal_execution_enabled") is False
        )
        selected_job_matches = selected_job_matches and safe
        scan_enabled = scan_enabled or job.get("strategy_scan_execution_enabled") is True
        signal_enabled = signal_enabled or job.get("signal_execution_enabled") is True
    if all_strategy_enabled:
        blockers.append("broad/all-strategy scheduler automation enabled")
    if matching_jobs and not selected_job_matches:
        blockers.append("selected strategy scheduled job does not match Stage 4L-2 safety constraints")
    if scan_enabled:
        blockers.append("scheduler snapshot strategy_scan_execution_enabled must be false")
    if signal_enabled:
        blockers.append("scheduler snapshot signal_execution_enabled must be false")
    return {
        "scheduler_snapshot_present": present,
        "scheduler_already_enabled": data.get("scheduler_automation_enabled") is True or bool(matching_jobs),
        "all_strategy_scheduler_enabled": all_strategy_enabled,
        "selected_strategy_job_matches": selected_job_matches,
        "strategy_scan_execution_enabled": scan_enabled,
        "signal_execution_enabled": signal_enabled,
    }, blockers, warnings


def _lifecycle_checks(snapshot: Any, selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; verify lifecycle state before Stage 4L-3")
    broad_enabled = any(data.get(key) is True for key in ("all_strategy_lifecycle_enabled", "all_strategies_enabled", "broad_lifecycle_enabled"))
    transition_enabled = data.get("lifecycle_transition_execution_enabled") is True
    lifecycle_enabled = data.get("lifecycle_automation_enabled") is True or broad_enabled
    matches = True
    if lifecycle_enabled:
        matches = (
            data.get("selected_strategy_id") == selected_strategy_id
            and data.get("broker_submission_enabled") is False
            and data.get("live_trading_enabled") is False
            and data.get("all_strategies_enabled") is False
            and broad_enabled is False
            and transition_enabled is False
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
        warnings.append("paper broker snapshot missing; verify PAPER broker config before Stage 4L-3")
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
        warnings.append("market is currently closed; gate validation remains report-only")
    if is_trading_day is False:
        warnings.append("snapshot is not a trading day; gate validation remains report-only")
    return {
        "market_window_snapshot_present": present,
        "allowed_to_schedule_paper_run": allowed,
        "is_trading_day": is_trading_day,
        "market_open": market_open,
        "reason": data.get("reason") if present else None,
    }, blockers, warnings


def _signal_readiness_gate(
    selected_strategy_id: str | None,
    operation_id: str | None,
    ready: bool,
    reasons: list[str],
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "ready_for_4l3_controlled_signal_readiness_validation": ready,
        "gate_status": "passed" if ready else "blocked",
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "allowed_next_phase": "stage4l3_controlled_signal_readiness_validator",
        "permission_scope": "single_strategy_signal_readiness_validation_only",
        "blocked_actions": BLOCKED_ACTIONS,
        "reasons": reasons,
        "blockers": blockers,
        "warnings": warnings,
    }


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    operation: dict[str, Any],
    stage4l1_plan_checks: dict[str, Any],
    operator_acknowledgement_checks: dict[str, Any],
    signal_input_checks: dict[str, Any],
    strategy_registry_checks: dict[str, Any],
    signal_schema_checks: dict[str, Any],
    signal_readiness_gate: dict[str, Any],
    proposed_4l3_signal_readiness_payload: dict[str, Any],
    proposed_4l3_validation_flow: list[dict[str, Any]],
    boundary_checks: dict[str, bool],
    required_inputs_for_4l3: list[dict[str, Any]],
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
        "stage4l2_signal_readiness_gate_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "operation": operation,
        "stage4l1_plan_checks": stage4l1_plan_checks,
        "operator_acknowledgement_checks": operator_acknowledgement_checks,
        "signal_input_checks": signal_input_checks,
        "strategy_registry_checks": strategy_registry_checks,
        "signal_schema_checks": signal_schema_checks,
        "signal_readiness_gate": signal_readiness_gate,
        "proposed_4l3_signal_readiness_payload": proposed_4l3_signal_readiness_payload,
        "proposed_4l3_validation_flow": proposed_4l3_validation_flow,
        "boundary_checks": boundary_checks,
        "required_inputs_for_4l3": required_inputs_for_4l3,
        "activation_snapshot_checks": activation_snapshot_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4l3": {
            "ready_to_execute_controlled_signal_readiness_validator": ready,
            "next_recommended_phase": NEXT_RECOMMENDED_PHASE,
            "blockers": blockers,
            "warnings": warnings,
        },
        "recommendations": {
            "ordered_next_steps": ORDERED_NEXT_STEPS,
            "do_not_do_yet": DO_NOT_DO_YET,
        },
        "success": ready,
        "errors": errors,
        "warnings": warnings,
    }


def _required_inputs_for_4l3() -> list[dict[str, Any]]:
    return [
        {"name": "accepted_stage4l1_plan_report", "required": True, "source": "Stage 4L-1"},
        {"name": "operator_acknowledgements", "required": True, "source": "operator supplied exact strings"},
        {"name": "selected_strategy_id", "required": True, "source": "Stage 4L-1 selected_strategy"},
        {"name": "operation_id", "required": True, "source": "Stage 4L-1 operation"},
        {"name": "proposed_signal_readiness_inputs", "required": True, "source": "Stage 4L-1"},
        {"name": "injected_controlled_signal_readiness_validator", "required": True, "source": "Stage 4L-3 only"},
    ]


def _selected_strategy(selected_strategy_id: str | None, paper_only: bool, one_strategy_only: bool) -> dict[str, Any]:
    return {"selected_strategy_id": selected_strategy_id, "paper_only": paper_only, "one_strategy_only": one_strategy_only}


def _operation(operation_id: str | None, operation_scope: str = "single_strategy_signal_readiness_gate") -> dict[str, Any]:
    return {
        "operation_id": operation_id,
        "operation_scope": operation_scope,
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }


def _default_artifact_checks() -> dict[str, bool]:
    return {
        "stage4l1_report_present": False,
        "stage4l1_report_ready": False,
        "selected_strategy_present": False,
        "operation_id_present": False,
        "signal_readiness_inputs_present": False,
        "proposed_4l2_validation_flow_present": False,
    }


def _default_stage4l1_plan_checks() -> dict[str, bool]:
    return {
        "stage4l1_signal_readiness_plan_report": False,
        "stage4l1_success": False,
        "stage4l1_ready_for_stage4l2": False,
    }


def _default_signal_input_checks() -> dict[str, Any]:
    return {
        "signal_readiness_inputs_valid": False,
        "selected_strategy_id_matches": False,
        "operation_id_matches": False,
        "paper_only": False,
        "one_strategy_only": False,
        "read_only": False,
        "market_data_inputs_available_bool": False,
        "contract_qualification_inputs_available_bool": False,
        "market_data_inputs_available": False,
        "contract_qualification_inputs_available": False,
        "at_least_one_input_category_available": False,
        "accepted_market_data_results_is_list": False,
        "accepted_contract_qualification_results_is_list": False,
        "accepted_market_data_result_count": 0,
        "accepted_contract_qualification_result_count": 0,
        "accepted_results_json_safe": False,
        "accepted_results_have_no_unsafe_flags": False,
    }


def _default_strategy_registry_checks() -> dict[str, Any]:
    return {
        "strategy_registry_snapshot_present": False,
        "candidate_strategy_ids": [],
        "selected_strategy_present": None,
        "selected_strategy_paper_eligible": None,
        "all_strategies_enabled": False,
        "registry_parse_warning": False,
    }


def _default_signal_schema_checks(present: bool = False, valid: bool = True) -> dict[str, Any]:
    return {
        "signal_schema_snapshot_present": present,
        "signal_schema_snapshot_valid": valid,
        "selected_strategy_matches": None,
        "expected_input_sections_valid": None,
        "missing_input_sections": [],
        "requires_market_data": False,
        "requires_contract_qualification": False,
        "execution_permissions_disabled": True,
    }


def _default_activation_snapshot_checks() -> dict[str, Any]:
    return {
        "scheduler_activation_snapshot_present": False,
        "scheduler_activation_snapshot_matches": True,
        "lifecycle_activation_snapshot_present": False,
        "lifecycle_activation_snapshot_matches": True,
        "activation_snapshot_present": False,
        "activation_snapshot_matches": True,
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
        "scheduler_already_enabled": False,
        "all_strategy_scheduler_enabled": False,
        "selected_strategy_job_matches": True,
        "strategy_scan_execution_enabled": False,
        "signal_execution_enabled": False,
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


def _selected_strategy_id(report: dict[str, Any]) -> str | None:
    value = _mapping(report.get("selected_strategy")).get("selected_strategy_id")
    return value if isinstance(value, str) and value.strip() else None


def _operation_id(report: dict[str, Any]) -> str | None:
    value = _mapping(report.get("operation")).get("operation_id")
    return value if isinstance(value, str) and value.strip() else None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_mapping(value: Any) -> tuple[bool, dict[str, Any]]:
    if value is None:
        return False, {}
    if isinstance(value, dict):
        return True, value
    return True, {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_string_list(value: Any) -> list[str]:
    return [item for item in _as_list(value) if isinstance(item, str)]


def _explicit_records(data: dict[str, Any], record_key: str, list_key: str) -> list[Any]:
    records: list[Any] = []
    if isinstance(data.get(record_key), dict):
        records.append(data.get(record_key))
    if isinstance(data.get(list_key), list):
        records.extend(data.get(list_key))
    if not records and data:
        records.append(data)
    return records


def _job_strategy_id(job: dict[str, Any]) -> Any:
    return job.get("selected_strategy_id", job.get("strategy_id"))


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return 0


def _first_present(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _stale_key_blockers(value: Any, path: str = "report") -> list[str]:
    blockers: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in STALE_4J_KEYS:
                blockers.append(f"stale 4J schema key present at {path}.{key}")
            blockers.extend(_stale_key_blockers(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            blockers.extend(_stale_key_blockers(item, f"{path}[{index}]"))
    return _dedupe(blockers)


def _copy_dict(value: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            copied[key] = _copy_dict(item)
        elif isinstance(item, list):
            copied[key] = [_copy_dict(child) if isinstance(child, dict) else child for child in item]
        else:
            copied[key] = item
    return copied


def _primitive_json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (Decimal, date, datetime)):
        return False
    if isinstance(value, list):
        return all(_primitive_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _primitive_json_safe(item) for key, item in value.items())
    return False


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _generated_at(now_provider: Callable[[], Any] | None) -> str:
    if now_provider is None:
        return DEFAULT_GENERATED_AT
    try:
        value = now_provider()
    except Exception as exc:  # noqa: BLE001 - timestamp provider failure becomes stable text.
        return f"{DEFAULT_GENERATED_AT} provider_error={type(exc).__name__}: {exc}"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return str(value)


def _has_raw_newline(value: str) -> bool:
    return "\n" in value or "\r" in value


def _has_memory_repr(value: str) -> bool:
    return re.search(r"<[^>]+ at 0x[0-9a-fA-F]+>", value) is not None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
