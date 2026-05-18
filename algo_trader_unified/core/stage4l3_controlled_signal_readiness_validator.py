"""Pure Stage 4L-3 controlled signal-readiness validator report."""

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
NEXT_RECOMMENDED_PHASE = "Stage 4L-4 signal readiness acceptance"
PAPER_IBKR_PORTS = {4004}
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
DISABLED_PAYLOAD_FLAGS = (
    "allow_strategy_scan",
    "allow_signal_execution",
    "allow_intent_creation",
    "allow_ticket_creation",
    "allow_order_submission",
    "allow_broker_submission",
    "allow_state_write",
    "allow_ledger_write",
    "allow_direct_reqMktData",
    "allow_direct_qualifyContracts",
    "allow_direct_reqContractDetails",
    "allow_provider_execution",
    "live_trading_enabled",
    "all_strategies_enabled",
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
REQUIRED_VALIDATOR_BOOL_FIELDS = (
    "signal_readiness_validated",
    "selected_strategy_ready_for_signal_evaluation",
    "validator_called",
    "strategy_scan_executed",
    "signal_execution_performed",
    "intent_created",
    "ticket_created",
    "order_submitted",
    "broker_submission_enabled",
    "state_written",
    "ledger_written",
    "direct_ib_call_made",
    "provider_call_made",
    "live_trading_enabled",
    "all_strategies_enabled",
)
DISABLED_VALIDATOR_FLAGS = (
    "strategy_scan_executed",
    "signal_execution_performed",
    "intent_created",
    "ticket_created",
    "order_submitted",
    "broker_submission_enabled",
    "state_written",
    "ledger_written",
    "direct_ib_call_made",
    "provider_call_made",
    "live_trading_enabled",
    "all_strategies_enabled",
)
FLOW_DISABLED_FLAGS = (
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
)
BOUNDARY_KEYS = (
    "no_signal_readiness_validator_called",
    "no_provider_called",
    "no_market_data_fetched",
    "no_contracts_qualified",
    "no_direct_ib_call",
    "no_direct_reqMktData",
    "no_direct_qualifyContracts",
    "no_direct_reqContractDetails",
    "no_strategy_scan",
    "no_signal_execution",
    "no_intents_created",
    "no_tickets_created",
    "no_orders_submitted",
    "no_broker_submission",
    "no_state_written",
    "no_ledger_written",
    "no_live_trading",
    "no_all_strategy_enablement",
    "no_scheduler_registration",
    "no_lifecycle_execution",
)
SAFETY_KEYS = (
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
ORDERED_NEXT_STEPS = [
    "Build Stage 4L-4 signal readiness acceptance.",
    "Keep live trading disabled.",
    "Keep all other strategies disabled until explicitly enabled.",
    "Keep broker submission separately gated until the explicit broker-submission phase.",
    "Keep intents, tickets, state writes, and ledger writes separately gated until their explicit phases.",
    "Use the accepted validator result only as read-only input for signal-readiness acceptance.",
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
    "Do not calculate executable signals now.",
]


def build_stage4l3_controlled_signal_readiness_validator_report(
    *,
    stage4l2_signal_readiness_gate_report: dict | None,
    controlled_signal_readiness_validator: Any = None,
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
    """Build the Stage 4L-3 controlled validator report from supplied artifacts only."""

    try:
        return _json_safe(
            _build_report(
                stage4l2_signal_readiness_gate_report=stage4l2_signal_readiness_gate_report,
                controlled_signal_readiness_validator=controlled_signal_readiness_validator,
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
        message = f"unexpected Stage 4L-3 controlled validator failure: {_flat_exception(exc)}"
        generated_at = _generated_at(now_provider)
        return _json_safe(
            _base_report(
                generated_at=generated_at,
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_selected_strategy(None, False, False),
                operation=_operation(None),
                stage4l2_gate_checks={"stage4l2_gate_ready": False},
                validator_payload_checks=_default_payload_checks(),
                validator_availability_checks=_validator_availability_checks(None, False),
                validator_execution_result=_validator_execution_result(False, False, False, {}, message, None, False),
                validator_call_trace=_validator_call_trace(None, None, {}, False, False, {}, message, None),
                signal_readiness_result_acceptance=_result_acceptance(None, None, False, {}, [message]),
                boundary_checks=_boundary_checks(False, False),
                required_inputs_for_4l4=_required_inputs_for_4l4(),
                activation_snapshot_checks=_default_activation_snapshot_checks(),
                state_checks=_default_state_checks(),
                risk_checks=_default_risk_checks(),
                scheduler_checks=_default_scheduler_checks(),
                lifecycle_checks=_default_lifecycle_checks(),
                paper_broker_checks=_default_paper_broker_checks(),
                market_window_checks=_default_market_window_checks(),
                safety_checks=_safety_checks(),
                ready=False,
                blockers=[message],
                warnings=[],
                errors=[message],
            )
        )


def _build_report(
    *,
    stage4l2_signal_readiness_gate_report: Any,
    controlled_signal_readiness_validator: Any,
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
    report = stage4l2_signal_readiness_gate_report if isinstance(stage4l2_signal_readiness_gate_report, dict) else None
    data = _mapping(report)
    errors = _as_string_list(data.get("errors"))
    blockers: list[str] = []
    warnings: list[str] = []

    if stage4l2_signal_readiness_gate_report is None:
        blockers.append("Stage 4L-2 signal readiness gate report is missing")
    elif report is None:
        blockers.append("Stage 4L-2 signal readiness gate report must be a dict")
        errors.append("Stage 4L-2 signal readiness gate report must be a dict")

    selected_strategy_id = _selected_strategy_id(data)
    operation_id = _operation_id(data)
    payload_raw = data.get("proposed_4l3_signal_readiness_payload")
    payload = _copy_dict(payload_raw) if isinstance(payload_raw, dict) else {}

    artifact_checks = _artifact_checks(stage4l2_signal_readiness_gate_report, data)
    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    operation, operation_blockers = _operation_checks(data, operation_id)
    gate_checks, gate_blockers = _stage4l2_gate_checks(data)
    payload_checks, payload_blockers = _payload_checks(payload_raw, payload, selected_strategy_id, operation_id)
    flow_checks, flow_blockers = _flow_checks(data.get("proposed_4l3_validation_flow"))
    registry_checks, registry_blockers, registry_warnings = _strategy_registry_checks(strategy_registry_snapshot, selected_strategy_id)
    schema_checks, schema_blockers, schema_warnings = _signal_schema_checks(signal_schema_snapshot, selected_strategy_id, payload)
    inherited_boundary_checks, inherited_boundary_blockers = _stage4l2_boundary_checks(data)
    inherited_safety_blockers = _stage4l2_safety_blockers(data)
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
    stale_blockers = _stale_key_blockers({"stage4l2": data, "payload": payload})

    allow_validator = payload.get("allow_controlled_signal_readiness_validator_call") is True
    availability_checks = _validator_availability_checks(controlled_signal_readiness_validator, allow_validator)
    availability_blockers = _validator_availability_blockers(availability_checks)

    pre_call_blockers = (
        blockers
        + selected_blockers
        + operation_blockers
        + _artifact_blockers(artifact_checks)
        + gate_blockers
        + payload_blockers
        + flow_blockers
        + registry_blockers
        + schema_blockers
        + inherited_boundary_blockers
        + inherited_safety_blockers
        + stale_blockers
        + activation_blockers
        + state_blockers
        + risk_blockers
        + scheduler_blockers
        + lifecycle_blockers
        + broker_blockers
        + market_blockers
        + availability_blockers
    )
    warnings.extend(
        registry_warnings
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
        pre_call_blockers.append("Stage 4L-2 report contains errors")

    validator_called = False
    call_success = False
    result: dict[str, Any] = {}
    failure_reason: str | None = None
    skipped_reason: str | None = None
    validator_result_blockers: list[str] = []

    if allow_validator and not availability_blockers and not pre_call_blockers:
        validator_called = True
        try:
            method = getattr(controlled_signal_readiness_validator, "validate_signal_readiness", None)
            raw_result = method(_copy_dict(payload)) if callable(method) else controlled_signal_readiness_validator(_copy_dict(payload))
        except Exception as exc:  # noqa: BLE001 - validator failures are expected report input.
            failure_reason = _flat_exception(exc)
            validator_result_blockers.append("controlled signal-readiness validator raised an exception")
        else:
            result, validator_result_blockers = _validator_result_checks(raw_result)
            call_success = not validator_result_blockers and result.get("signal_readiness_validated") is True
    else:
        if not allow_validator:
            skipped_reason = "Stage 4L-2 did not allow controlled signal-readiness validator call"
        elif availability_blockers:
            skipped_reason = "controlled signal-readiness validator unavailable"
        else:
            skipped_reason = "pre-call validation failed"

    if failure_reason and (_has_raw_newline(failure_reason) or _has_memory_repr(failure_reason)):
        validator_result_blockers.append("validator failure_reason must be flat JSON-safe string")
    if skipped_reason and (_has_raw_newline(skipped_reason) or _has_memory_repr(skipped_reason)):
        validator_result_blockers.append("validator skipped_reason must be flat JSON-safe string")

    execution_result = _validator_execution_result(
        attempted=allow_validator and availability_checks.get("validator_available") is True and not pre_call_blockers,
        validator_called=validator_called,
        success=call_success,
        result=result,
        failure_reason=failure_reason,
        selected_strategy_ready_for_signal_evaluation=result.get("selected_strategy_ready_for_signal_evaluation")
        if isinstance(result.get("selected_strategy_ready_for_signal_evaluation"), bool)
        else None,
        signal_readiness_validated=result.get("signal_readiness_validated") is True,
    )
    call_trace = _validator_call_trace(selected_strategy_id, operation_id, payload, validator_called, call_success, result, failure_reason, skipped_reason)
    trace_blockers = _call_trace_blockers(call_trace)
    acceptance_rejections = validator_result_blockers + trace_blockers
    if result.get("signal_readiness_validated") is not True and validator_called and not validator_result_blockers:
        acceptance_rejections.append("validator result did not validate signal readiness")
    acceptance = _result_acceptance(selected_strategy_id, operation_id, call_success, result, acceptance_rejections)
    boundary_checks = _boundary_checks(allow_validator, validator_called)
    safety_checks = _safety_checks()
    final_blockers = _dedupe(pre_call_blockers + validator_result_blockers + trace_blockers + _safety_blockers(safety_checks))
    final_errors = _dedupe(errors)
    if final_errors:
        final_blockers.append("Stage 4L-3 report contains errors")
        final_blockers = _dedupe(final_blockers)
    ready = (
        not final_blockers
        and not final_errors
        and gate_checks.get("stage4l2_ready_for_stage4l3") is True
        and payload_checks.get("payload_valid") is True
        and flow_checks.get("flow_ready") is True
        and availability_checks.get("validator_available") is True
        and execution_result.get("success") is True
        and acceptance.get("accepted") is True
        and acceptance.get("strategy_signal_ready") is True
        and all(boundary_checks.values())
        and all(safety_checks.values())
    )

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
        operation=operation,
        stage4l2_gate_checks={**gate_checks, "inherited_boundary_checks": inherited_boundary_checks, "inherited_flow_checks": flow_checks},
        validator_payload_checks=payload_checks,
        validator_availability_checks=availability_checks,
        validator_execution_result=execution_result,
        validator_call_trace=call_trace,
        signal_readiness_result_acceptance=acceptance,
        boundary_checks=boundary_checks,
        required_inputs_for_4l4=_required_inputs_for_4l4(),
        activation_snapshot_checks=activation_checks,
        state_checks=state_checks,
        risk_checks=risk_checks,
        scheduler_checks=scheduler_checks,
        lifecycle_checks=lifecycle_checks,
        paper_broker_checks=paper_broker_checks,
        market_window_checks=market_window_checks,
        safety_checks=safety_checks,
        ready=ready,
        blockers=[] if ready else final_blockers,
        warnings=_dedupe(warnings),
        errors=final_errors,
    )


def _artifact_checks(report: Any, data: dict[str, Any]) -> dict[str, bool]:
    readiness = _mapping(data.get("readiness_for_stage4l3"))
    return {
        "stage4l2_report_present": isinstance(report, dict),
        "stage4l2_report_ready": data.get("stage4l2_signal_readiness_gate_report") is True
        and readiness.get("ready_to_execute_controlled_signal_readiness_validator") is True
        and data.get("success") is True,
        "selected_strategy_present": isinstance(_selected_strategy_id(data), str) and bool(_selected_strategy_id(data)),
        "operation_id_present": isinstance(_operation_id(data), str) and bool(_operation_id(data)),
        "proposed_4l3_payload_present": isinstance(data.get("proposed_4l3_signal_readiness_payload"), dict),
        "proposed_4l3_validation_flow_present": isinstance(data.get("proposed_4l3_validation_flow"), list),
    }


def _artifact_blockers(checks: dict[str, bool]) -> list[str]:
    labels = {
        "stage4l2_report_present": "Stage 4L-2 signal readiness gate report is missing",
        "stage4l2_report_ready": "Stage 4L-2 signal readiness gate report is not ready for Stage 4L-3",
        "selected_strategy_present": "selected strategy is missing from Stage 4L-2 report",
        "operation_id_present": "operation_id is missing from Stage 4L-2 report",
        "proposed_4l3_payload_present": "proposed_4l3_signal_readiness_payload must be present",
        "proposed_4l3_validation_flow_present": "proposed_4l3_validation_flow must be present",
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
        blockers.append("operation_id is missing from Stage 4L-2 operation")
    if operation.get("live_trading_enabled") is True:
        blockers.append("Stage 4L-2 operation shows live trading enabled")
    if operation.get("broker_submission_enabled") is True:
        blockers.append("Stage 4L-2 operation shows broker submission enabled")
    return _operation(operation_id, operation.get("operation_scope") or "single_strategy_controlled_signal_readiness_validator"), blockers


def _stage4l2_gate_checks(report: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    readiness = _mapping(report.get("readiness_for_stage4l3"))
    checks = {
        "stage4l2_signal_readiness_gate_report": report.get("stage4l2_signal_readiness_gate_report") is True,
        "stage4l2_success": report.get("success") is True,
        "stage4l2_ready_for_stage4l3": readiness.get("ready_to_execute_controlled_signal_readiness_validator") is True,
    }
    blockers = [f"Stage 4L-2 gate check {key} failed" for key, value in checks.items() if value is not True]
    return checks, blockers


def _payload_checks(raw: Any, payload: dict[str, Any], selected_strategy_id: str | None, operation_id: str | None) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if not isinstance(raw, dict):
        return _default_payload_checks(), ["proposed_4l3_signal_readiness_payload must be a native dict"]
    if not _primitive_json_safe(payload):
        blockers.append("proposed_4l3_signal_readiness_payload must be JSON-safe")
    exact_checks = {
        "selected_strategy_id_matches": payload.get("selected_strategy_id") == selected_strategy_id,
        "operation_id_matches": payload.get("operation_id") == operation_id,
        "paper_only": payload.get("paper_only") is True,
        "one_strategy_only": payload.get("one_strategy_only") is True,
        "read_only": payload.get("read_only") is True,
        "allow_controlled_signal_readiness_validator_call": payload.get("allow_controlled_signal_readiness_validator_call") is True,
    }
    for key, value in exact_checks.items():
        if value is not True:
            blockers.append(f"proposed_4l3_signal_readiness_payload {key} failed")
    for key in DISABLED_PAYLOAD_FLAGS:
        if payload.get(key) is not False:
            blockers.append(f"proposed_4l3_signal_readiness_payload.{key} must be strict native boolean false")
        if payload.get(key) == "False":
            blockers.append(f"proposed_4l3_signal_readiness_payload.{key} must not be string False")
    input_checks, input_blockers = _signal_readiness_input_checks(payload.get("signal_readiness_inputs"), selected_strategy_id, operation_id)
    blockers.extend(input_blockers)
    return {
        "payload_valid": not blockers,
        "payload_json_safe": _primitive_json_safe(payload),
        **exact_checks,
        "signal_readiness_inputs": input_checks,
    }, _dedupe(blockers)


def _signal_readiness_input_checks(raw: Any, selected_strategy_id: str | None, operation_id: str | None) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    inputs = _mapping(raw)
    if not isinstance(raw, dict):
        return {"inputs_valid": False, "accepted_input_categories": []}, ["signal_readiness_inputs must be a native dict"]
    if not _primitive_json_safe(inputs):
        blockers.append("signal_readiness_inputs must be JSON-safe")
    exact_checks = {
        "selected_strategy_id_matches": inputs.get("selected_strategy_id") == selected_strategy_id,
        "operation_id_matches": inputs.get("operation_id") == operation_id,
        "read_only": inputs.get("read_only") is True,
        "paper_only": inputs.get("paper_only") is True,
        "one_strategy_only": inputs.get("one_strategy_only") is True,
    }
    for key, value in exact_checks.items():
        if value is not True:
            blockers.append(f"signal_readiness_inputs {key} failed")
    categories = _accepted_input_categories(inputs)
    if not categories:
        blockers.append("signal_readiness_inputs must contain at least one accepted input category")
    for key in DISABLED_INPUT_FLAGS:
        if inputs.get(key) is not False:
            blockers.append(f"signal_readiness_inputs.{key} must be strict native boolean false")
        if inputs.get(key) == "False":
            blockers.append(f"signal_readiness_inputs.{key} must not be string False")
    return {"inputs_valid": not blockers, **exact_checks, "accepted_input_categories": categories}, _dedupe(blockers)


def _accepted_input_categories(inputs: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    if inputs.get("market_data_inputs_available") is True or inputs.get("accepted_market_data_results"):
        categories.append("market_data")
    if inputs.get("contract_qualification_inputs_available") is True or inputs.get("accepted_contract_qualification_results"):
        categories.append("contract_qualification")
    for key, value in inputs.items():
        if key.startswith("accepted_") and value and key not in {"accepted_market_data_results", "accepted_contract_qualification_results"}:
            categories.append(key)
    return sorted(set(categories))


def _flow_checks(flow: Any) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if not isinstance(flow, list):
        return {"flow_ready": False, "step_count": 0, "sequence_numbers": [], "step_names": []}, ["proposed_4l3_validation_flow must be a native list"]
    if len(flow) != len(EXPECTED_4L3_FLOW_STEPS):
        blockers.append("proposed_4l3_validation_flow must contain exactly seven expected steps")
    sequence_numbers: list[Any] = []
    step_names: list[Any] = []
    for index, item in enumerate(flow or []):
        if not isinstance(item, dict):
            blockers.append("proposed_4l3_validation_flow contains malformed step")
            continue
        sequence_numbers.append(item.get("sequence_number"))
        step_names.append(item.get("step_name"))
        if item.get("sequence_number") != index + 1:
            blockers.append("proposed_4l3_validation_flow sequence_number values must be strictly sequential integers 1 through 7")
        if index < len(EXPECTED_4L3_FLOW_STEPS) and item.get("step_name") != EXPECTED_4L3_FLOW_STEPS[index]:
            blockers.append("proposed_4l3_validation_flow step_name values must exactly match the Stage 4L-2 contract")
        for key in FLOW_DISABLED_FLAGS:
            if item.get(key) is not False:
                blockers.append(f"proposed_4l3_validation_flow {key} must be false")
        if item.get("paper_only") is not True:
            blockers.append("proposed_4l3_validation_flow paper_only must be true")
    return {"flow_ready": not blockers, "step_count": len(flow), "sequence_numbers": sequence_numbers, "step_names": step_names}, _dedupe(blockers)


def _validator_availability_checks(validator: Any, allow_validator: bool) -> dict[str, Any]:
    method = getattr(validator, "validate_signal_readiness", None) if validator is not None else None
    has_method = callable(method)
    callable_validator = callable(validator)
    available = bool(allow_validator and validator is not None and (has_method or callable_validator))
    return {
        "validator_required": allow_validator,
        "validator_supplied": validator is not None,
        "has_validate_signal_readiness": has_method,
        "callable_validator": callable_validator,
        "validator_available": available,
    }


def _validator_availability_blockers(checks: dict[str, Any]) -> list[str]:
    if checks.get("validator_required") is not True:
        return ["Stage 4L-2 did not grant controlled signal-readiness validator permission"]
    if checks.get("validator_supplied") is not True:
        return ["controlled_signal_readiness_validator is required when Stage 4L-2 grants permission"]
    if checks.get("validator_available") is not True:
        return ["controlled_signal_readiness_validator must expose validate_signal_readiness or be callable"]
    return []


def _validator_result_checks(raw_result: Any) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if raw_result is None:
        return {}, ["controlled signal-readiness validator returned None"]
    if not isinstance(raw_result, dict):
        return {}, ["controlled signal-readiness validator result must be a native dict"]
    result = _copy_dict(raw_result)
    if not _primitive_json_safe(result):
        blockers.append("validator result must be JSON-safe")
    for key in REQUIRED_VALIDATOR_BOOL_FIELDS:
        if not isinstance(result.get(key), bool):
            blockers.append(f"validator result {key} must be native bool")
    if result.get("validator_called") is not True:
        blockers.append("validator result validator_called must be true")
    for key in DISABLED_VALIDATOR_FLAGS:
        if result.get(key) is not False:
            blockers.append(f"validator result {key} must be strict native boolean false")
        if result.get(key) == "False":
            blockers.append(f"validator result {key} must not be string False")
    for key in ("readiness_reasons", "blockers", "warnings"):
        value = result.get(key)
        if value is not None and (not isinstance(value, list) or not all(isinstance(item, str) for item in value)):
            blockers.append(f"validator result {key} must be a list of strings")
    if "signal_input_summary" in result and not isinstance(result.get("signal_input_summary"), dict):
        blockers.append("validator result signal_input_summary must be a native dict")
    disallowed_result_keys = (
        "executable_signals",
        "orders",
        "order_tickets",
        "broker_order_ids",
        "state_write_directives",
        "upstream_reports",
        "snapshots",
    )
    for key in disallowed_result_keys:
        if key in result:
            blockers.append(f"validator result must not include {key}")
    blockers.extend(_stale_key_blockers({"validator_result": result}))
    return result, _dedupe(blockers)


def _validator_execution_result(
    attempted: bool,
    validator_called: bool,
    success: bool,
    result: dict[str, Any],
    failure_reason: str | None,
    selected_strategy_ready_for_signal_evaluation: bool | None,
    signal_readiness_validated: bool,
) -> dict[str, Any]:
    return {
        "attempted": attempted,
        "validator_called": validator_called,
        "validator_method_name": "validate_signal_readiness",
        "direct_strategy_scan_called": False,
        "signal_execution_performed": False,
        "direct_ib_call_made": False,
        "provider_call_made": False,
        "success": success,
        "result": _copy_dict(result),
        "failure_reason": failure_reason,
        "selected_strategy_ready_for_signal_evaluation": selected_strategy_ready_for_signal_evaluation,
        "signal_readiness_validated": signal_readiness_validated,
    }


def _validator_call_trace(
    selected_strategy_id: str | None,
    operation_id: str | None,
    input_payload: dict[str, Any],
    validator_called: bool,
    success: bool,
    result: dict[str, Any],
    failure_reason: str | None,
    skipped_reason: str | None,
) -> dict[str, Any]:
    return {
        "validator_method": "validate_signal_readiness",
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "input_payload": _copy_dict(input_payload),
        "validator_called": validator_called,
        "direct_strategy_scan_called": False,
        "signal_execution_performed": False,
        "direct_ib_call_made": False,
        "provider_call_made": False,
        "success": success,
        "result": _copy_dict(result),
        "failure_reason": failure_reason,
        "skipped_reason": skipped_reason,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
        "order_submission_enabled": False,
        "state_write_enabled": False,
        "ledger_write_enabled": False,
    }


def _call_trace_blockers(trace: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for key in (
        "direct_strategy_scan_called",
        "signal_execution_performed",
        "direct_ib_call_made",
        "provider_call_made",
        "live_trading_enabled",
        "broker_submission_enabled",
        "order_submission_enabled",
        "state_write_enabled",
        "ledger_write_enabled",
    ):
        if trace.get(key) is not False:
            blockers.append(f"validator_call_trace {key} must be strict native boolean false")
        if trace.get(key) == "False":
            blockers.append(f"validator_call_trace {key} must not be string False")
    if not _primitive_json_safe(trace):
        blockers.append("validator_call_trace must be JSON-safe")
    for key in ("failure_reason", "skipped_reason"):
        value = trace.get(key)
        if isinstance(value, str) and (_has_raw_newline(value) or _has_memory_repr(value)):
            blockers.append(f"validator_call_trace {key} must be flat JSON-safe string")
    return blockers


def _result_acceptance(selected_strategy_id: str | None, operation_id: str | None, execution_safe: bool, result: dict[str, Any], rejected_reasons: list[str]) -> dict[str, Any]:
    result_blockers = _as_string_list(result.get("blockers"))
    result_warnings = _as_string_list(result.get("warnings"))
    signal_validated = result.get("signal_readiness_validated") is True
    strategy_ready = result.get("selected_strategy_ready_for_signal_evaluation") is True
    accepted = execution_safe and signal_validated and not rejected_reasons
    return {
        "accepted": accepted,
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "validator_execution_safe": execution_safe,
        "strategy_signal_ready": strategy_ready,
        "signal_readiness_validated": signal_validated,
        "accepted_result": _copy_dict(result) if accepted else {},
        "rejected_reasons": _dedupe(rejected_reasons),
        "blockers": result_blockers,
        "warnings": result_warnings,
    }


def _stage4l2_boundary_checks(report: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    boundary = _mapping(report.get("boundary_checks"))
    checks = {key: boundary.get(key) is True for key in BOUNDARY_KEYS}
    blockers = [f"Stage 4L-2 boundary check {key} failed" for key, value in checks.items() if value is not True]
    return checks, blockers


def _stage4l2_safety_blockers(report: dict[str, Any]) -> list[str]:
    safety = _mapping(report.get("safety_checks"))
    return [f"Stage 4L-2 safety check {key} failed" for key in SAFETY_KEYS if safety.get(key) is not True]


def _boundary_checks(allow_validator: bool, validator_called: bool) -> dict[str, bool]:
    return {
        "controlled_signal_readiness_validator_called_only_if_allowed": (not validator_called) or allow_validator is True,
        "no_direct_strategy_scan": True,
        "no_signal_execution": True,
        "no_provider_called": True,
        "no_market_data_fetched": True,
        "no_contracts_qualified": True,
        "no_direct_ib_call": True,
        "no_direct_reqMktData": True,
        "no_direct_qualifyContracts": True,
        "no_direct_reqContractDetails": True,
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


def _strategy_registry_checks(snapshot: Any, selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if snapshot is None:
        warnings.append("strategy registry snapshot missing; verify selected strategy eligibility before Stage 4L-4")
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
            selected_explicit_paper = _collect_strategy_records(strategies, candidates, selected_strategy_id, blockers)
        elif strategies is not None:
            malformed = True
    elif isinstance(snapshot, list):
        has_candidate_set = True
        if all(isinstance(item, str) for item in snapshot):
            candidates.update(item for item in snapshot if item)
        elif all(isinstance(item, dict) for item in snapshot):
            selected_explicit_paper = _collect_strategy_records(snapshot, candidates, selected_strategy_id, blockers)
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


def _collect_strategy_records(records: list[Any], candidates: set[str], selected_strategy_id: str | None, blockers: list[str]) -> bool | None:
    selected_explicit_paper = None
    for record in records:
        if not isinstance(record, dict):
            continue
        strategy_id = record.get("strategy_id")
        if isinstance(strategy_id, str) and strategy_id:
            if record.get("paper_eligible") is not False:
                candidates.add(strategy_id)
            if strategy_id == selected_strategy_id and "paper_eligible" in record:
                selected_explicit_paper = record.get("paper_eligible") is True
                if record.get("paper_eligible") is False:
                    blockers.append("strategy registry snapshot marks selected strategy paper_eligible false")
    return selected_explicit_paper


def _signal_schema_checks(snapshot: Any, selected_strategy_id: str | None, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    inputs = _mapping(payload.get("signal_readiness_inputs"))
    if snapshot is None:
        warnings.append("signal schema snapshot missing; verify Stage 4L-4 acceptance contract before proceeding")
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
                if item not in inputs:
                    missing_sections.append(item)
            if missing_sections:
                blockers.append("signal_readiness_inputs missing expected signal schema sections")
    for key in DISABLED_INPUT_FLAGS + ("broker_submission_enabled",):
        if snapshot.get(key) is True:
            blockers.append(f"signal schema {key} must not be true")
    return {
        "signal_schema_snapshot_present": True,
        "signal_schema_snapshot_valid": not blockers,
        "selected_strategy_matches": snapshot.get("selected_strategy_id") in (None, selected_strategy_id),
        "expected_input_sections_valid": expected_sections_valid,
        "missing_input_sections": sorted(missing_sections),
        "execution_permissions_disabled": not any(snapshot.get(key) is True for key in DISABLED_INPUT_FLAGS + ("broker_submission_enabled",)),
    }, blockers, warnings


def _activation_snapshot_group_checks(
    *,
    scheduler_activation_snapshot: Any,
    lifecycle_activation_snapshot: Any,
    activation_snapshot: Any,
    selected_strategy_id: str | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    checks: dict[str, Any] = {
        "scheduler_activation_snapshot_present": scheduler_activation_snapshot is not None,
        "lifecycle_activation_snapshot_present": lifecycle_activation_snapshot is not None,
        "activation_snapshot_present": activation_snapshot is not None,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    for label, snapshot in (
        ("scheduler_activation_snapshot", scheduler_activation_snapshot),
        ("lifecycle_activation_snapshot", lifecycle_activation_snapshot),
        ("activation_snapshot", activation_snapshot),
    ):
        if snapshot is None:
            continue
        if not isinstance(snapshot, dict):
            blockers.append(f"{label} must be a native dict")
            continue
        strategy_id = snapshot.get("selected_strategy_id") or snapshot.get("strategy_id")
        if strategy_id not in (None, selected_strategy_id):
            blockers.append(f"{label} selected strategy does not match")
        if snapshot.get("all_strategies_enabled") is True:
            blockers.append(f"{label} must not enable all strategies")
        checks[f"{label}_selected_strategy_matches"] = strategy_id in (None, selected_strategy_id)
    return checks, blockers, warnings


def _state_checks(snapshot: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if snapshot is None:
        warnings.append("state snapshot missing; verify no active halt or unresolved reconciliation before Stage 4L-4")
        return _default_state_checks(), blockers, warnings
    if not isinstance(snapshot, dict):
        return _default_state_checks(present=True, clean=False), ["state_snapshot must be a native dict"], warnings
    reconciliation_count = _first_present(snapshot, ("unresolved_needs_reconciliation_count", "needs_reconciliation_count"), 0)
    active_intents_count = snapshot.get("active_intents_count", 0)
    active_halt = bool(snapshot.get("active_halt"))
    if active_halt:
        blockers.append("state snapshot shows active halt")
    if _positive_number(reconciliation_count):
        blockers.append("state snapshot shows unresolved NEEDS_RECONCILIATION")
    if _positive_number(active_intents_count) and snapshot.get("active_intents_safe_for_enablement") is not True:
        blockers.append("state snapshot shows active intents not marked safe for enablement")
    if _positive_number(active_intents_count) and snapshot.get("active_intents_safe_for_enablement") is True:
        warnings.append("state snapshot has active intents marked safe for enablement; operator should verify before Stage 4L-4")
    return {
        "state_snapshot_present": True,
        "state_clean": not blockers,
        "active_halt": active_halt,
        "reconciliation_count": reconciliation_count,
        "active_intents_count": active_intents_count,
        "open_positions_count": snapshot.get("open_positions_count"),
    }, blockers, warnings


def _risk_checks(snapshot: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if snapshot is None:
        warnings.append("risk snapshot missing; verify core risk controls before Stage 4L-4")
        return _default_risk_checks(), blockers, warnings
    if not isinstance(snapshot, dict):
        return _default_risk_checks(present=True, valid=False), ["risk_snapshot must be a native dict"], warnings
    for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
        if snapshot.get(key) is not True:
            blockers.append(f"risk_snapshot {key} must be true")
    if "max_position_limit_available" in snapshot and snapshot.get("max_position_limit_available") is not True:
        blockers.append("risk_snapshot max_position_limit_available must be true when supplied")
    if snapshot.get("risk_bypass_enabled") is True:
        blockers.append("risk bypass must not be enabled")
    return {
        "risk_snapshot_present": True,
        "risk_controls_valid": not blockers,
        "kill_switch_available": snapshot.get("kill_switch_available") is True,
        "hard_halt_available": snapshot.get("hard_halt_available") is True,
        "daily_loss_limit_available": snapshot.get("daily_loss_limit_available") is True,
        "max_position_limit_available": snapshot.get("max_position_limit_available"),
        "risk_bypass_enabled": snapshot.get("risk_bypass_enabled") is True,
    }, blockers, warnings


def _scheduler_checks(snapshot: Any, selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if snapshot is None:
        warnings.append("scheduler snapshot missing; verify no broad scheduler automation before Stage 4L-4")
        return _default_scheduler_checks(), blockers, warnings
    if not isinstance(snapshot, dict):
        return _default_scheduler_checks(present=True, safe=False), ["scheduler_snapshot must be a native dict"], warnings
    if snapshot.get("all_strategy_scheduler_enabled") is True or snapshot.get("all_strategies_enabled") is True:
        blockers.append("broad/all-strategy scheduler automation must not be enabled")
    jobs = snapshot.get("jobs")
    job_count = 0
    if isinstance(jobs, list):
        for job in jobs or []:
            if not isinstance(job, dict):
                warnings.append("scheduler snapshot ignored malformed job")
                continue
            job_count += 1
            strategy_id = job.get("selected_strategy_id") or job.get("strategy_id")
            if strategy_id not in (None, selected_strategy_id):
                blockers.append("scheduler job strategy does not match selected strategy")
            for key in ("broker_submission_enabled", "live_trading_enabled", "all_strategies_enabled", "strategy_scan_execution_enabled", "signal_execution_enabled"):
                if job.get(key) is True:
                    blockers.append(f"scheduler job {key} must not be true")
    elif jobs is not None:
        blockers.append("scheduler_snapshot.jobs must be a list when supplied")
    return {"scheduler_snapshot_present": True, "scheduler_safe": not blockers, "job_count": job_count}, blockers, warnings


def _lifecycle_checks(snapshot: Any, selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if snapshot is None:
        warnings.append("lifecycle snapshot missing; verify no broad lifecycle automation before Stage 4L-4")
        return _default_lifecycle_checks(), blockers, warnings
    if not isinstance(snapshot, dict):
        return _default_lifecycle_checks(present=True, safe=False), ["lifecycle_snapshot must be a native dict"], warnings
    if snapshot.get("all_strategy_lifecycle_enabled") is True or snapshot.get("all_strategies_enabled") is True:
        blockers.append("broad/all-strategy lifecycle automation must not be enabled")
    strategy_id = snapshot.get("selected_strategy_id") or snapshot.get("strategy_id")
    if strategy_id not in (None, selected_strategy_id):
        blockers.append("lifecycle snapshot strategy does not match selected strategy")
    for key in ("broker_submission_enabled", "live_trading_enabled", "all_strategies_enabled", "lifecycle_transition_execution_enabled"):
        if snapshot.get(key) is True:
            blockers.append(f"lifecycle snapshot {key} must not be true")
    return {"lifecycle_snapshot_present": True, "lifecycle_safe": not blockers, "selected_strategy_matches": strategy_id in (None, selected_strategy_id)}, blockers, warnings


def _paper_broker_checks(snapshot: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if snapshot is None:
        warnings.append("paper broker snapshot missing; verify PAPER/non-live configuration before Stage 4L-4")
        return _default_paper_broker_checks(), blockers, warnings
    if not isinstance(snapshot, dict):
        return _default_paper_broker_checks(present=True, safe=False), ["paper_broker_snapshot must be a native dict"], warnings
    mode = snapshot.get("mode")
    if mode is not None and mode != "PAPER":
        blockers.append("paper broker snapshot mode must be PAPER")
    if snapshot.get("paper_trading") is False:
        blockers.append("paper broker snapshot paper_trading must not be false")
    port = snapshot.get("ibkr_port")
    if port is not None and port not in PAPER_IBKR_PORTS:
        blockers.append("paper broker snapshot ibkr_port must be a project paper port")
    if snapshot.get("live_trading_enabled") is True:
        blockers.append("paper broker snapshot live_trading_enabled must not be true")
    if snapshot.get("broker_submission_enabled") is True:
        blockers.append("paper broker snapshot broker_submission_enabled must not be true")
    return {"paper_broker_snapshot_present": True, "paper_broker_safe": not blockers, "mode": mode, "ibkr_port": port}, blockers, warnings


def _market_window_checks(snapshot: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if snapshot is None:
        warnings.append(MARKET_WINDOW_MANUAL_WARNING)
        return _default_market_window_checks(), blockers, warnings
    if not isinstance(snapshot, dict):
        return _default_market_window_checks(present=True, allowed=False), ["market_window_snapshot must be a native dict"], warnings
    if snapshot.get("allowed_to_schedule_paper_run") is False:
        blockers.append("market window explicitly blocks validator execution validation")
    if snapshot.get("market_open") is False:
        warnings.append("market is closed; safe validator output may still be reviewed but Stage 4L-4 should verify timing")
    if snapshot.get("is_trading_day") is False:
        warnings.append("not a trading day; safe validator output may still be reviewed but Stage 4L-4 should verify timing")
    return {
        "market_window_snapshot_present": True,
        "allowed_to_schedule_paper_run": snapshot.get("allowed_to_schedule_paper_run"),
        "market_open": snapshot.get("market_open"),
        "is_trading_day": snapshot.get("is_trading_day"),
    }, blockers, warnings


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    operation: dict[str, Any],
    stage4l2_gate_checks: dict[str, Any],
    validator_payload_checks: dict[str, Any],
    validator_availability_checks: dict[str, Any],
    validator_execution_result: dict[str, Any],
    validator_call_trace: dict[str, Any],
    signal_readiness_result_acceptance: dict[str, Any],
    boundary_checks: dict[str, bool],
    required_inputs_for_4l4: dict[str, Any],
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
        "stage4l3_controlled_signal_readiness_validator_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "operation": operation,
        "stage4l2_gate_checks": stage4l2_gate_checks,
        "validator_payload_checks": validator_payload_checks,
        "validator_availability_checks": validator_availability_checks,
        "validator_execution_result": validator_execution_result,
        "validator_call_trace": validator_call_trace,
        "signal_readiness_result_acceptance": signal_readiness_result_acceptance,
        "boundary_checks": boundary_checks,
        "required_inputs_for_4l4": required_inputs_for_4l4,
        "activation_snapshot_checks": activation_snapshot_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4l4": {
            "ready_to_build_signal_readiness_acceptance": ready,
            "next_recommended_phase": NEXT_RECOMMENDED_PHASE,
            "blockers": blockers,
            "warnings": warnings,
        },
        "recommendations": {
            "ordered_next_steps": list(ORDERED_NEXT_STEPS),
            "do_not_do_yet": list(DO_NOT_DO_YET),
        },
        "success": ready,
        "errors": errors,
        "warnings": warnings,
    }


def _selected_strategy(selected_strategy_id: str | None, paper_only: bool, one_strategy_only: bool) -> dict[str, Any]:
    return {"selected_strategy_id": selected_strategy_id, "paper_only": paper_only, "one_strategy_only": one_strategy_only}


def _operation(operation_id: str | None, operation_scope: str = "single_strategy_controlled_signal_readiness_validator") -> dict[str, Any]:
    return {
        "operation_id": operation_id,
        "operation_scope": operation_scope,
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }


def _required_inputs_for_4l4() -> dict[str, Any]:
    return {
        "accepted_stage4l3_report": True,
        "accepted_validator_result": True,
        "selected_strategy_id": "from Stage 4L-2 selected_strategy.selected_strategy_id",
        "operation_id": "from Stage 4L-2 operation.operation_id",
        "broker_submission_remains_separately_gated": True,
        "no_executable_signals_or_orders_created": True,
    }


def _safety_checks() -> dict[str, bool]:
    return {
        "no_live_trading": True,
        "no_all_strategy_enablement": True,
        "no_broker_submission_enabled": True,
        "no_direct_market_data": True,
        "no_direct_contract_qualification": True,
        "no_strategy_scan": True,
        "no_signal_execution": True,
        "no_order_submission": True,
        "no_intent_creation": True,
        "no_ticket_creation": True,
        "no_state_write": True,
        "no_ledger_write": True,
    }


def _safety_blockers(checks: dict[str, bool]) -> list[str]:
    return [f"safety check {key} failed" for key, value in checks.items() if value is not True]


def _default_artifact_checks() -> dict[str, bool]:
    return {
        "stage4l2_report_present": False,
        "stage4l2_report_ready": False,
        "selected_strategy_present": False,
        "operation_id_present": False,
        "proposed_4l3_payload_present": False,
        "proposed_4l3_validation_flow_present": False,
    }


def _default_payload_checks() -> dict[str, Any]:
    return {"payload_valid": False, "payload_json_safe": False, "signal_readiness_inputs": {"inputs_valid": False}}


def _default_strategy_registry_checks() -> dict[str, Any]:
    return {
        "strategy_registry_snapshot_present": False,
        "candidate_strategy_ids": [],
        "selected_strategy_present": None,
        "selected_strategy_paper_eligible": None,
        "all_strategies_enabled": False,
        "registry_parse_warning": False,
    }


def _default_signal_schema_checks(*, present: bool = False, valid: bool = True) -> dict[str, Any]:
    return {
        "signal_schema_snapshot_present": present,
        "signal_schema_snapshot_valid": valid,
        "selected_strategy_matches": None,
        "expected_input_sections_valid": None,
        "missing_input_sections": [],
        "execution_permissions_disabled": True,
    }


def _default_activation_snapshot_checks() -> dict[str, Any]:
    return {
        "scheduler_activation_snapshot_present": False,
        "lifecycle_activation_snapshot_present": False,
        "activation_snapshot_present": False,
    }


def _default_state_checks(*, present: bool = False, clean: bool = True) -> dict[str, Any]:
    return {"state_snapshot_present": present, "state_clean": clean, "active_halt": False, "reconciliation_count": 0, "active_intents_count": 0}


def _default_risk_checks(*, present: bool = False, valid: bool = True) -> dict[str, Any]:
    return {
        "risk_snapshot_present": present,
        "risk_controls_valid": valid,
        "kill_switch_available": None,
        "hard_halt_available": None,
        "daily_loss_limit_available": None,
        "max_position_limit_available": None,
        "risk_bypass_enabled": False,
    }


def _default_scheduler_checks(*, present: bool = False, safe: bool = True) -> dict[str, Any]:
    return {"scheduler_snapshot_present": present, "scheduler_safe": safe, "job_count": 0}


def _default_lifecycle_checks(*, present: bool = False, safe: bool = True) -> dict[str, Any]:
    return {"lifecycle_snapshot_present": present, "lifecycle_safe": safe, "selected_strategy_matches": None}


def _default_paper_broker_checks(*, present: bool = False, safe: bool = True) -> dict[str, Any]:
    return {"paper_broker_snapshot_present": present, "paper_broker_safe": safe, "mode": None, "ibkr_port": None}


def _default_market_window_checks(*, present: bool = False, allowed: bool | None = None) -> dict[str, Any]:
    return {"market_window_snapshot_present": present, "allowed_to_schedule_paper_run": allowed, "market_open": None, "is_trading_day": None}


def _selected_strategy_id(report: dict[str, Any]) -> str | None:
    value = _mapping(report.get("selected_strategy")).get("selected_strategy_id")
    return value if isinstance(value, str) and value else None


def _operation_id(report: dict[str, Any]) -> str | None:
    value = _mapping(report.get("operation")).get("operation_id")
    return value if isinstance(value, str) and value else None


def _stale_key_blockers(value: Any, path: str = "input") -> list[str]:
    blockers: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in STALE_4J_KEYS:
                blockers.append(f"stale Stage 4J schema key present at {path}.{key}")
            blockers.extend(_stale_key_blockers(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value or []):
            blockers.extend(_stale_key_blockers(item, f"{path}[{index}]"))
    return blockers


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _copy_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _copy_value(item) for key, item in value.items()}


def _copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _copy_dict(value)
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return [_copy_value(item) for item in value]
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def _primitive_json_safe(value: Any) -> bool:
    if isinstance(value, dict):
        return all(isinstance(key, str) and _primitive_json_safe(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_primitive_json_safe(item) for item in value)
    if isinstance(value, (str, int, bool)) or value is None:
        return True
    return isinstance(value, float) and math.isfinite(value)


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _generated_at(now_provider: Callable[[], Any] | None) -> str:
    if now_provider is None:
        return DEFAULT_GENERATED_AT
    value = now_provider()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _flat_exception(exc: BaseException) -> str:
    message = str(exc).replace("\r", " ").replace("\n", " ")
    message = re.sub(r"<([^<>]*?) at 0x[0-9A-Fa-f]+>", r"<\1>", message)
    return f"{type(exc).__name__}: {message}"


def _has_raw_newline(value: str) -> bool:
    return "\n" in value or "\r" in value


def _has_memory_repr(value: str) -> bool:
    return re.search(r"<[^<>]* at 0x[0-9A-Fa-f]+>", value) is not None


def _first_present(snapshot: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    for key in keys:
        if key in snapshot and snapshot.get(key) is not None:
            return snapshot.get(key)
    return default


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
