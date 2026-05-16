"""Pure Stage 4J-3 controlled scheduled PAPER operation dry-run report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from pathlib import Path
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
PAPER_IBKR_PORTS = {4004}
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding to 4J-4"
)
ORDERED_NEXT_STEPS = [
    "Build Stage 4J-4 controlled scheduled PAPER operation execution gate.",
    "Before any real selected-strategy operation, re-check activation artifacts, scheduler/lifecycle state, risk controls, paper broker config, market window, and state reconciliation.",
    "Keep live trading disabled.",
    "Keep all other strategies disabled.",
    "Keep broker submission separately gated until the explicit broker-submission phase.",
]
DO_NOT_DO_YET = [
    "Do not enable live trading.",
    "Do not enable all strategies.",
    "Do not place orders now.",
    "Do not run strategy scans now.",
    "Do not enable market data now.",
    "Do not enable contract qualification now.",
    "Do not change strategy thresholds.",
    "Do not change position sizing.",
    "Do not bypass risk controls.",
    "Do not enable broker submission broadly.",
]
DISABLED_COMPONENTS = [
    "live_trading",
    "all_strategy_automation",
    "broker_submission",
    "order_submission",
    "market_data_fetch",
    "contract_qualification",
    "strategy_scan_execution",
    "lifecycle_transition_execution",
    "direct_scheduler_registration",
    "direct_lifecycle_execution",
    "state_mutation",
    "ledger_write",
]
REQUIRED_INPUTS_FOR_4J4 = [
    "accepted_stage4j3_dry_run_report",
    "fresh_scheduler_activation_snapshot",
    "fresh_lifecycle_activation_snapshot",
    "fresh_activation_snapshot",
    "fresh_state_snapshot",
    "fresh_risk_snapshot",
    "fresh_scheduler_snapshot",
    "fresh_lifecycle_snapshot",
    "fresh_paper_broker_snapshot",
    "fresh_market_window_snapshot",
    "operator_approval_for_execution_gate_only",
]
EXPECTED_SIMULATED_RESULTS: dict[str, dict[str, Any]] = {
    "pre_operation_snapshot_check": {
        "simulated_pass": True,
        "checked_snapshots": [
            "state_snapshot",
            "risk_snapshot",
            "scheduler_snapshot",
            "lifecycle_snapshot",
            "paper_broker_snapshot",
            "market_window_snapshot",
        ],
    },
    "risk_gate_check": {
        "simulated_pass": True,
        "kill_switch_required": True,
        "hard_halt_required": True,
        "daily_loss_limit_required": True,
    },
    "activation_artifact_check": {
        "simulated_pass": True,
        "selected_strategy_id": None,
        "activation_required": True,
    },
    "scheduler_lifecycle_state_check": {
        "simulated_pass": True,
        "scheduler_checked": True,
        "lifecycle_checked": True,
    },
    "market_window_check": {
        "simulated_pass": True,
        "market_window_required_for_real_operation": True,
    },
    "selected_strategy_operation_preview": {
        "would_call_strategy": False,
        "strategy_call_placeholder": "not_executed_in_4J3",
    },
    "market_data_gate_preview": {
        "would_fetch_market_data": False,
        "market_data_placeholder": "not_fetched_in_4J3",
    },
    "contract_qualification_gate_preview": {
        "would_qualify_contracts": False,
        "contract_qualification_placeholder": "not_qualified_in_4J3",
    },
    "strategy_scan_gate_preview": {
        "would_call_strategy": False,
        "strategy_scan_placeholder": "not_executed_in_4J3",
    },
    "signal_to_intent_gate_preview": {
        "would_create_intent": False,
        "intent_placeholder": "not_created_in_4J3",
    },
    "order_ticket_gate_preview": {
        "would_create_ticket": False,
        "ticket_placeholder": "not_created_in_4J3",
    },
    "broker_submission_gate_preview": {
        "would_submit_order": False,
        "broker_submission_placeholder": "blocked_in_4J3",
    },
    "state_ledger_tracking_gate_preview": {
        "would_write_state": False,
        "would_write_ledger": False,
    },
    "alert_report_gate_preview": {
        "would_emit_alert": False,
        "alert_placeholder": "not_emitted_in_4J3",
    },
    "post_operation_reconciliation_gate_preview": {
        "would_reconcile_broker": False,
        "reconciliation_placeholder": "not_executed_in_4J3",
    },
}


def build_stage4j3_controlled_paper_operation_dry_run_report(
    *,
    stage4j2_operation_plan_report: dict | None,
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
    """Build a read-only dry run from an accepted Stage 4J-2 operation plan."""

    try:
        return _json_safe(
            _build_report(
                stage4j2_operation_plan_report=stage4j2_operation_plan_report,
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
    except Exception as exc:  # noqa: BLE001 - report boundary must stay data-only.
        message = f"unexpected Stage 4J-3 dry-run failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_default_selected_strategy(),
                operation=_default_operation(),
                dry_run_trace=[],
                dry_run_trace_checks=_default_trace_checks(),
                simulated_pre_operation_gates={},
                simulated_operation_results={},
                simulated_post_operation_checks=[],
                disabled_components_confirmed=_disabled_components_confirmed([]),
                activation_snapshot_checks=_default_activation_snapshot_checks(),
                state_checks=_default_state_checks(),
                risk_checks=_default_risk_checks(),
                scheduler_checks=_default_scheduler_checks(),
                lifecycle_checks=_default_lifecycle_checks(),
                paper_broker_checks=_default_paper_broker_checks(),
                market_window_checks=_default_market_window_checks(),
                safety_checks=_safety_checks({}, {}, []),
                ready=False,
                blockers=[message],
                warnings=[],
                errors=[message],
            )
        )


def _build_report(
    *,
    stage4j2_operation_plan_report: dict | None,
    scheduler_activation_snapshot: dict | None,
    lifecycle_activation_snapshot: dict | None,
    activation_snapshot: dict | None,
    state_snapshot: dict | None,
    risk_snapshot: dict | None,
    scheduler_snapshot: dict | None,
    lifecycle_snapshot: dict | None,
    paper_broker_snapshot: dict | None,
    market_window_snapshot: dict | None,
    now_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    generated_at = _generated_at(now_provider)
    report = stage4j2_operation_plan_report if isinstance(stage4j2_operation_plan_report, dict) else None
    data = _mapping(report)
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4j2_operation_plan_report is None:
        blockers.append("Stage 4J-2 operation plan report is missing")
    elif report is None:
        blockers.append("Stage 4J-2 operation plan report must be a dict")
        errors.append("Stage 4J-2 operation plan report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    selected_strategy_id = _selected_strategy_id(data)
    operation_plan = _mapping(data.get("operation_plan"))
    operation_id = _operation_id(operation_plan)
    artifact_checks = _artifact_checks(stage4j2_operation_plan_report, selected_strategy_id, operation_plan)
    blockers.extend(_artifact_blockers(artifact_checks, data))
    blockers.extend(_stage4j2_safety_blockers(_mapping(data.get("safety_checks"))))

    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    blockers.extend(selected_blockers)
    operation, operation_blockers = _operation_checks(operation_plan, operation_id)
    blockers.extend(operation_blockers)

    dry_run_trace, flow_blockers = _dry_run_trace(operation_plan, selected_strategy_id)
    blockers.extend(flow_blockers)
    trace_checks, trace_blockers = _dry_run_trace_checks(
        operation_plan.get("proposed_operation_flow"), dry_run_trace
    )
    blockers.extend(trace_blockers)

    activation_checks, activation_blockers, activation_warnings = _activation_snapshot_group_checks(
        scheduler_activation_snapshot=scheduler_activation_snapshot,
        lifecycle_activation_snapshot=lifecycle_activation_snapshot,
        activation_snapshot=activation_snapshot,
        selected_strategy_id=selected_strategy_id,
    )
    blockers.extend(activation_blockers)
    warnings.extend(activation_warnings)

    state_checks, state_blockers, state_warnings = _state_checks(state_snapshot)
    risk_checks, risk_blockers, risk_warnings = _risk_checks(risk_snapshot)
    scheduler_checks, scheduler_blockers, scheduler_warnings = _scheduler_checks(
        scheduler_snapshot, selected_strategy_id
    )
    lifecycle_checks, lifecycle_blockers, lifecycle_warnings = _lifecycle_checks(
        lifecycle_snapshot, selected_strategy_id
    )
    paper_broker_checks, broker_blockers, broker_warnings = _paper_broker_checks(
        paper_broker_snapshot
    )
    market_window_checks, market_blockers, market_warnings = _market_window_checks(
        market_window_snapshot
    )
    blockers.extend(
        state_blockers
        + risk_blockers
        + scheduler_blockers
        + lifecycle_blockers
        + broker_blockers
        + market_blockers
    )
    warnings.extend(
        state_warnings
        + risk_warnings
        + scheduler_warnings
        + lifecycle_warnings
        + broker_warnings
        + market_warnings
    )

    safety_checks = _safety_checks(
        _mapping(data.get("safety_checks")),
        _mapping(data.get("payload_checks")),
        [
            snapshot
            for snapshot in (
                scheduler_activation_snapshot,
                lifecycle_activation_snapshot,
                activation_snapshot,
                scheduler_snapshot,
                lifecycle_snapshot,
                paper_broker_snapshot,
            )
            if isinstance(snapshot, dict)
        ],
    )
    blockers.extend(_safety_blockers(safety_checks))

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    ready = not blocker_list and not error_list

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
        operation=operation,
        dry_run_trace=dry_run_trace,
        dry_run_trace_checks=trace_checks,
        simulated_pre_operation_gates=_simulated_pre_operation_gates(trace_checks),
        simulated_operation_results=_simulated_operation_results(dry_run_trace),
        simulated_post_operation_checks=_simulated_post_operation_checks(data),
        disabled_components_confirmed=_disabled_components_confirmed(dry_run_trace),
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


def _artifact_checks(
    report: dict | None, selected_strategy_id: str | None, operation_plan: dict[str, Any]
) -> dict[str, bool]:
    data = _mapping(report)
    readiness = _mapping(data.get("readiness_for_stage4j3"))
    flow = operation_plan.get("proposed_operation_flow")
    return {
        "stage4j2_report_present": isinstance(report, dict),
        "stage4j2_report_ready": (
            data.get("stage4j2_controlled_paper_operation_plan_report") is True
            and readiness.get("ready_to_build_controlled_paper_operation_dry_run") is True
            and data.get("success") is True
        ),
        "selected_strategy_present": isinstance(selected_strategy_id, str)
        and bool(selected_strategy_id),
        "operation_plan_present": bool(operation_plan),
        "operation_flow_present": isinstance(flow, list) and bool(flow),
        "operation_flow_valid": _operation_flow_valid(flow),
    }


def _artifact_blockers(checks: dict[str, bool], data: dict[str, Any]) -> list[str]:
    labels = {
        "stage4j2_report_present": "Stage 4J-2 operation plan report is missing",
        "stage4j2_report_ready": "Stage 4J-2 operation plan report is not ready for Stage 4J-3",
        "selected_strategy_present": "selected strategy is missing from accepted Stage 4J-2 report",
        "operation_plan_present": "operation plan is missing from Stage 4J-2 report",
        "operation_flow_present": "proposed operation flow is missing from Stage 4J-2 report",
        "operation_flow_valid": "proposed operation flow is malformed",
    }
    blockers = [label for key, label in labels.items() if checks.get(key) is not True]
    if _as_list(data.get("errors")):
        blockers.append("Stage 4J-2 report contains errors")
    return blockers


def _stage4j2_safety_blockers(safety_checks: dict[str, Any]) -> list[str]:
    labels = {
        "no_live_trading": "Stage 4J-2 safety checks do not confirm live trading is disabled",
        "no_all_strategy_enablement": "Stage 4J-2 safety checks do not confirm all-strategy automation is disabled",
        "no_broker_submission_enabled": "Stage 4J-2 safety checks do not confirm broker submission is disabled",
        "no_market_data": "Stage 4J-2 safety checks do not confirm market data is disabled",
        "no_contract_qualification": "Stage 4J-2 safety checks do not confirm contract qualification is disabled",
        "no_order_submission": "Stage 4J-2 safety checks do not confirm order submission is disabled",
        "no_strategy_scan_execution": "Stage 4J-2 safety checks do not confirm strategy scan execution is disabled",
        "no_lifecycle_transition_execution": "Stage 4J-2 safety checks do not confirm lifecycle transition execution is disabled",
        "no_direct_scheduler_registration": "Stage 4J-2 safety checks do not confirm direct scheduler registration is disabled",
        "no_direct_lifecycle_execution": "Stage 4J-2 safety checks do not confirm direct lifecycle execution is disabled",
        "no_state_write": "Stage 4J-2 safety checks do not confirm state writes are disabled",
        "no_ledger_write": "Stage 4J-2 safety checks do not confirm ledger writes are disabled",
    }
    return [label for key, label in labels.items() if safety_checks.get(key) is not True]


def _selected_strategy_checks(
    report: dict[str, Any], selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str]]:
    selected = _mapping(report.get("selected_strategy"))
    paper_only = selected.get("paper_only") is True
    one_strategy_only = selected.get("one_strategy_only") is True
    blockers: list[str] = []
    if not isinstance(selected_strategy_id, str) or not selected_strategy_id:
        blockers.append("selected strategy id is missing or invalid")
    if not paper_only:
        blockers.append("Stage 4J-2 selected strategy must be paper_only true")
    if not one_strategy_only:
        blockers.append("Stage 4J-2 selected strategy must be one_strategy_only true")
    return (
        {
            "selected_strategy_id": selected_strategy_id,
            "paper_only": paper_only,
            "one_strategy_only": one_strategy_only,
        },
        blockers,
    )


def _operation_checks(operation_plan: dict[str, Any], operation_id: str | None) -> tuple[dict[str, Any], list[str]]:
    window = _mapping(operation_plan.get("proposed_operation_window"))
    blockers: list[str] = []
    if operation_plan.get("available") is not True:
        blockers.append("operation plan available must be true")
    if not isinstance(operation_id, str) or not operation_id:
        blockers.append("operation_id is missing from Stage 4J-2 operation plan")
    if not window:
        blockers.append("proposed operation window is missing")
    elif window.get("would_execute_operation") is not False or window.get("live_trading_enabled") is not False:
        blockers.append("proposed operation window must remain disabled")
    return (
        {
            "operation_id": operation_id,
            "operation_scope": operation_plan.get("controlled_operation_scope"),
            "paper_only": True,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
        },
        blockers,
    )


def _dry_run_trace(
    operation_plan: dict[str, Any], selected_strategy_id: str | None
) -> tuple[list[dict[str, Any]], list[str]]:
    flow = operation_plan.get("proposed_operation_flow")
    blockers: list[str] = []
    trace: list[dict[str, Any]] = []
    if not isinstance(flow, list) or not flow:
        return trace, ["proposed operation flow is missing or not a list"]
    for index, step in enumerate(flow, start=1):
        if not isinstance(step, dict):
            blockers.append("proposed operation flow contains non-dict step")
            continue
        payload = step.get("payload")
        target_component = step.get("target_component")
        target_function = step.get("target_function")
        target = target_component if target_component is not None else target_function
        if "sequence_number" not in step:
            blockers.append("proposed operation flow step missing sequence_number")
        if not isinstance(target, str) or not target:
            blockers.append("proposed operation flow step missing target_component or target_function")
        if not isinstance(payload, dict):
            blockers.append("proposed operation flow step payload must be a dict")
            payload = {}
        elif not _primitive_json_safe(payload):
            blockers.append("proposed operation flow payload contains non-primitive JSON-unsafe data")
        for key in (
            "would_execute",
            "would_write_state",
            "would_write_ledger",
            "live_trading_enabled",
        ):
            if step.get(key) is not False:
                blockers.append(f"proposed operation flow step {key} must be strict boolean false")
        for key in (
            "would_call_strategy",
            "would_fetch_market_data",
            "would_qualify_contracts",
            "would_create_intent",
            "would_create_ticket",
            "would_submit_order",
        ):
            if key in step and step.get(key) is not False:
                blockers.append(f"proposed operation flow step {key} must be strict boolean false")
        stage = step.get("stage") if isinstance(step.get("stage"), str) else str(target or f"step_{index}")
        simulated_result = _simulated_result(stage, selected_strategy_id)
        if not _primitive_json_safe(simulated_result):
            blockers.append("dry-run simulated_result contains non-primitive JSON-unsafe data")
        item: dict[str, Any] = {
            "sequence_number": step.get("sequence_number"),
            "source_stage": "4J-2",
            "dry_run_stage": "4J-3",
            "input_payload": payload,
            "simulated_result": simulated_result,
            "would_execute": False,
            "would_write_state": False,
            "would_write_ledger": False,
            "paper_only": True,
            "live_trading_enabled": False,
            "status": "simulated",
        }
        if target_component is not None:
            item["target_component"] = target_component
        elif target_function is not None:
            item["target_function"] = target_function
        for key in (
            "would_call_strategy",
            "would_fetch_market_data",
            "would_qualify_contracts",
            "would_create_intent",
            "would_create_ticket",
            "would_submit_order",
        ):
            if key in step:
                item[key] = False
        trace.append(item)
    return trace, blockers


def _operation_flow_valid(flow: Any) -> bool:
    if not isinstance(flow, list) or not flow:
        return False
    for step in flow:
        if not isinstance(step, dict):
            return False
        if "sequence_number" not in step:
            return False
        if not (isinstance(step.get("target_component"), str) or isinstance(step.get("target_function"), str)):
            return False
        if not isinstance(step.get("payload"), dict) or not _primitive_json_safe(step.get("payload")):
            return False
    return True


def _simulated_result(stage: str, selected_strategy_id: str | None) -> dict[str, Any]:
    template = EXPECTED_SIMULATED_RESULTS.get(stage, {"simulated_pass": True})
    result = _copy_json_safe(template)
    if stage == "activation_artifact_check":
        result["selected_strategy_id"] = selected_strategy_id
    return result


def _dry_run_trace_checks(plan_flow: Any, trace: list[dict[str, Any]]) -> tuple[dict[str, bool], list[str]]:
    plan_steps = plan_flow if isinstance(plan_flow, list) else []
    plan_sequence = [
        step.get("sequence_number")
        for step in plan_steps
        if isinstance(step, dict)
    ]
    trace_sequence = [item.get("sequence_number") for item in trace]
    checks = {
        "trace_available": bool(trace),
        "trace_order_matches_plan": bool(trace) and trace_sequence == plan_sequence,
        "all_trace_items_simulated": bool(trace) and all(item.get("status") == "simulated" for item in trace),
        "no_strategy_call": all(item.get("would_call_strategy") is not True for item in trace),
        "no_market_data": all(item.get("would_fetch_market_data") is not True for item in trace),
        "no_contract_qualification": all(item.get("would_qualify_contracts") is not True for item in trace),
        "no_intent_created": all(item.get("would_create_intent") is not True for item in trace),
        "no_ticket_created": all(item.get("would_create_ticket") is not True for item in trace),
        "no_broker_submission": all(item.get("would_submit_order") is not True for item in trace),
        "no_state_write": all(item.get("would_write_state") is not True for item in trace),
        "no_ledger_write": all(item.get("would_write_ledger") is not True for item in trace),
        "payloads_json_safe": all(
            _primitive_json_safe(item.get("input_payload"))
            and _primitive_json_safe(item.get("simulated_result"))
            for item in trace
        ),
        "input_payloads_are_dicts": all(isinstance(item.get("input_payload"), dict) for item in trace),
        "simulated_results_are_dicts": all(isinstance(item.get("simulated_result"), dict) for item in trace),
    }
    blockers = [
        f"dry_run_trace check {key} must be true"
        for key, value in checks.items()
        if value is not True
    ]
    blockers.extend(_simulated_result_shape_blockers(trace))
    return checks, blockers


def _simulated_result_shape_blockers(trace: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for item in trace:
        stage = _stage_from_payload(item.get("input_payload"))
        if stage not in EXPECTED_SIMULATED_RESULTS:
            continue
        expected = _simulated_result(stage, _result_strategy_id(item.get("simulated_result")))
        result = item.get("simulated_result")
        if not isinstance(result, dict):
            blockers.append("dry_run_trace simulated_result must be a dict")
            continue
        if set(result.keys()) != set(expected.keys()):
            blockers.append(f"simulated_result keys for {stage} do not match expected shape")
            continue
        for key, expected_value in expected.items():
            if key == "selected_strategy_id":
                continue
            if result.get(key) != expected_value:
                blockers.append(f"simulated_result value for {stage}.{key} does not match expected placeholder")
            if key.endswith("_placeholder") and isinstance(result.get(key), dict):
                blockers.append(f"simulated_result placeholder for {stage}.{key} must be flat")
    return blockers


def _stage_from_payload(payload: Any) -> str | None:
    data = _mapping(payload)
    value = data.get("stage")
    return value if isinstance(value, str) else None


def _result_strategy_id(result: Any) -> str | None:
    value = _mapping(result).get("selected_strategy_id")
    return value if isinstance(value, str) else None


def _simulated_pre_operation_gates(trace_checks: dict[str, bool]) -> dict[str, Any]:
    return {
        "stage4j2_plan_consumed": True,
        "dry_run_trace_available": trace_checks.get("trace_available") is True,
        "trace_order_preserved": trace_checks.get("trace_order_matches_plan") is True,
        "broker_submission_disabled": True,
        "live_trading_disabled": True,
        "all_strategy_automation_disabled": True,
    }


def _simulated_operation_results(trace: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "steps_simulated": len(trace),
        "all_statuses_simulated": bool(trace) and all(item.get("status") == "simulated" for item in trace),
        "broker_submission_placeholder": "blocked_in_4J3",
        "state_write_placeholder": "not_written_in_4J3",
        "ledger_write_placeholder": "not_written_in_4J3",
    }


def _simulated_post_operation_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    plan = _mapping(report.get("operation_plan"))
    checks = _as_list(plan.get("proposed_post_operation_checks"))
    return [
        {
            "source_stage": "4J-2",
            "dry_run_stage": "4J-3",
            "check": _mapping(item).get("check") if isinstance(item, dict) else None,
            "status": "simulated",
            "would_execute": False,
            "would_write_state": False,
            "would_write_ledger": False,
        }
        for item in checks
        if isinstance(item, dict)
    ]


def _disabled_components_confirmed(trace: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "components": list(DISABLED_COMPONENTS),
        "all_confirmed_disabled": bool(trace)
        and all(
            item.get("would_execute") is False
            and item.get("would_write_state") is False
            and item.get("would_write_ledger") is False
            and item.get("live_trading_enabled") is False
            for item in trace
        ),
    }


def _activation_snapshot_group_checks(
    *,
    scheduler_activation_snapshot: dict | None,
    lifecycle_activation_snapshot: dict | None,
    activation_snapshot: dict | None,
    selected_strategy_id: str | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    warnings: list[str] = []
    for builder, snapshot in (
        (_scheduler_activation_snapshot_checks, scheduler_activation_snapshot),
        (_lifecycle_activation_snapshot_checks, lifecycle_activation_snapshot),
        (_activation_snapshot_checks, activation_snapshot),
    ):
        current_checks, current_blockers, current_warnings = builder(snapshot, selected_strategy_id)
        checks.update(current_checks)
        blockers.extend(current_blockers)
        warnings.extend(current_warnings)
    return checks, blockers, warnings


def _scheduler_activation_snapshot_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    return _activation_artifact_snapshot_checks(
        snapshot,
        selected_strategy_id,
        prefix="scheduler_activation",
        record_key="scheduler_activation_record",
        list_key="scheduler_activations",
        execution_key="strategy_scan_execution_enabled",
    )


def _lifecycle_activation_snapshot_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    return _activation_artifact_snapshot_checks(
        snapshot,
        selected_strategy_id,
        prefix="lifecycle_activation",
        record_key="lifecycle_activation_record",
        list_key="lifecycle_activations",
        execution_key="lifecycle_transition_execution_enabled",
    )


def _activation_artifact_snapshot_checks(
    snapshot: dict | None,
    selected_strategy_id: str | None,
    *,
    prefix: str,
    record_key: str,
    list_key: str,
    execution_key: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append(f"{prefix} snapshot missing; 4J-4 must verify activation artifact")
        return {f"{prefix}_snapshot_present": False, f"{prefix}_snapshot_matches": True}, blockers, warnings
    for record in _explicit_records(data, record_key, list_key):
        if not isinstance(record, dict):
            warnings.append(f"malformed {prefix} snapshot entry ignored")
            continue
        if record.get("selected_strategy_id") not in (None, selected_strategy_id):
            blockers.append(f"{prefix} snapshot selected_strategy_id does not match")
        if record.get("paper_only") is False:
            blockers.append(f"{prefix} snapshot paper_only contradicts selected strategy")
        if record.get("one_strategy_only") is False:
            blockers.append(f"{prefix} snapshot one_strategy_only contradicts selected strategy")
        for key in (
            "live_trading_enabled",
            "all_strategies_enabled",
            "broker_submission_enabled",
            execution_key,
        ):
            if record.get(key) is True:
                blockers.append(f"{prefix} snapshot {key} contradicts Stage 4J-3 safety")
    return (
        {f"{prefix}_snapshot_present": True, f"{prefix}_snapshot_matches": not blockers},
        blockers,
        warnings,
    )


def _activation_snapshot_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("activation snapshot missing; 4J-4 must verify one-strategy activation state")
        return {"activation_snapshot_present": False, "activation_snapshot_matches": True}, blockers, warnings
    active_ids = _as_string_list(data.get("active_strategy_ids"))
    if len(active_ids) > 1:
        blockers.append("activation snapshot shows more than one active strategy")
    elif active_ids and active_ids != [selected_strategy_id]:
        blockers.append("activation snapshot active_strategy_ids do not contain exactly selected strategy")
    for record in _explicit_records(data, "activation_record", "activations"):
        if not isinstance(record, dict):
            warnings.append("malformed activation snapshot entry ignored")
            continue
        if record.get("selected_strategy_id") not in (None, selected_strategy_id):
            blockers.append("activation snapshot selected_strategy_id does not match")
        if record.get("paper_only") is False:
            blockers.append("activation snapshot paper_only contradicts selected strategy")
        for key in ("live_trading_enabled", "all_strategies_enabled", "broker_submission_enabled"):
            if record.get(key) is True:
                blockers.append(f"activation snapshot {key} contradicts Stage 4J-3 safety")
    return {"activation_snapshot_present": True, "activation_snapshot_matches": not blockers}, blockers, warnings


def _state_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    active_intents_count = _safe_int(data.get("active_intents_count"))
    active_intents_safe = data.get("active_intents_safe_for_enablement") is True
    unresolved_count = _safe_int(
        _first_present(
            data.get("unresolved_needs_reconciliation_count"),
            data.get("needs_reconciliation_count"),
            default=0,
        )
    )
    active_halt = bool(data.get("active_halt")) if present else False
    checks = {
        "state_snapshot_present": present,
        "active_halt": active_halt,
        "unresolved_needs_reconciliation_count": unresolved_count,
        "active_intents_count": active_intents_count,
        "active_intents_safe_for_enablement": active_intents_safe,
        "open_positions_count": _safe_int(data.get("open_positions_count")),
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("state snapshot missing; 4J-4 must verify halt, reconciliation, intents, and positions")
    if active_halt:
        blockers.append("active halt is present")
    if unresolved_count > 0:
        blockers.append("unresolved NEEDS_RECONCILIATION is present")
    if active_intents_count > 0 and not active_intents_safe:
        blockers.append("unsafe active intents are present")
    if active_intents_count > 0 and active_intents_safe:
        warnings.append("active intents present but explicitly marked safe for enablement")
    return checks, blockers, warnings


def _risk_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
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
        warnings.append("risk snapshot missing; 4J-4 must verify kill switch, hard halt, and daily loss controls")
        return checks, blockers, warnings
    for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
        if checks[key] is not True:
            blockers.append(f"risk snapshot {key} must be true")
    if data.get("max_position_limit_available") is False:
        blockers.append("risk snapshot max_position_limit_available must not be false when supplied")
    if checks["risk_bypass_enabled"] is True:
        blockers.append("risk bypass is enabled")
    return checks, blockers, warnings


def _scheduler_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    jobs = _as_list(data.get("jobs")) + _as_list(data.get("scheduled_jobs"))
    matching_jobs = [job for job in jobs if isinstance(job, dict) and _job_strategy_id(job) == selected_strategy_id]
    selected_job_matches = True
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("scheduler snapshot missing; 4J-4 must verify scheduler state")
    all_strategy_enabled = any(
        data.get(key) is True
        for key in ("all_strategy_scheduler_enabled", "all_strategies_enabled", "broad_scheduler_enabled")
    )
    scheduler_enabled = data.get("scheduler_automation_enabled") is True or bool(matching_jobs)
    scan_enabled = data.get("strategy_scan_execution_enabled") is True
    for job in matching_jobs:
        selected_job_matches = selected_job_matches and _scheduler_job_safe(job, selected_strategy_id)
        scan_enabled = scan_enabled or job.get("strategy_scan_execution_enabled") is True
    if all_strategy_enabled:
        blockers.append("broad/all-strategy scheduler automation enabled")
    if matching_jobs and not selected_job_matches:
        blockers.append("selected strategy scheduler job does not match Stage 4J-3 safety constraints")
    if scan_enabled:
        blockers.append("scheduler snapshot strategy_scan_execution_enabled must be false")
    return (
        {
            "scheduler_snapshot_present": present,
            "scheduler_already_enabled": scheduler_enabled,
            "all_strategy_scheduler_enabled": all_strategy_enabled,
            "selected_strategy_job_matches": selected_job_matches,
            "strategy_scan_execution_enabled": scan_enabled,
        },
        blockers,
        warnings,
    )


def _lifecycle_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; 4J-4 must verify lifecycle state")
    broad_enabled = any(
        data.get(key) is True
        for key in ("all_strategy_lifecycle_enabled", "all_strategies_enabled", "broad_lifecycle_enabled")
    )
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
    return (
        {
            "lifecycle_snapshot_present": present,
            "lifecycle_already_enabled": lifecycle_enabled,
            "lifecycle_matches_selected_strategy": matches,
            "lifecycle_transition_execution_enabled": transition_enabled,
        },
        blockers,
        warnings,
    )


def _paper_broker_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    mode = data.get("mode")
    paper_trading = data.get("paper_trading")
    ibkr_port = data.get("ibkr_port")
    live_enabled = data.get("live_trading_enabled") is True
    broker_enabled = data.get("broker_submission_enabled") is True
    paper_config_valid = True
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("paper broker snapshot missing; 4J-4 must verify PAPER broker config")
    if present and mode is not None and str(mode).upper() != "PAPER":
        paper_config_valid = False
        blockers.append("paper broker snapshot mode must be PAPER")
    if present and paper_trading is False:
        paper_config_valid = False
        blockers.append("paper broker snapshot paper_trading must not be false")
    if present and ibkr_port is not None and ibkr_port not in PAPER_IBKR_PORTS:
        paper_config_valid = False
        blockers.append("paper broker snapshot ibkr_port must use the project PAPER port")
    if live_enabled:
        paper_config_valid = False
        blockers.append("paper broker snapshot live_trading_enabled must be false")
    if broker_enabled:
        paper_config_valid = False
        blockers.append("paper broker snapshot broker_submission_enabled must remain false")
    return (
        {
            "paper_broker_snapshot_present": present,
            "mode": mode,
            "paper_trading": paper_trading,
            "ibkr_port": ibkr_port,
            "paper_config_valid": paper_config_valid,
            "live_trading_enabled": live_enabled,
            "broker_submission_enabled": broker_enabled,
        },
        blockers,
        warnings,
    )


def _market_window_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    allowed = data.get("allowed_to_schedule_paper_run") if present else None
    is_trading_day = data.get("is_trading_day") if present else None
    market_open = data.get("market_open") if present else None
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append(MARKET_WINDOW_MANUAL_WARNING)
    if allowed is False:
        blockers.append("market window snapshot explicitly disallows scheduling a paper run")
    if market_open is False:
        warnings.append("market is currently closed; dry-run validation may continue but 4J-4 must verify timing")
    if is_trading_day is False:
        warnings.append("snapshot is not a trading day; dry-run validation may continue but 4J-4 must verify timing")
    return (
        {
            "market_window_snapshot_present": present,
            "allowed_to_schedule_paper_run": allowed,
            "is_trading_day": is_trading_day,
            "market_open": market_open,
            "reason": data.get("reason") if present else None,
        },
        blockers,
        warnings,
    )


def _safety_checks(
    report_safety: dict[str, Any], payload_checks: dict[str, Any], snapshots: list[dict[str, Any]]
) -> dict[str, bool]:
    return {
        "no_live_trading": report_safety.get("no_live_trading") is not False and _none_true("live_trading_enabled", *snapshots),
        "no_all_strategy_enablement": report_safety.get("no_all_strategy_enablement") is not False and _none_true("all_strategies_enabled", *snapshots),
        "no_broker_submission_enabled": report_safety.get("no_broker_submission_enabled") is not False and _none_true("broker_submission_enabled", *snapshots),
        "no_market_data": report_safety.get("no_market_data") is not False and payload_checks.get("market_data_disabled") is not False and _none_true("market_data_enabled", *snapshots),
        "no_contract_qualification": report_safety.get("no_contract_qualification") is not False and payload_checks.get("contract_qualification_disabled") is not False and _none_true("contract_qualification_enabled", *snapshots),
        "no_order_submission": report_safety.get("no_order_submission") is not False and _none_true("order_submission_enabled", *snapshots),
        "no_strategy_scan_execution": report_safety.get("no_strategy_scan_execution") is not False and payload_checks.get("strategy_scan_execution_disabled") is not False and _none_true("strategy_scan_execution_enabled", *snapshots),
        "no_lifecycle_transition_execution": report_safety.get("no_lifecycle_transition_execution") is not False and payload_checks.get("lifecycle_transition_execution_disabled") is not False and _none_true("lifecycle_transition_execution_enabled", *snapshots),
        "no_direct_scheduler_registration": report_safety.get("no_direct_scheduler_registration") is not False,
        "no_direct_lifecycle_execution": report_safety.get("no_direct_lifecycle_execution") is not False,
        "no_state_write": report_safety.get("no_state_write") is not False,
        "no_ledger_write": report_safety.get("no_ledger_write") is not False,
    }


def _safety_blockers(checks: dict[str, bool]) -> list[str]:
    labels = {
        "no_live_trading": "live trading safety flag is enabled",
        "no_all_strategy_enablement": "all-strategy safety flag is enabled",
        "no_broker_submission_enabled": "broker submission safety flag is enabled",
        "no_market_data": "market data safety flag is enabled",
        "no_contract_qualification": "contract qualification safety flag is enabled",
        "no_order_submission": "order submission safety flag is enabled",
        "no_strategy_scan_execution": "strategy scan execution safety flag is enabled",
        "no_lifecycle_transition_execution": "lifecycle transition execution safety flag is enabled",
        "no_direct_scheduler_registration": "direct scheduler registration safety flag is enabled",
        "no_direct_lifecycle_execution": "direct lifecycle execution safety flag is enabled",
        "no_state_write": "state write safety flag is enabled",
        "no_ledger_write": "ledger write safety flag is enabled",
    }
    return [label for key, label in labels.items() if checks.get(key) is not True]


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    operation: dict[str, Any],
    dry_run_trace: list[dict[str, Any]],
    dry_run_trace_checks: dict[str, Any],
    simulated_pre_operation_gates: dict[str, Any],
    simulated_operation_results: dict[str, Any],
    simulated_post_operation_checks: list[dict[str, Any]],
    disabled_components_confirmed: dict[str, Any],
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
        "stage4j3_controlled_paper_operation_dry_run_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "controlled_operation_scope": operation.get("operation_scope"),
        "operation_id": operation.get("operation_id"),
        "operation": operation,
        "dry_run_trace": dry_run_trace,
        "dry_run_trace_checks": dry_run_trace_checks,
        "simulated_pre_operation_gates": simulated_pre_operation_gates,
        "simulated_operation_results": simulated_operation_results,
        "simulated_post_operation_checks": simulated_post_operation_checks,
        "disabled_components_confirmed": disabled_components_confirmed,
        "required_inputs_for_4J4": list(REQUIRED_INPUTS_FOR_4J4),
        "activation_snapshot_checks": activation_snapshot_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4j4": {
            "ready_to_build_controlled_paper_operation_execution_gate": ready,
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


def _selected_strategy_id(report: dict[str, Any]) -> str | None:
    selected = _mapping(report.get("selected_strategy"))
    value = selected.get("selected_strategy_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _operation_id(operation_plan: dict[str, Any]) -> str | None:
    value = operation_plan.get("operation_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _explicit_records(data: dict[str, Any], record_key: str, list_key: str) -> list[Any]:
    records: list[Any] = []
    if record_key in data:
        records.append(data.get(record_key))
    records.extend(_as_list(data.get(list_key)))
    explicit_keys = {
        "selected_strategy_id",
        "paper_only",
        "one_strategy_only",
        "live_trading_enabled",
        "all_strategies_enabled",
        "broker_submission_enabled",
        "strategy_scan_execution_enabled",
        "lifecycle_transition_execution_enabled",
    }
    if any(key in data for key in explicit_keys):
        records.append(data)
    return records


def _job_strategy_id(job: dict[str, Any]) -> Any:
    return _first_present(job.get("selected_strategy_id"), job.get("strategy_id"), default=None)


def _scheduler_job_safe(job: dict[str, Any], selected_strategy_id: str | None) -> bool:
    required_false = (
        "broker_submission_enabled",
        "live_trading_enabled",
        "all_strategies_enabled",
        "strategy_scan_execution_enabled",
    )
    return (
        _job_strategy_id(job) == selected_strategy_id
        and job.get("paper_only") is not False
        and job.get("scheduler_job_scope") in (None, "single_strategy")
        and all(job.get(key) is False for key in required_false)
    )


def _none_true(key: str, *mappings: dict[str, Any]) -> bool:
    for mapping in mappings:
        if mapping.get(key) is True:
            return False
        for record_key in ("scheduler_activation_record", "lifecycle_activation_record", "activation_record"):
            record = _mapping(mapping.get(record_key))
            if record.get(key) is True:
                return False
        for list_key in ("scheduler_activations", "lifecycle_activations", "activations", "jobs", "scheduled_jobs"):
            for item in _as_list(mapping.get(list_key)):
                if isinstance(item, dict) and item.get(key) is True:
                    return False
    return True


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_mapping(value: Any) -> tuple[bool, dict[str, Any]]:
    return isinstance(value, dict), value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_string_list(value: Any) -> list[str]:
    return [item for item in _as_list(value) if isinstance(item, str)]


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


def _primitive_json_safe(value: Any) -> bool:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return not (isinstance(value, float) and not math.isfinite(value))
    if isinstance(value, list):
        return all(_primitive_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _primitive_json_safe(item) for key, item in value.items())
    if isinstance(value, (datetime, date, tuple, Decimal, Path, bytes)) or callable(value):
        return False
    return False


def _copy_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_json_safe(item) for item in value]
    return value


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
        "stage4j2_report_present": False,
        "stage4j2_report_ready": False,
        "selected_strategy_present": False,
        "operation_plan_present": False,
        "operation_flow_present": False,
        "operation_flow_valid": False,
    }


def _default_selected_strategy() -> dict[str, Any]:
    return {"selected_strategy_id": None, "paper_only": False, "one_strategy_only": False}


def _default_operation() -> dict[str, Any]:
    return {
        "operation_id": None,
        "operation_scope": None,
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }


def _default_trace_checks() -> dict[str, bool]:
    return {
        "trace_available": False,
        "trace_order_matches_plan": False,
        "all_trace_items_simulated": False,
        "no_strategy_call": True,
        "no_market_data": True,
        "no_contract_qualification": True,
        "no_intent_created": True,
        "no_ticket_created": True,
        "no_broker_submission": True,
        "no_state_write": True,
        "no_ledger_write": True,
        "payloads_json_safe": True,
        "input_payloads_are_dicts": True,
        "simulated_results_are_dicts": True,
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
