"""Pure Stage 4I-3 scheduled PAPER run dry-run report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
PAPER_IBKR_PORTS = {4004}
PRE_RUN_CHECKS_REQUIRED = [
    "state_snapshot",
    "risk_snapshot",
    "scheduler_snapshot",
    "lifecycle_snapshot",
    "paper_broker_snapshot",
    "market_window_snapshot",
]
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding to 4I-4"
)
ORDERED_NEXT_STEPS = [
    "Build Stage 4I-4 one-strategy scheduler/lifecycle activation gate.",
    "Before 4I-4, re-check state, activation artifact, risk controls, scheduler, lifecycle, paper broker config, and market window.",
    "Keep live trading disabled.",
    "Keep all other strategies disabled.",
    "Keep broker submission separately gated until the scheduled run phase explicitly permits it.",
]
DO_NOT_DO_YET = [
    "Do not enable live trading.",
    "Do not enable all strategies.",
    "Do not place orders now.",
    "Do not change strategy thresholds.",
    "Do not change position sizing.",
    "Do not bypass risk controls.",
    "Do not register scheduler jobs now.",
]


def build_stage4i3_scheduled_run_dry_run_report(
    *,
    stage4i2_run_plan_report: dict | None,
    activation_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    market_window_snapshot: dict | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a deterministic read-only trace for the planned 4I-2 run path."""

    try:
        return _json_safe(
            _build_report(
                stage4i2_run_plan_report=stage4i2_run_plan_report,
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
        message = f"unexpected scheduled run dry-run failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_default_selected_strategy(),
                schedule_checks=_default_schedule_checks(),
                flow_checks=_default_flow_checks(),
                dry_run_trace=[],
                dry_run_trace_checks=_default_trace_checks(),
                activation_checks=_default_activation_checks(),
                state_checks=_default_state_checks(),
                risk_checks=_default_risk_checks(),
                scheduler_checks=_default_scheduler_checks(),
                lifecycle_checks=_default_lifecycle_checks(),
                paper_broker_checks=_default_paper_broker_checks(),
                market_window_checks=_default_market_window_checks(),
                safety_checks=_safety_checks({}),
                ready=False,
                blockers=[message],
                warnings=[],
                errors=[message],
            )
        )


def _build_report(
    *,
    stage4i2_run_plan_report: dict | None,
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
    i2 = stage4i2_run_plan_report if isinstance(stage4i2_run_plan_report, dict) else None
    data = _mapping(i2)
    run_plan = _mapping(data.get("run_plan"))
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4i2_run_plan_report is None:
        blockers.append("Stage 4I-2 run plan report is missing")
    elif i2 is None:
        blockers.append("Stage 4I-2 run plan report must be a dict")
        errors.append("Stage 4I-2 run plan report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    artifact_checks = _artifact_checks(stage4i2_run_plan_report)
    blockers.extend(_artifact_blockers(artifact_checks, data))
    blockers.extend(_stage4i2_safety_blockers(_mapping(data.get("safety_checks"))))

    selected_strategy_id = _selected_strategy_id(data)
    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    blockers.extend(selected_blockers)

    schedule_checks, schedule_blockers = _schedule_checks(run_plan)
    blockers.extend(schedule_blockers)

    flow = run_plan.get("proposed_execution_flow") if isinstance(run_plan, dict) else None
    flow_checks, flow_blockers = _flow_checks(flow)
    blockers.extend(flow_blockers)

    activation_checks, activation_blockers, activation_warnings = _activation_checks(
        activation_snapshot, selected_strategy_id
    )
    blockers.extend(activation_blockers)
    warnings.extend(activation_warnings)

    state_checks, state_blockers, state_warnings = _state_checks(state_snapshot)
    blockers.extend(state_blockers)
    warnings.extend(state_warnings)

    risk_checks, risk_blockers, risk_warnings = _risk_checks(risk_snapshot)
    blockers.extend(risk_blockers)
    warnings.extend(risk_warnings)

    scheduler_checks, scheduler_blockers, scheduler_warnings = _scheduler_checks(
        scheduler_snapshot, selected_strategy_id
    )
    blockers.extend(scheduler_blockers)
    warnings.extend(scheduler_warnings)

    lifecycle_checks, lifecycle_blockers, lifecycle_warnings = _lifecycle_checks(
        lifecycle_snapshot
    )
    blockers.extend(lifecycle_blockers)
    warnings.extend(lifecycle_warnings)

    paper_broker_checks, broker_blockers, broker_warnings = _paper_broker_checks(
        paper_broker_snapshot
    )
    blockers.extend(broker_blockers)
    warnings.extend(broker_warnings)

    market_window_checks, market_blockers, market_warnings = _market_window_checks(
        market_window_snapshot
    )
    blockers.extend(market_blockers)
    warnings.extend(market_warnings)

    trace = _dry_run_trace(
        flow if isinstance(flow, list) else [],
        selected_strategy_id,
        activation_checks=activation_checks,
        state_checks=state_checks,
        risk_checks=risk_checks,
        scheduler_checks=scheduler_checks,
        lifecycle_checks=lifecycle_checks,
        paper_broker_checks=paper_broker_checks,
        market_window_checks=market_window_checks,
    )
    trace_checks = _trace_checks(flow if isinstance(flow, list) else [], trace, flow_checks)
    blockers.extend(_trace_blockers(trace_checks))

    safety_checks = _safety_checks(
        data,
        activation_snapshot if isinstance(activation_snapshot, dict) else {},
        scheduler_snapshot if isinstance(scheduler_snapshot, dict) else {},
        lifecycle_snapshot if isinstance(lifecycle_snapshot, dict) else {},
        paper_broker_snapshot if isinstance(paper_broker_snapshot, dict) else {},
        {"trace": trace},
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
        schedule_checks=schedule_checks,
        flow_checks=flow_checks,
        dry_run_trace=trace,
        dry_run_trace_checks=trace_checks,
        activation_checks=activation_checks,
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


def _artifact_checks(report: dict | None) -> dict[str, bool]:
    data = _mapping(report)
    readiness = _mapping(data.get("readiness_for_stage4i3"))
    run_plan = _mapping(data.get("run_plan"))
    schedule = _mapping(run_plan.get("proposed_schedule"))
    flow = run_plan.get("proposed_execution_flow")
    return {
        "stage4i2_report_present": isinstance(report, dict),
        "stage4i2_report_ready": (
            data.get("stage4i2_scheduled_paper_run_plan_report") is True
            and readiness.get("ready_to_build_scheduled_run_dry_run") is True
            and data.get("success") is True
        ),
        "selected_strategy_present": _selected_strategy_id(data) is not None,
        "run_plan_available": run_plan.get("available") is True,
        "proposed_schedule_valid": bool(schedule),
        "proposed_execution_flow_valid": isinstance(flow, list) and bool(flow),
    }


def _artifact_blockers(checks: dict[str, bool], data: dict[str, Any]) -> list[str]:
    labels = {
        "stage4i2_report_present": "Stage 4I-2 run plan report is missing",
        "stage4i2_report_ready": "Stage 4I-2 run plan report is not ready for Stage 4I-3",
        "selected_strategy_present": "selected strategy is missing from accepted Stage 4I-2 report",
        "run_plan_available": "run plan is not available",
        "proposed_schedule_valid": "proposed schedule is missing or malformed",
        "proposed_execution_flow_valid": "proposed execution flow is missing or malformed",
    }
    blockers = [label for key, label in labels.items() if checks.get(key) is not True]
    if _as_list(data.get("errors")):
        blockers.append("Stage 4I-2 report contains errors")
    return blockers


def _stage4i2_safety_blockers(safety_checks: dict[str, Any]) -> list[str]:
    labels = {
        "no_live_trading": "Stage 4I-2 safety checks do not confirm live trading is disabled",
        "no_all_strategy_enablement": "Stage 4I-2 safety checks do not confirm all-strategy automation is disabled",
        "no_broker_submission_enabled": "Stage 4I-2 safety checks do not confirm broker submission is disabled",
        "no_market_data": "Stage 4I-2 safety checks do not confirm market data is disabled",
        "no_contract_qualification": "Stage 4I-2 safety checks do not confirm contract qualification is disabled",
        "no_order_submission": "Stage 4I-2 safety checks do not confirm order submission is disabled",
        "no_strategy_scan_execution": "Stage 4I-2 safety checks do not confirm strategy scan execution is disabled",
        "no_scheduler_registration": "Stage 4I-2 safety checks do not confirm scheduler registration is disabled",
        "no_lifecycle_execution": "Stage 4I-2 safety checks do not confirm lifecycle execution is disabled",
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
        blockers.append("Stage 4I-2 selected strategy must be paper_only true")
    if not one_strategy_only:
        blockers.append("Stage 4I-2 selected strategy must be one_strategy_only true")
    return (
        {
            "selected_strategy_id": selected_strategy_id,
            "paper_only": paper_only,
            "one_strategy_only": one_strategy_only,
        },
        blockers,
    )


def _schedule_checks(run_plan: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    schedule = _mapping(run_plan.get("proposed_schedule"))
    checks = {
        "proposed_schedule_present": bool(schedule),
        "would_register": schedule.get("would_register"),
        "would_execute": schedule.get("would_execute"),
        "scheduler_job_enabled": schedule.get("scheduler_job_enabled"),
    }
    blockers: list[str] = []
    if not schedule:
        blockers.append("proposed schedule is missing")
    if schedule.get("would_register") is not False:
        blockers.append("proposed schedule would register a scheduler job")
    if schedule.get("would_execute") is not False:
        blockers.append("proposed schedule would execute")
    if schedule.get("scheduler_job_enabled") is not False:
        blockers.append("proposed schedule enables a scheduler job")
    return checks, blockers


def _flow_checks(flow: Any) -> tuple[dict[str, Any], list[str]]:
    present = isinstance(flow, list) and bool(flow)
    blockers: list[str] = []
    structured = present
    all_disabled = True
    broker_disabled = True
    target_schema_valid = True
    payload_schema_valid = True
    if not present:
        blockers.append("proposed execution flow is missing or malformed")
        structured = False
    for step in flow if isinstance(flow, list) else []:
        if not isinstance(step, dict):
            structured = False
            target_schema_valid = False
            payload_schema_valid = False
            blockers.append("proposed execution flow contains malformed step")
            continue
        target_name = step.get("target_function") or step.get("target_component")
        if not isinstance(target_name, str) or not target_name:
            target_schema_valid = False
            blockers.append("proposed execution flow step missing target function/component")
        if not isinstance(step.get("payload"), dict):
            payload_schema_valid = False
            blockers.append("proposed execution flow step missing payload dict")
        if step.get("would_execute") is not False:
            all_disabled = False
            blockers.append("proposed execution flow contains executable step")
        if _is_broker_or_order_step(step) and step.get("would_submit") is not False:
            broker_disabled = False
            blockers.append("broker/order flow step would submit")
    return (
        {
            "proposed_execution_flow_present": present,
            "flow_steps_structured": structured,
            "flow_steps_ordered": present,
            "all_steps_would_execute_false": all_disabled,
            "broker_steps_would_submit_false": broker_disabled,
            "target_schema_valid": target_schema_valid,
            "payload_schema_valid": payload_schema_valid,
        },
        _dedupe(blockers),
    )


def _dry_run_trace(
    flow: list[Any],
    selected_strategy_id: str | None,
    *,
    activation_checks: dict[str, Any],
    state_checks: dict[str, Any],
    risk_checks: dict[str, Any],
    scheduler_checks: dict[str, Any],
    lifecycle_checks: dict[str, Any],
    paper_broker_checks: dict[str, Any],
    market_window_checks: dict[str, Any],
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for index, step in enumerate(flow, start=1):
        if not isinstance(step, dict):
            continue
        target_function = step.get("target_function")
        target_component = step.get("target_component")
        source_stage = _safe_text(step.get("stage"), _safe_text(target_function, target_component))
        payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
        item: dict[str, Any] = {
            "sequence_number": index,
            "source_stage": source_stage,
            "input_payload": _json_safe(payload),
            "dry_run_result": _dry_run_result(
                source_stage,
                selected_strategy_id,
                activation_checks=activation_checks,
                state_checks=state_checks,
                risk_checks=risk_checks,
                scheduler_checks=scheduler_checks,
                lifecycle_checks=lifecycle_checks,
                paper_broker_checks=paper_broker_checks,
                market_window_checks=market_window_checks,
            ),
            "would_execute": False,
            "would_write_state": False,
            "would_write_ledger": False,
            "would_register_scheduler": False,
            "paper_only": True,
            "live_trading_enabled": False,
            "status": "simulated",
        }
        if isinstance(target_function, str) and target_function:
            item["target_function"] = target_function
        if isinstance(target_component, str) and target_component:
            item["target_component"] = target_component
        if _is_broker_or_order_step(step):
            item["would_submit"] = False
        trace.append(item)
    return trace


def _dry_run_result(
    stage: str | None,
    selected_strategy_id: str | None,
    *,
    activation_checks: dict[str, Any],
    state_checks: dict[str, Any],
    risk_checks: dict[str, Any],
    scheduler_checks: dict[str, Any],
    lifecycle_checks: dict[str, Any],
    paper_broker_checks: dict[str, Any],
    market_window_checks: dict[str, Any],
) -> dict[str, Any]:
    results = {
        "pre_run_snapshot_check": {
            "checks_required": list(PRE_RUN_CHECKS_REQUIRED),
            "simulated_pass": True,
        },
        "risk_gate_check": {
            "kill_switch_required": True,
            "hard_halt_required": True,
            "daily_loss_limit_required": True,
            "simulated_pass": (
                risk_checks["risk_snapshot_present"] is not True
                or (
                    risk_checks["kill_switch_available"] is True
                    and risk_checks["hard_halt_available"] is True
                    and risk_checks["daily_loss_limit_available"] is True
                    and risk_checks["risk_bypass_enabled"] is False
                )
            ),
        },
        "activation_artifact_check": {
            "selected_strategy_id": selected_strategy_id,
            "activation_required": True,
            "simulated_pass": activation_checks["activation_snapshot_matches"] is True,
        },
        "scheduler_gate_check": {
            "would_register_scheduler": False,
            "simulated_pass": (
                scheduler_checks["scheduler_already_enabled"] is False
                and scheduler_checks["all_strategy_scheduler_enabled"] is False
                and scheduler_checks["selected_strategy_job_already_enabled"] is False
            ),
        },
        "lifecycle_gate_check": {
            "would_execute_lifecycle": False,
            "simulated_pass": (
                lifecycle_checks["lifecycle_already_enabled"] is False
                and lifecycle_checks["lifecycle_transition_execution_enabled"] is False
            ),
        },
        "market_window_check": {
            "market_window_required_for_real_run": True,
            "simulated_pass": market_window_checks["allowed_to_schedule_paper_run"] is not False,
        },
        "strategy_scan_preview": {
            "would_call_strategy_scan": False,
            "scan_output_placeholder": "not_executed_in_4I3",
        },
        "signal_to_intent_preview": {
            "would_create_intent": False,
            "intent_placeholder": "not_created_in_4I3",
        },
        "intent_to_ticket_preview": {
            "would_create_ticket": False,
            "ticket_placeholder": "not_created_in_4I3",
        },
        "paper_order_submission_gate_preview": {
            "would_submit_order": False,
            "broker_submission_enabled": False,
            "submission_placeholder": "blocked_in_4I3",
        },
        "state_ledger_tracking_preview": {
            "would_write_state": False,
            "would_write_ledger": False,
        },
        "alert_report_preview": {
            "would_emit_alert": False,
            "alert_placeholder": "not_emitted_in_4I3",
        },
        "post_run_reconciliation_preview": {
            "would_reconcile_broker": False,
            "reconciliation_placeholder": "not_executed_in_4I3",
        },
    }
    result = results.get(stage or "", {"simulated_pass": True})
    if stage == "pre_run_snapshot_check":
        result = dict(result)
        result["simulated_pass"] = (
            state_checks["active_halt"] is False
            and state_checks["unresolved_needs_reconciliation_count"] == 0
            and (
                state_checks["active_intents_count"] == 0
                or state_checks["active_intents_safe_for_enablement"] is True
            )
            and paper_broker_checks["live_trading_enabled"] is False
            and paper_broker_checks["broker_submission_enabled"] is False
        )
    return result


def _trace_checks(
    flow: list[Any], trace: list[dict[str, Any]], flow_checks: dict[str, Any]
) -> dict[str, bool]:
    plan_targets = [
        (step.get("target_function") or step.get("target_component"))
        for step in flow
        if isinstance(step, dict)
    ]
    trace_targets = [
        (step.get("target_function") or step.get("target_component")) for step in trace
    ]
    return {
        "trace_available": bool(trace) and flow_checks["proposed_execution_flow_present"] is True,
        "trace_order_matches_plan": plan_targets == trace_targets,
        "all_trace_items_simulated": all(item.get("status") == "simulated" for item in trace),
        "no_strategy_scan_called": all(
            item.get("dry_run_result", {}).get("would_call_strategy_scan") is not True
            for item in trace
        ),
        "no_intent_created": all(
            item.get("dry_run_result", {}).get("would_create_intent") is not True
            for item in trace
        ),
        "no_ticket_created": all(
            item.get("dry_run_result", {}).get("would_create_ticket") is not True
            for item in trace
        ),
        "no_broker_submission": all(
            item.get("would_submit") is not True
            and item.get("dry_run_result", {}).get("would_submit_order") is not True
            for item in trace
        ),
        "no_state_write": all(item.get("would_write_state") is not True for item in trace),
        "no_ledger_write": all(item.get("would_write_ledger") is not True for item in trace),
        "no_scheduler_registration": all(
            item.get("would_register_scheduler") is not True for item in trace
        ),
        "no_lifecycle_execution": all(
            item.get("dry_run_result", {}).get("would_execute_lifecycle") is not True
            for item in trace
        ),
    }


def _trace_blockers(checks: dict[str, bool]) -> list[str]:
    labels = {
        "trace_available": "dry-run trace is unavailable",
        "trace_order_matches_plan": "dry-run trace order does not match the 4I-2 plan",
        "all_trace_items_simulated": "dry-run trace contains non-simulated item",
        "no_strategy_scan_called": "dry-run trace would call strategy scan",
        "no_intent_created": "dry-run trace would create intent",
        "no_ticket_created": "dry-run trace would create ticket",
        "no_broker_submission": "dry-run trace would submit broker order",
        "no_state_write": "dry-run trace would write state",
        "no_ledger_write": "dry-run trace would write ledger",
        "no_scheduler_registration": "dry-run trace would register scheduler",
        "no_lifecycle_execution": "dry-run trace would execute lifecycle",
    }
    return [label for key, label in labels.items() if checks.get(key) is not True]


def _activation_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    matches = True
    checks = {
        "activation_snapshot_present": present,
        "activation_snapshot_matches": True,
    }
    if not present:
        warnings.append("activation snapshot missing; 4I-4 must verify activation artifact before activation")
        return checks, blockers, warnings

    active_ids = _as_string_list(data.get("active_strategy_ids"))
    if len(active_ids) > 1:
        matches = False
        blockers.append("activation snapshot shows more than one activated strategy")
    if active_ids and active_ids != [selected_strategy_id]:
        matches = False
        blockers.append("activation snapshot active_strategy_ids do not match selected strategy")

    for record in _activation_records(data):
        if not isinstance(record, dict):
            warnings.append("malformed activation snapshot entry ignored")
            continue
        if record.get("selected_strategy_id") not in (None, selected_strategy_id):
            matches = False
            blockers.append("activation snapshot selected_strategy_id does not match")
        if record.get("paper_only") is False:
            matches = False
            blockers.append("activation snapshot paper_only contradicts selected strategy")
        if record.get("enabled_strategy_count") not in (None, 1):
            matches = False
            blockers.append("activation snapshot enabled_strategy_count must be 1")
        for key in ("live_trading_enabled", "all_strategies_enabled", "broker_submission_enabled"):
            if record.get(key) is True:
                matches = False
                blockers.append(f"activation snapshot {key} contradicts Stage 4I-3 safety")
    checks["activation_snapshot_matches"] = matches
    return checks, blockers, warnings


def _activation_records(snapshot: dict[str, Any]) -> list[Any]:
    records: list[Any] = []
    if "activation_record" in snapshot:
        records.append(snapshot.get("activation_record"))
    records.extend(_as_list(snapshot.get("activations")))
    if any(key in snapshot for key in ("selected_strategy_id", "paper_only", "enabled_strategy_count")):
        records.append(snapshot)
    return records


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
    checks = {
        "state_snapshot_present": present,
        "active_halt": bool(data.get("active_halt") or data.get("halt_active")),
        "unresolved_needs_reconciliation_count": unresolved_count,
        "active_intents_count": active_intents_count,
        "active_intents_safe_for_enablement": active_intents_safe,
        "open_positions_count": _safe_int(data.get("open_positions_count")),
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("state snapshot missing; 4I-4 must verify runtime state immediately")
        return checks, blockers, warnings
    if checks["active_halt"]:
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
        "kill_switch_available": data.get("kill_switch_available") is True,
        "hard_halt_available": data.get("hard_halt_available") is True,
        "daily_loss_limit_available": data.get("daily_loss_limit_available") is True,
        "max_position_limit_available": (
            data.get("max_position_limit_available")
            if "max_position_limit_available" in data
            else None
        ),
        "risk_bypass_enabled": data.get("risk_bypass_enabled") is True,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("risk snapshot missing; 4I-4 must verify risk controls immediately")
        return checks, blockers, warnings
    if checks["risk_bypass_enabled"]:
        blockers.append("risk bypass is enabled")
    for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
        if checks[key] is not True:
            blockers.append(f"{key} must be true in supplied risk snapshot")
    if checks["max_position_limit_available"] is False:
        blockers.append("max_position_limit_available must not be false in supplied risk snapshot")
    return checks, blockers, warnings


def _scheduler_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    already_enabled = _contains_truthy_flag(data, "scheduler_automation_enabled") or _contains_truthy_flag(
        data, "scheduler_wiring_enabled"
    )
    all_strategy_enabled = _contains_truthy_flag(data, "all_strategy_scheduler_enabled") or _contains_truthy_flag(
        data, "all_strategies_enabled"
    )
    selected_enabled = _selected_strategy_job_enabled(data, selected_strategy_id)
    checks = {
        "scheduler_snapshot_present": present,
        "scheduler_already_enabled": already_enabled,
        "all_strategy_scheduler_enabled": all_strategy_enabled,
        "selected_strategy_job_already_enabled": selected_enabled,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("scheduler snapshot missing; 4I-4 must verify scheduler state")
    if already_enabled:
        blockers.append("scheduler automation is already broadly enabled")
    if all_strategy_enabled:
        blockers.append("all-strategy scheduler automation is already enabled")
    if selected_enabled:
        blockers.append("selected strategy scheduled job is already active")
    return checks, blockers, warnings


def _lifecycle_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    already_enabled = _contains_truthy_flag(data, "lifecycle_automation_enabled") or _contains_truthy_flag(
        data, "lifecycle_wiring_enabled"
    )
    if data.get("disabled") is True and data.get("dry_run_only") is True:
        already_enabled = False
    transition_enabled = _contains_truthy_flag(data, "lifecycle_transition_execution_enabled")
    checks = {
        "lifecycle_snapshot_present": present,
        "lifecycle_already_enabled": already_enabled,
        "lifecycle_transition_execution_enabled": transition_enabled,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; 4I-4 must verify lifecycle state")
    if already_enabled:
        blockers.append("lifecycle automation is already enabled")
    if transition_enabled:
        blockers.append("lifecycle transition execution is enabled")
    return checks, blockers, warnings


def _paper_broker_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    mode = data.get("mode")
    port = data.get("ibkr_port")
    checks = {
        "paper_broker_snapshot_present": present,
        "mode": mode,
        "paper_trading": data.get("paper_trading"),
        "ibkr_port": port,
        "paper_config_valid": True,
        "live_trading_enabled": data.get("live_trading_enabled") is True,
        "broker_submission_enabled": data.get("broker_submission_enabled") is True,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("paper broker snapshot missing; 4I-4 must verify paper broker config immediately")
        return checks, blockers, warnings
    if mode not in (None, "PAPER"):
        blockers.append("paper broker mode must be PAPER")
    if data.get("paper_trading") is False:
        blockers.append("paper_trading must not be false")
    if port not in (None, *PAPER_IBKR_PORTS):
        blockers.append("ibkr_port must be a paper trading port")
    if checks["live_trading_enabled"]:
        blockers.append("paper broker snapshot enables live trading")
    if checks["broker_submission_enabled"]:
        blockers.append("paper broker snapshot enables broker submission")
    checks["paper_config_valid"] = not blockers
    return checks, blockers, warnings


def _market_window_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    checks = {
        "market_window_snapshot_present": present,
        "allowed_to_schedule_paper_run": data.get("allowed_to_schedule_paper_run"),
        "is_trading_day": data.get("is_trading_day"),
        "market_open": data.get("market_open"),
        "reason": data.get("reason"),
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append(MARKET_WINDOW_MANUAL_WARNING)
        return checks, blockers, warnings
    if data.get("allowed_to_schedule_paper_run") is False:
        blockers.append("market window snapshot explicitly disallows scheduling a paper run")
    if data.get("market_open") is False:
        warnings.append("market is currently closed; dry run may continue but 4I-4 must verify run timing")
    if data.get("is_trading_day") is False:
        warnings.append("snapshot is not a trading day; dry run may continue but 4I-4 must verify run timing")
    return checks, blockers, warnings


def _safety_checks(*sources: Any) -> dict[str, bool]:
    return {
        "no_live_trading": not _contains_truthy_flag(sources, "live_trading_enabled"),
        "no_all_strategy_enablement": not (
            _contains_truthy_flag(sources, "all_strategies_enabled")
            or _contains_truthy_flag(sources, "enable_all_strategies")
        ),
        "no_broker_submission_enabled": not _contains_truthy_flag(
            sources, "broker_submission_enabled"
        ),
        "no_market_data": not _contains_truthy_flag(sources, "market_data_enabled"),
        "no_contract_qualification": not _contains_truthy_flag(
            sources, "contract_qualification_enabled"
        ),
        "no_order_submission": not (
            _contains_truthy_flag(sources, "order_submission_enabled")
            or _contains_truthy_flag(sources, "live_orders_enabled")
        ),
        "no_strategy_scan_execution": not _contains_truthy_flag(
            sources, "strategy_scan_execution_enabled"
        ),
        "no_scheduler_registration": not _contains_truthy_flag(
            sources, "scheduler_registration_enabled"
        ),
        "no_lifecycle_execution": not (
            _contains_truthy_flag(sources, "lifecycle_execution_enabled")
            or _contains_truthy_flag(sources, "lifecycle_transition_execution_enabled")
        ),
        "no_state_write": not _contains_truthy_flag(sources, "state_write_enabled"),
        "no_ledger_write": not _contains_truthy_flag(sources, "ledger_write_enabled"),
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
        "no_scheduler_registration": "scheduler registration safety flag is enabled",
        "no_lifecycle_execution": "lifecycle execution safety flag is enabled",
        "no_state_write": "state write safety flag is enabled",
        "no_ledger_write": "ledger write safety flag is enabled",
    }
    return [label for key, label in labels.items() if checks.get(key) is not True]


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    schedule_checks: dict[str, Any],
    flow_checks: dict[str, Any],
    dry_run_trace: list[dict[str, Any]],
    dry_run_trace_checks: dict[str, Any],
    activation_checks: dict[str, Any],
    state_checks: dict[str, Any],
    risk_checks: dict[str, Any],
    scheduler_checks: dict[str, Any],
    lifecycle_checks: dict[str, Any],
    paper_broker_checks: dict[str, Any],
    market_window_checks: dict[str, Any],
    safety_checks: dict[str, Any],
    ready: bool,
    blockers: list[str],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "stage4i3_scheduled_run_dry_run_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "schedule_checks": schedule_checks,
        "flow_checks": flow_checks,
        "dry_run_trace": dry_run_trace,
        "dry_run_trace_checks": dry_run_trace_checks,
        "activation_checks": activation_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4i4": {
            "ready_to_build_scheduler_activation_gate": ready,
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
    selected_strategy = _mapping(report.get("selected_strategy"))
    value = selected_strategy.get("selected_strategy_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _selected_strategy_job_enabled(data: dict[str, Any], selected_strategy_id: str | None) -> bool:
    if not selected_strategy_id:
        return False
    for job in _as_list(data.get("jobs")) + _as_list(data.get("active_jobs")):
        if not isinstance(job, dict):
            continue
        if job.get("strategy_id") != selected_strategy_id:
            continue
        if job.get("disabled") is True and job.get("dry_run_only") is True:
            continue
        return True
    return False


def _is_broker_or_order_step(step: dict[str, Any]) -> bool:
    values = [
        step.get("stage"),
        step.get("target_function"),
        step.get("target_component"),
    ]
    text = " ".join(value.lower() for value in values if isinstance(value, str))
    return any(token in text for token in ("broker", "order", "submission", "ticket"))


def _default_artifact_checks() -> dict[str, bool]:
    return {
        "stage4i2_report_present": False,
        "stage4i2_report_ready": False,
        "selected_strategy_present": False,
        "run_plan_available": False,
        "proposed_schedule_valid": False,
        "proposed_execution_flow_valid": False,
    }


def _default_selected_strategy() -> dict[str, Any]:
    return {"selected_strategy_id": None, "paper_only": False, "one_strategy_only": False}


def _default_schedule_checks() -> dict[str, Any]:
    return {
        "proposed_schedule_present": False,
        "would_register": None,
        "would_execute": None,
        "scheduler_job_enabled": None,
    }


def _default_flow_checks() -> dict[str, Any]:
    return {
        "proposed_execution_flow_present": False,
        "flow_steps_structured": False,
        "flow_steps_ordered": False,
        "all_steps_would_execute_false": False,
        "broker_steps_would_submit_false": False,
        "target_schema_valid": False,
        "payload_schema_valid": False,
    }


def _default_trace_checks() -> dict[str, bool]:
    return {
        "trace_available": False,
        "trace_order_matches_plan": False,
        "all_trace_items_simulated": False,
        "no_strategy_scan_called": False,
        "no_intent_created": False,
        "no_ticket_created": False,
        "no_broker_submission": False,
        "no_state_write": False,
        "no_ledger_write": False,
        "no_scheduler_registration": False,
        "no_lifecycle_execution": False,
    }


def _default_activation_checks() -> dict[str, Any]:
    return {"activation_snapshot_present": False, "activation_snapshot_matches": False}


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
        "kill_switch_available": False,
        "hard_halt_available": False,
        "daily_loss_limit_available": False,
        "max_position_limit_available": None,
        "risk_bypass_enabled": False,
    }


def _default_scheduler_checks() -> dict[str, bool]:
    return {
        "scheduler_snapshot_present": False,
        "scheduler_already_enabled": False,
        "all_strategy_scheduler_enabled": False,
        "selected_strategy_job_already_enabled": False,
    }


def _default_lifecycle_checks() -> dict[str, bool]:
    return {
        "lifecycle_snapshot_present": False,
        "lifecycle_already_enabled": False,
        "lifecycle_transition_execution_enabled": False,
    }


def _default_paper_broker_checks() -> dict[str, Any]:
    return {
        "paper_broker_snapshot_present": False,
        "mode": None,
        "paper_trading": None,
        "ibkr_port": None,
        "paper_config_valid": False,
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


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_mapping(value: dict | None) -> tuple[bool, dict[str, Any]]:
    if isinstance(value, dict):
        return True, value
    return False, {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _as_string_list(value: Any) -> list[str]:
    return [item for item in _as_list(value) if isinstance(item, str)]


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_text(value: Any, default: str | None) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _first_present(*values: Any, default: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _contains_truthy_flag(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return any(
            (item_key == key and bool(item)) or _contains_truthy_flag(item, key)
            for item_key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_truthy_flag(item, key) for item in value)
    return False


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _generated_at(now_provider: Callable[[], Any] | None) -> str:
    if now_provider is None:
        return DEFAULT_GENERATED_AT
    value = now_provider()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
