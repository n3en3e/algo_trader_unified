"""Pure Stage 4H-3 controlled automation wiring dry-run report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
PAPER_IBKR_PORTS = {4004}
UNSAFE_TRUE_FLAGS = (
    "live_orders_enabled",
    "live_trading_enabled",
    "automated_paper_trading_enabled",
    "broker_submission_enabled",
    "scheduler_wiring_enabled",
    "scheduler_automation_enabled",
    "daemon_wiring_enabled",
    "lifecycle_wiring_enabled",
    "lifecycle_automation_enabled",
    "lifecycle_transition_execution_enabled",
    "market_data_enabled",
    "contract_qualification_enabled",
    "all_strategies_enabled",
    "enable_all_strategies",
    "automation_enabled",
)
ORDERED_NEXT_STEPS = [
    "Build Stage 4H-4 one-strategy automation enablement gate.",
    "Keep scheduler/lifecycle automation disabled until 4H-4 explicitly gates one strategy.",
    "Verify state, ledger, risk, and paper broker config before enablement.",
]
DO_NOT_DO_YET = [
    "Do not enable live trading.",
    "Do not enable all strategies at once.",
    "Do not place orders now.",
    "Do not enable scheduler jobs now.",
    "Do not change strategy thresholds.",
    "Do not change position sizing.",
    "Do not bypass risk controls.",
]
ORDERED_GROUPS = (
    "risk_gate_dry_run_operations",
    "state_ledger_tracking_dry_run_operations",
    "scheduler_dry_run_operations",
    "lifecycle_dry_run_operations",
    "signal_to_intent_dry_run_operations",
    "intent_to_ticket_dry_run_operations",
    "ticket_to_paper_submit_dry_run_operations",
    "state_ledger_tracking_dry_run_operations",
    "paper_broker_guard_dry_run_operations",
)


def build_stage4h3_automation_wiring_dry_run_report(
    *,
    stage4h2_wiring_preview_report: dict | None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a deterministic disabled wiring dry-run packet from Stage 4H-2."""

    try:
        return _json_safe(
            _build_report(
                stage4h2_wiring_preview_report=stage4h2_wiring_preview_report,
                state_snapshot=state_snapshot,
                risk_snapshot=risk_snapshot,
                scheduler_snapshot=scheduler_snapshot,
                lifecycle_snapshot=lifecycle_snapshot,
                paper_broker_snapshot=paper_broker_snapshot,
                now_provider=now_provider,
            )
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must remain data-only.
        message = f"unexpected report failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_selected_strategy(None),
                dry_run_packet=_dry_run_packet(False, _empty_operations(), {}),
                operation_checks=_operation_checks(_empty_operations()),
                scheduler_checks=_default_scheduler_checks(),
                lifecycle_checks=_default_lifecycle_checks(),
                risk_checks=_default_risk_checks(),
                state_checks=_default_state_checks(),
                paper_broker_checks=_default_paper_broker_checks(),
                safety_checks=_default_safety_checks(),
                ready=False,
                blockers=[message],
                warnings=[],
                errors=[message],
            )
        )


def _build_report(
    *,
    stage4h2_wiring_preview_report: dict | None,
    state_snapshot: dict | None,
    risk_snapshot: dict | None,
    scheduler_snapshot: dict | None,
    lifecycle_snapshot: dict | None,
    paper_broker_snapshot: dict | None,
    now_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    generated_at = _generated_at(now_provider)
    stage4h2 = (
        stage4h2_wiring_preview_report
        if isinstance(stage4h2_wiring_preview_report, dict)
        else None
    )
    data = _mapping(stage4h2)
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4h2_wiring_preview_report is None:
        blockers.append("Stage 4H-2 automation wiring preview report is missing")
    elif stage4h2 is None:
        blockers.append("Stage 4H-2 automation wiring preview report must be a dict")
        errors.append("Stage 4H-2 automation wiring preview report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    artifact_checks = _artifact_checks(stage4h2_wiring_preview_report)
    if not artifact_checks["stage4h2_preview_ready"]:
        blockers.append("Stage 4H-2 preview is not ready for Stage 4H-3")
    if not artifact_checks["selected_strategy_present"]:
        blockers.append("selected_preview_strategy_id is missing from Stage 4H-2 strategy_selection")
    if not artifact_checks["wiring_preview_available"]:
        blockers.append("Stage 4H-2 wiring_preview is not available")

    selected_id = _selected_strategy_id(data)
    selected_strategy = _selected_strategy(selected_id)
    wiring = _mapping(data.get("wiring_preview"))
    scheduler_preview = wiring.get("proposed_scheduler_wiring_preview")
    lifecycle_preview = wiring.get("proposed_lifecycle_wiring_preview")

    if not _is_structured_preview(scheduler_preview):
        blockers.append("proposed scheduler wiring preview must be structured JSON-safe data")
    if not _is_structured_preview(lifecycle_preview):
        blockers.append("proposed lifecycle wiring preview must be structured JSON-safe data")

    scheduler_checks, scheduler_blockers, scheduler_warnings = _scheduler_checks(
        scheduler_snapshot, selected_id, scheduler_preview
    )
    blockers.extend(scheduler_blockers)
    warnings.extend(scheduler_warnings)

    lifecycle_checks, lifecycle_blockers, lifecycle_warnings = _lifecycle_checks(
        lifecycle_snapshot, lifecycle_preview
    )
    blockers.extend(lifecycle_blockers)
    warnings.extend(lifecycle_warnings)

    risk_checks, risk_blockers, risk_warnings = _risk_checks(risk_snapshot)
    blockers.extend(risk_blockers)
    warnings.extend(risk_warnings)

    state_checks, state_blockers, state_warnings = _state_checks(state_snapshot)
    blockers.extend(state_blockers)
    warnings.extend(state_warnings)

    paper_checks, paper_blockers, paper_warnings = _paper_broker_checks(paper_broker_snapshot)
    blockers.extend(paper_blockers)
    warnings.extend(paper_warnings)

    safety_checks, safety_blockers = _safety_checks(
        stage4h2,
        scheduler_snapshot,
        lifecycle_snapshot,
        risk_snapshot,
        state_snapshot,
        paper_broker_snapshot,
    )
    blockers.extend(safety_blockers)

    operations = _build_operations(
        selected_strategy_id=selected_id,
        scheduler_preview=scheduler_preview,
        lifecycle_preview=lifecycle_preview,
        wiring_preview=wiring,
    )
    operation_checks = _operation_checks(operations)
    blockers.extend(_operation_blockers(operation_checks))
    dry_run_available = (
        artifact_checks["stage4h2_preview_ready"]
        and artifact_checks["selected_strategy_present"]
        and artifact_checks["wiring_preview_available"]
        and _is_structured_preview(scheduler_preview)
        and _is_structured_preview(lifecycle_preview)
        and operation_checks["operations_structured"]
    )

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    ready = (
        dry_run_available
        and all(operation_checks.values())
        and all(safety_checks.values())
        and not scheduler_checks["scheduler_already_enabled"]
        and not scheduler_checks["selected_strategy_job_already_enabled"]
        and scheduler_checks["proposed_jobs_disabled"]
        and scheduler_checks["proposed_jobs_would_register_false"]
        and scheduler_checks["proposed_jobs_would_execute_false"]
        and not lifecycle_checks["lifecycle_already_enabled"]
        and not lifecycle_checks["lifecycle_transition_execution_enabled"]
        and lifecycle_checks["proposed_flows_would_execute_false"]
        and not risk_checks["risk_bypass_enabled"]
        and (not risk_checks["risk_snapshot_present"] or (
            risk_checks["kill_switch_available"]
            and risk_checks["hard_halt_available"]
            and risk_checks["daily_loss_limit_available"]
        ))
        and not state_checks["active_halt"]
        and state_checks["unresolved_needs_reconciliation_count"] == 0
        and (not paper_checks["paper_broker_snapshot_present"] or paper_checks["paper_config_valid"])
        and not blocker_list
        and not error_list
    )

    packet = _dry_run_packet(
        dry_run_available,
        operations,
        _final_enablement_gate(selected_id, ready, blocker_list, warning_list),
    )
    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
        dry_run_packet=packet,
        operation_checks=operation_checks,
        scheduler_checks=scheduler_checks,
        lifecycle_checks=lifecycle_checks,
        risk_checks=risk_checks,
        state_checks=state_checks,
        paper_broker_checks=paper_checks,
        safety_checks=safety_checks,
        ready=ready,
        blockers=blocker_list,
        warnings=warning_list,
        errors=error_list,
    )


def _artifact_checks(report: dict | None) -> dict[str, bool]:
    data = _mapping(report)
    readiness = _mapping(data.get("readiness_for_stage4h3"))
    selected = _selected_strategy_id(data)
    wiring = _mapping(data.get("wiring_preview"))
    return {
        "stage4h2_preview_present": report is not None,
        "stage4h2_preview_ready": (
            data.get("stage4h2_automation_wiring_preview_report") is True
            and readiness.get("ready_to_build_automation_wiring_dry_run") is True
            and data.get("success") is True
        ),
        "selected_strategy_present": isinstance(selected, str) and bool(selected),
        "wiring_preview_available": wiring.get("available") is True,
    }


def _selected_strategy_id(report: dict[str, Any]) -> str | None:
    value = _mapping(report.get("strategy_selection")).get("selected_preview_strategy_id")
    return value if isinstance(value, str) and value else None


def _build_operations(
    *,
    selected_strategy_id: str | None,
    scheduler_preview: Any,
    lifecycle_preview: Any,
    wiring_preview: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    strategy_id = selected_strategy_id or ""
    sequence = 1
    operations = _empty_operations()

    def add(group_key: str, operation: dict[str, Any]) -> None:
        nonlocal sequence
        operation["sequence_number"] = sequence
        sequence += 1
        operations[group_key].append(operation)

    for gate in _as_list(wiring_preview.get("proposed_risk_gates")) or [
        {"name": "kill_switch_check"},
        {"name": "hard_halt_check"},
        {"name": "daily_loss_check"},
        {"name": "paper_only_config_check"},
    ]:
        if not isinstance(gate, dict):
            continue
        name = str(gate.get("name") or "risk_gate_check")
        add(
            "risk_gate_dry_run_operations",
            _operation(
                group="risk_gate_dry_run_operations",
                strategy_id=strategy_id,
                operation=name,
                target_component="RiskManager",
                payload={"strategy_id": strategy_id, "gate_name": name, "paper_only": True},
            ),
        )

    add(
        "state_ledger_tracking_dry_run_operations",
        _operation(
            group="state_ledger_tracking_dry_run_operations",
            strategy_id=strategy_id,
            operation="state_reconciliation_check",
            target_component="StateStore",
            payload={"strategy_id": strategy_id, "required_state": "clean_reconciliation"},
        ),
    )

    for job in _scheduler_jobs(scheduler_preview):
        add(
            "scheduler_dry_run_operations",
            _operation(
                group="scheduler_dry_run_operations",
                strategy_id=str(job.get("strategy_id") or strategy_id),
                operation="disabled_scheduler_registration_preview",
                target_component="Scheduler",
                payload={
                    "job_id": job.get("job_id"),
                    "strategy_id": job.get("strategy_id") or strategy_id,
                    "trigger_description": job.get("trigger_description"),
                    "disabled": True,
                    "would_register": False,
                    "would_execute": False,
                    "paper_only": True,
                },
                would_register=False,
            ),
        )

    for flow in _lifecycle_flows(lifecycle_preview):
        name = str(flow.get("name") or flow.get("flow_name") or "lifecycle_flow")
        add(
            "lifecycle_dry_run_operations",
            _operation(
                group="lifecycle_dry_run_operations",
                strategy_id=strategy_id,
                operation=name,
                target_component="LifecycleRouter",
                payload={"strategy_id": strategy_id, "flow_name": name, "paper_only": True, "would_execute": False},
            ),
        )

    add(
        "signal_to_intent_dry_run_operations",
        _operation(
            group="signal_to_intent_dry_run_operations",
            strategy_id=strategy_id,
            operation="signal_to_intent_preview",
            target_function="build_stage4g1_lifecycle_intake_report",
            payload={"strategy_id": strategy_id, "expected_signal_source": "selected_strategy_preview", "paper_only": True},
        ),
    )
    add(
        "intent_to_ticket_dry_run_operations",
        _operation(
            group="intent_to_ticket_dry_run_operations",
            strategy_id=strategy_id,
            operation="intent_to_ticket_preview",
            target_function="build_broker_order_request",
            payload={"strategy_id": strategy_id, "expected_input": "paper_intent", "paper_only": True},
        ),
    )
    add(
        "ticket_to_paper_submit_dry_run_operations",
        _operation(
            group="ticket_to_paper_submit_dry_run_operations",
            strategy_id=strategy_id,
            operation="ticket_to_paper_submit_preview",
            target_function="build_ibkr_paper_order_plan",
            payload={"strategy_id": strategy_id, "expected_input": "paper_order_ticket", "would_submit": False, "paper_only": True},
            would_submit=False,
        ),
    )
    add(
        "state_ledger_tracking_dry_run_operations",
        _operation(
            group="state_ledger_tracking_dry_run_operations",
            strategy_id=strategy_id,
            operation="submit_to_state_ledger_tracking_preview",
            target_component="Ledger",
            payload={"strategy_id": strategy_id, "would_mutate_state": False, "would_write_ledger": False, "paper_only": True},
        ),
    )
    add(
        "paper_broker_guard_dry_run_operations",
        _operation(
            group="paper_broker_guard_dry_run_operations",
            strategy_id=strategy_id,
            operation="paper_broker_config_guard_preview",
            target_component="PaperBrokerGuard",
            payload={"strategy_id": strategy_id, "required_mode": "PAPER", "allowed_ibkr_ports": sorted(PAPER_IBKR_PORTS), "live_trading_enabled": False},
        ),
    )
    return operations


def _operation(
    *,
    group: str,
    strategy_id: str,
    operation: str,
    payload: dict[str, Any],
    target_function: str | None = None,
    target_component: str | None = None,
    would_register: bool | None = None,
    would_submit: bool | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "group": group,
        "strategy_id": strategy_id,
        "operation": operation,
        "payload": payload,
        "would_execute": False,
        "paper_only": True,
        "live_trading_enabled": False,
    }
    if target_function:
        result["target_function"] = target_function
    if target_component:
        result["target_component"] = target_component
    if would_register is not None:
        result["would_register"] = would_register
    if would_submit is not None:
        result["would_submit"] = would_submit
    return result


def _scheduler_checks(
    snapshot: dict | None,
    selected_strategy_id: str | None,
    scheduler_preview: Any,
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    jobs = _scheduler_jobs(scheduler_preview)
    checks = {
        "scheduler_snapshot_present": present,
        "scheduler_already_enabled": _contains_truthy_flag(data, "scheduler_automation_enabled") or _contains_truthy_flag(data, "scheduler_wiring_enabled"),
        "selected_strategy_job_already_enabled": _selected_strategy_job_enabled(data, selected_strategy_id),
        "proposed_jobs_disabled": bool(jobs) and all(job.get("disabled") is True for job in jobs),
        "proposed_jobs_would_register_false": bool(jobs) and all(job.get("would_register") is False for job in jobs),
        "proposed_jobs_would_execute_false": bool(jobs) and all(job.get("would_execute") is False for job in jobs),
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("scheduler snapshot missing; 4H-4 must verify scheduler remains disabled")
    if checks["scheduler_already_enabled"]:
        blockers.append("scheduler automation is already enabled")
    if checks["selected_strategy_job_already_enabled"]:
        blockers.append("selected strategy scheduler job is already enabled")
    if not jobs:
        blockers.append("proposed scheduler jobs are missing or malformed")
    for key in ("proposed_jobs_disabled", "proposed_jobs_would_register_false", "proposed_jobs_would_execute_false"):
        if checks[key] is not True:
            blockers.append(f"{key} must be true")
    return checks, blockers, warnings


def _lifecycle_checks(
    snapshot: dict | None,
    lifecycle_preview: Any,
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    flows = _lifecycle_flows(lifecycle_preview)
    checks = {
        "lifecycle_snapshot_present": present,
        "lifecycle_already_enabled": _contains_truthy_flag(data, "lifecycle_automation_enabled") or _contains_truthy_flag(data, "lifecycle_wiring_enabled"),
        "lifecycle_transition_execution_enabled": _contains_truthy_flag(data, "lifecycle_transition_execution_enabled"),
        "proposed_flows_would_execute_false": bool(flows) and all(flow.get("would_execute") is False for flow in flows),
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; 4H-4 must verify lifecycle automation remains disabled")
    if checks["lifecycle_already_enabled"]:
        blockers.append("lifecycle automation is already enabled")
    if checks["lifecycle_transition_execution_enabled"]:
        blockers.append("lifecycle transition execution is enabled")
    if not flows:
        blockers.append("proposed lifecycle flows are missing or malformed")
    if checks["proposed_flows_would_execute_false"] is not True:
        blockers.append("proposed lifecycle flows must not execute")
    return checks, blockers, warnings


def _risk_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    checks = {
        "risk_snapshot_present": present,
        "kill_switch_available": data.get("kill_switch_available") is True,
        "hard_halt_available": data.get("hard_halt_available") is True,
        "daily_loss_limit_available": data.get("daily_loss_limit_available") is True,
        "risk_bypass_enabled": data.get("risk_bypass_enabled") is True,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("risk snapshot missing; 4H-4 must verify risk controls before enablement")
        return checks, blockers, warnings
    if checks["risk_bypass_enabled"]:
        blockers.append("risk controls are bypassed")
    for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
        if checks[key] is not True:
            blockers.append(f"{key} must be true in supplied risk snapshot")
    return checks, blockers, warnings


def _state_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    unresolved = _first_present(
        data.get("unresolved_needs_reconciliation_count"),
        data.get("needs_reconciliation_count"),
        default=0,
    )
    checks = {
        "state_snapshot_present": present,
        "active_halt": bool(data.get("active_halt") or data.get("halt_active")),
        "unresolved_needs_reconciliation_count": _safe_int(unresolved),
        "active_intents_count": _safe_int(data.get("active_intents_count")),
        "open_positions_count": _safe_int(data.get("open_positions_count")),
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("state snapshot missing; 4H-4 must verify state before enablement")
        return checks, blockers, warnings
    if checks["active_halt"]:
        blockers.append("active halt is present")
    if checks["unresolved_needs_reconciliation_count"] > 0:
        blockers.append("unresolved NEEDS_RECONCILIATION is present")
    if checks["active_intents_count"] > 0:
        warnings.append("active intents are present; 4H-4 must verify they are safe before enablement")
    return checks, blockers, warnings


def _paper_broker_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    mode = data.get("mode")
    port = data.get("ibkr_port")
    live_enabled = data.get("live_trading_enabled") is True
    submission_enabled = data.get("broker_submission_enabled") is True
    paper_valid = (
        not present
        or (
            (mode in (None, "PAPER"))
            and data.get("paper_trading") is not False
            and (port in (None, *PAPER_IBKR_PORTS))
            and not live_enabled
            and not submission_enabled
        )
    )
    checks = {
        "paper_broker_snapshot_present": present,
        "mode": mode,
        "paper_trading": data.get("paper_trading"),
        "ibkr_port": port,
        "paper_config_valid": paper_valid,
        "live_trading_enabled": live_enabled,
        "broker_submission_enabled": submission_enabled,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("paper broker snapshot missing; 4H-4 must verify PAPER broker config before enablement")
        return checks, blockers, warnings
    if mode not in (None, "PAPER"):
        blockers.append("paper broker mode must be PAPER")
    if data.get("paper_trading") is False:
        blockers.append("paper_trading must not be false")
    if port not in (None, *PAPER_IBKR_PORTS):
        blockers.append("ibkr_port must be a paper trading port")
    if live_enabled:
        blockers.append("live trading is enabled in paper broker snapshot")
    if submission_enabled:
        blockers.append("broker submission is enabled in paper broker snapshot")
    return checks, blockers, warnings


def _safety_checks(*sources: Any) -> tuple[dict[str, bool], list[str]]:
    unsafe = {key: any(_contains_truthy_flag(source, key) for source in sources) for key in UNSAFE_TRUE_FLAGS}
    checks = {
        "no_live_orders": not unsafe["live_orders_enabled"] and not unsafe["live_trading_enabled"],
        "no_market_data": not unsafe["market_data_enabled"],
        "no_contract_qualification": not unsafe["contract_qualification_enabled"],
        "no_broker_submission_enabled": not unsafe["broker_submission_enabled"],
        "no_scheduler_wiring_enabled": not unsafe["scheduler_wiring_enabled"] and not unsafe["scheduler_automation_enabled"],
        "no_lifecycle_wiring_enabled": not unsafe["lifecycle_wiring_enabled"] and not unsafe["lifecycle_automation_enabled"],
        "no_automated_paper_trading_enabled": not unsafe["automated_paper_trading_enabled"] and not unsafe["automation_enabled"],
        "no_daemon_wiring_enabled": not unsafe["daemon_wiring_enabled"],
        "no_all_strategy_enablement": not unsafe["all_strategies_enabled"] and not unsafe["enable_all_strategies"],
    }
    reasons = {
        "no_live_orders": "live order or live trading flag is enabled",
        "no_market_data": "market data flag is enabled",
        "no_contract_qualification": "contract qualification flag is enabled",
        "no_broker_submission_enabled": "broker submission automation is enabled",
        "no_scheduler_wiring_enabled": "scheduler automation is already enabled",
        "no_lifecycle_wiring_enabled": "lifecycle automation is already enabled",
        "no_automated_paper_trading_enabled": "automated paper trading is already enabled",
        "no_daemon_wiring_enabled": "daemon wiring is already enabled",
        "no_all_strategy_enablement": "all-strategy automation is enabled or requested",
    }
    return checks, [reasons[key] for key, value in checks.items() if value is not True]


def _operation_checks(operations: dict[str, list[dict[str, Any]]]) -> dict[str, bool]:
    flat = _flatten_operations(operations)
    return {
        "all_operations_would_execute_false": bool(flat) and all(op.get("would_execute") is False for op in flat),
        "all_scheduler_would_register_false": all(op.get("would_register") is False for op in operations.get("scheduler_dry_run_operations", [])),
        "all_broker_would_submit_false": all(op.get("would_submit") is False for op in operations.get("ticket_to_paper_submit_dry_run_operations", [])),
        "deterministic_operation_order": [op.get("sequence_number") for op in flat] == list(range(1, len(flat) + 1)),
        "operations_structured": bool(flat) and all(isinstance(op, dict) for op in flat),
        "target_schema_valid": bool(flat) and all(isinstance(op.get("target_function") or op.get("target_component"), str) for op in flat),
        "payload_schema_valid": bool(flat) and all(isinstance(op.get("payload"), dict) and _is_json_safe(op.get("payload")) for op in flat),
    }


def _operation_blockers(checks: dict[str, bool]) -> list[str]:
    reasons = {
        "all_operations_would_execute_false": "all dry-run operations must have would_execute false",
        "all_scheduler_would_register_false": "scheduler dry-run operations must have would_register false",
        "all_broker_would_submit_false": "broker dry-run operations must have would_submit false",
        "deterministic_operation_order": "dry-run operation order must be deterministic",
        "operations_structured": "dry-run operations must be structured",
        "target_schema_valid": "every dry-run operation must include target_function or target_component",
        "payload_schema_valid": "every dry-run operation must include a JSON-safe payload dict",
    }
    return [reasons[key] for key, value in checks.items() if value is not True]


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    dry_run_packet: dict[str, Any],
    operation_checks: dict[str, Any],
    scheduler_checks: dict[str, Any],
    lifecycle_checks: dict[str, Any],
    risk_checks: dict[str, Any],
    state_checks: dict[str, Any],
    paper_broker_checks: dict[str, Any],
    safety_checks: dict[str, Any],
    ready: bool,
    blockers: list[str],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "stage4h3_automation_wiring_dry_run_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "dry_run_packet": dry_run_packet,
        "operation_checks": operation_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "risk_checks": risk_checks,
        "state_checks": state_checks,
        "paper_broker_checks": paper_broker_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4h4": {
            "ready_to_build_one_strategy_automation_enablement_gate": ready,
            "blockers": list(blockers),
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


def _dry_run_packet(
    available: bool,
    operations: dict[str, list[dict[str, Any]]],
    final_gate: dict[str, Any],
) -> dict[str, Any]:
    packet = {"available": available}
    packet.update(operations)
    packet["final_enablement_gate_preview"] = final_gate
    return packet


def _empty_operations() -> dict[str, list[dict[str, Any]]]:
    return {
        "scheduler_dry_run_operations": [],
        "lifecycle_dry_run_operations": [],
        "signal_to_intent_dry_run_operations": [],
        "intent_to_ticket_dry_run_operations": [],
        "ticket_to_paper_submit_dry_run_operations": [],
        "state_ledger_tracking_dry_run_operations": [],
        "risk_gate_dry_run_operations": [],
        "paper_broker_guard_dry_run_operations": [],
    }


def _final_enablement_gate(
    selected_strategy_id: str | None,
    ready: bool,
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "strategy_id": selected_strategy_id,
        "target_component": "Stage4H4EnablementGate",
        "paper_only": True,
        "enabled": False,
        "would_execute": False,
        "would_register": False,
        "would_submit": False,
        "live_trading_enabled": False,
        "ready_for_4h4_gate_design": ready,
        "blockers": list(blockers),
        "warnings": list(warnings),
    }


def _selected_strategy(strategy_id: str | None) -> dict[str, Any]:
    return {
        "selected_preview_strategy_id": strategy_id,
        "paper_only": True,
        "enabled": False,
    }


def _default_artifact_checks() -> dict[str, bool]:
    return {
        "stage4h2_preview_present": False,
        "stage4h2_preview_ready": False,
        "selected_strategy_present": False,
        "wiring_preview_available": False,
    }


def _default_scheduler_checks() -> dict[str, Any]:
    return {
        "scheduler_snapshot_present": False,
        "scheduler_already_enabled": False,
        "selected_strategy_job_already_enabled": False,
        "proposed_jobs_disabled": False,
        "proposed_jobs_would_register_false": False,
        "proposed_jobs_would_execute_false": False,
    }


def _default_lifecycle_checks() -> dict[str, Any]:
    return {
        "lifecycle_snapshot_present": False,
        "lifecycle_already_enabled": False,
        "lifecycle_transition_execution_enabled": False,
        "proposed_flows_would_execute_false": False,
    }


def _default_risk_checks() -> dict[str, Any]:
    return {
        "risk_snapshot_present": False,
        "kill_switch_available": False,
        "hard_halt_available": False,
        "daily_loss_limit_available": False,
        "risk_bypass_enabled": False,
    }


def _default_state_checks() -> dict[str, Any]:
    return {
        "state_snapshot_present": False,
        "active_halt": False,
        "unresolved_needs_reconciliation_count": 0,
        "active_intents_count": 0,
        "open_positions_count": 0,
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


def _default_safety_checks() -> dict[str, bool]:
    return {
        "no_live_orders": False,
        "no_market_data": False,
        "no_contract_qualification": False,
        "no_broker_submission_enabled": False,
        "no_scheduler_wiring_enabled": False,
        "no_lifecycle_wiring_enabled": False,
        "no_automated_paper_trading_enabled": False,
        "no_daemon_wiring_enabled": False,
        "no_all_strategy_enablement": False,
    }


def _scheduler_jobs(value: Any) -> list[dict[str, Any]]:
    jobs = _mapping(value).get("jobs") if isinstance(value, dict) else value
    return [item for item in _as_list(jobs) if isinstance(item, dict)]


def _lifecycle_flows(value: Any) -> list[dict[str, Any]]:
    flows = _mapping(value).get("flows") if isinstance(value, dict) else value
    return [item for item in _as_list(flows) if isinstance(item, dict)]


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


def _is_structured_preview(value: Any) -> bool:
    if isinstance(value, dict):
        return all(not isinstance(item, str) for item in value.values())
    if isinstance(value, list):
        return all(isinstance(item, dict) for item in value)
    return False


def _flatten_operations(operations: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for key in (
        "risk_gate_dry_run_operations",
        "state_ledger_tracking_dry_run_operations",
        "scheduler_dry_run_operations",
        "lifecycle_dry_run_operations",
        "signal_to_intent_dry_run_operations",
        "intent_to_ticket_dry_run_operations",
        "ticket_to_paper_submit_dry_run_operations",
        "paper_broker_guard_dry_run_operations",
    ):
        flat.extend(operations.get(key, []))
    return sorted(flat, key=lambda item: _safe_int(item.get("sequence_number")))


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
    return [str(item) for item in _as_list(value)]


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
    if isinstance(value, list):
        return any(_contains_truthy_flag(item, key) for item in value)
    return False


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


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


def _is_json_safe(value: Any) -> bool:
    try:
        _json_safe(value)
        return True
    except (TypeError, ValueError):
        return False


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
