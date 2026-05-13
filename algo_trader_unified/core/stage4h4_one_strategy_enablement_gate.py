"""Pure Stage 4H-4 one-strategy paper automation enablement gate report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
PAPER_IBKR_PORTS = {4004}
REQUIRED_OPERATOR_ACKNOWLEDGEMENTS = [
    "I understand this enables automated PAPER trading for one strategy only.",
    "I understand this does not enable live trading.",
    "I understand this does not enable all strategies.",
    "I verified risk controls and kill switches are available.",
    "I verified PAPER broker configuration is active.",
    "I understand scheduler/lifecycle activation must remain limited to the selected strategy.",
]
ORDERED_NEXT_STEPS = [
    "Build Stage 4H-5 one-strategy activation executor.",
    "Enable only the selected strategy in PAPER mode after explicit activation review.",
    "Keep live trading disabled.",
    "Keep all other strategies disabled.",
    "Verify PAPER broker config immediately before activation.",
]
DO_NOT_DO_YET = [
    "Do not enable live trading.",
    "Do not enable all strategies.",
    "Do not place orders now.",
    "Do not change strategy thresholds.",
    "Do not change position sizing.",
    "Do not bypass risk controls.",
    "Do not enable broker submission in 4H-4.",
]
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
DRY_RUN_OPERATION_GROUPS = (
    "risk_gate_dry_run_operations",
    "state_ledger_tracking_dry_run_operations",
    "scheduler_dry_run_operations",
    "lifecycle_dry_run_operations",
    "signal_to_intent_dry_run_operations",
    "intent_to_ticket_dry_run_operations",
    "ticket_to_paper_submit_dry_run_operations",
    "paper_broker_guard_dry_run_operations",
)


def build_stage4h4_one_strategy_enablement_gate_report(
    *,
    stage4h3_wiring_dry_run_report: dict | None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    operator_acknowledgements: list[str] | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a deterministic future activation gate from a Stage 4H-3 dry run."""

    try:
        return _json_safe(
            _build_report(
                stage4h3_wiring_dry_run_report=stage4h3_wiring_dry_run_report,
                state_snapshot=state_snapshot,
                risk_snapshot=risk_snapshot,
                scheduler_snapshot=scheduler_snapshot,
                lifecycle_snapshot=lifecycle_snapshot,
                paper_broker_snapshot=paper_broker_snapshot,
                operator_acknowledgements=operator_acknowledgements,
                now_provider=now_provider,
            )
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must remain data-only.
        message = f"unexpected report failure: {type(exc).__name__}: {exc}"
        generated_at = _generated_at(now_provider)
        return _json_safe(
            _base_report(
                generated_at=generated_at,
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_selected_strategy(None),
                acknowledgement_checks=_acknowledgement_checks(operator_acknowledgements),
                activation_candidate=_activation_candidate(None, False, False),
                proposed_activation_flags=_proposed_activation_flags(False),
                proposed_scheduler_activation=_proposed_scheduler_activation(None, False),
                proposed_lifecycle_activation=_proposed_lifecycle_activation(None, False),
                proposed_runtime_guards=_runtime_guards(),
                proposed_monitoring_requirements=_monitoring_requirements(),
                proposed_kill_switch_requirements=_kill_switch_requirements(),
                safety_checks=_default_safety_checks(),
                scheduler_checks=_default_scheduler_checks(),
                lifecycle_checks=_default_lifecycle_checks(),
                risk_checks=_default_risk_checks(),
                state_checks=_default_state_checks(),
                paper_broker_checks=_default_paper_broker_checks(),
                ready=False,
                blockers=[message],
                warnings=[],
                errors=[message],
            )
        )


def _build_report(
    *,
    stage4h3_wiring_dry_run_report: dict | None,
    state_snapshot: dict | None,
    risk_snapshot: dict | None,
    scheduler_snapshot: dict | None,
    lifecycle_snapshot: dict | None,
    paper_broker_snapshot: dict | None,
    operator_acknowledgements: list[str] | None,
    now_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    generated_at = _generated_at(now_provider)
    h3 = stage4h3_wiring_dry_run_report if isinstance(stage4h3_wiring_dry_run_report, dict) else None
    data = _mapping(h3)
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4h3_wiring_dry_run_report is None:
        blockers.append("Stage 4H-3 wiring dry-run report is missing")
    elif h3 is None:
        blockers.append("Stage 4H-3 wiring dry-run report must be a dict")
        errors.append("Stage 4H-3 wiring dry-run report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    artifact_checks = _artifact_checks(stage4h3_wiring_dry_run_report)
    if not artifact_checks["stage4h3_dry_run_present"]:
        blockers.append("Stage 4H-3 dry-run report is missing")
    if not artifact_checks["stage4h3_dry_run_ready"]:
        blockers.append("Stage 4H-3 dry run is not ready for Stage 4H-4")
    if not artifact_checks["selected_strategy_present"]:
        blockers.append("selected_preview_strategy_id is missing from Stage 4H-3 selected_strategy")
    if not artifact_checks["dry_run_packet_available"]:
        blockers.append("Stage 4H-3 dry_run_packet is not available")

    selected_id = _selected_strategy_id(data)
    selected_strategy = _selected_strategy(selected_id)
    dry_run_packet = _mapping(data.get("dry_run_packet"))

    operation_checks, operation_blockers = _dry_run_operation_checks(dry_run_packet)
    blockers.extend(operation_blockers)

    acknowledgement_checks = _acknowledgement_checks(operator_acknowledgements)
    if acknowledgement_checks["missing"]:
        blockers.append("required operator acknowledgements missing")

    scheduler_checks, scheduler_blockers, scheduler_warnings = _scheduler_checks(
        scheduler_snapshot, selected_id
    )
    blockers.extend(scheduler_blockers)
    warnings.extend(scheduler_warnings)

    lifecycle_checks, lifecycle_blockers, lifecycle_warnings = _lifecycle_checks(
        lifecycle_snapshot
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
        h3,
        state_snapshot,
        risk_snapshot,
        scheduler_snapshot,
        lifecycle_snapshot,
        paper_broker_snapshot,
    )
    blockers.extend(safety_blockers)

    proposed_structures_ok = selected_id is not None

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    all_gate_conditions_pass = (
        all(artifact_checks.values())
        and all(operation_checks.values())
        and acknowledgement_checks["exact_match"] is True
        and all(safety_checks.values())
        and not scheduler_checks["scheduler_already_enabled"]
        and not scheduler_checks["selected_strategy_job_already_enabled"]
        and scheduler_checks["proposed_scheduler_activation_structured"]
        and not lifecycle_checks["lifecycle_already_enabled"]
        and not lifecycle_checks["lifecycle_transition_execution_enabled"]
        and lifecycle_checks["proposed_lifecycle_activation_structured"]
        and (not risk_checks["risk_snapshot_present"] or (
            risk_checks["kill_switch_available"]
            and risk_checks["hard_halt_available"]
            and risk_checks["daily_loss_limit_available"]
        ))
        and not risk_checks["risk_bypass_enabled"]
        and not state_checks["active_halt"]
        and state_checks["unresolved_needs_reconciliation_count"] == 0
        and (
            state_checks["active_intents_count"] == 0
            or state_checks["active_intents_safe_for_enablement"] is True
        )
        and paper_checks["paper_broker_snapshot_present"]
        and paper_checks["paper_config_valid"]
        and proposed_structures_ok
    )
    all_gates_pass = all_gate_conditions_pass and not blocker_list and not error_list

    activation_candidate = _activation_candidate(selected_id, selected_id is not None, all_gates_pass)
    proposed_activation_flags = _proposed_activation_flags(all_gates_pass)
    proposed_scheduler_activation = _proposed_scheduler_activation(selected_id, all_gates_pass)
    proposed_lifecycle_activation = _proposed_lifecycle_activation(selected_id, all_gates_pass)

    if proposed_activation_flags["enable_live_trading"]:
        blockers.append("proposed activation must not enable live trading")
    if proposed_activation_flags["enable_all_strategies"]:
        blockers.append("proposed activation must not enable all strategies")
    if activation_candidate["max_enabled_strategy_count"] != 1:
        blockers.append("activation candidate must be limited to one strategy")
    if activation_candidate["broker_submission_allowed_next_phase"]:
        blockers.append("broker submission must remain disallowed by Stage 4H-4")

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    all_gates_pass = all_gate_conditions_pass and not blocker_list and not error_list
    ready = all_gates_pass

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
        acknowledgement_checks=acknowledgement_checks,
        activation_candidate=activation_candidate,
        proposed_activation_flags=proposed_activation_flags,
        proposed_scheduler_activation=proposed_scheduler_activation,
        proposed_lifecycle_activation=proposed_lifecycle_activation,
        proposed_runtime_guards=_runtime_guards(),
        proposed_monitoring_requirements=_monitoring_requirements(),
        proposed_kill_switch_requirements=_kill_switch_requirements(),
        safety_checks=safety_checks,
        scheduler_checks=scheduler_checks,
        lifecycle_checks=lifecycle_checks,
        risk_checks=risk_checks,
        state_checks=state_checks,
        paper_broker_checks=paper_checks,
        ready=ready,
        blockers=blocker_list,
        warnings=warning_list,
        errors=error_list,
    )


def _artifact_checks(report: dict | None) -> dict[str, bool]:
    data = _mapping(report)
    readiness = _mapping(data.get("readiness_for_stage4h4"))
    packet = _mapping(data.get("dry_run_packet"))
    selected = _selected_strategy_id(data)
    return {
        "stage4h3_dry_run_present": report is not None,
        "stage4h3_dry_run_ready": (
            data.get("stage4h3_automation_wiring_dry_run_report") is True
            and readiness.get("ready_to_build_one_strategy_automation_enablement_gate")
            is True
            and data.get("success") is True
        ),
        "selected_strategy_present": isinstance(selected, str) and bool(selected),
        "dry_run_packet_available": packet.get("available") is True,
    }


def _selected_strategy_id(report: dict[str, Any]) -> str | None:
    selected = _mapping(report.get("selected_strategy"))
    value = selected.get("selected_preview_strategy_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _dry_run_operation_checks(packet: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    operations = _flatten_operations(packet)
    scheduler_ops = [op for op in operations if op.get("group") == "scheduler_dry_run_operations"]
    broker_ops = [
        op for op in operations if op.get("group") == "ticket_to_paper_submit_dry_run_operations"
    ]
    checks = {
        "operations_present": bool(operations),
        "operations_structured": bool(operations) and all(isinstance(op, dict) for op in operations),
        "target_schema_valid": bool(operations)
        and all(isinstance(op.get("target_function") or op.get("target_component"), str) for op in operations),
        "payload_schema_valid": bool(operations)
        and all(isinstance(op.get("payload"), dict) and _is_json_safe(op.get("payload")) for op in operations),
        "all_operations_would_execute_false": bool(operations)
        and all(op.get("would_execute") is False for op in operations),
        "all_scheduler_would_register_false": all(
            op.get("would_register") is False for op in scheduler_ops
        ),
        "all_broker_would_submit_false": all(op.get("would_submit") is False for op in broker_ops),
    }
    reasons = {
        "operations_present": "Stage 4H-3 dry-run operations are missing",
        "operations_structured": "Stage 4H-3 dry-run operations must be structured",
        "target_schema_valid": "every dry-run operation must include target_function or target_component",
        "payload_schema_valid": "every dry-run operation must include a JSON-safe payload dict",
        "all_operations_would_execute_false": "every dry-run operation must have would_execute false",
        "all_scheduler_would_register_false": "scheduler dry-run operations must have would_register false",
        "all_broker_would_submit_false": "broker/order dry-run operations must have would_submit false",
    }
    return checks, [reasons[key] for key, value in checks.items() if value is not True]


def _acknowledgement_checks(value: Any) -> dict[str, Any]:
    provided = [item.strip() for item in _as_list(value) if isinstance(item, str)]
    provided_set = set(provided)
    missing = [item for item in REQUIRED_OPERATOR_ACKNOWLEDGEMENTS if item not in provided_set]
    return {
        "required": list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS),
        "provided": provided,
        "missing": missing,
        "exact_match": not missing,
    }


def _scheduler_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    proposed = _proposed_scheduler_activation(selected_strategy_id, False)
    checks = {
        "scheduler_snapshot_present": present,
        "scheduler_already_enabled": _contains_truthy_flag(data, "scheduler_automation_enabled")
        or _contains_truthy_flag(data, "scheduler_wiring_enabled"),
        "selected_strategy_job_already_enabled": _selected_strategy_job_enabled(
            data, selected_strategy_id
        ),
        "proposed_scheduler_activation_structured": _is_json_safe(proposed)
        and isinstance(proposed.get("jobs"), list),
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("scheduler snapshot missing; 4H-5 must verify scheduler remains disabled")
    if checks["scheduler_already_enabled"]:
        blockers.append("scheduler automation is already enabled")
    if checks["selected_strategy_job_already_enabled"]:
        blockers.append("selected strategy scheduler job is already enabled")
    if not checks["proposed_scheduler_activation_structured"]:
        blockers.append("proposed scheduler activation must be structured JSON-safe data")
    return checks, blockers, warnings


def _lifecycle_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    proposed = _proposed_lifecycle_activation(None, False)
    checks = {
        "lifecycle_snapshot_present": present,
        "lifecycle_already_enabled": _contains_truthy_flag(data, "lifecycle_automation_enabled")
        or _contains_truthy_flag(data, "lifecycle_wiring_enabled"),
        "lifecycle_transition_execution_enabled": _contains_truthy_flag(
            data, "lifecycle_transition_execution_enabled"
        ),
        "proposed_lifecycle_activation_structured": _is_json_safe(proposed)
        and isinstance(proposed.get("flows"), list),
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; 4H-5 must verify lifecycle remains disabled")
    if checks["lifecycle_already_enabled"]:
        blockers.append("lifecycle automation is already enabled")
    if checks["lifecycle_transition_execution_enabled"]:
        blockers.append("lifecycle transition execution is enabled")
    if not checks["proposed_lifecycle_activation_structured"]:
        blockers.append("proposed lifecycle activation must be structured JSON-safe data")
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
        warnings.append("risk snapshot missing; 4H-5 must verify risk controls before activation")
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
    active_intents_count = _safe_int(data.get("active_intents_count"))
    active_intents_safe = data.get("active_intents_safe_for_enablement") is True
    checks = {
        "state_snapshot_present": present,
        "active_halt": bool(data.get("active_halt") or data.get("halt_active")),
        "unresolved_needs_reconciliation_count": _safe_int(unresolved),
        "active_intents_count": active_intents_count,
        "active_intents_safe_for_enablement": active_intents_safe,
        "open_positions_count": _safe_int(data.get("open_positions_count")),
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("state snapshot missing; 4H-5 must verify state before activation")
        return checks, blockers, warnings
    if checks["active_halt"]:
        blockers.append("active halt is present")
    if checks["unresolved_needs_reconciliation_count"] > 0:
        blockers.append("unresolved NEEDS_RECONCILIATION is present")
    if active_intents_count > 0 and not active_intents_safe:
        blockers.append("active intents are present without explicit safe-for-enablement flag")
    if active_intents_count > 0 and active_intents_safe:
        warnings.append("active intents are present but explicitly marked safe for enablement")
    return checks, blockers, warnings


def _paper_broker_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    mode = data.get("mode")
    port = data.get("ibkr_port")
    live_enabled = data.get("live_trading_enabled") is True
    submission_enabled = data.get("broker_submission_enabled") is True
    paper_valid = (
        present
        and mode in (None, "PAPER")
        and data.get("paper_trading") is not False
        and port in (None, *PAPER_IBKR_PORTS)
        and not live_enabled
        and not submission_enabled
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
        blockers.append("paper broker snapshot is required for Stage 4H-5 readiness")
        warnings.append("paper broker snapshot missing; PAPER config must be verified")
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


def _activation_candidate(
    selected_strategy_id: str | None, available: bool, allowed_next_phase: bool
) -> dict[str, Any]:
    return {
        "available": available,
        "selected_strategy_id": selected_strategy_id,
        "paper_only": True,
        "one_strategy_only": True,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "activation_scope": "single_strategy_paper_only",
        "max_enabled_strategy_count": 1,
        "scheduler_activation_allowed_next_phase": allowed_next_phase,
        "lifecycle_activation_allowed_next_phase": allowed_next_phase,
        "broker_submission_allowed_next_phase": False,
    }


def _proposed_activation_flags(allowed: bool) -> dict[str, bool]:
    return {
        "enable_automated_paper_trading_for_selected_strategy": allowed,
        "enable_scheduler_for_selected_strategy": allowed,
        "enable_lifecycle_for_selected_strategy": allowed,
        "enable_broker_submission_for_selected_strategy": False,
        "enable_live_trading": False,
        "enable_all_strategies": False,
    }


def _proposed_scheduler_activation(
    selected_strategy_id: str | None, allowed: bool
) -> dict[str, Any]:
    strategy_id = selected_strategy_id or ""
    return {
        "available": bool(selected_strategy_id),
        "future_enablement_only": True,
        "would_register_in_4h4": False,
        "jobs": [
            {
                "job_id": f"stage4h5_one_strategy_paper_{strategy_id}",
                "strategy_id": strategy_id,
                "paper_only": True,
                "future_enablement_only": True,
                "would_register_in_4h4": False,
                "proposed_enabled_in_4h5": allowed,
            }
        ],
    }


def _proposed_lifecycle_activation(
    selected_strategy_id: str | None, allowed: bool
) -> dict[str, Any]:
    strategy_id = selected_strategy_id or ""
    return {
        "available": bool(selected_strategy_id),
        "future_enablement_only": True,
        "would_execute_in_4h4": False,
        "flows": [
            {
                "strategy_id": strategy_id,
                "paper_only": True,
                "future_enablement_only": True,
                "would_execute_in_4h4": False,
                "proposed_enabled_in_4h5": allowed,
            }
        ],
    }


def _runtime_guards() -> list[dict[str, Any]]:
    return [
        {"name": "paper-only mode guard", "required": True},
        {"name": "live-trading disabled guard", "required": True},
        {"name": "IBKR paper port guard", "required": True, "allowed_ports": sorted(PAPER_IBKR_PORTS)},
        {"name": "kill switch guard", "required": True},
        {"name": "hard halt guard", "required": True},
        {"name": "daily loss guard", "required": True},
        {"name": "state reconciliation guard", "required": True},
        {"name": "duplicate intent/order guard", "required": True},
        {"name": "one-strategy-only guard", "required": True},
        {"name": "all-strategy-disabled guard", "required": True},
        {"name": "no-open-unsafe-intents guard", "required": True},
    ]


def _monitoring_requirements() -> list[dict[str, Any]]:
    return [
        {"name": "selected strategy health visible", "required": True},
        {"name": "paper broker mode visible before activation", "required": True},
        {"name": "scheduler job state visible before activation", "required": True},
        {"name": "lifecycle flow state visible before activation", "required": True},
    ]


def _kill_switch_requirements() -> list[dict[str, Any]]:
    return [
        {"name": "operator kill switch available", "required": True},
        {"name": "hard halt available", "required": True},
        {"name": "daily loss halt available", "required": True},
    ]


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    acknowledgement_checks: dict[str, Any],
    activation_candidate: dict[str, Any],
    proposed_activation_flags: dict[str, Any],
    proposed_scheduler_activation: dict[str, Any],
    proposed_lifecycle_activation: dict[str, Any],
    proposed_runtime_guards: list[dict[str, Any]],
    proposed_monitoring_requirements: list[dict[str, Any]],
    proposed_kill_switch_requirements: list[dict[str, Any]],
    safety_checks: dict[str, Any],
    scheduler_checks: dict[str, Any],
    lifecycle_checks: dict[str, Any],
    risk_checks: dict[str, Any],
    state_checks: dict[str, Any],
    paper_broker_checks: dict[str, Any],
    ready: bool,
    blockers: list[str],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "stage4h4_one_strategy_enablement_gate_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "acknowledgement_checks": acknowledgement_checks,
        "activation_candidate": activation_candidate,
        "proposed_activation_flags": proposed_activation_flags,
        "proposed_scheduler_activation": proposed_scheduler_activation,
        "proposed_lifecycle_activation": proposed_lifecycle_activation,
        "proposed_runtime_guards": proposed_runtime_guards,
        "proposed_monitoring_requirements": proposed_monitoring_requirements,
        "proposed_kill_switch_requirements": proposed_kill_switch_requirements,
        "safety_checks": safety_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "risk_checks": risk_checks,
        "state_checks": state_checks,
        "paper_broker_checks": paper_broker_checks,
        "readiness_for_stage4h5": {
            "ready_to_build_one_strategy_activation_executor": ready,
            "blockers": list(blockers),
            "warnings": list(warnings),
        },
        "recommendations": {
            "ordered_next_steps": list(ORDERED_NEXT_STEPS),
            "do_not_do_yet": list(DO_NOT_DO_YET),
        },
        "success": True,
        "errors": list(errors),
        "warnings": list(warnings),
    }


def _selected_strategy(strategy_id: str | None) -> dict[str, Any]:
    return {
        "selected_strategy_id": strategy_id,
        "paper_only": True,
        "enabled_now": False,
    }


def _default_artifact_checks() -> dict[str, bool]:
    return {
        "stage4h3_dry_run_present": False,
        "stage4h3_dry_run_ready": False,
        "selected_strategy_present": False,
        "dry_run_packet_available": False,
    }


def _default_scheduler_checks() -> dict[str, Any]:
    return {
        "scheduler_snapshot_present": False,
        "scheduler_already_enabled": False,
        "selected_strategy_job_already_enabled": False,
        "proposed_scheduler_activation_structured": False,
    }


def _default_lifecycle_checks() -> dict[str, Any]:
    return {
        "lifecycle_snapshot_present": False,
        "lifecycle_already_enabled": False,
        "lifecycle_transition_execution_enabled": False,
        "proposed_lifecycle_activation_structured": False,
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
        "active_intents_safe_for_enablement": False,
        "open_positions_count": 0,
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


def _flatten_operations(packet: dict[str, Any]) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for key in DRY_RUN_OPERATION_GROUPS:
        for item in _as_list(packet.get(key)):
            if isinstance(item, dict):
                operation = dict(item)
                operation.setdefault("group", key)
                operations.append(operation)
            else:
                operations.append({"group": key, "malformed_operation": str(item)})
    return operations


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
