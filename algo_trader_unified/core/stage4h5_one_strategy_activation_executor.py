"""Pure Stage 4H-5 one-strategy paper activation executor report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
PAPER_IBKR_PORTS = {4004}
REQUIRED_OPERATOR_ACKNOWLEDGEMENTS = [
    "I understand this activates automated PAPER trading for one strategy only.",
    "I understand this does not enable live trading.",
    "I understand this does not enable all strategies.",
    "I verified PAPER broker configuration is active.",
    "I verified risk controls and kill switches are active.",
    "I understand broker order submission remains separately gated.",
]
ORDERED_NEXT_STEPS = [
    "Build Stage 4H-6 one-strategy activation acceptance report.",
    "Verify the activation artifact/state before any scheduled run.",
    "Keep live trading disabled.",
    "Keep all other strategies disabled.",
    "Keep broker submission separately gated.",
]
DO_NOT_DO_YET = [
    "Do not enable live trading.",
    "Do not enable all strategies.",
    "Do not place orders now.",
    "Do not change strategy thresholds.",
    "Do not change position sizing.",
    "Do not bypass risk controls.",
    "Do not enable broker submission broadly.",
]
UNSAFE_TRUE_FLAGS = (
    "live_orders_enabled",
    "live_trading_enabled",
    "broker_submission_enabled",
    "market_data_enabled",
    "contract_qualification_enabled",
    "all_strategies_enabled",
    "enable_all_strategies",
    "daemon_wiring_enabled",
)


def build_stage4h5_one_strategy_activation_executor_report(
    *,
    stage4h4_enablement_gate_report: dict | None,
    activation_writer,
    audit_writer=None,
    operator_acknowledgements: list[str] | None = None,
    allow_activation_write: bool = False,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build and optionally apply a one-strategy PAPER activation artifact."""

    try:
        return _json_safe(
            _build_report(
                stage4h4_enablement_gate_report=stage4h4_enablement_gate_report,
                activation_writer=activation_writer,
                audit_writer=audit_writer,
                operator_acknowledgements=operator_acknowledgements,
                allow_activation_write=allow_activation_write,
                state_snapshot=state_snapshot,
                risk_snapshot=risk_snapshot,
                scheduler_snapshot=scheduler_snapshot,
                lifecycle_snapshot=lifecycle_snapshot,
                paper_broker_snapshot=paper_broker_snapshot,
                now_provider=now_provider,
            )
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must remain data-only.
        message = f"unexpected executor failure: {type(exc).__name__}: {exc}"
        generated_at = _generated_at(now_provider)
        return _json_safe(
            _base_report(
                generated_at=generated_at,
                gates=_default_gates([message]),
                acknowledgement_checks=_acknowledgement_checks(operator_acknowledgements),
                selected_strategy=_selected_strategy(None),
                activation_payload=_activation_payload(
                    selected_strategy_id="",
                    generated_at=generated_at,
                    required_runtime_guards=[],
                    required_monitoring=[],
                    required_kill_switches=[],
                ),
                execution=_default_execution(),
                applied_operations=[],
                skipped_operations=["activation_writer.activate_one_strategy"],
                rollback_required=False,
                rollback_status="no rollback required; activation was not attempted",
                ready=False,
                blockers=[message],
                warnings=[],
                errors=[message],
            )
        )


def _build_report(
    *,
    stage4h4_enablement_gate_report: dict | None,
    activation_writer: Any,
    audit_writer: Any,
    operator_acknowledgements: list[str] | None,
    allow_activation_write: bool,
    state_snapshot: dict | None,
    risk_snapshot: dict | None,
    scheduler_snapshot: dict | None,
    lifecycle_snapshot: dict | None,
    paper_broker_snapshot: dict | None,
    now_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    generated_at = _generated_at(now_provider)
    h4 = stage4h4_enablement_gate_report if isinstance(stage4h4_enablement_gate_report, dict) else None
    data = _mapping(h4)
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if not allow_activation_write:
        blockers.append("allow_activation_write must be true")
    if stage4h4_enablement_gate_report is None:
        blockers.append("Stage 4H-4 enablement gate report is missing")
    elif h4 is None:
        blockers.append("Stage 4H-4 enablement gate report must be a dict")
        errors.append("Stage 4H-4 enablement gate report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    selected_id = _selected_strategy_id(data)
    h4_gate_valid, h4_gate_ready, h4_blockers = _stage4h4_checks(data, selected_id)
    blockers.extend(h4_blockers)

    payload_warnings: list[str] = []
    runtime_guards = _proposal_list(data, "proposed_runtime_guards", payload_warnings)
    monitoring = _proposal_list(data, "proposed_monitoring_requirements", payload_warnings)
    kill_switches = _proposal_list(data, "proposed_kill_switch_requirements", payload_warnings)
    warnings.extend(payload_warnings)

    acknowledgement_checks = _acknowledgement_checks(operator_acknowledgements)
    if acknowledgement_checks["missing"]:
        blockers.append("required operator acknowledgements missing")

    paper_checks, paper_blockers, paper_warnings = _paper_broker_checks(paper_broker_snapshot)
    blockers.extend(paper_blockers)
    warnings.extend(paper_warnings)

    risk_checks, risk_blockers, risk_warnings = _risk_checks(risk_snapshot)
    blockers.extend(risk_blockers)
    warnings.extend(risk_warnings)

    state_checks, state_blockers, state_warnings = _state_checks(state_snapshot)
    blockers.extend(state_blockers)
    warnings.extend(state_warnings)

    scheduler_checks, scheduler_blockers, scheduler_warnings = _scheduler_checks(
        scheduler_snapshot, selected_id
    )
    blockers.extend(scheduler_blockers)
    warnings.extend(scheduler_warnings)

    lifecycle_checks, lifecycle_blockers, lifecycle_warnings = _lifecycle_checks(lifecycle_snapshot)
    blockers.extend(lifecycle_blockers)
    warnings.extend(lifecycle_warnings)

    safety_checks, safety_blockers = _safety_checks(
        data,
        state_snapshot,
        risk_snapshot,
        scheduler_snapshot,
        lifecycle_snapshot,
        paper_broker_snapshot,
    )
    blockers.extend(safety_blockers)

    activation_payload = _activation_payload(
        selected_strategy_id=selected_id or "",
        generated_at=generated_at,
        required_runtime_guards=runtime_guards,
        required_monitoring=monitoring,
        required_kill_switches=kill_switches,
    )
    gates = {
        "allow_activation_write": allow_activation_write is True,
        "stage4h4_gate_valid": h4_gate_valid,
        "stage4h4_gate_ready": h4_gate_ready,
        "selected_strategy_present": isinstance(selected_id, str) and bool(selected_id),
        "acknowledgements_ok": acknowledgement_checks["exact_match"] is True,
        "paper_broker_config_ok": paper_checks["paper_config_valid"] is True,
        "risk_controls_ok": risk_checks["risk_controls_ok"] is True,
        "state_clean": state_checks["state_clean"] is True,
        "scheduler_clean": scheduler_checks["scheduler_clean"] is True,
        "lifecycle_clean": lifecycle_checks["lifecycle_clean"] is True,
        "passed": False,
        "reasons": [],
    }

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    gates["passed"] = all(value is True for key, value in gates.items() if key not in ("passed", "reasons")) and not blocker_list and not error_list
    gates["reasons"] = blocker_list

    execution = _default_execution()
    applied_operations: list[str] = []
    skipped_operations: list[str] = []
    rollback_required = False
    rollback_status = "no rollback required"

    if not gates["passed"]:
        skipped_operations.extend(["activation_writer.activate_one_strategy"])
        if audit_writer is not None:
            skipped_operations.append("audit_writer.append_activation_audit")
    else:
        execution["attempted"] = True
        execution["activation_write_attempted"] = True
        activation_result = _call_activation_writer(activation_writer, activation_payload)
        if activation_result["ok"]:
            writer_response = activation_result["response"]
            status = writer_response.get("status")
            if status == "conflict":
                execution["failed_step"] = "activation_writer.activate_one_strategy"
                execution["failure_reason"] = "activation writer returned conflict"
                errors.append("activation writer returned conflict")
                skipped_operations.append("audit_writer.append_activation_audit")
            elif status == "already_exists" and not _already_exists_matches(
                writer_response.get("record"), selected_id
            ):
                execution["failed_step"] = "activation_writer.activate_one_strategy"
                execution["failure_reason"] = "existing activation record does not match requested one-strategy PAPER activation"
                errors.append(execution["failure_reason"])
                skipped_operations.append("audit_writer.append_activation_audit")
            else:
                execution["activation_write_succeeded"] = True
                applied_operations.append("activation_writer.activate_one_strategy")
                if audit_writer is not None:
                    execution["audit_write_attempted"] = True
                    audit_result = _call_audit_writer(
                        audit_writer,
                        {
                            "source_stage": "4H-5",
                            "generated_at": generated_at,
                            "selected_strategy_id": selected_id,
                            "activation_payload": activation_payload,
                            "activation_writer_response": writer_response,
                        },
                    )
                    if audit_result["ok"]:
                        execution["audit_write_succeeded"] = True
                        applied_operations.append("audit_writer.append_activation_audit")
                    else:
                        execution["failed_step"] = "audit_writer.append_activation_audit"
                        execution["failure_reason"] = audit_result["failure_reason"]
                        errors.append(audit_result["failure_reason"])
                        rollback_required = True
                        rollback_status = (
                            "manual rollback is required using standard backups or manual removal "
                            "of the activation artifact"
                        )
                        skipped_operations.append("automated_rollback")
                execution["completed"] = (
                    execution["activation_write_succeeded"] is True
                    and (audit_writer is None or execution["audit_write_succeeded"] is True)
                    and not rollback_required
                )
        else:
            execution["failed_step"] = "activation_writer.activate_one_strategy"
            execution["failure_reason"] = activation_result["failure_reason"]
            errors.append(activation_result["failure_reason"])
            if audit_writer is not None:
                skipped_operations.append("audit_writer.append_activation_audit")

    error_list = _dedupe(errors)
    ready = _ready_for_stage4h6(
        gates_passed=gates["passed"] is True,
        execution=execution,
        audit_writer_supplied=audit_writer is not None,
        rollback_required=rollback_required,
        activation_payload=activation_payload,
        errors=error_list,
        safety_checks=safety_checks,
    )
    return _base_report(
        generated_at=generated_at,
        gates=gates,
        acknowledgement_checks=acknowledgement_checks,
        selected_strategy=_selected_strategy(selected_id),
        activation_payload=activation_payload,
        execution=execution,
        applied_operations=applied_operations,
        skipped_operations=skipped_operations,
        rollback_required=rollback_required,
        rollback_status=rollback_status,
        ready=ready,
        blockers=blocker_list if not ready else [],
        warnings=warning_list,
        errors=error_list,
    )


def _stage4h4_checks(data: dict[str, Any], selected_id: str | None) -> tuple[bool, bool, list[str]]:
    readiness = _mapping(data.get("readiness_for_stage4h5"))
    candidate = _mapping(data.get("activation_candidate"))
    flags = _mapping(data.get("proposed_activation_flags"))
    blockers: list[str] = []
    gate_valid = data.get("stage4h4_one_strategy_enablement_gate_report") is True
    gate_ready = (
        readiness.get("ready_to_build_one_strategy_activation_executor") is True
        and data.get("success") is True
        and not _as_list(data.get("errors"))
    )
    checks = {
        "stage4h4 gate report must be valid": gate_valid,
        "Stage 4H-4 gate is not ready for Stage 4H-5": gate_ready,
        "selected_strategy.selected_strategy_id is missing": isinstance(selected_id, str)
        and bool(selected_id),
        "activation candidate is not available": candidate.get("available") is True,
        "activation candidate must be one strategy only": candidate.get("one_strategy_only") is True,
        "activation candidate must be paper only": candidate.get("paper_only") is True,
        "activation candidate max enabled strategy count must be 1": candidate.get("max_enabled_strategy_count") == 1,
        "proposed activation must not enable live trading": flags.get("enable_live_trading") is False,
        "proposed activation must not enable all strategies": flags.get("enable_all_strategies") is False,
        "proposed activation must enable selected strategy PAPER automation": flags.get("enable_automated_paper_trading_for_selected_strategy") is True,
        "proposed activation must enable selected strategy scheduler only": flags.get("enable_scheduler_for_selected_strategy") is True,
        "proposed activation must enable selected strategy lifecycle only": flags.get("enable_lifecycle_for_selected_strategy") is True,
        "proposed activation must keep broker submission disabled": flags.get("enable_broker_submission_for_selected_strategy") is False,
    }
    blockers.extend(reason for reason, ok in checks.items() if ok is not True)
    return gate_valid, gate_ready, blockers


def _selected_strategy_id(report: dict[str, Any]) -> str | None:
    selected = _mapping(report.get("selected_strategy"))
    value = selected.get("selected_strategy_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _proposal_list(data: dict[str, Any], key: str, warnings: list[str]) -> list[Any]:
    value = data.get(key)
    if isinstance(value, list):
        return _json_safe(value)
    warnings.append(f"{key} missing or not a list; using empty list")
    return []


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
        blockers.append("paper broker snapshot is required")
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


def _risk_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    checks = {
        "risk_snapshot_present": present,
        "kill_switch_available": data.get("kill_switch_available") is True,
        "hard_halt_available": data.get("hard_halt_available") is True,
        "daily_loss_limit_available": data.get("daily_loss_limit_available") is True,
        "risk_bypass_enabled": data.get("risk_bypass_enabled") is True,
    }
    checks["risk_controls_ok"] = (
        (not present)
        or (
            checks["kill_switch_available"]
            and checks["hard_halt_available"]
            and checks["daily_loss_limit_available"]
            and not checks["risk_bypass_enabled"]
        )
    )
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("risk snapshot missing; risk controls should be verified before activation")
        return checks, blockers, warnings
    if checks["risk_bypass_enabled"]:
        blockers.append("risk controls are bypassed")
    for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
        if checks[key] is not True:
            blockers.append(f"{key} must be true in supplied risk snapshot")
    return checks, blockers, warnings


def _state_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    active_intents_count = _safe_int(data.get("active_intents_count"))
    active_intents_safe = data.get("active_intents_safe_for_enablement") is True
    checks = {
        "state_snapshot_present": present,
        "active_halt": bool(data.get("active_halt") or data.get("halt_active")),
        "unresolved_needs_reconciliation_count": _safe_int(
            _first_present(
                data.get("unresolved_needs_reconciliation_count"),
                data.get("needs_reconciliation_count"),
                default=0,
            )
        ),
        "active_intents_count": active_intents_count,
        "active_intents_safe_for_enablement": active_intents_safe,
        "state_clean": True,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("state snapshot missing; state should be verified before activation")
        return checks, blockers, warnings
    if checks["active_halt"]:
        blockers.append("active halt is present")
    if checks["unresolved_needs_reconciliation_count"] > 0:
        blockers.append("unresolved NEEDS_RECONCILIATION is present")
    if active_intents_count > 0 and not active_intents_safe:
        blockers.append("active intents are present without explicit safe-for-enablement flag")
    checks["state_clean"] = not blockers
    return checks, blockers, warnings


def _scheduler_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    already_enabled = _contains_truthy_flag(data, "scheduler_automation_enabled") or _contains_truthy_flag(
        data, "scheduler_wiring_enabled"
    )
    selected_enabled = _selected_strategy_job_enabled(data, selected_strategy_id)
    checks = {
        "scheduler_snapshot_present": present,
        "scheduler_already_enabled": already_enabled,
        "selected_strategy_job_already_enabled": selected_enabled,
        "scheduler_clean": not already_enabled and not selected_enabled,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("scheduler snapshot missing; scheduler should be verified before activation")
    if already_enabled:
        blockers.append("scheduler automation is already enabled")
    if selected_enabled:
        blockers.append("selected strategy scheduler job is already enabled")
    return checks, blockers, warnings


def _lifecycle_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    already_enabled = _contains_truthy_flag(data, "lifecycle_automation_enabled") or _contains_truthy_flag(
        data, "lifecycle_wiring_enabled"
    )
    transition_enabled = _contains_truthy_flag(data, "lifecycle_transition_execution_enabled")
    checks = {
        "lifecycle_snapshot_present": present,
        "lifecycle_already_enabled": already_enabled,
        "lifecycle_transition_execution_enabled": transition_enabled,
        "lifecycle_clean": not already_enabled and not transition_enabled,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; lifecycle should be verified before activation")
    if already_enabled:
        blockers.append("lifecycle automation is already enabled")
    if transition_enabled:
        blockers.append("lifecycle transition execution is enabled")
    return checks, blockers, warnings


def _safety_checks(*sources: Any) -> tuple[dict[str, bool], list[str]]:
    unsafe = {key: any(_contains_truthy_flag(source, key) for source in sources) for key in UNSAFE_TRUE_FLAGS}
    checks = {
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "broker_submission_enabled": False,
        "market_data_enabled": False,
        "contract_qualification_enabled": False,
        "production_daemon_wiring_enabled": False,
        "no_unsafe_safety_flags": not any(unsafe.values()),
    }
    reasons = []
    if unsafe["live_orders_enabled"] or unsafe["live_trading_enabled"]:
        reasons.append("live order or live trading flag is enabled")
    if unsafe["broker_submission_enabled"]:
        reasons.append("broker submission automation is enabled")
    if unsafe["market_data_enabled"]:
        reasons.append("market data flag is enabled")
    if unsafe["contract_qualification_enabled"]:
        reasons.append("contract qualification flag is enabled")
    if unsafe["all_strategies_enabled"] or unsafe["enable_all_strategies"]:
        reasons.append("all-strategy automation is enabled or requested")
    if unsafe["daemon_wiring_enabled"]:
        reasons.append("daemon wiring is enabled")
    return checks, reasons


def _activation_payload(
    *,
    selected_strategy_id: str,
    generated_at: str,
    required_runtime_guards: list[Any],
    required_monitoring: list[Any],
    required_kill_switches: list[Any],
) -> dict[str, Any]:
    return {
        "selected_strategy_id": selected_strategy_id,
        "paper_only": True,
        "activation_scope": "single_strategy_paper_only",
        "enabled_strategy_count": 1,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "automated_paper_trading_enabled_for_selected_strategy": True,
        "scheduler_enabled_for_selected_strategy": True,
        "lifecycle_enabled_for_selected_strategy": True,
        "broker_submission_enabled": False,
        "source_stage": "4H-5",
        "generated_at": generated_at,
        "required_runtime_guards": _json_safe(required_runtime_guards),
        "required_monitoring": _json_safe(required_monitoring),
        "required_kill_switches": _json_safe(required_kill_switches),
    }


def _call_activation_writer(writer: Any, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = writer.activate_one_strategy(payload)
    except Exception as exc:  # noqa: BLE001 - injected writer failures must be reported.
        return {"ok": False, "failure_reason": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "response": _json_safe(_mapping(response))}


def _call_audit_writer(writer: Any, event: dict[str, Any]) -> dict[str, Any]:
    try:
        response = writer.append_activation_audit(_json_safe(event))
    except Exception as exc:  # noqa: BLE001 - injected writer failures must be reported.
        return {"ok": False, "failure_reason": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "response": _json_safe(_mapping(response))}


def _already_exists_matches(record: Any, selected_strategy_id: str | None) -> bool:
    data = _mapping(record)
    return (
        data.get("selected_strategy_id") == selected_strategy_id
        and data.get("paper_only") is True
        and data.get("live_trading_enabled") is False
        and data.get("all_strategies_enabled") is False
        and data.get("enabled_strategy_count") == 1
    )


def _ready_for_stage4h6(
    *,
    gates_passed: bool,
    execution: dict[str, Any],
    audit_writer_supplied: bool,
    rollback_required: bool,
    activation_payload: dict[str, Any],
    errors: list[str],
    safety_checks: dict[str, Any],
) -> bool:
    return (
        gates_passed
        and execution["attempted"] is True
        and execution["activation_write_succeeded"] is True
        and (not audit_writer_supplied or execution["audit_write_succeeded"] is True)
        and rollback_required is False
        and activation_payload["enabled_strategy_count"] == 1
        and activation_payload["paper_only"] is True
        and activation_payload["live_trading_enabled"] is False
        and activation_payload["all_strategies_enabled"] is False
        and activation_payload["broker_submission_enabled"] is False
        and not errors
        and safety_checks["no_unsafe_safety_flags"] is True
    )


def _base_report(
    *,
    generated_at: str,
    gates: dict[str, Any],
    acknowledgement_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    activation_payload: dict[str, Any],
    execution: dict[str, Any],
    applied_operations: list[str],
    skipped_operations: list[str],
    rollback_required: bool,
    rollback_status: str,
    ready: bool,
    blockers: list[str],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "dry_run": False,
        "stage4h5_one_strategy_activation_executor_report": True,
        "generated_at": generated_at,
        "gates": gates,
        "acknowledgement_checks": acknowledgement_checks,
        "selected_strategy": selected_strategy,
        "activation_payload": activation_payload,
        "execution": execution,
        "applied_operations": list(applied_operations),
        "skipped_operations": list(skipped_operations),
        "rollback": {
            "rollback_required": rollback_required,
            "rollback_attempted": False,
            "rollback_status": rollback_status,
            "rollback_limitations": "no automated rollback is supported in this phase",
        },
        "safety": {
            "live_trading_enabled": False,
            "all_strategies_enabled": False,
            "broker_submission_enabled": False,
            "market_data_enabled": False,
            "contract_qualification_enabled": False,
            "production_daemon_wiring_enabled": False,
        },
        "readiness_for_stage4h6": {
            "ready_to_build_one_strategy_activation_acceptance_report": ready,
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


def _selected_strategy(strategy_id: str | None) -> dict[str, Any]:
    return {"selected_strategy_id": strategy_id, "paper_only": True}


def _default_gates(reasons: list[str]) -> dict[str, Any]:
    return {
        "allow_activation_write": False,
        "stage4h4_gate_valid": False,
        "stage4h4_gate_ready": False,
        "selected_strategy_present": False,
        "acknowledgements_ok": False,
        "paper_broker_config_ok": False,
        "risk_controls_ok": False,
        "state_clean": False,
        "scheduler_clean": False,
        "lifecycle_clean": False,
        "passed": False,
        "reasons": list(reasons),
    }


def _default_execution() -> dict[str, Any]:
    return {
        "attempted": False,
        "activation_write_attempted": False,
        "activation_write_succeeded": False,
        "audit_write_attempted": False,
        "audit_write_succeeded": False,
        "completed": False,
        "failed_step": None,
        "failure_reason": None,
    }


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
    """Deduplicate a list of strings while preserving order."""
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
