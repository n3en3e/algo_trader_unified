"""Stage 4I-5 one-strategy scheduler/lifecycle activation executor."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
PAPER_IBKR_PORTS = {4004}
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding"
)
ROLLBACK_LIMITATIONS = "no automated rollback is supported in this phase"
ROLLBACK_STATUS = (
    "manual rollback is required using standard backups or manual removal of activation artifacts"
)
REQUIRED_OPERATOR_ACKNOWLEDGEMENTS = [
    "I understand this activates the scheduler/lifecycle path for one PAPER strategy only.",
    "I understand this does not enable live trading.",
    "I understand this does not enable all strategies.",
    "I understand broker order submission remains separately gated.",
    "I verified state, risk, scheduler, lifecycle, paper broker, and market window snapshots.",
    "I understand strategy scans and orders are not executed by this activation executor.",
]
ORDERED_NEXT_STEPS = [
    "Build Stage 4I-6 scheduler/lifecycle activation acceptance report.",
    "Before accepting activation, verify scheduler/lifecycle activation artifacts and snapshots.",
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
    "Do not enable broker submission broadly.",
]


def build_stage4i5_scheduler_lifecycle_activation_executor_report(
    *,
    stage4i4_activation_gate_report: dict | None,
    scheduler_activation_writer,
    lifecycle_activation_writer,
    audit_writer=None,
    activation_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    market_window_snapshot: dict | None = None,
    operator_acknowledgements: list[str] | None = None,
    allow_scheduler_lifecycle_activation: bool = False,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Validate and apply one PAPER strategy activation through injected writers."""

    try:
        return _json_safe(
            _build_report(
                stage4i4_activation_gate_report=stage4i4_activation_gate_report,
                scheduler_activation_writer=scheduler_activation_writer,
                lifecycle_activation_writer=lifecycle_activation_writer,
                audit_writer=audit_writer,
                activation_snapshot=activation_snapshot,
                state_snapshot=state_snapshot,
                risk_snapshot=risk_snapshot,
                scheduler_snapshot=scheduler_snapshot,
                lifecycle_snapshot=lifecycle_snapshot,
                paper_broker_snapshot=paper_broker_snapshot,
                market_window_snapshot=market_window_snapshot,
                operator_acknowledgements=operator_acknowledgements,
                allow_scheduler_lifecycle_activation=allow_scheduler_lifecycle_activation,
                now_provider=now_provider,
            )
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must stay data-only.
        message = f"unexpected activation executor failure: {type(exc).__name__}: {exc}"
        generated_at = _generated_at(now_provider)
        return _json_safe(
            _base_report(
                generated_at=generated_at,
                gates=_default_gates([message]),
                acknowledgement_checks=_acknowledgement_checks(operator_acknowledgements),
                selected_strategy=_selected_strategy_payload(None, {}),
                scheduler_activation_payload=_scheduler_payload(None, generated_at),
                lifecycle_activation_payload=_lifecycle_payload(None, generated_at),
                execution=_default_execution(failed_step="unexpected", failure_reason=message),
                applied_operations=[],
                skipped_operations=[
                    _operation("scheduler_activation", "skipped_after_unexpected_failure"),
                    _operation("lifecycle_activation", "skipped_after_unexpected_failure"),
                    _operation("audit", "skipped_after_unexpected_failure"),
                ],
                rollback_required=False,
                rollback_status="not required; no activation writes were attempted",
                blockers=[message],
                warnings=[],
                errors=[message],
            )
        )


def _build_report(
    *,
    stage4i4_activation_gate_report: dict | None,
    scheduler_activation_writer: Any,
    lifecycle_activation_writer: Any,
    audit_writer: Any,
    activation_snapshot: dict | None,
    state_snapshot: dict | None,
    risk_snapshot: dict | None,
    scheduler_snapshot: dict | None,
    lifecycle_snapshot: dict | None,
    paper_broker_snapshot: dict | None,
    market_window_snapshot: dict | None,
    operator_acknowledgements: list[str] | None,
    allow_scheduler_lifecycle_activation: bool,
    now_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    generated_at = _generated_at(now_provider)
    data = _mapping(stage4i4_activation_gate_report)
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4i4_activation_gate_report is None:
        blockers.append("Stage 4I-4 scheduler/lifecycle activation gate report is missing")
    elif not isinstance(stage4i4_activation_gate_report, dict):
        blockers.append("Stage 4I-4 scheduler/lifecycle activation gate report must be a dict")
        errors.append("Stage 4I-4 scheduler/lifecycle activation gate report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    selected_strategy_id = _selected_strategy_id(data)
    selected_strategy = _selected_strategy_payload(
        selected_strategy_id, _mapping(data.get("selected_strategy"))
    )
    candidate = _mapping(data.get("scheduler_lifecycle_activation_candidate"))
    proposed_scheduler = _mapping(data.get("proposed_scheduler_activation"))
    proposed_lifecycle = _mapping(data.get("proposed_lifecycle_activation"))

    acknowledgement_checks = _acknowledgement_checks(operator_acknowledgements)
    if acknowledgement_checks["exact_match"] is not True:
        blockers.append("required operator acknowledgements are missing")

    activation_checks, activation_blockers, activation_warnings = _activation_checks(
        activation_snapshot, selected_strategy_id
    )
    state_checks, state_blockers, state_warnings = _state_checks(state_snapshot)
    risk_checks, risk_blockers, risk_warnings = _risk_checks(risk_snapshot)
    scheduler_checks, scheduler_blockers, scheduler_warnings = _scheduler_checks(
        scheduler_snapshot, selected_strategy_id
    )
    lifecycle_checks, lifecycle_blockers, lifecycle_warnings = _lifecycle_checks(
        lifecycle_snapshot
    )
    paper_broker_checks, broker_blockers, broker_warnings = _paper_broker_checks(
        paper_broker_snapshot
    )
    market_window_checks, market_blockers, market_warnings = _market_window_checks(
        market_window_snapshot
    )
    warnings.extend(
        activation_warnings
        + state_warnings
        + risk_warnings
        + scheduler_warnings
        + lifecycle_warnings
        + broker_warnings
        + market_warnings
    )
    blockers.extend(
        activation_blockers
        + state_blockers
        + risk_blockers
        + scheduler_blockers
        + lifecycle_blockers
        + broker_blockers
        + market_blockers
    )

    stage4i4_gate_valid = data.get("stage4i4_scheduler_lifecycle_activation_gate_report") is True
    stage4i4_gate_ready = (
        _mapping(data.get("readiness_for_stage4i5")).get(
            "ready_to_build_scheduler_lifecycle_activation_executor"
        )
        is True
        and data.get("success") is True
    )
    stage4i4_gate_flag = data.get("stage4i4_scheduler_lifecycle_activation_gate_report") is True
    readiness_flag = (
        _mapping(data.get("readiness_for_stage4i5")).get(
            "ready_to_build_scheduler_lifecycle_activation_executor"
        )
        is True
    )

    _require(blockers, allow_scheduler_lifecycle_activation is True, "manual activation allow flag is not true")
    _require(blockers, isinstance(stage4i4_activation_gate_report, dict), "Stage 4I-4 report must exist")
    _require(blockers, stage4i4_gate_flag, "Stage 4I-4 gate report flag is not true")
    _require(blockers, readiness_flag, "Stage 4I-4 readiness for Stage 4I-5 is not true")
    _require(blockers, selected_strategy["selected_strategy_id"] is not None, "selected strategy id is missing")
    _require(blockers, selected_strategy["paper_only"] is True, "selected strategy must be paper_only true")
    _require(
        blockers,
        selected_strategy["one_strategy_only"] is True,
        "selected strategy must be one_strategy_only true",
    )

    blockers.extend(_candidate_blockers(candidate))
    blockers.extend(_proposed_scheduler_blockers(proposed_scheduler))
    blockers.extend(_proposed_lifecycle_blockers(proposed_lifecycle))

    safety = _safety()
    safety_checks = _safety_checks(
        data,
        activation_snapshot if isinstance(activation_snapshot, dict) else {},
        scheduler_snapshot if isinstance(scheduler_snapshot, dict) else {},
        lifecycle_snapshot if isinstance(lifecycle_snapshot, dict) else {},
        paper_broker_snapshot if isinstance(paper_broker_snapshot, dict) else {},
    )
    blockers.extend(_safety_blockers(safety_checks))

    scheduler_snapshot_clean = not scheduler_blockers
    lifecycle_snapshot_clean = not lifecycle_blockers
    state_clean = not state_blockers
    risk_controls_ok = not risk_blockers
    paper_broker_config_ok = not broker_blockers
    market_window_ok = not market_blockers
    scheduler_candidate_ok = not _candidate_blockers(candidate)
    lifecycle_candidate_ok = scheduler_candidate_ok and candidate.get(
        "lifecycle_activation_allowed_next_phase"
    ) is True

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    gates_passed = not blocker_list and not error_list
    gates = {
        "allow_scheduler_lifecycle_activation": allow_scheduler_lifecycle_activation is True,
        "stage4i4_gate_valid": stage4i4_gate_valid,
        "stage4i4_gate_ready": stage4i4_gate_ready,
        "selected_strategy_present": selected_strategy["selected_strategy_id"] is not None,
        "acknowledgements_ok": acknowledgement_checks["exact_match"] is True,
        "scheduler_candidate_ok": scheduler_candidate_ok,
        "lifecycle_candidate_ok": lifecycle_candidate_ok,
        "scheduler_snapshot_clean": scheduler_snapshot_clean,
        "lifecycle_snapshot_clean": lifecycle_snapshot_clean,
        "state_clean": state_clean,
        "risk_controls_ok": risk_controls_ok,
        "paper_broker_config_ok": paper_broker_config_ok,
        "market_window_ok": market_window_ok,
        "passed": gates_passed,
        "reasons": blocker_list,
    }

    scheduler_payload = _scheduler_payload(selected_strategy_id, generated_at)
    lifecycle_payload = _lifecycle_payload(selected_strategy_id, generated_at)
    execution = _default_execution()
    applied_operations: list[dict[str, Any]] = []
    skipped_operations: list[dict[str, Any]] = []
    rollback_required = False
    rollback_status = "not required"

    if gates_passed:
        execution["attempted"] = True
        execution["scheduler_activation_attempted"] = True
        scheduler_result = _call_scheduler_writer(scheduler_activation_writer, scheduler_payload)
        if scheduler_result["ok"] is not True:
            execution.update(
                scheduler_activation_succeeded=False,
                completed=False,
                failed_step="scheduler_activation",
                failure_reason=scheduler_result["failure_reason"],
            )
            errors.append(scheduler_result["failure_reason"])
            skipped_operations.extend(
                [
                    _operation("lifecycle_activation", "skipped_after_scheduler_failure"),
                    _operation("audit", "skipped_after_scheduler_failure"),
                ]
            )
        else:
            execution["scheduler_activation_succeeded"] = True
            applied_operations.append(
                _operation("scheduler_activation", "succeeded", scheduler_result["result"])
            )
            execution["lifecycle_activation_attempted"] = True
            lifecycle_result = _call_lifecycle_writer(lifecycle_activation_writer, lifecycle_payload)
            if lifecycle_result["ok"] is not True:
                execution.update(
                    lifecycle_activation_succeeded=False,
                    completed=False,
                    failed_step="lifecycle_activation",
                    failure_reason=lifecycle_result["failure_reason"],
                )
                errors.append(lifecycle_result["failure_reason"])
                rollback_required = True
                rollback_status = ROLLBACK_STATUS
                skipped_operations.append(
                    _operation("audit", "skipped_after_lifecycle_failure")
                )
            else:
                execution["lifecycle_activation_succeeded"] = True
                applied_operations.append(
                    _operation("lifecycle_activation", "succeeded", lifecycle_result["result"])
                )
                if audit_writer is not None:
                    execution["audit_write_attempted"] = True
                    audit_result = _call_audit_writer(
                        audit_writer,
                        {
                            "source_stage": "4I-5",
                            "generated_at": generated_at,
                            "selected_strategy_id": selected_strategy_id,
                            "scheduler_activation_payload": scheduler_payload,
                            "lifecycle_activation_payload": lifecycle_payload,
                            "scheduler_activation_result": scheduler_result["result"],
                            "lifecycle_activation_result": lifecycle_result["result"],
                        },
                    )
                    if audit_result["ok"] is not True:
                        execution.update(
                            audit_write_succeeded=False,
                            completed=False,
                            failed_step="audit",
                            failure_reason=audit_result["failure_reason"],
                        )
                        errors.append(audit_result["failure_reason"])
                        rollback_required = True
                        rollback_status = ROLLBACK_STATUS
                    else:
                        execution["audit_write_succeeded"] = True
                        applied_operations.append(
                            _operation("audit", "succeeded", audit_result["result"])
                        )
                        execution["completed"] = True
                else:
                    execution["completed"] = True
                    execution["audit_write_succeeded"] = False
    else:
        execution["attempted"] = False
        skipped_operations.extend(
            [
                _operation("scheduler_activation", "skipped_due_to_failed_gates"),
                _operation("lifecycle_activation", "skipped_due_to_failed_gates"),
                _operation("audit", "skipped_due_to_failed_gates"),
            ]
        )
        rollback_status = "not required; no activation writes were attempted"

    errors = _dedupe(errors)
    readiness = _ready_for_stage4i6(
        gates_passed=gates_passed,
        execution=execution,
        audit_writer_supplied=audit_writer is not None,
        rollback_required=rollback_required,
        selected_strategy=selected_strategy,
        safety=safety,
        errors=errors,
    )
    return _base_report(
        generated_at=generated_at,
        gates=gates,
        acknowledgement_checks=acknowledgement_checks,
        selected_strategy=selected_strategy,
        scheduler_activation_payload=scheduler_payload,
        lifecycle_activation_payload=lifecycle_payload,
        execution=execution,
        applied_operations=applied_operations,
        skipped_operations=skipped_operations,
        rollback_required=rollback_required,
        rollback_status=rollback_status,
        blockers=readiness["blockers"],
        warnings=warning_list,
        errors=errors,
    )


def _candidate_blockers(candidate: dict[str, Any]) -> list[str]:
    required = {
        "available": True,
        "scheduler_activation_allowed_next_phase": True,
        "lifecycle_activation_allowed_next_phase": True,
        "broker_submission_allowed_next_phase": False,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
    }
    blockers = [
        f"scheduler/lifecycle activation candidate {key} must be strict {expected}"
        for key, expected in required.items()
        if candidate.get(key) is not expected
    ]
    if candidate.get("enabled_strategy_count") != 1:
        blockers.append("scheduler/lifecycle activation candidate enabled_strategy_count must equal 1")
    return blockers


def _proposed_scheduler_blockers(proposed: dict[str, Any]) -> list[str]:
    required = {
        "available": True,
        "proposed_enabled_in_4I5": True,
        "would_register_in_4I4": False,
        "scheduler_job_enabled_now": False,
    }
    return [
        f"proposed scheduler activation {key} must be strict {expected}"
        for key, expected in required.items()
        if proposed.get(key) is not expected
    ]


def _proposed_lifecycle_blockers(proposed: dict[str, Any]) -> list[str]:
    required = {
        "available": True,
        "proposed_enabled_in_4I5": True,
        "would_execute_in_4I4": False,
        "lifecycle_execution_enabled_now": False,
    }
    return [
        f"proposed lifecycle activation {key} must be strict {expected}"
        for key, expected in required.items()
        if proposed.get(key) is not expected
    ]


def _scheduler_payload(selected_strategy_id: str | None, generated_at: str) -> dict[str, Any]:
    return {
        "selected_strategy_id": selected_strategy_id,
        "paper_only": True,
        "one_strategy_only": True,
        "activation_scope": "single_strategy_scheduled_paper_run",
        "source_stage": "4I-5",
        "scheduler_activation_enabled": True,
        "scheduler_job_enabled": True,
        "scheduler_job_scope": "single_strategy",
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "broker_submission_enabled": False,
        "strategy_scan_execution_enabled": False,
        "lifecycle_transition_execution_enabled": False,
        "market_data_enabled": False,
        "contract_qualification_enabled": False,
        "generated_at": generated_at,
        "required_runtime_guards": _required_runtime_guards(),
        "required_pre_run_checks": _required_pre_run_checks(),
    }


def _lifecycle_payload(selected_strategy_id: str | None, generated_at: str) -> dict[str, Any]:
    return {
        "selected_strategy_id": selected_strategy_id,
        "paper_only": True,
        "one_strategy_only": True,
        "activation_scope": "single_strategy_scheduled_paper_run",
        "source_stage": "4I-5",
        "lifecycle_activation_enabled": True,
        "lifecycle_scope": "single_strategy",
        "lifecycle_transition_execution_enabled": False,
        "strategy_scan_execution_enabled": False,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "broker_submission_enabled": False,
        "market_data_enabled": False,
        "contract_qualification_enabled": False,
        "generated_at": generated_at,
        "required_runtime_guards": _required_runtime_guards(),
        "required_pre_run_checks": _required_pre_run_checks(),
    }


def _required_runtime_guards() -> list[str]:
    return [
        "paper_only_mode",
        "one_strategy_only",
        "live_trading_disabled",
        "all_strategy_automation_disabled",
        "broker_submission_disabled",
        "strategy_scan_execution_disabled",
        "lifecycle_transition_execution_disabled",
        "market_data_disabled",
        "contract_qualification_disabled",
    ]


def _required_pre_run_checks() -> list[str]:
    return [
        "state_snapshot_clean",
        "risk_controls_available",
        "scheduler_snapshot_clean",
        "lifecycle_snapshot_clean",
        "paper_broker_snapshot_paper_only",
        "market_window_verified",
    ]


def _call_scheduler_writer(writer: Any, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = _dict_or_empty(writer.activate_scheduler(payload))
        return _verify_writer_result("scheduler_activation", result, payload["selected_strategy_id"])
    except Exception as exc:  # noqa: BLE001 - writer boundary reports type + text.
        return {"ok": False, "result": {}, "failure_reason": _exception_message(exc)}


def _call_lifecycle_writer(writer: Any, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = _dict_or_empty(writer.activate_lifecycle(payload))
        return _verify_writer_result("lifecycle_activation", result, payload["selected_strategy_id"])
    except Exception as exc:  # noqa: BLE001 - writer boundary reports type + text.
        return {"ok": False, "result": {}, "failure_reason": _exception_message(exc)}


def _call_audit_writer(writer: Any, event: dict[str, Any]) -> dict[str, Any]:
    try:
        result = _dict_or_empty(writer.append_scheduler_lifecycle_activation_audit(event))
        return {"ok": True, "result": result, "failure_reason": None}
    except Exception as exc:  # noqa: BLE001 - writer boundary reports type + text.
        return {"ok": False, "result": {}, "failure_reason": _exception_message(exc)}


def _verify_writer_result(target: str, result: dict[str, Any], selected_strategy_id: str | None) -> dict[str, Any]:
    status = result.get("status")
    if status == "conflict":
        return {
            "ok": False,
            "result": result,
            "failure_reason": f"{target} writer returned conflict",
        }
    if status == "already_exists":
        mismatches = _already_exists_mismatches(target, _mapping(result.get("record")), selected_strategy_id)
        if mismatches:
            return {
                "ok": False,
                "result": result,
                "failure_reason": f"{target} existing record is unsafe: {', '.join(mismatches)}",
            }
    return {"ok": True, "result": result, "failure_reason": None}


def _already_exists_mismatches(
    target: str, record: dict[str, Any], selected_strategy_id: str | None
) -> list[str]:
    checks = {
        "selected_strategy_id": record.get("selected_strategy_id") == selected_strategy_id,
        "paper_only": record.get("paper_only") is True,
        "one_strategy_only": record.get("one_strategy_only") is True,
        "live_trading_enabled": record.get("live_trading_enabled") is False,
        "all_strategies_enabled": record.get("all_strategies_enabled") is False,
        "broker_submission_enabled": record.get("broker_submission_enabled") is False,
    }
    if target == "scheduler_activation" and "scheduler_job_scope" in record:
        checks["scheduler_job_scope"] = record.get("scheduler_job_scope") == "single_strategy"
    if target == "lifecycle_activation" and "lifecycle_scope" in record:
        checks["lifecycle_scope"] = record.get("lifecycle_scope") == "single_strategy"
    for optional_false in ("strategy_scan_execution_enabled", "lifecycle_transition_execution_enabled"):
        if optional_false in record:
            checks[optional_false] = record.get(optional_false) is False
    return [key for key, ok in checks.items() if not ok]


def _activation_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    checks = {"activation_snapshot_present": present, "activation_snapshot_matches": True}
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("activation snapshot missing; operator must verify activation artifact")
        return checks, blockers, warnings
    active_ids = _as_string_list(data.get("active_strategy_ids"))
    if len(active_ids) > 1:
        blockers.append("activation snapshot shows more than one activated strategy")
    if active_ids and active_ids != [selected_strategy_id]:
        blockers.append("activation snapshot active_strategy_ids do not match selected strategy")
    for record in _activation_records(data):
        if not isinstance(record, dict):
            warnings.append("malformed activation snapshot entry ignored")
            continue
        if record.get("selected_strategy_id") not in (None, selected_strategy_id):
            blockers.append("activation snapshot selected_strategy_id does not match")
        if record.get("paper_only") is False:
            blockers.append("activation snapshot paper_only contradicts selected strategy")
        if record.get("enabled_strategy_count") not in (None, 1):
            blockers.append("activation snapshot enabled_strategy_count must be 1")
        for key in ("live_trading_enabled", "all_strategies_enabled", "broker_submission_enabled"):
            if record.get(key) is True:
                blockers.append(f"activation snapshot {key} contradicts Stage 4I-5 safety")
            elif key in record and record.get(key) is not False:
                blockers.append(f"activation snapshot {key} must be a strict false boolean")
    checks["activation_snapshot_matches"] = not blockers
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
        warnings.append("state snapshot missing; operator must verify runtime state")
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
        warnings.append("risk snapshot missing; operator must verify risk controls")
        return checks, blockers, warnings
    if checks["risk_bypass_enabled"]:
        blockers.append("risk bypass is enabled")
    for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
        if checks[key] is not True:
            blockers.append(f"{key} must be true in supplied risk snapshot")
    if "max_position_limit_available" in data and checks["max_position_limit_available"] is not True:
        blockers.append("max_position_limit_available must be true when supplied in risk snapshot")
    return checks, blockers, warnings


def _scheduler_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    already_enabled = _contains_truthy_or_malformed_flag(
        data, "scheduler_automation_enabled"
    ) or _contains_truthy_or_malformed_flag(data, "scheduler_wiring_enabled")
    all_strategy_enabled = _contains_truthy_or_malformed_flag(
        data, "all_strategy_scheduler_enabled"
    ) or _contains_truthy_or_malformed_flag(data, "all_strategies_enabled")
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
        warnings.append("scheduler snapshot missing; operator must verify scheduler state")
    if already_enabled:
        blockers.append("scheduler automation is already broadly enabled")
    if all_strategy_enabled:
        blockers.append("all-strategy scheduler automation is already enabled")
    if selected_enabled:
        blockers.append("selected strategy scheduled job is already active")
    return checks, blockers, warnings


def _lifecycle_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    already_enabled = _contains_truthy_or_malformed_flag(
        data, "lifecycle_automation_enabled"
    ) or _contains_truthy_or_malformed_flag(data, "lifecycle_wiring_enabled")
    if data.get("disabled") is True and data.get("dry_run_only") is True:
        already_enabled = False
    transition_enabled = _contains_truthy_or_malformed_flag(
        data, "lifecycle_transition_execution_enabled"
    )
    checks = {
        "lifecycle_snapshot_present": present,
        "lifecycle_already_enabled": already_enabled,
        "lifecycle_transition_execution_enabled": transition_enabled,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; operator must verify lifecycle state")
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
        warnings.append("paper broker snapshot missing; operator must verify paper broker config")
        return checks, blockers, warnings
    if mode not in (None, "PAPER"):
        blockers.append("paper broker mode must be PAPER")
    if data.get("paper_trading") is False:
        blockers.append("paper_trading must not be false")
    if port not in (None, *PAPER_IBKR_PORTS):
        blockers.append("ibkr_port must be a paper trading port")
    for key in ("live_trading_enabled", "broker_submission_enabled"):
        if data.get(key) is True:
            blockers.append(f"paper broker snapshot enables {key.replace('_', ' ')}")
        elif key in data and data.get(key) is not False:
            blockers.append(f"paper broker snapshot {key} must be a strict false boolean")
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
        warnings.append("market is currently closed; executor may continue but operator must verify run timing")
    if data.get("is_trading_day") is False:
        warnings.append("snapshot is not a trading day; executor may continue but operator must verify run timing")
    return checks, blockers, warnings


def _base_report(
    *,
    generated_at: str,
    gates: dict[str, Any],
    acknowledgement_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    scheduler_activation_payload: dict[str, Any],
    lifecycle_activation_payload: dict[str, Any],
    execution: dict[str, Any],
    applied_operations: list[dict[str, Any]],
    skipped_operations: list[dict[str, Any]],
    rollback_required: bool,
    rollback_status: str,
    blockers: list[str],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    ready = _ready_for_stage4i6(
        gates_passed=gates.get("passed") is True,
        execution=execution,
        audit_writer_supplied=execution.get("audit_write_attempted") is True,
        rollback_required=rollback_required,
        selected_strategy=selected_strategy,
        safety=_safety(),
        errors=errors,
    )
    return {
        "dry_run": False,
        "stage4i5_scheduler_lifecycle_activation_executor_report": True,
        "generated_at": generated_at,
        "gates": gates,
        "acknowledgement_checks": acknowledgement_checks,
        "selected_strategy": selected_strategy,
        "scheduler_activation_payload": scheduler_activation_payload,
        "lifecycle_activation_payload": lifecycle_activation_payload,
        "execution": execution,
        "applied_operations": applied_operations,
        "skipped_operations": skipped_operations,
        "rollback": {
            "rollback_required": rollback_required,
            "rollback_attempted": False,
            "rollback_status": rollback_status,
            "rollback_limitations": ROLLBACK_LIMITATIONS,
        },
        "safety": _safety(),
        "readiness_for_stage4i6": {
            "ready_to_build_scheduler_lifecycle_activation_acceptance": ready["ready"],
            "blockers": list(blockers if not ready["ready"] else []),
            "warnings": list(warnings),
        },
        "recommendations": {
            "ordered_next_steps": list(ORDERED_NEXT_STEPS),
            "do_not_do_yet": list(DO_NOT_DO_YET),
        },
        "success": ready["ready"],
        "errors": list(errors),
        "warnings": list(warnings),
    }


def _ready_for_stage4i6(
    *,
    gates_passed: bool,
    execution: dict[str, Any],
    audit_writer_supplied: bool,
    rollback_required: bool,
    selected_strategy: dict[str, Any],
    safety: dict[str, bool],
    errors: list[str],
) -> dict[str, Any]:
    blockers: list[str] = []
    if not gates_passed:
        blockers.append("activation gates did not pass")
    if execution.get("attempted") is not True:
        blockers.append("activation execution was not attempted")
    if execution.get("scheduler_activation_succeeded") is not True:
        blockers.append("scheduler activation did not succeed")
    if execution.get("lifecycle_activation_succeeded") is not True:
        blockers.append("lifecycle activation did not succeed")
    if audit_writer_supplied and execution.get("audit_write_succeeded") is not True:
        blockers.append("audit write did not succeed")
    if rollback_required:
        blockers.append("manual rollback review is required")
    if selected_strategy.get("selected_strategy_id") is None:
        blockers.append("exactly one selected strategy was not activated")
    for key in (
        "paper_only",
        "one_strategy_only",
    ):
        if selected_strategy.get(key) is not True:
            blockers.append(f"selected strategy {key} is not true")
    for key, value in safety.items():
        if value is not False and key.endswith("_enabled"):
            blockers.append(f"unsafe safety flag {key} is enabled")
    if errors:
        blockers.append("activation report contains errors")
    return {"ready": not blockers, "blockers": _dedupe(blockers)}


def _default_gates(reasons: list[str]) -> dict[str, Any]:
    return {
        "allow_scheduler_lifecycle_activation": False,
        "stage4i4_gate_valid": False,
        "stage4i4_gate_ready": False,
        "selected_strategy_present": False,
        "acknowledgements_ok": False,
        "scheduler_candidate_ok": False,
        "lifecycle_candidate_ok": False,
        "scheduler_snapshot_clean": False,
        "lifecycle_snapshot_clean": False,
        "state_clean": False,
        "risk_controls_ok": False,
        "paper_broker_config_ok": False,
        "market_window_ok": False,
        "passed": False,
        "reasons": list(reasons),
    }


def _default_execution(
    *, failed_step: str | None = None, failure_reason: str | None = None
) -> dict[str, Any]:
    return {
        "attempted": False,
        "scheduler_activation_attempted": False,
        "scheduler_activation_succeeded": False,
        "lifecycle_activation_attempted": False,
        "lifecycle_activation_succeeded": False,
        "audit_write_attempted": False,
        "audit_write_succeeded": False,
        "completed": False,
        "failed_step": failed_step,
        "failure_reason": failure_reason,
    }


def _selected_strategy_payload(
    selected_strategy_id: str | None, source: dict[str, Any]
) -> dict[str, Any]:
    return {
        "selected_strategy_id": selected_strategy_id,
        "paper_only": source.get("paper_only") is True,
        "one_strategy_only": source.get("one_strategy_only") is True,
    }


def _acknowledgement_checks(values: list[str] | None) -> dict[str, Any]:
    provided = [
        value.strip()
        for value in (values or [])
        if isinstance(value, str) and value.strip()
    ]
    missing = [item for item in REQUIRED_OPERATOR_ACKNOWLEDGEMENTS if item not in provided]
    return {
        "required": list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS),
        "provided": provided,
        "missing": missing,
        "exact_match": not missing,
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


def _safety() -> dict[str, bool]:
    return {
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "broker_submission_enabled": False,
        "strategy_scan_execution_enabled": False,
        "lifecycle_transition_execution_enabled": False,
        "market_data_enabled": False,
        "contract_qualification_enabled": False,
        "direct_scheduler_registration": False,
        "direct_lifecycle_execution": False,
    }


def _safety_checks(*sources: Any) -> dict[str, bool]:
    return {
        "no_live_trading": not _contains_truthy_or_malformed_flag(sources, "live_trading_enabled"),
        "no_all_strategy_enablement": not (
            _contains_truthy_or_malformed_flag(sources, "all_strategies_enabled")
            or _contains_truthy_or_malformed_flag(sources, "enable_all_strategies")
        ),
        "no_broker_submission_enabled": not _contains_truthy_or_malformed_flag(
            sources, "broker_submission_enabled"
        ),
        "no_market_data": not _contains_truthy_or_malformed_flag(sources, "market_data_enabled"),
        "no_contract_qualification": not _contains_truthy_or_malformed_flag(
            sources, "contract_qualification_enabled"
        ),
        "no_order_submission": not (
            _contains_truthy_or_malformed_flag(sources, "order_submission_enabled")
            or _contains_truthy_or_malformed_flag(sources, "live_orders_enabled")
        ),
        "no_strategy_scan_execution": not _contains_truthy_or_malformed_flag(
            sources, "strategy_scan_execution_enabled"
        ),
        "no_scheduler_registration": not _contains_truthy_or_malformed_flag(
            sources, "scheduler_registration_enabled"
        ),
        "no_lifecycle_execution": not (
            _contains_truthy_or_malformed_flag(sources, "lifecycle_execution_enabled")
            or _contains_truthy_or_malformed_flag(sources, "lifecycle_transition_execution_enabled")
        ),
    }


def _safety_blockers(checks: dict[str, bool]) -> list[str]:
    labels = {
        "no_live_trading": "live trading safety flag is enabled or malformed",
        "no_all_strategy_enablement": "all-strategy safety flag is enabled or malformed",
        "no_broker_submission_enabled": "broker submission safety flag is enabled or malformed",
        "no_market_data": "market data safety flag is enabled or malformed",
        "no_contract_qualification": "contract qualification safety flag is enabled or malformed",
        "no_order_submission": "order submission safety flag is enabled or malformed",
        "no_strategy_scan_execution": "strategy scan execution safety flag is enabled or malformed",
        "no_scheduler_registration": "scheduler registration safety flag is enabled or malformed",
        "no_lifecycle_execution": "lifecycle execution safety flag is enabled or malformed",
    }
    return [label for key, label in labels.items() if checks.get(key) is not True]


def _operation(target: str, status: str, result: dict[str, Any] | None = None) -> dict[str, Any]:
    item = {"target": target, "status": status}
    if result is not None:
        item["result"] = result
    return item


def _require(blockers: list[str], condition: bool, message: str) -> None:
    if not condition:
        blockers.append(message)


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return _json_safe(value) if isinstance(value, dict) else {}


def _exception_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _generated_at(now_provider: Callable[[], Any] | None) -> str:
    if now_provider is None:
        return DEFAULT_GENERATED_AT
    value = now_provider()
    if isinstance(value, datetime):
        current = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return current.isoformat()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc).isoformat()
    if isinstance(value, str) and value:
        return value
    return DEFAULT_GENERATED_AT


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_mapping(value: Any) -> tuple[bool, dict[str, Any]]:
    return isinstance(value, dict), _mapping(value)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_string_list(value: Any) -> list[str]:
    return [item.strip() for item in _as_list(value) if isinstance(item, str) and item.strip()]


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _first_present(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _contains_truthy_or_malformed_flag(source: Any, key: str) -> bool:
    if isinstance(source, dict):
        if key in source:
            value = source.get(key)
            if value is True:
                return True
            if value is False or value is None:
                return False
            return True
        return any(_contains_truthy_or_malformed_flag(value, key) for value in source.values())
    if isinstance(source, (list, tuple)):
        return any(_contains_truthy_or_malformed_flag(item, key) for item in source)
    return False


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
