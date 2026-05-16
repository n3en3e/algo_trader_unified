"""Pure Stage 4I-6 scheduler/lifecycle activation acceptance report."""

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
ORDERED_NEXT_STEPS = [
    "Proceed to the post-4I controlled scheduled PAPER operation planning/acceptance phase.",
    "Before any real scheduled run execution, re-check activation artifacts, scheduler/lifecycle state, risk controls, paper broker config, market window, and state reconciliation.",
    "Keep live trading disabled.",
    "Keep all other strategies disabled.",
    "Keep broker submission separately gated until the explicit broker-submission phase.",
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


def build_stage4i6_scheduler_lifecycle_activation_acceptance_report(
    *,
    stage4i5_activation_executor_report: dict | None,
    scheduler_activation_snapshot: dict | None = None,
    lifecycle_activation_snapshot: dict | None = None,
    audit_snapshot: dict | None = None,
    activation_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    market_window_snapshot: dict | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a read-only acceptance report for the 4I scheduler/lifecycle activation."""

    try:
        return _json_safe(
            _build_report(
                stage4i5_activation_executor_report=stage4i5_activation_executor_report,
                scheduler_activation_snapshot=scheduler_activation_snapshot,
                lifecycle_activation_snapshot=lifecycle_activation_snapshot,
                audit_snapshot=audit_snapshot,
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
        message = f"unexpected activation acceptance failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_default_selected_strategy(),
                payload_checks=_default_payload_checks(),
                applied_operation_checks=_default_applied_operation_checks(),
                snapshot_checks=_default_snapshot_checks(),
                state_checks=_default_state_checks(),
                risk_checks=_default_risk_checks(),
                scheduler_checks=_default_scheduler_checks(),
                lifecycle_checks=_default_lifecycle_checks(),
                paper_broker_checks=_default_paper_broker_checks(),
                market_window_checks=_default_market_window_checks(),
                safety_checks=_safety_checks({}, {}, {}, {}, {}),
                ready=False,
                blockers=[message],
                warnings=[],
                errors=[message],
            )
        )


def _build_report(
    *,
    stage4i5_activation_executor_report: dict | None,
    scheduler_activation_snapshot: dict | None,
    lifecycle_activation_snapshot: dict | None,
    audit_snapshot: dict | None,
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
    report = stage4i5_activation_executor_report if isinstance(stage4i5_activation_executor_report, dict) else None
    data = _mapping(report)
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4i5_activation_executor_report is None:
        blockers.append("Stage 4I-5 activation executor report is missing")
    elif report is None:
        blockers.append("Stage 4I-5 activation executor report must be a dict")
        errors.append("Stage 4I-5 activation executor report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    artifact_checks = _artifact_checks(stage4i5_activation_executor_report)
    blockers.extend(_artifact_blockers(artifact_checks, data))

    selected_strategy_id = _selected_strategy_id(data)
    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    blockers.extend(selected_blockers)

    payload_checks, payload_blockers = _payload_checks(data, selected_strategy_id)
    blockers.extend(payload_blockers)

    applied_operation_checks, operation_blockers = _applied_operation_checks(data)
    blockers.extend(operation_blockers)

    snapshot_checks: dict[str, Any] = {}
    sched_act_checks, sched_act_blockers, sched_act_warnings = _scheduler_activation_snapshot_checks(
        scheduler_activation_snapshot, selected_strategy_id
    )
    life_act_checks, life_act_blockers, life_act_warnings = _lifecycle_activation_snapshot_checks(
        lifecycle_activation_snapshot, selected_strategy_id
    )
    audit_checks, audit_blockers, audit_warnings = _audit_snapshot_checks(
        audit_snapshot, selected_strategy_id, _audit_attempted(data)
    )
    activation_checks, activation_blockers, activation_warnings = _activation_snapshot_checks(
        activation_snapshot, selected_strategy_id
    )
    snapshot_checks.update(sched_act_checks)
    snapshot_checks.update(life_act_checks)
    snapshot_checks.update(audit_checks)
    snapshot_checks.update(activation_checks)
    blockers.extend(sched_act_blockers + life_act_blockers + audit_blockers + activation_blockers)
    warnings.extend(sched_act_warnings + life_act_warnings + audit_warnings + activation_warnings)

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
        _mapping(data.get("scheduler_activation_payload")),
        _mapping(data.get("lifecycle_activation_payload")),
        scheduler_snapshot if isinstance(scheduler_snapshot, dict) else {},
        lifecycle_snapshot if isinstance(lifecycle_snapshot, dict) else {},
        paper_broker_snapshot if isinstance(paper_broker_snapshot, dict) else {},
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
        payload_checks=payload_checks,
        applied_operation_checks=applied_operation_checks,
        snapshot_checks=snapshot_checks,
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


def _artifact_checks(report: dict | None) -> dict[str, Any]:
    data = _mapping(report)
    execution = _mapping(data.get("execution"))
    rollback = _mapping(data.get("rollback"))
    readiness = _mapping(data.get("readiness_for_stage4i6"))
    audit_attempted = _audit_attempted(data)
    audit_write_succeeded = True if not audit_attempted else execution.get("audit_write_succeeded") is True
    selected_strategy_id = _selected_strategy_id(data)
    return {
        "stage4i5_report_present": isinstance(report, dict),
        "stage4i5_report_ready": (
            data.get("stage4i5_scheduler_lifecycle_activation_executor_report") is True
            and readiness.get("ready_to_build_scheduler_lifecycle_activation_acceptance") is True
            and data.get("success") is True
        ),
        "execution_attempted": execution.get("attempted") is True,
        "scheduler_activation_succeeded": execution.get("scheduler_activation_succeeded") is True,
        "lifecycle_activation_succeeded": execution.get("lifecycle_activation_succeeded") is True,
        "audit_write_succeeded": audit_write_succeeded,
        "execution_completed": execution.get("completed") is True,
        "rollback_not_required": _rollback_required(data, rollback) is False,
        "selected_strategy_present": isinstance(selected_strategy_id, str) and bool(selected_strategy_id),
    }


def _artifact_blockers(checks: dict[str, Any], data: dict[str, Any]) -> list[str]:
    labels = {
        "stage4i5_report_present": "Stage 4I-5 activation executor report is missing",
        "stage4i5_report_ready": "Stage 4I-5 activation executor report is not ready for Stage 4I-6",
        "execution_attempted": "Stage 4I-5 execution was not attempted",
        "scheduler_activation_succeeded": "scheduler activation did not succeed",
        "lifecycle_activation_succeeded": "lifecycle activation did not succeed",
        "audit_write_succeeded": "audit write did not succeed after it was attempted",
        "execution_completed": "Stage 4I-5 execution did not complete",
        "rollback_not_required": "Stage 4I-5 rollback is required",
        "selected_strategy_present": "selected strategy is missing from Stage 4I-5 report",
    }
    blockers = [label for key, label in labels.items() if checks.get(key) is not True]
    if _as_list(data.get("errors")):
        blockers.append("Stage 4I-5 report contains errors")
    return blockers


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
        blockers.append("selected strategy must be paper_only true")
    if not one_strategy_only:
        blockers.append("selected strategy must be one_strategy_only true")
    return (
        {
            "selected_strategy_id": selected_strategy_id,
            "paper_only": paper_only,
            "one_strategy_only": one_strategy_only,
        },
        blockers,
    )


def _payload_checks(
    report: dict[str, Any], selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str]]:
    scheduler_payload = _mapping(report.get("scheduler_activation_payload"))
    lifecycle_payload = _mapping(report.get("lifecycle_activation_payload"))
    scheduler_present = isinstance(report.get("scheduler_activation_payload"), dict)
    lifecycle_present = isinstance(report.get("lifecycle_activation_payload"), dict)
    scheduler_matches = scheduler_payload.get("selected_strategy_id") == selected_strategy_id
    lifecycle_matches = lifecycle_payload.get("selected_strategy_id") == selected_strategy_id
    scheduler_safe, scheduler_reasons = _payload_safe(scheduler_payload, selected_strategy_id)
    lifecycle_safe, lifecycle_reasons = _payload_safe(lifecycle_payload, selected_strategy_id)
    combined = _combined_payload_disabled(scheduler_payload, lifecycle_payload)
    checks = {
        "scheduler_payload_present": scheduler_present,
        "lifecycle_payload_present": lifecycle_present,
        "scheduler_payload_matches_selected_strategy": scheduler_matches,
        "lifecycle_payload_matches_selected_strategy": lifecycle_matches,
        "scheduler_payload_safe": scheduler_safe,
        "lifecycle_payload_safe": lifecycle_safe,
        "live_trading_disabled": combined["live_trading_enabled"] is False,
        "all_strategies_disabled": combined["all_strategies_enabled"] is False,
        "broker_submission_disabled": combined["broker_submission_enabled"] is False,
        "strategy_scan_execution_disabled": combined["strategy_scan_execution_enabled"] is False,
        "lifecycle_transition_execution_disabled": combined["lifecycle_transition_execution_enabled"] is False,
        "market_data_disabled": combined["market_data_enabled"] is False,
        "contract_qualification_disabled": combined["contract_qualification_enabled"] is False,
    }
    blockers: list[str] = []
    if not scheduler_present:
        blockers.append("scheduler activation payload is missing")
    if not lifecycle_present:
        blockers.append("lifecycle activation payload is missing")
    if scheduler_present and not scheduler_matches:
        blockers.append("scheduler activation payload selected_strategy_id does not match")
    if lifecycle_present and not lifecycle_matches:
        blockers.append("lifecycle activation payload selected_strategy_id does not match")
    blockers.extend(f"scheduler activation payload {reason}" for reason in scheduler_reasons)
    blockers.extend(f"lifecycle activation payload {reason}" for reason in lifecycle_reasons)
    return checks, blockers


def _payload_safe(payload: dict[str, Any], selected_strategy_id: str | None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    required = {
        "selected_strategy_id": selected_strategy_id,
        "paper_only": True,
        "one_strategy_only": True,
        "activation_scope": "single_strategy_scheduled_paper_run",
        "source_stage": "4I-5",
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "broker_submission_enabled": False,
        "strategy_scan_execution_enabled": False,
        "lifecycle_transition_execution_enabled": False,
        "market_data_enabled": False,
        "contract_qualification_enabled": False,
    }
    for key, expected in required.items():
        if payload.get(key) is not expected if isinstance(expected, bool) else payload.get(key) != expected:
            reasons.append(f"{key} must be strict {expected!r}")
    return not reasons, reasons


def _combined_payload_disabled(
    scheduler_payload: dict[str, Any], lifecycle_payload: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "live_trading_enabled",
        "all_strategies_enabled",
        "broker_submission_enabled",
        "strategy_scan_execution_enabled",
        "lifecycle_transition_execution_enabled",
        "market_data_enabled",
        "contract_qualification_enabled",
    ):
        result[key] = scheduler_payload.get(key) if scheduler_payload.get(key) is lifecycle_payload.get(key) else None
    return result


def _applied_operation_checks(report: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    applied = _as_list(report.get("applied_operations"))
    skipped = _as_list(report.get("skipped_operations"))
    targets = {item.get("target") for item in applied if isinstance(item, dict)}
    audit_attempted = _audit_attempted(report)
    checks = {
        "scheduler_activation_applied": "scheduler_activation" in targets,
        "lifecycle_activation_applied": "lifecycle_activation" in targets,
        "audit_applied": (not audit_attempted) or "audit" in targets,
        "skipped_operations_empty": len(skipped) == 0,
        "applied_operation_targets_present": all(
            isinstance(item, dict) and isinstance(item.get("target"), str) and bool(item.get("target"))
            for item in applied
        )
        and bool(applied),
    }
    blockers: list[str] = []
    if not checks["scheduler_activation_applied"]:
        blockers.append("applied_operations missing scheduler_activation target")
    if not checks["lifecycle_activation_applied"]:
        blockers.append("applied_operations missing lifecycle_activation target")
    if not checks["audit_applied"]:
        blockers.append("applied_operations missing audit target after audit was attempted")
    if not checks["skipped_operations_empty"]:
        blockers.append("skipped_operations must be empty for clean acceptance")
    if not checks["applied_operation_targets_present"]:
        blockers.append("applied_operations contains malformed or missing target labels")
    return checks, blockers


def _scheduler_activation_snapshot_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    return _activation_artifact_snapshot_checks(
        snapshot,
        selected_strategy_id,
        prefix="scheduler_activation",
        record_key="scheduler_activation_record",
        list_key="scheduler_activations",
        scope_key="scheduler_job_scope",
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
        scope_key="lifecycle_scope",
    )


def _activation_artifact_snapshot_checks(
    snapshot: dict | None,
    selected_strategy_id: str | None,
    *,
    prefix: str,
    record_key: str,
    list_key: str,
    scope_key: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append(f"{prefix} snapshot missing; operator must verify activation artifact")
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
        if record.get(scope_key) not in (None, "single_strategy"):
            blockers.append(f"{prefix} snapshot {scope_key} must be single_strategy")
        for key in (
            "live_trading_enabled",
            "all_strategies_enabled",
            "broker_submission_enabled",
            "strategy_scan_execution_enabled",
            "lifecycle_transition_execution_enabled",
        ):
            if record.get(key) is True:
                blockers.append(f"{prefix} snapshot {key} contradicts Stage 4I safety")
    return (
        {f"{prefix}_snapshot_present": True, f"{prefix}_snapshot_matches": not blockers},
        blockers,
        warnings,
    )


def _audit_snapshot_checks(
    snapshot: dict | None, selected_strategy_id: str | None, audit_attempted: bool
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        if audit_attempted:
            warnings.append("audit snapshot missing after audit was attempted; operator must verify audit artifact")
        return {"audit_snapshot_present": False, "audit_snapshot_matches": True}, blockers, warnings
    records = _audit_records(data)
    matched = False
    explicit_mismatch = False
    if data.get("selected_strategy_id") not in (None, selected_strategy_id):
        explicit_mismatch = True
    if data.get("source_stage") not in (None, "4I-5"):
        blockers.append("audit snapshot source_stage does not match 4I-5")
    for event in records:
        if not isinstance(event, dict):
            warnings.append("malformed audit snapshot event ignored")
            continue
        if _audit_event_matches(event, selected_strategy_id):
            matched = True
        if _audit_event_mismatches(event, selected_strategy_id):
            explicit_mismatch = True
    if explicit_mismatch:
        blockers.append("audit snapshot selected_strategy_id does not match")
    if records and not matched and audit_attempted:
        blockers.append("audit snapshot does not include selected strategy activation evidence")
    return {"audit_snapshot_present": True, "audit_snapshot_matches": not blockers}, blockers, warnings


def _activation_snapshot_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("activation snapshot missing; operator must verify one-strategy activation state")
        return {"activation_snapshot_present": False, "activation_snapshot_matches": True}, blockers, warnings
    active_ids = _as_string_list(data.get("active_strategy_ids"))
    if active_ids and active_ids != [selected_strategy_id]:
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
                blockers.append(f"activation snapshot {key} contradicts Stage 4I safety")
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
        warnings.append("state snapshot missing; next phase must verify halt, reconciliation, and intents")
    if active_halt:
        blockers.append("state snapshot shows active halt")
    if unresolved_count > 0:
        blockers.append("state snapshot shows unresolved NEEDS_RECONCILIATION")
    if active_intents_count > 0 and not active_intents_safe:
        blockers.append("state snapshot shows unsafe active intents")
    if active_intents_count > 0 and active_intents_safe:
        warnings.append("active intents present but marked safe for enablement")
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
        warnings.append("risk snapshot missing; next phase must verify kill switch, hard halt, and daily loss controls")
        return checks, blockers, warnings
    for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
        if checks[key] is not True:
            blockers.append(f"risk snapshot {key} must be true")
    if data.get("max_position_limit_available") is False:
        blockers.append("risk snapshot max_position_limit_available must not be false when supplied")
    if checks["risk_bypass_enabled"] is True:
        blockers.append("risk snapshot risk_bypass_enabled must be false")
    return checks, blockers, warnings


def _scheduler_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    jobs = _as_list(data.get("jobs")) + _as_list(data.get("scheduled_jobs"))
    matching_jobs = [
        job for job in jobs if isinstance(job, dict) and _job_strategy_id(job) == selected_strategy_id
    ]
    selected_job_matches = True
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("scheduler snapshot missing; next phase must verify scheduler state")
    all_strategy_enabled = any(
        data.get(key) is True
        for key in ("all_strategy_scheduler_enabled", "all_strategies_enabled", "broad_scheduler_enabled")
    )
    scheduler_enabled = data.get("scheduler_automation_enabled") is True or bool(matching_jobs)
    for job in matching_jobs:
        selected_job_matches = selected_job_matches and _scheduler_job_safe(job, selected_strategy_id)
    if all_strategy_enabled:
        blockers.append("scheduler snapshot shows broad/all-strategy scheduler automation enabled")
    if matching_jobs and not selected_job_matches:
        blockers.append("selected strategy scheduler job does not match 4I-5 safety constraints")
    checks = {
        "scheduler_snapshot_present": present,
        "scheduler_already_enabled": scheduler_enabled,
        "all_strategy_scheduler_enabled": all_strategy_enabled,
        "selected_strategy_job_matches": selected_job_matches,
    }
    return checks, blockers, warnings


def _lifecycle_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; next phase must verify lifecycle state")
    lifecycle_enabled = data.get("lifecycle_automation_enabled") is True
    transition_enabled = data.get("lifecycle_transition_execution_enabled") is True
    matches = True
    if lifecycle_enabled:
        matches = (
            data.get("selected_strategy_id") == selected_strategy_id
            and data.get("broker_submission_enabled") is False
            and data.get("live_trading_enabled") is False
            and data.get("all_strategies_enabled") is False
            and transition_enabled is False
        )
        if not matches:
            blockers.append("lifecycle snapshot automation does not match selected strategy safety constraints")
    if transition_enabled:
        blockers.append("lifecycle snapshot lifecycle_transition_execution_enabled must be false")
    checks = {
        "lifecycle_snapshot_present": present,
        "lifecycle_already_enabled": lifecycle_enabled,
        "lifecycle_matches_selected_strategy": matches,
        "lifecycle_transition_execution_enabled": transition_enabled,
    }
    return checks, blockers, warnings


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
        warnings.append("paper broker snapshot missing; next phase must verify PAPER broker config")
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
        blockers.append("market window explicitly disallows scheduled PAPER run")
    if market_open is False:
        warnings.append("market window shows market_open false; verify timing before proceeding")
    if is_trading_day is False:
        warnings.append("market window shows is_trading_day false; verify holiday schedule before proceeding")
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
    scheduler_payload: dict[str, Any],
    lifecycle_payload: dict[str, Any],
    scheduler_snapshot: dict[str, Any],
    lifecycle_snapshot: dict[str, Any],
    paper_broker_snapshot: dict[str, Any],
) -> dict[str, bool]:
    return {
        "no_live_trading": _none_true("live_trading_enabled", scheduler_payload, lifecycle_payload, scheduler_snapshot, lifecycle_snapshot, paper_broker_snapshot),
        "no_all_strategy_enablement": _none_true("all_strategies_enabled", scheduler_payload, lifecycle_payload, scheduler_snapshot, lifecycle_snapshot),
        "no_broker_submission_enabled": _none_true("broker_submission_enabled", scheduler_payload, lifecycle_payload, lifecycle_snapshot, paper_broker_snapshot),
        "no_market_data": _none_true("market_data_enabled", scheduler_payload, lifecycle_payload),
        "no_contract_qualification": _none_true("contract_qualification_enabled", scheduler_payload, lifecycle_payload),
        "no_order_submission": True,
        "no_strategy_scan_execution": _none_true("strategy_scan_execution_enabled", scheduler_payload, lifecycle_payload, scheduler_snapshot),
        "no_lifecycle_transition_execution": _none_true("lifecycle_transition_execution_enabled", scheduler_payload, lifecycle_payload, lifecycle_snapshot),
        "no_direct_scheduler_registration": True,
        "no_direct_lifecycle_execution": True,
        "no_state_write": True,
        "no_ledger_write": True,
    }


def _safety_blockers(checks: dict[str, bool]) -> list[str]:
    labels = {
        "no_live_trading": "safety checks do not confirm live trading is disabled",
        "no_all_strategy_enablement": "safety checks do not confirm all-strategy enablement is disabled",
        "no_broker_submission_enabled": "safety checks do not confirm broker submission is disabled",
        "no_market_data": "safety checks do not confirm market data is disabled",
        "no_contract_qualification": "safety checks do not confirm contract qualification is disabled",
        "no_order_submission": "safety checks do not confirm order submission is disabled",
        "no_strategy_scan_execution": "safety checks do not confirm strategy scan execution is disabled",
        "no_lifecycle_transition_execution": "safety checks do not confirm lifecycle transition execution is disabled",
        "no_direct_scheduler_registration": "safety checks do not confirm direct scheduler registration is disabled",
        "no_direct_lifecycle_execution": "safety checks do not confirm direct lifecycle execution is disabled",
        "no_state_write": "safety checks do not confirm state writes are disabled",
        "no_ledger_write": "safety checks do not confirm ledger writes are disabled",
    }
    return [label for key, label in labels.items() if checks.get(key) is not True]


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    payload_checks: dict[str, Any],
    applied_operation_checks: dict[str, Any],
    snapshot_checks: dict[str, Any],
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
        "stage4i6_scheduler_lifecycle_activation_acceptance_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "payload_checks": payload_checks,
        "applied_operation_checks": applied_operation_checks,
        "snapshot_checks": snapshot_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_next_phase": {
            "ready_to_proceed_after_stage4i": ready,
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


def _selected_strategy_id(report: dict[str, Any]) -> str | None:
    selected = _mapping(report.get("selected_strategy"))
    value = selected.get("selected_strategy_id")
    if isinstance(value, str) and value:
        return value
    return None


def _rollback_required(report: dict[str, Any], rollback: dict[str, Any]) -> bool:
    if "rollback_required" in rollback:
        return rollback.get("rollback_required") is True
    return report.get("rollback_required") is True


def _audit_attempted(report: dict[str, Any]) -> bool:
    return _mapping(report.get("execution")).get("audit_write_attempted") is True


def _explicit_records(data: dict[str, Any], record_key: str, list_key: str) -> list[Any]:
    records: list[Any] = []
    if record_key in data:
        records.append(data.get(record_key))
    records.extend(_as_list(data.get(list_key)))
    explicit_keys = {
        "selected_strategy_id",
        "paper_only",
        "one_strategy_only",
        "scheduler_job_scope",
        "lifecycle_scope",
        "live_trading_enabled",
        "all_strategies_enabled",
        "broker_submission_enabled",
        "strategy_scan_execution_enabled",
        "lifecycle_transition_execution_enabled",
    }
    if any(key in data for key in explicit_keys):
        records.append(data)
    return records


def _audit_records(data: dict[str, Any]) -> list[Any]:
    records: list[Any] = []
    records.extend(_as_list(data.get("events")))
    records.extend(_as_list(data.get("audit_events")))
    if any(key in data for key in ("selected_strategy_id", "payload", "data", "source_stage")):
        records.append(data)
    return records


def _audit_event_matches(event: dict[str, Any], selected_strategy_id: str | None) -> bool:
    return any(
        _mapping(part).get("selected_strategy_id") == selected_strategy_id
        for part in (event, event.get("payload"), event.get("data"))
    )


def _audit_event_mismatches(event: dict[str, Any], selected_strategy_id: str | None) -> bool:
    for part in (event, event.get("payload"), event.get("data")):
        data = _mapping(part)
        value = data.get("selected_strategy_id")
        if value is not None and value != selected_strategy_id:
            return True
    return False


def _job_strategy_id(job: dict[str, Any]) -> Any:
    return _first_present(job.get("selected_strategy_id"), job.get("strategy_id"))


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
        and all(job.get(key) is not True for key in required_false)
    )


def _none_true(key: str, *mappings: dict[str, Any]) -> bool:
    return all(mapping.get(key) is not True for mapping in mappings)


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


def _default_artifact_checks() -> dict[str, Any]:
    return {
        "stage4i5_report_present": False,
        "stage4i5_report_ready": False,
        "execution_attempted": False,
        "scheduler_activation_succeeded": False,
        "lifecycle_activation_succeeded": False,
        "audit_write_succeeded": False,
        "execution_completed": False,
        "rollback_not_required": False,
        "selected_strategy_present": False,
    }


def _default_selected_strategy() -> dict[str, Any]:
    return {"selected_strategy_id": None, "paper_only": False, "one_strategy_only": False}


def _default_payload_checks() -> dict[str, Any]:
    return {
        "scheduler_payload_present": False,
        "lifecycle_payload_present": False,
        "scheduler_payload_matches_selected_strategy": False,
        "lifecycle_payload_matches_selected_strategy": False,
        "scheduler_payload_safe": False,
        "lifecycle_payload_safe": False,
        "live_trading_disabled": False,
        "all_strategies_disabled": False,
        "broker_submission_disabled": False,
        "strategy_scan_execution_disabled": False,
        "lifecycle_transition_execution_disabled": False,
        "market_data_disabled": False,
        "contract_qualification_disabled": False,
    }


def _default_applied_operation_checks() -> dict[str, Any]:
    return {
        "scheduler_activation_applied": False,
        "lifecycle_activation_applied": False,
        "audit_applied": False,
        "skipped_operations_empty": False,
        "applied_operation_targets_present": False,
    }


def _default_snapshot_checks() -> dict[str, Any]:
    return {
        "scheduler_activation_snapshot_present": False,
        "scheduler_activation_snapshot_matches": True,
        "lifecycle_activation_snapshot_present": False,
        "lifecycle_activation_snapshot_matches": True,
        "audit_snapshot_present": False,
        "audit_snapshot_matches": True,
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
