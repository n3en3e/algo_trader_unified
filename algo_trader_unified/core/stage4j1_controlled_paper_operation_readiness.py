"""Pure Stage 4J-1 controlled scheduled PAPER operation readiness report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
PAPER_IBKR_PORTS = {4004}
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding to 4J-2"
)
ORDERED_NEXT_STEPS = [
    "Build Stage 4J-2 controlled scheduled PAPER operation plan.",
    "Before any real scheduled strategy operation, re-check activation artifacts, scheduler/lifecycle state, risk controls, paper broker config, market window, and state reconciliation.",
    "Keep live trading disabled.",
    "Keep all other strategies disabled.",
    "Keep broker submission separately gated until the explicit broker-submission phase.",
]
DO_NOT_DO_YET = [
    "Do not enable live trading.",
    "Do not enable all strategies.",
    "Do not place orders now.",
    "Do not run strategy scans now.",
    "Do not change strategy thresholds.",
    "Do not change position sizing.",
    "Do not bypass risk controls.",
    "Do not enable broker submission broadly.",
]


def build_stage4j1_controlled_paper_operation_readiness_report(
    *,
    stage4i6_acceptance_report: dict | None,
    scheduler_activation_snapshot: dict | None = None,
    lifecycle_activation_snapshot: dict | None = None,
    activation_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    market_window_snapshot: dict | None = None,
    strategy_registry_snapshot: dict | list | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a read-only readiness report for Stage 4J-2 planning."""

    try:
        return _json_safe(
            _build_report(
                stage4i6_acceptance_report=stage4i6_acceptance_report,
                scheduler_activation_snapshot=scheduler_activation_snapshot,
                lifecycle_activation_snapshot=lifecycle_activation_snapshot,
                activation_snapshot=activation_snapshot,
                state_snapshot=state_snapshot,
                risk_snapshot=risk_snapshot,
                scheduler_snapshot=scheduler_snapshot,
                lifecycle_snapshot=lifecycle_snapshot,
                paper_broker_snapshot=paper_broker_snapshot,
                market_window_snapshot=market_window_snapshot,
                strategy_registry_snapshot=strategy_registry_snapshot,
                now_provider=now_provider,
            )
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must stay data-only.
        message = f"unexpected Stage 4J-1 readiness failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_default_selected_strategy(),
                stage4i_acceptance_checks=_default_stage4i_acceptance_checks(),
                registry_checks=_default_registry_checks(),
                activation_snapshot_checks=_default_activation_snapshot_checks(),
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
    stage4i6_acceptance_report: dict | None,
    scheduler_activation_snapshot: dict | None,
    lifecycle_activation_snapshot: dict | None,
    activation_snapshot: dict | None,
    state_snapshot: dict | None,
    risk_snapshot: dict | None,
    scheduler_snapshot: dict | None,
    lifecycle_snapshot: dict | None,
    paper_broker_snapshot: dict | None,
    market_window_snapshot: dict | None,
    strategy_registry_snapshot: dict | list | None,
    now_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    generated_at = _generated_at(now_provider)
    report = stage4i6_acceptance_report if isinstance(stage4i6_acceptance_report, dict) else None
    data = _mapping(report)
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4i6_acceptance_report is None:
        blockers.append("Stage 4I-6 scheduler/lifecycle activation acceptance report is missing")
    elif report is None:
        blockers.append("Stage 4I-6 scheduler/lifecycle activation acceptance report must be a dict")
        errors.append("Stage 4I-6 scheduler/lifecycle activation acceptance report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    selected_strategy_id = _selected_strategy_id(data)
    artifact_checks = _artifact_checks(stage4i6_acceptance_report, selected_strategy_id)
    blockers.extend(_artifact_blockers(artifact_checks, data))

    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    blockers.extend(selected_blockers)

    stage4i_acceptance_checks, acceptance_blockers = _stage4i_acceptance_checks(data)
    blockers.extend(acceptance_blockers)

    registry_checks, registry_blockers, registry_warnings = _registry_checks(
        strategy_registry_snapshot, selected_strategy_id
    )
    blockers.extend(registry_blockers)
    warnings.extend(registry_warnings)

    activation_snapshot_checks, activation_blockers, activation_warnings = _activation_snapshot_group_checks(
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
        scheduler_activation_snapshot if isinstance(scheduler_activation_snapshot, dict) else {},
        lifecycle_activation_snapshot if isinstance(lifecycle_activation_snapshot, dict) else {},
        activation_snapshot if isinstance(activation_snapshot, dict) else {},
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
        stage4i_acceptance_checks=stage4i_acceptance_checks,
        registry_checks=registry_checks,
        activation_snapshot_checks=activation_snapshot_checks,
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


def _artifact_checks(report: dict | None, selected_strategy_id: str | None) -> dict[str, Any]:
    data = _mapping(report)
    readiness = _mapping(data.get("readiness_for_next_phase"))
    stage4i_complete = readiness.get("ready_to_proceed_after_stage4i") is True
    return {
        "stage4i6_report_present": isinstance(report, dict),
        "stage4i6_report_ready": (
            data.get("stage4i6_scheduler_lifecycle_activation_acceptance_report") is True
            and stage4i_complete
            and data.get("success") is True
        ),
        "stage4i_complete": stage4i_complete,
        "selected_strategy_present": isinstance(selected_strategy_id, str) and bool(selected_strategy_id),
        "scheduler_lifecycle_activation_accepted": (
            data.get("stage4i6_scheduler_lifecycle_activation_acceptance_report") is True
            and data.get("success") is True
        ),
    }


def _artifact_blockers(checks: dict[str, Any], data: dict[str, Any]) -> list[str]:
    labels = {
        "stage4i6_report_present": "Stage 4I-6 acceptance report is missing",
        "stage4i6_report_ready": "Stage 4I-6 acceptance report is not ready",
        "stage4i_complete": "Stage 4I completion flag is not true",
        "selected_strategy_present": "selected strategy is missing from Stage 4I-6 report",
        "scheduler_lifecycle_activation_accepted": "scheduler/lifecycle activation acceptance is not confirmed",
    }
    blockers = [label for key, label in labels.items() if checks.get(key) is not True]
    if _as_list(data.get("errors")):
        blockers.append("Stage 4I-6 report contains errors")
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


def _stage4i_acceptance_checks(report: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    payload = _mapping(report.get("payload_checks"))
    snapshot = _mapping(report.get("snapshot_checks"))
    artifact = _mapping(report.get("artifact_checks"))
    checks = {
        "scheduler_activation_accepted": _first_present(
            snapshot.get("scheduler_activation_snapshot_matches"),
            artifact.get("scheduler_activation_succeeded"),
            default=report.get("success") is True,
        )
        is True,
        "lifecycle_activation_accepted": _first_present(
            snapshot.get("lifecycle_activation_snapshot_matches"),
            artifact.get("lifecycle_activation_succeeded"),
            default=report.get("success") is True,
        )
        is True,
        "broker_submission_disabled": payload.get("broker_submission_disabled") is True,
        "strategy_scan_execution_disabled": payload.get("strategy_scan_execution_disabled") is True,
        "lifecycle_transition_execution_disabled": payload.get("lifecycle_transition_execution_disabled") is True,
        "market_data_disabled": payload.get("market_data_disabled") is True,
        "contract_qualification_disabled": payload.get("contract_qualification_disabled") is True,
    }
    labels = {
        "scheduler_activation_accepted": "Stage 4I scheduler activation acceptance is not confirmed",
        "lifecycle_activation_accepted": "Stage 4I lifecycle activation acceptance is not confirmed",
        "broker_submission_disabled": "Stage 4I report does not confirm broker submission disabled",
        "strategy_scan_execution_disabled": "Stage 4I report does not confirm strategy scan execution disabled",
        "lifecycle_transition_execution_disabled": "Stage 4I report does not confirm lifecycle transition execution disabled",
        "market_data_disabled": "Stage 4I report does not confirm market data disabled",
        "contract_qualification_disabled": "Stage 4I report does not confirm contract qualification disabled",
    }
    return checks, [label for key, label in labels.items() if checks.get(key) is not True]


def _registry_checks(
    snapshot: dict | list | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    present = snapshot is not None
    warnings: list[str] = []
    blockers: list[str] = []
    candidate_ids: set[str] = set()
    eligible_ids: set[str] = set()
    parse_warning: str | None = None
    parsed_successfully = True

    if not present:
        warnings.append("strategy registry snapshot missing; next phase must verify selected strategy paper eligibility")
        return (
            {
                "strategy_registry_snapshot_present": False,
                "selected_strategy_in_registry": None,
                "selected_strategy_paper_eligible": None,
                "candidate_strategy_ids": [],
                "parse_warning": None,
            },
            blockers,
            warnings,
        )

    try:
        records: list[Any] = []
        if isinstance(snapshot, dict):
            if isinstance(snapshot.get("paper_eligible_strategy_ids"), list):
                records.extend(snapshot.get("paper_eligible_strategy_ids"))
            if isinstance(snapshot.get("strategies"), list):
                records.extend(snapshot.get("strategies"))
        elif isinstance(snapshot, list):
            records.extend(snapshot)
        else:
            parsed_successfully = False
            parse_warning = "strategy registry snapshot has unsupported structure"
        for item in records:
            if isinstance(item, str):
                candidate_ids.add(item)
                eligible_ids.add(item)
            elif isinstance(item, dict):
                strategy_id = item.get("strategy_id")
                if isinstance(strategy_id, str) and strategy_id:
                    candidate_ids.add(strategy_id)
                    if item.get("paper_eligible") is True:
                        eligible_ids.add(strategy_id)
            else:
                continue
    except TypeError as exc:
        parsed_successfully = False
        parse_warning = f"strategy registry parse failure: {type(exc).__name__}: {exc}"

    if parse_warning:
        warnings.append(parse_warning)
    candidate_strategy_ids = sorted(candidate_ids)
    in_registry = selected_strategy_id in candidate_ids if parsed_successfully else None
    paper_eligible = selected_strategy_id in eligible_ids if parsed_successfully else None
    if parsed_successfully and selected_strategy_id not in candidate_ids:
        blockers.append("selected strategy is missing from supplied strategy registry snapshot")
    elif parsed_successfully and selected_strategy_id not in eligible_ids:
        blockers.append("selected strategy is not paper-eligible in supplied strategy registry snapshot")
    return (
        {
            "strategy_registry_snapshot_present": True,
            "selected_strategy_in_registry": in_registry,
            "selected_strategy_paper_eligible": paper_eligible,
            "candidate_strategy_ids": candidate_strategy_ids,
            "parse_warning": parse_warning,
        },
        blockers,
        warnings,
    )


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
    for check_builder, snapshot in (
        (_scheduler_activation_snapshot_checks, scheduler_activation_snapshot),
        (_lifecycle_activation_snapshot_checks, lifecycle_activation_snapshot),
        (_activation_snapshot_checks, activation_snapshot),
    ):
        current_checks, current_blockers, current_warnings = check_builder(
            snapshot, selected_strategy_id
        )
        checks.update(current_checks)
        blockers.extend(current_blockers)
        warnings.extend(current_warnings)
    return checks, blockers, warnings


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
        warnings.append(f"{prefix} snapshot missing; 4J-2 must verify activation artifact")
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
                blockers.append(f"{prefix} snapshot {key} contradicts Stage 4J-1 safety")
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
        warnings.append("activation snapshot missing; 4J-2 must verify one-strategy activation state")
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
                blockers.append(f"activation snapshot {key} contradicts Stage 4J-1 safety")
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
        warnings.append("state snapshot missing; 4J-2 must verify halt, reconciliation, intents, and positions")
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
        warnings.append("risk snapshot missing; 4J-2 must verify kill switch, hard halt, and daily loss controls")
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
    matching_jobs = [job for job in jobs if isinstance(job, dict) and _job_strategy_id(job) == selected_strategy_id]
    selected_job_matches = True
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("scheduler snapshot missing; 4J-2 must verify scheduler state")
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
        blockers.append("scheduler snapshot shows broad/all-strategy scheduler automation enabled")
    if matching_jobs and not selected_job_matches:
        blockers.append("selected strategy scheduler job does not match Stage 4J-1 safety constraints")
    if scan_enabled:
        blockers.append("scheduler snapshot strategy_scan_execution_enabled must be false")
    checks = {
        "scheduler_snapshot_present": present,
        "scheduler_already_enabled": scheduler_enabled,
        "all_strategy_scheduler_enabled": all_strategy_enabled,
        "selected_strategy_job_matches": selected_job_matches,
        "strategy_scan_execution_enabled": scan_enabled,
    }
    return checks, blockers, warnings


def _lifecycle_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; 4J-2 must verify lifecycle state")
    broad_enabled = any(
        data.get(key) is True
        for key in ("all_strategy_lifecycle_enabled", "all_strategies_enabled", "broad_lifecycle_enabled")
    )
    lifecycle_enabled = data.get("lifecycle_automation_enabled") is True or broad_enabled
    transition_enabled = data.get("lifecycle_transition_execution_enabled") is True
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
        blockers.append("lifecycle snapshot shows broad/all-strategy lifecycle automation enabled")
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
        warnings.append("paper broker snapshot missing; 4J-2 must verify PAPER broker config")
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
        blockers.append("market window explicitly blocks controlled PAPER operation planning")
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


def _safety_checks(*mappings: dict[str, Any]) -> dict[str, bool]:
    report_safety = mappings[0] if mappings else {}
    payload_checks = mappings[1] if len(mappings) > 1 else {}
    rest = mappings[2:] if len(mappings) > 2 else ()
    return {
        "no_live_trading": report_safety.get("no_live_trading") is not False and _none_true("live_trading_enabled", *rest),
        "no_all_strategy_enablement": report_safety.get("no_all_strategy_enablement") is not False and _none_true("all_strategies_enabled", *rest),
        "no_broker_submission_enabled": report_safety.get("no_broker_submission_enabled") is not False and _none_true("broker_submission_enabled", *rest),
        "no_market_data": report_safety.get("no_market_data") is not False and payload_checks.get("market_data_disabled") is not False and _none_true("market_data_enabled", *rest),
        "no_contract_qualification": report_safety.get("no_contract_qualification") is not False and payload_checks.get("contract_qualification_disabled") is not False and _none_true("contract_qualification_enabled", *rest),
        "no_order_submission": report_safety.get("no_order_submission") is not False,
        "no_strategy_scan_execution": report_safety.get("no_strategy_scan_execution") is not False and payload_checks.get("strategy_scan_execution_disabled") is not False and _none_true("strategy_scan_execution_enabled", *rest),
        "no_lifecycle_transition_execution": report_safety.get("no_lifecycle_transition_execution") is not False and payload_checks.get("lifecycle_transition_execution_disabled") is not False and _none_true("lifecycle_transition_execution_enabled", *rest),
        "no_direct_scheduler_registration": report_safety.get("no_direct_scheduler_registration") is not False,
        "no_direct_lifecycle_execution": report_safety.get("no_direct_lifecycle_execution") is not False,
        "no_state_write": report_safety.get("no_state_write") is not False,
        "no_ledger_write": report_safety.get("no_ledger_write") is not False,
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
    stage4i_acceptance_checks: dict[str, Any],
    registry_checks: dict[str, Any],
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
        "stage4j1_controlled_paper_operation_readiness_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "stage4i_acceptance_checks": stage4i_acceptance_checks,
        "registry_checks": registry_checks,
        "activation_snapshot_checks": activation_snapshot_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4j2": {
            "ready_to_build_controlled_paper_operation_plan": ready,
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
        "stage4i6_report_present": False,
        "stage4i6_report_ready": False,
        "stage4i_complete": False,
        "selected_strategy_present": False,
        "scheduler_lifecycle_activation_accepted": False,
    }


def _default_selected_strategy() -> dict[str, Any]:
    return {"selected_strategy_id": None, "paper_only": False, "one_strategy_only": False}


def _default_stage4i_acceptance_checks() -> dict[str, Any]:
    return {
        "scheduler_activation_accepted": False,
        "lifecycle_activation_accepted": False,
        "broker_submission_disabled": False,
        "strategy_scan_execution_disabled": False,
        "lifecycle_transition_execution_disabled": False,
        "market_data_disabled": False,
        "contract_qualification_disabled": False,
    }


def _default_registry_checks() -> dict[str, Any]:
    return {
        "strategy_registry_snapshot_present": False,
        "selected_strategy_in_registry": None,
        "selected_strategy_paper_eligible": None,
        "candidate_strategy_ids": [],
        "parse_warning": None,
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
