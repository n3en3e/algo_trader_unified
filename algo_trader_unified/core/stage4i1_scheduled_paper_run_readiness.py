"""Pure Stage 4I-1 scheduled PAPER run readiness report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
PAPER_IBKR_PORTS = {4004}
ORDERED_NEXT_STEPS = [
    "Build Stage 4I-2 first scheduled PAPER automation run plan.",
    "Before any scheduled run, re-check state, activation artifact, risk controls, scheduler, lifecycle, and paper broker config.",
    "Keep live trading disabled.",
    "Keep all other strategies disabled.",
    "Keep broker submission separately gated until the scheduled run phase explicitly permits it.",
]
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding to 4I-2"
)
DO_NOT_DO_YET = [
    "Do not enable live trading.",
    "Do not enable all strategies.",
    "Do not place orders now.",
    "Do not change strategy thresholds.",
    "Do not change position sizing.",
    "Do not bypass risk controls.",
    "Do not enable broad scheduler or lifecycle automation.",
]


def build_stage4i1_scheduled_paper_run_readiness_report(
    *,
    stage4h6_activation_acceptance_report: dict | None,
    activation_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    market_window_snapshot: dict | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a read-only readiness report for Stage 4I-2 run-plan design."""

    try:
        return _json_safe(
            _build_report(
                stage4h6_activation_acceptance_report=stage4h6_activation_acceptance_report,
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
        message = f"unexpected readiness failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_default_selected_strategy(),
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
    stage4h6_activation_acceptance_report: dict | None,
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
    h6 = (
        stage4h6_activation_acceptance_report
        if isinstance(stage4h6_activation_acceptance_report, dict)
        else None
    )
    data = _mapping(h6)
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4h6_activation_acceptance_report is None:
        blockers.append("Stage 4H-6 activation acceptance report is missing")
    elif h6 is None:
        blockers.append("Stage 4H-6 activation acceptance report must be a dict")
        errors.append("Stage 4H-6 activation acceptance report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    artifact_checks = _artifact_checks(stage4h6_activation_acceptance_report)
    blockers.extend(_artifact_blockers(artifact_checks, data))

    selected_strategy_id = _selected_strategy_id(data)
    payload_checks = _mapping(data.get("activation_payload_checks"))
    selected_strategy, selected_blockers = _selected_strategy_checks(
        selected_strategy_id, payload_checks
    )
    blockers.extend(selected_blockers)

    activation_checks, activation_blockers, activation_warnings = _activation_checks(
        activation_snapshot, selected_strategy_id, payload_checks, data
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

    safety_checks = _safety_checks(
        payload_checks,
        activation_snapshot if isinstance(activation_snapshot, dict) else {},
        scheduler_snapshot if isinstance(scheduler_snapshot, dict) else {},
        lifecycle_snapshot if isinstance(lifecycle_snapshot, dict) else {},
        paper_broker_snapshot if isinstance(paper_broker_snapshot, dict) else {},
    )
    blockers.extend(_safety_blockers(safety_checks))

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    no_errors_ready = not blocker_list and not error_list
    artifacts_ready = (
        artifact_checks["stage4h6_report_present"] is True
        and artifact_checks["stage4h6_report_ready"] is True
        and artifact_checks["selected_strategy_present"] is True
        and artifact_checks["activation_accepted"] is True
    )
    selected_strategy_ready = (
        selected_strategy["one_strategy_only"] is True
        and selected_strategy["paper_only"] is True
    )
    activation_ready = (
        activation_checks["activation_snapshot_matches"] is True
        and activation_checks["paper_only"] is True
        and activation_checks["live_trading_disabled"] is True
        and activation_checks["all_strategies_disabled"] is True
        and activation_checks["broker_submission_disabled"] is True
    )
    state_ready = (
        state_checks["active_halt"] is False
        and state_checks["unresolved_needs_reconciliation_count"] == 0
        and (
            state_checks["active_intents_count"] == 0
            or state_checks["active_intents_safe_for_enablement"] is True
        )
    )
    scheduler_ready = (
        scheduler_checks["scheduler_already_enabled"] is False
        and scheduler_checks["all_strategy_scheduler_enabled"] is False
        and scheduler_checks["selected_strategy_job_already_enabled"] is False
    )
    lifecycle_ready = (
        lifecycle_checks["lifecycle_already_enabled"] is False
        and lifecycle_checks["lifecycle_transition_execution_enabled"] is False
    )
    risk_ready = risk_checks["risk_bypass_enabled"] is False
    paper_broker_ready = (
        paper_broker_checks["live_trading_enabled"] is False
        and paper_broker_checks["broker_submission_enabled"] is False
    )
    market_window_ready = market_window_checks["allowed_to_schedule_paper_run"] is not False
    safety_ready = (
        safety_checks["no_live_trading"] is True
        and safety_checks["no_all_strategy_enablement"] is True
        and safety_checks["no_broker_submission_enabled"] is True
        and safety_checks["no_market_data"] is True
        and safety_checks["no_contract_qualification"] is True
        and safety_checks["no_order_submission"] is True
    )
    ready = (
        no_errors_ready
        and artifacts_ready
        and selected_strategy_ready
        and activation_ready
        and state_ready
        and risk_ready
        and scheduler_ready
        and lifecycle_ready
        and paper_broker_ready
        and market_window_ready
        and safety_ready
    )

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
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
    readiness = _mapping(data.get("readiness_for_next_phase"))
    payload_checks = _mapping(data.get("activation_payload_checks"))
    selected_strategy_id = _selected_strategy_id(data)
    return {
        "stage4h6_report_present": isinstance(report, dict),
        "stage4h6_report_ready": (
            data.get("stage4h6_one_strategy_activation_acceptance_report") is True
            and readiness.get("ready_to_build_first_scheduled_paper_automation_run") is True
            and data.get("success") is True
        ),
        "selected_strategy_present": isinstance(selected_strategy_id, str)
        and bool(selected_strategy_id),
        "activation_accepted": (
            payload_checks.get("paper_only") is True
            and payload_checks.get("one_strategy_only") is True
            and payload_checks.get("live_trading_disabled") is True
            and payload_checks.get("all_strategies_disabled") is True
            and payload_checks.get("broker_submission_disabled") is True
        ),
    }


def _artifact_blockers(checks: dict[str, bool], data: dict[str, Any]) -> list[str]:
    blockers = []
    labels = {
        "stage4h6_report_present": "Stage 4H-6 activation acceptance report is missing",
        "stage4h6_report_ready": "Stage 4H-6 activation acceptance report is not ready for Stage 4I-1",
        "selected_strategy_present": "selected strategy is missing from accepted Stage 4H-6 report",
        "activation_accepted": "Stage 4H-6 activation acceptance checks are not clean",
    }
    for key, label in labels.items():
        if checks.get(key) is not True:
            blockers.append(label)
    if _as_list(data.get("errors")):
        blockers.append("Stage 4H-6 report contains errors")
    return blockers


def _selected_strategy_checks(
    selected_strategy_id: str | None, payload_checks: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    one_strategy_only = payload_checks.get("one_strategy_only") is True
    paper_only = payload_checks.get("paper_only") is True
    blockers: list[str] = []
    if not isinstance(selected_strategy_id, str) or not selected_strategy_id:
        blockers.append("selected strategy id is missing or invalid")
    if not one_strategy_only:
        blockers.append("accepted activation must contain exactly one strategy")
    if not paper_only:
        blockers.append("accepted activation must be paper_only true")
    return (
        {
            "selected_strategy_id": selected_strategy_id,
            "paper_only": paper_only,
            "one_strategy_only": one_strategy_only,
        },
        blockers,
    )


def _activation_checks(
    snapshot: dict | None,
    selected_strategy_id: str | None,
    payload_checks: dict[str, Any],
    h6_report: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    matches = True

    checks = {
        "activation_snapshot_present": present,
        "activation_snapshot_matches": True,
        "paper_only": payload_checks.get("paper_only") is True,
        "live_trading_disabled": payload_checks.get("live_trading_disabled") is True,
        "all_strategies_disabled": payload_checks.get("all_strategies_disabled") is True,
        "broker_submission_disabled": payload_checks.get("broker_submission_disabled") is True,
    }
    if not present:
        warnings.append("activation snapshot missing; 4I-2 must verify activation artifact before run planning")
    else:
        records = _activation_records(data)
        active_ids = _as_string_list(data.get("active_strategy_ids"))
        if active_ids and active_ids != [selected_strategy_id]:
            matches = False
            blockers.append("activation snapshot active_strategy_ids do not match selected strategy")
        for record in records:
            if not isinstance(record, dict):
                warnings.append("malformed activation snapshot entry ignored")
                continue
            if record.get("selected_strategy_id") not in (None, selected_strategy_id):
                matches = False
                blockers.append("activation snapshot selected_strategy_id does not match")
            for key, expected in (
                ("paper_only", True),
                ("live_trading_enabled", False),
                ("all_strategies_enabled", False),
                ("broker_submission_enabled", False),
            ):
                if key in record and record.get(key) is not expected:
                    matches = False
                    blockers.append(f"activation snapshot {key} contradicts accepted activation")
            if "enabled_strategy_count" in record and record.get("enabled_strategy_count") != 1:
                matches = False
                blockers.append("activation snapshot enabled_strategy_count must be 1")
        accepted_selected = _selected_strategy_id(h6_report)
        if data.get("selected_strategy_id") not in (None, accepted_selected):
            matches = False
            blockers.append("activation snapshot contradicts Stage 4H-6 selected strategy")

    expected_payload_checks = {
        "paper_only": "activation payload must remain paper_only true",
        "live_trading_disabled": "activation payload enables live trading",
        "all_strategies_disabled": "activation payload enables all strategies",
        "broker_submission_disabled": "activation payload enables broker submission",
    }
    for key, label in expected_payload_checks.items():
        if checks[key] is not True:
            blockers.append(label)
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
        "open_positions_count": _safe_int(data.get("open_positions_count")),
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("state snapshot missing; 4I-2 must verify state immediately")
        return checks, blockers, warnings
    if checks["active_halt"]:
        blockers.append("active halt is present")
    if checks["unresolved_needs_reconciliation_count"] > 0:
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
        warnings.append("risk snapshot missing; 4I-2 must verify risk controls immediately")
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
        warnings.append("scheduler snapshot missing; 4I-2 must verify scheduler state")
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
    transition_enabled = _contains_truthy_flag(data, "lifecycle_transition_execution_enabled")
    checks = {
        "lifecycle_snapshot_present": present,
        "lifecycle_already_enabled": already_enabled,
        "lifecycle_transition_execution_enabled": transition_enabled,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; 4I-2 must verify lifecycle state")
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
        warnings.append("paper broker snapshot missing; 4I-2 must verify paper broker config immediately")
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
        warnings.append("market is currently closed; planning may continue but 4I-2 must verify run timing")
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
    }


def _safety_blockers(checks: dict[str, bool]) -> list[str]:
    labels = {
        "no_live_trading": "live trading safety flag is enabled",
        "no_all_strategy_enablement": "all-strategy safety flag is enabled",
        "no_broker_submission_enabled": "broker submission safety flag is enabled",
        "no_market_data": "market data safety flag is enabled",
        "no_contract_qualification": "contract qualification safety flag is enabled",
        "no_order_submission": "order submission safety flag is enabled",
    }
    return [label for key, label in labels.items() if checks.get(key) is not True]


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
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
    recommendations = {
        "ordered_next_steps": list(ORDERED_NEXT_STEPS),
        "do_not_do_yet": list(DO_NOT_DO_YET),
    }
    if market_window_checks.get("market_window_snapshot_present") is not True:
        recommendations["ordered_next_steps"].append(
            "Manually verify exchange hours and holiday schedules before 4I-2 if no market window snapshot is supplied."
        )
    return {
        "dry_run": True,
        "stage4i1_scheduled_paper_run_readiness_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "activation_checks": activation_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4i2": {
            "ready_to_build_first_scheduled_run_plan": ready,
            "blockers": list(blockers if not ready else []),
            "warnings": list(warnings),
        },
        "recommendations": recommendations,
        "success": ready,
        "errors": list(errors),
        "warnings": list(warnings),
    }


def _selected_strategy_id(report: dict[str, Any]) -> str | None:
    payload_checks = _mapping(report.get("activation_payload_checks"))
    selected_strategy = _mapping(report.get("selected_strategy"))
    activation_payload = _mapping(report.get("activation_payload"))
    for value in (
        payload_checks.get("selected_strategy_id"),
        selected_strategy.get("selected_strategy_id"),
        activation_payload.get("selected_strategy_id"),
        report.get("selected_strategy_id"),
    ):
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


def _default_artifact_checks() -> dict[str, bool]:
    return {
        "stage4h6_report_present": False,
        "stage4h6_report_ready": False,
        "selected_strategy_present": False,
        "activation_accepted": False,
    }


def _default_selected_strategy() -> dict[str, Any]:
    return {"selected_strategy_id": None, "paper_only": False, "one_strategy_only": False}


def _default_activation_checks() -> dict[str, Any]:
    return {
        "activation_snapshot_present": False,
        "activation_snapshot_matches": False,
        "paper_only": False,
        "live_trading_disabled": False,
        "all_strategies_disabled": False,
        "broker_submission_disabled": False,
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
