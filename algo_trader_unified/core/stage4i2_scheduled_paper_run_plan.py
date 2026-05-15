"""Pure Stage 4I-2 scheduled PAPER run plan report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
PAPER_IBKR_PORTS = {4004}
ALLOWED_CADENCES = {"once", "daily", "market_open", "market_close"}
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding to 4I-3"
)
ORDERED_NEXT_STEPS = [
    "Build Stage 4I-3 scheduled run dry run.",
    "Before 4I-3, re-check state, activation artifact, risk controls, scheduler, lifecycle, paper broker config, and market window.",
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
    "Do not enable scheduler jobs now.",
]
DISABLED_COMPONENTS = [
    "broker_submission",
    "live_trading",
    "all_strategy_automation",
    "real_scheduler_registration",
    "lifecycle_transition_execution",
    "market_data_fetch",
    "contract_qualification",
    "strategy_scan_execution",
    "state_mutation",
    "ledger_write",
]
REQUIRED_SNAPSHOTS_FOR_4I3 = [
    "stage4i1_readiness_report",
    "activation_snapshot",
    "state_snapshot",
    "risk_snapshot",
    "scheduler_snapshot",
    "lifecycle_snapshot",
    "paper_broker_snapshot",
    "market_window_snapshot",
]


def build_stage4i2_scheduled_paper_run_plan_report(
    *,
    stage4i1_readiness_report: dict | None,
    activation_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    market_window_snapshot: dict | None = None,
    run_window_config: dict | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a read-only plan for the first scheduled one-strategy PAPER run."""

    try:
        return _json_safe(
            _build_report(
                stage4i1_readiness_report=stage4i1_readiness_report,
                activation_snapshot=activation_snapshot,
                state_snapshot=state_snapshot,
                risk_snapshot=risk_snapshot,
                scheduler_snapshot=scheduler_snapshot,
                lifecycle_snapshot=lifecycle_snapshot,
                paper_broker_snapshot=paper_broker_snapshot,
                market_window_snapshot=market_window_snapshot,
                run_window_config=run_window_config,
                now_provider=now_provider,
            )
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must stay data-only.
        message = f"unexpected run plan failure: {type(exc).__name__}: {exc}"
        generated_at = _generated_at(now_provider)
        return _json_safe(
            _base_report(
                generated_at=generated_at,
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_default_selected_strategy(),
                run_plan=_empty_run_plan(),
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
    stage4i1_readiness_report: dict | None,
    activation_snapshot: dict | None,
    state_snapshot: dict | None,
    risk_snapshot: dict | None,
    scheduler_snapshot: dict | None,
    lifecycle_snapshot: dict | None,
    paper_broker_snapshot: dict | None,
    market_window_snapshot: dict | None,
    run_window_config: dict | None,
    now_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    generated_at = _generated_at(now_provider)
    generated_date = _date_string(generated_at)
    i1 = stage4i1_readiness_report if isinstance(stage4i1_readiness_report, dict) else None
    data = _mapping(i1)
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4i1_readiness_report is None:
        blockers.append("Stage 4I-1 readiness report is missing")
    elif i1 is None:
        blockers.append("Stage 4I-1 readiness report must be a dict")
        errors.append("Stage 4I-1 readiness report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    artifact_checks = _artifact_checks(stage4i1_readiness_report)
    blockers.extend(_artifact_blockers(artifact_checks, data))
    blockers.extend(_stage4i1_safety_blockers(_mapping(data.get("safety_checks"))))

    selected_strategy_id = _selected_strategy_id(data)
    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    blockers.extend(selected_blockers)

    run_window, window_blockers, window_warnings = _run_window(run_window_config)
    blockers.extend(window_blockers)
    warnings.extend(window_warnings)

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

    safety_checks = _safety_checks(
        data,
        activation_snapshot if isinstance(activation_snapshot, dict) else {},
        scheduler_snapshot if isinstance(scheduler_snapshot, dict) else {},
        lifecycle_snapshot if isinstance(lifecycle_snapshot, dict) else {},
        paper_broker_snapshot if isinstance(paper_broker_snapshot, dict) else {},
    )
    blockers.extend(_safety_blockers(safety_checks))

    run_plan = _run_plan(
        selected_strategy_id=selected_strategy_id,
        run_window=run_window,
        generated_date=generated_date,
        available=(
            artifact_checks["stage4i1_report_ready"] is True
            and selected_strategy["selected_strategy_id"] is not None
            and selected_strategy["paper_only"] is True
            and selected_strategy["one_strategy_only"] is True
        ),
    )
    plan_blockers = _run_plan_blockers(run_plan)
    blockers.extend(plan_blockers)

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    ready = not blocker_list and not error_list

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
        run_plan=run_plan,
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
    readiness = _mapping(data.get("readiness_for_stage4i2"))
    selected_strategy_id = _selected_strategy_id(data)
    return {
        "stage4i1_report_present": isinstance(report, dict),
        "stage4i1_report_ready": (
            data.get("stage4i1_scheduled_paper_run_readiness_report") is True
            and readiness.get("ready_to_build_first_scheduled_run_plan") is True
            and data.get("success") is True
        ),
        "selected_strategy_present": isinstance(selected_strategy_id, str)
        and bool(selected_strategy_id),
    }


def _artifact_blockers(checks: dict[str, bool], data: dict[str, Any]) -> list[str]:
    blockers = []
    labels = {
        "stage4i1_report_present": "Stage 4I-1 readiness report is missing",
        "stage4i1_report_ready": "Stage 4I-1 readiness report is not ready for Stage 4I-2",
        "selected_strategy_present": "selected strategy is missing from accepted Stage 4I-1 report",
    }
    for key, label in labels.items():
        if checks.get(key) is not True:
            blockers.append(label)
    if _as_list(data.get("errors")):
        blockers.append("Stage 4I-1 report contains errors")
    return blockers


def _stage4i1_safety_blockers(safety_checks: dict[str, Any]) -> list[str]:
    labels = {
        "no_live_trading": "Stage 4I-1 safety checks do not confirm live trading is disabled",
        "no_all_strategy_enablement": "Stage 4I-1 safety checks do not confirm all-strategy automation is disabled",
        "no_broker_submission_enabled": "Stage 4I-1 safety checks do not confirm broker submission is disabled",
        "no_market_data": "Stage 4I-1 safety checks do not confirm market data is disabled",
        "no_contract_qualification": "Stage 4I-1 safety checks do not confirm contract qualification is disabled",
        "no_order_submission": "Stage 4I-1 safety checks do not confirm order submission is disabled",
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
        blockers.append("Stage 4I-1 selected strategy must be paper_only true")
    if not one_strategy_only:
        blockers.append("Stage 4I-1 selected strategy must be one_strategy_only true")
    return (
        {
            "selected_strategy_id": selected_strategy_id,
            "paper_only": paper_only,
            "one_strategy_only": one_strategy_only,
        },
        blockers,
    )


def _run_window(config: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(config)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("run window config missing; deterministic safe defaults are used for planning only")
    cadence = _normalize_cadence(data.get("cadence"))
    if cadence not in ALLOWED_CADENCES:
        blockers.append("run_window_config cadence is unsupported")
    if data.get("dry_run_only") is False:
        blockers.append("run_window_config dry_run_only must not be false")
    return (
        {
            "schedule_label": _safe_text(
                data.get("schedule_label"), "stage4i2_first_scheduled_paper_run"
            ),
            "timezone": _safe_text(data.get("timezone"), "America/New_York"),
            "first_run_date": _safe_text(data.get("first_run_date"), None),
            "run_time": _safe_text(data.get("run_time"), "09:45"),
            "cadence": cadence,
            "max_runtime_seconds": _safe_optional_int(data.get("max_runtime_seconds")),
            "dry_run_only": True,
            "operator_notes": _safe_text(data.get("operator_notes"), None),
        },
        blockers,
        warnings,
    )


def _run_plan(
    *,
    selected_strategy_id: str | None,
    run_window: dict[str, Any],
    generated_date: str,
    available: bool,
) -> dict[str, Any]:
    strategy_id = selected_strategy_id or ""
    cadence = run_window["cadence"]
    schedule_label = run_window["schedule_label"]
    run_time = run_window["run_time"]
    proposed_schedule = {
        "schedule_label": schedule_label,
        "selected_strategy_id": selected_strategy_id,
        "cadence": cadence,
        "timezone": run_window["timezone"],
        "first_run_date": run_window["first_run_date"] or generated_date,
        "run_time": run_time,
        "dry_run_only": True,
        "would_register": False,
        "would_execute": False,
        "scheduler_job_enabled": False,
    }
    return {
        "available": available,
        "proposed_run_id": _proposed_run_id(
            strategy_id, schedule_label, cadence, run_time, generated_date
        ),
        "run_scope": "single_strategy_paper_scheduled_run",
        "paper_only": True,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "broker_submission_enabled": False,
        "proposed_schedule": proposed_schedule,
        "proposed_pre_run_gates": _proposed_pre_run_gates(),
        "proposed_execution_flow": _proposed_execution_flow(selected_strategy_id),
        "proposed_post_run_checks": _proposed_post_run_checks(selected_strategy_id),
        "proposed_alerts_and_reports": _proposed_alerts_and_reports(selected_strategy_id),
        "disabled_components": list(DISABLED_COMPONENTS),
        "required_snapshots_for_4I3": list(REQUIRED_SNAPSHOTS_FOR_4I3),
    }


def _proposed_pre_run_gates() -> dict[str, Any]:
    return {
        "activation_accepted": True,
        "state_clean": True,
        "risk_controls_available": True,
        "scheduler_clean": True,
        "lifecycle_clean": True,
        "paper_broker_config_valid": True,
        "market_window_allowed": True,
        "one_strategy_only": True,
        "live_trading_disabled": True,
        "all_strategies_disabled": True,
        "broker_submission_disabled": True,
        "manual_operator_review_required_before_4I3": True,
    }


def _proposed_execution_flow(selected_strategy_id: str | None) -> list[dict[str, Any]]:
    stages = [
        ("pre_run_snapshot_check", "read_only_snapshot_inputs"),
        ("risk_gate_check", "risk_controls_snapshot"),
        ("activation_artifact_check", "accepted_activation_artifact"),
        ("scheduler_gate_check", "scheduler_snapshot"),
        ("lifecycle_gate_check", "lifecycle_snapshot"),
        ("market_window_check", "market_window_snapshot"),
        ("strategy_scan_preview", "strategy_preview_only"),
        ("signal_to_intent_preview", "signal_intent_preview"),
        ("intent_to_ticket_preview", "paper_ticket_preview"),
        ("paper_order_submission_gate_preview", "manual_broker_submission_gate"),
        ("state_ledger_tracking_preview", "state_and_ledger_preview"),
        ("alert_report_preview", "operator_alert_report_preview"),
        ("post_run_reconciliation_preview", "reconciliation_preview"),
    ]
    flow = []
    for index, (stage, component) in enumerate(stages, start=1):
        flow.append(
            {
                "sequence_number": index,
                "stage": stage,
                "target_component": component,
                "payload": {"selected_strategy_id": selected_strategy_id},
                "would_execute": False,
                "would_submit": False,
                "paper_only": True,
                "live_trading_enabled": False,
            }
        )
    return flow


def _proposed_post_run_checks(selected_strategy_id: str | None) -> list[dict[str, Any]]:
    return [
        {
            "check": "dry_run_summary_review",
            "selected_strategy_id": selected_strategy_id,
            "would_execute": False,
        },
        {
            "check": "state_and_ledger_reconciliation_preview",
            "selected_strategy_id": selected_strategy_id,
            "would_execute": False,
        },
    ]


def _proposed_alerts_and_reports(selected_strategy_id: str | None) -> list[dict[str, Any]]:
    return [
        {
            "report": "operator_stage4i2_plan_summary",
            "selected_strategy_id": selected_strategy_id,
            "would_send": False,
        },
        {
            "report": "stage4i3_snapshot_requirements",
            "selected_strategy_id": selected_strategy_id,
            "would_send": False,
        },
    ]


def _run_plan_blockers(run_plan: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if run_plan.get("available") is not True:
        blockers.append("run plan is not available")
    schedule = _mapping(run_plan.get("proposed_schedule"))
    if not schedule:
        blockers.append("proposed schedule is missing")
    if schedule.get("would_register") is not False:
        blockers.append("proposed schedule would register a job")
    if schedule.get("would_execute") is not False:
        blockers.append("proposed schedule would execute")
    if schedule.get("scheduler_job_enabled") is not False:
        blockers.append("proposed schedule enables a scheduler job")
    flow = _as_list(run_plan.get("proposed_execution_flow"))
    if not flow:
        blockers.append("proposed execution flow is missing")
    for step in flow:
        if not isinstance(step, dict):
            blockers.append("proposed execution flow contains malformed step")
            continue
        if step.get("would_execute") is not False:
            blockers.append("proposed execution flow contains executable step")
        if step.get("would_submit") is not False:
            blockers.append("proposed execution flow contains submission step")
    return blockers


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
        warnings.append("activation snapshot missing; 4I-3 must verify activation artifact before dry run")
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
                blockers.append(f"activation snapshot {key} contradicts Stage 4I-2 safety")
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
        warnings.append("state snapshot missing; 4I-3 must verify runtime state immediately")
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
        warnings.append("risk snapshot missing; 4I-3 must verify risk controls immediately")
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
        warnings.append("scheduler snapshot missing; 4I-3 must verify scheduler state")
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
        warnings.append("lifecycle snapshot missing; 4I-3 must verify lifecycle state")
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
        warnings.append("paper broker snapshot missing; 4I-3 must verify paper broker config immediately")
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
        warnings.append("market is currently closed; planning may continue but 4I-3 must verify run timing")
    if data.get("is_trading_day") is False:
        warnings.append("snapshot is not a trading day; planning may continue but 4I-3 must verify run timing")
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
        "no_lifecycle_execution": not _contains_truthy_flag(
            sources, "lifecycle_execution_enabled"
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
        "no_strategy_scan_execution": "strategy scan execution safety flag is enabled",
        "no_scheduler_registration": "scheduler registration safety flag is enabled",
        "no_lifecycle_execution": "lifecycle execution safety flag is enabled",
    }
    return [label for key, label in labels.items() if checks.get(key) is not True]


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    run_plan: dict[str, Any],
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
        "stage4i2_scheduled_paper_run_plan_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "run_plan": run_plan,
        "activation_checks": activation_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4i3": {
            "ready_to_build_scheduled_run_dry_run": ready,
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


def _empty_run_plan() -> dict[str, Any]:
    return {
        "available": False,
        "proposed_run_id": None,
        "run_scope": "single_strategy_paper_scheduled_run",
        "proposed_schedule": {},
        "proposed_pre_run_gates": {},
        "proposed_execution_flow": [],
        "proposed_post_run_checks": [],
        "proposed_alerts_and_reports": [],
        "disabled_components": list(DISABLED_COMPONENTS),
        "required_snapshots_for_4I3": list(REQUIRED_SNAPSHOTS_FOR_4I3),
    }


def _default_artifact_checks() -> dict[str, bool]:
    return {
        "stage4i1_report_present": False,
        "stage4i1_report_ready": False,
        "selected_strategy_present": False,
    }


def _default_selected_strategy() -> dict[str, Any]:
    return {"selected_strategy_id": None, "paper_only": False, "one_strategy_only": False}


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


def _safe_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return _safe_int(value)


def _safe_text(value: Any, default: str | None) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _normalize_cadence(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return "once"


def _proposed_run_id(
    strategy_id: str, schedule_label: str | None, cadence: str, run_time: str, generated_date: str
) -> str:
    return "_".join(
        [
            _id_part(strategy_id or "unknown_strategy"),
            _id_part(schedule_label or "stage4i2"),
            _id_part(cadence or "once"),
            _id_part(run_time or "time"),
            _id_part(generated_date),
        ]
    )


def _id_part(value: str) -> str:
    cleaned = []
    for char in value.strip().lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in ("-", "_"):
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "unknown"


def _date_string(generated_at: str) -> str:
    if "T" in generated_at:
        return generated_at.split("T", 1)[0]
    if len(generated_at) >= 10:
        return generated_at[:10]
    return "1970-01-01"


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
