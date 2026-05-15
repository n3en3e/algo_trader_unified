"""Pure Stage 4I-4 scheduler/lifecycle activation gate report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
PAPER_IBKR_PORTS = {4004}
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding to 4I-5"
)
REQUIRED_OPERATOR_ACKNOWLEDGEMENTS = [
    "I understand this may allow scheduler/lifecycle activation for one PAPER strategy in the next phase.",
    "I understand this does not enable live trading.",
    "I understand this does not enable all strategies.",
    "I understand broker order submission remains separately gated.",
    "I verified state, risk, scheduler, lifecycle, paper broker, and market window snapshots.",
    "I understand strategy scans and orders are not executed in this gate phase.",
]
ORDERED_NEXT_STEPS = [
    "Build Stage 4I-5 one-strategy scheduler/lifecycle activation executor.",
    "Before 4I-5, re-check state, activation artifact, risk controls, scheduler, lifecycle, paper broker config, and market window.",
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
    "Do not register scheduler jobs in 4I-4.",
]


def build_stage4i4_scheduler_lifecycle_activation_gate_report(
    *,
    stage4i3_dry_run_report: dict | None,
    activation_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    market_window_snapshot: dict | None = None,
    operator_acknowledgements: list[str] | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a read-only gate for a future one-strategy scheduled PAPER executor."""

    try:
        return _json_safe(
            _build_report(
                stage4i3_dry_run_report=stage4i3_dry_run_report,
                activation_snapshot=activation_snapshot,
                state_snapshot=state_snapshot,
                risk_snapshot=risk_snapshot,
                scheduler_snapshot=scheduler_snapshot,
                lifecycle_snapshot=lifecycle_snapshot,
                paper_broker_snapshot=paper_broker_snapshot,
                market_window_snapshot=market_window_snapshot,
                operator_acknowledgements=operator_acknowledgements,
                now_provider=now_provider,
            )
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must stay data-only.
        message = f"unexpected activation gate failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_default_selected_strategy(),
                acknowledgement_checks=_acknowledgement_checks(operator_acknowledgements),
                candidate=_activation_candidate(None, False),
                scheduler_activation=_proposed_scheduler_activation(None, False),
                lifecycle_activation=_proposed_lifecycle_activation(None, False),
                runtime_guards=_runtime_guards(),
                pre_activation_checks=_pre_activation_checks(
                    _default_artifact_checks(),
                    _default_activation_checks(),
                    _default_state_checks(),
                    _default_risk_checks(),
                    _default_scheduler_checks(),
                    _default_lifecycle_checks(),
                    _default_paper_broker_checks(),
                    _default_market_window_checks(),
                    _acknowledgement_checks(operator_acknowledgements),
                ),
                post_activation_checks=_post_activation_checks(),
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
    stage4i3_dry_run_report: dict | None,
    activation_snapshot: dict | None,
    state_snapshot: dict | None,
    risk_snapshot: dict | None,
    scheduler_snapshot: dict | None,
    lifecycle_snapshot: dict | None,
    paper_broker_snapshot: dict | None,
    market_window_snapshot: dict | None,
    operator_acknowledgements: list[str] | None,
    now_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    generated_at = _generated_at(now_provider)
    i3 = stage4i3_dry_run_report if isinstance(stage4i3_dry_run_report, dict) else None
    data = _mapping(i3)
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4i3_dry_run_report is None:
        blockers.append("Stage 4I-3 scheduled run dry-run report is missing")
    elif i3 is None:
        blockers.append("Stage 4I-3 scheduled run dry-run report must be a dict")
        errors.append("Stage 4I-3 scheduled run dry-run report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    artifact_checks = _artifact_checks(stage4i3_dry_run_report)
    blockers.extend(_artifact_blockers(artifact_checks, data))
    blockers.extend(_stage4i3_safety_blockers(_mapping(data.get("safety_checks"))))

    selected_strategy_id = _selected_strategy_id(data)
    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    blockers.extend(selected_blockers)

    trace = data.get("dry_run_trace", [])
    trace_checks, trace_blockers = _dry_run_trace_checks(trace)
    artifact_checks["dry_run_trace_present"] = trace_checks["trace_available"]
    artifact_checks["dry_run_trace_clean"] = not trace_blockers
    blockers.extend(trace_blockers)
    blockers.extend(_source_trace_check_blockers(_mapping(data.get("dry_run_trace_checks"))))

    acknowledgement_checks = _acknowledgement_checks(operator_acknowledgements)
    if acknowledgement_checks["exact_match"] is not True:
        blockers.append("required operator acknowledgements are missing")

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

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    gates_pass = not blocker_list and not error_list

    candidate = _activation_candidate(selected_strategy_id, gates_pass)
    scheduler_activation = _proposed_scheduler_activation(selected_strategy_id, gates_pass)
    lifecycle_activation = _proposed_lifecycle_activation(selected_strategy_id, gates_pass)
    pre_activation_checks = _pre_activation_checks(
        artifact_checks,
        activation_checks,
        state_checks,
        risk_checks,
        scheduler_checks,
        lifecycle_checks,
        paper_broker_checks,
        market_window_checks,
        acknowledgement_checks,
    )
    ready = (
        gates_pass
        and candidate["available"] is True
        and scheduler_activation["proposed_enabled_in_4I5"] is True
        and lifecycle_activation["proposed_enabled_in_4I5"] is True
        and candidate["broker_submission_allowed_next_phase"] is False
        and safety_checks["no_live_trading"] is True
        and safety_checks["no_all_strategy_enablement"] is True
        and safety_checks["no_market_data"] is True
        and safety_checks["no_contract_qualification"] is True
        and safety_checks["no_order_submission"] is True
    )

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
        acknowledgement_checks=acknowledgement_checks,
        candidate=candidate,
        scheduler_activation=scheduler_activation,
        lifecycle_activation=lifecycle_activation,
        runtime_guards=_runtime_guards(),
        pre_activation_checks=pre_activation_checks,
        post_activation_checks=_post_activation_checks(),
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
    readiness = _mapping(data.get("readiness_for_stage4i4"))
    selected_strategy_id = _selected_strategy_id(data)
    trace = data.get("dry_run_trace", [])
    trace_present = isinstance(trace, list) and bool(trace)
    trace_checks, trace_blockers = _dry_run_trace_checks(trace)
    return {
        "stage4i3_report_present": isinstance(report, dict),
        "stage4i3_report_ready": (
            data.get("stage4i3_scheduled_run_dry_run_report") is True
            and readiness.get("ready_to_build_scheduler_activation_gate") is True
            and data.get("success") is True
        ),
        "selected_strategy_present": isinstance(selected_strategy_id, str)
        and bool(selected_strategy_id),
        "dry_run_trace_present": trace_present,
        "dry_run_trace_clean": trace_present and trace_checks["trace_available"] and not trace_blockers,
    }


def _artifact_blockers(checks: dict[str, bool], data: dict[str, Any]) -> list[str]:
    labels = {
        "stage4i3_report_present": "Stage 4I-3 scheduled run dry-run report is missing",
        "stage4i3_report_ready": "Stage 4I-3 scheduled run dry-run report is not ready for Stage 4I-4",
        "selected_strategy_present": "selected strategy is missing from accepted Stage 4I-3 report",
        "dry_run_trace_present": "dry-run trace is missing from accepted Stage 4I-3 report",
        "dry_run_trace_clean": "dry-run trace is not clean",
    }
    blockers = [label for key, label in labels.items() if checks.get(key) is not True]
    if _as_list(data.get("errors")):
        blockers.append("Stage 4I-3 report contains errors")
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
    if paper_only is not True:
        blockers.append("Stage 4I-3 selected strategy must be paper_only true")
    if one_strategy_only is not True:
        blockers.append("Stage 4I-3 selected strategy must be one_strategy_only true")
    return (
        {
            "selected_strategy_id": selected_strategy_id,
            "paper_only": paper_only,
            "one_strategy_only": one_strategy_only,
        },
        blockers,
    )


def _dry_run_trace_checks(trace: Any) -> tuple[dict[str, bool], list[str]]:
    trace_items = trace if isinstance(trace, list) else []
    checks = {
        "trace_available": isinstance(trace, list) and bool(trace),
        "all_trace_items_simulated": True,
        "all_trace_items_would_execute_false": True,
        "all_trace_items_would_submit_false_where_supplied": True,
        "all_trace_items_would_write_state_false": True,
        "all_trace_items_would_write_ledger_false": True,
        "all_trace_items_would_register_scheduler_false": True,
    }
    blockers: list[str] = []
    if not checks["trace_available"]:
        blockers.append("dry-run trace is unavailable")
        checks.update({key: False for key in checks if key != "trace_available"})
        return checks, blockers
    for item in trace_items:
        if not isinstance(item, dict):
            for key in checks:
                if key != "trace_available":
                    checks[key] = False
            blockers.append("dry-run trace contains malformed item")
            continue
        if not (item.get("status") == "simulated" or item.get("simulated") is True):
            checks["all_trace_items_simulated"] = False
            blockers.append("dry-run trace contains non-simulated item")
        if item.get("would_execute") is not False:
            checks["all_trace_items_would_execute_false"] = False
            blockers.append("dry-run trace item would execute")
        if "would_submit" in item and item.get("would_submit") is not False:
            checks["all_trace_items_would_submit_false_where_supplied"] = False
            blockers.append("dry-run trace item would submit")
        if item.get("would_write_state") is not False:
            checks["all_trace_items_would_write_state_false"] = False
            blockers.append("dry-run trace item would write state")
        if item.get("would_write_ledger") is not False:
            checks["all_trace_items_would_write_ledger_false"] = False
            blockers.append("dry-run trace item would write ledger")
        if item.get("would_register_scheduler") is not False:
            checks["all_trace_items_would_register_scheduler_false"] = False
            blockers.append("dry-run trace item would register scheduler")
    return checks, _dedupe(blockers)


def _source_trace_check_blockers(checks: dict[str, Any]) -> list[str]:
    labels = {
        "no_strategy_scan_called": "Stage 4I-3 dry-run checks do not confirm strategy scan was skipped",
        "no_intent_created": "Stage 4I-3 dry-run checks do not confirm intent creation was skipped",
        "no_ticket_created": "Stage 4I-3 dry-run checks do not confirm ticket creation was skipped",
        "no_broker_submission": "Stage 4I-3 dry-run checks do not confirm broker submission was skipped",
        "no_state_write": "Stage 4I-3 dry-run checks do not confirm state writes were skipped",
        "no_ledger_write": "Stage 4I-3 dry-run checks do not confirm ledger writes were skipped",
        "no_scheduler_registration": "Stage 4I-3 dry-run checks do not confirm scheduler registration was skipped",
        "no_lifecycle_execution": "Stage 4I-3 dry-run checks do not confirm lifecycle execution was skipped",
    }
    return [label for key, label in labels.items() if checks.get(key) is not True]


def _stage4i3_safety_blockers(safety_checks: dict[str, Any]) -> list[str]:
    labels = {
        "no_live_trading": "Stage 4I-3 safety checks do not confirm live trading is disabled",
        "no_all_strategy_enablement": "Stage 4I-3 safety checks do not confirm all-strategy automation is disabled",
        "no_broker_submission_enabled": "Stage 4I-3 safety checks do not confirm broker submission is disabled",
        "no_market_data": "Stage 4I-3 safety checks do not confirm market data is disabled",
        "no_contract_qualification": "Stage 4I-3 safety checks do not confirm contract qualification is disabled",
        "no_order_submission": "Stage 4I-3 safety checks do not confirm order submission is disabled",
        "no_strategy_scan_execution": "Stage 4I-3 safety checks do not confirm strategy scan execution is disabled",
        "no_scheduler_registration": "Stage 4I-3 safety checks do not confirm scheduler registration is disabled",
        "no_lifecycle_execution": "Stage 4I-3 safety checks do not confirm lifecycle execution is disabled",
        "no_state_write": "Stage 4I-3 safety checks do not confirm state writes are disabled",
        "no_ledger_write": "Stage 4I-3 safety checks do not confirm ledger writes are disabled",
    }
    return [label for key, label in labels.items() if safety_checks.get(key) is not True]


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


def _activation_candidate(selected_strategy_id: str | None, available: bool) -> dict[str, Any]:
    return {
        "available": available,
        "selected_strategy_id": selected_strategy_id,
        "paper_only": True,
        "one_strategy_only": True,
        "activation_scope": "single_strategy_scheduled_paper_run",
        "scheduler_activation_allowed_next_phase": available,
        "lifecycle_activation_allowed_next_phase": available,
        "broker_submission_allowed_next_phase": False,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "strategy_scan_execution_enabled_now": False,
        "order_submission_enabled_now": False,
        "enabled_strategy_count": 1,
    }


def _proposed_scheduler_activation(
    selected_strategy_id: str | None, available: bool
) -> dict[str, Any]:
    return {
        "available": available,
        "selected_strategy_id": selected_strategy_id,
        "paper_only": True,
        "one_strategy_only": True,
        "would_register_in_4I4": False,
        "proposed_enabled_in_4I5": available,
        "scheduler_job_enabled_now": False,
        "job_scope": "single_strategy",
        "job_payload": {
            "selected_strategy_id": selected_strategy_id,
            "dry_run_only_until_4I5": True,
            "live_trading_enabled": False,
            "all_strategies_enabled": False,
            "broker_submission_enabled": False,
        },
    }


def _proposed_lifecycle_activation(
    selected_strategy_id: str | None, available: bool
) -> dict[str, Any]:
    return {
        "available": available,
        "selected_strategy_id": selected_strategy_id,
        "paper_only": True,
        "one_strategy_only": True,
        "would_execute_in_4I4": False,
        "proposed_enabled_in_4I5": available,
        "lifecycle_execution_enabled_now": False,
        "lifecycle_scope": "single_strategy",
        "lifecycle_payload": {
            "selected_strategy_id": selected_strategy_id,
            "dry_run_only_until_4I5": True,
            "live_trading_enabled": False,
            "all_strategies_enabled": False,
            "broker_submission_enabled": False,
        },
    }


def _runtime_guards() -> dict[str, bool]:
    return {
        "paper_only_mode": True,
        "live_trading_disabled": True,
        "one_strategy_only": True,
        "all_strategy_automation_disabled": True,
        "broker_submission_disabled": True,
        "market_data_disabled_until_explicit_phase": True,
        "contract_qualification_disabled_until_explicit_phase": True,
        "state_clean_required": True,
        "risk_controls_required": True,
        "scheduler_clean_required": True,
        "lifecycle_clean_required": True,
        "market_window_required": True,
        "activation_artifact_required": True,
    }


def _pre_activation_checks(
    artifact_checks: dict[str, Any],
    activation_checks: dict[str, Any],
    state_checks: dict[str, Any],
    risk_checks: dict[str, Any],
    scheduler_checks: dict[str, Any],
    lifecycle_checks: dict[str, Any],
    paper_broker_checks: dict[str, Any],
    market_window_checks: dict[str, Any],
    acknowledgement_checks: dict[str, Any],
) -> dict[str, bool]:
    return {
        "stage4i3_dry_run_accepted": artifact_checks.get("stage4i3_report_ready") is True,
        "dry_run_trace_clean": artifact_checks.get("dry_run_trace_clean") is True,
        "activation_snapshot_clean": activation_checks.get("activation_snapshot_matches") is True,
        "state_snapshot_clean": (
            state_checks.get("active_halt") is False
            and state_checks.get("unresolved_needs_reconciliation_count") == 0
            and (
                state_checks.get("active_intents_count") == 0
                or state_checks.get("active_intents_safe_for_enablement") is True
            )
        ),
        "risk_snapshot_clean": (
            risk_checks.get("risk_snapshot_present") is not True
            or (
                risk_checks.get("kill_switch_available") is True
                and risk_checks.get("hard_halt_available") is True
                and risk_checks.get("daily_loss_limit_available") is True
                and risk_checks.get("risk_bypass_enabled") is False
                and risk_checks.get("max_position_limit_available") is not False
            )
        ),
        "scheduler_snapshot_clean": (
            scheduler_checks.get("scheduler_already_enabled") is False
            and scheduler_checks.get("all_strategy_scheduler_enabled") is False
            and scheduler_checks.get("selected_strategy_job_already_enabled") is False
        ),
        "lifecycle_snapshot_clean": (
            lifecycle_checks.get("lifecycle_already_enabled") is False
            and lifecycle_checks.get("lifecycle_transition_execution_enabled") is False
        ),
        "paper_broker_snapshot_clean": (
            paper_broker_checks.get("paper_config_valid") is True
            and paper_broker_checks.get("live_trading_enabled") is False
            and paper_broker_checks.get("broker_submission_enabled") is False
        ),
        "market_window_snapshot_clean": (
            market_window_checks.get("allowed_to_schedule_paper_run") is not False
        ),
        "acknowledgements_exact": acknowledgement_checks.get("exact_match") is True,
    }


def _post_activation_checks() -> dict[str, bool]:
    return {
        "verify_single_strategy_enabled_only": True,
        "verify_scheduler_scope_single_strategy": True,
        "verify_lifecycle_scope_single_strategy": True,
        "verify_live_trading_still_disabled": True,
        "verify_all_strategy_automation_still_disabled": True,
        "verify_broker_submission_still_separately_gated": True,
        "verify_no_orders_submitted_by_gate": True,
        "verify_no_state_or_ledger_writes_by_gate": True,
    }


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
        warnings.append("activation snapshot missing; 4I-5 must verify activation artifact before activation")
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
                blockers.append(f"activation snapshot {key} contradicts Stage 4I-4 safety")
            elif key in record and record.get(key) is not False:
                matches = False
                blockers.append(f"activation snapshot {key} must be a strict false boolean")
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
        warnings.append("state snapshot missing; 4I-5 must verify runtime state immediately")
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
        warnings.append("risk snapshot missing; 4I-5 must verify risk controls immediately")
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
        warnings.append("scheduler snapshot missing; 4I-5 must verify scheduler state")
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
        warnings.append("lifecycle snapshot missing; 4I-5 must verify lifecycle state")
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
        warnings.append("paper broker snapshot missing; 4I-5 must verify paper broker config immediately")
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
    checks["live_trading_enabled"] = data.get("live_trading_enabled") is True
    checks["broker_submission_enabled"] = data.get("broker_submission_enabled") is True
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
        warnings.append("market is currently closed; gate may continue but 4I-5 must verify run timing")
    if data.get("is_trading_day") is False:
        warnings.append("snapshot is not a trading day; gate may continue but 4I-5 must verify run timing")
    return checks, blockers, warnings


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
        "no_state_write": not _contains_truthy_or_malformed_flag(sources, "state_write_enabled"),
        "no_ledger_write": not _contains_truthy_or_malformed_flag(sources, "ledger_write_enabled"),
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
        "no_state_write": "state write safety flag is enabled or malformed",
        "no_ledger_write": "ledger write safety flag is enabled or malformed",
    }
    return [label for key, label in labels.items() if checks.get(key) is not True]


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    acknowledgement_checks: dict[str, Any],
    candidate: dict[str, Any],
    scheduler_activation: dict[str, Any],
    lifecycle_activation: dict[str, Any],
    runtime_guards: dict[str, Any],
    pre_activation_checks: dict[str, Any],
    post_activation_checks: dict[str, Any],
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
        "stage4i4_scheduler_lifecycle_activation_gate_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "acknowledgement_checks": acknowledgement_checks,
        "scheduler_lifecycle_activation_candidate": candidate,
        "proposed_scheduler_activation": scheduler_activation,
        "proposed_lifecycle_activation": lifecycle_activation,
        "proposed_runtime_guards": runtime_guards,
        "proposed_pre_activation_checks": pre_activation_checks,
        "proposed_post_activation_checks": post_activation_checks,
        "activation_checks": activation_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4i5": {
            "ready_to_build_scheduler_lifecycle_activation_executor": ready,
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


def _default_artifact_checks() -> dict[str, bool]:
    return {
        "stage4i3_report_present": False,
        "stage4i3_report_ready": False,
        "selected_strategy_present": False,
        "dry_run_trace_present": False,
        "dry_run_trace_clean": False,
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
