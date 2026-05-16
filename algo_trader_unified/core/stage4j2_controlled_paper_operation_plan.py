"""Pure Stage 4J-2 controlled scheduled PAPER operation plan report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from pathlib import Path
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
PAPER_IBKR_PORTS = {4004}
ALLOWED_CADENCES = {"once", "daily", "market_open", "market_close"}
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding to 4J-3"
)
ORDERED_NEXT_STEPS = [
    "Build Stage 4J-3 controlled scheduled PAPER operation dry run.",
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
    "Do not enable market data now.",
    "Do not enable contract qualification now.",
    "Do not change strategy thresholds.",
    "Do not change position sizing.",
    "Do not bypass risk controls.",
    "Do not enable broker submission broadly.",
]
DISABLED_COMPONENTS = [
    "live_trading",
    "all_strategy_automation",
    "broker_submission",
    "order_submission",
    "market_data_fetch",
    "contract_qualification",
    "strategy_scan_execution",
    "lifecycle_transition_execution",
    "direct_scheduler_registration",
    "direct_lifecycle_execution",
    "state_mutation",
    "ledger_write",
]
REQUIRED_INPUTS_FOR_4J3 = [
    "stage4j2_operation_plan_report",
    "scheduler_activation_snapshot",
    "lifecycle_activation_snapshot",
    "activation_snapshot",
    "state_snapshot",
    "risk_snapshot",
    "scheduler_snapshot",
    "lifecycle_snapshot",
    "paper_broker_snapshot",
    "market_window_snapshot",
    "operator_approval_for_dry_run_only",
]
FLOW_STAGES = [
    ("pre_operation_snapshot_check", "read_only_snapshot_inputs"),
    ("risk_gate_check", "risk_controls_snapshot"),
    ("activation_artifact_check", "accepted_activation_artifacts"),
    ("scheduler_lifecycle_state_check", "scheduler_lifecycle_snapshots"),
    ("market_window_check", "market_window_snapshot"),
    ("selected_strategy_operation_preview", "selected_strategy_preview_only"),
    ("market_data_gate_preview", "market_data_gate"),
    ("contract_qualification_gate_preview", "contract_qualification_gate"),
    ("strategy_scan_gate_preview", "strategy_scan_gate"),
    ("signal_to_intent_gate_preview", "signal_to_intent_gate"),
    ("order_ticket_gate_preview", "order_ticket_gate"),
    ("broker_submission_gate_preview", "broker_submission_gate"),
    ("state_ledger_tracking_gate_preview", "state_ledger_tracking_gate"),
    ("alert_report_gate_preview", "operator_alert_report_gate"),
    ("post_operation_reconciliation_gate_preview", "post_operation_reconciliation_gate"),
]


def build_stage4j2_controlled_paper_operation_plan_report(
    *,
    stage4j1_readiness_report: dict | None,
    operation_window_config: dict | None = None,
    scheduler_activation_snapshot: dict | None = None,
    lifecycle_activation_snapshot: dict | None = None,
    activation_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    market_window_snapshot: dict | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a read-only plan for the first controlled scheduled PAPER operation."""

    try:
        return _json_safe(
            _build_report(
                stage4j1_readiness_report=stage4j1_readiness_report,
                operation_window_config=operation_window_config,
                scheduler_activation_snapshot=scheduler_activation_snapshot,
                lifecycle_activation_snapshot=lifecycle_activation_snapshot,
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
        message = f"unexpected Stage 4J-2 operation plan failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_default_selected_strategy(),
                controlled_operation_scope=_controlled_operation_scope(None),
                operation_plan=_empty_operation_plan(),
                operation_window_checks=_default_operation_window_checks(),
                activation_snapshot_checks=_default_activation_snapshot_checks(),
                state_checks=_default_state_checks(),
                risk_checks=_default_risk_checks(),
                scheduler_checks=_default_scheduler_checks(),
                lifecycle_checks=_default_lifecycle_checks(),
                paper_broker_checks=_default_paper_broker_checks(),
                market_window_checks=_default_market_window_checks(),
                safety_checks=_safety_checks(
                    report_safety={},
                    payload_checks={},
                    snapshots=[],
                ),
                ready=False,
                blockers=[message],
                warnings=[],
                errors=[message],
            )
        )


def _build_report(
    *,
    stage4j1_readiness_report: dict | None,
    operation_window_config: dict | None,
    scheduler_activation_snapshot: dict | None,
    lifecycle_activation_snapshot: dict | None,
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
    generated_date = _date_string(generated_at)
    report = stage4j1_readiness_report if isinstance(stage4j1_readiness_report, dict) else None
    data = _mapping(report)
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4j1_readiness_report is None:
        blockers.append("Stage 4J-1 readiness report is missing")
    elif report is None:
        blockers.append("Stage 4J-1 readiness report must be a dict")
        errors.append("Stage 4J-1 readiness report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    selected_strategy_id = _selected_strategy_id(data)
    artifact_checks = _artifact_checks(stage4j1_readiness_report, selected_strategy_id)
    blockers.extend(_artifact_blockers(artifact_checks, data))
    blockers.extend(_stage4j1_safety_blockers(_mapping(data.get("safety_checks"))))

    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    blockers.extend(selected_blockers)

    window, window_checks, window_blockers, window_warnings = _operation_window(
        operation_window_config
    )
    blockers.extend(window_blockers)
    warnings.extend(window_warnings)

    activation_checks, activation_blockers, activation_warnings = _activation_snapshot_group_checks(
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
        report_safety=_mapping(data.get("safety_checks")),
        payload_checks=_mapping(data.get("payload_checks")),
        snapshots=[s for s in (
            scheduler_activation_snapshot,
            lifecycle_activation_snapshot,
            activation_snapshot,
            scheduler_snapshot,
            lifecycle_snapshot,
            paper_broker_snapshot,
        ) if isinstance(s, dict)],
    )
    blockers.extend(_safety_blockers(safety_checks))

    controlled_scope = _controlled_operation_scope(selected_strategy_id)
    operation_plan = _operation_plan(
        selected_strategy_id=selected_strategy_id,
        window=window,
        generated_date=generated_date,
        available=(
            artifact_checks["stage4j1_report_ready"] is True
            and selected_strategy["selected_strategy_id"] is not None
            and selected_strategy["paper_only"] is True
            and selected_strategy["one_strategy_only"] is True
        ),
    )
    blockers.extend(_operation_plan_blockers(operation_plan))

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    ready = not blocker_list and not error_list

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
        controlled_operation_scope=controlled_scope,
        operation_plan=operation_plan,
        operation_window_checks=window_checks,
        activation_snapshot_checks=activation_checks,
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


def _artifact_checks(report: dict | None, selected_strategy_id: str | None) -> dict[str, bool]:
    data = _mapping(report)
    readiness = _mapping(data.get("readiness_for_stage4j2"))
    return {
        "stage4j1_report_present": isinstance(report, dict),
        "stage4j1_report_ready": (
            data.get("stage4j1_controlled_paper_operation_readiness_report") is True
            and readiness.get("ready_to_build_controlled_paper_operation_plan") is True
            and data.get("success") is True
        ),
        "selected_strategy_present": isinstance(selected_strategy_id, str)
        and bool(selected_strategy_id),
    }


def _artifact_blockers(checks: dict[str, bool], data: dict[str, Any]) -> list[str]:
    labels = {
        "stage4j1_report_present": "Stage 4J-1 readiness report is missing",
        "stage4j1_report_ready": "Stage 4J-1 readiness report is not ready for Stage 4J-2",
        "selected_strategy_present": "selected strategy is missing from accepted Stage 4J-1 report",
    }
    blockers = [label for key, label in labels.items() if checks.get(key) is not True]
    if _as_list(data.get("errors")):
        blockers.append("Stage 4J-1 report contains errors")
    return blockers


def _stage4j1_safety_blockers(safety_checks: dict[str, Any]) -> list[str]:
    labels = {
        "no_live_trading": "Stage 4J-1 safety checks do not confirm live trading is disabled",
        "no_all_strategy_enablement": "Stage 4J-1 safety checks do not confirm all-strategy automation is disabled",
        "no_broker_submission_enabled": "Stage 4J-1 safety checks do not confirm broker submission is disabled",
        "no_market_data": "Stage 4J-1 safety checks do not confirm market data is disabled",
        "no_contract_qualification": "Stage 4J-1 safety checks do not confirm contract qualification is disabled",
        "no_order_submission": "Stage 4J-1 safety checks do not confirm order submission is disabled",
        "no_strategy_scan_execution": "Stage 4J-1 safety checks do not confirm strategy scan execution is disabled",
        "no_lifecycle_transition_execution": "Stage 4J-1 safety checks do not confirm lifecycle transition execution is disabled",
        "no_direct_scheduler_registration": "Stage 4J-1 safety checks do not confirm direct scheduler registration is disabled",
        "no_direct_lifecycle_execution": "Stage 4J-1 safety checks do not confirm direct lifecycle execution is disabled",
        "no_state_write": "Stage 4J-1 safety checks do not confirm state writes are disabled",
        "no_ledger_write": "Stage 4J-1 safety checks do not confirm ledger writes are disabled",
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
        blockers.append("Stage 4J-1 selected strategy must be paper_only true")
    if not one_strategy_only:
        blockers.append("Stage 4J-1 selected strategy must be one_strategy_only true")
    return (
        {
            "selected_strategy_id": selected_strategy_id,
            "paper_only": paper_only,
            "one_strategy_only": one_strategy_only,
        },
        blockers,
    )


def _operation_window(
    config: dict | None,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(config)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("operation window config missing; deterministic safe defaults are used for planning only")
    cadence = _normalize_cadence(data.get("cadence"))
    cadence_supported = cadence in ALLOWED_CADENCES
    if not cadence_supported:
        blockers.append("operation_window_config cadence is unsupported")
    if data.get("dry_run_only") is False:
        blockers.append("operation_window_config dry_run_only must not be false")
    window = {
        "operation_label": _safe_text(data.get("operation_label"), "stage4j2_controlled_paper_operation"),
        "timezone": _safe_text(data.get("timezone"), "America/New_York"),
        "first_operation_date": _safe_text(data.get("first_operation_date"), None),
        "operation_time": _safe_text(data.get("operation_time"), "09:45"),
        "cadence": cadence,
        "max_runtime_seconds": _safe_optional_int(data.get("max_runtime_seconds")),
        "dry_run_only": True,
        "operator_notes": _safe_text(data.get("operator_notes"), None),
    }
    return (
        window,
        {
            "operation_window_config_present": present,
            "cadence": cadence,
            "cadence_supported": cadence_supported,
            "dry_run_only": True,
        },
        blockers,
        warnings,
    )


def _controlled_operation_scope(selected_strategy_id: str | None) -> dict[str, Any]:
    return {
        "selected_strategy_id": selected_strategy_id,
        "operation_scope": "single_strategy_controlled_scheduled_paper_operation",
        "paper_only": True,
        "one_strategy_only": True,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "broker_submission_enabled": False,
        "strategy_scan_execution_enabled_now": False,
        "lifecycle_transition_execution_enabled_now": False,
        "market_data_enabled_now": False,
        "contract_qualification_enabled_now": False,
    }


def _operation_plan(
    *,
    selected_strategy_id: str | None,
    window: dict[str, Any],
    generated_date: str,
    available: bool,
) -> dict[str, Any]:
    operation_id = _operation_id(selected_strategy_id or "unknown_strategy", window["cadence"], generated_date)
    proposed_window = {
        "operation_id": operation_id,
        "operation_label": window["operation_label"],
        "selected_strategy_id": selected_strategy_id,
        "cadence": window["cadence"],
        "timezone": window["timezone"],
        "first_operation_date": window["first_operation_date"] or generated_date,
        "operation_time": window["operation_time"],
        "max_runtime_seconds": window["max_runtime_seconds"],
        "dry_run_only": True,
        "would_register_scheduler": False,
        "would_execute_operation": False,
        "would_submit_orders": False,
        "paper_only": True,
        "live_trading_enabled": False,
    }
    return {
        "available": available,
        "operation_id": operation_id,
        "operation_mode": "controlled_scheduled_paper_planning_only",
        "controlled_operation_scope": "single_strategy_controlled_scheduled_paper_operation",
        "proposed_operation_window": proposed_window,
        "proposed_pre_operation_gates": _proposed_pre_operation_gates(),
        "proposed_operation_flow": _proposed_operation_flow(selected_strategy_id, operation_id),
        "proposed_post_operation_checks": _proposed_post_operation_checks(selected_strategy_id),
        "disabled_components": list(DISABLED_COMPONENTS),
        "required_inputs_for_4J3": list(REQUIRED_INPUTS_FOR_4J3),
    }


def _proposed_pre_operation_gates() -> dict[str, bool]:
    return {
        "stage4j1_readiness_accepted": True,
        "selected_strategy_present": True,
        "one_strategy_only": True,
        "paper_only": True,
        "scheduler_activation_valid": True,
        "lifecycle_activation_valid": True,
        "activation_snapshot_clean": True,
        "state_snapshot_clean": True,
        "risk_controls_available": True,
        "scheduler_snapshot_clean": True,
        "lifecycle_snapshot_clean": True,
        "paper_broker_config_valid": True,
        "market_window_allowed": True,
        "broker_submission_disabled": True,
        "live_trading_disabled": True,
        "all_strategy_automation_disabled": True,
        "strategy_scan_execution_disabled_in_4J2": True,
        "lifecycle_transition_execution_disabled_in_4J2": True,
        "market_data_disabled_in_4J2": True,
        "contract_qualification_disabled_in_4J2": True,
    }


def _proposed_operation_flow(selected_strategy_id: str | None, operation_id: str) -> list[dict[str, Any]]:
    flow = []
    for index, (stage, component) in enumerate(FLOW_STAGES, start=1):
        flow.append(
            {
                "sequence_number": index,
                "stage": stage,
                "target_component": component,
                "payload": {
                    "operation_id": operation_id,
                    "selected_strategy_id": selected_strategy_id,
                    "stage": stage,
                    "preview_only": True,
                },
                "would_execute": False,
                "would_call_strategy": False,
                "would_fetch_market_data": False,
                "would_qualify_contracts": False,
                "would_create_intent": False,
                "would_create_ticket": False,
                "would_submit_order": False,
                "would_write_state": False,
                "would_write_ledger": False,
                "paper_only": True,
                "live_trading_enabled": False,
            }
        )
    return flow


def _proposed_post_operation_checks(selected_strategy_id: str | None) -> list[dict[str, Any]]:
    return [
        {
            "check": "dry_run_plan_summary_review",
            "selected_strategy_id": selected_strategy_id,
            "would_execute": False,
            "would_write_state": False,
            "would_write_ledger": False,
        },
        {
            "check": "post_operation_reconciliation_preview",
            "selected_strategy_id": selected_strategy_id,
            "would_execute": False,
            "would_write_state": False,
            "would_write_ledger": False,
        },
    ]


def _operation_plan_blockers(operation_plan: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if operation_plan.get("available") is not True:
        blockers.append("operation plan is not available")
    window = _mapping(operation_plan.get("proposed_operation_window"))
    if not window:
        blockers.append("proposed operation window is missing")
    for key in ("would_register_scheduler", "would_execute_operation", "would_submit_orders"):
        if window.get(key) is not False:
            blockers.append(f"proposed operation window {key} must be false")
    flow = _as_list(operation_plan.get("proposed_operation_flow"))
    if not flow:
        blockers.append("proposed operation flow is missing")
    expected_stages = [stage for stage, _component in FLOW_STAGES]
    actual_stages: list[str] = []
    for step in flow:
        if not isinstance(step, dict):
            blockers.append("proposed operation flow contains malformed step")
            continue
        actual_stages.append(step.get("stage") if isinstance(step.get("stage"), str) else "")
        payload = step.get("payload")
        if not isinstance(payload, dict):
            blockers.append("proposed operation flow step payload must be a dict")
        elif not _primitive_json_safe(payload):
            blockers.append("proposed operation flow payload contains non-primitive JSON-unsafe data")
        for key in (
            "would_execute",
            "would_call_strategy",
            "would_fetch_market_data",
            "would_qualify_contracts",
            "would_create_intent",
            "would_create_ticket",
            "would_submit_order",
            "would_write_state",
            "would_write_ledger",
        ):
            if step.get(key) is not False:
                blockers.append(f"proposed operation flow step {key} must be false")
        if step.get("paper_only") is not True:
            blockers.append("proposed operation flow step paper_only must be true")
        if step.get("live_trading_enabled") is not False:
            blockers.append("proposed operation flow step live_trading_enabled must be false")
    if actual_stages and actual_stages != expected_stages:
        blockers.append("proposed operation flow stages are not in the required deterministic order")
    return blockers


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
    for builder, snapshot in (
        (_scheduler_activation_snapshot_checks, scheduler_activation_snapshot),
        (_lifecycle_activation_snapshot_checks, lifecycle_activation_snapshot),
        (_activation_snapshot_checks, activation_snapshot),
    ):
        current_checks, current_blockers, current_warnings = builder(snapshot, selected_strategy_id)
        checks.update(current_checks)
        blockers.extend(current_blockers)
        warnings.extend(current_warnings)
    return checks, blockers, warnings


def _scheduler_activation_snapshot_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    return _activation_artifact_snapshot_checks(
        snapshot,
        selected_strategy_id,
        prefix="scheduler_activation",
        record_key="scheduler_activation_record",
        list_key="scheduler_activations",
        execution_key="strategy_scan_execution_enabled",
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
        execution_key="lifecycle_transition_execution_enabled",
    )


def _activation_artifact_snapshot_checks(
    snapshot: dict | None,
    selected_strategy_id: str | None,
    *,
    prefix: str,
    record_key: str,
    list_key: str,
    execution_key: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append(f"{prefix} snapshot missing; 4J-3 must verify activation artifact")
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
        for key in (
            "live_trading_enabled",
            "all_strategies_enabled",
            "broker_submission_enabled",
            execution_key,
        ):
            if record.get(key) is True:
                blockers.append(f"{prefix} snapshot {key} contradicts Stage 4J-2 safety")
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
        warnings.append("activation snapshot missing; 4J-3 must verify one-strategy activation state")
        return {"activation_snapshot_present": False, "activation_snapshot_matches": True}, blockers, warnings
    active_ids = _as_string_list(data.get("active_strategy_ids"))
    if len(active_ids) > 1:
        blockers.append("activation snapshot shows more than one active strategy")
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
                blockers.append(f"activation snapshot {key} contradicts Stage 4J-2 safety")
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
        warnings.append("state snapshot missing; 4J-3 must verify halt, reconciliation, intents, and positions")
    if active_halt:
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
        "kill_switch_available": data.get("kill_switch_available") is True if present else None,
        "hard_halt_available": data.get("hard_halt_available") is True if present else None,
        "daily_loss_limit_available": data.get("daily_loss_limit_available") is True if present else None,
        "max_position_limit_available": data.get("max_position_limit_available") is not False if present else None,
        "risk_bypass_enabled": data.get("risk_bypass_enabled") is True if present else False,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("risk snapshot missing; 4J-3 must verify kill switch, hard halt, and daily loss controls")
        return checks, blockers, warnings
    for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
        if checks[key] is not True:
            blockers.append(f"risk snapshot {key} must be true")
    if data.get("max_position_limit_available") is False:
        blockers.append("risk snapshot max_position_limit_available must not be false when supplied")
    if checks["risk_bypass_enabled"] is True:
        blockers.append("risk bypass is enabled")
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
        warnings.append("scheduler snapshot missing; 4J-3 must verify scheduler state")
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
        blockers.append("broad/all-strategy scheduler automation enabled")
    if matching_jobs and not selected_job_matches:
        blockers.append("selected strategy scheduler job does not match Stage 4J-2 safety constraints")
    if scan_enabled:
        blockers.append("scheduler snapshot strategy_scan_execution_enabled must be false")
    return (
        {
            "scheduler_snapshot_present": present,
            "scheduler_already_enabled": scheduler_enabled,
            "all_strategy_scheduler_enabled": all_strategy_enabled,
            "selected_strategy_job_matches": selected_job_matches,
            "strategy_scan_execution_enabled": scan_enabled,
        },
        blockers,
        warnings,
    )


def _lifecycle_checks(
    snapshot: dict | None, selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; 4J-3 must verify lifecycle state")
    broad_enabled = any(
        data.get(key) is True
        for key in ("all_strategy_lifecycle_enabled", "all_strategies_enabled", "broad_lifecycle_enabled")
    )
    transition_enabled = data.get("lifecycle_transition_execution_enabled") is True
    lifecycle_enabled = data.get("lifecycle_automation_enabled") is True or broad_enabled
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
        blockers.append("broad/all-strategy lifecycle automation enabled")
    if transition_enabled:
        blockers.append("lifecycle transition execution is enabled")
    return (
        {
            "lifecycle_snapshot_present": present,
            "lifecycle_already_enabled": lifecycle_enabled,
            "lifecycle_matches_selected_strategy": matches,
            "lifecycle_transition_execution_enabled": transition_enabled,
        },
        blockers,
        warnings,
    )


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
        warnings.append("paper broker snapshot missing; 4J-3 must verify PAPER broker config")
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
        blockers.append("market window snapshot explicitly disallows scheduling a paper run")
    if market_open is False:
        warnings.append("market is currently closed; planning may continue but 4J-3 must verify timing")
    if is_trading_day is False:
        warnings.append("snapshot is not a trading day; planning may continue but 4J-3 must verify timing")
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
    *, report_safety: dict[str, Any], payload_checks: dict[str, Any], snapshots: list[dict[str, Any]]
) -> dict[str, bool]:
    return {
        "no_live_trading": report_safety.get("no_live_trading") is not False and _none_true("live_trading_enabled", *snapshots),
        "no_all_strategy_enablement": report_safety.get("no_all_strategy_enablement") is not False and _none_true("all_strategies_enabled", *snapshots),
        "no_broker_submission_enabled": report_safety.get("no_broker_submission_enabled") is not False and _none_true("broker_submission_enabled", *snapshots),
        "no_market_data": report_safety.get("no_market_data") is not False and payload_checks.get("market_data_disabled") is not False and _none_true("market_data_enabled", *snapshots),
        "no_contract_qualification": report_safety.get("no_contract_qualification") is not False and payload_checks.get("contract_qualification_disabled") is not False and _none_true("contract_qualification_enabled", *snapshots),
        "no_order_submission": report_safety.get("no_order_submission") is not False and _none_true("order_submission_enabled", *snapshots),
        "no_strategy_scan_execution": report_safety.get("no_strategy_scan_execution") is not False and payload_checks.get("strategy_scan_execution_disabled") is not False and _none_true("strategy_scan_execution_enabled", *snapshots),
        "no_lifecycle_transition_execution": report_safety.get("no_lifecycle_transition_execution") is not False and payload_checks.get("lifecycle_transition_execution_disabled") is not False and _none_true("lifecycle_transition_execution_enabled", *snapshots),
        "no_direct_scheduler_registration": report_safety.get("no_direct_scheduler_registration") is not False,
        "no_direct_lifecycle_execution": report_safety.get("no_direct_lifecycle_execution") is not False,
        "no_state_write": report_safety.get("no_state_write") is not False,
        "no_ledger_write": report_safety.get("no_ledger_write") is not False,
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
        "no_lifecycle_transition_execution": "lifecycle transition execution safety flag is enabled",
        "no_direct_scheduler_registration": "direct scheduler registration safety flag is enabled",
        "no_direct_lifecycle_execution": "direct lifecycle execution safety flag is enabled",
        "no_state_write": "state write safety flag is enabled",
        "no_ledger_write": "ledger write safety flag is enabled",
    }
    return [label for key, label in labels.items() if checks.get(key) is not True]


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    controlled_operation_scope: dict[str, Any],
    operation_plan: dict[str, Any],
    operation_window_checks: dict[str, Any],
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
        "stage4j2_controlled_paper_operation_plan_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "controlled_operation_scope": controlled_operation_scope,
        "operation_plan": operation_plan,
        "operation_window_checks": operation_window_checks,
        "activation_snapshot_checks": activation_snapshot_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4j3": {
            "ready_to_build_controlled_paper_operation_dry_run": ready,
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
    selected = _mapping(report.get("selected_strategy"))
    value = selected.get("selected_strategy_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
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
    return _first_present(job.get("selected_strategy_id"), job.get("strategy_id"), default=None)


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


def _first_present(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _operation_id(strategy_id: str, cadence: str, generated_date: str) -> str:
    return "_".join([_id_part(strategy_id), _id_part(cadence or "once"), _id_part(generated_date)])


def _id_part(value: str) -> str:
    cleaned = []
    for char in str(value).strip().lower():
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


def _primitive_json_safe(value: Any) -> bool:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return not (isinstance(value, float) and not math.isfinite(value))
    if isinstance(value, list):
        return all(_primitive_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _primitive_json_safe(item) for key, item in value.items())
    if isinstance(value, (datetime, date, tuple, Decimal, Path)) or callable(value):
        return False
    return False


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


def _empty_operation_plan() -> dict[str, Any]:
    return {
        "available": False,
        "operation_id": None,
        "operation_mode": "controlled_scheduled_paper_planning_only",
        "proposed_operation_window": {},
        "proposed_pre_operation_gates": {},
        "proposed_operation_flow": [],
        "proposed_post_operation_checks": [],
        "disabled_components": list(DISABLED_COMPONENTS),
        "required_inputs_for_4J3": list(REQUIRED_INPUTS_FOR_4J3),
    }


def _default_artifact_checks() -> dict[str, bool]:
    return {
        "stage4j1_report_present": False,
        "stage4j1_report_ready": False,
        "selected_strategy_present": False,
    }


def _default_selected_strategy() -> dict[str, Any]:
    return {"selected_strategy_id": None, "paper_only": False, "one_strategy_only": False}


def _default_operation_window_checks() -> dict[str, Any]:
    return {
        "operation_window_config_present": False,
        "cadence": "once",
        "cadence_supported": True,
        "dry_run_only": True,
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
