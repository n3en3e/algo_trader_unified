"""Pure Stage 4J-4 controlled scheduled PAPER operation execution-gate report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from pathlib import Path
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
PAPER_IBKR_PORTS = {4004}
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding to 4J-5"
)
REQUIRED_OPERATOR_ACKNOWLEDGEMENTS = [
    "I understand this may allow building the first selected-strategy PAPER operation executor in the next phase.",
    "I understand this does not enable live trading.",
    "I understand this does not enable all strategies.",
    "I understand broker order submission remains separately gated.",
    "I understand market data and contract qualification remain separately gated.",
    "I verified state, risk, scheduler, lifecycle, paper broker, and market window snapshots.",
    "I understand this gate phase does not run strategy code or place orders.",
]
ORDERED_NEXT_STEPS = [
    "Build Stage 4J-5 controlled scheduled PAPER operation executor.",
    "Before any real selected-strategy operation, re-check activation artifacts, scheduler/lifecycle state, risk controls, paper broker config, market window, and state reconciliation.",
    "Keep live trading disabled.",
    "Keep all other strategies disabled.",
    "Keep broker submission separately gated until the explicit broker-submission phase.",
    "Keep market data and contract qualification separately gated until their explicit phase.",
]
DO_NOT_DO_YET = [
    "Do not enable live trading.",
    "Do not enable all strategies.",
    "Do not place orders now.",
    "Do not enable broker submission now.",
    "Do not enable market data now.",
    "Do not enable contract qualification now.",
    "Do not create intents or tickets now.",
    "Do not write state or ledger now.",
    "Do not change strategy thresholds.",
    "Do not change position sizing.",
    "Do not bypass risk controls.",
]
DISABLED_COMPONENTS = [
    "live_trading",
    "all_strategy_automation",
    "broker_submission",
    "order_submission",
    "market_data_fetch",
    "contract_qualification",
    "intent_creation",
    "ticket_creation",
    "state_mutation",
    "ledger_write",
    "direct_scheduler_registration",
    "direct_lifecycle_execution",
]
REQUIRED_INPUTS_FOR_4J5 = [
    "accepted_stage4j4_execution_gate_report",
    "accepted_stage4j3_dry_run_report",
    "fresh_scheduler_activation_snapshot",
    "fresh_lifecycle_activation_snapshot",
    "fresh_activation_snapshot",
    "fresh_state_snapshot",
    "fresh_risk_snapshot",
    "fresh_scheduler_snapshot",
    "fresh_lifecycle_snapshot",
    "fresh_paper_broker_snapshot",
    "fresh_market_window_snapshot",
    "exact_operator_acknowledgements",
]
TRACE_FALSE_FLAGS = (
    "would_execute",
    "would_call_strategy",
    "would_fetch_market_data",
    "would_qualify_contracts",
    "would_create_intent",
    "would_create_ticket",
    "would_submit_order",
    "would_write_state",
    "would_write_ledger",
)
SAFETY_KEYS = (
    "no_live_trading",
    "no_all_strategy_enablement",
    "no_broker_submission_enabled",
    "no_market_data",
    "no_contract_qualification",
    "no_order_submission",
    "no_strategy_scan_execution",
    "no_lifecycle_transition_execution",
    "no_state_write",
    "no_ledger_write",
)


def build_stage4j4_controlled_paper_operation_execution_gate_report(
    *,
    stage4j3_dry_run_report: dict | None,
    scheduler_activation_snapshot: dict | None = None,
    lifecycle_activation_snapshot: dict | None = None,
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
    """Build a read-only execution gate report from an accepted Stage 4J-3 dry run."""

    operator_acknowledgements = [] if operator_acknowledgements is None else operator_acknowledgements
    try:
        return _json_safe(
            _build_report(
                stage4j3_dry_run_report=stage4j3_dry_run_report,
                scheduler_activation_snapshot=scheduler_activation_snapshot,
                lifecycle_activation_snapshot=lifecycle_activation_snapshot,
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
        message = f"unexpected Stage 4J-4 execution-gate failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_default_selected_strategy(),
                operation=_default_operation(),
                acknowledgement_checks=_acknowledgement_checks(operator_acknowledgements),
                execution_gate=_execution_gate(None, None, False, False, False, False),
                proposed_execution_permissions_for_4J5=_permissions(None, None, False),
                proposed_pre_execution_checks={},
                proposed_execution_trace_requirements=_trace_requirements(),
                proposed_post_execution_checks=_post_execution_checks(),
                disabled_components=list(DISABLED_COMPONENTS),
                required_inputs_for_4J5=list(REQUIRED_INPUTS_FOR_4J5),
                dry_run_trace_checks=_default_trace_checks(),
                activation_snapshot_checks=_default_activation_snapshot_checks(),
                state_checks=_default_state_checks(),
                risk_checks=_default_risk_checks(),
                scheduler_checks=_default_scheduler_checks(),
                lifecycle_checks=_default_lifecycle_checks(),
                paper_broker_checks=_default_paper_broker_checks(),
                market_window_checks=_default_market_window_checks(),
                safety_checks=_default_safety_checks(),
                ready=False,
                blockers=[message],
                warnings=[],
                errors=[message],
            )
        )


def _build_report(
    *,
    stage4j3_dry_run_report: dict | None,
    scheduler_activation_snapshot: dict | None,
    lifecycle_activation_snapshot: dict | None,
    activation_snapshot: dict | None,
    state_snapshot: dict | None,
    risk_snapshot: dict | None,
    scheduler_snapshot: dict | None,
    lifecycle_snapshot: dict | None,
    paper_broker_snapshot: dict | None,
    market_window_snapshot: dict | None,
    operator_acknowledgements: list[Any],
    now_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    generated_at = _generated_at(now_provider)
    report = stage4j3_dry_run_report if isinstance(stage4j3_dry_run_report, dict) else None
    data = _mapping(report)
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4j3_dry_run_report is None:
        blockers.append("Stage 4J-3 dry-run report is missing")
    elif report is None:
        blockers.append("Stage 4J-3 dry-run report must be a dict")
        errors.append("Stage 4J-3 dry-run report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    selected_strategy_id = _selected_strategy_id(data)
    operation_id = _operation_id(data)
    artifact_checks = _artifact_checks(stage4j3_dry_run_report, data, selected_strategy_id, operation_id)
    blockers.extend(_artifact_blockers(artifact_checks, data))

    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    operation, operation_blockers = _operation_checks(data, operation_id)
    blockers.extend(selected_blockers + operation_blockers)

    trace = data.get("dry_run_trace")
    dry_run_trace_checks, trace_blockers = _dry_run_trace_checks(
        trace,
        _mapping(data.get("dry_run_trace_checks")),
    )
    blockers.extend(trace_blockers)

    acknowledgement_checks = _acknowledgement_checks(operator_acknowledgements)
    if acknowledgement_checks["exact_match"] is not True:
        blockers.append("required operator acknowledgements are missing or not exact")

    activation_checks, activation_blockers, activation_warnings = _activation_snapshot_group_checks(
        scheduler_activation_snapshot=scheduler_activation_snapshot,
        lifecycle_activation_snapshot=lifecycle_activation_snapshot,
        activation_snapshot=activation_snapshot,
        selected_strategy_id=selected_strategy_id,
    )
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
    snapshot_blockers = (
        activation_blockers
        + state_blockers
        + risk_blockers
        + scheduler_blockers
        + lifecycle_blockers
        + broker_blockers
        + market_blockers
    )
    blockers.extend(snapshot_blockers)
    warnings.extend(
        activation_warnings
        + state_warnings
        + risk_warnings
        + scheduler_warnings
        + lifecycle_warnings
        + broker_warnings
        + market_warnings
    )

    safety_checks = _safety_checks(
        _mapping(data.get("safety_checks")),
        [
            snapshot
            for snapshot in (
                scheduler_activation_snapshot,
                lifecycle_activation_snapshot,
                activation_snapshot,
                scheduler_snapshot,
                lifecycle_snapshot,
                paper_broker_snapshot,
            )
            if isinstance(snapshot, dict)
        ],
    )
    blockers.extend(_safety_blockers(safety_checks))

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    ready = not blocker_list and not error_list
    snapshots_clean = not snapshot_blockers

    permissions = _permissions(selected_strategy_id, operation_id, ready)
    execution_gate = _execution_gate(
        selected_strategy_id,
        operation_id,
        artifact_checks["stage4j3_report_ready"],
        acknowledgement_checks["exact_match"],
        snapshots_clean,
        ready,
    )
    pre_execution_checks = _pre_execution_checks(
        artifact_checks=artifact_checks,
        trace_checks=dry_run_trace_checks,
        acknowledgement_checks=acknowledgement_checks,
        snapshots_clean=snapshots_clean,
        state_checks=state_checks,
        risk_checks=risk_checks,
        scheduler_checks=scheduler_checks,
        lifecycle_checks=lifecycle_checks,
        paper_broker_checks=paper_broker_checks,
        market_window_checks=market_window_checks,
        safety_checks=safety_checks,
    )

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
        operation=operation,
        acknowledgement_checks=acknowledgement_checks,
        execution_gate=execution_gate,
        proposed_execution_permissions_for_4J5=permissions,
        proposed_pre_execution_checks=pre_execution_checks,
        proposed_execution_trace_requirements=_trace_requirements(),
        proposed_post_execution_checks=_post_execution_checks(),
        disabled_components=list(DISABLED_COMPONENTS),
        required_inputs_for_4J5=list(REQUIRED_INPUTS_FOR_4J5),
        dry_run_trace_checks=dry_run_trace_checks,
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


def _artifact_checks(
    report: Any, data: dict[str, Any], selected_strategy_id: str | None, operation_id: str | None
) -> dict[str, bool]:
    readiness = _mapping(data.get("readiness_for_stage4j4"))
    trace = data.get("dry_run_trace")
    trace_checks = _mapping(data.get("dry_run_trace_checks"))
    trace_clean = _required_trace_report_checks_pass(trace_checks) and _trace_items_clean(trace)
    return {
        "stage4j3_report_present": isinstance(report, dict),
        "stage4j3_report_ready": (
            data.get("stage4j3_controlled_paper_operation_dry_run_report") is True
            and readiness.get("ready_to_build_controlled_paper_operation_execution_gate") is True
            and data.get("success") is True
        ),
        "selected_strategy_present": isinstance(selected_strategy_id, str) and bool(selected_strategy_id),
        "operation_id_present": isinstance(operation_id, str) and bool(operation_id),
        "dry_run_trace_present": isinstance(trace, list) and bool(trace),
        "dry_run_trace_clean": trace_clean,
    }


def _artifact_blockers(checks: dict[str, bool], data: dict[str, Any]) -> list[str]:
    labels = {
        "stage4j3_report_present": "Stage 4J-3 dry-run report is missing",
        "stage4j3_report_ready": "Stage 4J-3 dry-run report is not ready for Stage 4J-4",
        "selected_strategy_present": "selected strategy is missing from accepted Stage 4J-3 report",
        "operation_id_present": "operation_id is missing from accepted Stage 4J-3 report",
        "dry_run_trace_present": "dry_run_trace is missing from Stage 4J-3 report",
        "dry_run_trace_clean": "dry_run_trace is not clean",
    }
    blockers = [label for key, label in labels.items() if checks.get(key) is not True]
    if _as_list(data.get("errors")):
        blockers.append("Stage 4J-3 report contains errors")
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
        blockers.append("Stage 4J-3 selected strategy must be paper_only true")
    if not one_strategy_only:
        blockers.append("Stage 4J-3 selected strategy must be one_strategy_only true")
    return (
        {
            "selected_strategy_id": selected_strategy_id,
            "paper_only": paper_only,
            "one_strategy_only": one_strategy_only,
        },
        blockers,
    )


def _operation_checks(report: dict[str, Any], operation_id: str | None) -> tuple[dict[str, Any], list[str]]:
    operation = _mapping(report.get("operation"))
    blockers: list[str] = []
    if not isinstance(operation_id, str) or not operation_id:
        blockers.append("operation_id is missing from Stage 4J-3 operation")
    if operation.get("live_trading_enabled") is True:
        blockers.append("Stage 4J-3 operation shows live trading enabled")
    if operation.get("broker_submission_enabled") is True:
        blockers.append("Stage 4J-3 operation shows broker submission enabled")
    return (
        {
            "operation_id": operation_id,
            "operation_scope": operation.get("operation_scope"),
            "paper_only": True,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
        },
        blockers,
    )


def _dry_run_trace_checks(
    trace_value: Any, source_checks: dict[str, Any]
) -> tuple[dict[str, bool], list[str]]:
    trace = trace_value if isinstance(trace_value, list) else []
    statuses_ok = bool(trace)
    payloads_json_safe = bool(trace)
    input_payloads_are_dicts = bool(trace)
    simulated_results_are_dicts = bool(trace)
    flags_ok = {key: bool(trace) for key in TRACE_FALSE_FLAGS}
    blockers: list[str] = []
    if not isinstance(trace_value, list) or not trace:
        blockers.append("dry_run_trace must be a non-empty list")
    for item in trace:
        if not isinstance(item, dict):
            blockers.append("dry_run_trace contains non-dict item")
            statuses_ok = False
            payloads_json_safe = False
            input_payloads_are_dicts = False
            simulated_results_are_dicts = False
            for key in TRACE_FALSE_FLAGS:
                flags_ok[key] = False
            continue
        if item.get("status") != "simulated":
            statuses_ok = False
            blockers.append("every dry_run_trace item status must be simulated")
        payload = item.get("input_payload", None)
        result = item.get("simulated_result", None)
        if not isinstance(payload, dict):
            input_payloads_are_dicts = False
            blockers.append("every dry_run_trace item input_payload must be a native dict")
        elif not _primitive_json_safe(payload):
            payloads_json_safe = False
            blockers.append("dry_run_trace input_payload contains non-primitive JSON-unsafe data")
        if not isinstance(result, dict):
            simulated_results_are_dicts = False
            blockers.append("every dry_run_trace item simulated_result must be a native dict")
        elif not _primitive_json_safe(result):
            payloads_json_safe = False
            blockers.append("dry_run_trace simulated_result contains non-primitive JSON-unsafe data")
        for key in TRACE_FALSE_FLAGS:
            if item.get(key) is not False:
                flags_ok[key] = False
                blockers.append(f"dry_run_trace item {key} must be strict boolean false")

    checks = {
        "trace_available": bool(trace),
        "trace_order_matches_plan": source_checks.get("trace_order_matches_plan") is True,
        "all_trace_items_simulated": statuses_ok,
        "no_strategy_call": flags_ok["would_call_strategy"],
        "no_market_data": flags_ok["would_fetch_market_data"],
        "no_contract_qualification": flags_ok["would_qualify_contracts"],
        "no_intent_created": flags_ok["would_create_intent"],
        "no_ticket_created": flags_ok["would_create_ticket"],
        "no_broker_submission": flags_ok["would_submit_order"],
        "no_state_write": flags_ok["would_write_state"],
        "no_ledger_write": flags_ok["would_write_ledger"],
        "payloads_json_safe": payloads_json_safe,
        "input_payloads_are_dicts": input_payloads_are_dicts,
        "simulated_results_are_dicts": simulated_results_are_dicts,
    }
    for key in (
        "trace_available",
        "trace_order_matches_plan",
        "all_trace_items_simulated",
        "no_strategy_call",
        "no_market_data",
        "no_contract_qualification",
        "no_intent_created",
        "no_ticket_created",
        "no_broker_submission",
        "no_state_write",
        "no_ledger_write",
        "payloads_json_safe",
        "input_payloads_are_dicts",
        "simulated_results_are_dicts",
    ):
        if checks[key] is not True:
            blockers.append(f"dry_run_trace check {key} must be true")
    return checks, _dedupe(blockers)


def _required_trace_report_checks_pass(checks: dict[str, Any]) -> bool:
    return all(checks.get(key) is True for key in _default_trace_checks())


def _trace_items_clean(trace_value: Any) -> bool:
    checks, blockers = _dry_run_trace_checks(trace_value, {"trace_order_matches_plan": True})
    return not blockers and all(value is True for value in checks.values())


def _acknowledgement_checks(values: list[Any]) -> dict[str, Any]:
    provided = [item.strip() for item in values if isinstance(item, str)]
    missing = [item for item in REQUIRED_OPERATOR_ACKNOWLEDGEMENTS if item not in provided]
    return {
        "required": list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS),
        "provided": provided,
        "missing": missing,
        "exact_match": not missing,
    }


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
        warnings.append(f"{prefix} snapshot missing; 4J-5 must verify activation artifact")
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
                blockers.append(f"{prefix} snapshot {key} contradicts Stage 4J-4 safety")
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
        warnings.append("activation snapshot missing; 4J-5 must verify one-strategy activation state")
        return {"activation_snapshot_present": False, "activation_snapshot_matches": True}, blockers, warnings
    active_ids = _as_string_list(data.get("active_strategy_ids"))
    if len(active_ids) > 1:
        blockers.append("activation snapshot shows more than one active strategy")
    elif active_ids and active_ids != [selected_strategy_id]:
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
                blockers.append(f"activation snapshot {key} contradicts Stage 4J-4 safety")
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
        "state_snapshot_clean": not active_halt
        and unresolved_count == 0
        and (active_intents_count == 0 or active_intents_safe),
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("state snapshot missing; 4J-5 must verify halt, reconciliation, intents, and positions")
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
        warnings.append("risk snapshot missing; 4J-5 must verify kill switch, hard halt, and daily loss controls")
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
        warnings.append("scheduler snapshot missing; 4J-5 must verify scheduler state")
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
        blockers.append("selected strategy scheduler job does not match Stage 4J-4 safety constraints")
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
        warnings.append("lifecycle snapshot missing; 4J-5 must verify lifecycle state")
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
        warnings.append("paper broker snapshot missing; 4J-5 must verify PAPER broker config")
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
        blockers.append("market window snapshot explicitly disallows execution-gate validation")
    if market_open is False:
        warnings.append("market is currently closed; execution-gate validation may continue but 4J-5 must verify timing")
    if is_trading_day is False:
        warnings.append("snapshot is not a trading day; execution-gate validation may continue but 4J-5 must verify timing")
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


def _safety_checks(report_safety: dict[str, Any], snapshots: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "no_live_trading": report_safety.get("no_live_trading") is True and _none_true("live_trading_enabled", *snapshots),
        "no_all_strategy_enablement": report_safety.get("no_all_strategy_enablement") is True and _none_true("all_strategies_enabled", *snapshots),
        "no_broker_submission_enabled": report_safety.get("no_broker_submission_enabled") is True and _none_true("broker_submission_enabled", *snapshots),
        "no_market_data": report_safety.get("no_market_data") is True and _none_true("market_data_enabled", *snapshots),
        "no_contract_qualification": report_safety.get("no_contract_qualification") is True and _none_true("contract_qualification_enabled", *snapshots),
        "no_order_submission": report_safety.get("no_order_submission") is True and _none_true("order_submission_enabled", *snapshots),
        "no_strategy_scan_execution": report_safety.get("no_strategy_scan_execution") is True and _none_true("strategy_scan_execution_enabled", *snapshots),
        "no_lifecycle_transition_execution": report_safety.get("no_lifecycle_transition_execution") is True and _none_true("lifecycle_transition_execution_enabled", *snapshots),
        "no_state_write": report_safety.get("no_state_write") is True,
        "no_ledger_write": report_safety.get("no_ledger_write") is True,
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
        "no_state_write": "state write safety flag is enabled",
        "no_ledger_write": "ledger write safety flag is enabled",
    }
    return [label for key, label in labels.items() if checks.get(key) is not True]


def _permissions(selected_strategy_id: str | None, operation_id: str | None, ready: bool) -> dict[str, Any]:
    return {
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "permission_scope": "single_strategy_controlled_paper_operation_executor",
        "paper_only": True,
        "one_strategy_only": True,
        "may_build_executor_next_phase": ready,
        "may_call_strategy_next_phase": ready,
        "may_fetch_market_data_next_phase": False,
        "may_qualify_contracts_next_phase": False,
        "may_create_intent_next_phase": False,
        "may_create_ticket_next_phase": False,
        "may_submit_order_next_phase": False,
        "may_write_state_next_phase": False,
        "may_write_ledger_next_phase": False,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "broker_submission_enabled": False,
    }


def _execution_gate(
    selected_strategy_id: str | None,
    operation_id: str | None,
    dry_run_accepted: bool,
    acknowledgements_ok: bool,
    snapshots_clean: bool,
    ready: bool,
) -> dict[str, Any]:
    return {
        "available": ready,
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "gate_scope": "single_strategy_controlled_paper_operation",
        "dry_run_accepted": dry_run_accepted,
        "acknowledgements_ok": acknowledgements_ok,
        "snapshots_clean": snapshots_clean,
        "broker_submission_still_disabled": True,
        "live_trading_still_disabled": True,
        "all_strategies_still_disabled": True,
        "market_data_still_disabled": True,
        "contract_qualification_still_disabled": True,
        "state_ledger_writes_still_disabled": True,
        "ready_for_4J5": ready,
    }


def _pre_execution_checks(
    *,
    artifact_checks: dict[str, Any],
    trace_checks: dict[str, Any],
    acknowledgement_checks: dict[str, Any],
    snapshots_clean: bool,
    state_checks: dict[str, Any],
    risk_checks: dict[str, Any],
    scheduler_checks: dict[str, Any],
    lifecycle_checks: dict[str, Any],
    paper_broker_checks: dict[str, Any],
    market_window_checks: dict[str, Any],
    safety_checks: dict[str, Any],
) -> dict[str, bool]:
    return {
        "stage4j3_dry_run_accepted": artifact_checks.get("stage4j3_report_ready") is True,
        "selected_strategy_present": artifact_checks.get("selected_strategy_present") is True,
        "operation_id_present": artifact_checks.get("operation_id_present") is True,
        "dry_run_trace_clean": trace_checks.get("trace_available") is True
        and all(trace_checks.get(key) is True for key in trace_checks),
        "dry_run_trace_payloads_native_dicts": trace_checks.get("input_payloads_are_dicts") is True,
        "dry_run_trace_results_native_dicts": trace_checks.get("simulated_results_are_dicts") is True,
        "dry_run_trace_payloads_json_safe": trace_checks.get("payloads_json_safe") is True,
        "exact_acknowledgements": acknowledgement_checks.get("exact_match") is True,
        "activation_artifacts_clean": snapshots_clean,
        "state_snapshot_clean": state_checks.get("state_snapshot_clean") is not False,
        "risk_controls_available": risk_checks.get("risk_snapshot_present") is not True
        or (
            risk_checks.get("kill_switch_available") is True
            and risk_checks.get("hard_halt_available") is True
            and risk_checks.get("daily_loss_limit_available") is True
        ),
        "scheduler_snapshot_clean": scheduler_checks.get("all_strategy_scheduler_enabled") is not True
        and scheduler_checks.get("strategy_scan_execution_enabled") is not True
        and scheduler_checks.get("selected_strategy_job_matches") is True,
        "lifecycle_snapshot_clean": lifecycle_checks.get("lifecycle_transition_execution_enabled") is not True
        and lifecycle_checks.get("lifecycle_matches_selected_strategy") is True,
        "paper_broker_config_valid": paper_broker_checks.get("paper_config_valid") is True,
        "market_window_allowed": market_window_checks.get("allowed_to_schedule_paper_run") is not False,
        "broker_submission_disabled": safety_checks.get("no_broker_submission_enabled") is True,
        "live_trading_disabled": safety_checks.get("no_live_trading") is True,
        "all_strategy_automation_disabled": safety_checks.get("no_all_strategy_enablement") is True,
        "market_data_disabled": safety_checks.get("no_market_data") is True,
        "contract_qualification_disabled": safety_checks.get("no_contract_qualification") is True,
        "state_ledger_writes_disabled": safety_checks.get("no_state_write") is True
        and safety_checks.get("no_ledger_write") is True,
    }


def _trace_requirements() -> dict[str, bool]:
    return {
        "executor_must_be_single_strategy_only": True,
        "executor_must_be_paper_only": True,
        "executor_must_not_submit_orders": True,
        "executor_must_not_write_state": True,
        "executor_must_not_write_ledger": True,
        "executor_must_not_fetch_market_data_until_explicit_gate": True,
        "executor_must_not_qualify_contracts_until_explicit_gate": True,
        "executor_must_not_create_intents_until_explicit_gate": True,
        "executor_must_not_create_tickets_until_explicit_gate": True,
        "executor_must_emit_report_only": True,
        "executor_must_preserve_operation_id": True,
    }


def _post_execution_checks() -> list[dict[str, Any]]:
    return [
        {"check": "executor_report_review", "required": True, "write_state": False, "write_ledger": False},
        {"check": "confirm_no_broker_submission", "required": True, "write_state": False, "write_ledger": False},
        {"check": "confirm_no_live_trading", "required": True, "write_state": False, "write_ledger": False},
    ]


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    operation: dict[str, Any],
    acknowledgement_checks: dict[str, Any],
    execution_gate: dict[str, Any],
    proposed_execution_permissions_for_4J5: dict[str, Any],
    proposed_pre_execution_checks: dict[str, Any],
    proposed_execution_trace_requirements: dict[str, Any],
    proposed_post_execution_checks: list[dict[str, Any]],
    disabled_components: list[str],
    required_inputs_for_4J5: list[str],
    dry_run_trace_checks: dict[str, Any],
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
        "stage4j4_controlled_paper_operation_execution_gate_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "operation": operation,
        "acknowledgement_checks": acknowledgement_checks,
        "execution_gate": execution_gate,
        "proposed_execution_permissions_for_4J5": proposed_execution_permissions_for_4J5,
        "proposed_pre_execution_checks": proposed_pre_execution_checks,
        "proposed_execution_trace_requirements": proposed_execution_trace_requirements,
        "proposed_post_execution_checks": proposed_post_execution_checks,
        "disabled_components": disabled_components,
        "required_inputs_for_4J5": required_inputs_for_4J5,
        "dry_run_trace_checks": dry_run_trace_checks,
        "activation_snapshot_checks": activation_snapshot_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4j5": {
            "ready_to_build_controlled_paper_operation_executor": ready,
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


def _operation_id(report: dict[str, Any]) -> str | None:
    operation = _mapping(report.get("operation"))
    value = operation.get("operation_id")
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


def _primitive_json_safe(value: Any) -> bool:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return not (isinstance(value, float) and not math.isfinite(value))
    if isinstance(value, list):
        return all(_primitive_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _primitive_json_safe(item) for key, item in value.items())
    if isinstance(value, (datetime, date, tuple, Decimal, Path, bytes)) or callable(value):
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


def _default_artifact_checks() -> dict[str, bool]:
    return {
        "stage4j3_report_present": False,
        "stage4j3_report_ready": False,
        "selected_strategy_present": False,
        "operation_id_present": False,
        "dry_run_trace_present": False,
        "dry_run_trace_clean": False,
    }


def _default_selected_strategy() -> dict[str, Any]:
    return {"selected_strategy_id": None, "paper_only": False, "one_strategy_only": False}


def _default_operation() -> dict[str, Any]:
    return {
        "operation_id": None,
        "operation_scope": None,
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }


def _default_trace_checks() -> dict[str, bool]:
    return {
        "trace_available": False,
        "trace_order_matches_plan": False,
        "all_trace_items_simulated": False,
        "no_strategy_call": True,
        "no_market_data": True,
        "no_contract_qualification": True,
        "no_intent_created": True,
        "no_ticket_created": True,
        "no_broker_submission": True,
        "no_state_write": True,
        "no_ledger_write": True,
        "payloads_json_safe": True,
        "input_payloads_are_dicts": True,
        "simulated_results_are_dicts": True,
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
        "state_snapshot_clean": True,
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


def _default_safety_checks() -> dict[str, bool]:
    return {key: False for key in SAFETY_KEYS}
