"""Pure Stage 4J-5 controlled scheduled PAPER operation executor report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from pathlib import Path
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
PAPER_IBKR_PORTS = {4004}
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding"
)
ALLOWED_RUNNER_STATUSES = {"completed_report_only", "skipped_no_signal", "blocked_by_gate"}
UNSAFE_RUNNER_FALSE_FLAGS = (
    "market_data_requested",
    "contracts_qualified",
    "intents_created",
    "tickets_created",
    "orders_submitted",
    "state_written",
    "ledger_written",
    "live_trading_enabled",
    "all_strategies_enabled",
    "broker_submission_enabled",
)
ORDERED_NEXT_STEPS = [
    "Build Stage 4J-6 controlled scheduled PAPER operation acceptance report.",
    "Review runner output and confirm no market data, contract qualification, order, state, or ledger side effects occurred.",
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


def build_stage4j5_controlled_paper_operation_executor_report(
    *,
    stage4j4_execution_gate_report: dict | None,
    selected_strategy_operation_runner,
    scheduler_activation_snapshot: dict | None = None,
    lifecycle_activation_snapshot: dict | None = None,
    activation_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    market_window_snapshot: dict | None = None,
    allow_controlled_paper_operation_execution: bool = False,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a controlled executor report and optionally call the injected runner."""

    try:
        return _json_safe(
            _build_report(
                stage4j4_execution_gate_report=stage4j4_execution_gate_report,
                selected_strategy_operation_runner=selected_strategy_operation_runner,
                scheduler_activation_snapshot=scheduler_activation_snapshot,
                lifecycle_activation_snapshot=lifecycle_activation_snapshot,
                activation_snapshot=activation_snapshot,
                state_snapshot=state_snapshot,
                risk_snapshot=risk_snapshot,
                scheduler_snapshot=scheduler_snapshot,
                lifecycle_snapshot=lifecycle_snapshot,
                paper_broker_snapshot=paper_broker_snapshot,
                market_window_snapshot=market_window_snapshot,
                allow_controlled_paper_operation_execution=allow_controlled_paper_operation_execution,
                now_provider=now_provider,
            )
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must stay JSON-safe.
        generated_at = _generated_at(now_provider)
        message = f"unexpected Stage 4J-5 executor failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=generated_at,
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_default_selected_strategy(),
                operation=_default_operation(),
                controlled_operation_payload=_controlled_operation_payload(None, None, generated_at),
                gate_checks=_gate_checks(False, {}, [], passed=False),
                execution=_execution_not_attempted("unexpected_exception", message),
                runner_output_checks=_default_runner_output_checks(False),
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
    stage4j4_execution_gate_report: dict | None,
    selected_strategy_operation_runner: Any,
    scheduler_activation_snapshot: dict | None,
    lifecycle_activation_snapshot: dict | None,
    activation_snapshot: dict | None,
    state_snapshot: dict | None,
    risk_snapshot: dict | None,
    scheduler_snapshot: dict | None,
    lifecycle_snapshot: dict | None,
    paper_broker_snapshot: dict | None,
    market_window_snapshot: dict | None,
    allow_controlled_paper_operation_execution: bool,
    now_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    generated_at = _generated_at(now_provider)
    report = stage4j4_execution_gate_report if isinstance(stage4j4_execution_gate_report, dict) else None
    data = _mapping(report)
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4j4_execution_gate_report is None:
        blockers.append("Stage 4J-4 execution gate report is missing")
    elif report is None:
        blockers.append("Stage 4J-4 execution gate report must be a dict")
        errors.append("Stage 4J-4 execution gate report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    selected_strategy_id = _selected_strategy_id(data)
    operation_id = _operation_id(data)
    artifact_checks = _artifact_checks(stage4j4_execution_gate_report, data, selected_strategy_id, operation_id)
    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    operation, operation_blockers = _operation_checks(data, operation_id)
    blockers.extend(_artifact_blockers(artifact_checks, data) + selected_blockers + operation_blockers)

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
    paper_broker_checks, broker_blockers, broker_warnings = _paper_broker_checks(paper_broker_snapshot)
    market_window_checks, market_blockers, market_warnings = _market_window_checks(market_window_snapshot)

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
        [item for item in (scheduler_activation_snapshot, lifecycle_activation_snapshot, activation_snapshot, scheduler_snapshot, lifecycle_snapshot, paper_broker_snapshot) if isinstance(item, dict)],
    )
    blockers.extend(_safety_blockers(safety_checks))

    permissions = _mapping(data.get("proposed_execution_permissions_for_4J5"))
    permission_blockers = _permission_blockers(permissions)
    blockers.extend(permission_blockers)
    if allow_controlled_paper_operation_execution is not True:
        blockers.append("allow_controlled_paper_operation_execution must be true")

    runner_method = getattr(selected_strategy_operation_runner, "run_controlled_paper_operation", None)
    if not callable(runner_method):
        blockers.append("selected_strategy_operation_runner must expose run_controlled_paper_operation")

    gate_blockers = _dedupe(blockers)
    gates_passed = not gate_blockers and not errors
    payload = _controlled_operation_payload(selected_strategy_id, operation_id, generated_at)
    gate_checks = _gate_checks(
        allow_controlled_paper_operation_execution,
        permissions,
        gate_blockers,
        passed=gates_passed,
    )

    execution = _execution_not_attempted(None, None)
    runner_checks = _default_runner_output_checks(False)
    unsafe_runner_flags: list[str] = []
    if gates_passed:
        try:
            runner_return = runner_method(payload)
            runner_succeeded = True
            failure_reason = None
        except Exception as exc:  # noqa: BLE001 - injected boundary becomes flat report data.
            runner_return = {}
            runner_succeeded = False
            failure_reason = f"{type(exc).__name__}: {exc}"
            errors.append(failure_reason)
        raw_runner_result = {} if runner_return is None else runner_return
        runner_result = raw_runner_result if isinstance(raw_runner_result, dict) else {"_non_dict_runner_result": raw_runner_result}
        runner_checks, runner_blockers, runner_warnings, unsafe_runner_flags = _runner_output_checks(
            raw_runner_result,
            selected_strategy_id,
            operation_id,
            gates_passed,
            allow_controlled_paper_operation_execution,
        )
        blockers.extend(runner_blockers)
        warnings.extend(runner_warnings)
        result_status = runner_result.get("status") if isinstance(runner_result, dict) else None
        execution = {
            "attempted": True,
            "runner_called": True,
            "runner_succeeded": runner_succeeded,
            "failed_step": "runner_call" if not runner_succeeded else ("runner_output_validation" if runner_blockers else None),
            "failure_reason": failure_reason if failure_reason is not None else ("; ".join(runner_blockers) if runner_blockers else None),
            "result_status": result_status,
            "runner_result": _json_safe(runner_result),
            "unsafe_runner_flags": list(unsafe_runner_flags),
            "completed": runner_succeeded and not runner_blockers,
        }

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    ready = (
        gates_passed
        and execution.get("runner_called") is True
        and execution.get("runner_succeeded") is True
        and execution.get("completed") is True
        and not blocker_list
        and not error_list
    )

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
        operation=operation,
        controlled_operation_payload=payload,
        gate_checks=gate_checks,
        execution=execution,
        runner_output_checks=runner_checks,
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
    readiness = _mapping(data.get("readiness_for_stage4j5"))
    gate = _mapping(data.get("execution_gate"))
    permissions = _mapping(data.get("proposed_execution_permissions_for_4J5"))
    return {
        "stage4j4_report_present": isinstance(report, dict),
        "stage4j4_report_ready": (
            data.get("stage4j4_controlled_paper_operation_execution_gate_report") is True
            and readiness.get("ready_to_build_controlled_paper_operation_executor") is True
            and data.get("success") is True
        ),
        "selected_strategy_present": isinstance(selected_strategy_id, str) and bool(selected_strategy_id),
        "operation_id_present": isinstance(operation_id, str) and bool(operation_id),
        "execution_gate_ready": gate.get("ready_for_4J5") is True,
        "permissions_valid": not _permission_blockers(permissions),
    }


def _artifact_blockers(checks: dict[str, bool], data: dict[str, Any]) -> list[str]:
    labels = {
        "stage4j4_report_present": "Stage 4J-4 execution gate report is missing",
        "stage4j4_report_ready": "Stage 4J-4 execution gate report is not ready for Stage 4J-5",
        "selected_strategy_present": "selected strategy is missing from accepted Stage 4J-4 report",
        "operation_id_present": "operation_id is missing from accepted Stage 4J-4 report",
        "execution_gate_ready": "Stage 4J-4 execution gate is not ready_for_4J5",
        "permissions_valid": "Stage 4J-4 proposed execution permissions are unsafe",
    }
    blockers = [label for key, label in labels.items() if checks.get(key) is not True]
    if _as_list(data.get("errors")):
        blockers.append("Stage 4J-4 report contains errors")
    return blockers


def _selected_strategy_checks(
    report: dict[str, Any], selected_strategy_id: str | None
) -> tuple[dict[str, Any], list[str]]:
    selected = _mapping(report.get("selected_strategy"))
    paper_only = selected.get("paper_only") is True
    one_strategy_only = selected.get("one_strategy_only") is True
    blockers: list[str] = []
    if not selected_strategy_id:
        blockers.append("selected strategy id is missing or invalid")
    if not paper_only:
        blockers.append("Stage 4J-4 selected strategy must be paper_only true")
    if not one_strategy_only:
        blockers.append("Stage 4J-4 selected strategy must be one_strategy_only true")
    return {"selected_strategy_id": selected_strategy_id, "paper_only": paper_only, "one_strategy_only": one_strategy_only}, blockers


def _operation_checks(report: dict[str, Any], operation_id: str | None) -> tuple[dict[str, Any], list[str]]:
    operation = _mapping(report.get("operation"))
    blockers: list[str] = []
    if not operation_id:
        blockers.append("operation_id is missing from Stage 4J-4 operation")
    if operation.get("live_trading_enabled") is True:
        blockers.append("Stage 4J-4 operation shows live trading enabled")
    if operation.get("broker_submission_enabled") is True:
        blockers.append("Stage 4J-4 operation shows broker submission enabled")
    return {
        "operation_id": operation_id,
        "operation_scope": "single_strategy_controlled_paper_operation_executor",
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }, blockers


def _permission_blockers(permissions: dict[str, Any]) -> list[str]:
    required_true = ("may_build_executor_next_phase", "may_call_strategy_next_phase")
    required_false = (
        "may_fetch_market_data_next_phase",
        "may_qualify_contracts_next_phase",
        "may_create_intent_next_phase",
        "may_create_ticket_next_phase",
        "may_submit_order_next_phase",
        "may_write_state_next_phase",
        "may_write_ledger_next_phase",
        "live_trading_enabled",
        "all_strategies_enabled",
        "broker_submission_enabled",
    )
    blockers = [f"permission {key} must be true" for key in required_true if permissions.get(key) is not True]
    blockers.extend(f"permission {key} must be false" for key in required_false if permissions.get(key) is not False)
    return blockers


def _gate_checks(
    allow_execution: bool, permissions: dict[str, Any], reasons: list[str], *, passed: bool
) -> dict[str, Any]:
    return {
        "allow_controlled_paper_operation_execution": allow_execution is True,
        "may_build_executor_next_phase": permissions.get("may_build_executor_next_phase") is True,
        "may_call_strategy_next_phase": permissions.get("may_call_strategy_next_phase") is True,
        "market_data_blocked": permissions.get("may_fetch_market_data_next_phase") is False,
        "contract_qualification_blocked": permissions.get("may_qualify_contracts_next_phase") is False,
        "intent_creation_blocked": permissions.get("may_create_intent_next_phase") is False,
        "ticket_creation_blocked": permissions.get("may_create_ticket_next_phase") is False,
        "order_submission_blocked": permissions.get("may_submit_order_next_phase") is False,
        "state_write_blocked": permissions.get("may_write_state_next_phase") is False,
        "ledger_write_blocked": permissions.get("may_write_ledger_next_phase") is False,
        "live_trading_disabled": permissions.get("live_trading_enabled") is False,
        "all_strategies_disabled": permissions.get("all_strategies_enabled") is False,
        "broker_submission_disabled": permissions.get("broker_submission_enabled") is False,
        "passed": passed,
        "reasons": list(reasons),
    }


def _controlled_operation_payload(
    selected_strategy_id: str | None, operation_id: str | None, generated_at: str
) -> dict[str, Any]:
    return {
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "source_stage": "4J-5",
        "permission_source_stage": "4J-4",
        "operation_scope": "single_strategy_controlled_paper_operation_executor",
        "paper_only": True,
        "one_strategy_only": True,
        "allow_strategy_call": True,
        "allow_market_data": False,
        "allow_contract_qualification": False,
        "allow_intent_creation": False,
        "allow_ticket_creation": False,
        "allow_order_submission": False,
        "allow_state_write": False,
        "allow_ledger_write": False,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "broker_submission_enabled": False,
        "generated_at": generated_at,
        "required_runtime_guards": [
            "single_strategy_only",
            "paper_only",
            "report_only",
            "no_market_data",
            "no_contract_qualification",
            "no_intents",
            "no_tickets",
            "no_orders",
            "no_state_write",
            "no_ledger_write",
        ],
        "required_pre_execution_checks": [
            "accepted_stage4j4_gate",
            "selected_strategy_matches_gate",
            "operation_id_matches_gate",
            "risk_controls_available_if_snapshot_supplied",
            "state_clean_if_snapshot_supplied",
            "scheduler_lifecycle_safe_if_snapshot_supplied",
            "paper_broker_safe_if_snapshot_supplied",
            "market_window_not_explicitly_disallowed",
        ],
    }


def _runner_output_checks(
    raw: Any,
    selected_strategy_id: str | None,
    operation_id: str | None,
    gates_passed: bool,
    allow_execution: bool,
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    result = raw if isinstance(raw, dict) else {}
    status = result.get("status")
    warnings: list[str] = []
    blockers: list[str] = []
    unsafe_flags: list[str] = []
    checks = {
        "runner_result_present": isinstance(raw, dict) and bool(raw),
        "status_allowed": status in ALLOWED_RUNNER_STATUSES,
        "selected_strategy_matches": result.get("selected_strategy_id") == selected_strategy_id,
        "operation_id_matches": result.get("operation_id") == operation_id,
        "paper_only": result.get("paper_only") is True,
        "one_strategy_only": result.get("one_strategy_only") is True,
        "strategy_call_allowed": result.get("strategy_called") is True and gates_passed and allow_execution is True
        or status in ("skipped_no_signal", "blocked_by_gate") and result.get("strategy_called") is not True,
        "no_market_data_requested": result.get("market_data_requested", False) is False,
        "no_contracts_qualified": result.get("contracts_qualified", False) is False,
        "no_intents_created": result.get("intents_created", False) is False,
        "no_tickets_created": result.get("tickets_created", False) is False,
        "no_orders_submitted": result.get("orders_submitted", False) is False,
        "no_state_written": result.get("state_written", False) is False,
        "no_ledger_written": result.get("ledger_written", False) is False,
        "no_live_trading": result.get("live_trading_enabled", False) is False,
        "no_all_strategy_enablement": result.get("all_strategies_enabled", False) is False,
        "no_broker_submission": result.get("broker_submission_enabled", False) is False,
        "output_json_safe": isinstance(raw, dict) and _primitive_json_safe(raw),
    }
    if raw is None:
        blockers.append("runner returned no output")
    elif not isinstance(raw, dict):
        blockers.append("runner output must be a dict")
    if status == "skipped_no_signal":
        warnings.append("runner skipped with no signal inside report-only boundary")
    if status == "blocked_by_gate":
        warnings.append("runner blocked internally inside report-only boundary")
    for flag in UNSAFE_RUNNER_FALSE_FLAGS:
        if isinstance(raw, dict) and flag not in raw:
            warnings.append(f"runner output missing optional safety field {flag}")
    labels = {
        "runner_result_present": "runner result is missing",
        "status_allowed": "runner status is unsupported",
        "selected_strategy_matches": "runner selected_strategy_id does not match",
        "operation_id_matches": "runner operation_id does not match",
        "paper_only": "runner paper_only must be true",
        "one_strategy_only": "runner one_strategy_only must be true",
        "strategy_call_allowed": "runner strategy_called is not allowed for this status",
        "output_json_safe": "runner output must be JSON-safe",
    }
    for key, label in labels.items():
        if checks.get(key) is not True:
            blockers.append(label)
    for flag in UNSAFE_RUNNER_FALSE_FLAGS:
        check_key = _runner_false_check_key(flag)
        if checks.get(check_key) is not True:
            unsafe_flags.append(flag)
            blockers.append(f"runner {flag} must be strict boolean false when supplied")
    if result.get("state_written") is True:
        blockers.append("critical failure: runner reported state_written true")
    if result.get("ledger_written") is True:
        blockers.append("critical failure: runner reported ledger_written true")
    return checks, _dedupe(blockers), _dedupe(warnings), _dedupe(unsafe_flags)


def _runner_false_check_key(flag: str) -> str:
    return {
        "market_data_requested": "no_market_data_requested",
        "contracts_qualified": "no_contracts_qualified",
        "intents_created": "no_intents_created",
        "tickets_created": "no_tickets_created",
        "orders_submitted": "no_orders_submitted",
        "state_written": "no_state_written",
        "ledger_written": "no_ledger_written",
        "live_trading_enabled": "no_live_trading",
        "all_strategies_enabled": "no_all_strategy_enablement",
        "broker_submission_enabled": "no_broker_submission",
    }[flag]


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
    for prefix, snapshot, record_key, list_key, execution_key in (
        ("scheduler_activation", scheduler_activation_snapshot, "scheduler_activation_record", "scheduler_activations", "strategy_scan_execution_enabled"),
        ("lifecycle_activation", lifecycle_activation_snapshot, "lifecycle_activation_record", "lifecycle_activations", "lifecycle_transition_execution_enabled"),
        ("activation", activation_snapshot, "activation_record", "activations", None),
    ):
        current_checks, current_blockers, current_warnings = _activation_snapshot_checks(
            snapshot, selected_strategy_id, prefix, record_key, list_key, execution_key
        )
        checks.update(current_checks)
        blockers.extend(current_blockers)
        warnings.extend(current_warnings)
    return checks, blockers, warnings


def _activation_snapshot_checks(
    snapshot: dict | None,
    selected_strategy_id: str | None,
    prefix: str,
    record_key: str,
    list_key: str,
    execution_key: str | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append(f"{prefix} snapshot missing; 4J-6 must verify activation artifact")
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
        for key in ("live_trading_enabled", "all_strategies_enabled", "broker_submission_enabled"):
            if record.get(key) is True:
                blockers.append(f"{prefix} snapshot {key} contradicts Stage 4J-5 safety")
        if execution_key and record.get(execution_key) is True:
            blockers.append(f"{prefix} snapshot {execution_key} contradicts Stage 4J-5 safety")
    return {f"{prefix}_snapshot_present": True, f"{prefix}_snapshot_matches": not blockers}, blockers, warnings


def _state_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    active_intents_count = _safe_int(data.get("active_intents_count"))
    active_intents_safe = data.get("active_intents_safe_for_enablement") is True
    unresolved_count = _safe_int(_first_present(data.get("unresolved_needs_reconciliation_count"), data.get("needs_reconciliation_count"), default=0))
    active_halt = bool(data.get("active_halt")) if present else False
    clean = not active_halt and unresolved_count == 0 and (active_intents_count == 0 or active_intents_safe)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("state snapshot missing; 4J-6 must verify halt, reconciliation, intents, and positions")
    if active_halt:
        blockers.append("active halt is present")
    if unresolved_count > 0:
        blockers.append("unresolved NEEDS_RECONCILIATION is present")
    if active_intents_count > 0 and not active_intents_safe:
        blockers.append("unsafe active intents are present")
    if active_intents_count > 0 and active_intents_safe:
        warnings.append("active intents present but explicitly marked safe for enablement")
    return {
        "state_snapshot_present": present,
        "active_halt": active_halt,
        "unresolved_needs_reconciliation_count": unresolved_count,
        "active_intents_count": active_intents_count,
        "active_intents_safe_for_enablement": active_intents_safe,
        "open_positions_count": _safe_int(data.get("open_positions_count")),
        "state_snapshot_clean": clean,
    }, blockers, warnings


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
        warnings.append("risk snapshot missing; 4J-6 must verify kill switch, hard halt, and daily loss controls")
        return checks, blockers, warnings
    for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
        if checks[key] is not True:
            blockers.append(f"risk snapshot {key} must be true")
    if data.get("max_position_limit_available") is False:
        blockers.append("risk snapshot max_position_limit_available must not be false when supplied")
    if checks["risk_bypass_enabled"] is True:
        blockers.append("risk bypass is enabled")
    return checks, blockers, warnings


def _scheduler_checks(snapshot: dict | None, selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    jobs = _as_list(data.get("jobs")) + _as_list(data.get("scheduled_jobs"))
    matching_jobs = [job for job in jobs if isinstance(job, dict) and _job_strategy_id(job) == selected_strategy_id]
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("scheduler snapshot missing; 4J-6 must verify scheduler state")
    all_strategy_enabled = any(data.get(key) is True for key in ("all_strategy_scheduler_enabled", "all_strategies_enabled", "broad_scheduler_enabled"))
    scan_enabled = data.get("strategy_scan_execution_enabled") is True
    selected_job_matches = True
    for job in matching_jobs:
        selected_job_matches = selected_job_matches and _scheduler_job_safe(job, selected_strategy_id)
        scan_enabled = scan_enabled or job.get("strategy_scan_execution_enabled") is True
    if all_strategy_enabled:
        blockers.append("broad/all-strategy scheduler automation enabled")
    if matching_jobs and not selected_job_matches:
        blockers.append("selected strategy scheduled job does not match Stage 4J-5 safety constraints")
    if scan_enabled:
        blockers.append("scheduler snapshot strategy_scan_execution_enabled must be false")
    return {
        "scheduler_snapshot_present": present,
        "scheduler_already_enabled": data.get("scheduler_automation_enabled") is True or bool(matching_jobs),
        "all_strategy_scheduler_enabled": all_strategy_enabled,
        "selected_strategy_job_matches": selected_job_matches,
        "strategy_scan_execution_enabled": scan_enabled,
    }, blockers, warnings


def _lifecycle_checks(snapshot: dict | None, selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; 4J-6 must verify lifecycle state")
    broad_enabled = any(data.get(key) is True for key in ("all_strategy_lifecycle_enabled", "all_strategies_enabled", "broad_lifecycle_enabled"))
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
    return {
        "lifecycle_snapshot_present": present,
        "lifecycle_already_enabled": lifecycle_enabled,
        "lifecycle_matches_selected_strategy": matches,
        "lifecycle_transition_execution_enabled": transition_enabled,
    }, blockers, warnings


def _paper_broker_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    mode = data.get("mode")
    paper_trading = data.get("paper_trading")
    ibkr_port = data.get("ibkr_port")
    live_enabled = data.get("live_trading_enabled") is True
    broker_enabled = data.get("broker_submission_enabled") is True
    valid = True
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("paper broker snapshot missing; 4J-6 must verify PAPER broker config")
    if present and mode is not None and str(mode).upper() != "PAPER":
        valid = False
        blockers.append("paper broker snapshot mode must be PAPER")
    if present and paper_trading is False:
        valid = False
        blockers.append("paper broker snapshot paper_trading must not be false")
    if present and ibkr_port is not None and ibkr_port not in PAPER_IBKR_PORTS:
        valid = False
        blockers.append("paper broker snapshot ibkr_port must use the project PAPER port")
    if live_enabled:
        valid = False
        blockers.append("paper broker snapshot live_trading_enabled must be false")
    if broker_enabled:
        valid = False
        blockers.append("paper broker snapshot broker_submission_enabled must remain false")
    return {
        "paper_broker_snapshot_present": present,
        "mode": mode,
        "paper_trading": paper_trading,
        "ibkr_port": ibkr_port,
        "paper_config_valid": valid,
        "live_trading_enabled": live_enabled,
        "broker_submission_enabled": broker_enabled,
    }, blockers, warnings


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
        blockers.append("market window snapshot explicitly disallows controlled paper operation execution")
    if market_open is False:
        warnings.append("market is currently closed; controlled paper operation may continue only inside report-only boundary")
    if is_trading_day is False:
        warnings.append("snapshot is not a trading day; controlled paper operation may continue only inside report-only boundary")
    return {
        "market_window_snapshot_present": present,
        "allowed_to_schedule_paper_run": allowed,
        "is_trading_day": is_trading_day,
        "market_open": market_open,
        "reason": data.get("reason") if present else None,
    }, blockers, warnings


def _safety_checks(report_safety: dict[str, Any], snapshots: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "no_live_trading": report_safety.get("no_live_trading") is True and _none_true("live_trading_enabled", *snapshots),
        "no_all_strategy_enablement": report_safety.get("no_all_strategy_enablement") is True and _none_true("all_strategies_enabled", *snapshots),
        "no_broker_submission_enabled": report_safety.get("no_broker_submission_enabled") is True and _none_true("broker_submission_enabled", *snapshots),
        "no_market_data": report_safety.get("no_market_data") is True and _none_true("market_data_enabled", *snapshots),
        "no_contract_qualification": report_safety.get("no_contract_qualification") is True and _none_true("contract_qualification_enabled", *snapshots),
        "no_order_submission": report_safety.get("no_order_submission") is True and _none_true("order_submission_enabled", *snapshots),
        "no_intent_creation": _none_true("intent_creation_enabled", *snapshots),
        "no_ticket_creation": _none_true("ticket_creation_enabled", *snapshots),
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
        "no_intent_creation": "intent creation safety flag is enabled",
        "no_ticket_creation": "ticket creation safety flag is enabled",
        "no_state_write": "state write safety flag is enabled",
        "no_ledger_write": "ledger write safety flag is enabled",
    }
    return [label for key, label in labels.items() if checks.get(key) is not True]


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    operation: dict[str, Any],
    controlled_operation_payload: dict[str, Any],
    gate_checks: dict[str, Any],
    execution: dict[str, Any],
    runner_output_checks: dict[str, Any],
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
        "dry_run": False,
        "stage4j5_controlled_paper_operation_executor_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "operation": operation,
        "controlled_operation_payload": controlled_operation_payload,
        "gate_checks": gate_checks,
        "execution": execution,
        "runner_output_checks": runner_output_checks,
        "activation_snapshot_checks": activation_snapshot_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4j6": {
            "ready_to_build_controlled_paper_operation_acceptance": ready,
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


def _execution_not_attempted(failed_step: str | None, failure_reason: str | None) -> dict[str, Any]:
    return {
        "attempted": False,
        "runner_called": False,
        "runner_succeeded": False,
        "failed_step": failed_step,
        "failure_reason": failure_reason,
        "result_status": None,
        "runner_result": {},
        "unsafe_runner_flags": [],
        "completed": False,
    }


def _default_runner_output_checks(present: bool) -> dict[str, Any]:
    return {
        "runner_result_present": present,
        "status_allowed": False,
        "selected_strategy_matches": False,
        "operation_id_matches": False,
        "paper_only": False,
        "one_strategy_only": False,
        "strategy_call_allowed": False,
        "no_market_data_requested": True,
        "no_contracts_qualified": True,
        "no_intents_created": True,
        "no_tickets_created": True,
        "no_orders_submitted": True,
        "no_state_written": True,
        "no_ledger_written": True,
        "no_live_trading": True,
        "no_all_strategy_enablement": True,
        "no_broker_submission": True,
        "output_json_safe": False,
    }


def _selected_strategy_id(report: dict[str, Any]) -> str | None:
    value = _mapping(report.get("selected_strategy")).get("selected_strategy_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _operation_id(report: dict[str, Any]) -> str | None:
    value = _mapping(report.get("operation")).get("operation_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


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
        "stage4j4_report_present": False,
        "stage4j4_report_ready": False,
        "selected_strategy_present": False,
        "operation_id_present": False,
        "execution_gate_ready": False,
        "permissions_valid": False,
    }


def _default_selected_strategy() -> dict[str, Any]:
    return {"selected_strategy_id": None, "paper_only": False, "one_strategy_only": False}


def _default_operation() -> dict[str, Any]:
    return {
        "operation_id": None,
        "operation_scope": "single_strategy_controlled_paper_operation_executor",
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
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
    return {
        "no_live_trading": False,
        "no_all_strategy_enablement": False,
        "no_broker_submission_enabled": False,
        "no_market_data": False,
        "no_contract_qualification": False,
        "no_order_submission": False,
        "no_intent_creation": False,
        "no_ticket_creation": False,
        "no_state_write": False,
        "no_ledger_write": False,
    }
