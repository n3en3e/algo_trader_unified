"""Pure Stage 4K-6 market data and contract qualification acceptance report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
import re
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding"
)
PAPER_IBKR_PORTS = {4004}
NEXT_RECOMMENDED_PHASE = "Stage 4L signal/readiness integration"
STALE_4J_KEYS = (
    "proposed_execution_permissions_for_" + "4J5",
    "may_call_" + "strategy_next_phase",
    "may_build_" + "executor_next_phase",
    "may_fetch_" + "market_data_next_phase",
)
REQUIRED_TRACE_FIELDS = (
    "sequence_number",
    "provider_type",
    "provider_method",
    "payload_id",
    "selected_strategy_id",
    "operation_id",
    "input_payload",
    "provider_called",
    "direct_ib_call_made",
    "success",
    "result",
    "failure_reason",
    "skipped_reason",
)
REQUIRED_OPERATION_FIELDS = (
    "provider_type",
    "provider_method",
    "payload_id",
    "selected_strategy_id",
    "operation_id",
    "target",
    "status",
)
RESULT_UNSAFE_TRUE_FLAGS = (
    "live_trading_enabled",
    "broker_submission_enabled",
    "order_submitted",
    "order_submission_enabled",
    "state_written",
    "state_write_enabled",
    "ledger_written",
    "ledger_write_enabled",
    "direct_ib_call_made",
    "reqMktData_called",
    "qualifyContracts_called",
    "reqContractDetails_called",
)
RESULT_DISABLED_FLAGS = RESULT_UNSAFE_TRUE_FLAGS
TRACE_DISABLED_FLAGS = (
    "direct_ib_call_made",
    "live_trading_enabled",
    "broker_submission_enabled",
    "order_submission_enabled",
    "state_write_enabled",
    "ledger_write_enabled",
)
EXPECTED_SKIP_REASONS = (
    "not permitted",
    "not required",
    "provider category not permitted",
    "provider category not required",
)
ORDERED_NEXT_STEPS = [
    "Proceed to Stage 4L signal/readiness integration using accepted 4K outputs as read-only inputs.",
    "Keep live trading disabled.",
    "Keep all other strategies disabled until explicitly enabled.",
    "Keep broker submission separately gated until the explicit broker-submission phase.",
    "Keep intents, tickets, state writes, and ledger writes separately gated until their explicit phases.",
    "Use accepted controlled provider results only as read-only inputs for future signal planning.",
]
DO_NOT_DO_YET = [
    "Do not enable live trading.",
    "Do not enable all strategies.",
    "Do not place orders now.",
    "Do not enable broker submission now.",
    "Do not create intents or tickets now.",
    "Do not write state or ledger now.",
    "Do not change strategy thresholds.",
    "Do not change position sizing.",
    "Do not bypass risk controls.",
    "Do not call direct market data methods.",
    "Do not call direct contract qualification methods.",
    "Do not run strategy scans now.",
]


def build_stage4k6_market_data_contract_acceptance_report(
    *,
    stage4k5_executor_report: dict | None,
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
    """Build a read-only acceptance report from an accepted Stage 4K-5 executor report."""

    try:
        return _json_safe(
            _build_report(
                stage4k5_executor_report=stage4k5_executor_report,
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
    except Exception as exc:  # noqa: BLE001 - report boundary must stay JSON-safe.
        message = f"unexpected Stage 4K-6 acceptance failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_default_selected_strategy(),
                operation=_default_operation(),
                stage4k5_executor_checks=_default_stage4k5_executor_checks(),
                provider_result_acceptance=_provider_result_acceptance(False, None, None, False, False, [], [], 0, 0, 0, 0, [], [message], []),
                operation_audit=_operation_audit(False, False, False, False, False, False, False, False, False, False, False, [message], []),
                accepted_market_data_outputs=_accepted_outputs("market_data", None, None, []),
                accepted_contract_qualification_outputs=_accepted_outputs("contract_qualification", None, None, []),
                boundary_checks=_default_boundary_checks(),
                required_inputs_for_next_phase=_required_inputs_for_next_phase(),
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
    stage4k5_executor_report: Any,
    scheduler_activation_snapshot: Any,
    lifecycle_activation_snapshot: Any,
    activation_snapshot: Any,
    state_snapshot: Any,
    risk_snapshot: Any,
    scheduler_snapshot: Any,
    lifecycle_snapshot: Any,
    paper_broker_snapshot: Any,
    market_window_snapshot: Any,
    now_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    generated_at = _generated_at(now_provider)
    report = stage4k5_executor_report if isinstance(stage4k5_executor_report, dict) else None
    data = _mapping(report)
    blockers: list[str] = []
    warnings: list[str] = []
    errors = _as_string_list(data.get("errors"))

    if stage4k5_executor_report is None:
        blockers.append("Stage 4K-5 executor report is missing")
    elif report is None:
        blockers.append("Stage 4K-5 executor report must be a dict")
        errors.append("Stage 4K-5 executor report must be a dict")

    selected_strategy_id = _selected_strategy_id(data)
    operation_id = _operation_id(data)
    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    operation, operation_blockers = _operation_checks(data, operation_id)
    artifact_checks = _artifact_checks(stage4k5_executor_report, data)
    stage4k5_checks, executor_blockers = _stage4k5_executor_checks(data, selected_strategy_id, operation_id)
    trace_items, trace_blockers, trace_warnings = _trace_checks(data, selected_strategy_id, operation_id)
    applied_items, applied_blockers = _operation_list_checks(data.get("applied_operations"), "applied_operations", selected_strategy_id, operation_id)
    failed_items, failed_blockers = _operation_list_checks(data.get("failed_operations"), "failed_operations", selected_strategy_id, operation_id)
    skipped_items, skipped_blockers = _operation_list_checks(data.get("skipped_operations"), "skipped_operations", selected_strategy_id, operation_id)
    boundary_checks, boundary_blockers = _boundary_checks(data, trace_items)
    safety_checks, safety_blockers = _safety_checks(boundary_checks)
    provider_acceptance, provider_blockers, provider_warnings, accepted_md_results, accepted_cq_results = _provider_result_acceptance_checks(
        data, trace_items, selected_strategy_id, operation_id
    )
    operation_audit, audit_blockers, audit_warnings = _operation_audit_checks(
        trace_items, applied_items, failed_items, skipped_items, selected_strategy_id, operation_id
    )
    stale_blockers = _stale_key_blockers(data)
    activation_checks, activation_blockers, activation_warnings = _activation_snapshot_group_checks(
        scheduler_activation_snapshot=scheduler_activation_snapshot,
        lifecycle_activation_snapshot=lifecycle_activation_snapshot,
        activation_snapshot=activation_snapshot,
        selected_strategy_id=selected_strategy_id,
    )
    state_checks, state_blockers, state_warnings = _state_checks(state_snapshot)
    risk_checks, risk_blockers, risk_warnings = _risk_checks(risk_snapshot)
    scheduler_checks, scheduler_blockers, scheduler_warnings = _scheduler_checks(scheduler_snapshot, selected_strategy_id)
    lifecycle_checks, lifecycle_blockers, lifecycle_warnings = _lifecycle_checks(lifecycle_snapshot, selected_strategy_id)
    paper_broker_checks, broker_blockers, broker_warnings = _paper_broker_checks(paper_broker_snapshot)
    market_window_checks, market_blockers, market_warnings = _market_window_checks(market_window_snapshot)

    blockers.extend(
        selected_blockers
        + operation_blockers
        + _artifact_blockers(artifact_checks, data)
        + executor_blockers
        + trace_blockers
        + applied_blockers
        + failed_blockers
        + skipped_blockers
        + boundary_blockers
        + safety_blockers
        + provider_blockers
        + audit_blockers
        + stale_blockers
        + activation_blockers
        + state_blockers
        + risk_blockers
        + scheduler_blockers
        + lifecycle_blockers
        + broker_blockers
        + market_blockers
    )
    warnings.extend(
        trace_warnings
        + provider_warnings
        + audit_warnings
        + activation_warnings
        + state_warnings
        + risk_warnings
        + scheduler_warnings
        + lifecycle_warnings
        + broker_warnings
        + market_warnings
    )
    if errors:
        blockers.append("Stage 4K-5 report contains errors")

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    accepted_outputs_available = bool(accepted_md_results or accepted_cq_results)
    ready = (
        not blocker_list
        and not error_list
        and provider_acceptance.get("accepted") is True
        and operation_audit.get("operation_audit_passed") is True
        and accepted_outputs_available
    )

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
        operation=operation,
        stage4k5_executor_checks=stage4k5_checks,
        provider_result_acceptance=provider_acceptance,
        operation_audit=operation_audit,
        accepted_market_data_outputs=_accepted_outputs("market_data", selected_strategy_id, operation_id, accepted_md_results),
        accepted_contract_qualification_outputs=_accepted_outputs("contract_qualification", selected_strategy_id, operation_id, accepted_cq_results),
        boundary_checks=boundary_checks,
        required_inputs_for_next_phase=_required_inputs_for_next_phase(),
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


def _artifact_checks(report: Any, data: dict[str, Any]) -> dict[str, bool]:
    readiness = _mapping(data.get("readiness_for_stage4k6"))
    return {
        "stage4k5_report_present": isinstance(report, dict),
        "stage4k5_report_ready": (
            data.get("stage4k5_market_data_contract_executor_report") is True
            and readiness.get("ready_to_build_market_data_contract_acceptance") is True
            and data.get("success") is True
        ),
        "selected_strategy_present": isinstance(_selected_strategy_id(data), str),
        "operation_id_present": isinstance(_operation_id(data), str),
        "executor_results_present": isinstance(data.get("market_data_execution_results"), dict)
        and isinstance(data.get("contract_qualification_execution_results"), dict),
        "provider_call_trace_present": isinstance(data.get("provider_call_trace"), list),
        "operation_lists_present": all(isinstance(data.get(key), list) for key in ("applied_operations", "failed_operations", "skipped_operations")),
    }


def _artifact_blockers(checks: dict[str, bool], data: dict[str, Any]) -> list[str]:
    labels = {
        "stage4k5_report_present": "Stage 4K-5 executor report is missing",
        "stage4k5_report_ready": "Stage 4K-5 executor report is not ready for Stage 4K-6",
        "selected_strategy_present": "selected strategy is missing from Stage 4K-5 report",
        "operation_id_present": "operation_id is missing from Stage 4K-5 report",
        "executor_results_present": "market data and contract qualification execution result dicts are required",
        "provider_call_trace_present": "provider_call_trace must be a native list",
        "operation_lists_present": "applied_operations, failed_operations, and skipped_operations must be native lists",
    }
    blockers = [label for key, label in labels.items() if checks.get(key) is not True]
    if _as_list(data.get("errors")):
        blockers.append("Stage 4K-5 report contains errors")
    return blockers


def _selected_strategy_checks(report: dict[str, Any], selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str]]:
    selected = _mapping(report.get("selected_strategy"))
    paper_only = selected.get("paper_only") is True
    one_strategy_only = selected.get("one_strategy_only") is True
    blockers: list[str] = []
    if not selected_strategy_id:
        blockers.append("selected strategy id is missing or invalid")
    if not paper_only:
        blockers.append("Stage 4K-5 selected strategy must be paper_only true")
    if not one_strategy_only:
        blockers.append("Stage 4K-5 selected strategy must be one_strategy_only true")
    return {"selected_strategy_id": selected_strategy_id, "paper_only": paper_only, "one_strategy_only": one_strategy_only}, blockers


def _operation_checks(report: dict[str, Any], operation_id: str | None) -> tuple[dict[str, Any], list[str]]:
    operation = _mapping(report.get("operation"))
    blockers: list[str] = []
    if not operation_id:
        blockers.append("operation_id is missing from Stage 4K-5 operation")
    if operation.get("live_trading_enabled") is True:
        blockers.append("Stage 4K-5 operation shows live trading enabled")
    if operation.get("broker_submission_enabled") is True:
        blockers.append("Stage 4K-5 operation shows broker submission enabled")
    return {
        "operation_id": operation_id,
        "operation_scope": operation.get("operation_scope") or "single_strategy_market_data_contract_acceptance",
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }, blockers


def _stage4k5_executor_checks(report: dict[str, Any], selected_strategy_id: str | None, operation_id: str | None) -> tuple[dict[str, Any], list[str]]:
    md = _mapping(report.get("market_data_execution_results"))
    cq = _mapping(report.get("contract_qualification_execution_results"))
    readiness = _mapping(report.get("readiness_for_stage4k6"))
    checks = {
        "stage4k5_report_true": report.get("stage4k5_market_data_contract_executor_report") is True,
        "stage4k5_success": report.get("success") is True,
        "ready_to_build_market_data_contract_acceptance": readiness.get("ready_to_build_market_data_contract_acceptance") is True,
        "dry_run_false": report.get("dry_run") is False,
        "selected_strategy_matches": _selected_strategy_id(report) == selected_strategy_id,
        "operation_id_matches": _operation_id(report) == operation_id,
        "market_data_execution_results_is_dict": isinstance(report.get("market_data_execution_results"), dict),
        "contract_qualification_execution_results_is_dict": isinstance(report.get("contract_qualification_execution_results"), dict),
        "market_data_attempted_native_bool": isinstance(md.get("attempted"), bool),
        "market_data_provider_called_native_bool": isinstance(md.get("provider_called"), bool),
        "market_data_success_native_bool": isinstance(md.get("success"), bool),
        "contract_qualification_attempted_native_bool": isinstance(cq.get("attempted"), bool),
        "contract_qualification_provider_called_native_bool": isinstance(cq.get("provider_called"), bool),
        "contract_qualification_success_native_bool": isinstance(cq.get("success"), bool),
        "at_least_one_provider_category_attempted": md.get("attempted") is True or cq.get("attempted") is True,
    }
    blockers = [f"Stage 4K-5 executor check {key} failed" for key, value in checks.items() if value is not True]
    blockers.extend(_result_summary_blockers("market_data_execution_results", md, ("direct_ib_call_made", "reqMktData_called")))
    blockers.extend(_result_summary_blockers("contract_qualification_execution_results", cq, ("direct_ib_call_made", "qualifyContracts_called", "reqContractDetails_called")))
    if md.get("attempted") is True and (md.get("success") is not True or md.get("provider_called") is not True):
        blockers.append("market_data_execution_results attempted true requires success and provider_called true")
    if cq.get("attempted") is True and (cq.get("success") is not True or cq.get("provider_called") is not True):
        blockers.append("contract_qualification_execution_results attempted true requires success and provider_called true")
    return checks, _dedupe(blockers)


def _result_summary_blockers(label: str, result: dict[str, Any], false_fields: tuple[str, ...]) -> list[str]:
    blockers: list[str] = []
    for key in false_fields:
        if result.get(key) is not False:
            blockers.append(f"{label} {key} must be strict native boolean false")
    for key in RESULT_UNSAFE_TRUE_FLAGS:
        if result.get(key) is True:
            blockers.append(f"{label} {key} must not be true")
        if result.get(key) == "False":
            blockers.append(f"{label} {key} must be native bool false, not string")
    return blockers


def _trace_checks(report: dict[str, Any], selected_strategy_id: str | None, operation_id: str | None) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    raw = report.get("provider_call_trace")
    blockers: list[str] = []
    warnings: list[str] = []
    if raw is None:
        return [], ["provider_call_trace is missing"], warnings
    if not isinstance(raw, list):
        return [], ["provider_call_trace must be a native list"], warnings
    items: list[dict[str, Any]] = []
    provider_order: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            blockers.append("provider_call_trace contains malformed item")
            continue
        items.append(item)
        provider_type = item.get("provider_type")
        if isinstance(provider_type, str):
            provider_order.append(provider_type)
        for field in REQUIRED_TRACE_FIELDS:
            if field not in item:
                blockers.append(f"provider_call_trace item missing required field {field}")
        if item.get("sequence_number") != index + 1:
            blockers.append("provider_call_trace sequence_number must be deterministic and ordered")
        if item.get("provider_type") not in ("market_data", "contract_qualification"):
            blockers.append("provider_call_trace provider_type is unsupported")
        if item.get("selected_strategy_id") != selected_strategy_id:
            blockers.append("provider_call_trace selected_strategy_id does not match selected strategy")
        if item.get("operation_id") != operation_id:
            blockers.append("provider_call_trace operation_id does not match operation")
        if not isinstance(item.get("input_payload"), dict) or not _primitive_json_safe(item.get("input_payload")):
            blockers.append("provider_call_trace input_payload must be a native JSON-safe dict")
        if not isinstance(item.get("result"), dict) or not _primitive_json_safe(item.get("result")):
            blockers.append("provider_call_trace result must be a native JSON-safe dict")
        if not isinstance(item.get("provider_called"), bool):
            blockers.append("provider_call_trace provider_called must be native bool")
        if item.get("direct_ib_call_made") is not False:
            blockers.append("provider_call_trace direct_ib_call_made must remain false")
        if not isinstance(item.get("success"), bool):
            blockers.append("provider_call_trace success must be native bool")
        for key in ("failure_reason", "skipped_reason"):
            value = item.get(key)
            if value is not None and not isinstance(value, str):
                blockers.append(f"provider_call_trace {key} must be None or a flat JSON-safe string")
            if isinstance(value, str) and (_has_raw_newline(value) or _has_memory_repr(value)):
                blockers.append(f"provider_call_trace {key} must be flat JSON-safe string")
        blockers.extend(_provider_result_blockers(item.get("result") if isinstance(item.get("result"), dict) else {}, "provider_call_trace result"))
        for key in TRACE_DISABLED_FLAGS:
            if key in item and item.get(key) == "False":
                blockers.append(f"provider_call_trace {key} must be native bool false, not string")
    if "market_data" in provider_order and "contract_qualification" in provider_order:
        seen_contract = False
        for provider_type in provider_order:
            if provider_type == "contract_qualification":
                seen_contract = True
            if seen_contract and provider_type == "market_data":
                blockers.append("provider_call_trace must list market_data traces before contract_qualification traces")
                break
    return items, _dedupe(blockers), _dedupe(warnings)


def _operation_list_checks(raw: Any, list_name: str, selected_strategy_id: str | None, operation_id: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    blockers: list[str] = []
    if raw is None:
        return [], [f"{list_name} is missing"]
    if not isinstance(raw, list):
        return [], [f"{list_name} must be a native list"]
    items: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            blockers.append(f"{list_name} contains malformed item")
            continue
        items.append(item)
        for field in REQUIRED_OPERATION_FIELDS:
            if field not in item:
                blockers.append(f"{list_name} item missing required field {field}")
        if item.get("selected_strategy_id") != selected_strategy_id:
            blockers.append(f"{list_name} selected_strategy_id does not match selected strategy")
        if item.get("operation_id") != operation_id:
            blockers.append(f"{list_name} operation_id does not match operation")
        for key in ("failure_reason", "skipped_reason"):
            value = item.get(key)
            if isinstance(value, str) and (_has_raw_newline(value) or _has_memory_repr(value)):
                blockers.append(f"{list_name} {key} must be flat JSON-safe string")
    return items, _dedupe(blockers)


def _provider_result_acceptance_checks(
    report: dict[str, Any],
    trace_items: list[dict[str, Any]],
    selected_strategy_id: str | None,
    operation_id: str | None,
) -> tuple[dict[str, Any], list[str], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    md = _mapping(report.get("market_data_execution_results"))
    cq = _mapping(report.get("contract_qualification_execution_results"))
    blockers: list[str] = []
    warnings: list[str] = []
    accepted_md = _accepted_trace_results(trace_items, "market_data")
    accepted_cq = _accepted_trace_results(trace_items, "contract_qualification")
    md_attempted = md.get("attempted") is True
    cq_attempted = cq.get("attempted") is True
    market_data_accepted = md_attempted and md.get("success") is True and md.get("provider_called") is True and bool(accepted_md)
    contract_accepted = cq_attempted and cq.get("success") is True and cq.get("provider_called") is True and bool(accepted_cq)
    if md_attempted and not market_data_accepted:
        blockers.append("attempted market data provider results were not accepted")
    if cq_attempted and not contract_accepted:
        blockers.append("attempted contract qualification provider results were not accepted")
    if not (market_data_accepted or contract_accepted):
        blockers.append("at least one provider category must be accepted")
    if not md_attempted and not cq_attempted:
        blockers.append("at least one provider category must be attempted")
    accepted_types = []
    rejected_types = []
    if market_data_accepted:
        accepted_types.append("market_data")
    elif md_attempted:
        rejected_types.append("market_data")
    if contract_accepted:
        accepted_types.append("contract_qualification")
    elif cq_attempted:
        rejected_types.append("contract_qualification")
    failed = _as_list(report.get("failed_operations"))
    skipped = _as_list(report.get("skipped_operations"))
    applied = _as_list(report.get("applied_operations"))
    trace_called_count = sum(1 for item in trace_items if item.get("provider_called") is True)
    accepted = market_data_accepted or contract_accepted
    acceptance = _provider_result_acceptance(
        accepted and not blockers,
        selected_strategy_id,
        operation_id,
        market_data_accepted,
        contract_accepted,
        accepted_types,
        rejected_types,
        trace_called_count,
        len(applied),
        len(failed),
        len(skipped),
        ["accepted controlled provider results are read-only inputs"] if accepted else [],
        blockers,
        warnings,
    )
    return acceptance, _dedupe(blockers), warnings, accepted_md if market_data_accepted else [], accepted_cq if contract_accepted else []


def _accepted_trace_results(trace_items: list[dict[str, Any]], provider_type: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in trace_items:
        result = item.get("result")
        if (
            item.get("provider_type") == provider_type
            and item.get("provider_called") is True
            and item.get("success") is True
            and item.get("direct_ib_call_made") is False
            and isinstance(result, dict)
            and _primitive_json_safe(result)
            and not _provider_result_blockers(result, "provider result")
        ):
            results.append(_json_safe(result))
    return results


def _operation_audit_checks(
    trace_items: list[dict[str, Any]],
    applied_items: list[dict[str, Any]],
    failed_items: list[dict[str, Any]],
    skipped_items: list[dict[str, Any]],
    selected_strategy_id: str | None,
    operation_id: str | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    failed_empty = len(failed_items) == 0
    skipped_expected = all(_expected_skip(item) for item in skipped_items)
    if not failed_empty:
        blockers.append("failed_operations must be empty")
    if not skipped_expected:
        blockers.append("skipped_operations contains unexpected skipped operation")
    attempted = [item for item in trace_items if item.get("provider_called") is True]
    if attempted and not applied_items:
        blockers.append("applied_operations must be non-empty when provider execution was attempted")
    successful_keys = {_stable_key(item) for item in trace_items if item.get("provider_called") is True and item.get("success") is True}
    applied_keys = {_stable_key(item) for item in applied_items}
    failed_keys = {_stable_key(item) for item in failed_items}
    trace_failed_keys = {_stable_key(item) for item in trace_items if item.get("provider_called") is True and item.get("success") is False}
    trace_matches = successful_keys == applied_keys and trace_failed_keys.issubset(failed_keys)
    if successful_keys != applied_keys:
        blockers.append("provider_call_trace successful items must match applied_operations by stable IDs")
    if trace_failed_keys and not trace_failed_keys.issubset(failed_keys):
        blockers.append("failed provider_call_trace items must be accounted for by failed_operations")
    selected_consistent = all(item.get("selected_strategy_id") == selected_strategy_id for item in trace_items + applied_items + failed_items + skipped_items)
    operation_consistent = all(item.get("operation_id") == operation_id for item in trace_items + applied_items + failed_items + skipped_items)
    no_unsafe_results = all(not _provider_result_blockers(_mapping(item.get("result")), "provider result") for item in trace_items)
    no_direct_ib = all(item.get("direct_ib_call_made") is False for item in trace_items)
    no_order_write = no_unsafe_results
    passed = (
        not blockers
        and failed_empty
        and skipped_expected
        and trace_matches
        and selected_consistent
        and operation_consistent
        and no_unsafe_results
        and no_direct_ib
        and no_order_write
    )
    return _operation_audit(
        passed,
        bool(applied_items) and all(isinstance(item, dict) for item in applied_items),
        failed_empty,
        skipped_expected,
        trace_matches,
        selected_consistent,
        operation_consistent,
        failed_empty,
        no_unsafe_results,
        no_direct_ib,
        no_order_write,
        blockers,
        warnings,
    ), _dedupe(blockers), warnings


def _boundary_checks(report: dict[str, Any], trace_items: list[dict[str, Any]]) -> tuple[dict[str, bool], list[str]]:
    source = _mapping(report.get("boundary_checks"))
    checks = {
        "no_provider_called_in_4k6": True,
        "no_direct_ib_call": source.get("no_direct_ib_call") is True and all(item.get("direct_ib_call_made") is False for item in trace_items),
        "no_direct_reqMktData": source.get("no_direct_reqMktData", source.get("no_reqMktData")) is True,
        "no_direct_qualifyContracts": source.get("no_direct_qualifyContracts", source.get("no_qualifyContracts")) is True,
        "no_direct_reqContractDetails": source.get("no_direct_reqContractDetails", source.get("no_reqContractDetails")) is True,
        "no_strategy_scan": source.get("no_strategy_scan") is True,
        "no_intents_created": source.get("no_intents_created") is True,
        "no_tickets_created": source.get("no_tickets_created") is True,
        "no_orders_submitted": source.get("no_orders_submitted") is True,
        "no_broker_submission": source.get("no_broker_submission") is True,
        "no_state_written": source.get("no_state_written") is True,
        "no_ledger_written": source.get("no_ledger_written") is True,
        "no_live_trading": source.get("no_live_trading") is True,
        "no_all_strategy_enablement": source.get("no_all_strategy_enablement") is True,
        "no_scheduler_registration": source.get("no_scheduler_registration") is True,
        "no_lifecycle_execution": source.get("no_lifecycle_execution") is True,
    }
    blockers = [f"Stage 4K-5 boundary check {key} must be strict native boolean true" for key, value in checks.items() if value is not True]
    return checks, blockers


def _safety_checks(boundary_checks: dict[str, bool]) -> tuple[dict[str, bool], list[str]]:
    checks = {
        "no_live_trading": boundary_checks.get("no_live_trading") is True,
        "no_all_strategy_enablement": boundary_checks.get("no_all_strategy_enablement") is True,
        "no_broker_submission_enabled": boundary_checks.get("no_broker_submission") is True,
        "no_direct_market_data": boundary_checks.get("no_direct_reqMktData") is True,
        "no_direct_contract_qualification": boundary_checks.get("no_direct_qualifyContracts") is True and boundary_checks.get("no_direct_reqContractDetails") is True,
        "no_order_submission": boundary_checks.get("no_orders_submitted") is True,
        "no_intent_creation": boundary_checks.get("no_intents_created") is True,
        "no_ticket_creation": boundary_checks.get("no_tickets_created") is True,
        "no_state_write": boundary_checks.get("no_state_written") is True,
        "no_ledger_write": boundary_checks.get("no_ledger_written") is True,
    }
    labels = {
        "no_live_trading": "live trading safety flag is enabled",
        "no_all_strategy_enablement": "all-strategy safety flag is enabled",
        "no_broker_submission_enabled": "broker submission safety flag is enabled",
        "no_direct_market_data": "direct market data safety flag is enabled",
        "no_direct_contract_qualification": "direct contract qualification safety flag is enabled",
        "no_order_submission": "order submission safety flag is enabled",
        "no_intent_creation": "intent creation safety flag is enabled",
        "no_ticket_creation": "ticket creation safety flag is enabled",
        "no_state_write": "state write safety flag is enabled",
        "no_ledger_write": "ledger write safety flag is enabled",
    }
    return checks, [label for key, label in labels.items() if checks.get(key) is not True]


def _provider_result_blockers(result: dict[str, Any], label: str) -> list[str]:
    blockers: list[str] = []
    if not _primitive_json_safe(result):
        blockers.append(f"{label} must be JSON-safe")
    for key in RESULT_UNSAFE_TRUE_FLAGS:
        if result.get(key) is True:
            blockers.append(f"{label} {key} must not be true")
    for key in RESULT_DISABLED_FLAGS:
        if result.get(key) == "False":
            blockers.append(f"{label} {key} must be native bool false, not string")
    return blockers


def _stale_key_blockers(value: Any, path: str = "report") -> list[str]:
    blockers: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in STALE_4J_KEYS:
                blockers.append(f"stale 4J schema key present at {path}.{key}")
            blockers.extend(_stale_key_blockers(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            blockers.extend(_stale_key_blockers(item, f"{path}[{index}]"))
    return _dedupe(blockers)


def _activation_snapshot_group_checks(
    *,
    scheduler_activation_snapshot: Any,
    lifecycle_activation_snapshot: Any,
    activation_snapshot: Any,
    selected_strategy_id: str | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    warnings: list[str] = []
    snapshot_configs = [
        {
            "prefix": "scheduler_activation",
            "snapshot": scheduler_activation_snapshot,
            "record_key": "scheduler_activation_record",
            "list_key": "scheduler_activations",
            "execution_key": "strategy_scan_execution_enabled",
        },
        {
            "prefix": "lifecycle_activation",
            "snapshot": lifecycle_activation_snapshot,
            "record_key": "lifecycle_activation_record",
            "list_key": "lifecycle_activations",
            "execution_key": "lifecycle_transition_execution_enabled",
        },
        {
            "prefix": "activation",
            "snapshot": activation_snapshot,
            "record_key": "activation_record",
            "list_key": "activations",
            "execution_key": None,
        },
    ]
    for config in snapshot_configs:
        current_checks, current_blockers, current_warnings = _activation_snapshot_checks(
            config["snapshot"],
            selected_strategy_id,
            config["prefix"],
            config["record_key"],
            config["list_key"],
            config["execution_key"],
        )
        checks.update(current_checks)
        blockers.extend(current_blockers)
        warnings.extend(current_warnings)
    return checks, blockers, warnings


def _activation_snapshot_checks(snapshot: Any, selected_strategy_id: str | None, prefix: str, record_key: str, list_key: str, execution_key: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append(f"{prefix} snapshot missing; verify activation artifact before Stage 4L")
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
                blockers.append(f"{prefix} snapshot {key} contradicts Stage 4K-6 safety")
        if execution_key and record.get(execution_key) is True:
            blockers.append(f"{prefix} snapshot {execution_key} contradicts Stage 4K-6 safety")
    return {f"{prefix}_snapshot_present": True, f"{prefix}_snapshot_matches": not blockers}, blockers, warnings


def _state_checks(snapshot: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    active_intents_count = _safe_int(data.get("active_intents_count"))
    active_intents_safe = data.get("active_intents_safe_for_enablement") is True
    unresolved_count = _safe_int(_first_present(data.get("unresolved_needs_reconciliation_count"), data.get("needs_reconciliation_count"), default=0))
    active_halt = bool(data.get("active_halt")) if present else False
    clean = not active_halt and unresolved_count == 0 and (active_intents_count == 0 or active_intents_safe)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("state snapshot missing; verify halt, reconciliation, intents, and positions before Stage 4L")
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


def _risk_checks(snapshot: Any) -> tuple[dict[str, Any], list[str], list[str]]:
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
        warnings.append("risk snapshot missing; verify kill switch, hard halt, and daily loss controls before Stage 4L")
        return checks, blockers, warnings
    for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
        if checks[key] is not True:
            blockers.append(f"risk snapshot {key} must be true")
    if data.get("max_position_limit_available") is False:
        blockers.append("risk snapshot max_position_limit_available must not be false when supplied")
    if checks["risk_bypass_enabled"] is True:
        blockers.append("risk bypass is enabled")
    return checks, blockers, warnings


def _scheduler_checks(snapshot: Any, selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    jobs = _as_list(data.get("jobs")) + _as_list(data.get("scheduled_jobs"))
    matching_jobs = [job for job in jobs if isinstance(job, dict) and _job_strategy_id(job) == selected_strategy_id]
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("scheduler snapshot missing; verify scheduler state before Stage 4L")
    all_strategy_enabled = any(data.get(key) is True for key in ("all_strategy_scheduler_enabled", "all_strategies_enabled", "broad_scheduler_enabled"))
    scan_enabled = data.get("strategy_scan_execution_enabled") is True
    selected_job_matches = True
    for job in matching_jobs:
        selected_job_matches = selected_job_matches and _scheduler_job_safe(job, selected_strategy_id)
        scan_enabled = scan_enabled or job.get("strategy_scan_execution_enabled") is True
    if all_strategy_enabled:
        blockers.append("broad/all-strategy scheduler automation enabled")
    if matching_jobs and not selected_job_matches:
        blockers.append("selected strategy scheduled job does not match Stage 4K-6 safety constraints")
    if scan_enabled:
        blockers.append("scheduler snapshot strategy_scan_execution_enabled must be false")
    return {
        "scheduler_snapshot_present": present,
        "scheduler_already_enabled": data.get("scheduler_automation_enabled") is True or bool(matching_jobs),
        "all_strategy_scheduler_enabled": all_strategy_enabled,
        "selected_strategy_job_matches": selected_job_matches,
        "strategy_scan_execution_enabled": scan_enabled,
    }, blockers, warnings


def _lifecycle_checks(snapshot: Any, selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; verify lifecycle state before Stage 4L")
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


def _paper_broker_checks(snapshot: Any) -> tuple[dict[str, Any], list[str], list[str]]:
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
        warnings.append("paper broker snapshot missing; verify PAPER broker config before Stage 4L")
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


def _market_window_checks(snapshot: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    allowed = data.get("allowed_to_schedule_paper_run") if present else None
    is_trading_day = data.get("is_trading_day") if present else None
    market_open = data.get("market_open") if present else None
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append(MARKET_WINDOW_MANUAL_WARNING)
    if allowed is False:
        blockers.append("market window snapshot explicitly disallows acceptance validation")
    if market_open is False:
        warnings.append("market is currently closed; acceptance validation remains report-only")
    if is_trading_day is False:
        warnings.append("snapshot is not a trading day; acceptance validation remains report-only")
    return {
        "market_window_snapshot_present": present,
        "allowed_to_schedule_paper_run": allowed,
        "is_trading_day": is_trading_day,
        "market_open": market_open,
        "reason": data.get("reason") if present else None,
    }, blockers, warnings


def _accepted_outputs(provider_type: str, selected_strategy_id: str | None, operation_id: str | None, results: list[dict[str, Any]]) -> dict[str, Any]:
    source = "request_controlled_market_data" if provider_type == "market_data" else "qualify_controlled_contracts"
    return {
        "available": bool(results),
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "accepted_result_count": len(results),
        "accepted_results": [_json_safe(item) for item in results],
        "source_provider_method": source,
        "read_only_for_future_stages": True,
    }


def _provider_result_acceptance(
    accepted: bool,
    selected_strategy_id: str | None,
    operation_id: str | None,
    market_data_accepted: bool,
    contract_qualification_accepted: bool,
    accepted_provider_types: list[str],
    rejected_provider_types: list[str],
    provider_call_count: int,
    applied_operation_count: int,
    failed_operation_count: int,
    skipped_operation_count: int,
    reasons: list[str],
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "accepted": accepted,
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "market_data_accepted": market_data_accepted,
        "contract_qualification_accepted": contract_qualification_accepted,
        "accepted_provider_types": list(accepted_provider_types),
        "rejected_provider_types": list(rejected_provider_types),
        "provider_call_count": provider_call_count,
        "applied_operation_count": applied_operation_count,
        "failed_operation_count": failed_operation_count,
        "skipped_operation_count": skipped_operation_count,
        "reasons": list(reasons),
        "blockers": list(blockers),
        "warnings": list(warnings),
    }


def _operation_audit(
    operation_audit_passed: bool,
    applied_operations_valid: bool,
    failed_operations_empty: bool,
    skipped_operations_empty_or_expected: bool,
    trace_matches_operations: bool,
    selected_strategy_consistent: bool,
    operation_id_consistent: bool,
    no_unexpected_provider_failures: bool,
    no_unsafe_provider_results: bool,
    no_direct_ib_calls: bool,
    no_order_or_write_side_effects: bool,
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "operation_audit_passed": operation_audit_passed,
        "applied_operations_valid": applied_operations_valid,
        "failed_operations_empty": failed_operations_empty,
        "skipped_operations_empty_or_expected": skipped_operations_empty_or_expected,
        "trace_matches_operations": trace_matches_operations,
        "selected_strategy_consistent": selected_strategy_consistent,
        "operation_id_consistent": operation_id_consistent,
        "no_unexpected_provider_failures": no_unexpected_provider_failures,
        "no_unsafe_provider_results": no_unsafe_provider_results,
        "no_direct_ib_calls": no_direct_ib_calls,
        "no_order_or_write_side_effects": no_order_or_write_side_effects,
        "blockers": list(blockers),
        "warnings": list(warnings),
    }


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    operation: dict[str, Any],
    stage4k5_executor_checks: dict[str, Any],
    provider_result_acceptance: dict[str, Any],
    operation_audit: dict[str, Any],
    accepted_market_data_outputs: dict[str, Any],
    accepted_contract_qualification_outputs: dict[str, Any],
    boundary_checks: dict[str, bool],
    required_inputs_for_next_phase: list[str],
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
        "stage4k6_market_data_contract_acceptance_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "operation": operation,
        "stage4k5_executor_checks": stage4k5_executor_checks,
        "provider_result_acceptance": provider_result_acceptance,
        "operation_audit": operation_audit,
        "accepted_market_data_outputs": accepted_market_data_outputs,
        "accepted_contract_qualification_outputs": accepted_contract_qualification_outputs,
        "boundary_checks": boundary_checks,
        "required_inputs_for_next_phase": required_inputs_for_next_phase,
        "activation_snapshot_checks": activation_snapshot_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_next_phase": {
            "stage4k_complete": ready,
            "ready_to_proceed_after_stage4k": ready,
            "next_recommended_phase": NEXT_RECOMMENDED_PHASE,
            "blockers": list(blockers if not ready else []),
            "warnings": list(warnings),
        },
        "recommendations": {"ordered_next_steps": list(ORDERED_NEXT_STEPS), "do_not_do_yet": list(DO_NOT_DO_YET)},
        "success": ready,
        "errors": list(errors),
        "warnings": list(warnings),
    }


def _selected_strategy_id(report: dict[str, Any]) -> str | None:
    value = _mapping(report.get("selected_strategy")).get("selected_strategy_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _operation_id(report: dict[str, Any]) -> str | None:
    value = _mapping(report.get("operation")).get("operation_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _stable_key(item: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    return (
        item.get("payload_id"),
        item.get("provider_type"),
        item.get("provider_method"),
        item.get("selected_strategy_id"),
        item.get("operation_id"),
    )


def _expected_skip(item: dict[str, Any]) -> bool:
    if item.get("status") not in (None, "skipped"):
        return False
    reason = item.get("skipped_reason")
    if not isinstance(reason, str):
        return False
    lowered = reason.lower()
    return any(marker in lowered for marker in EXPECTED_SKIP_REASONS)


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
    if isinstance(value, (datetime, date, tuple, Decimal, bytes)) or callable(value):
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


def _has_raw_newline(value: str) -> bool:
    return "\n" in value or "\r" in value


def _has_memory_repr(value: str) -> bool:
    return re.search(r"<[^>]+ at 0x[0-9a-fA-F]+>", value) is not None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _required_inputs_for_next_phase() -> list[str]:
    return [
        "accepted Stage 4K-6 report",
        "accepted read-only controlled market data outputs when available",
        "accepted read-only controlled contract qualification outputs when available",
        "same selected_strategy_id and operation_id from Stage 4K-5",
        "fresh state, risk, scheduler, lifecycle, paper broker, and market window snapshots",
    ]


def _default_artifact_checks() -> dict[str, bool]:
    return {
        "stage4k5_report_present": False,
        "stage4k5_report_ready": False,
        "selected_strategy_present": False,
        "operation_id_present": False,
        "executor_results_present": False,
        "provider_call_trace_present": False,
        "operation_lists_present": False,
    }


def _default_selected_strategy() -> dict[str, Any]:
    return {"selected_strategy_id": None, "paper_only": False, "one_strategy_only": False}


def _default_operation() -> dict[str, Any]:
    return {
        "operation_id": None,
        "operation_scope": "single_strategy_market_data_contract_acceptance",
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }


def _default_stage4k5_executor_checks() -> dict[str, Any]:
    return {
        "stage4k5_report_true": False,
        "stage4k5_success": False,
        "ready_to_build_market_data_contract_acceptance": False,
        "dry_run_false": False,
        "selected_strategy_matches": False,
        "operation_id_matches": False,
        "market_data_execution_results_is_dict": False,
        "contract_qualification_execution_results_is_dict": False,
        "market_data_attempted_native_bool": False,
        "market_data_provider_called_native_bool": False,
        "market_data_success_native_bool": False,
        "contract_qualification_attempted_native_bool": False,
        "contract_qualification_provider_called_native_bool": False,
        "contract_qualification_success_native_bool": False,
        "at_least_one_provider_category_attempted": False,
    }


def _default_boundary_checks() -> dict[str, bool]:
    return {
        "no_provider_called_in_4k6": True,
        "no_direct_ib_call": False,
        "no_direct_reqMktData": False,
        "no_direct_qualifyContracts": False,
        "no_direct_reqContractDetails": False,
        "no_strategy_scan": False,
        "no_intents_created": False,
        "no_tickets_created": False,
        "no_orders_submitted": False,
        "no_broker_submission": False,
        "no_state_written": False,
        "no_ledger_written": False,
        "no_live_trading": False,
        "no_all_strategy_enablement": False,
        "no_scheduler_registration": False,
        "no_lifecycle_execution": False,
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
        "no_direct_market_data": False,
        "no_direct_contract_qualification": False,
        "no_order_submission": False,
        "no_intent_creation": False,
        "no_ticket_creation": False,
        "no_state_write": False,
        "no_ledger_write": False,
    }
