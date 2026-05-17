"""Pure Stage 4K-2 market data and contract qualification plan report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding"
)
PAPER_IBKR_PORTS = {4004}

STAGE4K1_BOUNDARY_FLAGS = (
    "no_market_data_fetched",
    "no_contracts_qualified",
    "no_strategy_scan",
    "no_intents_created",
    "no_tickets_created",
    "no_orders_submitted",
    "no_state_written",
    "no_ledger_written",
    "no_broker_submission",
    "no_live_trading",
    "no_all_strategy_enablement",
    "no_scheduler_registration",
    "no_lifecycle_execution",
)
STAGE4K1_SAFETY_FLAGS = (
    "no_live_trading",
    "no_all_strategy_enablement",
    "no_broker_submission_enabled",
    "no_market_data",
    "no_contract_qualification",
    "no_order_submission",
    "no_intent_creation",
    "no_ticket_creation",
    "no_state_write",
    "no_ledger_write",
)
PROPOSED_SCOPE_FALSE_FLAGS = (
    "may_fetch_market_data_now",
    "may_qualify_contracts_now",
    "may_submit_orders_now",
    "may_create_intents_now",
    "may_create_tickets_now",
    "may_write_state_now",
    "may_write_ledger_now",
    "live_trading_enabled",
    "all_strategies_enabled",
    "broker_submission_enabled",
)
MARKET_DATA_DISABLED_FLAGS = (
    "live_market_data_enabled",
    "streaming_market_data_enabled",
    "snapshot_market_data_enabled",
    "market_data_currently_enabled",
    "reqMktData_enabled",
)
CONTRACT_DISABLED_FLAGS = (
    "live_contract_qualification_enabled",
    "contract_qualification_currently_enabled",
    "qualifyContracts_enabled",
    "reqContractDetails_enabled",
)
REQUIREMENT_FIELDS = (
    "strategy_id",
    "selected_strategy_id",
    "symbol",
    "asset_class",
    "sec_type",
    "exchange",
    "currency",
    "trading_class",
    "expiry",
    "strike",
    "right",
    "paper_eligible",
    "market_data_required",
    "qualification_required",
)
FLOW_STEPS = (
    ("pre_execution_snapshot_check", "read_only_snapshot_inputs"),
    ("market_data_provider_gate_check", "controlled_market_data_provider"),
    ("contract_qualification_provider_gate_check", "controlled_contract_qualification_provider"),
    ("selected_strategy_contract_scope_check", "selected_strategy_contract_requirements"),
    ("market_data_plan_preview", "controlled_market_data_provider"),
    ("contract_qualification_plan_preview", "controlled_contract_qualification_provider"),
    ("post_plan_safety_check", "stage4k2_safety_boundary"),
)
ORDERED_NEXT_STEPS = [
    "Build Stage 4K-3 market data and contract qualification dry run.",
    "Keep live trading disabled.",
    "Keep all other strategies disabled.",
    "Keep broker submission separately gated until the explicit broker-submission phase.",
    "Keep intents, tickets, state writes, and ledger writes separately gated until their explicit phases.",
    "Do not fetch market data or qualify contracts until the controlled execution phase for this gate.",
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
    "Do not fetch market data now.",
    "Do not qualify contracts now.",
]


def build_stage4k2_market_data_contract_plan_report(
    *,
    stage4k1_readiness_report: dict | None,
    strategy_contract_requirements_snapshot: dict | list | None = None,
    market_data_capability_snapshot: dict | None = None,
    contract_qualification_capability_snapshot: dict | None = None,
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
    """Build a read-only Stage 4K-2 plan report from an accepted Stage 4K-1 report."""

    try:
        return _json_safe(
            _build_report(
                stage4k1_readiness_report=stage4k1_readiness_report,
                strategy_contract_requirements_snapshot=strategy_contract_requirements_snapshot,
                market_data_capability_snapshot=market_data_capability_snapshot,
                contract_qualification_capability_snapshot=contract_qualification_capability_snapshot,
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
        message = f"unexpected Stage 4K-2 plan failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_default_selected_strategy(),
                operation=_default_operation(),
                stage4k1_readiness_checks=_default_stage4k1_readiness_checks(),
                strategy_contract_requirements_checks=_default_requirements_checks(),
                market_data_readiness=_default_market_data_readiness(),
                contract_qualification_readiness=_default_contract_readiness(),
                market_data_plan=_market_data_plan(None, None, [], False),
                contract_qualification_plan=_contract_plan(None, None, [], False),
                proposed_operation_flow=_operation_flow(None, None, [], []),
                proposed_provider_payloads={"market_data_provider_payloads": [], "contract_qualification_provider_payloads": []},
                boundary_checks=_default_boundary_checks(),
                required_inputs_for_4k3=_required_inputs_for_4k3(),
                activation_snapshot_checks={},
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
    stage4k1_readiness_report: dict | None,
    strategy_contract_requirements_snapshot: Any,
    market_data_capability_snapshot: Any,
    contract_qualification_capability_snapshot: Any,
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
    data = stage4k1_readiness_report if isinstance(stage4k1_readiness_report, dict) else {}
    blockers: list[str] = []
    warnings: list[str] = []
    errors = _as_string_list(data.get("errors"))

    if stage4k1_readiness_report is None:
        blockers.append("Stage 4K-1 readiness report is missing")
    elif not isinstance(stage4k1_readiness_report, dict):
        blockers.append("Stage 4K-1 readiness report must be a dict")
        errors.append("Stage 4K-1 readiness report must be a dict")

    selected_strategy_id = _selected_strategy_id(data)
    operation_id = _operation_id(data)
    artifact_checks = _artifact_checks(stage4k1_readiness_report, data, selected_strategy_id, operation_id)
    blockers.extend(_artifact_blockers(artifact_checks, data))

    stage4k1_checks, stage4k1_blockers = _stage4k1_readiness_checks(data)
    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    operation, operation_blockers = _operation_checks(data, operation_id)
    boundary_checks, boundary_blockers = _boundary_checks(data)
    safety_checks, safety_blockers = _safety_checks(data)
    blockers.extend(stage4k1_blockers + selected_blockers + operation_blockers + boundary_blockers + safety_blockers)

    requirements_checks, requirements, req_blockers, req_warnings = _requirements_checks(
        strategy_contract_requirements_snapshot,
        selected_strategy_id,
    )
    market_data_readiness, md_blockers, md_warnings = _capability_checks(
        market_data_capability_snapshot,
        selected_strategy_id,
        capability_name="market_data",
        disabled_flags=MARKET_DATA_DISABLED_FLAGS,
        provider_key="market_data_provider_available",
        required_truthy_keys=("market_data_request_limit_configured", "symbol_universe_defined"),
        optional_not_false_key="paper_market_data_mode",
    )
    contract_readiness, cq_blockers, cq_warnings = _capability_checks(
        contract_qualification_capability_snapshot,
        selected_strategy_id,
        capability_name="contract_qualification",
        disabled_flags=CONTRACT_DISABLED_FLAGS,
        provider_key="contract_qualification_provider_available",
        required_truthy_keys=("contract_universe_defined",),
        optional_not_false_key=None,
    )
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

    all_blockers = [req_blockers, md_blockers, cq_blockers, activation_blockers, state_blockers, risk_blockers, scheduler_blockers, lifecycle_blockers, broker_blockers, market_blockers]
    for bl in all_blockers:
        blockers.extend(bl)

    all_warnings = [req_warnings, md_warnings, cq_warnings, activation_warnings, state_warnings, risk_warnings, scheduler_warnings, lifecycle_warnings, broker_warnings, market_warnings]
    for wl in all_warnings:
        warnings.extend(wl)

    pre_plan_ready = not _dedupe(blockers) and not _dedupe(errors)
    market_data_plan = _market_data_plan(selected_strategy_id, operation_id, requirements, pre_plan_ready)
    contract_plan = _contract_plan(selected_strategy_id, operation_id, requirements, pre_plan_ready)
    proposed_operation_flow = _operation_flow(selected_strategy_id, operation_id, market_data_plan["planned_requests"], contract_plan["planned_qualifications"])
    proposed_provider_payloads = _provider_payloads(market_data_plan["planned_requests"], contract_plan["planned_qualifications"], operation_id)
    plan_blockers = _plan_integrity_blockers(market_data_plan, contract_plan, proposed_operation_flow, proposed_provider_payloads)
    blockers.extend(plan_blockers)

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    ready = not blocker_list and not error_list
    if not ready:
        market_data_plan = _market_data_plan(selected_strategy_id, operation_id, requirements, False)
        contract_plan = _contract_plan(selected_strategy_id, operation_id, requirements, False)
        proposed_operation_flow = _operation_flow(selected_strategy_id, operation_id, market_data_plan["planned_requests"], contract_plan["planned_qualifications"])
        proposed_provider_payloads = _provider_payloads(market_data_plan["planned_requests"], contract_plan["planned_qualifications"], operation_id)

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
        operation=operation,
        stage4k1_readiness_checks=stage4k1_checks,
        strategy_contract_requirements_checks=requirements_checks,
        market_data_readiness=market_data_readiness,
        contract_qualification_readiness=contract_readiness,
        market_data_plan=market_data_plan,
        contract_qualification_plan=contract_plan,
        proposed_operation_flow=proposed_operation_flow,
        proposed_provider_payloads=proposed_provider_payloads,
        boundary_checks=boundary_checks,
        required_inputs_for_4k3=_required_inputs_for_4k3(),
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


def _artifact_checks(report: Any, data: dict[str, Any], selected_strategy_id: str | None, operation_id: str | None) -> dict[str, bool]:
    readiness = _mapping(data.get("readiness_for_stage4k2"))
    scope = _mapping(data.get("proposed_4k_scope"))
    return {
        "stage4k1_report_present": isinstance(report, dict),
        "stage4k1_report_ready": (
            data.get("stage4k1_market_data_contract_readiness_report") is True
            and readiness.get("ready_to_build_market_data_contract_plan") is True
            and data.get("success") is True
        ),
        "selected_strategy_present": isinstance(selected_strategy_id, str),
        "operation_id_present": isinstance(operation_id, str),
        "proposed_4k_scope_present": isinstance(data.get("proposed_4k_scope"), dict),
        "proposed_4k_scope_valid": _proposed_scope_valid(scope),
    }


def _artifact_blockers(checks: dict[str, bool], data: dict[str, Any]) -> list[str]:
    labels = {
        "stage4k1_report_present": "Stage 4K-1 readiness report is missing",
        "stage4k1_report_ready": "Stage 4K-1 readiness report is not ready for Stage 4K-2",
        "selected_strategy_present": "selected strategy is missing from Stage 4K-1 report",
        "operation_id_present": "operation_id is missing from Stage 4K-1 report",
        "proposed_4k_scope_present": "proposed_4k_scope is missing or malformed",
        "proposed_4k_scope_valid": "proposed_4k_scope does not preserve Stage 4K-2 safety boundaries",
    }
    blockers = [label for key, label in labels.items() if checks.get(key) is not True]
    if _as_list(data.get("errors")):
        blockers.append("Stage 4K-1 report contains errors")
    return blockers


def _stage4k1_readiness_checks(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    readiness = _mapping(data.get("readiness_for_stage4k2"))
    checks = {
        "stage4k1_report_true": data.get("stage4k1_market_data_contract_readiness_report") is True,
        "stage4k1_success": data.get("success") is True,
        "ready_to_build_market_data_contract_plan": readiness.get("ready_to_build_market_data_contract_plan") is True,
    }
    labels = {
        "stage4k1_report_true": "Stage 4K-1 report flag must be true",
        "stage4k1_success": "Stage 4K-1 report success must be true",
        "ready_to_build_market_data_contract_plan": "Stage 4K-1 must be ready to build the market data contract plan",
    }
    return checks, [label for key, label in labels.items() if checks.get(key) is not True]


def _proposed_scope_valid(scope: dict[str, Any]) -> bool:
    if not scope:
        return False
    if scope.get("may_build_market_data_contract_plan_next_phase") is not True:
        return False
    return all(scope.get(key) is False for key in PROPOSED_SCOPE_FALSE_FLAGS)


def _selected_strategy_checks(report: dict[str, Any], selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str]]:
    selected = _mapping(report.get("selected_strategy"))
    blockers: list[str] = []
    if not selected_strategy_id:
        blockers.append("selected_strategy.selected_strategy_id is missing or invalid")
    if selected.get("paper_only") is not True:
        blockers.append("selected strategy must be paper_only true")
    if selected.get("one_strategy_only") is not True:
        blockers.append("selected strategy must be one_strategy_only true")
    return {"selected_strategy_id": selected_strategy_id, "paper_only": True, "one_strategy_only": True}, blockers


def _operation_checks(report: dict[str, Any], operation_id: str | None) -> tuple[dict[str, Any], list[str]]:
    operation = _mapping(report.get("operation"))
    blockers: list[str] = []
    if not operation_id:
        blockers.append("operation.operation_id is missing or invalid")
    if operation.get("live_trading_enabled") is True:
        blockers.append("Stage 4K-1 operation shows live trading enabled")
    if operation.get("broker_submission_enabled") is True:
        blockers.append("Stage 4K-1 operation shows broker submission enabled")
    return {
        "operation_id": operation_id,
        "operation_scope": operation.get("operation_scope") or "single_strategy_market_data_contract_plan",
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }, blockers


def _boundary_checks(report: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    source = _mapping(report.get("boundary_checks"))
    checks = {key: source.get(key) is True for key in STAGE4K1_BOUNDARY_FLAGS}
    blockers = [
        f"Stage 4K-1 boundary check {key} must be strict native boolean true"
        for key in STAGE4K1_BOUNDARY_FLAGS
        if checks.get(key) is not True
    ]
    return checks, blockers


def _safety_checks(report: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    source = _mapping(report.get("safety_checks"))
    checks = {key: source.get(key) is True for key in STAGE4K1_SAFETY_FLAGS}
    blockers = [
        f"Stage 4K-1 safety check {key} must be strict native boolean true"
        for key in STAGE4K1_SAFETY_FLAGS
        if checks.get(key) is not True
    ]
    return checks, blockers


def _requirements_checks(snapshot: Any, selected_strategy_id: str | None) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[str]]:
    supplied = snapshot is not None
    blockers: list[str] = []
    warnings: list[str] = []
    raw_entries: list[Any] = []
    parseable = False
    if not supplied:
        return _default_requirements_checks(), [], blockers, ["strategy contract requirements snapshot missing; verify selected strategy contract scope before Stage 4K-3"]
    if isinstance(snapshot, dict):
        parseable = True
        for key in ("requirements", "contracts", "symbols"):
            if key in snapshot:
                value = snapshot.get(key)
                raw_entries.extend(value if isinstance(value, list) else [value])
        if not raw_entries:
            raw_entries.append(snapshot)
    elif isinstance(snapshot, list):
        parseable = True
        raw_entries.extend(snapshot)
    else:
        warnings.append("malformed strategy contract requirements snapshot ignored")
    normalized: list[dict[str, Any]] = []
    strategy_ids: set[str] = set()
    malformed = False
    explicit_contradiction = False
    for entry in raw_entries:
        requirement: dict[str, Any] | None = None
        if isinstance(entry, str):
            symbol = entry.strip()
            if symbol:
                requirement = _normalize_requirement({"symbol": symbol, "market_data_required": True, "qualification_required": True}, selected_strategy_id)
        elif isinstance(entry, dict):
            requirement = _normalize_requirement(entry, selected_strategy_id)
        else:
            malformed = True
        if not requirement:
            malformed = True
            continue
        entry_strategy = requirement.get("strategy_id") or requirement.get("selected_strategy_id")
        if isinstance(entry_strategy, str) and entry_strategy:
            strategy_ids.add(entry_strategy)
            if selected_strategy_id and entry_strategy != selected_strategy_id:
                explicit_contradiction = True
                blockers.append("strategy contract requirements include a mismatched strategy ID")
        if requirement.get("paper_eligible") is False:
            explicit_contradiction = True
            blockers.append("strategy contract requirements include paper_eligible false")
        normalized.append(requirement)
    if malformed:
        warnings.append("malformed strategy contract requirement entries ignored")
    if len(strategy_ids) > 1:
        blockers.append("multiple strategy IDs are present in contract requirements")
    required_count = sum(1 for item in normalized if item.get("market_data_required") is True or item.get("qualification_required") is True)
    if parseable and normalized and required_count == 0:
        blockers.append("strategy contract requirements contain no market data or qualification required items")
    if parseable and not normalized and not explicit_contradiction:
        warnings.append("strategy contract requirements snapshot parse yielded no usable requirements")
    normalized = sorted(normalized, key=_requirement_sort_key)
    return {
        "strategy_contract_requirements_snapshot_present": supplied,
        "strategy_contract_requirements_parseable": parseable,
        "selected_strategy_id": selected_strategy_id,
        "strategy_ids_seen": sorted(strategy_ids),
        "multiple_strategy_ids_present": len(strategy_ids) > 1,
        "normalized_requirement_count": len(normalized),
        "required_item_count": required_count,
        "requirements_safe_for_plan": not blockers,
    }, normalized, blockers, warnings


def _normalize_requirement(entry: dict[str, Any], selected_strategy_id: str | None) -> dict[str, Any] | None:
    symbol = entry.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    item = {key: entry.get(key) for key in REQUIREMENT_FIELDS if key in entry}
    item["symbol"] = symbol.strip()
    if "strategy_id" not in item and "selected_strategy_id" not in item and selected_strategy_id:
        item["selected_strategy_id"] = selected_strategy_id
    if "market_data_required" not in item and "qualification_required" not in item:
        item["market_data_required"] = True
        item["qualification_required"] = True
    if "sec_type" not in item or item.get("sec_type") is None:
        item["sec_type"] = entry.get("asset_class") or "STK"
    return item


def _requirement_sort_key(item: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        _sort_text(item.get("symbol")),
        _sort_text(item.get("sec_type")),
        _sort_text(item.get("expiry")),
        _sort_text(item.get("strike")),
        _sort_text(item.get("right")),
    )


def _capability_checks(
    snapshot: Any,
    selected_strategy_id: str | None,
    *,
    capability_name: str,
    disabled_flags: tuple[str, ...],
    provider_key: str,
    required_truthy_keys: tuple[str, ...],
    optional_not_false_key: str | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    supplied, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not supplied:
        warnings.append(f"{capability_name} capability snapshot missing; verify capability can be planned but remains disabled before Stage 4K-3")
        return _capability_result(False, capability_name, {}, selected_strategy_id, True), blockers, warnings
    merged = _merge_capability_layers(data)
    strategy_id = merged.get("selected_strategy_id")
    if isinstance(strategy_id, str) and selected_strategy_id and strategy_id != selected_strategy_id:
        blockers.append(f"{capability_name} capability selected_strategy_id does not match")
    provider = merged.get(provider_key)
    if provider is not None and provider is not True:
        blockers.append(f"{capability_name} capability {provider_key} must be native boolean true when supplied")
    if optional_not_false_key:
        mode_value = merged.get(optional_not_false_key)
        if mode_value is False or isinstance(mode_value, str):
            blockers.append(f"{capability_name} capability {optional_not_false_key} must not be false or stringified when supplied")
    for key in disabled_flags:
        value = merged.get(key)
        if isinstance(value, str):
            blockers.append(f"{capability_name} capability {key} must be a native boolean false when supplied")
        elif value is True:
            blockers.append(f"{capability_name} capability {key} must remain false")
    for key in required_truthy_keys:
        value = merged.get(key)
        if isinstance(value, str):
            blockers.append(f"{capability_name} capability {key} must be a native boolean true when supplied")
        elif value is False:
            blockers.append(f"{capability_name} capability {key} must not be false when supplied")
        elif value is None:
            warnings.append(f"{capability_name} capability {key} missing; verify required universe before Stage 4K-3")
    return _capability_result(True, capability_name, merged, selected_strategy_id, not blockers), blockers, warnings


def _merge_capability_layers(data: dict[str, Any]) -> dict[str, Any]:
    merged = dict(data)
    for key in ("capabilities", "config"):
        nested = _mapping(data.get(key))
        for nested_key, value in nested.items():
            merged.setdefault(nested_key, value)
    return merged


def _capability_result(present: bool, capability_name: str, data: dict[str, Any], selected_strategy_id: str | None, safe: bool) -> dict[str, Any]:
    return {
        f"{capability_name}_capability_snapshot_present": present,
        "selected_strategy_id": data.get("selected_strategy_id"),
        "selected_strategy_matches": data.get("selected_strategy_id") in (None, selected_strategy_id),
        "can_be_planned": safe,
        "currently_enabled": any(data.get(key) is True for key in MARKET_DATA_DISABLED_FLAGS + CONTRACT_DISABLED_FLAGS),
        "safe_to_plan_future_phase": safe,
    }


def _market_data_plan(selected_strategy_id: str | None, operation_id: str | None, requirements: list[dict[str, Any]], gates_pass: bool) -> dict[str, Any]:
    md_requirements = [item for item in requirements if item.get("market_data_required") is not False]
    symbols = sorted({_sort_text(item.get("symbol")) for item in md_requirements if _sort_text(item.get("symbol"))})
    planned = []
    for index, symbol in enumerate(symbols, start=1):
        payload = {
            "selected_strategy_id": selected_strategy_id,
            "operation_id": operation_id,
            "symbol": symbol,
            "data_type": "snapshot_plan",
            "allow_live_trading": False,
            "allow_broker_submission": False,
            "allow_order_submission": False,
            "allow_state_write": False,
            "allow_ledger_write": False,
        }
        planned.append(
            {
                "request_id": f"md-{index:03d}-{symbol}",
                "selected_strategy_id": selected_strategy_id,
                "symbol": symbol,
                "data_type": "snapshot_plan",
                "provider_target": "controlled_market_data_provider",
                "target_function": "request_controlled_market_data",
                "payload": payload,
                "would_execute_now": False,
                "paper_only": True,
                "live_trading_enabled": False,
                "broker_submission_enabled": False,
            }
        )
    return {
        "available": gates_pass,
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "plan_scope": "single_strategy_market_data_plan",
        "paper_only": True,
        "may_fetch_market_data_in_4k2": False,
        "may_fetch_market_data_in_4k3": False,
        "may_fetch_market_data_in_4k4": False,
        "may_fetch_market_data_in_4k5": gates_pass,
        "allowed_provider_method_name": "request_controlled_market_data",
        "disallowed_methods": ["reqMktData", "direct_ib_reqMktData", "yfinance", "requests", "urllib"],
        "allowed_data_mode": "controlled_paper_provider_only",
        "request_limit_required": True,
        "selected_strategy_only": True,
        "normalized_symbols": symbols,
        "planned_requests": planned,
    }


def _contract_plan(selected_strategy_id: str | None, operation_id: str | None, requirements: list[dict[str, Any]], gates_pass: bool) -> dict[str, Any]:
    cq_requirements = [item for item in requirements if item.get("qualification_required") is not False]
    normalized = sorted(cq_requirements, key=_requirement_sort_key)
    planned = []
    for index, item in enumerate(normalized, start=1):
        symbol = _sort_text(item.get("symbol"))
        sec_type = _sort_text(item.get("sec_type")) or "STK"
        payload = _contract_payload(item, selected_strategy_id, operation_id)
        planned.append(
            {
                "qualification_id": f"cq-{index:03d}-{symbol}-{sec_type}",
                "selected_strategy_id": selected_strategy_id,
                "symbol": symbol,
                "sec_type": sec_type,
                "provider_target": "controlled_contract_qualification_provider",
                "target_function": "qualify_controlled_contracts",
                "payload": payload,
                "would_execute_now": False,
                "paper_only": True,
                "live_trading_enabled": False,
                "broker_submission_enabled": False,
            }
        )
    return {
        "available": gates_pass,
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "plan_scope": "single_strategy_contract_qualification_plan",
        "paper_only": True,
        "may_qualify_contracts_in_4k2": False,
        "may_qualify_contracts_in_4k3": False,
        "may_qualify_contracts_in_4k4": False,
        "may_qualify_contracts_in_4k5": gates_pass,
        "allowed_provider_method_name": "qualify_controlled_contracts",
        "disallowed_methods": ["qualifyContracts", "reqContractDetails", "direct_ib_qualifyContracts"],
        "allowed_qualification_mode": "controlled_paper_provider_only",
        "selected_strategy_only": True,
        "normalized_contract_requirements": normalized,
        "planned_qualifications": planned,
    }


def _contract_payload(item: dict[str, Any], selected_strategy_id: str | None, operation_id: str | None) -> dict[str, Any]:
    payload = {
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "symbol": item.get("symbol"),
        "sec_type": item.get("sec_type") or "STK",
        "allow_live_trading": False,
        "allow_broker_submission": False,
        "allow_order_submission": False,
        "allow_state_write": False,
        "allow_ledger_write": False,
    }
    for key in ("asset_class", "exchange", "currency", "trading_class", "expiry", "strike", "right"):
        if item.get(key) is not None:
            payload[key] = item.get(key)
    return payload


def _operation_flow(selected_strategy_id: str | None, operation_id: str | None, planned_requests: list[dict[str, Any]], planned_qualifications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flow = []
    for index, (step_name, target_component) in enumerate(FLOW_STEPS, start=1):
        flow.append(
            {
                "sequence_number": index,
                "step_name": step_name,
                "target_component": target_component,
                "payload": {
                    "selected_strategy_id": selected_strategy_id,
                    "operation_id": operation_id,
                    "planned_market_data_request_count": len(planned_requests),
                    "planned_contract_qualification_count": len(planned_qualifications),
                    "preview_only": True,
                },
                "would_execute_now": False,
                "would_fetch_market_data_now": False,
                "would_qualify_contracts_now": False,
                "would_call_strategy_now": False,
                "would_create_intents_now": False,
                "would_create_tickets_now": False,
                "would_submit_orders_now": False,
                "would_write_state_now": False,
                "would_write_ledger_now": False,
                "paper_only": True,
                "live_trading_enabled": False,
                "broker_submission_enabled": False,
            }
        )
    return flow


def _provider_payloads(planned_requests: list[dict[str, Any]], planned_qualifications: list[dict[str, Any]], operation_id: str | None) -> dict[str, list[dict[str, Any]]]:
    md_payloads = [_future_provider_payload(item["payload"], operation_id) for item in planned_requests if isinstance(item.get("payload"), dict)]
    cq_payloads = [_future_provider_payload(item["payload"], operation_id) for item in planned_qualifications if isinstance(item.get("payload"), dict)]
    return {
        "market_data_provider_payloads": md_payloads,
        "contract_qualification_provider_payloads": cq_payloads,
    }


def _future_provider_payload(payload: dict[str, Any], operation_id: str | None) -> dict[str, Any]:
    result = dict(payload)
    result["operation_id"] = result.get("operation_id") or operation_id
    result["allow_live_trading"] = False
    result["allow_broker_submission"] = False
    result["allow_order_submission"] = False
    result["allow_state_write"] = False
    result["allow_ledger_write"] = False
    return result


def _plan_integrity_blockers(
    market_data_plan: dict[str, Any],
    contract_plan: dict[str, Any],
    flow: list[dict[str, Any]],
    payloads: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if market_data_plan.get("available") is not True:
        blockers.append("market data plan is not available")
    if contract_plan.get("available") is not True:
        blockers.append("contract qualification plan is not available")
    for item in _as_list(market_data_plan.get("planned_requests")):
        if not isinstance(item, dict) or not isinstance(item.get("payload"), dict):
            blockers.append("planned market data requests must contain native dict payloads")
        if _unsafe_plan_item(item):
            blockers.append("planned market data request enables execution or unsafe permissions")
    for item in _as_list(contract_plan.get("planned_qualifications")):
        if not isinstance(item, dict) or not isinstance(item.get("payload"), dict):
            blockers.append("planned contract qualifications must contain native dict payloads")
        if _unsafe_plan_item(item):
            blockers.append("planned contract qualification enables execution or unsafe permissions")
    expected_steps = [name for name, _target in FLOW_STEPS]
    if [item.get("step_name") for item in flow if isinstance(item, dict)] != expected_steps:
        blockers.append("proposed operation flow is missing required ordered steps")
    for item in flow:
        if not isinstance(item, dict) or not isinstance(item.get("payload"), dict):
            blockers.append("proposed operation flow payloads must be native dicts")
            continue
        for key in (
            "would_execute_now",
            "would_fetch_market_data_now",
            "would_qualify_contracts_now",
            "would_call_strategy_now",
            "would_create_intents_now",
            "would_create_tickets_now",
            "would_submit_orders_now",
            "would_write_state_now",
            "would_write_ledger_now",
            "live_trading_enabled",
            "broker_submission_enabled",
        ):
            if item.get(key) is not False:
                blockers.append(f"proposed operation flow {key} must remain false")
    for key in ("market_data_provider_payloads", "contract_qualification_provider_payloads"):
        for payload in _as_list(payloads.get(key)):
            if not isinstance(payload, dict):
                blockers.append("proposed provider payloads must be native dicts")
                continue
            for permission in ("allow_live_trading", "allow_broker_submission", "allow_order_submission", "allow_state_write", "allow_ledger_write"):
                if payload.get(permission) is not False:
                    blockers.append(f"proposed provider payload {permission} must remain false")
    return _dedupe(blockers)


def _unsafe_plan_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return True
    return any(item.get(key) is not False for key in ("would_execute_now", "live_trading_enabled", "broker_submission_enabled"))


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
    for prefix, snapshot, record_key, list_key, execution_key in (
        ("scheduler_activation", scheduler_activation_snapshot, "scheduler_activation_record", "scheduler_activations", "strategy_scan_execution_enabled"),
        ("lifecycle_activation", lifecycle_activation_snapshot, "lifecycle_activation_record", "lifecycle_activations", "lifecycle_transition_execution_enabled"),
        ("activation", activation_snapshot, "activation_record", "activations", None),
    ):
        current_checks, current_blockers, current_warnings = _activation_snapshot_checks(snapshot, selected_strategy_id, prefix, record_key, list_key, execution_key)
        for key, value in current_checks.items():
            checks[key] = value
        blockers.extend(current_blockers)
        warnings.extend(current_warnings)
    return checks, blockers, warnings


def _activation_snapshot_checks(snapshot: Any, selected_strategy_id: str | None, prefix: str, record_key: str, list_key: str, execution_key: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append(f"{prefix} snapshot missing; verify activation artifact before Stage 4K-3")
        return {f"{prefix}_snapshot_present": False, f"{prefix}_snapshot_matches": True}, blockers, warnings
    for record in _explicit_records(data, record_key, list_key):
        if not isinstance(record, dict):
            warnings.append(f"malformed {prefix} snapshot entry ignored")
            continue
        if record.get("selected_strategy_id") not in (None, selected_strategy_id):
            blockers.append(f"{prefix} snapshot selected_strategy_id does not match")
        for key in ("live_trading_enabled", "all_strategies_enabled", "broker_submission_enabled"):
            if record.get(key) is True:
                blockers.append(f"{prefix} snapshot {key} contradicts Stage 4K-2 safety")
        if execution_key and record.get(execution_key) is True:
            blockers.append(f"{prefix} snapshot {execution_key} contradicts Stage 4K-2 safety")
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
        warnings.append("state snapshot missing; verify halt, reconciliation, intents, and positions before Stage 4K-3")
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
        warnings.append("risk snapshot missing; verify kill switch, hard halt, and daily loss controls before Stage 4K-3")
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
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("scheduler snapshot missing; verify scheduler state before Stage 4K-3")
    broad_enabled = any(data.get(key) is True for key in ("all_strategy_scheduler_enabled", "all_strategies_enabled", "broad_scheduler_enabled"))
    scan_enabled = data.get("strategy_scan_execution_enabled") is True
    selected_job_matches = True
    for job in jobs:
        if not isinstance(job, dict):
            continue
        job_strategy = _first_present(job.get("selected_strategy_id"), job.get("strategy_id"), default=None)
        if job_strategy == selected_strategy_id:
            selected_job_matches = selected_job_matches and _scheduler_job_safe(job, selected_strategy_id)
        elif job.get("active") is True or job.get("enabled") is True:
            blockers.append("scheduler snapshot contains active job for another strategy")
    if broad_enabled:
        blockers.append("broad/all-strategy scheduler automation enabled")
    if scan_enabled:
        blockers.append("scheduler strategy scan execution is enabled")
    if jobs and not selected_job_matches:
        blockers.append("scheduler selected strategy job is not within Stage 4K-2 safety constraints")
    return {
        "scheduler_snapshot_present": present,
        "all_strategy_scheduler_enabled": broad_enabled,
        "selected_strategy_job_matches": selected_job_matches,
        "strategy_scan_execution_enabled": scan_enabled,
    }, blockers, warnings


def _scheduler_job_safe(job: dict[str, Any], selected_strategy_id: str | None) -> bool:
    return (
        _first_present(job.get("selected_strategy_id"), job.get("strategy_id"), default=None) == selected_strategy_id
        and job.get("broker_submission_enabled") is False
        and job.get("live_trading_enabled") is False
        and job.get("all_strategies_enabled") is False
        and job.get("strategy_scan_execution_enabled") is False
    )


def _lifecycle_checks(snapshot: Any, selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; verify lifecycle state before Stage 4K-3")
    broad_enabled = any(data.get(key) is True for key in ("all_strategy_lifecycle_enabled", "all_strategies_enabled", "broad_lifecycle_enabled"))
    transition_enabled = data.get("lifecycle_transition_execution_enabled") is True
    lifecycle_enabled = data.get("lifecycle_automation_enabled") is True
    matches = True
    if present and lifecycle_enabled:
        matches = (
            data.get("selected_strategy_id") == selected_strategy_id
            and data.get("broker_submission_enabled") is False
            and data.get("live_trading_enabled") is False
            and data.get("all_strategies_enabled") is False
            and transition_enabled is False
            and broad_enabled is False
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
        warnings.append("paper broker snapshot missing; verify PAPER broker config before Stage 4K-3")
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
        blockers.append("market window snapshot explicitly disallows plan validation")
    if market_open is False:
        warnings.append("market is currently closed; plan validation remains report-only")
    if is_trading_day is False:
        warnings.append("snapshot is not a trading day; plan validation remains report-only")
    return {
        "market_window_snapshot_present": present,
        "allowed_to_schedule_paper_run": allowed,
        "is_trading_day": is_trading_day,
        "market_open": market_open,
        "reason": data.get("reason") if present else None,
    }, blockers, warnings


def _required_inputs_for_4k3() -> list[str]:
    return [
        "accepted Stage 4K-2 market data and contract qualification plan report",
        "controlled market data provider dry-run interface that does not call direct IBKR market data methods",
        "controlled contract qualification provider dry-run interface that does not call direct IBKR qualification methods",
        "native JSON-safe provider payloads for the selected strategy only",
        "fresh state, risk, scheduler, lifecycle, paper broker, and market window snapshots",
        "operator confirmation that broker submission, intents, tickets, state writes, ledger writes, live trading, and all-strategy automation remain disabled",
    ]


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    operation: dict[str, Any],
    stage4k1_readiness_checks: dict[str, Any],
    strategy_contract_requirements_checks: dict[str, Any],
    market_data_readiness: dict[str, Any],
    contract_qualification_readiness: dict[str, Any],
    market_data_plan: dict[str, Any],
    contract_qualification_plan: dict[str, Any],
    proposed_operation_flow: list[dict[str, Any]],
    proposed_provider_payloads: dict[str, Any],
    boundary_checks: dict[str, bool],
    required_inputs_for_4k3: list[str],
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
        "stage4k2_market_data_contract_plan_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "operation": operation,
        "stage4k1_readiness_checks": stage4k1_readiness_checks,
        "strategy_contract_requirements_checks": strategy_contract_requirements_checks,
        "market_data_readiness": market_data_readiness,
        "contract_qualification_readiness": contract_qualification_readiness,
        "market_data_plan": market_data_plan,
        "contract_qualification_plan": contract_qualification_plan,
        "proposed_operation_flow": proposed_operation_flow,
        "proposed_provider_payloads": proposed_provider_payloads,
        "boundary_checks": boundary_checks,
        "required_inputs_for_4k3": required_inputs_for_4k3,
        "activation_snapshot_checks": activation_snapshot_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4k3": {
            "ready_to_build_market_data_contract_dry_run": ready,
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
    if any(key in data for key in ("selected_strategy_id", "live_trading_enabled", "all_strategies_enabled", "broker_submission_enabled")):
        records.append(data)
    return records


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_mapping(value: Any) -> tuple[bool, dict[str, Any]]:
    return isinstance(value, dict), value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_string_list(value: Any) -> list[str]:
    return [item.strip() for item in _as_list(value) if isinstance(item, str) and item.strip()]


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


def _sort_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ("" if value is None else str(value))


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


def _default_artifact_checks() -> dict[str, bool]:
    return {
        "stage4k1_report_present": False,
        "stage4k1_report_ready": False,
        "selected_strategy_present": False,
        "operation_id_present": False,
        "proposed_4k_scope_present": False,
        "proposed_4k_scope_valid": False,
    }


def _default_selected_strategy() -> dict[str, Any]:
    return {"selected_strategy_id": None, "paper_only": True, "one_strategy_only": True}


def _default_operation() -> dict[str, Any]:
    return {
        "operation_id": None,
        "operation_scope": "single_strategy_market_data_contract_plan",
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }


def _default_stage4k1_readiness_checks() -> dict[str, Any]:
    return {
        "stage4k1_report_true": False,
        "stage4k1_success": False,
        "ready_to_build_market_data_contract_plan": False,
    }


def _default_requirements_checks() -> dict[str, Any]:
    return {
        "strategy_contract_requirements_snapshot_present": False,
        "strategy_contract_requirements_parseable": False,
        "selected_strategy_id": None,
        "strategy_ids_seen": [],
        "multiple_strategy_ids_present": False,
        "normalized_requirement_count": 0,
        "required_item_count": 0,
        "requirements_safe_for_plan": True,
    }


def _default_market_data_readiness() -> dict[str, Any]:
    return _capability_result(False, "market_data", {}, None, True)


def _default_contract_readiness() -> dict[str, Any]:
    return _capability_result(False, "contract_qualification", {}, None, True)


def _default_boundary_checks() -> dict[str, bool]:
    return {key: False for key in STAGE4K1_BOUNDARY_FLAGS}


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
    return {key: False for key in STAGE4K1_SAFETY_FLAGS}
