"""Pure Stage 4K-1 market data and contract qualification readiness report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
RECOMMENDED_NEXT_GATE = "stage4k_market_data_and_contract_qualification_gate"
MARKET_WINDOW_MANUAL_WARNING = (
    "market window snapshot missing; operator must manually verify exchange hours and holiday schedules before proceeding"
)
PAPER_IBKR_PORTS = {4004}

FOUR_J_BOUNDARY_TRUE_FLAGS = (
    "no_market_data_requested",
    "no_contracts_qualified",
    "no_intents_created",
    "no_tickets_created",
    "no_orders_submitted",
    "no_state_written",
    "no_ledger_written",
    "no_live_trading",
    "no_all_strategy_enablement",
    "no_broker_submission",
)
FOUR_J_SAFETY_TRUE_FLAGS = (
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
ORDERED_NEXT_STEPS = [
    "Build Stage 4K-2 market data and contract qualification plan.",
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


def build_stage4k1_market_data_contract_readiness_report(
    *,
    stage4j6_acceptance_report: dict | None,
    strategy_registry_snapshot: dict | list | None = None,
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
    """Build a read-only Stage 4K-1 readiness report from an accepted Stage 4J-6 report."""

    try:
        return _json_safe(
            _build_report(
                stage4j6_acceptance_report=stage4j6_acceptance_report,
                strategy_registry_snapshot=strategy_registry_snapshot,
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
        message = f"unexpected Stage 4K-1 readiness failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks=_default_artifact_checks(),
                selected_strategy=_default_selected_strategy(),
                operation=_default_operation(),
                stage4j_completion_checks=_default_stage4j_completion_checks(),
                strategy_registry_checks=_default_strategy_registry_checks(),
                market_data_readiness=_default_market_data_readiness(),
                contract_qualification_readiness=_default_contract_readiness(),
                proposed_4k_scope=_proposed_scope(None, None, False),
                boundary_checks=_default_boundary_checks(),
                required_inputs_for_4k2=_required_inputs_for_4k2(),
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
    stage4j6_acceptance_report: dict | None,
    strategy_registry_snapshot: dict | list | None,
    market_data_capability_snapshot: dict | None,
    contract_qualification_capability_snapshot: dict | None,
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
    data = stage4j6_acceptance_report if isinstance(stage4j6_acceptance_report, dict) else {}
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4j6_acceptance_report is None:
        blockers.append("Stage 4J-6 acceptance report is missing")
    elif not isinstance(stage4j6_acceptance_report, dict):
        blockers.append("Stage 4J-6 acceptance report must be a dict")
        errors.append("Stage 4J-6 acceptance report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    selected_strategy_id = _selected_strategy_id(data)
    operation_id = _operation_id(data)
    artifact_checks = _artifact_checks(stage4j6_acceptance_report, data, selected_strategy_id, operation_id)
    blockers.extend(_artifact_blockers(artifact_checks, data))

    stage4j_checks, stage4j_blockers = _stage4j_completion_checks(data)
    blockers.extend(stage4j_blockers)
    selected_strategy, selected_blockers = _selected_strategy_checks(data, selected_strategy_id)
    operation, operation_blockers = _operation_checks(data, operation_id)
    blockers.extend(selected_blockers + operation_blockers)

    boundary_checks, boundary_blockers = _boundary_checks(data)
    safety_checks, safety_blockers = _stage4j_safety_checks(data)
    blockers.extend(boundary_blockers + safety_blockers)

    registry_checks, registry_blockers, registry_warnings = _strategy_registry_checks(strategy_registry_snapshot, selected_strategy_id)
    market_data_readiness, market_data_blockers, market_data_warnings = _capability_checks(
        market_data_capability_snapshot,
        selected_strategy_id,
        capability_name="market_data",
        disabled_flags=MARKET_DATA_DISABLED_FLAGS,
        provider_key="market_data_provider_available",
        mode_key="paper_market_data_mode",
        required_truthy_keys=("market_data_request_limit_configured", "symbol_universe_defined"),
        universe_label="symbol universe",
    )
    contract_readiness, contract_blockers, contract_warnings = _capability_checks(
        contract_qualification_capability_snapshot,
        selected_strategy_id,
        capability_name="contract_qualification",
        disabled_flags=CONTRACT_DISABLED_FLAGS,
        provider_key="contract_qualification_provider_available",
        mode_key=None,
        required_truthy_keys=("contract_universe_defined",),
        universe_label="contract universe",
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
    blockers.extend(
        registry_blockers
        + market_data_blockers
        + contract_blockers
        + activation_blockers
        + state_blockers
        + risk_blockers
        + scheduler_blockers
        + lifecycle_blockers
        + broker_blockers
        + market_blockers
    )
    warnings.extend(
        registry_warnings
        + market_data_warnings
        + contract_warnings
        + activation_warnings
        + state_warnings
        + risk_warnings
        + scheduler_warnings
        + lifecycle_warnings
        + broker_warnings
        + market_warnings
    )

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    ready = not blocker_list and not error_list
    proposed_scope = _proposed_scope(selected_strategy_id, operation_id, ready)
    if ready and proposed_scope["may_build_market_data_contract_plan_next_phase"] is not True:
        blocker_list = _dedupe(blocker_list + ["proposed 4K scope does not permit building the next plan"])
        ready = False

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        selected_strategy=selected_strategy,
        operation=operation,
        stage4j_completion_checks=stage4j_checks,
        strategy_registry_checks=registry_checks,
        market_data_readiness=market_data_readiness,
        contract_qualification_readiness=contract_readiness,
        proposed_4k_scope=proposed_scope,
        boundary_checks=boundary_checks,
        required_inputs_for_4k2=_required_inputs_for_4k2(),
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
    readiness = _mapping(data.get("readiness_for_stage4j_complete_or_next_gate"))
    return {
        "stage4j6_report_present": isinstance(report, dict),
        "stage4j6_report_ready": (
            data.get("stage4j6_controlled_paper_operation_acceptance_report") is True
            and data.get("success") is True
        ),
        "stage4j_complete": readiness.get("stage4j_complete") is True,
        "selected_strategy_present": isinstance(selected_strategy_id, str),
        "operation_id_present": isinstance(operation_id, str),
        "recommended_next_gate_matches": readiness.get("recommended_next_gate") == RECOMMENDED_NEXT_GATE,
    }


def _artifact_blockers(checks: dict[str, bool], data: dict[str, Any]) -> list[str]:
    labels = {
        "stage4j6_report_present": "Stage 4J-6 acceptance report is missing",
        "stage4j6_report_ready": "Stage 4J-6 acceptance report is not ready",
        "stage4j_complete": "Stage 4J is not complete",
        "selected_strategy_present": "selected strategy is missing from Stage 4J-6 report",
        "operation_id_present": "operation_id is missing from Stage 4J-6 report",
        "recommended_next_gate_matches": "Stage 4J-6 recommended next gate is not Stage 4K",
    }
    blockers = [label for key, label in labels.items() if checks.get(key) is not True]
    if _as_list(data.get("errors")):
        blockers.append("Stage 4J-6 report contains errors")
    return blockers


def _stage4j_completion_checks(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    readiness = _mapping(data.get("readiness_for_stage4j_complete_or_next_gate"))
    acceptance = _mapping(data.get("executor_acceptance"))
    checks = {
        "stage4j6_acceptance_report_true": data.get("stage4j6_controlled_paper_operation_acceptance_report") is True,
        "stage4j_complete": readiness.get("stage4j_complete") is True,
        "ready_for_next_explicit_gate": readiness.get("ready_for_next_explicit_gate") is True,
        "recommended_next_gate_matches": readiness.get("recommended_next_gate") == RECOMMENDED_NEXT_GATE,
        "executor_acceptance_accepted": acceptance.get("accepted") is True,
    }
    labels = {
        "stage4j6_acceptance_report_true": "Stage 4J-6 report flag must be true",
        "stage4j_complete": "Stage 4J completion must be true",
        "ready_for_next_explicit_gate": "Stage 4J-6 must be ready for the next explicit gate",
        "recommended_next_gate_matches": "Stage 4J-6 must recommend the Stage 4K gate",
        "executor_acceptance_accepted": "Stage 4J-6 executor acceptance must be accepted",
    }
    return checks, [label for key, label in labels.items() if checks.get(key) is not True]


def _selected_strategy_checks(report: dict[str, Any], selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str]]:
    selected = _mapping(report.get("selected_strategy"))
    paper_only = selected.get("paper_only") is True
    one_strategy_only = selected.get("one_strategy_only") is True
    blockers: list[str] = []
    if not selected_strategy_id:
        blockers.append("selected_strategy.selected_strategy_id is missing or invalid")
    if not paper_only:
        blockers.append("selected strategy must be paper_only true")
    if not one_strategy_only:
        blockers.append("selected strategy must be one_strategy_only true")
    return {"selected_strategy_id": selected_strategy_id, "paper_only": True, "one_strategy_only": True}, blockers


def _operation_checks(report: dict[str, Any], operation_id: str | None) -> tuple[dict[str, Any], list[str]]:
    operation = _mapping(report.get("operation"))
    blockers: list[str] = []
    if not operation_id:
        blockers.append("operation.operation_id is missing or invalid")
    if operation.get("live_trading_enabled") is True:
        blockers.append("Stage 4J-6 operation shows live trading enabled")
    if operation.get("broker_submission_enabled") is True:
        blockers.append("Stage 4J-6 operation shows broker submission enabled")
    return {
        "operation_id": operation_id,
        "operation_scope": operation.get("operation_scope") or "single_strategy_market_data_contract_readiness",
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }, blockers


def _boundary_checks(report: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    source = _mapping(report.get("boundary_checks"))
    checks = {
        "no_market_data_fetched": source.get("no_market_data_requested") is True,
        "no_contracts_qualified": source.get("no_contracts_qualified") is True,
        "no_strategy_scan": source.get("no_direct_strategy_scan", True) is True,
        "no_intents_created": source.get("no_intents_created") is True,
        "no_tickets_created": source.get("no_tickets_created") is True,
        "no_orders_submitted": source.get("no_orders_submitted") is True,
        "no_state_written": source.get("no_state_written") is True,
        "no_ledger_written": source.get("no_ledger_written") is True,
        "no_broker_submission": source.get("no_broker_submission") is True,
        "no_live_trading": source.get("no_live_trading") is True,
        "no_all_strategy_enablement": source.get("no_all_strategy_enablement") is True,
        "no_scheduler_registration": source.get("no_scheduler_registration", True) is True,
        "no_lifecycle_execution": source.get("no_lifecycle_execution", True) is True,
    }
    blockers: list[str] = []
    for flag in FOUR_J_BOUNDARY_TRUE_FLAGS:
        if source.get(flag) is not True:
            blockers.append(f"Stage 4J-6 boundary check {flag} must be strict native boolean true")
    return checks, blockers


def _stage4j_safety_checks(report: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    source = _mapping(report.get("safety_checks"))
    checks = {key: source.get(key) is True for key in FOUR_J_SAFETY_TRUE_FLAGS}
    blockers = [
        f"Stage 4J-6 safety check {key} must be strict native boolean true"
        for key in FOUR_J_SAFETY_TRUE_FLAGS
        if checks.get(key) is not True
    ]
    return checks, blockers


def _strategy_registry_checks(snapshot: Any, selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    supplied = snapshot is not None
    blockers: list[str] = []
    warnings: list[str] = []
    entries: list[Any] = []
    parseable = False
    if not supplied:
        return _default_strategy_registry_checks(), blockers, ["strategy registry snapshot missing; verify selected strategy remains paper eligible before Stage 4K-2"]
    if isinstance(snapshot, dict):
        parseable = True
        raw = snapshot.get("strategies")
        if isinstance(raw, list):
            entries.extend(raw)
        else:
            entries.append(snapshot)
        entries.extend(_as_list(snapshot.get("active_strategies")))
        entries.extend(_as_list(snapshot.get("selected_strategies")))
    elif isinstance(snapshot, list):
        parseable = True
        entries.extend(snapshot)
    else:
        warnings.append("malformed strategy registry snapshot ignored")
    ids: set[str] = set()
    selected_present = False
    paper_eligible = None
    active_or_selected_ids: set[str] = set()
    malformed = False
    for entry in entries:
        if isinstance(entry, str):
            strategy_id = entry.strip()
            if strategy_id:
                ids.add(strategy_id)
                selected_present = selected_present or strategy_id == selected_strategy_id
            continue
        if not isinstance(entry, dict):
            malformed = True
            continue
        strategy_id = _strategy_id_from_entry(entry)
        if strategy_id:
            ids.add(strategy_id)
            selected_present = selected_present or strategy_id == selected_strategy_id
            if entry.get("active") is True or entry.get("selected") is True or entry.get("enabled") is True:
                active_or_selected_ids.add(strategy_id)
            if strategy_id == selected_strategy_id and "paper_eligible" in entry:
                paper_eligible = entry.get("paper_eligible")
        else:
            malformed = True
    if isinstance(snapshot, dict):
        for key in ("active_strategy_ids", "selected_strategy_ids"):
            for item in _as_string_list(snapshot.get(key)):
                active_or_selected_ids.add(item)
        for key in ("active_strategy_id", "selected_strategy_id"):
            value = snapshot.get(key)
            if isinstance(value, str) and value.strip():
                active_or_selected_ids.add(value.strip())
    if malformed:
        warnings.append("malformed strategy registry entries ignored")
    if parseable and ids and selected_strategy_id not in ids:
        blockers.append("strategy registry snapshot does not contain selected_strategy_id")
    elif parseable and not ids:
        warnings.append("strategy registry snapshot parse yielded no strategy ids")
    if paper_eligible is False:
        blockers.append("strategy registry marks selected strategy paper_eligible false")
    if len(active_or_selected_ids) > 1:
        blockers.append("multiple active/selected strategy IDs detected")
    return {
        "strategy_registry_snapshot_present": supplied,
        "strategy_registry_parseable": parseable,
        "selected_strategy_id_present": selected_present,
        "selected_strategy_paper_eligible": paper_eligible,
        "multiple_active_or_selected_strategies": len(active_or_selected_ids) > 1,
        "strategy_ids_seen": sorted(ids),
    }, blockers, warnings


def _strategy_id_from_entry(entry: dict[str, Any]) -> str | None:
    for key in ("strategy_id", "selected_strategy_id", "id"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _capability_checks(
    snapshot: Any,
    selected_strategy_id: str | None,
    *,
    capability_name: str,
    disabled_flags: tuple[str, ...],
    provider_key: str,
    mode_key: str | None,
    required_truthy_keys: tuple[str, ...],
    universe_label: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    supplied, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not supplied:
        warnings.append(f"{capability_name} capability snapshot missing; verify capability can be planned but remains disabled before Stage 4K-2")
        return _capability_result(False, capability_name, {}, selected_strategy_id, True), blockers, warnings
    merged = _merge_capability_layers(data)
    strategy_id = merged.get("selected_strategy_id")
    if isinstance(strategy_id, str) and selected_strategy_id and strategy_id != selected_strategy_id:
        blockers.append(f"{capability_name} capability selected_strategy_id does not match")
    provider = merged.get(provider_key)
    if provider is not None and provider is not True:
        blockers.append(f"{capability_name} capability {provider_key} must be true when supplied")
    if mode_key:
        mode_value = merged.get(mode_key)
        if mode_value is False:
            blockers.append(f"{capability_name} capability {mode_key} must not be false when supplied")
    for key in disabled_flags:
        value = merged.get(key)
        if isinstance(value, str):
            blockers.append(f"{capability_name} capability {key} must be a native boolean false when supplied")
        elif value is True:
            blockers.append(f"{capability_name} capability {key} must remain false")
    for key in required_truthy_keys:
        value = merged.get(key)
        if value is False:
            blockers.append(f"{capability_name} capability {key} must not be false when supplied")
        elif value is None:
            warnings.append(f"{capability_name} capability {key} missing; verify {universe_label} before Stage 4K-2")
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
        "safe_to_plan_next_phase": safe,
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


def _activation_snapshot_checks(snapshot: dict | None, selected_strategy_id: str | None, prefix: str, record_key: str, list_key: str, execution_key: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append(f"{prefix} snapshot missing; verify activation artifact before Stage 4K-2")
        return {f"{prefix}_snapshot_present": False, f"{prefix}_snapshot_matches": True}, blockers, warnings
    for record in _explicit_records(data, record_key, list_key):
        if not isinstance(record, dict):
            warnings.append(f"malformed {prefix} snapshot entry ignored")
            continue
        if record.get("selected_strategy_id") not in (None, selected_strategy_id):
            blockers.append(f"{prefix} snapshot selected_strategy_id does not match")
        for key in ("live_trading_enabled", "all_strategies_enabled", "broker_submission_enabled"):
            if record.get(key) is True:
                blockers.append(f"{prefix} snapshot {key} contradicts Stage 4K-1 safety")
        if execution_key and record.get(execution_key) is True:
            blockers.append(f"{prefix} snapshot {execution_key} contradicts Stage 4K-1 safety")
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
        warnings.append("state snapshot missing; verify halt, reconciliation, intents, and positions before Stage 4K-2")
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
        warnings.append("risk snapshot missing; verify kill switch, hard halt, and daily loss controls before Stage 4K-2")
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
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("scheduler snapshot missing; verify scheduler state before Stage 4K-2")
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
        blockers.append("scheduler selected strategy job is not within Stage 4K-1 safety constraints")
    return {
        "scheduler_snapshot_present": present,
        "all_strategy_scheduler_enabled": broad_enabled,
        "selected_strategy_job_matches": selected_job_matches,
        "strategy_scan_execution_enabled": scan_enabled,
    }, blockers, warnings


def _scheduler_job_safe(job: dict[str, Any], selected_strategy_id: str | None) -> bool:
    return (
        _first_present(job.get("selected_strategy_id"), job.get("strategy_id"), default=None) == selected_strategy_id
        and job.get("paper_only") is not False
        and job.get("broker_submission_enabled") is False
        and job.get("live_trading_enabled") is False
        and job.get("all_strategies_enabled") is False
        and job.get("strategy_scan_execution_enabled") is False
    )


def _lifecycle_checks(snapshot: dict | None, selected_strategy_id: str | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; verify lifecycle state before Stage 4K-2")
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
        warnings.append("paper broker snapshot missing; verify PAPER broker config before Stage 4K-2")
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
        blockers.append("market window snapshot explicitly disallows readiness validation")
    if market_open is False:
        warnings.append("market is currently closed; readiness validation remains report-only")
    if is_trading_day is False:
        warnings.append("snapshot is not a trading day; readiness validation remains report-only")
    return {
        "market_window_snapshot_present": present,
        "allowed_to_schedule_paper_run": allowed,
        "is_trading_day": is_trading_day,
        "market_open": market_open,
        "reason": data.get("reason") if present else None,
    }, blockers, warnings


def _proposed_scope(selected_strategy_id: str | None, operation_id: str | None, may_build_next: bool) -> dict[str, Any]:
    return {
        "selected_strategy_id": selected_strategy_id,
        "operation_id": operation_id,
        "proposed_scope": "single_strategy_market_data_contract_qualification_gate",
        "paper_only": True,
        "one_strategy_only": True,
        "may_build_market_data_contract_plan_next_phase": may_build_next,
        "may_fetch_market_data_now": False,
        "may_qualify_contracts_now": False,
        "may_submit_orders_now": False,
        "may_create_intents_now": False,
        "may_create_tickets_now": False,
        "may_write_state_now": False,
        "may_write_ledger_now": False,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "broker_submission_enabled": False,
    }


def _required_inputs_for_4k2() -> list[str]:
    return [
        "accepted Stage 4J-6 controlled scheduled PAPER operation acceptance report",
        "selected strategy market data capability snapshot with native boolean flags",
        "selected strategy contract qualification capability snapshot with native boolean flags",
        "fresh state, risk, scheduler, lifecycle, paper broker, and market window snapshots",
        "operator confirmation that live trading, all-strategy automation, broker submission, intents, tickets, state writes, and ledger writes remain separately gated",
    ]


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    selected_strategy: dict[str, Any],
    operation: dict[str, Any],
    stage4j_completion_checks: dict[str, Any],
    strategy_registry_checks: dict[str, Any],
    market_data_readiness: dict[str, Any],
    contract_qualification_readiness: dict[str, Any],
    proposed_4k_scope: dict[str, Any],
    boundary_checks: dict[str, bool],
    required_inputs_for_4k2: list[str],
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
        "stage4k1_market_data_contract_readiness_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "selected_strategy": selected_strategy,
        "operation": operation,
        "stage4j_completion_checks": stage4j_completion_checks,
        "strategy_registry_checks": strategy_registry_checks,
        "market_data_readiness": market_data_readiness,
        "contract_qualification_readiness": contract_qualification_readiness,
        "proposed_4k_scope": proposed_4k_scope,
        "boundary_checks": boundary_checks,
        "required_inputs_for_4k2": required_inputs_for_4k2,
        "activation_snapshot_checks": activation_snapshot_checks,
        "state_checks": state_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "paper_broker_checks": paper_broker_checks,
        "market_window_checks": market_window_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4k2": {
            "ready_to_build_market_data_contract_plan": ready,
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
        "stage4j6_report_present": False,
        "stage4j6_report_ready": False,
        "stage4j_complete": False,
        "selected_strategy_present": False,
        "operation_id_present": False,
        "recommended_next_gate_matches": False,
    }


def _default_selected_strategy() -> dict[str, Any]:
    return {"selected_strategy_id": None, "paper_only": True, "one_strategy_only": True}


def _default_operation() -> dict[str, Any]:
    return {
        "operation_id": None,
        "operation_scope": "single_strategy_market_data_contract_readiness",
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }


def _default_stage4j_completion_checks() -> dict[str, Any]:
    return {
        "stage4j6_acceptance_report_true": False,
        "stage4j_complete": False,
        "ready_for_next_explicit_gate": False,
        "recommended_next_gate_matches": False,
        "executor_acceptance_accepted": False,
    }


def _default_strategy_registry_checks() -> dict[str, Any]:
    return {
        "strategy_registry_snapshot_present": False,
        "strategy_registry_parseable": False,
        "selected_strategy_id_present": False,
        "selected_strategy_paper_eligible": None,
        "multiple_active_or_selected_strategies": False,
        "strategy_ids_seen": [],
    }


def _default_market_data_readiness() -> dict[str, Any]:
    return _capability_result(False, "market_data", {}, None, True)


def _default_contract_readiness() -> dict[str, Any]:
    return _capability_result(False, "contract_qualification", {}, None, True)


def _default_boundary_checks() -> dict[str, bool]:
    return {
        "no_market_data_fetched": False,
        "no_contracts_qualified": False,
        "no_strategy_scan": True,
        "no_intents_created": False,
        "no_tickets_created": False,
        "no_orders_submitted": False,
        "no_state_written": False,
        "no_ledger_written": False,
        "no_broker_submission": False,
        "no_live_trading": False,
        "no_all_strategy_enablement": False,
        "no_scheduler_registration": True,
        "no_lifecycle_execution": True,
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
    return {key: False for key in FOUR_J_SAFETY_TRUE_FLAGS}
