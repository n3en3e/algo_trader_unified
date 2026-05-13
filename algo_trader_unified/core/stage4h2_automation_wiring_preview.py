"""Pure Stage 4H-2 controlled automation wiring preview report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
UNSAFE_TRUE_FLAGS = (
    "live_orders_enabled",
    "live_trading_enabled",
    "automated_paper_trading_enabled",
    "broker_submission_enabled",
    "scheduler_wiring_enabled",
    "scheduler_automation_enabled",
    "daemon_wiring_enabled",
    "lifecycle_wiring_enabled",
    "lifecycle_automation_enabled",
    "lifecycle_transition_execution_enabled",
    "market_data_enabled",
    "contract_qualification_enabled",
    "all_strategies_enabled",
    "enable_all_strategies",
    "automation_enabled",
)
ORDERED_NEXT_STEPS = [
    "Build Stage 4H-3 automation wiring dry run.",
    "Dry-run one-strategy paper automation only.",
    "Keep scheduler/lifecycle automation disabled until explicitly gated.",
    "Verify state, ledger, risk, and paper broker readiness before enabling automation.",
]
DO_NOT_DO_YET = [
    "Do not enable live trading.",
    "Do not enable all strategies at once.",
    "Do not place orders.",
    "Do not enable scheduler jobs now.",
    "Do not change strategy thresholds.",
    "Do not change position sizing.",
    "Do not bypass risk controls.",
]
RISK_GATES = [
    "kill_switch_check",
    "hard_halt_check",
    "soft_halt_check",
    "daily_loss_check",
    "max_position_order_size_check",
    "duplicate_intent_order_check",
    "state_reconciliation_clean_check",
    "paper_only_config_check",
    "live_trading_disabled_check",
    "broker_paper_port_config_check",
]


def build_stage4h2_automation_wiring_preview_report(
    *,
    stage4h1_readiness_report: dict | None,
    strategy_registry_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    explicit_preview_strategy_id: str | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Preview the smallest safe one-strategy automated paper wiring plan."""

    try:
        return _json_safe(
            _build_report(
                stage4h1_readiness_report=stage4h1_readiness_report,
                strategy_registry_snapshot=strategy_registry_snapshot,
                scheduler_snapshot=scheduler_snapshot,
                lifecycle_snapshot=lifecycle_snapshot,
                risk_snapshot=risk_snapshot,
                state_snapshot=state_snapshot,
                explicit_preview_strategy_id=explicit_preview_strategy_id,
                now_provider=now_provider,
            )
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must stay data-only.
        message = f"unexpected report failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks={
                    "stage4h1_readiness_present": stage4h1_readiness_report is not None,
                    "stage4h1_readiness_ready": False,
                },
                strategy_selection=_default_strategy_selection(explicit_preview_strategy_id),
                wiring_preview=_wiring_preview(None),
                safety_checks=_default_safety_checks(),
                risk_checks=_default_risk_checks(),
                scheduler_checks=_default_scheduler_checks(),
                lifecycle_checks=_default_lifecycle_checks(),
                state_checks=_default_state_checks(),
                blockers=[message],
                warnings=[],
                errors=[message],
                ready=False,
            )
        )


def _build_report(
    *,
    stage4h1_readiness_report: dict | None,
    strategy_registry_snapshot: dict | None,
    scheduler_snapshot: dict | None,
    lifecycle_snapshot: dict | None,
    risk_snapshot: dict | None,
    state_snapshot: dict | None,
    explicit_preview_strategy_id: str | None,
    now_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    generated_at = _generated_at(now_provider)
    stage4h1 = stage4h1_readiness_report if isinstance(stage4h1_readiness_report, dict) else None
    registry_present, registry = _optional_mapping(strategy_registry_snapshot)
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    artifact_checks = _artifact_checks(stage4h1_readiness_report)
    if stage4h1_readiness_report is None:
        blockers.append("Stage 4H-1 readiness report is missing")
    elif stage4h1 is None:
        blockers.append("Stage 4H-1 readiness report must be a dict")
        errors.append("Stage 4H-1 readiness report must be a dict")
    if not artifact_checks["stage4h1_readiness_ready"]:
        blockers.append("Stage 4H-1 readiness is not ready for Stage 4H-2")
    errors.extend(_as_string_list(_mapping(stage4h1).get("errors")))

    strategy_selection, strategy_blockers = _strategy_selection(
        stage4h1=stage4h1,
        registry=registry,
        registry_present=registry_present,
        explicit_preview_strategy_id=explicit_preview_strategy_id,
    )
    blockers.extend(strategy_blockers)
    selected = strategy_selection["selected_preview_strategy_id"]

    wiring_preview = _wiring_preview(selected)
    scheduler_checks, scheduler_blockers, scheduler_warnings = _scheduler_checks(
        scheduler_snapshot, selected, wiring_preview
    )
    blockers.extend(scheduler_blockers)
    warnings.extend(scheduler_warnings)

    lifecycle_checks, lifecycle_blockers, lifecycle_warnings = _lifecycle_checks(
        lifecycle_snapshot, wiring_preview
    )
    blockers.extend(lifecycle_blockers)
    warnings.extend(lifecycle_warnings)

    risk_checks, risk_blockers, risk_warnings = _risk_checks(risk_snapshot)
    blockers.extend(risk_blockers)
    warnings.extend(risk_warnings)

    state_checks, state_blockers, state_warnings = _state_checks(state_snapshot)
    blockers.extend(state_blockers)
    warnings.extend(state_warnings)

    safety_checks, safety_blockers = _safety_checks(
        stage4h1,
        strategy_registry_snapshot,
        scheduler_snapshot,
        lifecycle_snapshot,
        risk_snapshot,
        state_snapshot,
    )
    blockers.extend(safety_blockers)

    if not wiring_preview["available"]:
        blockers.append("wiring preview requires exactly one selected preview strategy")
    if not _is_structured_preview(wiring_preview["proposed_scheduler_wiring_preview"]):
        blockers.append("proposed scheduler wiring preview must be structured JSON-safe data")
    if not _is_structured_preview(wiring_preview["proposed_lifecycle_wiring_preview"]):
        blockers.append("proposed lifecycle wiring preview must be structured JSON-safe data")

    proposed_job = _first_job(wiring_preview["proposed_scheduler_wiring_preview"])
    if proposed_job and (
        proposed_job.get("disabled") is not True
        or proposed_job.get("would_register") is not False
        or proposed_job.get("would_execute") is not False
    ):
        blockers.append("proposed scheduler job must remain disabled and non-registering")
    if _any_lifecycle_flow_would_execute(wiring_preview["proposed_lifecycle_wiring_preview"]):
        blockers.append("proposed lifecycle preview must not execute")

    ready = (
        artifact_checks["stage4h1_readiness_present"] is True
        and artifact_checks["stage4h1_readiness_ready"] is True
        and strategy_selection["single_strategy_selected"] is True
        and wiring_preview["available"] is True
        and all(safety_checks.values())
        and not scheduler_checks["scheduler_already_enabled"]
        and not scheduler_checks["active_selected_strategy_job_present"]
        and not lifecycle_checks["lifecycle_already_enabled"]
        and lifecycle_checks["lifecycle_transition_execution_enabled"] is False
        and not state_checks["active_halt"]
        and state_checks["unresolved_needs_reconciliation_count"] == 0
        and not blockers
        and not errors
    )

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        strategy_selection=strategy_selection,
        wiring_preview=wiring_preview,
        safety_checks=safety_checks,
        risk_checks=risk_checks,
        scheduler_checks=scheduler_checks,
        lifecycle_checks=lifecycle_checks,
        state_checks=state_checks,
        blockers=_dedupe(blockers),
        warnings=_dedupe(warnings),
        errors=_dedupe(errors),
        ready=ready,
    )


def _artifact_checks(report: dict | None) -> dict[str, bool]:
    data = _mapping(report)
    readiness = _mapping(data.get("readiness_for_stage4h2"))
    return {
        "stage4h1_readiness_present": report is not None,
        "stage4h1_readiness_ready": (
            data.get("stage4h1_automation_readiness_report") is True
            and readiness.get("ready_to_build_automation_wiring_preview") is True
            and data.get("success") is True
        ),
    }


def _strategy_selection(
    *,
    stage4h1: dict[str, Any] | None,
    registry: dict[str, Any],
    registry_present: bool,
    explicit_preview_strategy_id: str | None,
) -> tuple[dict[str, Any], list[str]]:
    h1_candidates = _ids_from_string_list(
        _mapping(_mapping(stage4h1).get("strategy_candidate_checks")).get("candidate_strategy_ids")
    )
    registry_candidates, unsafe_ids = _registry_candidate_ids(registry)
    source_ids = h1_candidates if h1_candidates else registry_candidates
    candidate_ids = sorted(item for item in source_ids if item not in unsafe_ids)
    snapshot_preview = registry.get("preview_strategy_id")
    snapshot_preview_id = snapshot_preview if isinstance(snapshot_preview, str) and snapshot_preview else None
    requested = explicit_preview_strategy_id or snapshot_preview_id
    blockers: list[str] = []
    selected: str | None = None
    reason = "no eligible strategy candidates"

    if not candidate_ids:
        blockers.append("zero paper-eligible strategy candidates are available")
    elif requested:
        if requested in candidate_ids:
            selected = requested
            reason = "explicit preview strategy selected" if explicit_preview_strategy_id else "snapshot preview strategy selected"
        else:
            blockers.append(f"preview strategy {requested} is not eligible")
            reason = "requested preview strategy is not eligible"
    elif len(candidate_ids) == 1:
        selected = candidate_ids[0]
        reason = "exactly one eligible preview strategy candidate"
    else:
        blockers.append("one preview strategy must be explicitly selected")
        reason = "multiple eligible candidates require explicit preview selection"

    return (
        {
            "strategy_registry_present": registry_present,
            "candidate_strategy_ids": candidate_ids,
            "explicit_preview_strategy_id": explicit_preview_strategy_id,
            "selected_preview_strategy_id": selected,
            "single_strategy_selected": selected is not None,
            "selection_reason": reason,
        },
        blockers,
    )


def _registry_candidate_ids(registry: dict[str, Any]) -> tuple[set[str], set[str]]:
    candidates = _ids_from_string_list(registry.get("paper_eligible_strategy_ids"))
    unsafe: set[str] = set()
    for key in ("strategies", "candidates"):
        ids, unsafe_ids = _ids_from_dict_list(registry.get(key))
        candidates |= ids
        unsafe |= unsafe_ids
    return candidates, unsafe


def _ids_from_string_list(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str) and item}


def _ids_from_dict_list(value: Any) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    unsafe: set[str] = set()
    if not isinstance(value, list):
        return ids, unsafe
    for item in value:
        if not isinstance(item, dict):
            continue
        strategy_id = item.get("strategy_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            continue
        if item.get("live_only") is True or item.get("enabled") is True or item.get("automated_enabled") is True:
            unsafe.add(strategy_id)
            continue
        if item.get("paper_eligible") is True:
            ids.add(strategy_id)
    return ids, unsafe


def _wiring_preview(selected_strategy_id: str | None) -> dict[str, Any]:
    scheduler = {
        "jobs": [
            {
                "job_id": f"stage4h3_dry_run_{selected_strategy_id or 'unselected'}",
                "strategy_id": selected_strategy_id,
                "trigger_description": "operator-controlled Stage 4H-3 dry-run cadence preview",
                "disabled": True,
                "would_register": False,
                "would_execute": False,
                "paper_only": True,
            }
        ]
    }
    lifecycle = {
        "flows": [
            {"name": "signal_to_intent", "would_execute": False, "paper_only": True},
            {"name": "intent_to_ticket", "would_execute": False, "paper_only": True},
            {"name": "ticket_to_paper_submit", "would_execute": False, "paper_only": True},
            {"name": "submit_to_state_ledger_tracking", "would_execute": False, "paper_only": True},
            {"name": "reconciliation_check", "would_execute": False, "paper_only": True},
            {"name": "halt_check", "would_execute": False, "paper_only": True},
        ]
    }
    return {
        "available": selected_strategy_id is not None,
        "proposed_scheduler_wiring_preview": scheduler,
        "proposed_lifecycle_wiring_preview": lifecycle,
        "proposed_signal_to_intent_flow": {
            "strategy_id": selected_strategy_id,
            "would_execute": False,
            "paper_only": True,
            "output_preview": "paper_intent",
        },
        "proposed_intent_to_order_ticket_flow": {
            "strategy_id": selected_strategy_id,
            "would_execute": False,
            "paper_only": True,
            "output_preview": "paper_order_ticket",
        },
        "proposed_order_ticket_to_paper_submit_flow": {
            "strategy_id": selected_strategy_id,
            "would_execute": False,
            "would_submit": False,
            "paper_only": True,
        },
        "proposed_state_and_ledger_tracking_flow": {
            "strategy_id": selected_strategy_id,
            "would_execute": False,
            "would_mutate_state": False,
            "would_write_ledger": False,
            "paper_only": True,
        },
        "proposed_risk_gates": [
            {"name": name, "required_for_4h3": True, "would_execute": False}
            for name in RISK_GATES
        ],
        "disabled_components": [
            "scheduler_wiring",
            "daemon_wiring",
            "lifecycle_wiring",
            "lifecycle_transition_execution",
            "broker_submission",
            "market_data",
            "contract_qualification",
            "state_mutation",
            "ledger_writes",
            "live_trading",
            "all_strategy_automation",
        ],
        "dry_run_plan_for_4H3": {
            "strategy_id": selected_strategy_id,
            "paper_only": True,
            "would_register_scheduler_job": False,
            "would_execute_lifecycle_transition": False,
            "would_submit_order": False,
        },
    }


def _scheduler_checks(
    snapshot: dict | None,
    selected_strategy_id: str | None,
    wiring_preview: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    already_enabled = _contains_truthy_flag(data, "scheduler_automation_enabled") or _contains_truthy_flag(data, "scheduler_wiring_enabled")
    active_job = _active_selected_strategy_job_present(data, selected_strategy_id)
    proposed_job = _first_job(wiring_preview["proposed_scheduler_wiring_preview"])
    checks = {
        "scheduler_snapshot_present": present,
        "scheduler_already_enabled": already_enabled,
        "active_selected_strategy_job_present": active_job,
        "proposed_job_disabled": _mapping(proposed_job).get("disabled") is True,
        "proposed_job_would_register": False,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("scheduler snapshot missing; 4H-3 must verify scheduler remains disabled")
    if already_enabled:
        blockers.append("scheduler automation is already enabled")
    if active_job:
        blockers.append("active scheduler job exists for selected preview strategy")
    return checks, blockers, warnings


def _active_selected_strategy_job_present(data: dict[str, Any], selected_strategy_id: str | None) -> bool:
    if not selected_strategy_id:
        return False
    for job in _as_list(data.get("jobs")) + _as_list(data.get("active_jobs")):
        if not isinstance(job, dict):
            continue
        if job.get("strategy_id") != selected_strategy_id:
            continue
        if job.get("dry_run_only") is True and job.get("disabled") is True:
            continue
        return True
    return False


def _lifecycle_checks(
    snapshot: dict | None,
    wiring_preview: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    already_enabled = _contains_truthy_flag(data, "lifecycle_automation_enabled") or _contains_truthy_flag(data, "lifecycle_wiring_enabled")
    transition_enabled = _contains_truthy_flag(data, "lifecycle_transition_execution_enabled")
    would_execute = _any_lifecycle_flow_would_execute(wiring_preview["proposed_lifecycle_wiring_preview"])
    checks = {
        "lifecycle_snapshot_present": present,
        "lifecycle_already_enabled": already_enabled,
        "lifecycle_transition_execution_enabled": transition_enabled,
        "proposed_lifecycle_would_execute": would_execute,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; 4H-3 must verify lifecycle automation remains disabled")
    if already_enabled:
        blockers.append("lifecycle automation is already enabled")
    if transition_enabled:
        blockers.append("lifecycle transition execution is enabled")
    if would_execute:
        blockers.append("proposed lifecycle preview would execute")
    return checks, blockers, warnings


def _risk_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    checks = {
        "risk_snapshot_present": present,
        "kill_switch_available": data.get("kill_switch_available") is True,
        "hard_halt_available": data.get("hard_halt_available") is True,
        "daily_loss_limit_available": data.get("daily_loss_limit_available") is True,
        "risk_bypass_enabled": data.get("risk_bypass_enabled") is True,
        "risk_gates_previewed": list(RISK_GATES),
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("risk snapshot missing; 4H-3 must verify risk controls before dry run")
        return checks, blockers, warnings
    if checks["risk_bypass_enabled"]:
        blockers.append("risk controls are bypassed")
    for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
        if checks[key] is not True:
            blockers.append(f"{key} must be true in supplied risk snapshot")
    return checks, blockers, warnings


def _state_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    unresolved = _first_present(
        data.get("unresolved_needs_reconciliation_count"),
        data.get("needs_reconciliation_count"),
        default=0,
    )
    active_intents = _safe_int(data.get("active_intents_count"))
    checks = {
        "state_snapshot_present": present,
        "active_halt": bool(data.get("active_halt") or data.get("halt_active")),
        "unresolved_needs_reconciliation_count": _safe_int(unresolved),
        "active_intents_count": active_intents,
        "open_positions_count": _safe_int(data.get("open_positions_count")),
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("state snapshot missing; 4H-3 must include read-only state verification")
        return checks, blockers, warnings
    if checks["active_halt"]:
        blockers.append("active halt is present")
    if checks["unresolved_needs_reconciliation_count"] > 0:
        blockers.append("unresolved NEEDS_RECONCILIATION is present")
    if active_intents > 0:
        warnings.append("active intents are present; 4H-3 must verify they are safe before dry run")
    if data.get("active_intents_unsafe") is True and active_intents > 0:
        blockers.append("active intents are marked unsafe")
    return checks, blockers, warnings


def _safety_checks(*sources: Any) -> tuple[dict[str, bool], list[str]]:
    unsafe = {key: any(_contains_truthy_flag(source, key) for source in sources) for key in UNSAFE_TRUE_FLAGS}
    checks = {
        "no_live_orders": not unsafe["live_orders_enabled"] and not unsafe["live_trading_enabled"],
        "no_market_data": not unsafe["market_data_enabled"],
        "no_contract_qualification": not unsafe["contract_qualification_enabled"],
        "no_broker_submission_enabled": not unsafe["broker_submission_enabled"],
        "no_scheduler_wiring_enabled": not unsafe["scheduler_wiring_enabled"] and not unsafe["scheduler_automation_enabled"],
        "no_lifecycle_wiring_enabled": not unsafe["lifecycle_wiring_enabled"] and not unsafe["lifecycle_automation_enabled"],
        "no_automated_paper_trading_enabled": not unsafe["automated_paper_trading_enabled"] and not unsafe["automation_enabled"],
        "no_daemon_wiring_enabled": not unsafe["daemon_wiring_enabled"],
        "no_all_strategy_enablement": not unsafe["all_strategies_enabled"] and not unsafe["enable_all_strategies"],
    }
    reasons = {
        "no_live_orders": "live trading flag is enabled",
        "no_market_data": "market data flag is enabled",
        "no_contract_qualification": "contract qualification flag is enabled",
        "no_broker_submission_enabled": "broker submission automation is enabled",
        "no_scheduler_wiring_enabled": "scheduler automation is already enabled",
        "no_lifecycle_wiring_enabled": "lifecycle automation is already enabled",
        "no_automated_paper_trading_enabled": "automated paper trading is already enabled",
        "no_daemon_wiring_enabled": "daemon wiring is already enabled",
        "no_all_strategy_enablement": "all-strategy automation is enabled or requested",
    }
    return checks, [reasons[key] for key, value in checks.items() if value is not True]


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    strategy_selection: dict[str, Any],
    wiring_preview: dict[str, Any],
    safety_checks: dict[str, Any],
    risk_checks: dict[str, Any],
    scheduler_checks: dict[str, Any],
    lifecycle_checks: dict[str, Any],
    state_checks: dict[str, Any],
    blockers: list[str],
    warnings: list[str],
    errors: list[str],
    ready: bool,
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "stage4h2_automation_wiring_preview_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "strategy_selection": strategy_selection,
        "wiring_preview": wiring_preview,
        "safety_checks": safety_checks,
        "risk_checks": risk_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "state_checks": state_checks,
        "readiness_for_stage4h3": {
            "ready_to_build_automation_wiring_dry_run": ready,
            "blockers": list(blockers),
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


def _default_strategy_selection(explicit_preview_strategy_id: str | None) -> dict[str, Any]:
    return {
        "strategy_registry_present": False,
        "candidate_strategy_ids": [],
        "explicit_preview_strategy_id": explicit_preview_strategy_id,
        "selected_preview_strategy_id": None,
        "single_strategy_selected": False,
        "selection_reason": "report construction failed",
    }


def _default_safety_checks() -> dict[str, bool]:
    return {
        "no_live_orders": False,
        "no_market_data": False,
        "no_contract_qualification": False,
        "no_broker_submission_enabled": False,
        "no_scheduler_wiring_enabled": False,
        "no_lifecycle_wiring_enabled": False,
        "no_automated_paper_trading_enabled": False,
        "no_daemon_wiring_enabled": False,
        "no_all_strategy_enablement": False,
    }


def _default_risk_checks() -> dict[str, Any]:
    return {
        "risk_snapshot_present": False,
        "kill_switch_available": False,
        "hard_halt_available": False,
        "daily_loss_limit_available": False,
        "risk_bypass_enabled": False,
        "risk_gates_previewed": [],
    }


def _default_scheduler_checks() -> dict[str, Any]:
    return {
        "scheduler_snapshot_present": False,
        "scheduler_already_enabled": False,
        "active_selected_strategy_job_present": False,
        "proposed_job_disabled": False,
        "proposed_job_would_register": False,
    }


def _default_lifecycle_checks() -> dict[str, Any]:
    return {
        "lifecycle_snapshot_present": False,
        "lifecycle_already_enabled": False,
        "lifecycle_transition_execution_enabled": False,
        "proposed_lifecycle_would_execute": False,
    }


def _default_state_checks() -> dict[str, Any]:
    return {
        "state_snapshot_present": False,
        "active_halt": False,
        "unresolved_needs_reconciliation_count": 0,
        "active_intents_count": 0,
        "open_positions_count": 0,
    }


def _is_structured_preview(value: Any) -> bool:
    if isinstance(value, dict):
        return all(not isinstance(item, str) for item in value.values())
    if isinstance(value, list):
        return all(isinstance(item, dict) for item in value)
    return False


def _first_job(value: Any) -> dict[str, Any] | None:
    jobs = _mapping(value).get("jobs") if isinstance(value, dict) else value
    for item in _as_list(jobs):
        if isinstance(item, dict):
            return item
    return None


def _any_lifecycle_flow_would_execute(value: Any) -> bool:
    flows = _mapping(value).get("flows") if isinstance(value, dict) else value
    return any(isinstance(item, dict) and item.get("would_execute") is True for item in _as_list(flows))


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
    return [str(item) for item in _as_list(value)]


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
            (item_key == key and item is True) or _contains_truthy_flag(item, key)
            for item_key, item in value.items()
        )
    if isinstance(value, list):
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
        return value.astimezone(timezone.utc).isoformat()
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Decimal):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return "<object>"
