"""Pure Stage 4H-1 controlled automation readiness report."""

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
    "scheduler_changes_enabled",
    "scheduler_wiring_enabled",
    "lifecycle_wiring_enabled",
    "market_data_enabled",
    "contract_qualification_enabled",
    "all_strategies_enabled",
    "enable_all_strategies",
    "automation_enabled",
)
ORDERED_NEXT_STEPS = [
    "Build Stage 4H-2 automation wiring preview.",
    "Preview one-strategy paper automation only.",
    "Keep scheduler/lifecycle automation disabled until explicitly gated.",
    "Start 4H with read-only state, ledger, and risk verification.",
]
DO_NOT_DO_YET = [
    "Do not enable live trading.",
    "Do not enable all strategies at once.",
    "Do not bypass risk controls.",
    "Do not change strategy thresholds.",
    "Do not change position sizing.",
    "Do not place orders.",
]


def build_stage4h1_automation_readiness_report(
    *,
    stage4g_acceptance_report: dict | None,
    module_checks: dict | None = None,
    safety_checks: dict | None = None,
    state_snapshot: dict | None = None,
    strategy_registry_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Evaluate whether Stage 4H-2 automation wiring preview design may begin."""

    try:
        return _json_safe(
            _build_report(
                stage4g_acceptance_report=stage4g_acceptance_report,
                module_checks=module_checks,
                safety_checks=safety_checks,
                state_snapshot=state_snapshot,
                strategy_registry_snapshot=strategy_registry_snapshot,
                risk_snapshot=risk_snapshot,
                now_provider=now_provider,
            )
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must remain data-only.
        message = f"unexpected report failure: {type(exc).__name__}: {exc}"
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                artifact_checks=_default_artifact_checks(),
                module_checks=_default_module_checks(),
                state_checks=_default_state_checks(),
                strategy_candidate_checks=_default_strategy_candidate_checks(),
                risk_checks=_default_risk_checks(),
                safety_checks=_default_safety_checks(),
                blockers=[message],
                warnings=[],
                errors=[message],
            )
        )


def _build_report(
    *,
    stage4g_acceptance_report: dict | None,
    module_checks: dict | None,
    safety_checks: dict | None,
    state_snapshot: dict | None,
    strategy_registry_snapshot: dict | None,
    risk_snapshot: dict | None,
    now_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    generated_at = _generated_at(now_provider)
    stage4g = stage4g_acceptance_report if isinstance(stage4g_acceptance_report, dict) else None
    modules = module_checks if isinstance(module_checks, dict) else {}
    supplied_safety = safety_checks if isinstance(safety_checks, dict) else {}
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4g_acceptance_report is None:
        blockers.append("Stage 4G acceptance report is missing")
    elif stage4g is None:
        blockers.append("Stage 4G acceptance report must be a dict")
        errors.append("Stage 4G acceptance report must be a dict")

    artifact_checks, artifact_blockers = _artifact_checks(stage4g)
    blockers.extend(artifact_blockers)
    errors.extend(_as_string_list(_mapping(stage4g).get("errors")))

    module_report = _module_checks(modules)
    safety_report, safety_blockers = _safety_checks(stage4g, modules, supplied_safety)
    blockers.extend(safety_blockers)

    state_report, state_blockers, state_warnings = _state_checks(state_snapshot)
    blockers.extend(state_blockers)
    warnings.extend(state_warnings)

    strategy_report, strategy_blockers, strategy_warnings = _strategy_candidate_checks(
        strategy_registry_snapshot
    )
    blockers.extend(strategy_blockers)
    warnings.extend(strategy_warnings)

    risk_report, risk_blockers, risk_warnings = _risk_checks(risk_snapshot)
    blockers.extend(risk_blockers)
    warnings.extend(risk_warnings)

    if module_report["automation_wiring_enabled"]:
        blockers.append("automation wiring is already enabled")

    ready = (
        all(artifact_checks.values())
        and all(safety_report.values())
        and not blockers
        and not errors
    )
    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        module_checks=module_report,
        state_checks=state_report,
        strategy_candidate_checks=strategy_report,
        risk_checks=risk_report,
        safety_checks=safety_report,
        blockers=_dedupe(blockers),
        warnings=_dedupe(warnings),
        errors=_dedupe(errors),
        ready=ready,
    )


def _artifact_checks(report: dict[str, Any] | None) -> tuple[dict[str, bool], list[str]]:
    data = _mapping(report)
    readiness = _mapping(data.get("readiness_for_stage4h"))
    artifact_checks = _mapping(data.get("artifact_checks"))
    rollback = _mapping(data.get("rollback"))
    checks = {
        "stage4g_acceptance_present": report is not None,
        "stage4g_acceptance_ready": (
            data.get("stage4g6_lifecycle_write_acceptance_report") is True
            and readiness.get("ready_to_begin_controlled_automated_paper_trading_launch")
            is True
            and data.get("success") is True
        ),
        "stage4g_executor_clean": (
            not _as_list(data.get("errors"))
            and not _as_list(readiness.get("blockers"))
            and not _has_nonempty_key(data, "skipped_operations")
        ),
        "stage4g_no_rollback": (
            artifact_checks.get("rollback_not_required") is True
            or rollback.get("rollback_required") is False
            or "rollback" not in data
        ),
    }
    reasons = {
        "stage4g_acceptance_present": "Stage 4G acceptance report is missing",
        "stage4g_acceptance_ready": "Stage 4G acceptance is not ready for Stage 4H",
        "stage4g_executor_clean": "Stage 4G acceptance has errors, blockers, or skipped operations",
        "stage4g_no_rollback": "Stage 4G acceptance indicates rollback is required",
    }
    return checks, [reasons[key] for key, value in checks.items() if value is not True]


def _module_checks(data: dict[str, Any]) -> dict[str, bool]:
    return {
        "strategy_registry_present": _truthy(data.get("strategy_registry_present")),
        "risk_controls_present": _truthy(data.get("risk_controls_present")),
        "state_store_lifecycle_foundation_present": _truthy(
            data.get("state_store_lifecycle_foundation_present")
        ),
        "ledger_lifecycle_foundation_present": _truthy(
            data.get("ledger_lifecycle_foundation_present")
        ),
        "scheduler_components_present": _truthy(data.get("scheduler_components_present")),
        "automation_wiring_present": _truthy(data.get("automation_wiring_present")),
        "automation_wiring_enabled": _truthy(data.get("automation_wiring_enabled"))
        or _contains_truthy_flag(data, "automation_enabled"),
    }


def _state_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    unresolved = _first_present(
        data.get("unresolved_needs_reconciliation_count"),
        data.get("needs_reconciliation_count"),
        _count_status(data, "NEEDS_RECONCILIATION"),
        default=0,
    )
    active_halt = bool(data.get("active_halt") or data.get("halt_active"))
    checks = {
        "state_snapshot_present": present,
        "unresolved_needs_reconciliation_count": _safe_int(unresolved),
        "active_halt": active_halt,
        "active_intents_count": _safe_int(data.get("active_intents_count")),
        "open_positions_count": _safe_int(data.get("open_positions_count")),
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("state snapshot missing; 4H-2 must begin with read-only state verification")
    if active_halt:
        blockers.append("active halt is present")
    if checks["unresolved_needs_reconciliation_count"] > 0:
        blockers.append("unresolved NEEDS_RECONCILIATION is present")
    return checks, blockers, warnings


def _strategy_candidate_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    ids = (
        _ids_from_string_list(data.get("paper_eligible_strategy_ids"))
        | _ids_from_dict_list(data.get("strategies"))
        | _ids_from_dict_list(data.get("candidates"))
    )
    sorted_ids = sorted(ids)
    checks = {
        "strategy_registry_snapshot_present": present,
        "paper_eligible_strategy_count": len(sorted_ids),
        "candidate_strategy_ids": sorted_ids,
        "single_strategy_launch_required": True,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("strategy registry snapshot missing; 4H-2 must verify one paper-eligible candidate")
    elif not sorted_ids:
        blockers.append("strategy registry snapshot has no safe paper-eligible candidates")
    return checks, blockers, warnings


def _risk_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    checks = {
        "risk_snapshot_present": present,
        "kill_switch_available": _truthy(data.get("kill_switch_available")),
        "soft_halt_available": _truthy(data.get("soft_halt_available")),
        "hard_halt_available": _truthy(data.get("hard_halt_available")),
        "daily_loss_limit_available": _truthy(data.get("daily_loss_limit_available")),
        "max_position_limit_available": _truthy(data.get("max_position_limit_available")),
        "risk_controls_clean": data.get("risk_controls_clean") is not False
        and data.get("risk_bypass_enabled") is not True,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("risk snapshot missing; 4H-2/4H-3 must verify risk controls before wiring")
        return checks, blockers, warnings
    if data.get("risk_bypass_enabled") is True:
        blockers.append("risk controls are bypassed")
    for key in (
        "kill_switch_available",
        "hard_halt_available",
        "daily_loss_limit_available",
    ):
        if data.get(key) is not True:
            blockers.append(f"{key} must be true in supplied risk snapshot")
    if data.get("risk_controls_clean") is False:
        blockers.append("risk controls are not clean")
    return checks, blockers, warnings


def _safety_checks(
    stage4g: dict[str, Any] | None,
    module_checks: dict[str, Any],
    safety_checks: dict[str, Any],
) -> tuple[dict[str, bool], list[str]]:
    sources = [stage4g or {}, module_checks, safety_checks]
    unsafe = {key: any(_contains_truthy_flag(source, key) for source in sources) for key in UNSAFE_TRUE_FLAGS}
    checks = {
        "no_live_orders": not unsafe["live_orders_enabled"] and not unsafe["live_trading_enabled"],
        "no_market_data": not unsafe["market_data_enabled"],
        "no_contract_qualification": not unsafe["contract_qualification_enabled"],
        "no_broker_submission_enabled": not unsafe["broker_submission_enabled"],
        "no_scheduler_changes": not unsafe["scheduler_changes_enabled"]
        and not unsafe["scheduler_wiring_enabled"],
        "no_lifecycle_wiring": not unsafe["lifecycle_wiring_enabled"],
        "no_automated_paper_trading_enabled": not unsafe["automated_paper_trading_enabled"]
        and not unsafe["automation_enabled"],
        "no_all_strategy_enablement": not unsafe["all_strategies_enabled"]
        and not unsafe["enable_all_strategies"],
    }
    reasons = {
        "no_live_orders": "live order or live trading flag is enabled",
        "no_market_data": "market data flag is enabled",
        "no_contract_qualification": "contract qualification flag is enabled",
        "no_broker_submission_enabled": "broker submission automation is enabled",
        "no_scheduler_changes": "scheduler automation flag is enabled",
        "no_lifecycle_wiring": "lifecycle wiring flag is enabled",
        "no_automated_paper_trading_enabled": "automated paper trading is already enabled",
        "no_all_strategy_enablement": "all-strategy automation is enabled or requested",
    }
    return checks, [reasons[key] for key, value in checks.items() if value is not True]


def _ids_from_string_list(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value if isinstance(item, str) and item}


def _ids_from_dict_list(value: Any) -> set[str]:
    ids: set[str] = set()
    if not isinstance(value, list):
        return ids
    for item in value:
        if not isinstance(item, dict):
            continue
        strategy_id = item.get("strategy_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            continue
        if item.get("paper_eligible") is not True:
            continue
        if item.get("live_only") is True:
            continue
        if item.get("enabled") is True or item.get("automated_enabled") is True:
            continue
        ids.add(strategy_id)
    return ids


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    module_checks: dict[str, Any],
    state_checks: dict[str, Any],
    strategy_candidate_checks: dict[str, Any],
    risk_checks: dict[str, Any],
    safety_checks: dict[str, Any],
    blockers: list[str],
    warnings: list[str],
    errors: list[str],
    ready: bool = False,
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "stage4h1_automation_readiness_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "module_checks": module_checks,
        "state_checks": state_checks,
        "strategy_candidate_checks": strategy_candidate_checks,
        "risk_checks": risk_checks,
        "safety_checks": safety_checks,
        "readiness_for_stage4h2": {
            "ready_to_build_automation_wiring_preview": ready,
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


def _default_artifact_checks() -> dict[str, bool]:
    return {
        "stage4g_acceptance_present": False,
        "stage4g_acceptance_ready": False,
        "stage4g_executor_clean": False,
        "stage4g_no_rollback": False,
    }


def _default_module_checks() -> dict[str, bool]:
    return {
        "strategy_registry_present": False,
        "risk_controls_present": False,
        "state_store_lifecycle_foundation_present": False,
        "ledger_lifecycle_foundation_present": False,
        "scheduler_components_present": False,
        "automation_wiring_present": False,
        "automation_wiring_enabled": False,
    }


def _default_state_checks() -> dict[str, Any]:
    return {
        "state_snapshot_present": False,
        "unresolved_needs_reconciliation_count": 0,
        "active_halt": False,
        "active_intents_count": 0,
        "open_positions_count": 0,
    }


def _default_strategy_candidate_checks() -> dict[str, Any]:
    return {
        "strategy_registry_snapshot_present": False,
        "paper_eligible_strategy_count": 0,
        "candidate_strategy_ids": [],
        "single_strategy_launch_required": True,
    }


def _default_risk_checks() -> dict[str, bool]:
    return {
        "risk_snapshot_present": False,
        "kill_switch_available": False,
        "soft_halt_available": False,
        "hard_halt_available": False,
        "daily_loss_limit_available": False,
        "max_position_limit_available": False,
        "risk_controls_clean": True,
    }


def _default_safety_checks() -> dict[str, bool]:
    return {
        "no_live_orders": False,
        "no_market_data": False,
        "no_contract_qualification": False,
        "no_broker_submission_enabled": False,
        "no_scheduler_changes": False,
        "no_lifecycle_wiring": False,
        "no_automated_paper_trading_enabled": False,
        "no_all_strategy_enablement": False,
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
    return [str(item) for item in _as_list(value)]


def _truthy(value: Any) -> bool:
    return value is True


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


def _count_status(value: Any, status: str) -> int:
    if isinstance(value, dict):
        count = 0
        if value.get("status") == status or value.get("lifecycle_state") == status:
            count += 1
        return count + sum(_count_status(item, status) for item in value.values())
    if isinstance(value, list):
        return sum(_count_status(item, status) for item in value)
    return 0


def _contains_truthy_flag(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return any(
            (item_key == key and bool(item)) or _contains_truthy_flag(item, key)
            for item_key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_truthy_flag(item, key) for item in value)
    return False


def _has_nonempty_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        if key in value and bool(_as_list(value.get(key))):
            return True
        return any(_has_nonempty_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_has_nonempty_key(item, key) for item in value)
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
