"""Pure Stage 4H-6 one-strategy activation acceptance report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
PAPER_IBKR_PORTS = {4004}
ORDERED_NEXT_STEPS = [
    "Build the first controlled scheduled PAPER automation run phase.",
    "Before first scheduled run, re-check state, ledger/audit, risk, paper broker config, scheduler, and lifecycle snapshots.",
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
    "Do not enable broad scheduler or lifecycle automation.",
]


def build_stage4h6_one_strategy_activation_acceptance_report(
    *,
    stage4h5_activation_executor_report: dict | None,
    activation_snapshot: dict | None = None,
    audit_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Build a read-only acceptance report for one-strategy PAPER activation."""

    try:
        return _json_safe(
            _build_report(
                stage4h5_activation_executor_report=stage4h5_activation_executor_report,
                activation_snapshot=activation_snapshot,
                audit_snapshot=audit_snapshot,
                state_snapshot=state_snapshot,
                scheduler_snapshot=scheduler_snapshot,
                lifecycle_snapshot=lifecycle_snapshot,
                risk_snapshot=risk_snapshot,
                paper_broker_snapshot=paper_broker_snapshot,
                now_provider=now_provider,
            )
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must stay data-only.
        message = f"unexpected acceptance failure: {type(exc).__name__}: {exc}"
        generated_at = _generated_at(now_provider)
        return _json_safe(
            _base_report(
                generated_at=generated_at,
                artifact_checks=_default_artifact_checks(),
                activation_payload_checks=_default_activation_payload_checks(),
                snapshot_checks=_default_snapshot_checks(),
                state_checks=_default_state_checks(),
                scheduler_checks=_default_scheduler_checks(),
                lifecycle_checks=_default_lifecycle_checks(),
                risk_checks=_default_risk_checks(),
                paper_broker_checks=_default_paper_broker_checks(),
                safety_checks=_safety_checks({}),
                ready=False,
                blockers=[message],
                warnings=[],
                errors=[message],
            )
        )


def _build_report(
    *,
    stage4h5_activation_executor_report: dict | None,
    activation_snapshot: dict | None,
    audit_snapshot: dict | None,
    state_snapshot: dict | None,
    scheduler_snapshot: dict | None,
    lifecycle_snapshot: dict | None,
    risk_snapshot: dict | None,
    paper_broker_snapshot: dict | None,
    now_provider: Callable[[], Any] | None,
) -> dict[str, Any]:
    generated_at = _generated_at(now_provider)
    h5 = stage4h5_activation_executor_report if isinstance(stage4h5_activation_executor_report, dict) else None
    data = _mapping(h5)
    blockers: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if stage4h5_activation_executor_report is None:
        blockers.append("Stage 4H-5 activation executor report is missing")
    elif h5 is None:
        blockers.append("Stage 4H-5 activation executor report must be a dict")
        errors.append("Stage 4H-5 activation executor report must be a dict")
    errors.extend(_as_string_list(data.get("errors")))

    artifact_checks = _artifact_checks(stage4h5_activation_executor_report)
    blockers.extend(_artifact_blockers(artifact_checks, data))

    selected_id = _selected_strategy_id(data)
    payload = _mapping(data.get("activation_payload"))
    activation_payload_checks, payload_blockers, payload_warnings = _activation_payload_checks(
        payload, selected_id, _as_list(data.get("applied_operations"))
    )
    blockers.extend(payload_blockers)
    warnings.extend(payload_warnings)

    activation_snapshot_checks, activation_blockers, activation_warnings = (
        _activation_snapshot_checks(activation_snapshot, selected_id, payload)
    )
    blockers.extend(activation_blockers)
    warnings.extend(activation_warnings)

    audit_snapshot_checks, audit_blockers, audit_warnings = _audit_snapshot_checks(
        audit_snapshot, selected_id
    )
    blockers.extend(audit_blockers)
    warnings.extend(audit_warnings)

    state_checks, state_blockers, state_warnings = _state_checks(state_snapshot)
    blockers.extend(state_blockers)
    warnings.extend(state_warnings)

    scheduler_checks, scheduler_blockers, scheduler_warnings = _scheduler_checks(
        scheduler_snapshot, selected_id
    )
    blockers.extend(scheduler_blockers)
    warnings.extend(scheduler_warnings)

    lifecycle_checks, lifecycle_blockers, lifecycle_warnings = _lifecycle_checks(
        lifecycle_snapshot
    )
    blockers.extend(lifecycle_blockers)
    warnings.extend(lifecycle_warnings)

    risk_checks, risk_blockers, risk_warnings = _risk_checks(risk_snapshot)
    blockers.extend(risk_blockers)
    warnings.extend(risk_warnings)

    paper_broker_checks, paper_blockers, paper_warnings = _paper_broker_checks(
        paper_broker_snapshot
    )
    blockers.extend(paper_blockers)
    warnings.extend(paper_warnings)

    snapshot_checks = {
        "activation_snapshot_present": isinstance(activation_snapshot, dict),
        "activation_snapshot_matches": activation_snapshot_checks["activation_snapshot_matches"],
        "audit_snapshot_present": isinstance(audit_snapshot, dict),
        "audit_snapshot_matches": audit_snapshot_checks["audit_snapshot_matches"],
        "state_snapshot_present": isinstance(state_snapshot, dict),
        "scheduler_snapshot_present": isinstance(scheduler_snapshot, dict),
        "lifecycle_snapshot_present": isinstance(lifecycle_snapshot, dict),
        "risk_snapshot_present": isinstance(risk_snapshot, dict),
        "paper_broker_snapshot_present": isinstance(paper_broker_snapshot, dict),
    }

    safety_sources = [
        payload,
        activation_snapshot if isinstance(activation_snapshot, dict) else {},
        scheduler_snapshot if isinstance(scheduler_snapshot, dict) else {},
        lifecycle_snapshot if isinstance(lifecycle_snapshot, dict) else {},
        paper_broker_snapshot if isinstance(paper_broker_snapshot, dict) else {},
    ]
    safety_checks = _safety_checks(*safety_sources)
    blockers.extend(_safety_blockers(safety_checks))

    skipped_operations = _as_list(data.get("skipped_operations"))
    if skipped_operations:
        blockers.append("Stage 4H-5 skipped operations must be empty for clean acceptance")

    blocker_list = _dedupe(blockers)
    warning_list = _dedupe(warnings)
    error_list = _dedupe(errors)
    ready = (
        not blocker_list
        and not error_list
        and all(artifact_checks.values())
        and activation_payload_checks["activation_payload_present"] is True
        and activation_payload_checks["selected_strategy_consistent"] is True
        and activation_payload_checks["paper_only"] is True
        and activation_payload_checks["one_strategy_only"] is True
        and activation_payload_checks["live_trading_disabled"] is True
        and activation_payload_checks["all_strategies_disabled"] is True
        and activation_payload_checks["broker_submission_disabled"] is True
        and activation_payload_checks["applied_operation_payloads_safe"] is True
        and activation_snapshot_checks["activation_snapshot_matches"] is True
        and audit_snapshot_checks["audit_snapshot_matches"] is True
        and safety_checks["no_live_orders"] is True
        and safety_checks["no_market_data"] is True
        and safety_checks["no_contract_qualification"] is True
        and safety_checks["no_broker_submission_enabled"] is True
        and safety_checks["no_all_strategy_enablement"] is True
        and safety_checks["no_live_trading"] is True
    )

    return _base_report(
        generated_at=generated_at,
        artifact_checks=artifact_checks,
        activation_payload_checks=activation_payload_checks,
        snapshot_checks=snapshot_checks,
        state_checks=state_checks,
        scheduler_checks=scheduler_checks,
        lifecycle_checks=lifecycle_checks,
        risk_checks=risk_checks,
        paper_broker_checks=paper_broker_checks,
        safety_checks=safety_checks,
        ready=ready,
        blockers=blocker_list if not ready else [],
        warnings=warning_list,
        errors=error_list,
    )


def _artifact_checks(report: dict | None) -> dict[str, bool]:
    data = _mapping(report)
    readiness = _mapping(data.get("readiness_for_stage4h6"))
    execution = _mapping(data.get("execution"))
    rollback = _mapping(data.get("rollback"))
    audit_attempted = execution.get("audit_write_attempted") is True
    audit_succeeded = execution.get("audit_write_succeeded") is True
    return {
        "stage4h5_report_present": isinstance(report, dict),
        "stage4h5_report_ready": (
            data.get("stage4h5_one_strategy_activation_executor_report") is True
            and readiness.get("ready_to_build_one_strategy_activation_acceptance_report") is True
            and data.get("success") is True
        ),
        "execution_completed": execution.get("completed") is True,
        "activation_write_succeeded": execution.get("activation_write_succeeded") is True,
        "audit_write_succeeded": (not audit_attempted) or audit_succeeded,
        "rollback_not_required": rollback.get("rollback_required") is False,
    }


def _artifact_blockers(checks: dict[str, bool], data: dict[str, Any]) -> list[str]:
    blockers = []
    labels = {
        "stage4h5_report_present": "Stage 4H-5 activation executor report is missing",
        "stage4h5_report_ready": "Stage 4H-5 activation executor report is not ready for Stage 4H-6",
        "execution_completed": "Stage 4H-5 execution did not complete",
        "activation_write_succeeded": "Stage 4H-5 activation write did not succeed",
        "audit_write_succeeded": "Stage 4H-5 audit write did not succeed after being attempted",
        "rollback_not_required": "Stage 4H-5 rollback is required",
    }
    for key, label in labels.items():
        if checks.get(key) is not True:
            blockers.append(label)
    if _as_list(data.get("errors")):
        blockers.append("Stage 4H-5 report contains errors")
    return blockers


def _activation_payload_checks(
    payload: dict[str, Any],
    selected_id: str | None,
    applied_operations: list[Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    warnings: list[str] = []
    blockers: list[str] = []
    payload_present = bool(payload)
    enabled_count = payload.get("enabled_strategy_count")
    payload_selected = payload.get("selected_strategy_id")
    active_ids = _strategy_ids_from_payload(payload)
    one_strategy_only = (
        payload_present
        and enabled_count == 1
        and (not active_ids or active_ids == [selected_id])
    )
    applied_safe, applied_visible, applied_warnings = _applied_operation_payloads_safe(
        applied_operations
    )
    warnings.extend(applied_warnings)
    if not applied_visible and applied_operations:
        warnings.append("applied operation payloads are not visible; relying on top-level activation payload")

    checks = {
        "activation_payload_present": payload_present,
        "selected_strategy_id": selected_id,
        "selected_strategy_consistent": (
            isinstance(selected_id, str)
            and bool(selected_id)
            and payload_selected == selected_id
        ),
        "paper_only": payload.get("paper_only") is True,
        "one_strategy_only": one_strategy_only,
        "enabled_strategy_count": enabled_count,
        "live_trading_disabled": payload.get("live_trading_enabled") is False,
        "all_strategies_disabled": payload.get("all_strategies_enabled") is False,
        "broker_submission_disabled": payload.get("broker_submission_enabled") is False,
        "scheduler_enabled_for_selected_strategy": (
            payload.get("scheduler_enabled_for_selected_strategy") is True
        ),
        "lifecycle_enabled_for_selected_strategy": (
            payload.get("lifecycle_enabled_for_selected_strategy") is True
        ),
        "runtime_guards_present": isinstance(payload.get("required_runtime_guards"), list),
        "monitoring_present": isinstance(payload.get("required_monitoring"), list),
        "kill_switches_present": isinstance(payload.get("required_kill_switches"), list),
        "applied_operation_payloads_safe": applied_safe,
        "activation_scope_is_single_strategy_paper_only": payload.get("activation_scope")
        == "single_strategy_paper_only",
        "automated_paper_trading_enabled": payload.get(
            "automated_paper_trading_enabled_for_selected_strategy"
        )
        is True,
    }
    expected = {
        "activation_payload_present": "activation payload is missing",
        "selected_strategy_consistent": "activation payload selected strategy does not match selected strategy",
        "paper_only": "activation payload must be paper_only true",
        "one_strategy_only": "activation payload must activate exactly one strategy",
        "live_trading_disabled": "activation payload enables live trading",
        "all_strategies_disabled": "activation payload enables all strategies",
        "broker_submission_disabled": "activation payload enables broker submission",
        "scheduler_enabled_for_selected_strategy": "selected strategy scheduler flag must be enabled",
        "lifecycle_enabled_for_selected_strategy": "selected strategy lifecycle flag must be enabled",
        "runtime_guards_present": "required runtime guards must be a JSON-safe list",
        "monitoring_present": "required monitoring must be a JSON-safe list",
        "kill_switches_present": "required kill switches must be a JSON-safe list",
        "applied_operation_payloads_safe": "applied operation payload contradicts live trading or broker submission safety",
        "activation_scope_is_single_strategy_paper_only": "activation scope must be single_strategy_paper_only",
        "automated_paper_trading_enabled": "selected strategy automated PAPER trading flag must be enabled",
    }
    for key, label in expected.items():
        if checks.get(key) is not True:
            blockers.append(label)
    return checks, blockers, warnings


def _applied_operation_payloads_safe(operations: list[Any]) -> tuple[bool, bool, list[str]]:
    visible = False
    warnings: list[str] = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        payload = operation.get("payload")
        if not isinstance(payload, dict):
            payload = operation.get("activation_payload")
        if not isinstance(payload, dict):
            payload = _mapping(operation.get("record"))
        if not payload:
            warnings.append("applied operation entry has no visible payload")
            continue
        visible = True
        if payload.get("live_trading_enabled") is True:
            return False, visible, warnings
        if payload.get("broker_submission_enabled") is True:
            return False, visible, warnings
    return True, visible, warnings


def _activation_snapshot_checks(
    snapshot: dict | None, selected_id: str | None, payload: dict[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    if not isinstance(snapshot, dict):
        return (
            {"activation_snapshot_matches": True},
            [],
            [
                "activation snapshot missing; next phase must verify activation artifact before first scheduled run"
            ],
        )
    blockers: list[str] = []
    warnings: list[str] = []
    records = _activation_records(snapshot)
    active_ids = _as_string_list(snapshot.get("active_strategy_ids"))
    explicit = bool(records or active_ids or "selected_strategy_id" in snapshot)
    matched = True
    if active_ids and active_ids != [selected_id]:
        matched = False
        blockers.append("activation snapshot active_strategy_ids do not match selected strategy")
    for record in records:
        if not isinstance(record, dict):
            warnings.append("malformed activation snapshot entry ignored")
            continue
        record_selected = record.get("selected_strategy_id")
        if record_selected is not None and record_selected != selected_id:
            matched = False
            blockers.append("activation snapshot selected strategy does not match")
        for key, expected in (
            ("paper_only", True),
            ("live_trading_enabled", False),
            ("all_strategies_enabled", False),
            ("broker_submission_enabled", False),
        ):
            if key in record and record.get(key) is not expected:
                matched = False
                blockers.append(f"activation snapshot {key} contradicts activation payload")
        if "enabled_strategy_count" in record and record.get("enabled_strategy_count") != 1:
            matched = False
            blockers.append("activation snapshot enabled_strategy_count must be 1")
    if explicit and not records and "selected_strategy_id" in snapshot:
        if snapshot.get("selected_strategy_id") != selected_id:
            matched = False
            blockers.append("activation snapshot selected_strategy_id does not match")
    if payload and snapshot.get("selected_strategy_id") not in (None, payload.get("selected_strategy_id")):
        matched = False
        blockers.append("activation snapshot contradicts Stage 4H-5 activation payload")
    return {"activation_snapshot_matches": matched}, blockers, warnings


def _activation_records(snapshot: dict[str, Any]) -> list[Any]:
    records: list[Any] = []
    if "activation_record" in snapshot:
        records.append(snapshot.get("activation_record"))
    records.extend(_as_list(snapshot.get("activations")))
    if any(key in snapshot for key in ("selected_strategy_id", "paper_only", "enabled_strategy_count")):
        records.append(snapshot)
    return records


def _audit_snapshot_checks(
    snapshot: dict | None, selected_id: str | None
) -> tuple[dict[str, Any], list[str], list[str]]:
    if not isinstance(snapshot, dict):
        return (
            {"audit_snapshot_matches": True},
            [],
            ["audit snapshot missing; manual audit verification remains required"],
        )
    blockers: list[str] = []
    warnings: list[str] = []
    matched = True
    if snapshot.get("selected_strategy_id") not in (None, selected_id):
        matched = False
        blockers.append("audit snapshot selected_strategy_id does not match")
    if snapshot.get("source_stage") not in (None, "4H-5"):
        matched = False
        blockers.append("audit snapshot source_stage must be 4H-5")
    events = _as_list(snapshot.get("events")) + _as_list(snapshot.get("activation_audit_events"))
    dict_events = [event for event in events if isinstance(event, dict)]
    if len(dict_events) != len(events):
        warnings.append("malformed audit snapshot entry ignored")
    if dict_events and not any(event.get("selected_strategy_id") == selected_id for event in dict_events):
        matched = False
        blockers.append("audit snapshot events do not reference selected strategy")
    return {"audit_snapshot_matches": matched}, blockers, warnings


def _state_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    active_intents_count = _safe_int(data.get("active_intents_count"))
    active_intents_safe = data.get("active_intents_safe_for_enablement") is True
    checks = {
        "active_halt": bool(data.get("active_halt") or data.get("halt_active")),
        "unresolved_needs_reconciliation_count": _safe_int(
            _first_present(
                data.get("unresolved_needs_reconciliation_count"),
                data.get("needs_reconciliation_count"),
                default=0,
            )
        ),
        "active_intents_count": active_intents_count,
        "active_intents_safe_for_enablement": active_intents_safe,
        "open_positions_count": _safe_int(data.get("open_positions_count")),
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("state snapshot missing; next phase must verify state immediately")
        return checks, blockers, warnings
    if checks["active_halt"]:
        blockers.append("active halt is present")
    if checks["unresolved_needs_reconciliation_count"] > 0:
        blockers.append("unresolved NEEDS_RECONCILIATION is present")
    if active_intents_count > 0 and not active_intents_safe:
        blockers.append("unsafe active intents are present")
    if active_intents_count > 0 and active_intents_safe:
        warnings.append("active intents present but explicitly marked safe for enablement")
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
        "scheduler_already_enabled": already_enabled,
        "all_strategy_scheduler_enabled": all_strategy_enabled,
        "selected_strategy_job_already_enabled": selected_enabled,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("scheduler snapshot missing; next phase remains scheduler-gated")
    if already_enabled:
        blockers.append("scheduler automation is already broadly enabled")
    if all_strategy_enabled:
        blockers.append("all-strategy scheduler automation is already enabled")
    if selected_enabled:
        blockers.append("selected strategy scheduler job is already active")
    return checks, blockers, warnings


def _lifecycle_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    already_enabled = _contains_truthy_flag(data, "lifecycle_automation_enabled") or _contains_truthy_flag(
        data, "lifecycle_wiring_enabled"
    )
    transition_enabled = _contains_truthy_flag(data, "lifecycle_transition_execution_enabled")
    checks = {
        "lifecycle_already_enabled": already_enabled,
        "lifecycle_transition_execution_enabled": transition_enabled,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("lifecycle snapshot missing; next phase remains lifecycle-gated")
    if already_enabled:
        blockers.append("lifecycle automation is already enabled")
    if transition_enabled:
        blockers.append("lifecycle transition execution is enabled")
    return checks, blockers, warnings


def _risk_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    checks = {
        "kill_switch_available": data.get("kill_switch_available") is True,
        "hard_halt_available": data.get("hard_halt_available") is True,
        "daily_loss_limit_available": data.get("daily_loss_limit_available") is True,
        "risk_bypass_enabled": data.get("risk_bypass_enabled") is True,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        warnings.append("risk snapshot missing; next phase must verify risk immediately")
        return checks, blockers, warnings
    if checks["risk_bypass_enabled"]:
        blockers.append("risk bypass is enabled")
    for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
        if checks[key] is not True:
            blockers.append(f"{key} must be true in supplied risk snapshot")
    return checks, blockers, warnings


def _paper_broker_checks(snapshot: dict | None) -> tuple[dict[str, Any], list[str], list[str]]:
    present, data = _optional_mapping(snapshot)
    mode = data.get("mode")
    port = data.get("ibkr_port")
    checks = {
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
        warnings.append("paper broker snapshot missing; next phase must verify paper config immediately")
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


def _safety_checks(*sources: Any) -> dict[str, bool]:
    return {
        "no_live_orders": not _contains_truthy_flag(sources, "live_orders_enabled"),
        "no_market_data": not _contains_truthy_flag(sources, "market_data_enabled"),
        "no_contract_qualification": not _contains_truthy_flag(
            sources, "contract_qualification_enabled"
        ),
        "no_broker_submission_enabled": not _contains_truthy_flag(
            sources, "broker_submission_enabled"
        ),
        "no_all_strategy_enablement": not (
            _contains_truthy_flag(sources, "all_strategies_enabled")
            or _contains_truthy_flag(sources, "enable_all_strategies")
        ),
        "no_live_trading": not _contains_truthy_flag(sources, "live_trading_enabled"),
    }


def _safety_blockers(checks: dict[str, bool]) -> list[str]:
    labels = {
        "no_live_orders": "live orders safety flag is enabled",
        "no_market_data": "market data safety flag is enabled",
        "no_contract_qualification": "contract qualification safety flag is enabled",
        "no_broker_submission_enabled": "broker submission safety flag is enabled",
        "no_all_strategy_enablement": "all-strategy safety flag is enabled",
        "no_live_trading": "live trading safety flag is enabled",
    }
    return [label for key, label in labels.items() if checks.get(key) is not True]


def _base_report(
    *,
    generated_at: str,
    artifact_checks: dict[str, Any],
    activation_payload_checks: dict[str, Any],
    snapshot_checks: dict[str, Any],
    state_checks: dict[str, Any],
    scheduler_checks: dict[str, Any],
    lifecycle_checks: dict[str, Any],
    risk_checks: dict[str, Any],
    paper_broker_checks: dict[str, Any],
    safety_checks: dict[str, Any],
    ready: bool,
    blockers: list[str],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "stage4h6_one_strategy_activation_acceptance_report": True,
        "generated_at": generated_at,
        "artifact_checks": artifact_checks,
        "activation_payload_checks": activation_payload_checks,
        "snapshot_checks": snapshot_checks,
        "state_checks": state_checks,
        "scheduler_checks": scheduler_checks,
        "lifecycle_checks": lifecycle_checks,
        "risk_checks": risk_checks,
        "paper_broker_checks": paper_broker_checks,
        "safety_checks": safety_checks,
        "readiness_for_next_phase": {
            "ready_to_build_first_scheduled_paper_automation_run": ready,
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
    for value in (
        selected.get("selected_strategy_id"),
        _mapping(report.get("activation_payload")).get("selected_strategy_id"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _strategy_ids_from_payload(payload: dict[str, Any]) -> list[str]:
    for key in ("active_strategy_ids", "enabled_strategy_ids", "activated_strategy_ids"):
        values = _as_string_list(payload.get(key))
        if values:
            return values
    return []


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
        "stage4h5_report_present": False,
        "stage4h5_report_ready": False,
        "execution_completed": False,
        "activation_write_succeeded": False,
        "audit_write_succeeded": False,
        "rollback_not_required": False,
    }


def _default_activation_payload_checks() -> dict[str, Any]:
    return {
        "activation_payload_present": False,
        "selected_strategy_id": None,
        "selected_strategy_consistent": False,
        "paper_only": False,
        "one_strategy_only": False,
        "enabled_strategy_count": None,
        "live_trading_disabled": False,
        "all_strategies_disabled": False,
        "broker_submission_disabled": False,
        "scheduler_enabled_for_selected_strategy": False,
        "lifecycle_enabled_for_selected_strategy": False,
        "runtime_guards_present": False,
        "monitoring_present": False,
        "kill_switches_present": False,
        "applied_operation_payloads_safe": False,
    }


def _default_snapshot_checks() -> dict[str, bool]:
    return {
        "activation_snapshot_present": False,
        "activation_snapshot_matches": False,
        "audit_snapshot_present": False,
        "audit_snapshot_matches": False,
        "state_snapshot_present": False,
        "scheduler_snapshot_present": False,
        "lifecycle_snapshot_present": False,
        "risk_snapshot_present": False,
        "paper_broker_snapshot_present": False,
    }


def _default_state_checks() -> dict[str, Any]:
    return {
        "active_halt": False,
        "unresolved_needs_reconciliation_count": 0,
        "active_intents_count": 0,
        "active_intents_safe_for_enablement": False,
        "open_positions_count": 0,
    }


def _default_scheduler_checks() -> dict[str, bool]:
    return {
        "scheduler_already_enabled": False,
        "all_strategy_scheduler_enabled": False,
        "selected_strategy_job_already_enabled": False,
    }


def _default_lifecycle_checks() -> dict[str, bool]:
    return {
        "lifecycle_already_enabled": False,
        "lifecycle_transition_execution_enabled": False,
    }


def _default_risk_checks() -> dict[str, bool]:
    return {
        "kill_switch_available": False,
        "hard_halt_available": False,
        "daily_loss_limit_available": False,
        "risk_bypass_enabled": False,
    }


def _default_paper_broker_checks() -> dict[str, Any]:
    return {
        "mode": None,
        "paper_trading": None,
        "ibkr_port": None,
        "paper_config_valid": False,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
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
