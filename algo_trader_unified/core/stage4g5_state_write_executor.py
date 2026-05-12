"""Pure Stage 4G-5 manual paper lifecycle state write executor."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable


DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"
REQUIRED_ACKNOWLEDGEMENTS = [
    "I understand this will write paper lifecycle state.",
    "I understand this will write ledger events.",
    "I understand this is still PAPER only.",
    "I understand this does not enable scheduler automation.",
    "I reviewed the proposed StateStore and ledger payloads.",
]
WRITE_PLAN_KEYS = (
    "state_store_write_enabled",
    "ledger_write_enabled",
    "lifecycle_transition_enabled",
    "daemon_wiring_enabled",
    "scheduler_wiring_enabled",
)
SAFETY_RESULT_KEYS = (
    "no_live_orders",
    "no_market_data",
    "no_contract_qualification",
    "no_scheduler_changes",
    "no_lifecycle_wiring",
    "no_state_mutation",
    "no_ledger_writes",
)
UNSAFE_TRUE_FLAG_KEYS = (
    "live_orders_enabled",
    "market_data_enabled",
    "contract_qualification_enabled",
    "scheduler_changes_enabled",
    "daemon_wiring_enabled",
    "scheduler_wiring_enabled",
    "lifecycle_wiring_enabled",
    "state_mutation_enabled",
    "ledger_writes_enabled",
    "state_store_write_enabled",
    "ledger_write_enabled",
    "lifecycle_transition_enabled",
    "live_override",
    "allow_live_override",
    "scheduler_override",
    "lifecycle_override",
    "scheduler_lifecycle_override",
)
STATE_OPERATIONS = ("upsert_order", "upsert_position")
ORDERED_NEXT_STEPS = [
    "Build Stage 4G-6 manual lifecycle write acceptance report.",
    "Verify the written StateStore and ledger records manually.",
    "Keep scheduler/lifecycle automation disabled.",
    "Do not begin live trading.",
]
DO_NOT_DO_YET = [
    "Do not enable automated paper execution.",
    "Do not enable scheduler or lifecycle automation.",
    "Do not place live orders.",
    "Do not begin live trading.",
]


def build_stage4g5_state_write_executor_report(
    *,
    state_write_dry_run_report: dict | None,
    state_store_writer,
    ledger_writer,
    operator_acknowledgements: list[str] | None = None,
    allow_state_write: bool = False,
    allow_ledger_write: bool = False,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """Apply a Stage 4G-4 dry-run packet through injected writer abstractions only."""

    try:
        return _build_report(
            state_write_dry_run_report=state_write_dry_run_report,
            state_store_writer=state_store_writer,
            ledger_writer=ledger_writer,
            operator_acknowledgements=operator_acknowledgements,
            allow_state_write=allow_state_write,
            allow_ledger_write=allow_ledger_write,
            now_provider=now_provider,
        )
    except Exception as exc:  # noqa: BLE001 - report boundary must stay data-only.
        return _json_safe(
            _base_report(
                generated_at=_generated_at(now_provider),
                allow_state_write=allow_state_write,
                allow_ledger_write=allow_ledger_write,
                acknowledgement_checks=_acknowledgement_checks(
                    operator_acknowledgements
                ),
                gates={
                    "allow_state_write": allow_state_write is True,
                    "allow_ledger_write": allow_ledger_write is True,
                    "dry_run_report_valid": False,
                    "dry_run_ready": False,
                    "acknowledgements_ok": False,
                    "write_plan_safe": False,
                    "safety_flags_clean": False,
                    "passed": False,
                    "reasons": ["unexpected report failure"],
                },
                execution=_execution(
                    attempted=False,
                    failure_reason=f"unexpected report failure: {type(exc).__name__}: {exc}",
                ),
                errors=[f"unexpected report failure: {type(exc).__name__}: {exc}"],
                warnings=[],
            )
        )


def _build_report(
    *,
    state_write_dry_run_report: dict | None,
    state_store_writer: Any,
    ledger_writer: Any,
    operator_acknowledgements: list[str] | None,
    allow_state_write: bool,
    allow_ledger_write: bool,
    now_provider: Callable[[], Any] | None,
) -> dict:
    generated_at = _generated_at(now_provider)
    report = state_write_dry_run_report if isinstance(state_write_dry_run_report, dict) else None
    acknowledgements = _acknowledgement_checks(operator_acknowledgements)
    operations = _deterministic_operations(report)
    gates, errors, warnings = _gate_checks(
        report=report,
        raw_report=state_write_dry_run_report,
        operations=operations,
        acknowledgements=acknowledgements,
        allow_state_write=allow_state_write,
        allow_ledger_write=allow_ledger_write,
    )

    if not gates["passed"]:
        return _json_safe(
            _base_report(
                generated_at=generated_at,
                allow_state_write=allow_state_write,
                allow_ledger_write=allow_ledger_write,
                acknowledgement_checks=acknowledgements,
                gates=gates,
                execution=_execution(
                    attempted=False,
                    failure_reason="; ".join(gates["reasons"]) or None,
                ),
                skipped_operations=[
                    _operation_summary(item, skip_reason="gate refused execution")
                    for item in operations
                    if item.get("target") != "Lifecycle"
                ],
                errors=errors,
                warnings=warnings,
            )
        )

    execution = _execution(attempted=True)
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    executable = [item for item in operations if item.get("target") != "Lifecycle"]
    failed_index: int | None = None
    failure_reason: str | None = None

    for index, item in enumerate(executable):
        target = item.get("target")
        operation = item.get("operation")
        payload = _mapping(item.get("payload"))
        try:
            if target == "StateStore":
                execution["state_store_write_attempted"] = True
                method = getattr(state_store_writer, str(operation))
                result = method(payload)
            elif target == "Ledger" and operation == "append_event":
                execution["ledger_write_attempted"] = True
                result = ledger_writer.append_event(payload)
            else:
                raise ValueError(f"unrecognized executable operation: {target}.{operation}")
            sanitized_result = _json_safe(result)
            _validate_idempotent_result(payload, sanitized_result, operation=str(operation))
            applied.append(
                _operation_summary(item, result=_result_mapping(sanitized_result))
            )
        except Exception as exc:  # noqa: BLE001 - writer failures must be reported.
            failure_reason = f"{type(exc).__name__}: {exc}"
            failed_index = index
            execution["failed_step"] = _step_name(item)
            execution["failure_reason"] = failure_reason
            errors.append(f"write failed at {_step_name(item)}: {failure_reason}")
            break

    if failed_index is not None:
        for item in executable[failed_index:]:
            reason = (
                f"failed write: {failure_reason}"
                if item is executable[failed_index]
                else "abandoned after prior write failure"
            )
            skipped.append(_operation_summary(item, skip_reason=reason))

    state_ops = [item for item in executable if item.get("target") == "StateStore"]
    ledger_ops = [item for item in executable if item.get("target") == "Ledger"]
    applied_sequences = {item["sequence_number"] for item in applied}
    execution["state_store_write_succeeded"] = bool(state_ops) and all(
        item.get("sequence_number") in applied_sequences for item in state_ops
    )
    execution["ledger_write_succeeded"] = bool(ledger_ops) and all(
        item.get("sequence_number") in applied_sequences for item in ledger_ops
    )
    execution["completed"] = (
        failed_index is None
        and execution["state_store_write_succeeded"]
        and execution["ledger_write_succeeded"]
    )

    rollback_required = bool(applied and failed_index is not None)
    return _json_safe(
        _base_report(
            generated_at=generated_at,
            allow_state_write=allow_state_write,
            allow_ledger_write=allow_ledger_write,
            acknowledgement_checks=acknowledgements,
            gates=gates,
            execution=execution,
            applied_operations=applied,
            skipped_operations=skipped,
            rollback_required=rollback_required,
            errors=errors,
            warnings=warnings,
        )
    )


def _gate_checks(
    *,
    report: dict[str, Any] | None,
    raw_report: Any,
    operations: list[dict[str, Any]],
    acknowledgements: dict[str, Any],
    allow_state_write: bool,
    allow_ledger_write: bool,
) -> tuple[dict[str, Any], list[str], list[str]]:
    reasons: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    if raw_report is not None and report is None:
        errors.append("state_write_dry_run_report must be a dict")
    if allow_state_write is not True:
        reasons.append("allow_state_write must be True")
    if allow_ledger_write is not True:
        reasons.append("allow_ledger_write must be True")
    if report is None:
        reasons.append("state_write_dry_run_report missing")

    dry_run_report_valid = _dry_run_report_valid(report)
    dry_run_ready = _dry_run_ready(report)
    write_plan_safe = _write_plan_safe(report)
    safety_flags_clean = _safety_flags_clean(report)
    operations_ok, operation_reasons = _operations_gate_reasons(operations)
    if not dry_run_report_valid:
        reasons.append("Stage 4G-4 dry-run report is missing or malformed")
    if not dry_run_ready:
        reasons.append("Stage 4G-4 dry-run report is not ready for Stage 4G-5")
    if not operations_ok:
        reasons.extend(operation_reasons)
    if not acknowledgements["exact_match"]:
        reasons.append("required operator acknowledgements missing")
    if not write_plan_safe:
        reasons.append("Stage 4G-4 write_plan flags must all be False")
    if not safety_flags_clean:
        reasons.append("Stage 4G-4 safety flags are not clean")
    if _contains_any_true_flag(report, ("live_override", "allow_live_override")):
        reasons.append("live override flag is forbidden")
    if _contains_any_true_flag(
        report,
        ("scheduler_override", "lifecycle_override", "scheduler_lifecycle_override"),
    ):
        reasons.append("scheduler/lifecycle override flag is forbidden")

    gates = {
        "allow_state_write": allow_state_write is True,
        "allow_ledger_write": allow_ledger_write is True,
        "dry_run_report_valid": dry_run_report_valid,
        "dry_run_ready": dry_run_ready and operations_ok,
        "acknowledgements_ok": acknowledgements["exact_match"],
        "write_plan_safe": write_plan_safe,
        "safety_flags_clean": safety_flags_clean,
        "passed": False,
        "reasons": _dedupe(reasons),
    }
    gates["passed"] = (
        gates["allow_state_write"]
        and gates["allow_ledger_write"]
        and gates["dry_run_report_valid"]
        and gates["dry_run_ready"]
        and gates["acknowledgements_ok"]
        and gates["write_plan_safe"]
        and gates["safety_flags_clean"]
        and not gates["reasons"]
    )
    return gates, _dedupe(errors), _dedupe(warnings)


def _dry_run_report_valid(report: dict[str, Any] | None) -> bool:
    packet = _mapping(_mapping(report).get("dry_run_packet"))
    checks = _mapping(_mapping(report).get("operation_schema_checks"))
    return (
        report is not None
        and report.get("stage4g4_state_write_dry_run") is True
        and packet.get("available") is True
        and checks.get("operations_structured") is True
        and checks.get("recognized_operations") is True
    )


def _dry_run_ready(report: dict[str, Any] | None) -> bool:
    readiness = _mapping(_mapping(report).get("readiness_for_stage4g5"))
    return readiness.get("ready_to_build_manual_state_write_executor") is True


def _write_plan_safe(report: dict[str, Any] | None) -> bool:
    write_plan = _mapping(_mapping(report).get("write_plan"))
    return all(write_plan.get(key) is False for key in WRITE_PLAN_KEYS)


def _safety_flags_clean(report: dict[str, Any] | None) -> bool:
    safety = _mapping(_mapping(report).get("safety_checks"))
    if not all(safety.get(key) is True for key in SAFETY_RESULT_KEYS):
        return False
    return not _contains_any_true_flag(report, UNSAFE_TRUE_FLAG_KEYS)


def _operations_gate_reasons(
    operations: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not operations:
        reasons.append("dry_run_operations are required")
        return False, reasons
    if any(item.get("would_execute") is not False for item in operations):
        reasons.append("all dry_run_operations must have would_execute=False")
    if not _operations_are_ordered(operations):
        reasons.append("dry_run_operations are not deterministically ordered")
    for item in operations:
        target = item.get("target")
        operation = item.get("operation")
        payload = item.get("payload")
        if target == "StateStore" and operation not in STATE_OPERATIONS:
            reasons.append("unrecognized StateStore operation")
        if target == "StateStore" and not isinstance(payload, dict):
            reasons.append("StateStore operation payload must be a JSON-safe dict")
        if target == "Ledger" and operation != "append_event":
            reasons.append("unrecognized Ledger operation")
        if target == "Ledger" and not isinstance(payload, dict):
            reasons.append("Ledger event payload must be a JSON-safe dict")
        if target == "Lifecycle" and operation != "record_lifecycle_transition":
            reasons.append("unrecognized Lifecycle operation")
        if target not in ("StateStore", "Ledger", "Lifecycle"):
            reasons.append("unrecognized operation target")
    return not reasons, _dedupe(reasons)


def _operations_are_ordered(operations: list[dict[str, Any]]) -> bool:
    phases = []
    for item in operations:
        target = item.get("target")
        operation = item.get("operation")
        if target == "StateStore" and operation == "upsert_order":
            phases.append(0)
        elif target == "StateStore" and operation == "upsert_position":
            phases.append(1)
        elif target == "Ledger" and operation == "append_event":
            phases.append(2)
        elif target == "Lifecycle" and operation == "record_lifecycle_transition":
            phases.append(3)
        else:
            return False
    return phases == sorted(phases) and 3 in phases


def _deterministic_operations(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    raw_operations = _as_list(
        _mapping(_mapping(report).get("dry_run_packet")).get("dry_run_operations")
    )
    operations: list[dict[str, Any]] = []
    for index, item in enumerate(raw_operations):
        if isinstance(item, dict):
            operations.append(dict(item))
        else:
            operations.append(
                {
                    "sequence_number": index + 1,
                    "target": "<malformed>",
                    "operation": "<malformed>",
                    "payload": None,
                    "would_execute": None,
                }
            )
    return operations


def _validate_idempotent_result(
    payload: dict[str, Any],
    result: Any,
    *,
    operation: str,
) -> None:
    if not isinstance(result, dict):
        return
    status = str(result.get("status", "")).lower()
    duplicate_markers = {"already_exists", "duplicate", "idempotent"}
    if status not in duplicate_markers and result.get("duplicate") is not True:
        return
    record = _mapping(result.get("record"))
    for key in ("client_order_id", "broker_order_id"):
        expected = payload.get(key)
        if _present(expected) and str(record.get(key)) != str(expected):
            raise ValueError(
                f"{operation} idempotent result mismatched {key}: "
                f"requested {expected}, got {record.get(key)}"
            )


def _operation_summary(
    item: dict[str, Any],
    *,
    result: dict[str, Any] | None = None,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    payload = _mapping(item.get("payload"))
    summary = {
        "sequence_number": item.get("sequence_number"),
        "target": item.get("target"),
        "operation": item.get("operation"),
        "client_order_id": payload.get("client_order_id"),
        "broker_order_id": payload.get("broker_order_id"),
    }
    if result is not None:
        summary["result"] = result
    if skip_reason is not None:
        summary["skip_reason"] = skip_reason
    return summary


def _base_report(
    *,
    generated_at: str,
    allow_state_write: bool,
    allow_ledger_write: bool,
    acknowledgement_checks: dict[str, Any],
    gates: dict[str, Any],
    execution: dict[str, Any],
    applied_operations: list[dict[str, Any]] | None = None,
    skipped_operations: list[dict[str, Any]] | None = None,
    rollback_required: bool = False,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    errors = list(errors or [])
    warnings = list(warnings or [])
    ready = (
        gates.get("passed") is True
        and execution.get("attempted") is True
        and execution.get("state_store_write_succeeded") is True
        and execution.get("ledger_write_succeeded") is True
        and execution.get("lifecycle_transition_executed") is False
        and rollback_required is False
        and not errors
        and gates.get("safety_flags_clean") is True
    )
    blockers: list[str] = []
    if not ready:
        blockers.extend(gates.get("reasons") or [])
        if rollback_required:
            blockers.append("manual rollback is required before Stage 4G-6 readiness")
        if execution.get("state_store_write_succeeded") is not True:
            blockers.append("StateStore writes did not complete successfully")
        if execution.get("ledger_write_succeeded") is not True:
            blockers.append("ledger writes did not complete successfully")
        if errors:
            blockers.append("errors must be resolved before Stage 4G-6")
    return {
        "dry_run": False,
        "stage4g5_state_write_executor": True,
        "generated_at": generated_at,
        "gates": gates,
        "acknowledgement_checks": acknowledgement_checks,
        "execution": execution,
        "applied_operations": list(applied_operations or []),
        "skipped_operations": list(skipped_operations or []),
        "rollback": {
            "rollback_required": rollback_required,
            "rollback_attempted": False,
            "rollback_status": (
                "manual rollback is required using standard backups"
                if rollback_required
                else "no rollback required"
            ),
            "rollback_limitations": "no automated rollback is supported in this phase",
        },
        "write_plan": {
            "state_store_write_enabled": allow_state_write is True and gates.get("passed") is True,
            "ledger_write_enabled": allow_ledger_write is True and gates.get("passed") is True,
            "lifecycle_transition_enabled": False,
            "daemon_wiring_enabled": False,
            "scheduler_wiring_enabled": False,
        },
        "safety": {
            "paper_state_write_enabled": allow_state_write is True and gates.get("passed") is True,
            "paper_ledger_write_enabled": allow_ledger_write is True and gates.get("passed") is True,
            "live_orders_enabled": False,
            "market_data_enabled": False,
            "contract_qualification_enabled": False,
            "scheduler_changes_enabled": False,
            "lifecycle_wiring_enabled": False,
            "automated_paper_trading_enabled": False,
        },
        "readiness_for_stage4g6": {
            "ready_to_build_manual_lifecycle_write_acceptance_report": ready,
            "blockers": _dedupe(blockers),
            "warnings": list(warnings),
        },
        "recommendations": {
            "ordered_next_steps": list(ORDERED_NEXT_STEPS),
            "do_not_do_yet": list(DO_NOT_DO_YET),
        },
        "success": True,
        "errors": errors,
        "warnings": warnings,
    }


def _execution(
    *,
    attempted: bool,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "attempted": attempted,
        "state_store_write_attempted": False,
        "ledger_write_attempted": False,
        "state_store_write_succeeded": False,
        "ledger_write_succeeded": False,
        "lifecycle_transition_executed": False,
        "completed": False,
        "failed_step": None,
        "failure_reason": failure_reason,
    }


def _result_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {"value": value}


def _step_name(item: dict[str, Any]) -> str:
    return f"{item.get('target')}.{item.get('operation')}#{item.get('sequence_number')}"


def _acknowledgement_checks(values: list[str] | None) -> dict[str, Any]:
    provided = [
        value.strip()
        for value in values
        if isinstance(value, str)
    ] if isinstance(values, list) else []
    missing = [value for value in REQUIRED_ACKNOWLEDGEMENTS if value not in provided]
    return {
        "required": list(REQUIRED_ACKNOWLEDGEMENTS),
        "provided": provided,
        "missing": missing,
        "exact_match": not missing,
    }


def _contains_any_true_flag(value: Any, keys: tuple[str, ...]) -> bool:
    if isinstance(value, dict):
        return any(
            (item_key in keys and item is True) or _contains_any_true_flag(item, keys)
            for item_key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_any_true_flag(item, keys) for item in value)
    return False


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _present(value: Any) -> bool:
    return value not in (None, "")


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
