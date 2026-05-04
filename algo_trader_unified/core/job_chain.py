"""One-shot dry-run scheduler job chain runner."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.scheduler import (
    JOB_INTENT_CONFIRMATION,
    JOB_INTENT_FILL_CONFIRMATION,
    JOB_INTENT_SUBMISSION,
    JOB_POSITION_TRANSITIONS,
    JOB_S01_MANAGEMENT_SCAN,
    JOB_S01_VOL_SCAN,
    JOB_S02_MANAGEMENT_SCAN,
    JOB_S02_VOL_SCAN,
)


_STRATEGY_SCAN_JOBS = {
    S01_VOL_BASELINE: JOB_S01_VOL_SCAN,
    S02_VOL_ENHANCED: JOB_S02_VOL_SCAN,
}
_STRATEGY_MANAGEMENT_JOBS = {
    S01_VOL_BASELINE: JOB_S01_MANAGEMENT_SCAN,
    S02_VOL_ENHANCED: JOB_S02_MANAGEMENT_SCAN,
}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _step_status(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("errors_count", 0):
            return "completed_with_errors"
        return str(result.get("status") or "completed")
    status = getattr(result, "status", None)
    if isinstance(status, str):
        return status
    return "completed"


def _step_entry(*, step: str, job_id: str, result: Any) -> dict[str, Any]:
    return {
        "step": step,
        "job_id": job_id,
        "status": _step_status(result),
        "result": _json_safe(result),
    }


def _error_entry(*, step: str, job_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "step": step,
        "job_id": job_id,
        "error": str(exc),
        "error_type": type(exc).__name__,
    }


def _strategy_ids(strategy_id: str | None) -> list[str]:
    if strategy_id is None:
        return [S01_VOL_BASELINE, S02_VOL_ENHANCED]
    if strategy_id in {S01_VOL_BASELINE, S02_VOL_ENHANCED}:
        return [strategy_id]
    return []


def _skip(result: dict[str, Any], *, step: str, reason: str, job_id: str | None = None) -> None:
    entry = {"step": step, "reason": reason}
    if job_id is not None:
        entry["job_id"] = job_id
    result["steps_skipped"].append(entry)


def run_dry_run_job_chain(
    *,
    scheduler,
    state_store,
    ledger,
    now,
    strategy_id: str | None = None,
    signal_context_provider=None,
    management_signal_provider=None,
    execution_adapter=None,
    include_entry_scan: bool = True,
    include_management_scan: bool = True,
    include_submission: bool = True,
    include_confirmation: bool = True,
    include_fill_confirmation: bool = True,
    include_position_transitions: bool = True,
    fail_fast: bool = False,
) -> dict:
    result = {
        "dry_run": True,
        "status": "completed",
        "strategy_id": strategy_id,
        "steps_run": [],
        "steps_skipped": [],
        "errors_count": 0,
        "errors": [],
        "summary": {
            "entry_scan_runs": 0,
            "management_scan_runs": 0,
            "submission_runs": 0,
            "confirmation_runs": 0,
            "fill_confirmation_runs": 0,
            "position_transition_runs": 0,
        },
    }

    def run_step(step: str, job_id: str, **kwargs) -> None:
        try:
            step_result = scheduler.run_job_once(job_id, **kwargs)
        except Exception as exc:
            result["errors_count"] += 1
            result["errors"].append(_error_entry(step=step, job_id=job_id, exc=exc))
            if fail_fast:
                raise
            return
        result["steps_run"].append(_step_entry(step=step, job_id=job_id, result=step_result))

    if include_entry_scan:
        scan_strategy_ids = _strategy_ids(strategy_id)
        if signal_context_provider is None:
            _skip(result, step="entry_scan", reason="missing_signal_context_provider")
        elif not scan_strategy_ids:
            _skip(result, step="entry_scan", reason="unsupported_strategy_id")
        else:
            for scan_strategy_id in scan_strategy_ids:
                job_id = _STRATEGY_SCAN_JOBS[scan_strategy_id]
                run_step(
                    "entry_scan",
                    job_id,
                    state_store=state_store,
                    ledger=ledger,
                    current_time=now,
                    signal_context_provider=signal_context_provider,
                )
                result["summary"]["entry_scan_runs"] += 1
    else:
        _skip(result, step="entry_scan", reason="disabled")

    if include_management_scan:
        management_strategy_ids = _strategy_ids(strategy_id)
        if not management_strategy_ids:
            _skip(result, step="management_scan", reason="unsupported_strategy_id")
        else:
            for management_strategy_id in management_strategy_ids:
                job_id = _STRATEGY_MANAGEMENT_JOBS[management_strategy_id]
                kwargs = {
                    "state_store": state_store,
                    "ledger": ledger,
                    "now": now if isinstance(now, str) else now.isoformat(),
                }
                if management_signal_provider is not None:
                    kwargs["management_signal_provider"] = management_signal_provider
                run_step("management_scan", job_id, **kwargs)
                result["summary"]["management_scan_runs"] += 1
    else:
        _skip(result, step="management_scan", reason="disabled")

    downstream_kwargs = {
        "state_store": state_store,
        "ledger": ledger,
        "now": now if isinstance(now, str) else now.isoformat(),
    }
    if strategy_id is not None:
        downstream_kwargs["strategy_id"] = strategy_id
    if execution_adapter is not None:
        downstream_kwargs["execution_adapter"] = execution_adapter

    if include_submission:
        run_step("intent_submission", JOB_INTENT_SUBMISSION, **downstream_kwargs)
        result["summary"]["submission_runs"] += 1
    else:
        _skip(result, step="intent_submission", job_id=JOB_INTENT_SUBMISSION, reason="disabled")

    if include_confirmation:
        run_step("intent_confirmation", JOB_INTENT_CONFIRMATION, **downstream_kwargs)
        result["summary"]["confirmation_runs"] += 1
    else:
        _skip(
            result,
            step="intent_confirmation",
            job_id=JOB_INTENT_CONFIRMATION,
            reason="disabled",
        )

    if include_fill_confirmation:
        run_step("intent_fill_confirmation", JOB_INTENT_FILL_CONFIRMATION, **downstream_kwargs)
        result["summary"]["fill_confirmation_runs"] += 1
    else:
        _skip(
            result,
            step="intent_fill_confirmation",
            job_id=JOB_INTENT_FILL_CONFIRMATION,
            reason="disabled",
        )

    position_kwargs = {
        "state_store": state_store,
        "ledger": ledger,
        "now": downstream_kwargs["now"],
    }
    if strategy_id is not None:
        position_kwargs["strategy_id"] = strategy_id
    if include_position_transitions:
        run_step("position_transitions", JOB_POSITION_TRANSITIONS, **position_kwargs)
        result["summary"]["position_transition_runs"] += 1
    else:
        _skip(
            result,
            step="position_transitions",
            job_id=JOB_POSITION_TRANSITIONS,
            reason="disabled",
        )

    if result["errors_count"]:
        result["status"] = "completed_with_errors"
    return result
