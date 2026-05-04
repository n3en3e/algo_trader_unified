"""Read-only dry-run readiness report for operator checks."""

from __future__ import annotations

import importlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from algo_trader_unified.config.risk import ORDER_INTENT_TTL_MINUTES
from algo_trader_unified.config.scheduler import (
    JOB_INTENT_CONFIRMATION,
    JOB_INTENT_FILL_CONFIRMATION,
    JOB_INTENT_SUBMISSION,
    JOB_POSITION_TRANSITIONS,
    JOB_S01_MANAGEMENT_SCAN,
    JOB_S01_VOL_SCAN,
    JOB_S02_MANAGEMENT_SCAN,
    JOB_S02_VOL_SCAN,
    JOB_SPECS,
)
from algo_trader_unified.core.ledger_paths import (
    EXECUTION_LEDGER_RELATIVE_PATH,
    LEDGER_DIR_RELATIVE_PATH,
    ORDER_LEDGER_RELATIVE_PATH,
)
from algo_trader_unified.core.ledger_reader import LedgerReadError, LedgerReader
from algo_trader_unified.core.state_store import (
    ACTIVE_CLOSE_INTENT_STATUSES,
    StateStoreCorruptError,
)
from algo_trader_unified.tools.system_status import build_summary


REQUIRED_JOB_IDS = (
    JOB_S01_VOL_SCAN,
    JOB_S01_MANAGEMENT_SCAN,
    JOB_INTENT_SUBMISSION,
    JOB_INTENT_CONFIRMATION,
    JOB_INTENT_FILL_CONFIRMATION,
    JOB_POSITION_TRANSITIONS,
)
OPTIONAL_S02_JOB_IDS = (JOB_S02_VOL_SCAN, JOB_S02_MANAGEMENT_SCAN)
FORBIDDEN_JOB_FRAGMENTS = (
    "live_broker_submission",
    "live_order_submission",
    "auto_live_submit",
    "auto_live_fill",
    "auto_position_adjust",
)
REQUIRED_FUNCTIONS = (
    ("algo_trader_unified.core.job_chain", "run_dry_run_job_chain"),
    ("algo_trader_unified.core.management", "run_management_scan"),
    ("algo_trader_unified.jobs.submission", "run_intent_submission_job"),
    ("algo_trader_unified.jobs.confirmation", "run_intent_confirmation_job"),
    ("algo_trader_unified.jobs.fill_confirmation", "run_intent_fill_confirmation_job"),
    ("algo_trader_unified.jobs.position_transitions", "run_position_transitions_job"),
    ("algo_trader_unified.tools.run_dry_run_chain", "main"),
)
DRY_RUN_ORCHESTRATION_FILES = (
    Path("algo_trader_unified/tools/run_dry_run_chain.py"),
    Path("algo_trader_unified/core/job_chain.py"),
    Path("algo_trader_unified/jobs/submission.py"),
    Path("algo_trader_unified/jobs/confirmation.py"),
    Path("algo_trader_unified/jobs/fill_confirmation.py"),
    Path("algo_trader_unified/jobs/position_transitions.py"),
)
FORBIDDEN_SOURCE_SNIPPETS = (
    "scheduler" + ".start(",
    "ib_" + "insync",
    "yf" + "inance",
    "req" + "uests",
    "place" + "Order",
    "cancel" + "Order",
)
PENDING_STATUSES = ("created", "submitted", "confirmed", "filled")


def build_dry_run_readiness_report(
    *,
    root_dir,
    state_store,
    ledger_reader=None,
    now,
    job_specs=None,
) -> dict:
    root = Path(root_dir)
    checked_at = _normalize_now(now).isoformat()
    report = {
        "dry_run": True,
        "ready": False,
        "status": "blocked",
        "checked_at": checked_at,
        "checks": [],
        "blocking_issues": [],
        "warnings": [],
        "summary": _empty_summary(),
        "next_action": "inspect_state_file",
    }

    specs = JOB_SPECS if job_specs is None else job_specs
    _check_scheduler_jobs(report, specs)
    _check_required_functions(report)
    state_loaded = _check_state_store(report, state_store)
    _check_ledgers(report, root, ledger_reader)
    _check_live_safety(report, root)

    if state_loaded:
        order_intents, positions, close_intents = _records_from_store(state_store)
        report["summary"] = build_summary(order_intents, positions, close_intents)
        report["summary"]["active_close_intents_count"] = _active_close_intent_count(
            close_intents
        )
        report["summary"]["active_order_intents_count"] = _active_order_intent_count(
            order_intents
        )
        _check_pending_lifecycle(report, order_intents, close_intents, checked_at)
        _check_inconsistent_state(report, order_intents, positions, close_intents)
    else:
        _warn(
            report,
            "empty_state",
            "No StateStore file is present yet; fresh dry-run state has no lifecycle work.",
        )

    _finish_report(report)
    return _json_safe(report)


def _normalize_now(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("now must be a datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _empty_summary() -> dict[str, Any]:
    return build_summary([], [], []) | {
        "active_close_intents_count": 0,
        "active_order_intents_count": 0,
    }


def _check(report: dict, name: str, status: str, message: str, **details: Any) -> None:
    entry = {"name": name, "status": status, "message": message}
    entry.update({key: value for key, value in details.items() if value is not None})
    report["checks"].append(entry)


def _block(report: dict, code: str, message: str, **details: Any) -> None:
    issue = {"code": code, "message": message}
    issue.update({key: value for key, value in details.items() if value is not None})
    report["blocking_issues"].append(issue)


def _warn(report: dict, code: str, message: str, **details: Any) -> None:
    warning = {"code": code, "message": message}
    warning.update({key: value for key, value in details.items() if value is not None})
    report["warnings"].append(warning)


def _check_scheduler_jobs(report: dict, job_specs: dict[str, Any]) -> None:
    missing = [job_id for job_id in REQUIRED_JOB_IDS if job_id not in job_specs]
    for job_id in missing:
        _block(report, "missing_required_job", "Required scheduler job is missing.", job_id=job_id)
    optional_missing = [job_id for job_id in OPTIONAL_S02_JOB_IDS if job_id not in job_specs]
    for job_id in optional_missing:
        _warn(report, "missing_optional_s02_job", "Optional S02 scheduler job is missing.", job_id=job_id)
    forbidden = [
        job_id
        for job_id in job_specs
        if any(fragment in str(job_id).lower() for fragment in FORBIDDEN_JOB_FRAGMENTS)
    ]
    for job_id in forbidden:
        _block(report, "forbidden_scheduler_job", "Forbidden live or auto job is registered.", job_id=job_id)
    status = "pass" if not missing and not forbidden else "fail"
    _check(report, "scheduler_jobs", status, "Scheduler job registration inspected.")


def _check_required_functions(report: dict) -> None:
    failed = False
    for module_name, function_name in REQUIRED_FUNCTIONS:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            failed = True
            _block(
                report,
                "missing_required_module",
                "Required dry-run module cannot be imported.",
                module=module_name,
                error=str(exc),
            )
            continue
        if not callable(getattr(module, function_name, None)):
            failed = True
            _block(
                report,
                "missing_required_function",
                "Required dry-run function is missing.",
                module=module_name,
                function=function_name,
            )
    _check(
        report,
        "required_functions",
        "fail" if failed else "pass",
        "Required dry-run modules and functions inspected.",
    )


def _check_state_store(report: dict, state_store: Any) -> bool:
    if state_store is None:
        _check(report, "state_store", "warning", "StateStore file is absent.")
        return False
    if isinstance(state_store, StateStoreCorruptError):
        _block(report, "corrupt_state_store", "StateStore is corrupt.", error=str(state_store))
        _check(report, "state_store", "fail", "StateStore could not be read.")
        return False
    try:
        state = getattr(state_store, "state")
        if not isinstance(state, dict):
            raise TypeError("StateStore state is not a dict")
    except StateStoreCorruptError as exc:
        _block(report, "corrupt_state_store", "StateStore is corrupt.", error=str(exc))
        _check(report, "state_store", "fail", "StateStore could not be read.")
        return False
    except (AttributeError, TypeError) as exc:
        _block(report, "unreadable_state_store", "StateStore could not be read.", error=str(exc))
        _check(report, "state_store", "fail", "StateStore could not be read.")
        return False
    _check(report, "state_store", "pass", "StateStore loaded for read-only inspection.")
    return True


def _check_ledgers(report: dict, root: Path, ledger_reader: Any) -> None:
    reader = ledger_reader if ledger_reader is not None else LedgerReader.from_root(root)
    ledger_dir = root / LEDGER_DIR_RELATIVE_PATH
    if ledger_dir.exists():
        if not ledger_dir.is_dir():
            _block(report, "ledger_path_not_directory", "Ledger path exists but is not a directory.", path=str(ledger_dir))
        else:
            _check(report, "ledger_directory", "pass", "Ledger directory exists.", path=str(ledger_dir))
    else:
        parent = ledger_dir.parent
        status = "pass" if parent.exists() else "warning"
        _check(report, "ledger_directory", status, "Ledger directory is absent; fresh ledger files are acceptable.", path=str(ledger_dir))
        _warn(report, "missing_ledger_directory", "Ledger directory is absent; fresh ledger files are acceptable.", path=str(ledger_dir))

    for name, path in (
        ("order", root / ORDER_LEDGER_RELATIVE_PATH),
        ("execution", root / EXECUTION_LEDGER_RELATIVE_PATH),
    ):
        if not path.exists():
            _warn(report, "missing_ledger_file", "Ledger file is absent; fresh dry-run roots may start empty.", ledger=name, path=str(path))
            continue
        try:
            reader.read_events(name)
        except LedgerReadError as exc:
            _block(report, "corrupt_ledger", "Ledger JSONL could not be read.", ledger=name, error=str(exc))
        else:
            _check(report, f"{name}_ledger", "pass", "Ledger file is readable.", path=str(path))


def _check_live_safety(report: dict, root: Path) -> None:
    failed = False
    for relative_path in DRY_RUN_ORCHESTRATION_FILES:
        path = root / relative_path
        if not path.exists():
            _warn(report, "source_file_missing", "Source file was not available for safety inspection.", path=str(relative_path))
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in FORBIDDEN_SOURCE_SNIPPETS:
            if snippet in text:
                failed = True
                _block(report, "forbidden_source_reference", "Forbidden live/scheduler source reference found.", path=str(relative_path), snippet=snippet)
    service_files = list(root.glob("*.service")) + list((root / "systemd").glob("*.service")) if (root / "systemd").exists() else list(root.glob("*.service"))
    for path in service_files:
        failed = True
        _block(report, "systemd_file_present", "Systemd service files are outside dry-run readiness scope.", path=str(path))
    _check(report, "live_safety_sources", "fail" if failed else "pass", "Dry-run orchestration source safety inspected.")


def _records_from_store(state_store: Any) -> tuple[list[dict], list[dict], list[dict]]:
    state = getattr(state_store, "state", {})
    return (
        _dict_records(state.get("order_intents", {})),
        _dict_records(state.get("positions", {})),
        _dict_records(state.get("close_intents", {})),
    )


def _dict_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [deepcopy(record) for record in value.values() if isinstance(record, dict)]
    if isinstance(value, list):
        return [deepcopy(record) for record in value if isinstance(record, dict)]
    return []


def _active_close_intent_count(close_intents: list[dict]) -> int:
    return sum(1 for intent in close_intents if intent.get("status") in ACTIVE_CLOSE_INTENT_STATUSES)


def _active_order_intent_count(order_intents: list[dict]) -> int:
    return sum(1 for intent in order_intents if intent.get("status") in PENDING_STATUSES)


def _check_pending_lifecycle(
    report: dict,
    order_intents: list[dict],
    close_intents: list[dict],
    checked_at: str,
) -> None:
    now = datetime.fromisoformat(checked_at)
    for intent in order_intents:
        status = intent.get("status")
        if status not in PENDING_STATUSES:
            continue
        code = f"pending_order_intent_{status}"
        _warn(report, code, f"Order intent is waiting for {status} lifecycle progression.", intent_id=intent.get("intent_id"), status=status)
        _warn_if_stale(report, "stale_order_intent", intent, now, "intent_id")
    for intent in close_intents:
        status = intent.get("status")
        if status not in PENDING_STATUSES:
            continue
        code = f"pending_close_intent_{status}"
        _warn(report, code, f"Close intent is waiting for {status} lifecycle progression.", close_intent_id=intent.get("close_intent_id"), position_id=intent.get("position_id"), status=status)
        _warn_if_stale(report, "stale_close_intent", intent, now, "close_intent_id")
    _check(report, "pending_lifecycle", "pass", "Pending lifecycle work inspected.")


def _warn_if_stale(report: dict, code: str, record: dict, now: datetime, id_field: str) -> None:
    timestamp = record.get("updated_at") or record.get("created_at")
    if not isinstance(timestamp, str):
        return
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        _warn(report, code, "Lifecycle record timestamp is not parseable.", record_id=record.get(id_field), timestamp=timestamp)
        return
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_minutes = (now - parsed).total_seconds() / 60
    if age_minutes > ORDER_INTENT_TTL_MINUTES:
        _warn(
            report,
            code,
            "Lifecycle record is older than ORDER_INTENT_TTL_MINUTES.",
            record_id=record.get(id_field),
            status=record.get("status"),
            age_minutes=round(age_minutes, 2),
            ttl_minutes=ORDER_INTENT_TTL_MINUTES,
        )


def _check_inconsistent_state(
    report: dict,
    order_intents: list[dict],
    positions: list[dict],
    close_intents: list[dict],
) -> None:
    close_by_id = {intent.get("close_intent_id"): intent for intent in close_intents}
    positions_by_id = {position.get("position_id"): position for position in positions}
    active_by_position: dict[str, list[dict]] = {}
    for intent in close_intents:
        if intent.get("status") in ACTIVE_CLOSE_INTENT_STATUSES:
            position_id = intent.get("position_id")
            if isinstance(position_id, str):
                active_by_position.setdefault(position_id, []).append(intent)
                if position_id not in positions_by_id:
                    _block(report, "close_intent_missing_position", "Active close intent references a missing position.", close_intent_id=intent.get("close_intent_id"), position_id=position_id)
                elif intent.get("status") == "filled" and positions_by_id[position_id].get("status") != "open":
                    _block(report, "filled_close_non_open_position", "Filled close intent references a non-open position.", close_intent_id=intent.get("close_intent_id"), position_id=position_id)
    for position_id, intents in active_by_position.items():
        if len(intents) > 1:
            _block(report, "duplicate_active_close_intents", "More than one active close intent exists for a position.", position_id=position_id, close_intent_ids=[item.get("close_intent_id") for item in intents])

    open_by_strategy_symbol: dict[tuple[str, str], list[str]] = {}
    for position in positions:
        position_id = position.get("position_id")
        active_id = position.get("active_close_intent_id")
        if active_id:
            active = close_by_id.get(active_id)
            if active is None or active.get("status") not in ACTIVE_CLOSE_INTENT_STATUSES:
                _block(report, "position_missing_active_close_intent", "Position has active_close_intent_id without a matching active close intent.", position_id=position_id, active_close_intent_id=active_id)
            elif active.get("position_id") != position_id:
                _block(report, "active_close_intent_mismatch", "Position active_close_intent_id points to a close intent for another position.", position_id=position_id, active_close_intent_id=active_id)
        if position.get("status") == "open":
            linked = close_by_id.get(active_id)
            if linked is not None and linked.get("status") == "position_closed":
                _block(report, "open_position_closed_close_intent", "Open position links to a position_closed close intent.", position_id=position_id, close_intent_id=active_id)
            key = (str(position.get("strategy_id")), str(position.get("symbol")))
            open_by_strategy_symbol.setdefault(key, []).append(str(position_id))
    for (strategy_id, symbol), position_ids in open_by_strategy_symbol.items():
        if len(position_ids) > 1:
            _block(report, "duplicate_open_positions", "Duplicate open positions exist for strategy_id and symbol.", strategy_id=strategy_id, symbol=symbol, position_ids=position_ids)

    for intent in close_intents:
        if intent.get("status") == "position_closed":
            position_id = intent.get("position_id")
            if active_by_position.get(str(position_id)):
                continue
            position = positions_by_id.get(position_id)
            if position is not None and position.get("active_close_intent_id") == intent.get("close_intent_id"):
                _block(report, "position_closed_intent_still_active", "position_closed close intent is still linked as active on its position.", close_intent_id=intent.get("close_intent_id"), position_id=position_id)
    for intent in order_intents:
        if intent.get("position_opened_event_id") and intent.get("status") != "position_opened":
            _block(report, "order_intent_opened_event_status_mismatch", "Order intent has position_opened_event_id but is not position_opened.", intent_id=intent.get("intent_id"), status=intent.get("status"))
    _check(report, "state_consistency", "pass", "State consistency inspected.")


def _finish_report(report: dict) -> None:
    if report["blocking_issues"]:
        report["ready"] = False
        report["status"] = "blocked"
        report["next_action"] = _blocked_next_action(report["blocking_issues"])
        return
    report["ready"] = True
    if report["warnings"]:
        report["status"] = "warning"
    else:
        report["status"] = "ready"
    report["next_action"] = "run_dry_run_chain"


def _blocked_next_action(issues: list[dict]) -> str:
    codes = {issue.get("code") for issue in issues}
    if "corrupt_state_store" in codes:
        return "inspect_state_file"
    if "missing_required_job" in codes:
        return "fix_scheduler_registration"
    if any("close_intent" in str(code) or "position" in str(code) for code in codes):
        return "manual_reconciliation_required"
    return "manual_reconciliation_required"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value
