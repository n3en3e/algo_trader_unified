"""Read-only Stage 4 dry-run acceptance reporting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core import scheduler_cadence
from algo_trader_unified.core.halt_state_utils import halt_is_active


EXPECTED_4A_JOBS = scheduler_cadence.STAGE4A_JOB_IDS
EXPECTED_4B_LIFECYCLE_JOBS = scheduler_cadence.STAGE4B_LIFECYCLE_JOB_IDS
_ACTIVE_INTENT_STATUSES = {"created", "submitted", "confirmed", "filled"}
_SNAPSHOT_MAX_STALENESS_MINUTES = 15
_SNAPSHOT_TIMESTAMP_FIELDS = ("timestamp", "captured_at", "snapshot_at", "generated_at")


def build_dry_run_acceptance_report(
    *,
    state_store,
    ledger_reader,
    readiness_provider,
    snapshots_dir,
    halt_state_path,
    scheduler_builder,
    now_provider: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe, read-only acceptance report for Stage 4 dry-run mode."""

    now = _current_time(now_provider)
    errors: list[str] = []
    snapshots_path = Path(snapshots_dir)
    halt_path = Path(halt_state_path)
    halt_file_loadable, halt_state = _load_halt_state(halt_path, errors)
    scheduler = _scheduler_report(
        scheduler_builder=scheduler_builder,
        state_store=state_store,
        ledger_reader=ledger_reader,
        readiness_provider=readiness_provider,
        snapshots_dir=snapshots_path,
        halt_state_path=halt_path,
        now=now,
        errors=errors,
    )
    readiness = _readiness_by_strategy(state_store, errors)

    report = {
        "dry_run": True,
        "acceptance_report": True,
        "generated_at": now.isoformat(),
        "startup_gate": {
            "dry_run_only_required": True,
            "state_store_loadable": _state_store_loadable(state_store),
            "halt_file_loadable": halt_file_loadable,
            "halt_active": halt_is_active(halt_state),
            "diagnostics_passed": "unavailable",
        },
        "scheduler": scheduler,
        "expected_4a_jobs": list(EXPECTED_4A_JOBS),
        "expected_4b_lifecycle_jobs": list(EXPECTED_4B_LIFECYCLE_JOBS),
        "state": {
            "open_positions_count": _open_positions_count(state_store),
            "active_intents_count": _active_intents_count(state_store),
            "readiness_available_by_strategy": {
                strategy_id: readiness.get(strategy_id) is not None
                for strategy_id in _strategy_ids()
            },
            "dirty_state_by_strategy": {
                strategy_id: _readiness_value(readiness.get(strategy_id), "dirty_state")
                for strategy_id in _strategy_ids()
            },
        },
        "snapshots": _snapshots_report(snapshots_path, now, errors),
        "ledger": _ledger_report(ledger_reader, errors),
        "safety": {
            "broker_calls_enabled": False,
            "market_data_enabled": False,
            "systemd_enabled": False,
            "paper_live_orders_enabled": False,
        },
        "recommended_operator_commands": _recommended_operator_commands(),
        "errors": errors,
    }
    report["success"] = not errors
    return _json_safe(report)


def _scheduler_report(
    *,
    scheduler_builder,
    state_store,
    ledger_reader,
    readiness_provider,
    snapshots_dir: Path,
    halt_state_path: Path,
    now: datetime,
    errors: list[str],
) -> dict[str, Any]:
    without_triggers = _registered_jobs(
        scheduler_builder=scheduler_builder,
        enable_triggers=False,
        enable_lifecycle_pipeline=False,
        state_store=state_store,
        ledger=ledger_reader,
        readiness_provider=readiness_provider,
        snapshots_dir=snapshots_dir,
        halt_state_path=halt_state_path,
        now=now,
        errors=errors,
    )
    observation_jobs = _registered_jobs(
        scheduler_builder=scheduler_builder,
        enable_triggers=True,
        enable_lifecycle_pipeline=False,
        state_store=state_store,
        ledger=ledger_reader,
        readiness_provider=readiness_provider,
        snapshots_dir=snapshots_dir,
        halt_state_path=halt_state_path,
        now=now,
        errors=errors,
    )
    lifecycle_jobs = _registered_jobs(
        scheduler_builder=scheduler_builder,
        enable_triggers=True,
        enable_lifecycle_pipeline=True,
        state_store=state_store,
        ledger=ledger_reader,
        readiness_provider=readiness_provider,
        snapshots_dir=snapshots_dir,
        halt_state_path=halt_state_path,
        now=now,
        errors=errors,
    )
    if without_triggers:
        errors.append("scheduler registered jobs with enable_triggers=False")
    if observation_jobs != list(EXPECTED_4A_JOBS):
        errors.append("scheduler observation job registration does not match Stage 4A")
    if lifecycle_jobs != [*EXPECTED_4A_JOBS, *EXPECTED_4B_LIFECYCLE_JOBS]:
        errors.append("scheduler lifecycle job registration does not match Stage 4B")
    return {
        "observation_jobs_registered": observation_jobs,
        "lifecycle_jobs_registered": lifecycle_jobs,
        "lifecycle_pipeline_enabled_by_default": False,
        "jobs_registered_without_triggers": without_triggers,
    }


def _registered_jobs(
    *,
    scheduler_builder,
    enable_triggers: bool,
    enable_lifecycle_pipeline: bool,
    state_store,
    ledger,
    readiness_provider,
    snapshots_dir: Path,
    halt_state_path: Path,
    now: datetime,
    errors: list[str],
) -> list[str]:
    collector = _CollectingScheduler()
    try:
        scheduler = scheduler_builder(
            enable_triggers=enable_triggers,
            enable_lifecycle_pipeline=enable_lifecycle_pipeline,
            state_store=state_store,
            ledger=ledger,
            readiness_provider=readiness_provider,
            snapshots_dir=snapshots_dir,
            halt_state_path=halt_state_path,
            scheduler_factory=lambda: collector,
            now_provider=lambda: now,
        )
    except Exception as exc:
        errors.append(f"scheduler registration unavailable: {type(exc).__name__}: {exc}")
        return []
    jobs = getattr(scheduler, "jobs", getattr(collector, "jobs", []))
    if not isinstance(jobs, list):
        return []
    return [str(job["id"]) for job in jobs if isinstance(job, dict) and "id" in job]


class _CollectingScheduler:
    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []

    def add_job(self, func, **kwargs) -> None:
        self.jobs.append({"func": func, **kwargs})


def _load_halt_state(path: Path, errors: list[str]) -> tuple[bool, dict[str, Any] | None]:
    try:
        if not path.exists():
            return True, None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"halt_state unavailable: {exc}")
        return False, None
    if not isinstance(payload, dict):
        errors.append("halt_state is not a JSON object")
        return False, None
    return True, payload


def _state_store_loadable(state_store) -> bool:
    return isinstance(getattr(state_store, "state", None), dict)


def _open_positions_count(state_store) -> int | str:
    try:
        if hasattr(state_store, "list_positions"):
            return len(state_store.list_positions(status="open"))
        positions = getattr(state_store, "state", {}).get("positions", {})
        values = positions.values() if isinstance(positions, dict) else positions
        return sum(
            1
            for position in values
            if isinstance(position, dict) and position.get("status") == "open"
        )
    except Exception:
        return "unavailable"


def _active_intents_count(state_store) -> int | str:
    try:
        total = 0
        for method_name, key in (
            ("list_order_intents", "order_intents"),
            ("list_close_intents", "close_intents"),
        ):
            if hasattr(state_store, method_name):
                records = getattr(state_store, method_name)()
            else:
                records = getattr(state_store, "state", {}).get(key, {})
            values = records.values() if isinstance(records, dict) else records
            total += sum(
                1
                for record in values
                if isinstance(record, dict)
                and record.get("status") in _ACTIVE_INTENT_STATUSES
            )
        return total
    except Exception:
        return "unavailable"


def _readiness_by_strategy(state_store, errors: list[str]) -> dict[str, dict[str, Any] | None]:
    readiness: dict[str, dict[str, Any] | None] = {}
    for strategy_id in _strategy_ids():
        try:
            record = state_store.get_readiness(strategy_id)
        except Exception as exc:
            errors.append(f"readiness unavailable for {strategy_id}: {exc}")
            record = None
        readiness[strategy_id] = record if isinstance(record, dict) else None
    return readiness


def _readiness_value(record: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(record, dict):
        return None
    return record.get(key)


def _snapshots_report(snapshots_dir: Path, now: datetime, errors: list[str]) -> dict[str, Any]:
    latest_json = _latest_file(snapshots_dir, ".json")
    latest_digest = _latest_digest_file(snapshots_dir)
    fresh = None
    if latest_json is not None:
        fresh = _snapshot_fresh(latest_json, now)
    elif not snapshots_dir.exists():
        errors.append("snapshots directory unavailable")
    return {
        "latest_account_snapshot_path": str(latest_json) if latest_json is not None else None,
        "account_snapshot_fresh": fresh,
        "latest_digest_path": str(latest_digest) if latest_digest is not None else None,
    }


def _ledger_report(ledger_reader, errors: list[str]) -> dict[str, Any]:
    paths = _ledger_paths(ledger_reader)
    size = 0
    for path in paths:
        try:
            if path.exists():
                size += path.stat().st_size
        except OSError as exc:
            errors.append(f"ledger size unavailable for {path}: {exc}")
    return {
        "latest_event_timestamp": _latest_event_timestamp(paths, errors),
        "malformed_event_count": "unavailable",
        "total_event_count": "unavailable",
        "ledger_size_bytes": size if paths else "unavailable",
    }


def _ledger_paths(ledger_reader) -> list[Path]:
    paths = []
    for name in ("execution_ledger_path", "order_ledger_path"):
        value = getattr(ledger_reader, name, None)
        if value is not None:
            paths.append(Path(value))
    return paths


def _latest_event_timestamp(paths: list[Path], errors: list[str]) -> str | None:
    latest: datetime | None = None
    latest_raw: str | None = None
    for path in paths:
        line = _last_non_empty_line(path, errors)
        if line is None:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"latest ledger event malformed in {path}: {exc.msg}")
            continue
        timestamp = event.get("timestamp") if isinstance(event, dict) else None
        parsed = _parse_datetime(timestamp)
        if parsed is not None and (latest is None or parsed > latest):
            latest = parsed
            latest_raw = timestamp
    return latest_raw


def _last_non_empty_line(path: Path, errors: list[str]) -> str | None:
    try:
        if not path.exists():
            return None
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            if position == 0:
                return None
            buffer = b""
            while position > 0 and buffer.count(b"\n") < 2:
                read_size = min(1024, position)
                position -= read_size
                handle.seek(position)
                buffer = handle.read(read_size) + buffer
        for line in reversed(buffer.splitlines()):
            stripped = line.strip()
            if stripped:
                return stripped.decode("utf-8")
        return None
    except OSError as exc:
        errors.append(f"ledger unavailable for {path}: {exc}")
        return None
    except UnicodeDecodeError as exc:
        errors.append(f"ledger tail decode failed for {path}: {exc}")
        return None


def _latest_file(directory: Path, suffix: str) -> Path | None:
    try:
        candidates = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == suffix
        ]
    except OSError:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _latest_digest_file(directory: Path) -> Path | None:
    try:
        candidates = [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.name.startswith("digest_")
            and path.suffix.lower() == ".txt"
        ]
    except OSError:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _snapshot_fresh(path: Path, now: datetime) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    timestamp = _timestamp_from_payload(payload)
    if timestamp is None:
        try:
            timestamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        except OSError:
            return False
    return (now - timestamp).total_seconds() <= _SNAPSHOT_MAX_STALENESS_MINUTES * 60


def _timestamp_from_payload(payload: Any) -> datetime | None:
    if not isinstance(payload, dict):
        return None
    for field in _SNAPSHOT_TIMESTAMP_FIELDS:
        parsed = _parse_datetime(payload.get(field))
        if parsed is not None:
            return parsed
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _current_time(now_provider: Callable[[], datetime] | None) -> datetime:
    value = now_provider() if now_provider is not None else datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _strategy_ids() -> tuple[str, str]:
    return (S01_VOL_BASELINE, S02_VOL_ENHANCED)


def _recommended_operator_commands() -> dict[str, str]:
    return {
        "ayobot_pull_test": (
            "cd /home/algobot/algo_trader_unified && "
            "git pull && "
            "python3 -m py_compile $(find algo_trader_unified -name '*.py' "
            "-not -path '*/__pycache__/*') && "
            "python3 -m unittest discover -s algo_trader_unified/tests"
        ),
        "acceptance_report": (
            "python3 -m algo_trader_unified.tools.daemon "
            "--dry-run-only --acceptance-report"
        ),
        "bounded_smoke": (
            "python3 -m algo_trader_unified.tools.daemon "
            "--dry-run-only --smoke-cycles 1"
        ),
        "bounded_foreground": (
            "python3 -m algo_trader_unified.tools.daemon "
            "--dry-run-only --foreground-runtime-seconds 60 --enable-triggers"
        ),
        "daily_digest": (
            "python3 -m algo_trader_unified.tools.daemon "
            "--dry-run-only --smoke-cycles 1"
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
