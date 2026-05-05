"""Stage 4A dry-run scheduler cadence wiring."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.scheduler import (
    JOB_DAILY_DIGEST,
    JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS,
    JOB_DRY_RUN_CONFIRM_FILLS,
    JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS,
    JOB_DRY_RUN_SUBMIT_PENDING_INTENTS,
    JOB_EOD_REVIEW,
    JOB_HEARTBEAT,
    JOB_KEEPALIVE,
    JOB_MARKET_OPEN_SCAN,
    JOB_RISK_MONITOR,
    JOB_S01_VOL_SCAN,
    JOB_S02_VOL_SCAN,
    SCHEDULER_TIMEZONE,
)
from algo_trader_unified.config.variants import S01_CONFIG, S02_CONFIG, StrategyVariantConfig
from algo_trader_unified.core.readiness import ReadinessManager, ReadinessStatus
from algo_trader_unified.core.skip_reasons import (
    SKIP_READINESS_NOT_EVALUATED,
    SKIP_STATESTORE_UNREADABLE,
)
from algo_trader_unified.jobs.confirmation import run_intent_confirmation_job
from algo_trader_unified.jobs.daily_digest import run_daily_digest
from algo_trader_unified.jobs.fill_confirmation import run_intent_fill_confirmation_job
from algo_trader_unified.jobs.position_transitions import run_position_transitions_job
from algo_trader_unified.jobs.readiness import market_open_scan
from algo_trader_unified.jobs.submission import run_intent_submission_job


STAGE4A_JOB_IDS = (
    JOB_KEEPALIVE,
    JOB_RISK_MONITOR,
    JOB_HEARTBEAT,
    JOB_MARKET_OPEN_SCAN,
    JOB_S01_VOL_SCAN,
    JOB_S02_VOL_SCAN,
    JOB_EOD_REVIEW,
    JOB_DAILY_DIGEST,
)

STAGE4B_LIFECYCLE_JOB_IDS = (
    JOB_DRY_RUN_SUBMIT_PENDING_INTENTS,
    JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS,
    JOB_DRY_RUN_CONFIRM_FILLS,
    JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS,
)

_ACTIVE_INTENT_STATUSES = {"created", "submitted", "confirmed", "filled"}
_LOCAL_SNAPSHOT_MAX_STALENESS_MINUTES = 15
_SNAPSHOT_TIMESTAMP_FIELDS = ("timestamp", "captured_at", "snapshot_at", "generated_at")


def build_scheduler(
    *,
    enable_triggers: bool,
    enable_lifecycle_pipeline: bool = False,
    state_store,
    ledger,
    readiness_provider,
    snapshots_dir,
    halt_state_path,
    scheduler_factory: Callable[[], Any] | None = None,
    now_provider: Callable[[], datetime] | None = None,
):
    scheduler = _new_scheduler(scheduler_factory)
    if not enable_triggers:
        return scheduler

    snapshots_path = Path(snapshots_dir)
    halt_path = Path(halt_state_path)
    readiness_manager = ReadinessManager(state_store, ledger)
    current_time = now_provider or _scheduler_now_datetime

    _add_interval_job(
        scheduler,
        JOB_KEEPALIVE,
        lambda: run_keepalive(),
        seconds=60,
    )
    _add_interval_job(
        scheduler,
        JOB_RISK_MONITOR,
        lambda: run_risk_monitor(
            state_store=state_store,
            snapshots_dir=snapshots_path,
            halt_state_path=halt_path,
        ),
        minutes=5,
    )
    _add_interval_job(
        scheduler,
        JOB_HEARTBEAT,
        lambda: run_heartbeat(
            state_store=state_store,
            snapshots_dir=snapshots_path,
            halt_state_path=halt_path,
        ),
        minutes=5,
    )
    _add_cron_job(
        scheduler,
        JOB_MARKET_OPEN_SCAN,
        lambda: run_market_open_scan_job(
            readiness_manager=readiness_manager,
            readiness_provider=readiness_provider,
            current_time=current_time(),
        ),
        hour=9,
        minute=35,
        coalesce=True,
    )
    _add_cron_job(
        scheduler,
        JOB_S01_VOL_SCAN,
        lambda: run_safe_vol_scan(
            config=S01_CONFIG,
            readiness_manager=readiness_manager,
            state_store=state_store,
            ledger=ledger,
        ),
        hour=9,
        minute=40,
        coalesce=True,
    )
    _add_cron_job(
        scheduler,
        JOB_S02_VOL_SCAN,
        lambda: run_safe_vol_scan(
            config=S02_CONFIG,
            readiness_manager=readiness_manager,
            state_store=state_store,
            ledger=ledger,
        ),
        hour=9,
        minute=45,
        coalesce=True,
    )
    _add_cron_job(
        scheduler,
        JOB_EOD_REVIEW,
        lambda: run_eod_review(),
        hour=16,
        minute=5,
        coalesce=True,
    )
    _add_cron_job(
        scheduler,
        JOB_DAILY_DIGEST,
        lambda: run_daily_digest_job(
            state_store=state_store,
            ledger=ledger,
            snapshots_dir=snapshots_path,
            halt_state_path=halt_path,
            now=current_time(),
        ),
        hour=17,
        minute=0,
        coalesce=False,
    )
    if enable_lifecycle_pipeline:
        _add_stage4b_lifecycle_jobs(
            scheduler=scheduler,
            state_store=state_store,
            ledger=ledger,
            now_provider=current_time,
        )
    return scheduler


def run_bounded_dry_run_smoke(
    *,
    state_store,
    ledger,
    readiness_provider,
    snapshots_dir,
    halt_state_path,
    cycles: int,
    include_lifecycle_pipeline: bool,
    now_provider: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    if cycles <= 0:
        raise ValueError("smoke cycles must be a positive finite integer")

    collector = _SmokeScheduler()
    build_scheduler(
        enable_triggers=True,
        enable_lifecycle_pipeline=include_lifecycle_pipeline,
        state_store=state_store,
        ledger=ledger,
        readiness_provider=readiness_provider,
        snapshots_dir=snapshots_dir,
        halt_state_path=halt_state_path,
        scheduler_factory=lambda: collector,
        now_provider=now_provider,
    )
    jobs_by_id = {job["id"]: job["func"] for job in collector.jobs}
    job_order = _smoke_job_order(include_lifecycle_pipeline)
    jobs_run = {job_id: 0 for job_id in job_order}
    job_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    completed = 0
    nap = sleep_fn or (lambda seconds: None)

    for cycle_index in range(cycles):
        try:
            for job_id in job_order:
                result = jobs_by_id[job_id]()
                jobs_run[job_id] += 1
                job_results.append(
                    {
                        "cycle": cycle_index + 1,
                        "job_id": job_id,
                        "dry_run": True,
                        "result": _json_safe(result),
                    }
                )
            completed += 1
        except Exception as exc:
            errors.append(
                {
                    "cycle": cycle_index + 1,
                    "job_id": job_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            break
        if cycle_index + 1 < cycles:
            nap(0)

    return {
        "dry_run": True,
        "cycles_requested": cycles,
        "cycles_completed": completed,
        "include_lifecycle_pipeline": include_lifecycle_pipeline,
        "jobs_run": jobs_run,
        "job_results": job_results,
        "errors": errors,
        "success": not errors and completed == cycles,
    }


def run_keepalive() -> dict[str, str]:
    return {"status": "alive"}


def run_risk_monitor(*, state_store, snapshots_dir: Path, halt_state_path: Path) -> dict[str, Any]:
    snapshot_fresh = _latest_snapshot_fresh(snapshots_dir)
    halt_state = _halt_state_summary(halt_state_path)
    open_positions = _open_positions_count(state_store)
    active_intents = _active_intents_count(state_store)
    print(
        "Risk monitor. "
        f"Snapshot fresh: {snapshot_fresh}. "
        f"Open positions: {open_positions}. "
        f"Active intents: {active_intents}. "
        f"Halt: {halt_state}"
    )
    return {
        "snapshot_fresh": snapshot_fresh,
        "open_positions": open_positions,
        "active_intents": active_intents,
        "halt": halt_state,
    }


def run_heartbeat(*, state_store, snapshots_dir: Path, halt_state_path: Path) -> dict[str, Any]:
    open_positions = _open_positions_count(state_store)
    active_intents = _active_intents_count(state_store)
    halt_state = _halt_state_summary(halt_state_path)
    print(
        f"Daemon alive. Open positions: {open_positions}, "
        f"Active intents: {active_intents}, Halt: {halt_state}"
    )
    payload = {
        "type": "heartbeat",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "open_positions": open_positions,
        "active_intents": active_intents,
        "halt": halt_state,
    }
    _write_snapshot(snapshots_dir, "heartbeat", payload)
    return payload


def run_market_open_scan_job(
    *,
    readiness_manager: ReadinessManager,
    readiness_provider,
    current_time: datetime | None = None,
):
    strategy_ids = (S01_VOL_BASELINE, S02_VOL_ENHANCED)
    return market_open_scan(
        readiness_manager=readiness_manager,
        current_time=current_time,
        strategy_ids=strategy_ids,
        health_snapshot=readiness_provider(),
    )


def run_safe_vol_scan(
    *,
    config: StrategyVariantConfig,
    readiness_manager: ReadinessManager,
    state_store,
    ledger,
):
    try:
        readiness = readiness_manager.get_readiness(config.strategy_id)
    except Exception:
        _append_vol_readiness_skip(
            config=config,
            ledger=ledger,
            skip_reason=SKIP_STATESTORE_UNREADABLE,
        )
        return {
            "job_id": _job_id_for_config(config),
            "strategy_id": config.strategy_id,
            "status": "skipped",
            "detail": "statestore_unreadable",
        }
    if readiness is None:
        _append_vol_readiness_skip(
            config=config,
            ledger=ledger,
            skip_reason=SKIP_READINESS_NOT_EVALUATED,
        )
        return {
            "job_id": _job_id_for_config(config),
            "strategy_id": config.strategy_id,
            "status": "skipped",
            "detail": "readiness_not_evaluated",
        }
    if not _readiness_allows_entries(readiness):
        _append_vol_readiness_skip(
            config=config,
            ledger=ledger,
            skip_reason=_readiness_skip_reason(readiness),
        )
        return {
            "job_id": _job_id_for_config(config),
            "strategy_id": config.strategy_id,
            "status": "skipped",
            "detail": "readiness_skipped",
        }
    print(f"{config.strategy_id} vol scan readiness clear; live signal provider not wired")
    return {
        "job_id": _job_id_for_config(config),
        "strategy_id": config.strategy_id,
        "status": "skipped",
        "detail": "signal_provider_not_wired",
    }


def run_eod_review() -> dict[str, str]:
    print("EOD review placeholder")
    return {"status": "placeholder"}


def run_daily_digest_job(
    *,
    state_store,
    ledger,
    snapshots_dir: Path,
    halt_state_path: Path,
    telegram_sender=None,
    now: datetime | None = None,
):
    return run_daily_digest(
        state_store=state_store,
        ledger=ledger,
        snapshots_dir=snapshots_dir,
        halt_state_path=halt_state_path,
        telegram_sender=telegram_sender,
        now=now,
        strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
    )


def _add_stage4b_lifecycle_jobs(*, scheduler, state_store, ledger, now_provider) -> None:
    _add_interval_job(
        scheduler,
        JOB_DRY_RUN_SUBMIT_PENDING_INTENTS,
        lambda: run_intent_submission_job(
            state_store=state_store,
            ledger=ledger,
            now=_scheduler_now(now_provider),
        ),
        seconds=60,
    )
    _add_interval_job(
        scheduler,
        JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS,
        lambda: run_intent_confirmation_job(
            state_store=state_store,
            ledger=ledger,
            now=_scheduler_now(now_provider),
        ),
        seconds=60,
    )
    _add_interval_job(
        scheduler,
        JOB_DRY_RUN_CONFIRM_FILLS,
        lambda: run_intent_fill_confirmation_job(
            state_store=state_store,
            ledger=ledger,
            now=_scheduler_now(now_provider),
        ),
        seconds=60,
    )
    _add_interval_job(
        scheduler,
        JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS,
        lambda: run_position_transitions_job(
            state_store=state_store,
            ledger=ledger,
            now=_scheduler_now(now_provider),
        ),
        seconds=60,
    )


def _scheduler_now(now_provider: Callable[[], datetime]) -> str:
    return now_provider().astimezone(timezone.utc).isoformat()


def _scheduler_now_datetime() -> datetime:
    return datetime.now(timezone.utc)


def _smoke_job_order(include_lifecycle_pipeline: bool) -> tuple[str, ...]:
    lifecycle_jobs = STAGE4B_LIFECYCLE_JOB_IDS if include_lifecycle_pipeline else ()
    return (
        JOB_KEEPALIVE,
        JOB_RISK_MONITOR,
        JOB_HEARTBEAT,
        JOB_MARKET_OPEN_SCAN,
        JOB_S01_VOL_SCAN,
        JOB_S02_VOL_SCAN,
        *lifecycle_jobs,
        JOB_EOD_REVIEW,
        JOB_DAILY_DIGEST,
    )


class _SmokeScheduler:
    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []

    def add_job(self, func, **kwargs) -> None:
        self.jobs.append({"func": func, **kwargs})


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _new_scheduler(scheduler_factory: Callable[[], Any] | None):
    if scheduler_factory is not None:
        return scheduler_factory()
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError as exc:
        from algo_trader_unified.core.scheduler import MissingSchedulerDependencyError

        raise MissingSchedulerDependencyError(
            "APScheduler is required to build the runtime scheduler"
        ) from exc
    return BackgroundScheduler(timezone=SCHEDULER_TIMEZONE)


def _add_interval_job(scheduler, job_id: str, func, **interval_kwargs: int) -> None:
    scheduler.add_job(
        func,
        trigger="interval",
        id=job_id,
        max_instances=1,
        coalesce=True,
        **interval_kwargs,
    )


def _add_cron_job(
    scheduler,
    job_id: str,
    func,
    *,
    hour: int,
    minute: int,
    coalesce: bool,
) -> None:
    scheduler.add_job(
        func,
        trigger="cron",
        id=job_id,
        day_of_week="mon-fri",
        hour=hour,
        minute=minute,
        timezone=SCHEDULER_TIMEZONE,
        max_instances=1,
        coalesce=coalesce,
    )


def _append_vol_readiness_skip(
    *,
    config: StrategyVariantConfig,
    ledger,
    skip_reason: str,
) -> None:
    ledger.append(
        event_type="SIGNAL_SKIPPED",
        strategy_id=config.strategy_id,
        execution_mode=config.execution_mode,
        source_module="core.scheduler_cadence",
        payload={
            "strategy_id": config.strategy_id,
            "sleeve_id": config.sleeve_id,
            "skip_reason": skip_reason,
            "skip_detail": f"{config.strategy_id} vol readiness gate blocked entries",
            "gate_name": "vol_readiness_gate",
            "execution_mode": config.execution_mode,
        },
    )


def _readiness_allows_entries(readiness: ReadinessStatus | dict) -> bool:
    if isinstance(readiness, ReadinessStatus):
        return readiness.ready_for_entries
    if isinstance(readiness, dict):
        return bool(readiness.get("ready_for_entries"))
    return False


def _readiness_skip_reason(readiness: ReadinessStatus | dict) -> str:
    if isinstance(readiness, ReadinessStatus):
        return readiness.reason or SKIP_READINESS_NOT_EVALUATED
    if isinstance(readiness, dict):
        return readiness.get("reason") or SKIP_READINESS_NOT_EVALUATED
    return SKIP_READINESS_NOT_EVALUATED


def _job_id_for_config(config: StrategyVariantConfig) -> str:
    if config.strategy_id == S01_VOL_BASELINE:
        return JOB_S01_VOL_SCAN
    return JOB_S02_VOL_SCAN


def _open_positions_count(state_store) -> int:
    if hasattr(state_store, "list_positions"):
        return len(state_store.list_positions(status="open"))
    positions = getattr(state_store, "state", {}).get("positions", {})
    values = positions.values() if isinstance(positions, dict) else positions
    return sum(1 for position in values if isinstance(position, dict) and position.get("status") == "open")


def _active_intents_count(state_store) -> int:
    total = 0
    if hasattr(state_store, "list_order_intents"):
        total += _count_active(state_store.list_order_intents())
    else:
        total += _count_active(getattr(state_store, "state", {}).get("order_intents", {}))
    if hasattr(state_store, "list_close_intents"):
        total += _count_active(state_store.list_close_intents())
    else:
        total += _count_active(getattr(state_store, "state", {}).get("close_intents", {}))
    return total


def _count_active(records: Any) -> int:
    values = records.values() if isinstance(records, dict) else records
    return sum(
        1
        for record in values
        if isinstance(record, dict) and record.get("status") in _ACTIVE_INTENT_STATUSES
    )


def _halt_state_summary(halt_state_path: Path) -> str:
    try:
        if not halt_state_path.exists():
            return "inactive"
        payload = json.loads(halt_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unreadable"
    if not isinstance(payload, dict):
        return "unreadable"
    if payload.get("resumed") is True or payload.get("tier") not in {"soft", "hard"}:
        return "inactive"
    scope = payload.get("scope") or "unknown"
    tier = payload.get("tier")
    halted_id = payload.get("id") or payload.get("scope_id")
    if halted_id:
        return f"active:{scope}:{halted_id}:{tier}"
    return f"active:{scope}:{tier}"


def _latest_snapshot_fresh(snapshots_dir: Path) -> bool:
    try:
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_file() and path.suffix == ".json"]
    except OSError:
        return False
    if not snapshots:
        return False
    latest = max(snapshots, key=lambda path: path.stat().st_mtime)
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    timestamp = _snapshot_timestamp(payload)
    if timestamp is None:
        timestamp = datetime.fromtimestamp(latest.stat().st_mtime, timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - timestamp).total_seconds()
    return age_seconds <= _LOCAL_SNAPSHOT_MAX_STALENESS_MINUTES * 60


def _snapshot_timestamp(payload: dict[str, Any]) -> datetime | None:
    for field in _SNAPSHOT_TIMESTAMP_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _write_snapshot(snapshots_dir: Path, prefix: str, payload: dict[str, Any]) -> Path:
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = snapshots_dir / f"{prefix}_{stamp}.json"
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    return path
