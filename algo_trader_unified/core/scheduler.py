"""Scheduler skeleton for Phase 3A readiness jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.scheduler import (
    JOB_INTENT_SUBMISSION,
    JOB_MARKET_OPEN_SCAN,
    JOB_S01_MANAGEMENT_SCAN,
    JOB_S01_VOL_SCAN,
    JOB_S02_MANAGEMENT_SCAN,
    JOB_S02_VOL_SCAN,
    JOB_SPECS,
    SCHEDULER_TIMEZONE,
    JobSpec,
)
from algo_trader_unified.core.readiness import ReadinessManager
from algo_trader_unified.jobs.readiness import (
    HealthSnapshot,
    all_clear_health_snapshot,
    market_open_scan,
)
from algo_trader_unified.jobs.management import run_management_scan_job
from algo_trader_unified.jobs.submission import run_intent_submission_job
from algo_trader_unified.jobs.vol import run_s01_vol_scan, run_s02_vol_scan


class SchedulerBuildError(RuntimeError):
    """Raised when the scheduler cannot be built or a job cannot be run."""


class MissingSchedulerDependencyError(SchedulerBuildError):
    """Raised when APScheduler is needed but unavailable."""


class JobNotFoundError(SchedulerBuildError):
    """Raised when a requested job_id is not registered."""


@dataclass(frozen=True)
class JobRunResult:
    job_id: str
    status: str
    started_at: str
    finished_at: str
    detail: str | None
    event_id: str | None


HealthSnapshotProvider = Callable[[], HealthSnapshot]


class UnifiedScheduler:
    def __init__(
        self,
        *,
        state_store,
        ledger,
        readiness_manager: ReadinessManager,
        job_specs: dict[str, JobSpec] | None = None,
        timezone: str = SCHEDULER_TIMEZONE,
        health_snapshot_provider: HealthSnapshotProvider | None = None,
    ) -> None:
        self.state_store = state_store
        self.ledger = ledger
        self.readiness_manager = readiness_manager
        self.job_specs = job_specs or JOB_SPECS
        self.timezone = timezone
        self.health_snapshot_provider = health_snapshot_provider
        self.scheduler = None

    def list_job_specs(self) -> tuple[JobSpec, ...]:
        return tuple(self.job_specs.values())

    def build_scheduler(self):
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError as exc:
            raise MissingSchedulerDependencyError(
                "APScheduler is required to build the runtime scheduler"
            ) from exc

        self.scheduler = BackgroundScheduler(timezone=self.timezone)
        self.add_jobs()
        return self.scheduler

    def add_jobs(self) -> None:
        if self.scheduler is None:
            raise SchedulerBuildError("build_scheduler() must be called before add_jobs()")
        for spec in self.job_specs.values():
            if not spec.enabled:
                continue
            self.scheduler.add_job(
                self.run_job_once,
                trigger=spec.trigger_type,
                kwargs={"job_id": spec.job_id},
                id=spec.job_id,
                max_instances=spec.max_instances,
                coalesce=spec.coalesce,
                misfire_grace_time=spec.misfire_grace_time,
                **spec.trigger_kwargs,
            )

    def run_job_once(self, job_id: str, **kwargs) -> JobRunResult | dict:
        if job_id not in self.job_specs:
            raise JobNotFoundError(f"Unknown scheduler job_id: {job_id}")
        started_at = datetime.now(timezone.utc).isoformat()
        event_id = None
        detail = None
        status = "success"

        if job_id == JOB_MARKET_OPEN_SCAN:
            strategy_ids = (S01_VOL_BASELINE, S02_VOL_ENHANCED)
            provider = kwargs.pop("health_snapshot_provider", self.health_snapshot_provider)
            snapshot = kwargs.pop(
                "health_snapshot",
                provider() if provider else all_clear_health_snapshot(strategy_ids),
            )
            result = market_open_scan(
                readiness_manager=kwargs.pop("readiness_manager", self.readiness_manager),
                current_time=kwargs.pop("current_time", datetime.now(timezone.utc)),
                strategy_ids=kwargs.pop("strategy_ids", strategy_ids),
                health_snapshot=snapshot,
                **kwargs,
            )
            status = "success" if result.all_clear else "skipped"
            detail = "all_clear" if result.all_clear else "readiness_failures"
        elif job_id == JOB_S01_VOL_SCAN:
            result = run_s01_vol_scan(
                readiness_manager=kwargs.pop("readiness_manager", self.readiness_manager),
                state_store=kwargs.pop("state_store", self.state_store),
                ledger=kwargs.pop("ledger", self.ledger),
                **kwargs,
            )
            status = result.status
            detail = result.detail
        elif job_id == JOB_S02_VOL_SCAN:
            result = run_s02_vol_scan(
                readiness_manager=kwargs.pop("readiness_manager", self.readiness_manager),
                state_store=kwargs.pop("state_store", self.state_store),
                ledger=kwargs.pop("ledger", self.ledger),
                **kwargs,
            )
            status = result.status
            detail = result.detail
        elif job_id == JOB_S01_MANAGEMENT_SCAN:
            kwargs.pop("strategy_id", None)
            return run_management_scan_job(
                strategy_id=S01_VOL_BASELINE,
                state_store=kwargs.pop("state_store", self.state_store),
                ledger=kwargs.pop("ledger", self.ledger),
                **kwargs,
            )
        elif job_id == JOB_S02_MANAGEMENT_SCAN:
            kwargs.pop("strategy_id", None)
            return run_management_scan_job(
                strategy_id=S02_VOL_ENHANCED,
                state_store=kwargs.pop("state_store", self.state_store),
                ledger=kwargs.pop("ledger", self.ledger),
                **kwargs,
            )
        elif job_id == JOB_INTENT_SUBMISSION:
            return run_intent_submission_job(
                state_store=kwargs.pop("state_store", self.state_store),
                ledger=kwargs.pop("ledger", self.ledger),
                **kwargs,
            )
        else:
            detail = "noop"

        return JobRunResult(
            job_id=job_id,
            status=status,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            detail=detail,
            event_id=event_id,
        )

    def start(self) -> None:
        if self.scheduler is None:
            self.build_scheduler()
        self.scheduler.start()

    def shutdown(self) -> None:
        if self.scheduler is not None:
            self.scheduler.shutdown()
