"""Scheduler job registry for the unified runner.

Scheduler config includes readiness infrastructure and dry-run vol scan wiring.
Job specs here must not imply live market data, order submission, contract
selection, or position lifecycle mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEDULER_TIMEZONE = "America/New_York"
DEFAULT_COALESCE = True
DEFAULT_MAX_INSTANCES = 1
DEFAULT_MISFIRE_GRACE_SEC = 60
SCHEDULER_SHUTDOWN_TIMEOUT_SEC = 10

JOB_KEEPALIVE = "keepalive"
JOB_RISK_MONITOR = "risk_monitor"
JOB_HEARTBEAT = "heartbeat"
JOB_MARKET_OPEN_SCAN = "market_open_scan"
JOB_EOD_REVIEW = "eod_review"
JOB_DAILY_DIGEST = "daily_digest"
JOB_WEEKLY_DIGEST = "weekly_digest"
JOB_S01_VOL_SCAN = "s01_vol_scan_stub"
JOB_S02_VOL_SCAN = "s02_vol_scan_stub"
JOB_S01_MANAGEMENT_SCAN = "s01_management_scan"
JOB_S02_MANAGEMENT_SCAN = "s02_management_scan"
JOB_INTENT_SUBMISSION = "intent_submission"
JOB_INTENT_CONFIRMATION = "intent_confirmation"
JOB_INTENT_FILL_CONFIRMATION = "intent_fill_confirmation"
JOB_POSITION_TRANSITIONS = "position_transitions"
JOB_DRY_RUN_SUBMIT_PENDING_INTENTS = "dry_run_submit_pending_intents"
JOB_DRY_RUN_EXPIRE_INTENTS = "dry_run_expire_intents"
JOB_DRY_RUN_EOD_INTENT_CLEANUP = "dry_run_eod_intent_cleanup"
JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS = "dry_run_confirm_submitted_orders"
JOB_DRY_RUN_CONFIRM_FILLS = "dry_run_confirm_fills"
JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS = "dry_run_apply_position_transitions"


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    description: str
    trigger_type: str
    trigger_kwargs: dict[str, Any] = field(default_factory=dict)
    max_instances: int = DEFAULT_MAX_INSTANCES
    coalesce: bool = DEFAULT_COALESCE
    misfire_grace_time: int = DEFAULT_MISFIRE_GRACE_SEC
    enabled: bool = True


JOB_SPECS: dict[str, JobSpec] = {
    JOB_KEEPALIVE: JobSpec(
        job_id=JOB_KEEPALIVE,
        description="Dry-run keepalive wrapper",
        trigger_type="interval",
        trigger_kwargs={"minutes": 15},
    ),
    JOB_RISK_MONITOR: JobSpec(
        job_id=JOB_RISK_MONITOR,
        description="Dry-run risk monitor wrapper",
        trigger_type="interval",
        trigger_kwargs={"minutes": 1},
    ),
    JOB_MARKET_OPEN_SCAN: JobSpec(
        job_id=JOB_MARKET_OPEN_SCAN,
        description="Market-open readiness sweep",
        trigger_type="cron",
        trigger_kwargs={"day_of_week": "mon-fri", "hour": 9, "minute": 35},
    ),
    JOB_DAILY_DIGEST: JobSpec(
        job_id=JOB_DAILY_DIGEST,
        description="Dry-run daily digest wrapper",
        trigger_type="cron",
        trigger_kwargs={
            "day_of_week": "mon-fri",
            "hour": 17,
            "minute": 0,
            "timezone": SCHEDULER_TIMEZONE,
        },
        coalesce=False,
    ),
    JOB_WEEKLY_DIGEST: JobSpec(
        job_id=JOB_WEEKLY_DIGEST,
        description="Dry-run weekly digest wrapper",
        trigger_type="cron",
        trigger_kwargs={"day_of_week": "fri", "hour": 16, "minute": 30},
    ),
    JOB_S01_VOL_SCAN: JobSpec(
        job_id=JOB_S01_VOL_SCAN,
        description="Dry-run S01 vol signal scan",
        trigger_type="cron",
        trigger_kwargs={"day_of_week": "mon-fri", "hour": 9, "minute": 40},
        enabled=True,
    ),
    JOB_S02_VOL_SCAN: JobSpec(
        job_id=JOB_S02_VOL_SCAN,
        description="Dry-run S02 vol signal scan",
        trigger_type="cron",
        trigger_kwargs={"day_of_week": "mon-fri", "hour": 9, "minute": 45},
        enabled=True,
    ),
    JOB_S01_MANAGEMENT_SCAN: JobSpec(
        job_id=JOB_S01_MANAGEMENT_SCAN,
        description="Dry-run S01 management scan",
        trigger_type="cron",
        trigger_kwargs={"day_of_week": "mon-fri", "hour": 15, "minute": 50},
        enabled=True,
    ),
    JOB_S02_MANAGEMENT_SCAN: JobSpec(
        job_id=JOB_S02_MANAGEMENT_SCAN,
        description="Dry-run S02 management scan",
        trigger_type="cron",
        trigger_kwargs={"day_of_week": "mon-fri", "hour": 15, "minute": 55},
        enabled=True,
    ),
    JOB_INTENT_SUBMISSION: JobSpec(
        job_id=JOB_INTENT_SUBMISSION,
        description="Dry-run created intent submission",
        trigger_type="cron",
        trigger_kwargs={"day_of_week": "mon-fri", "hour": 15, "minute": 59},
        enabled=True,
    ),
    JOB_INTENT_CONFIRMATION: JobSpec(
        job_id=JOB_INTENT_CONFIRMATION,
        description="Dry-run submitted intent confirmation",
        trigger_type="cron",
        trigger_kwargs={"day_of_week": "mon-fri", "hour": 16, "minute": 0},
        enabled=True,
    ),
    JOB_INTENT_FILL_CONFIRMATION: JobSpec(
        job_id=JOB_INTENT_FILL_CONFIRMATION,
        description="Dry-run confirmed intent fill confirmation",
        trigger_type="cron",
        trigger_kwargs={"day_of_week": "mon-fri", "hour": 16, "minute": 1},
        enabled=True,
    ),
    JOB_POSITION_TRANSITIONS: JobSpec(
        job_id=JOB_POSITION_TRANSITIONS,
        description="Dry-run filled intent position transitions",
        trigger_type="cron",
        trigger_kwargs={"day_of_week": "mon-fri", "hour": 16, "minute": 2},
        enabled=True,
    ),
    JOB_DRY_RUN_SUBMIT_PENDING_INTENTS: JobSpec(
        job_id=JOB_DRY_RUN_SUBMIT_PENDING_INTENTS,
        description="Disabled dry-run pending intent submission cadence",
        trigger_type="interval",
        trigger_kwargs={"seconds": 60},
        enabled=False,
    ),
    JOB_DRY_RUN_EXPIRE_INTENTS: JobSpec(
        job_id=JOB_DRY_RUN_EXPIRE_INTENTS,
        description="Disabled dry-run intent TTL expiry cadence",
        trigger_type="interval",
        trigger_kwargs={"seconds": 60},
        enabled=False,
    ),
    JOB_DRY_RUN_EOD_INTENT_CLEANUP: JobSpec(
        job_id=JOB_DRY_RUN_EOD_INTENT_CLEANUP,
        description="Disabled dry-run EOD intent cleanup cadence",
        trigger_type="cron",
        trigger_kwargs={
            "day_of_week": "mon-fri",
            "hour": 16,
            "minute": 10,
            "timezone": SCHEDULER_TIMEZONE,
        },
        enabled=False,
    ),
    JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS: JobSpec(
        job_id=JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS,
        description="Disabled dry-run submitted order confirmation cadence",
        trigger_type="interval",
        trigger_kwargs={"seconds": 60},
        enabled=False,
    ),
    JOB_DRY_RUN_CONFIRM_FILLS: JobSpec(
        job_id=JOB_DRY_RUN_CONFIRM_FILLS,
        description="Disabled dry-run fill confirmation cadence",
        trigger_type="interval",
        trigger_kwargs={"seconds": 60},
        enabled=False,
    ),
    JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS: JobSpec(
        job_id=JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS,
        description="Disabled dry-run position transition cadence",
        trigger_type="interval",
        trigger_kwargs={"seconds": 60},
        enabled=False,
    ),
}
