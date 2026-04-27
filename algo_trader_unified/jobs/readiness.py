"""Market-open readiness sweep job."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.variants import S01_CONFIG, S02_CONFIG
from algo_trader_unified.core.readiness import (
    MarketOpenReadinessResult,
    ReadinessManager,
    ReadinessStatus,
)
from algo_trader_unified.core.skip_reasons import (
    SKIP_ACCOUNT_SNAPSHOT_STALE,
    SKIP_BLACKOUT_DATE,
    SKIP_HALTED,
    SKIP_IV_BASELINE_MISSING,
    SKIP_NEEDS_RECONCILIATION,
    SKIP_NLV_DEGRADED,
    SKIP_STATESTORE_UNREADABLE,
    SKIP_UNKNOWN_BROKER_EXPOSURE,
)


@dataclass(frozen=True)
class HealthSnapshot:
    account_snapshot_fresh: bool
    nlv_valid: bool
    state_store_readable: bool
    halt_active_by_strategy: dict[str, bool]
    dirty_state_by_strategy: dict[str, bool]
    unknown_broker_exposure_by_strategy: dict[str, bool]
    calendar_expired_by_strategy: dict[str, bool]
    iv_baseline_available_by_strategy: dict[str, bool | None]


DEFAULT_EXECUTION_MODES = {
    S01_VOL_BASELINE: S01_CONFIG.execution_mode,
    S02_VOL_ENHANCED: S02_CONFIG.execution_mode,
}


def all_clear_health_snapshot(strategy_ids: Iterable[str]) -> HealthSnapshot:
    ids = tuple(strategy_ids)
    return HealthSnapshot(
        account_snapshot_fresh=True,
        nlv_valid=True,
        state_store_readable=True,
        halt_active_by_strategy={strategy_id: False for strategy_id in ids},
        dirty_state_by_strategy={strategy_id: False for strategy_id in ids},
        unknown_broker_exposure_by_strategy={strategy_id: False for strategy_id in ids},
        calendar_expired_by_strategy={strategy_id: False for strategy_id in ids},
        iv_baseline_available_by_strategy={strategy_id: True for strategy_id in ids},
    )


def _iso_now(current_time: datetime | None) -> str:
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    return current_time.isoformat()


def _failed_checks(snapshot: HealthSnapshot, strategy_id: str) -> list[str]:
    failed: list[str] = []
    if not snapshot.nlv_valid:
        failed.append("nlv_valid")
    if snapshot.dirty_state_by_strategy.get(strategy_id, False):
        failed.append("dirty_state")
    if snapshot.unknown_broker_exposure_by_strategy.get(strategy_id, False):
        failed.append("unknown_broker_exposure")
    if snapshot.halt_active_by_strategy.get(strategy_id, False):
        failed.append("halt_active")
    if snapshot.calendar_expired_by_strategy.get(strategy_id, False):
        failed.append("calendar_expired")
    if snapshot.iv_baseline_available_by_strategy.get(strategy_id) is False:
        failed.append("iv_baseline_available")
    if not snapshot.account_snapshot_fresh:
        failed.append("account_snapshot_fresh")
    if not snapshot.state_store_readable:
        failed.append("state_store_readable")
    return failed


def _skip_reason_for_failed_checks(failed_checks: list[str]) -> str | None:
    reason_by_check = {
        "nlv_valid": SKIP_NLV_DEGRADED,
        "dirty_state": SKIP_NEEDS_RECONCILIATION,
        "unknown_broker_exposure": SKIP_UNKNOWN_BROKER_EXPOSURE,
        "halt_active": SKIP_HALTED,
        "calendar_expired": SKIP_BLACKOUT_DATE,
        "iv_baseline_available": SKIP_IV_BASELINE_MISSING,
        "account_snapshot_fresh": SKIP_ACCOUNT_SNAPSHOT_STALE,
        "state_store_readable": SKIP_STATESTORE_UNREADABLE,
    }
    for check in reason_by_check:
        if check in failed_checks:
            return reason_by_check[check]
    return None


def market_open_scan(
    *,
    readiness_manager: ReadinessManager,
    current_time: datetime | None = None,
    strategy_ids: Iterable[str] = (S01_VOL_BASELINE, S02_VOL_ENHANCED),
    health_snapshot: HealthSnapshot | None = None,
    execution_mode_by_strategy: dict[str, str] | None = None,
) -> MarketOpenReadinessResult:
    strategy_ids = tuple(strategy_ids)
    checked_at = _iso_now(current_time)
    snapshot = health_snapshot or all_clear_health_snapshot(strategy_ids)
    execution_modes = execution_mode_by_strategy or DEFAULT_EXECUTION_MODES
    statuses: dict[str, ReadinessStatus] = {}

    for strategy_id in strategy_ids:
        failed_checks = _failed_checks(snapshot, strategy_id)
        skip_reason = _skip_reason_for_failed_checks(failed_checks)
        status = ReadinessStatus(
            strategy_id=strategy_id,
            ready_for_entries=skip_reason is None,
            reason=skip_reason,
            checked_at=checked_at,
            dirty_state=snapshot.dirty_state_by_strategy.get(strategy_id, False),
            unknown_broker_exposure=snapshot.unknown_broker_exposure_by_strategy.get(
                strategy_id, False
            ),
            nlv_degraded=not snapshot.nlv_valid,
            halt_active=snapshot.halt_active_by_strategy.get(strategy_id, False),
            calendar_expired=snapshot.calendar_expired_by_strategy.get(strategy_id, False),
            iv_baseline_available=snapshot.iv_baseline_available_by_strategy.get(strategy_id),
        )
        statuses[strategy_id] = status
        if skip_reason is not None:
            readiness_manager.ledger.append(
                event_type="SIGNAL_SKIPPED",
                strategy_id=strategy_id,
                execution_mode=execution_modes.get(strategy_id, "paper_only"),
                source_module="jobs.readiness",
                payload={
                    "strategy_id": strategy_id,
                    "skip_reason": skip_reason,
                    "skip_detail": f"market open readiness failed: {', '.join(failed_checks)}",
                    "failed_checks": failed_checks,
                    "gate_name": "market_open_readiness",
                    "execution_mode": execution_modes.get(strategy_id, "paper_only"),
                },
            )

    result = MarketOpenReadinessResult(
        checked_at=checked_at,
        statuses=statuses,
        all_clear=all(status.ready_for_entries for status in statuses.values()),
    )
    readiness_manager.store_market_open_result(result)
    return result
