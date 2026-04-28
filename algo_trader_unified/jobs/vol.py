"""Dry-run vol signal scan jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE
from algo_trader_unified.config.scheduler import JOB_S01_VOL_SCAN
from algo_trader_unified.config.variants import S01_CONFIG
from algo_trader_unified.core.readiness import ReadinessManager, ReadinessStatus
from algo_trader_unified.core.skip_reasons import SKIP_NEEDS_RECONCILIATION
from algo_trader_unified.strategies.base import Phase2ARiskManagerStub
from algo_trader_unified.strategies.vol.engine import VolSellingEngine
from algo_trader_unified.strategies.vol.signals import SignalResult, VolSignalInput


@dataclass(frozen=True)
class VolScanJobResult:
    job_id: str
    strategy_id: str
    status: str
    detail: str
    signal_result: SignalResult | None = None


SignalContextProvider = Callable[[], VolSignalInput]


def default_s01_signal_context_provider(
    current_time: datetime | None = None,
) -> VolSignalInput:
    """Returns a test-safe context with no live data; this intentionally yields SIGNAL_SKIPPED until a real provider is injected."""
    now = current_time or datetime.now(timezone.utc)
    return VolSignalInput(
        symbol="XSP",
        current_date=now.date(),
        vix=None,
        iv_rank=None,
        target_dte=int(S01_CONFIG.params["target_dte"]),
        blackout_dates=(),
        order_ref_candidate=None,
    )


def _readiness_allows_entries(readiness: ReadinessStatus | dict | None) -> bool:
    if isinstance(readiness, ReadinessStatus):
        return readiness.ready_for_entries
    if isinstance(readiness, dict):
        return bool(readiness.get("ready_for_entries"))
    return False


def _readiness_skip_reason(readiness: ReadinessStatus | dict | None) -> str:
    if isinstance(readiness, ReadinessStatus):
        return readiness.reason or SKIP_NEEDS_RECONCILIATION
    if isinstance(readiness, dict):
        return readiness.get("reason") or SKIP_NEEDS_RECONCILIATION
    return SKIP_NEEDS_RECONCILIATION


def run_s01_vol_scan(
    *,
    readiness_manager: ReadinessManager,
    state_store,
    ledger,
    broker=None,
    risk_manager=None,
    current_time: datetime | None = None,
    signal_context_provider: SignalContextProvider | None = None,
    engine: VolSellingEngine | None = None,
) -> VolScanJobResult:
    readiness = readiness_manager.get_readiness(S01_VOL_BASELINE)
    if not _readiness_allows_entries(readiness):
        skip_reason = _readiness_skip_reason(readiness)
        ledger.append(
            event_type="SIGNAL_SKIPPED",
            strategy_id=S01_VOL_BASELINE,
            execution_mode=S01_CONFIG.execution_mode,
            source_module="jobs.vol",
            payload={
                "strategy_id": S01_VOL_BASELINE,
                "skip_reason": skip_reason,
                "skip_detail": "S01 vol readiness gate blocked entries",
                "gate_name": "s01_vol_readiness_gate",
                "execution_mode": S01_CONFIG.execution_mode,
            },
        )
        return VolScanJobResult(
            job_id=JOB_S01_VOL_SCAN,
            strategy_id=S01_VOL_BASELINE,
            status="skipped",
            detail="readiness_skipped",
        )

    provider = signal_context_provider or (
        lambda: default_s01_signal_context_provider(current_time)
    )
    signal_input = provider()
    if not isinstance(signal_input, VolSignalInput):
        raise TypeError("signal_context_provider must return VolSignalInput")

    vol_engine = engine or VolSellingEngine(
        config=S01_CONFIG,
        state_store=state_store,
        ledger=ledger,
        broker=broker,
        risk_manager=risk_manager or Phase2ARiskManagerStub(),
    )
    signal_result = vol_engine.generate_standard_strangle_signal(signal_input)
    if signal_result.should_enter:
        return VolScanJobResult(
            job_id=JOB_S01_VOL_SCAN,
            strategy_id=S01_VOL_BASELINE,
            status="success",
            detail="signal_generated",
            signal_result=signal_result,
        )
    return VolScanJobResult(
        job_id=JOB_S01_VOL_SCAN,
        strategy_id=S01_VOL_BASELINE,
        status="skipped",
        detail="signal_skipped",
        signal_result=signal_result,
    )
