"""Dry-run vol signal scan jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from algo_trader_unified.config.scheduler import JOB_S01_VOL_SCAN, JOB_S02_VOL_SCAN
from algo_trader_unified.config.variants import S01_CONFIG, S02_CONFIG, StrategyVariantConfig
from algo_trader_unified.core.ledger_reader import LedgerReader, LedgerReadError
from algo_trader_unified.core.readiness import ReadinessManager, ReadinessStatus
from algo_trader_unified.core.skip_reasons import (
    SKIP_ACTIVE_ORDER_INTENT,
    SKIP_ALREADY_SIGNALED_TODAY,
    SKIP_NEEDS_RECONCILIATION,
)
from algo_trader_unified.strategies.base import Phase2ARiskManagerStub
from algo_trader_unified.strategies.vol.engine import VolSellingEngine
from algo_trader_unified.strategies.vol.signals import (
    SignalResult,
    VolSignalInput,
    signal_generated_detail,
)


@dataclass(frozen=True)
class VolScanJobResult:
    job_id: str
    strategy_id: str
    status: str
    detail: str
    signal_result: SignalResult | None = None
    order_intent_id: str | None = None
    order_intent_created_event_id: str | None = None


SignalContextProvider = Callable[[], VolSignalInput]


_VOL_SCAN_JOB_IDS = {
    S01_CONFIG.strategy_id: JOB_S01_VOL_SCAN,
    S02_CONFIG.strategy_id: JOB_S02_VOL_SCAN,
}


def default_vol_signal_context_provider(
    config: StrategyVariantConfig,
    current_time: datetime | None = None,
) -> VolSignalInput:
    """Returns a test-safe context with no live data; this intentionally yields SIGNAL_SKIPPED until a real provider is injected."""
    now = current_time or datetime.now(timezone.utc)
    return VolSignalInput(
        symbol="XSP",
        current_date=now.date(),
        vix=None,
        iv_rank=None,
        target_dte=int(config.params["target_dte"]),
        blackout_dates=(),
        order_ref_candidate=None,
    )


def default_vol_signal_context_provider_for(
    config: StrategyVariantConfig,
    current_time: datetime | None = None,
) -> SignalContextProvider:
    def provider() -> VolSignalInput:
        return default_vol_signal_context_provider(config, current_time)

    return provider


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


def _job_id_for_config(config: StrategyVariantConfig) -> str:
    try:
        return _VOL_SCAN_JOB_IDS[config.strategy_id]
    except KeyError as exc:
        raise KeyError(f"No job_id registered for strategy_id={config.strategy_id}") from exc


def _idempotency_skip_detail(config: StrategyVariantConfig) -> str:
    return f"{config.strategy_id} signal already generated today"


def _readiness_skip_detail(config: StrategyVariantConfig) -> str:
    return f"{config.strategy_id} vol readiness gate blocked entries"


def _signal_payload(
    *,
    config: StrategyVariantConfig,
    signal_input: VolSignalInput,
    signal_result: SignalResult,
) -> dict:
    payload = {
        "symbol": signal_input.symbol,
        "target_dte": int(signal_input.target_dte),
        "vix": signal_input.vix,
        "iv_rank": signal_input.iv_rank,
        "order_ref_candidate": signal_input.order_ref_candidate,
        "sizing_context": signal_result.sizing_context,
        "risk_context": signal_result.risk_context,
    }
    if signal_result.should_enter:
        payload["event_detail"] = signal_generated_detail(config)
    else:
        payload["skip_reason"] = signal_result.skip_reason
        payload["skip_detail"] = signal_result.skip_detail
    return payload


def _active_order_intent_skip_payload(config: StrategyVariantConfig) -> dict:
    return {
        "strategy_id": config.strategy_id,
        "sleeve_id": config.sleeve_id,
        "skip_reason": SKIP_ACTIVE_ORDER_INTENT,
        "skip_detail": f"{config.strategy_id} already has an active order intent",
        "gate_name": "vol_order_intent_gate",
        "execution_mode": config.execution_mode,
    }


def _append_active_order_intent_skip(*, config: StrategyVariantConfig, ledger) -> None:
    ledger.append(
        event_type="SIGNAL_SKIPPED",
        strategy_id=config.strategy_id,
        execution_mode=config.execution_mode,
        source_module="jobs.vol",
        payload=_active_order_intent_skip_payload(config),
    )


def _intent_id_for_signal(
    *,
    config: StrategyVariantConfig,
    source_signal_event_id: str,
) -> str:
    return f"{config.strategy_id}:{source_signal_event_id}"


def _order_ref_for_signal(
    *,
    config: StrategyVariantConfig,
    signal_input: VolSignalInput,
    source_signal_event_id: str,
) -> str:
    if signal_input.order_ref_candidate and signal_input.order_ref_candidate.strip():
        return signal_input.order_ref_candidate.strip()
    return f"{config.strategy_id}|{source_signal_event_id}|OPEN"


def run_vol_scan(
    *,
    config: StrategyVariantConfig,
    readiness_manager: ReadinessManager,
    state_store,
    ledger,
    broker=None,
    risk_manager=None,
    current_time: datetime | None = None,
    signal_context_provider: SignalContextProvider | None = None,
    engine: VolSellingEngine | None = None,
    ledger_reader: LedgerReader | None = None,
) -> VolScanJobResult:
    now = current_time or datetime.now(timezone.utc)
    job_id = _job_id_for_config(config)
    with state_store.get_strategy_lock(config.strategy_id):
        if state_store.get_active_order_intent(config.strategy_id) is not None:
            _append_active_order_intent_skip(config=config, ledger=ledger)
            return VolScanJobResult(
                job_id=job_id,
                strategy_id=config.strategy_id,
                status="skipped",
                detail="active_order_intent",
            )

    if ledger_reader is None:
        if not hasattr(ledger, "root_dir"):
            raise LedgerReadError(
                "ledger_reader is required when ledger has no root_dir"
            )
        ledger_reader = LedgerReader.from_root(ledger.root_dir)
    reader = ledger_reader
    same_day_signals = reader.read_today(
        strategy_id=config.strategy_id,
        event_type="SIGNAL_GENERATED",
        now=now,
        timezone="America/New_York",
    )
    if same_day_signals:
        ledger.append(
            event_type="SIGNAL_SKIPPED",
            strategy_id=config.strategy_id,
            execution_mode=config.execution_mode,
            source_module="jobs.vol",
            payload={
                "strategy_id": config.strategy_id,
                "sleeve_id": config.sleeve_id,
                "skip_reason": SKIP_ALREADY_SIGNALED_TODAY,
                "skip_detail": _idempotency_skip_detail(config),
                "gate_name": "vol_idempotency_gate",
                "execution_mode": config.execution_mode,
                "matched_event_count": len(same_day_signals),
            },
        )
        return VolScanJobResult(
            job_id=job_id,
            strategy_id=config.strategy_id,
            status="skipped",
            detail="already_signaled_today",
            signal_result=None,
        )

    readiness = readiness_manager.get_readiness(config.strategy_id)
    if not _readiness_allows_entries(readiness):
        skip_reason = _readiness_skip_reason(readiness)
        ledger.append(
            event_type="SIGNAL_SKIPPED",
            strategy_id=config.strategy_id,
            execution_mode=config.execution_mode,
            source_module="jobs.vol",
            payload={
                "strategy_id": config.strategy_id,
                "sleeve_id": config.sleeve_id,
                "skip_reason": skip_reason,
                "skip_detail": _readiness_skip_detail(config),
                "gate_name": "vol_readiness_gate",
                "execution_mode": config.execution_mode,
            },
        )
        return VolScanJobResult(
            job_id=job_id,
            strategy_id=config.strategy_id,
            status="skipped",
            detail="readiness_skipped",
        )

    provider = signal_context_provider or default_vol_signal_context_provider_for(
        config,
        current_time,
    )
    signal_input = provider()
    if not isinstance(signal_input, VolSignalInput):
        raise TypeError("signal_context_provider must return VolSignalInput")

    vol_engine = engine or VolSellingEngine(
        config=config,
        state_store=state_store,
        ledger=ledger,
        broker=broker,
        risk_manager=risk_manager or Phase2ARiskManagerStub(),
    )
    signal_result = vol_engine.generate_standard_strangle_signal(
        signal_input,
        log_to_ledger=False,
    )
    signal_payload = _signal_payload(
        config=config,
        signal_input=signal_input,
        signal_result=signal_result,
    )
    # jobs.vol owns ledger attribution for scheduler-driven vol scans because
    # the engine evaluates the signal with log_to_ledger=False.
    if not signal_result.should_enter:
        ledger.append(
            event_type="SIGNAL_SKIPPED",
            strategy_id=config.strategy_id,
            execution_mode=config.execution_mode,
            source_module="jobs.vol",
            position_id=None,
            opportunity_id=None,
            payload=signal_payload,
        )
        return VolScanJobResult(
            job_id=job_id,
            strategy_id=config.strategy_id,
            status="skipped",
            detail="signal_skipped",
            signal_result=signal_result,
        )

    source_signal_event_id = str(
        ledger.append(
            event_type="SIGNAL_GENERATED",
            strategy_id=config.strategy_id,
            execution_mode=config.execution_mode,
            source_module="jobs.vol",
            position_id=None,
            opportunity_id=None,
            payload=signal_payload,
        )
    )

    with state_store.get_strategy_lock(config.strategy_id):
        if state_store.get_active_order_intent(config.strategy_id) is not None:
            _append_active_order_intent_skip(config=config, ledger=ledger)
            return VolScanJobResult(
                job_id=job_id,
                strategy_id=config.strategy_id,
                status="skipped",
                detail="active_order_intent",
                signal_result=signal_result,
            )
        created_at = now.isoformat()
        order_ref = _order_ref_for_signal(
            config=config,
            signal_input=signal_input,
            source_signal_event_id=source_signal_event_id,
        )
        intent_id = _intent_id_for_signal(
            config=config,
            source_signal_event_id=source_signal_event_id,
        )
        order_intent_payload = {
            "intent_id": intent_id,
            "strategy_id": config.strategy_id,
            "sleeve_id": config.sleeve_id,
            "symbol": signal_input.symbol,
            "execution_mode": config.execution_mode,
            "source_signal_event_id": source_signal_event_id,
            "order_ref": order_ref,
            "intent_status": "created",
            "created_at": created_at,
            "event_detail": "ORDER_INTENT_CREATED",
            "sizing_context": signal_result.sizing_context,
            "risk_context": signal_result.risk_context,
            "signal_payload_snapshot": signal_payload,
            "dry_run": True,
        }
        order_intent_created_event_id = str(
            ledger.append(
                event_type="ORDER_INTENT_CREATED",
                strategy_id=config.strategy_id,
                execution_mode=config.execution_mode,
                source_module="jobs.vol",
                position_id=None,
                opportunity_id=None,
                payload=order_intent_payload,
            )
        )
        intent_record = {
            "intent_id": intent_id,
            "strategy_id": config.strategy_id,
            "sleeve_id": config.sleeve_id,
            "symbol": signal_input.symbol,
            "execution_mode": config.execution_mode,
            "status": "created",
            "source_signal_event_id": source_signal_event_id,
            "order_intent_created_event_id": order_intent_created_event_id,
            "order_ref": order_ref,
            "created_at": created_at,
            "updated_at": created_at,
            "sizing_context": signal_result.sizing_context,
            "risk_context": signal_result.risk_context,
            "signal_payload_snapshot": signal_payload,
            "dry_run": True,
        }
        state_store.create_order_intent(intent_record)
        return VolScanJobResult(
            job_id=job_id,
            strategy_id=config.strategy_id,
            status="success",
            detail="order_intent_created",
            signal_result=signal_result,
            order_intent_id=intent_id,
            order_intent_created_event_id=order_intent_created_event_id,
        )


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
    ledger_reader: LedgerReader | None = None,
) -> VolScanJobResult:
    return run_vol_scan(
        config=S01_CONFIG,
        readiness_manager=readiness_manager,
        state_store=state_store,
        ledger=ledger,
        broker=broker,
        risk_manager=risk_manager,
        current_time=current_time,
        signal_context_provider=signal_context_provider,
        ledger_reader=ledger_reader,
        engine=engine,
    )


def run_s02_vol_scan(
    *,
    readiness_manager: ReadinessManager,
    state_store,
    ledger,
    broker=None,
    risk_manager=None,
    current_time: datetime | None = None,
    signal_context_provider: SignalContextProvider | None = None,
    engine: VolSellingEngine | None = None,
    ledger_reader: LedgerReader | None = None,
) -> VolScanJobResult:
    return run_vol_scan(
        config=S02_CONFIG,
        readiness_manager=readiness_manager,
        state_store=state_store,
        ledger=ledger,
        broker=broker,
        risk_manager=risk_manager,
        current_time=current_time,
        signal_context_provider=signal_context_provider,
        ledger_reader=ledger_reader,
        engine=engine,
    )
