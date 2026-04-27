"""Typed standard XSP strangle signal/gate helpers.

Phase 2A context schema:
  sizing_context = {"capital": float, "allocation_pct": float}
  risk_context = {"execution_mode": str, "strategy_id": str}

Phase 2B may extend these dictionaries, but Phase 2A never returns empty
contexts on either clean or skip paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from algo_trader_unified.config.portfolio import PAPER_RESEARCH_CAPITAL
from algo_trader_unified.config.variants import StrategyVariantConfig
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.strategies.base import RiskManagerProtocol


ENTRY_EXECUTION_MODES = {"shadow", "paper_only", "paper_proxy_for_live", "live_enabled"}

SKIP_HALTED = "SKIP_HALTED"
SKIP_EXISTING_POSITION = "SKIP_EXISTING_POSITION"
SKIP_BLACKOUT_DATE = "SKIP_BLACKOUT_DATE"
SKIP_VIX_GATE = "SKIP_VIX_GATE"
SKIP_IV_RANK_BELOW_MIN = "SKIP_IV_RANK_BELOW_MIN"
SKIP_NEEDS_RECONCILIATION = "SKIP_NEEDS_RECONCILIATION"
SKIP_ORDERREF_MISSING = "SKIP_ORDERREF_MISSING"

ACTION_HOLD = "hold"
ACTION_CLOSE_MANUAL = "close_manual"
ACTION_CLOSE_STOP_LOSS = "close_stop_loss"
ACTION_CLOSE_PROFIT_TARGET = "close_profit_target"
ACTION_CLOSE_DTE = "close_dte"

ALLOWED_MANAGEMENT_ACTIONS = {
    ACTION_HOLD,
    ACTION_CLOSE_MANUAL,
    ACTION_CLOSE_STOP_LOSS,
    ACTION_CLOSE_PROFIT_TARGET,
    ACTION_CLOSE_DTE,
}


@dataclass(frozen=True)
class VolSignalInput:
    symbol: str
    current_date: date
    vix: float | None
    iv_rank: float | None
    target_dte: int
    blackout_dates: tuple[date, ...]
    order_ref_candidate: str | None


@dataclass(frozen=True)
class SignalResult:
    should_enter: bool
    skip_reason: str | None
    skip_detail: str | None
    sizing_context: dict[str, float]
    risk_context: dict[str, str]


@dataclass(frozen=True)
class ManagementSignalInput:
    position_id: str
    current_date: date
    entry_date: date
    expiry: date
    entry_credit: float
    current_mark_to_close: float
    manual_close_requested: bool = False


@dataclass(frozen=True)
class ManagementResult:
    position_id: str
    action: str
    reason: str | None
    should_close: bool
    entry_credit: float
    current_mark_to_close: float
    days_to_expiry: int
    context: dict[str, object]


class ManagementInputError(ValueError):
    """Raised when injected management signal inputs are invalid."""


def build_sizing_context(config: StrategyVariantConfig) -> dict[str, float]:
    return {
        "capital": float(config.nominal_research_allocation),
        "allocation_pct": float(config.nominal_research_allocation) / float(PAPER_RESEARCH_CAPITAL),
    }


def build_risk_context(config: StrategyVariantConfig) -> dict[str, str]:
    return {
        "execution_mode": str(config.execution_mode),
        "strategy_id": config.strategy_id,
    }


def _result(
    *,
    config: StrategyVariantConfig,
    should_enter: bool,
    skip_reason: str | None,
    skip_detail: str | None,
) -> SignalResult:
    return SignalResult(
        should_enter=should_enter,
        skip_reason=skip_reason,
        skip_detail=skip_detail,
        sizing_context=build_sizing_context(config),
        risk_context=build_risk_context(config),
    )


def _normalized_dates(dates: Iterable[date]) -> set[date]:
    return set(dates)


def has_existing_open_position(
    state_store: StateStore,
    *,
    strategy_id: str,
    symbol: str,
) -> bool:
    open_statuses = {"pending_open", "open", "pending_close", "partial_fill_error"}
    with state_store.get_strategy_lock(strategy_id):
        for position in state_store.state.get("positions", []):
            if position.get("strategy_id") != strategy_id:
                continue
            if position.get("status") not in open_statuses:
                continue
            if position.get("contract_identity", {}).get("underlying") == symbol:
                return True
            for leg in position.get("legs", []):
                if leg.get("symbol") == symbol:
                    return True
    return False


def has_needs_reconciliation(
    state_store: StateStore,
    *,
    strategy_id: str,
    symbol: str,
) -> bool:
    with state_store.get_strategy_lock(strategy_id):
        for position in state_store.state.get("positions", []):
            if position.get("strategy_id") != strategy_id:
                continue
            if position.get("status") != "NEEDS_RECONCILIATION":
                continue
            if position.get("contract_identity", {}).get("underlying") == symbol:
                return True
            for leg in position.get("legs", []):
                if leg.get("symbol") == symbol:
                    return True
    return False


def evaluate_standard_strangle_signal(
    *,
    config: StrategyVariantConfig,
    state_store: StateStore,
    risk_manager: RiskManagerProtocol,
    signal_input: VolSignalInput,
) -> SignalResult:
    if config.execution_mode not in ENTRY_EXECUTION_MODES:
        return _result(
            config=config,
            should_enter=False,
            skip_reason=SKIP_HALTED,
            skip_detail=f"execution_mode {config.execution_mode} does not allow entries",
        )

    if risk_manager.is_halted(config.strategy_id) or not risk_manager.can_enter(config.strategy_id):
        return _result(
            config=config,
            should_enter=False,
            skip_reason=SKIP_HALTED,
            skip_detail=f"{config.strategy_id} halted or entries blocked by risk manager",
        )

    if has_existing_open_position(
        state_store,
        strategy_id=config.strategy_id,
        symbol=signal_input.symbol,
    ):
        return _result(
            config=config,
            should_enter=False,
            skip_reason=SKIP_EXISTING_POSITION,
            skip_detail=f"{config.strategy_id} already has open {signal_input.symbol} exposure",
        )

    if has_needs_reconciliation(
        state_store,
        strategy_id=config.strategy_id,
        symbol=signal_input.symbol,
    ):
        return _result(
            config=config,
            should_enter=False,
            skip_reason=SKIP_NEEDS_RECONCILIATION,
            skip_detail=f"{config.strategy_id} {signal_input.symbol} requires reconciliation",
        )

    if signal_input.current_date in _normalized_dates(signal_input.blackout_dates):
        return _result(
            config=config,
            should_enter=False,
            skip_reason=SKIP_BLACKOUT_DATE,
            skip_detail=f"{signal_input.symbol} entry blocked on blackout date {signal_input.current_date.isoformat()}",
        )

    vix_gate_min = config.params.get("vix_gate_min")
    if vix_gate_min not in (None, 0):
        if signal_input.vix is None or signal_input.vix < float(vix_gate_min):
            return _result(
                config=config,
                should_enter=False,
                skip_reason=SKIP_VIX_GATE,
                skip_detail=f"VIX {signal_input.vix} below minimum {vix_gate_min}",
            )

    iv_rank_min = float(config.params["iv_rank_min"])
    if signal_input.iv_rank is None or signal_input.iv_rank < iv_rank_min:
        return _result(
            config=config,
            should_enter=False,
            skip_reason=SKIP_IV_RANK_BELOW_MIN,
            skip_detail=f"IV rank {signal_input.iv_rank} below minimum {iv_rank_min:g}",
        )

    if signal_input.order_ref_candidate is None or not signal_input.order_ref_candidate.strip():
        return _result(
            config=config,
            should_enter=False,
            skip_reason=SKIP_ORDERREF_MISSING,
            skip_detail="orderRef candidate is missing",
        )

    return _result(
        config=config,
        should_enter=True,
        skip_reason=None,
        skip_detail=None,
    )


def evaluate_management_signal(
    *,
    config: StrategyVariantConfig,
    signal_input: ManagementSignalInput,
) -> ManagementResult:
    if signal_input.entry_credit <= 0:
        raise ManagementInputError("entry_credit must be > 0")
    if signal_input.current_mark_to_close < 0:
        raise ManagementInputError("current_mark_to_close must be >= 0")
    profit_target_pct = float(config.params["profit_target_pct"])
    stop_loss_mult = float(config.params["stop_loss_mult"])
    dte_close_threshold = int(config.params["dte_close_threshold"])
    days_to_expiry = (signal_input.expiry - signal_input.current_date).days
    profit_target_mark = signal_input.entry_credit * (1 - profit_target_pct)
    stop_loss_mark = signal_input.entry_credit * stop_loss_mult
    context = {
        "strategy_id": config.strategy_id,
        "execution_mode": config.execution_mode,
        "profit_target_pct": profit_target_pct,
        "stop_loss_mult": stop_loss_mult,
        "dte_close_threshold": dte_close_threshold,
        "profit_target_mark": profit_target_mark,
        "stop_loss_mark": stop_loss_mark,
        "entry_date": signal_input.entry_date.isoformat(),
    }
    if signal_input.manual_close_requested:
        action = ACTION_CLOSE_MANUAL
        reason = "manual close requested"
    elif signal_input.current_mark_to_close >= stop_loss_mark:
        action = ACTION_CLOSE_STOP_LOSS
        reason = "stop loss threshold reached"
    elif signal_input.current_mark_to_close <= profit_target_mark:
        action = ACTION_CLOSE_PROFIT_TARGET
        reason = "profit target threshold reached"
    elif days_to_expiry <= dte_close_threshold:
        action = ACTION_CLOSE_DTE
        reason = "DTE close threshold reached"
    else:
        action = ACTION_HOLD
        reason = None
    return ManagementResult(
        position_id=signal_input.position_id,
        action=action,
        reason=reason,
        should_close=action != ACTION_HOLD,
        entry_credit=signal_input.entry_credit,
        current_mark_to_close=signal_input.current_mark_to_close,
        days_to_expiry=days_to_expiry,
        context=context,
    )
