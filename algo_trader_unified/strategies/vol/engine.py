"""Shared S01/S02 standard XSP strangle signal/gate engine.

The evaluate-and-reserve pattern is not atomic: signal generation and
create_pending_position are currently separate calls. Phase 2C must not
introduce any code path that assumes atomicity between signal evaluation and
position creation. The atomic fix -- evaluate-and-reserve under one held
strategy_state_lock, or re-check-and-reserve inside create_pending_position --
is deferred to Phase 3 scheduler/order-intent wiring. No Phase 2C code may
widen this gap.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE
from algo_trader_unified.config.variants import StrategyVariantConfig
from algo_trader_unified.core.broker import IBKRBrokerWrapper
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.strategies.base import BaseStrategy, RiskManagerProtocol
from algo_trader_unified.strategies.vol.signals import (
    ACTION_HOLD,
    ManagementResult,
    ManagementSignalInput,
    SignalResult,
    VolSignalInput,
    evaluate_management_signal,
    evaluate_standard_strangle_signal,
)


class InvalidSignalError(RuntimeError):
    """Raised when a lifecycle mutation is requested from a skipped signal."""


class LifecycleTransitionError(RuntimeError):
    """Raised when a position lifecycle transition is invalid."""


@dataclass
class VolSellingEngine(BaseStrategy):
    config: StrategyVariantConfig
    state_store: StateStore
    ledger: LedgerAppender
    broker: IBKRBrokerWrapper
    risk_manager: RiskManagerProtocol

    def generate_standard_strangle_signal(
        self,
        signal_input: VolSignalInput | None = None,
        *,
        symbol: str = "XSP",
        current_date: date | None = None,
        vix: float | None = None,
        iv_rank: float | None = None,
        target_dte: int | None = None,
        blackout_dates: Iterable[date] = (),
        order_ref_candidate: str | None = None,
    ) -> SignalResult:
        if signal_input is None:
            if current_date is None:
                raise ValueError("current_date is required when signal_input is not provided")
            selected_target_dte = (
                int(target_dte)
                if target_dte is not None
                else int(self.config.params["target_dte"])
            )
            signal_input = VolSignalInput(
                symbol=symbol,
                current_date=current_date,
                vix=vix,
                iv_rank=iv_rank,
                target_dte=selected_target_dte,
                blackout_dates=tuple(blackout_dates),
                order_ref_candidate=order_ref_candidate,
            )
        else:
            selected_target_dte = int(signal_input.target_dte)
        result = evaluate_standard_strangle_signal(
            config=self.config,
            state_store=self.state_store,
            risk_manager=self.risk_manager,
            signal_input=signal_input,
        )
        event_type = "SIGNAL_GENERATED" if result.should_enter else "SIGNAL_SKIPPED"
        payload = {
            "symbol": signal_input.symbol,
            "target_dte": selected_target_dte,
            "vix": signal_input.vix,
            "iv_rank": signal_input.iv_rank,
            "order_ref_candidate": signal_input.order_ref_candidate,
            "sizing_context": result.sizing_context,
            "risk_context": result.risk_context,
        }
        if result.should_enter and self.config.strategy_id == S01_VOL_BASELINE:
            payload["event_detail"] = "S01_VOL_SIGNAL_GENERATED"
        if not result.should_enter:
            payload["skip_reason"] = result.skip_reason
            payload["skip_detail"] = result.skip_detail
        self.ledger.append(
            event_type=event_type,
            strategy_id=self.config.strategy_id,
            execution_mode=self.config.execution_mode,
            source_module="strategies.vol.engine",
            position_id=None,
            opportunity_id=None,
            payload=payload,
        )
        return result

    def evaluate_management_signal(
        self,
        *,
        position_id: str,
        current_date: date,
        entry_date: date,
        expiry: date,
        entry_credit: float,
        current_mark_to_close: float,
        manual_close_requested: bool = False,
    ) -> ManagementResult:
        result = evaluate_management_signal(
            config=self.config,
            signal_input=ManagementSignalInput(
                position_id=position_id,
                current_date=current_date,
                entry_date=entry_date,
                expiry=expiry,
                entry_credit=entry_credit,
                current_mark_to_close=current_mark_to_close,
                manual_close_requested=manual_close_requested,
            ),
        )
        if result.action != ACTION_HOLD:
            self.ledger.append(
                event_type="SIGNAL_GENERATED",
                strategy_id=self.config.strategy_id,
                execution_mode=self.config.execution_mode,
                source_module="strategies.vol.engine",
                position_id=position_id,
                opportunity_id=None,
                payload={
                    "event_detail": "MANAGEMENT_CLOSE_SIGNAL",
                    "position_id": position_id,
                    "strategy_id": self.config.strategy_id,
                    "action": result.action,
                    "reason": result.reason,
                    "entry_credit": result.entry_credit,
                    "current_mark_to_close": result.current_mark_to_close,
                    "days_to_expiry": result.days_to_expiry,
                },
            )
        return result

    def create_pending_position(
        self,
        *,
        signal_result: SignalResult,
        position_id: str,
        leg_specs: list[dict[str, Any]],
        current_time: datetime | None = None,
    ) -> dict[str, Any]:
        if not signal_result.should_enter:
            raise InvalidSignalError("create_pending_position requires a clean signal")
        now = self._timestamp(current_time)
        position = {
            "position_id": position_id,
            "strategy_id": self.config.strategy_id,
            "sleeve_id": self.config.sleeve_id,
            "symbol": "XSP",
            "status": "pending_open",
            "execution_mode_at_entry": self.config.execution_mode,
            "managed_under_mode": self.config.execution_mode,
            "contract_identity": {"underlying": "XSP", "structure": "strangle"},
            "legs": deepcopy(leg_specs),
            "created_at": now,
            "updated_at": now,
        }
        with self.state_store.get_strategy_lock(self.config.strategy_id):
            self.state_store.state.setdefault("positions", []).append(position)
            self.state_store.save()
        self.ledger.append(
            event_type="POSITION_ADJUSTED",
            strategy_id=self.config.strategy_id,
            execution_mode=self.config.execution_mode,
            source_module="strategies.vol.engine",
            position_id=position_id,
            opportunity_id=None,
            payload={
                "lifecycle_event": "PENDING_OPEN_CREATED",
                "position_id": position_id,
                "strategy_id": self.config.strategy_id,
                "sleeve_id": self.config.sleeve_id,
                "symbol": "XSP",
                "status": "pending_open",
            },
        )
        return deepcopy(position)

    def mark_position_open(
        self,
        *,
        position_id: str,
        current_time: datetime | None = None,
    ) -> dict[str, Any]:
        now = self._timestamp(current_time)
        with self.state_store.get_strategy_lock(self.config.strategy_id):
            position = self._find_position(position_id)
            previous_status = position.get("status")
            if previous_status != "pending_open":
                raise LifecycleTransitionError(f"{position_id} is not pending_open")
            position["status"] = "open"
            position["updated_at"] = now
            self.state_store.save()
            snapshot = deepcopy(position)
        self.ledger.append(
            event_type="POSITION_OPENED",
            strategy_id=self.config.strategy_id,
            execution_mode=self.config.execution_mode,
            source_module="strategies.vol.engine",
            position_id=position_id,
            opportunity_id=None,
            payload={
                "position_id": position_id,
                "previous_status": previous_status,
                "status": "open",
            },
        )
        return snapshot

    def mark_position_pending_close(
        self,
        *,
        position_id: str,
        source_management_action: str | None = None,
        close_reason: str | None = None,
        close_intent_id: str | None = None,
        close_order_ref: str | None = None,
        current_time: datetime | None = None,
    ) -> dict[str, Any]:
        now = self._timestamp(current_time)
        with self.state_store.get_strategy_lock(self.config.strategy_id):
            position = self._find_position(position_id)
            previous_status = position.get("status")
            if previous_status != "open":
                raise LifecycleTransitionError(f"{position_id} is not open")
            position["status"] = "pending_close"
            position["updated_at"] = now
            if source_management_action is not None:
                position["source_management_action"] = source_management_action
            if close_reason is not None:
                position["close_reason"] = close_reason
            if close_intent_id is not None:
                position["close_intent_id"] = close_intent_id
            if close_order_ref is not None:
                position["close_order_ref"] = close_order_ref
            self.state_store.save()
            snapshot = deepcopy(position)
        self.ledger.append(
            event_type="POSITION_ADJUSTED",
            strategy_id=self.config.strategy_id,
            execution_mode=self.config.execution_mode,
            source_module="strategies.vol.engine",
            position_id=position_id,
            opportunity_id=None,
            payload={
                "lifecycle_event": "PENDING_CLOSE_CREATED",
                "position_id": position_id,
                "previous_status": "open",
                "status": "pending_close",
                "source_management_action": source_management_action,
                "close_reason": close_reason,
                "close_intent_id": close_intent_id,
                "close_order_ref": close_order_ref,
            },
        )
        return snapshot

    def record_close(
        self,
        *,
        position_id: str,
        close_reason: str,
        realized_pnl: float,
        current_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Finalize a position close and write POSITION_CLOSED.

        record_close accepts both 'open' and 'pending_close' as valid prior
        statuses. In the standard close flow, confirm_close_fill enforces
        pending_close before calling this method. Direct calls from outside
        VolOrderManager.confirm_close_fill bypass FILL_CONFIRMED. Phase 3
        scheduler integration should route all standard closes through
        VolOrderManager.confirm_close_fill.
        """
        now = self._timestamp(current_time)
        with self.state_store.get_strategy_lock(self.config.strategy_id):
            position = self._find_position(position_id)
            previous_status = position.get("status")
            if previous_status not in {"open", "pending_close"}:
                raise LifecycleTransitionError(f"{position_id} is not open or pending_close")
            position["status"] = "closed"
            position["closed_at"] = now
            position["realized_pnl"] = realized_pnl
            position["lifecycle_reason"] = close_reason
            position["updated_at"] = now
            self.state_store.save()
            snapshot = deepcopy(position)
        event = self.ledger.append(
            event_type="POSITION_CLOSED",
            strategy_id=self.config.strategy_id,
            execution_mode=self.config.execution_mode,
            source_module="strategies.vol.engine",
            position_id=position_id,
            opportunity_id=None,
            payload={
                "position_id": position_id,
                "previous_status": previous_status,
                "status": "closed",
                "close_reason": close_reason,
                "realized_pnl": realized_pnl,
            },
        )
        snapshot["position_closed_event_id"] = event.event_id
        return snapshot

    @staticmethod
    def _timestamp(current_time: datetime | None) -> str:
        return (current_time or datetime.now(timezone.utc)).isoformat()

    def _find_position(self, position_id: str) -> dict[str, Any]:
        for position in self.state_store.state.get("positions", []):
            if (
                position.get("position_id") == position_id
                and position.get("strategy_id") == self.config.strategy_id
            ):
                return position
        raise KeyError(f"Position not found for {self.config.strategy_id}: {position_id}")
