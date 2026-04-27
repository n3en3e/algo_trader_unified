"""Dry-run close order-intent coordinator for standard XSP strangles."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Number
from typing import Any

from algo_trader_unified.config.variants import StrategyVariantConfig
from algo_trader_unified.core.broker import IBKRBrokerWrapper, MissingOrderRefError
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.strategies.vol.engine import VolSellingEngine
from algo_trader_unified.strategies.vol.signals import ACTION_HOLD, ManagementResult


class InvalidManagementResultError(RuntimeError):
    """Raised when a management result is not a close signal."""


class StaleManagementResultError(RuntimeError):
    """Raised when current StateStore status no longer supports close execution."""


class CloseFillValidationError(RuntimeError):
    """Raised when injected close-fill data or persisted close metadata is invalid."""


class StaleCloseFillError(RuntimeError):
    """Raised when current StateStore status no longer supports fill confirmation."""


@dataclass(frozen=True)
class CloseOrderIntent:
    intent_id: str
    position_id: str
    strategy_id: str
    sleeve_id: str
    symbol: str
    action: str
    reason: str
    order_ref: str
    status: str
    created_at: str
    source_management_action: str
    dry_run: bool
    payload: dict[str, Any]


@dataclass(frozen=True)
class CloseExecutionResult:
    intent_id: str
    position_id: str
    strategy_id: str
    status: str
    order_submitted_event_id: str | None
    pending_close_created: bool
    order_ref: str
    source_management_action: str


@dataclass(frozen=True)
class CloseConfirmResult:
    position_id: str
    strategy_id: str
    order_ref: str
    realized_pnl: float
    fill_price: float
    fill_time: Any
    fill_confirmed_event_id: str
    position_closed_event_id: str
    close_reason: str
    status: str


class VolOrderManager:
    def __init__(
        self,
        *,
        config: StrategyVariantConfig,
        state_store: StateStore,
        ledger: LedgerAppender,
        broker: IBKRBrokerWrapper,
        engine: VolSellingEngine,
        client_id: int = 20,
    ) -> None:
        self.config = config
        self.state_store = state_store
        self.ledger = ledger
        self.broker = broker
        self.engine = engine
        self.client_id = client_id

    def execute_close(
        self,
        *,
        management_result: ManagementResult,
        order_ref: str | None,
        intent_id: str,
        current_time: datetime | None = None,
    ) -> CloseExecutionResult:
        if not management_result.should_close or management_result.action == ACTION_HOLD:
            raise InvalidManagementResultError("execute_close requires a close management result")
        self.broker._assert_submission_allowed(self.client_id, order_ref)
        assert order_ref is not None
        if not order_ref.strip():
            raise MissingOrderRefError("close order intent requires order_ref")

        intent = self._build_close_intent(
            management_result=management_result,
            order_ref=order_ref,
            intent_id=intent_id,
            current_time=current_time,
        )
        with self.state_store.get_strategy_lock(self.config.strategy_id):
            position = self._find_position(management_result.position_id)
            if position.get("status") != "open":
                raise StaleManagementResultError(
                    f"{management_result.position_id} is not open"
                )
            self.engine.mark_position_pending_close(
                position_id=management_result.position_id,
                source_management_action=intent.source_management_action,
                close_reason=intent.reason,
                close_intent_id=intent.intent_id,
                close_order_ref=intent.order_ref,
                current_time=current_time,
            )

        event = self.ledger.append(
            event_type="ORDER_SUBMITTED",
            strategy_id=self.config.strategy_id,
            execution_mode=self.config.execution_mode,
            source_module="strategies.vol.order_manager",
            position_id=management_result.position_id,
            opportunity_id=None,
            payload=deepcopy(intent.payload),
        )
        return CloseExecutionResult(
            intent_id=intent.intent_id,
            position_id=intent.position_id,
            strategy_id=intent.strategy_id,
            status=intent.status,
            order_submitted_event_id=event.event_id,
            pending_close_created=True,
            order_ref=order_ref,
            source_management_action=management_result.action,
        )

    def confirm_close_fill(
        self,
        *,
        position_id: str,
        order_ref: str,
        realized_pnl: float,
        fill_price: float,
        fill_time: Any,
        fill_id: str | None = None,
        current_time: datetime | None = None,
    ) -> CloseConfirmResult:
        self._validate_close_fill_inputs(
            order_ref=order_ref,
            realized_pnl=realized_pnl,
            fill_price=fill_price,
            fill_time=fill_time,
        )
        with self.state_store.get_strategy_lock(self.config.strategy_id):
            position = self._find_pending_close_position(position_id)
            if position.get("close_order_ref") != order_ref:
                raise CloseFillValidationError("close fill order_ref does not match pending close")
            source_management_action, close_reason = self._close_metadata(position)
            payload = {
                "position_id": position_id,
                "strategy_id": self.config.strategy_id,
                "sleeve_id": position.get("sleeve_id"),
                "symbol": position.get("symbol"),
                "order_ref": order_ref,
                "fill_id": fill_id,
                "fill_price": fill_price,
                "fill_time": fill_time,
                "realized_pnl": realized_pnl,
                "source_management_action": source_management_action,
                "close_reason": close_reason,
            }
            fill_event = self.ledger.append(
                event_type="FILL_CONFIRMED",
                strategy_id=self.config.strategy_id,
                execution_mode=self.config.execution_mode,
                source_module="strategies.vol.order_manager",
                position_id=position_id,
                opportunity_id=None,
                payload=deepcopy(payload),
            )
            closed = self.engine.record_close(
                position_id=position_id,
                close_reason=close_reason,
                realized_pnl=realized_pnl,
                current_time=current_time,
            )
        position_closed_event_id = closed.get("position_closed_event_id")
        if not position_closed_event_id:
            raise CloseFillValidationError("position close event id was not returned")
        return CloseConfirmResult(
            position_id=position_id,
            strategy_id=self.config.strategy_id,
            order_ref=order_ref,
            realized_pnl=realized_pnl,
            fill_price=fill_price,
            fill_time=fill_time,
            fill_confirmed_event_id=fill_event.event_id,
            position_closed_event_id=position_closed_event_id,
            close_reason=close_reason,
            status="close_confirmed",
        )

    def _build_close_intent(
        self,
        *,
        management_result: ManagementResult,
        order_ref: str,
        intent_id: str,
        current_time: datetime | None,
    ) -> CloseOrderIntent:
        created_at = (current_time or datetime.now(timezone.utc)).isoformat()
        reason = management_result.reason or management_result.action
        payload = {
            "intent_id": intent_id,
            "position_id": management_result.position_id,
            "strategy_id": self.config.strategy_id,
            "sleeve_id": self.config.sleeve_id,
            "symbol": "XSP",
            "action": "close",
            "reason": reason,
            "order_ref": order_ref,
            "dry_run": True,
            "source_management_action": management_result.action,
            "current_mark_to_close": management_result.current_mark_to_close,
            "entry_credit": management_result.entry_credit,
            "days_to_expiry": management_result.days_to_expiry,
        }
        return CloseOrderIntent(
            intent_id=intent_id,
            position_id=management_result.position_id,
            strategy_id=self.config.strategy_id,
            sleeve_id=self.config.sleeve_id,
            symbol="XSP",
            action="close",
            reason=reason,
            order_ref=order_ref,
            status="close_intent_created",
            created_at=created_at,
            source_management_action=management_result.action,
            dry_run=True,
            payload=payload,
        )

    def _find_position(self, position_id: str) -> dict[str, Any]:
        for position in self.state_store.state.get("positions", []):
            if (
                position.get("position_id") == position_id
                and position.get("strategy_id") == self.config.strategy_id
            ):
                return position
        raise StaleManagementResultError(f"{position_id} not found")

    def _find_pending_close_position(self, position_id: str) -> dict[str, Any]:
        for position in self.state_store.state.get("positions", []):
            if position.get("position_id") != position_id:
                continue
            if position.get("strategy_id") != self.config.strategy_id:
                raise StaleCloseFillError(f"{position_id} does not belong to {self.config.strategy_id}")
            if position.get("status") != "pending_close":
                raise StaleCloseFillError(f"{position_id} is not pending_close")
            return position
        raise StaleCloseFillError(f"{position_id} not found")

    @staticmethod
    def _validate_close_fill_inputs(
        *,
        order_ref: str,
        realized_pnl: float,
        fill_price: float,
        fill_time: Any,
    ) -> None:
        if order_ref is None or not str(order_ref).strip():
            raise MissingOrderRefError("close fill requires order_ref")
        if not isinstance(realized_pnl, Number) or isinstance(realized_pnl, bool):
            raise CloseFillValidationError("realized_pnl must be numeric")
        if not isinstance(fill_price, Number) or isinstance(fill_price, bool):
            raise CloseFillValidationError("fill_price must be numeric")
        if fill_price < 0:
            raise CloseFillValidationError("fill_price must be non-negative")
        if fill_time is None or fill_time == "":
            raise CloseFillValidationError("fill_time is required")

    @staticmethod
    def _close_metadata(position: dict[str, Any]) -> tuple[str, str]:
        source_management_action = position.get("source_management_action")
        close_reason = position.get("close_reason")
        if not isinstance(source_management_action, str) or not source_management_action.strip():
            raise CloseFillValidationError("pending close is missing source_management_action")
        if not isinstance(close_reason, str) or not close_reason.strip():
            raise CloseFillValidationError("pending close is missing close_reason")
        return source_management_action, close_reason
