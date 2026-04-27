"""Base strategy interfaces for dependency-injected engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from algo_trader_unified.config.variants import StrategyVariantConfig
from algo_trader_unified.core.broker import IBKRBrokerWrapper
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.state_store import StateStore


@runtime_checkable
class RiskManagerProtocol(Protocol):
    def can_enter(self, strategy_id: str) -> bool:
        """Return True if new entries are currently allowed."""

    def is_halted(self, strategy_id: str) -> bool:
        """Return True if the strategy is halted."""


class Phase2ARiskManagerStub:
    def can_enter(self, strategy_id: str) -> bool:
        return True

    def is_halted(self, strategy_id: str) -> bool:
        return False


@dataclass
class BaseStrategy:
    config: StrategyVariantConfig
    state_store: StateStore
    ledger: LedgerAppender
    broker: IBKRBrokerWrapper
    risk_manager: RiskManagerProtocol
