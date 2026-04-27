"""Readiness state manager for scheduler-driven gate checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.state_store import StateStore


@dataclass(frozen=True)
class ReadinessStatus:
    strategy_id: str
    ready_for_entries: bool
    reason: str | None
    checked_at: str
    dirty_state: bool
    unknown_broker_exposure: bool
    nlv_degraded: bool
    halt_active: bool
    calendar_expired: bool
    iv_baseline_available: bool | None


@dataclass(frozen=True)
class MarketOpenReadinessResult:
    checked_at: str
    statuses: dict[str, ReadinessStatus]
    all_clear: bool


class ReadinessManager:
    """Owns readiness state updates through StateStore only."""

    def __init__(self, state_store: StateStore, ledger: LedgerAppender) -> None:
        self.state_store = state_store
        self.ledger = ledger

    def update_readiness(self, status: ReadinessStatus) -> None:
        self.state_store.update_readiness(status.strategy_id, asdict(status))

    def store_market_open_result(self, result: MarketOpenReadinessResult) -> None:
        for status in result.statuses.values():
            self.update_readiness(status)

    def get_readiness(self, strategy_id: str) -> ReadinessStatus | dict[str, Any] | None:
        payload = self.state_store.get_readiness(strategy_id)
        if payload is None:
            return None
        expected = set(ReadinessStatus.__dataclass_fields__)
        if expected.issubset(payload):
            return ReadinessStatus(**{key: payload[key] for key in expected})
        return payload

    def get_all_readiness(self) -> dict[str, Any]:
        return self.state_store.get_all_readiness()
