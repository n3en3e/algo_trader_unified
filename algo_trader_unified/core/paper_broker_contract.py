"""Stage 4D paper broker safety contract helpers."""

from __future__ import annotations

from typing import Any

from algo_trader_unified.core.broker_adapter import (
    BrokerAccountSnapshot,
    BrokerAdapter,
    BrokerCancelResult,
    BrokerOrderStatus,
    BrokerPosition,
    BrokerSubmitResult,
)


VALID_BROKER_MODES = frozenset({"DRY_RUN", "PAPER"})


class BrokerModeError(ValueError):
    """Raised when a broker adapter mode is not explicitly allowed."""


def validate_broker_mode(mode: str) -> str:
    """Return a validated broker mode, rejecting live or unknown modes."""

    if mode in VALID_BROKER_MODES:
        return mode
    raise BrokerModeError(f"Broker mode is not allowed for Stage 4D-1: {mode!r}")


class NullBrokerAdapter:
    """Inert adapter for interface contract tests only.

    This class does not connect to broker services, fetch market data, write
    ledgers, or mutate StateStore. Raw result payloads are already serialized
    JSON-safe dictionaries.
    """

    def __init__(
        self,
        *,
        mode: str = "DRY_RUN",
        accept_submissions: bool = True,
        cancel_succeeds: bool = True,
        broker_order_id: str = "null_order_0001",
        client_order_id: str | None = "null_client_0001",
    ) -> None:
        self.mode = validate_broker_mode(mode)
        self.accept_submissions = accept_submissions
        self.cancel_succeeds = cancel_succeeds
        self.broker_order_id = broker_order_id
        self.client_order_id = client_order_id

    def submit_order_intent(self, intent: dict[str, Any]) -> BrokerSubmitResult:
        client_order_id = intent.get("client_order_id", self.client_order_id)
        if client_order_id is not None:
            client_order_id = str(client_order_id)
        return BrokerSubmitResult(
            accepted=self.accept_submissions,
            dry_run=True,
            broker_order_id=self.broker_order_id if self.accept_submissions else None,
            client_order_id=client_order_id,
            reason=None if self.accept_submissions else "null_adapter_rejected",
            raw={
                "adapter": "NullBrokerAdapter",
                "mode": self.mode,
                "intent_keys": sorted(str(key) for key in intent.keys()),
            },
        )

    def cancel_order(
        self,
        broker_order_id: str,
        *,
        reason: str | None = None,
    ) -> BrokerCancelResult:
        return BrokerCancelResult(
            cancelled=self.cancel_succeeds,
            dry_run=True,
            broker_order_id=broker_order_id,
            reason=reason if reason is not None else "null_adapter_cancel",
            raw={
                "adapter": "NullBrokerAdapter",
                "mode": self.mode,
                "cancel_succeeds": self.cancel_succeeds,
            },
        )

    def get_order_status(self, broker_order_id: str) -> BrokerOrderStatus:
        return BrokerOrderStatus(
            broker_order_id=broker_order_id,
            client_order_id=self.client_order_id,
            status="OPEN",
            filled_quantity=0,
            remaining_quantity=None,
            avg_fill_price=None,
            raw={
                "adapter": "NullBrokerAdapter",
                "mode": self.mode,
                "source": "deterministic",
            },
        )

    def list_open_orders(self) -> list[BrokerOrderStatus]:
        return [self.get_order_status(self.broker_order_id)] if self.accept_submissions else []

    def list_positions(self) -> list[BrokerPosition]:
        return []

    def get_account_snapshot(self) -> BrokerAccountSnapshot:
        return BrokerAccountSnapshot(
            net_liquidation=None,
            available_funds=None,
            buying_power=None,
            timestamp=None,
            raw={
                "adapter": "NullBrokerAdapter",
                "mode": self.mode,
                "positions": 0,
            },
        )


def assert_broker_adapter(adapter: BrokerAdapter) -> BrokerAdapter:
    """Return adapter after runtime protocol validation for tests and factories."""

    if not isinstance(adapter, BrokerAdapter):
        raise TypeError("object does not satisfy BrokerAdapter")
    return adapter
