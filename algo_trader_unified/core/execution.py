"""Internal dry-run execution adapters."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
from typing import Any, TypedDict


class DryRunOrderSubmission(TypedDict):
    dry_run: bool
    simulated_order_id: str
    submitted_at: str
    order_ref: str | None
    intent_id: str
    strategy_id: str
    symbol: str | None
    action: str
    status: str


class DryRunCloseOrderSubmission(TypedDict):
    dry_run: bool
    close_intent_id: str
    position_id: str
    strategy_id: str
    symbol: str | None
    close_order_ref: str
    simulated_close_order_id: str
    submitted_at: str
    status: str
    action: str


class DryRunOrderStatus(TypedDict):
    dry_run: bool
    simulated_order_id: str
    checked_at: str
    confirmed_at: str
    intent_id: str
    strategy_id: str
    symbol: str | None
    order_ref: str | None
    status: str
    action: str


class DryRunFillStatus(TypedDict):
    dry_run: bool
    simulated_order_id: str
    checked_at: str
    filled_at: str
    fill_id: str
    fill_price: float | Decimal
    fill_quantity: int | float | Decimal
    intent_id: str
    strategy_id: str
    symbol: str | None
    order_ref: str | None
    status: str
    action: str


class DryRunExecutionAdapter:
    """Pure adapter that simulates submission for order-intent lifecycle tests."""

    def submit_order_intent(
        self,
        intent: dict[str, Any],
        *,
        submitted_at: str,
    ) -> DryRunOrderSubmission:
        intent_id = intent["intent_id"]
        strategy_id = intent["strategy_id"]
        order_ref = intent.get("order_ref")
        digest = sha256(
            f"{intent_id}|{strategy_id}|{order_ref}|{submitted_at}".encode("utf-8")
        ).hexdigest()[:16]
        return {
            "dry_run": True,
            "simulated_order_id": f"sim_{digest}",
            "submitted_at": submitted_at,
            "order_ref": order_ref,
            "intent_id": intent_id,
            "strategy_id": strategy_id,
            "symbol": intent.get("symbol"),
            "action": "open",
            "status": "submitted",
        }

    def submit_close_intent(
        self,
        close_intent: dict[str, Any],
        *,
        submitted_at: str,
    ) -> DryRunCloseOrderSubmission:
        close_intent_id = close_intent["close_intent_id"]
        position_id = close_intent["position_id"]
        strategy_id = close_intent["strategy_id"]
        close_order_ref = f"{strategy_id}:{position_id}:{close_intent_id}:close"
        digest = sha256(
            f"{close_intent_id}|{position_id}|{strategy_id}|{close_order_ref}|{submitted_at}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        return {
            "dry_run": True,
            "close_intent_id": close_intent_id,
            "position_id": position_id,
            "strategy_id": strategy_id,
            "symbol": close_intent.get("symbol"),
            "close_order_ref": close_order_ref,
            "simulated_close_order_id": f"sim_close_{digest}",
            "submitted_at": submitted_at,
            "status": "submitted",
            "action": "close",
        }

    def check_order_status(
        self,
        *,
        simulated_order_id: str,
        intent: dict[str, Any],
        checked_at: str,
    ) -> DryRunOrderStatus:
        return {
            "dry_run": True,
            "simulated_order_id": simulated_order_id,
            "checked_at": checked_at,
            "confirmed_at": checked_at,
            "intent_id": intent["intent_id"],
            "strategy_id": intent["strategy_id"],
            "symbol": intent.get("symbol"),
            "order_ref": intent.get("order_ref"),
            "status": "confirmed",
            "action": "open",
        }

    def check_for_fills(
        self,
        *,
        simulated_order_id: str,
        intent: dict[str, Any],
        checked_at: str,
    ) -> DryRunFillStatus:
        digest = sha256(
            f"{simulated_order_id}|{intent['intent_id']}|{checked_at}".encode("utf-8")
        ).hexdigest()[:16]
        return {
            "dry_run": True,
            "simulated_order_id": simulated_order_id,
            "checked_at": checked_at,
            "filled_at": checked_at,
            "fill_id": f"fill_{digest}",
            "fill_price": 0.0,
            "fill_quantity": 1,
            "intent_id": intent["intent_id"],
            "strategy_id": intent["strategy_id"],
            "symbol": intent.get("symbol"),
            "order_ref": intent.get("order_ref"),
            "status": "filled",
            "action": "open",
        }
