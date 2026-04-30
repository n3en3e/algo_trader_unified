"""Internal dry-run execution adapters."""

from __future__ import annotations

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
