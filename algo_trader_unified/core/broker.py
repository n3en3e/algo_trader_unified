"""Execution-facing broker wrapper stub for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from algo_trader_unified.config.portfolio import DIAGNOSTIC_CLIENT_ID_RANGE


class DiagnosticClientOrderError(RuntimeError):
    """Raised when a diagnostic client attempts to submit an order."""


class MissingOrderRefError(RuntimeError):
    """Raised when an IBKR order lacks orderRef attribution."""


@dataclass(frozen=True)
class BrokerSubmissionResult:
    submitted: bool
    detail: str
    trade: Any | None = None


class IBKRBrokerWrapper:
    def __init__(self, ibkr_client: Any | None = None) -> None:
        self.ibkr_client = ibkr_client

    def _assert_submission_allowed(self, client_id: int, order_ref: str | None) -> None:
        if client_id in DIAGNOSTIC_CLIENT_ID_RANGE:
            raise DiagnosticClientOrderError(
                f"Diagnostic client_id {client_id} may not submit orders"
            )
        if order_ref is None or str(order_ref).strip() == "":
            raise MissingOrderRefError("IBKR order submission requires orderRef")

    def submit_order(
        self,
        *,
        client_id: int,
        contract: Any,
        order: Any,
        order_ref: str | None,
    ) -> BrokerSubmissionResult:
        self._assert_submission_allowed(client_id, order_ref)
        if hasattr(order, "orderRef"):
            order.orderRef = order_ref
        if self.ibkr_client is None:
            return BrokerSubmissionResult(submitted=False, detail="No IBKR client attached")
        trade = self.ibkr_client.placeOrder(contract, order)
        return BrokerSubmissionResult(submitted=True, detail="submitted", trade=trade)

