"""Stage 4D-3 paper broker adapter translation and validation layer.

This module accepts an injected fake/test client only. It does not import or
instantiate broker clients, fetch market data, or wire execution into lifecycle
jobs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol

from algo_trader_unified.core.broker_adapter import (
    BrokerAccountSnapshot,
    BrokerCancelResult,
    BrokerOrderStatus,
    BrokerPosition,
    BrokerSubmitResult,
)
from algo_trader_unified.core.paper_broker_contract import validate_broker_mode


ALLOWED_SIDES = frozenset({"BUY", "SELL"})
ALLOWED_ORDER_TYPES = frozenset({"LIMIT", "MARKET"})


@dataclass(frozen=True)
class BrokerOrderRequest:
    client_order_id: str
    strategy_id: str
    symbol: str
    asset_type: str
    side: str
    quantity: float | int
    order_type: str
    limit_price: float | None
    time_in_force: str | None
    intent_id: str | None
    metadata: dict[str, Any] | None


class PaperBrokerClient(Protocol):
    def submit_order(self, request: BrokerOrderRequest) -> dict[str, Any]:
        ...

    def cancel_order(
        self,
        broker_order_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        ...

    def get_order_status(self, broker_order_id: str) -> dict[str, Any]:
        ...

    def list_open_orders(self) -> list[dict[str, Any]]:
        ...

    def list_positions(self) -> list[dict[str, Any]]:
        ...

    def get_account_snapshot(self) -> dict[str, Any]:
        ...


def sanitize_json_safe(value: Any) -> Any:
    """Return a recursively JSON-safe primitive representation."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Decimal):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if isinstance(value, dict):
        return {str(key): sanitize_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_safe(item) for item in value]
    return f"<{type(value).__module__}.{type(value).__name__}>"


def _sanitize_raw(raw: Any) -> dict[str, Any]:
    sanitized = sanitize_json_safe(raw)
    if isinstance(sanitized, dict):
        return sanitized
    return {"value": sanitized}


def _require_non_empty_string(intent: dict[str, Any], field: str) -> str:
    value = intent.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value.strip()


def _positive_number(value: Any, field: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{field} must be positive numeric")
    numeric = float(value) if isinstance(value, Decimal) else value
    if not math.isfinite(float(numeric)) or numeric <= 0:
        raise ValueError(f"{field} must be positive numeric")
    return numeric


def _optional_string(intent: dict[str, Any], field: str) -> str | None:
    value = intent.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string when present")
    stripped = value.strip()
    return stripped or None


def build_broker_order_request(intent: dict[str, Any]) -> BrokerOrderRequest:
    if not isinstance(intent, dict):
        raise ValueError("intent must be a dict")

    strategy_id = _require_non_empty_string(intent, "strategy_id")
    symbol = intent.get("symbol") or intent.get("underlying")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol or underlying is required")
    symbol = symbol.strip()

    side = _require_non_empty_string(intent, "side").upper()
    if side not in ALLOWED_SIDES:
        raise ValueError(f"side must be one of {sorted(ALLOWED_SIDES)}")

    quantity = _positive_number(intent.get("quantity"), "quantity")

    order_type = _require_non_empty_string(intent, "order_type").upper()
    if order_type not in ALLOWED_ORDER_TYPES:
        raise ValueError(f"order_type must be one of {sorted(ALLOWED_ORDER_TYPES)}")

    limit_price = None
    if order_type == "LIMIT":
        limit_price = float(_positive_number(intent.get("limit_price"), "limit_price"))
    elif intent.get("limit_price") is not None:
        raise ValueError("limit_price is only applicable for LIMIT orders")

    intent_id = intent.get("intent_id")
    if not isinstance(intent_id, str) or not intent_id.strip():
        raise ValueError("intent_id is required for deterministic client_order_id")
    intent_id = intent_id.strip()

    metadata = intent.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict when present")

    asset_type = intent.get("asset_type") or intent.get("security_type") or "UNKNOWN"
    if not isinstance(asset_type, str) or not asset_type.strip():
        raise ValueError("asset_type must be a string when present")

    return BrokerOrderRequest(
        client_order_id=intent_id,
        strategy_id=strategy_id,
        symbol=symbol,
        asset_type=asset_type.strip().upper(),
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        time_in_force=_optional_string(intent, "time_in_force"),
        intent_id=intent_id,
        metadata=sanitize_json_safe(metadata) if metadata is not None else None,
    )


class PaperBrokerAdapter:
    def __init__(
        self,
        *,
        mode: str,
        client: PaperBrokerClient,
        account_id: str | None = None,
        allow_live: bool = False,
    ) -> None:
        del allow_live
        self.mode = validate_broker_mode(mode)
        self.client = client
        self.account_id = account_id

    def submit_order_intent(self, intent: dict[str, Any]) -> BrokerSubmitResult:
        try:
            request = build_broker_order_request(intent)
        except (TypeError, ValueError) as exc:
            return BrokerSubmitResult(
                accepted=False,
                dry_run=self.mode == "DRY_RUN",
                broker_order_id=None,
                client_order_id=None,
                reason=str(exc),
                raw={"adapter": "PaperBrokerAdapter", "mode": self.mode},
            )

        if self.mode == "DRY_RUN":
            return BrokerSubmitResult(
                accepted=True,
                dry_run=True,
                broker_order_id=f"dry_run_{request.client_order_id}",
                client_order_id=request.client_order_id,
                reason=None,
                raw={
                    "adapter": "PaperBrokerAdapter",
                    "mode": self.mode,
                    "request": sanitize_json_safe(request.__dict__),
                },
            )

        try:
            response = self.client.submit_order(request)
        except Exception as exc:  # noqa: BLE001 - adapter boundary must fail closed.
            return BrokerSubmitResult(
                accepted=False,
                dry_run=False,
                broker_order_id=None,
                client_order_id=request.client_order_id,
                reason=f"{type(exc).__name__}: {exc}",
                raw={"adapter": "PaperBrokerAdapter", "mode": self.mode},
            )

        raw = _sanitize_raw(response)
        return BrokerSubmitResult(
            accepted=bool(response.get("accepted", True)) if isinstance(response, dict) else False,
            dry_run=False,
            broker_order_id=_optional_response_string(response, "broker_order_id"),
            client_order_id=_optional_response_string(response, "client_order_id")
            or request.client_order_id,
            reason=_optional_response_string(response, "reason"),
            raw=raw,
        )

    def cancel_order(
        self,
        broker_order_id: str,
        *,
        reason: str | None = None,
    ) -> BrokerCancelResult:
        if self.mode == "DRY_RUN":
            return BrokerCancelResult(
                cancelled=True,
                dry_run=True,
                broker_order_id=broker_order_id,
                reason=reason,
                raw={"adapter": "PaperBrokerAdapter", "mode": self.mode},
            )
        try:
            response = self.client.cancel_order(broker_order_id, reason)
        except Exception as exc:  # noqa: BLE001
            return BrokerCancelResult(
                cancelled=False,
                dry_run=False,
                broker_order_id=broker_order_id,
                reason=f"{type(exc).__name__}: {exc}",
                raw={"adapter": "PaperBrokerAdapter", "mode": self.mode},
            )
        return BrokerCancelResult(
            cancelled=bool(response.get("cancelled", False)),
            dry_run=False,
            broker_order_id=_optional_response_string(response, "broker_order_id")
            or broker_order_id,
            reason=_optional_response_string(response, "reason"),
            raw=_sanitize_raw(response),
        )

    def get_order_status(self, broker_order_id: str) -> BrokerOrderStatus:
        if self.mode == "DRY_RUN":
            return BrokerOrderStatus(
                broker_order_id=broker_order_id,
                client_order_id=None,
                status="DRY_RUN",
                filled_quantity=0,
                remaining_quantity=None,
                avg_fill_price=None,
                raw={"adapter": "PaperBrokerAdapter", "mode": self.mode},
            )
        try:
            response = self.client.get_order_status(broker_order_id)
        except Exception as exc:  # noqa: BLE001
            return _failure_status(broker_order_id, exc, self.mode)
        return _order_status_from_response(response, fallback_broker_order_id=broker_order_id)

    def list_open_orders(self) -> list[BrokerOrderStatus]:
        if self.mode == "DRY_RUN":
            return []
        # List-returning protocol methods cannot carry error details, so client
        # failures must remain visible instead of looking like empty results.
        responses = self.client.list_open_orders()
        return [
            _order_status_from_response(response, fallback_broker_order_id=f"open_{index}")
            for index, response in enumerate(responses)
        ]

    def list_positions(self) -> list[BrokerPosition]:
        if self.mode == "DRY_RUN":
            return []
        # List-returning protocol methods cannot carry error details, so client
        # failures must remain visible instead of looking like empty results.
        responses = self.client.list_positions()
        return [
            BrokerPosition(
                symbol=str(response.get("symbol", "")),
                quantity=_optional_number(response.get("quantity")) or 0,
                avg_price=_optional_float(response.get("avg_price")),
                raw=_sanitize_raw(response),
            )
            for response in responses
        ]

    def get_account_snapshot(self) -> BrokerAccountSnapshot:
        if self.mode == "DRY_RUN":
            return BrokerAccountSnapshot(
                net_liquidation=None,
                available_funds=None,
                buying_power=None,
                timestamp=None,
                raw={"adapter": "PaperBrokerAdapter", "mode": self.mode},
            )
        try:
            response = self.client.get_account_snapshot()
        except Exception as exc:  # noqa: BLE001
            return BrokerAccountSnapshot(
                net_liquidation=None,
                available_funds=None,
                buying_power=None,
                timestamp=None,
                raw={
                    "adapter": "PaperBrokerAdapter",
                    "mode": self.mode,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        return BrokerAccountSnapshot(
            net_liquidation=_optional_float(response.get("net_liquidation")),
            available_funds=_optional_float(response.get("available_funds")),
            buying_power=_optional_float(response.get("buying_power")),
            timestamp=_optional_response_string(response, "timestamp"),
            raw=_sanitize_raw(response),
        )


def _optional_response_string(response: dict[str, Any], field: str) -> str | None:
    value = response.get(field)
    if value is None:
        return None
    return str(sanitize_json_safe(value))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _optional_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    numeric = float(value) if isinstance(value, Decimal) else value
    return numeric if math.isfinite(float(numeric)) else None


def _order_status_from_response(
    response: dict[str, Any],
    *,
    fallback_broker_order_id: str,
) -> BrokerOrderStatus:
    return BrokerOrderStatus(
        broker_order_id=_optional_response_string(response, "broker_order_id")
        or fallback_broker_order_id,
        client_order_id=_optional_response_string(response, "client_order_id"),
        status=str(response.get("status", "UNKNOWN")),
        filled_quantity=_optional_number(response.get("filled_quantity")),
        remaining_quantity=_optional_number(response.get("remaining_quantity")),
        avg_fill_price=_optional_float(response.get("avg_fill_price")),
        raw=_sanitize_raw(response),
    )


def _failure_status(
    broker_order_id: str,
    exc: Exception,
    mode: str,
) -> BrokerOrderStatus:
    return BrokerOrderStatus(
        broker_order_id=broker_order_id,
        client_order_id=None,
        status="ERROR",
        filled_quantity=None,
        remaining_quantity=None,
        avg_fill_price=None,
        raw={
            "adapter": "PaperBrokerAdapter",
            "mode": mode,
            "error": f"{type(exc).__name__}: {exc}",
        },
    )
