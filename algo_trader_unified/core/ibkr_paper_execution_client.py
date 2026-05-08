"""Stage 4E-3 isolated IBKR paper submit/cancel client contract.

This module accepts an injected fake/test IB-like object only. It does not
import IBKR runtime libraries, instantiate broker clients, fetch market data,
qualify contracts, or wire execution into lifecycle jobs.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from decimal import Decimal
from typing import Any, Protocol

from algo_trader_unified.core.broker_adapter import (
    BrokerCancelResult,
    BrokerOrderStatus,
    BrokerSubmitResult,
)
from algo_trader_unified.core.ibkr_paper_client import IbkrConnectionStatus
from algo_trader_unified.core.ibkr_paper_order_mapper import (
    IBKR_PAPER_PORT,
    PAPER_TRADING_MODE,
    IbkrPaperConfig,
    IbkrPaperOrderPlan,
    validate_ibkr_paper_config,
)
from algo_trader_unified.core.paper_broker_adapter import sanitize_json_safe


class IbExecutionLike(Protocol):
    def connect(self, host: str, port: int, clientId: int) -> Any:
        ...

    def disconnect(self) -> Any:
        ...

    def isConnected(self) -> bool:
        ...

    def placeOrder(self, contract: dict[str, Any], order: dict[str, Any]) -> Any:
        ...

    def cancelOrder(self, broker_order_id: str, reason: str | None = None) -> Any:
        ...

    def get_order_status(self, broker_order_id: str) -> Any:
        ...


class IbkrPaperExecutionClient:
    """Paper-only submit/cancel wrapper around an injected fake IB-like client."""

    def __init__(
        self,
        *,
        ib: IbExecutionLike,
        config: IbkrPaperConfig,
    ) -> None:
        self.ib = ib
        self.config = _validate_submit_config(config)

    def connect(self) -> IbkrConnectionStatus:
        try:
            result = self.ib.connect(
                self.config.host,
                self.config.port,
                clientId=self.config.client_id,
            )
            connected = self._is_connected()
        except Exception as exc:  # noqa: BLE001
            return self._connection_status(
                connected=False,
                reason=f"{type(exc).__name__}: {exc}",
                raw={"operation": "connect", "error": f"{type(exc).__name__}: {exc}"},
            )
        return self._connection_status(
            connected=connected,
            reason=None if connected else "connect returned but client is not connected",
            raw={"operation": "connect", "result": result},
        )

    def disconnect(self) -> IbkrConnectionStatus:
        try:
            result = self.ib.disconnect()
            connected = self._is_connected()
        except Exception as exc:  # noqa: BLE001
            return self._connection_status(
                connected=False,
                reason=f"{type(exc).__name__}: {exc}",
                raw={"operation": "disconnect", "error": f"{type(exc).__name__}: {exc}"},
            )
        return self._connection_status(
            connected=connected,
            reason=None,
            raw={"operation": "disconnect", "result": result},
        )

    def submit_order_plan(self, plan: IbkrPaperOrderPlan) -> BrokerSubmitResult:
        client_order_id = _optional_string(_plan_value(plan, "client_order_id"))

        reason = _submit_blocker(plan)
        if reason is not None:
            return BrokerSubmitResult(
                accepted=False,
                dry_run=True,
                broker_order_id=None,
                client_order_id=client_order_id,
                reason=reason,
                raw={"operation": "placeOrder", "validation_error": reason},
            )

        contract_hint = _json_safe_dict(_plan_value(plan, "ibkr_contract_hint"))
        order_hint = _json_safe_dict(_plan_value(plan, "ibkr_order_hint"))
        order_hint.setdefault("orderRef", client_order_id)

        try:
            response = self.ib.placeOrder(dict(contract_hint), dict(order_hint))
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            return BrokerSubmitResult(
                accepted=False,
                dry_run=True,
                broker_order_id=None,
                client_order_id=client_order_id,
                reason=reason,
                raw={"operation": "placeOrder", "error": reason},
            )

        parsed = _parse_trade_response(response, fallback_client_order_id=client_order_id)
        accepted = _response_accepted(response, parsed["status"])
        return BrokerSubmitResult(
            accepted=accepted,
            dry_run=True,
            broker_order_id=parsed["broker_order_id"],
            client_order_id=parsed["client_order_id"] or client_order_id,
            reason=_response_reason(response) if not accepted else None,
            raw=_json_safe_dict(
                {
                    "operation": "placeOrder",
                    "contract": contract_hint,
                    "order": order_hint,
                    "response": parsed,
                }
            ),
        )

    def cancel_order(
        self,
        broker_order_id: str,
        *,
        reason: str | None = None,
    ) -> BrokerCancelResult:
        broker_order_id = broker_order_id.strip() if isinstance(broker_order_id, str) else ""
        if not broker_order_id:
            return BrokerCancelResult(
                cancelled=False,
                dry_run=True,
                broker_order_id=None,
                reason="broker_order_id must be a non-empty string",
                raw={"operation": "cancelOrder", "validation_error": "empty broker_order_id"},
            )

        try:
            response = self.ib.cancelOrder(broker_order_id, reason)
        except Exception as exc:  # noqa: BLE001
            failure_reason = f"{type(exc).__name__}: {exc}"
            return BrokerCancelResult(
                cancelled=False,
                dry_run=True,
                broker_order_id=broker_order_id,
                reason=failure_reason,
                raw={"operation": "cancelOrder", "error": failure_reason},
            )

        parsed = _parse_trade_response(response, fallback_broker_order_id=broker_order_id)
        cancelled = _response_cancelled(response, parsed["status"])
        return BrokerCancelResult(
            cancelled=cancelled,
            dry_run=True,
            broker_order_id=parsed["broker_order_id"] or broker_order_id,
            reason=_response_reason(response) if not cancelled else reason,
            raw=_json_safe_dict(
                {
                    "operation": "cancelOrder",
                    "broker_order_id": broker_order_id,
                    "reason": reason,
                    "response": parsed,
                }
            ),
        )

    def get_order_status(self, broker_order_id: str) -> BrokerOrderStatus:
        broker_order_id = broker_order_id.strip() if isinstance(broker_order_id, str) else ""
        if not broker_order_id:
            return BrokerOrderStatus(
                broker_order_id="",
                client_order_id=None,
                status="ERROR",
                filled_quantity=None,
                remaining_quantity=None,
                avg_fill_price=None,
                raw={"operation": "get_order_status", "validation_error": "empty broker_order_id"},
            )

        try:
            response = self.ib.get_order_status(broker_order_id)
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            return BrokerOrderStatus(
                broker_order_id=broker_order_id,
                client_order_id=None,
                status="ERROR",
                filled_quantity=None,
                remaining_quantity=None,
                avg_fill_price=None,
                raw={"operation": "get_order_status", "error": reason},
            )

        parsed = _parse_trade_response(response, fallback_broker_order_id=broker_order_id)
        return BrokerOrderStatus(
            broker_order_id=parsed["broker_order_id"] or broker_order_id,
            client_order_id=parsed["client_order_id"],
            status=parsed["status"] or "UNKNOWN",
            filled_quantity=_optional_number(parsed["filled_quantity"]),
            remaining_quantity=_optional_number(parsed["remaining_quantity"]),
            avg_fill_price=_optional_float(parsed["avg_fill_price"]),
            raw=_json_safe_dict({"operation": "get_order_status", "response": parsed}),
        )

    def _is_connected(self) -> bool:
        return bool(self.ib.isConnected())

    def _connection_status(
        self,
        *,
        connected: bool,
        reason: str | None,
        raw: dict[str, Any] | None,
    ) -> IbkrConnectionStatus:
        return IbkrConnectionStatus(
            connected=connected,
            paper_mode=self.config.trading_mode == PAPER_TRADING_MODE
            and self.config.port == IBKR_PAPER_PORT,
            host=self.config.host,
            port=self.config.port,
            client_id=self.config.client_id,
            account_id=self.config.account_id,
            reason=reason,
            raw=_json_safe_dict(raw) if raw is not None else None,
        )


def _validate_submit_config(config: IbkrPaperConfig) -> IbkrPaperConfig:
    if not isinstance(config, IbkrPaperConfig):
        raise TypeError("config must be an IbkrPaperConfig")
    return validate_ibkr_paper_config(asdict(config))


def _submit_blocker(plan: Any) -> str | None:
    if not isinstance(plan, (IbkrPaperOrderPlan, dict)):
        return "plan must be an IbkrPaperOrderPlan or dict-compatible object"
    if _plan_value(plan, "ready_for_submission") is not True:
        return "ready_for_submission must be True"
    if _plan_value(plan, "paper_only") is not True:
        return "paper_only must be True"
    if _plan_value(plan, "dry_run") is not True:
        return "dry_run must be True"
    blockers = _plan_value(plan, "blockers")
    if blockers:
        return "blockers must be empty"
    client_order_id = _plan_value(plan, "client_order_id")
    if not isinstance(client_order_id, str) or not client_order_id.strip():
        return "client_order_id must be a non-empty string"
    if not isinstance(_plan_value(plan, "ibkr_contract_hint"), dict):
        return "ibkr_contract_hint must be a plain dict"
    if not isinstance(_plan_value(plan, "ibkr_order_hint"), dict):
        return "ibkr_order_hint must be a plain dict"
    return None


def _plan_value(plan: Any, name: str) -> Any:
    if isinstance(plan, dict):
        return plan.get(name)
    return getattr(plan, name, None)


def _parse_trade_response(
    response: Any,
    *,
    fallback_broker_order_id: str | None = None,
    fallback_client_order_id: str | None = None,
) -> dict[str, Any]:
    order = _get_value(response, "order")
    status_obj = _get_value(response, "orderStatus")
    broker_order_id = (
        _optional_string(_get_value(response, "broker_order_id"))
        or _optional_string(_get_value(response, "orderId"))
        or _optional_string(_get_value(order, "orderId"))
        or fallback_broker_order_id
    )
    client_order_id = (
        _optional_string(_get_value(response, "client_order_id"))
        or _optional_string(_get_value(response, "orderRef"))
        or _optional_string(_get_value(order, "orderRef"))
        or fallback_client_order_id
    )
    status = (
        _optional_string(_get_value(response, "status"))
        or _optional_string(_get_value(status_obj, "status"))
        or "UNKNOWN"
    )
    return _json_safe_dict(
        {
            "broker_order_id": broker_order_id,
            "client_order_id": client_order_id,
            "status": status,
            "filled_quantity": _first_value(
                _get_value(response, "filled_quantity"),
                _get_value(status_obj, "filled"),
            ),
            "remaining_quantity": _first_value(
                _get_value(response, "remaining_quantity"),
                _get_value(status_obj, "remaining"),
            ),
            "avg_fill_price": _first_value(
                _get_value(response, "avg_fill_price"),
                _get_value(status_obj, "avgFillPrice"),
            ),
            "reason": _response_reason(response),
            "accepted": _get_value(response, "accepted"),
            "cancelled": _get_value(response, "cancelled"),
            "contract": _known_attrs(
                _get_value(response, "contract"),
                ("symbol", "secType", "lastTradeDateOrContractMonth"),
            ),
        }
    )


def _response_accepted(response: Any, status: Any) -> bool:
    accepted = _get_value(response, "accepted")
    if isinstance(accepted, bool):
        return accepted
    normalized = str(status or "").strip().upper()
    return normalized not in {"REJECTED", "INACTIVE", "ERROR"}


def _response_cancelled(response: Any, status: Any) -> bool:
    cancelled = _get_value(response, "cancelled")
    if isinstance(cancelled, bool):
        return cancelled
    normalized = str(status or "").strip().upper()
    return normalized in {"CANCELLED", "CANCELED", "PENDINGCANCEL"}


def _response_reason(response: Any) -> str | None:
    return (
        _optional_string(_get_value(response, "reason"))
        or _optional_string(_get_value(response, "message"))
        or _optional_string(_get_value(response, "error"))
    )


def _json_safe_dict(value: Any) -> dict[str, Any]:
    safe = sanitize_json_safe(value)
    if isinstance(safe, dict):
        return safe
    return {"value": safe}


def _get_value(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    safe = sanitize_json_safe(value)
    if safe is None:
        return None
    text = str(safe)
    return text if text else None


def _optional_float(value: Any) -> float | None:
    number = _optional_number(value)
    if number is None:
        return None
    return float(number)


def _optional_number(value: Any) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (float, Decimal)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            number = float(stripped)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _known_attrs(value: Any, names: tuple[str, ...]) -> dict[str, Any] | None:
    if value is None:
        return None
    return _json_safe_dict({name: _get_value(value, name) for name in names})
