"""Stage 4E-1 read-only IBKR paper connectivity client.

This module is intentionally dependency-injected and does not import IBKR
runtime libraries. It exposes read-only paper Gateway checks and safe broker
state snapshots through synchronous methods only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Protocol

from algo_trader_unified.core.broker_adapter import (
    BrokerAccountSnapshot,
    BrokerOrderStatus,
    BrokerPosition,
    assert_json_safe_raw,
)
from algo_trader_unified.core.ibkr_paper_order_mapper import (
    IBKR_PAPER_PORT,
    PAPER_TRADING_MODE,
    IbkrPaperConfig,
    validate_ibkr_paper_config,
)
from algo_trader_unified.core.paper_broker_adapter import sanitize_json_safe


class IbkrPaperClientError(RuntimeError):
    """Raised when a read-only IBKR client operation cannot be completed."""


class IbLike(Protocol):
    def connect(self, host: str, port: int, clientId: int) -> Any:
        ...

    def disconnect(self) -> Any:
        ...

    def isConnected(self) -> bool:
        ...

    def reqCurrentTime(self) -> Any:
        ...

    def accountSummary(self) -> Any:
        ...

    def openOrders(self) -> Any:
        ...

    def positions(self) -> Any:
        ...


@dataclass(frozen=True)
class IbkrConnectionStatus:
    connected: bool
    paper_mode: bool
    host: str
    port: int
    client_id: int
    account_id: str | None
    reason: str | None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        assert_json_safe_raw(self.raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IbkrPaperReadOnlyClient:
    """Read-only, paper-only IBKR client wrapper for injected IB-like clients."""

    def __init__(
        self,
        *,
        ib: IbLike,
        config: IbkrPaperConfig,
    ) -> None:
        self.ib = ib
        self.config = _validate_config_dataclass(config)

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
                raw={"operation": "connect", "error": exc},
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
                raw={"operation": "disconnect", "error": exc},
            )
        return self._connection_status(
            connected=connected,
            reason=None,
            raw={"operation": "disconnect", "result": result},
        )

    def get_connection_status(self) -> IbkrConnectionStatus:
        try:
            connected = self._is_connected()
        except Exception as exc:  # noqa: BLE001
            return self._connection_status(
                connected=False,
                reason=f"{type(exc).__name__}: {exc}",
                raw={"operation": "isConnected", "error": exc},
            )
        return self._connection_status(connected=connected, reason=None, raw=None)

    def get_current_time(self) -> dict[str, Any]:
        try:
            current_time = self.ib.reqCurrentTime()
        except Exception as exc:  # noqa: BLE001
            raise IbkrPaperClientError(
                f"IBKR read-only current time request failed: {type(exc).__name__}: {exc}"
            ) from exc
        return _json_safe_dict({"current_time": current_time})

    def get_account_snapshot(self) -> BrokerAccountSnapshot:
        try:
            response = self.ib.accountSummary()
        except Exception as exc:  # noqa: BLE001
            return BrokerAccountSnapshot(
                net_liquidation=None,
                available_funds=None,
                buying_power=None,
                timestamp=None,
                raw=_json_safe_dict(
                    {
                        "operation": "accountSummary",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ),
            )

        rows = [_normalize_account_summary_row(row) for row in _as_list(response)]
        values = _account_summary_values(rows)
        return BrokerAccountSnapshot(
            net_liquidation=_optional_float(values.get("NetLiquidation")),
            available_funds=_optional_float(values.get("AvailableFunds")),
            buying_power=_optional_float(values.get("BuyingPower")),
            timestamp=_optional_string(values.get("timestamp")),
            raw=_json_safe_dict({"operation": "accountSummary", "rows": rows}),
        )

    def list_open_orders(self) -> list[BrokerOrderStatus]:
        try:
            orders = self.ib.openOrders()
        except Exception as exc:  # noqa: BLE001
            raise IbkrPaperClientError(
                f"IBKR read-only open orders request failed: {type(exc).__name__}: {exc}"
            ) from exc
        return [
            _open_order_status(order, fallback_broker_order_id=f"open_{index}")
            for index, order in enumerate(_as_list(orders))
        ]

    def list_positions(self) -> list[BrokerPosition]:
        try:
            positions = self.ib.positions()
        except Exception as exc:  # noqa: BLE001
            raise IbkrPaperClientError(
                f"IBKR read-only positions request failed: {type(exc).__name__}: {exc}"
            ) from exc
        return [_broker_position(position) for position in _as_list(positions)]

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


def _validate_config_dataclass(config: IbkrPaperConfig) -> IbkrPaperConfig:
    if not isinstance(config, IbkrPaperConfig):
        raise TypeError("config must be an IbkrPaperConfig")
    return validate_ibkr_paper_config(asdict(config))


def _json_safe_dict(value: Any) -> dict[str, Any]:
    safe = sanitize_json_safe(value)
    if isinstance(safe, dict):
        return safe
    return {"value": safe}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _get_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _get_nested(value: Any, *names: str) -> Any:
    current = value
    for name in names:
        current = _get_value(current, name)
        if current is None:
            return None
    return current


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(sanitize_json_safe(value))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalize_account_summary_row(row: Any) -> dict[str, Any]:
    return _json_safe_dict(
        {
            "account": _get_value(row, "account"),
            "tag": _get_value(row, "tag"),
            "value": _get_value(row, "value"),
            "currency": _get_value(row, "currency"),
        }
    )


def _account_summary_values(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for row in rows:
        tag = row.get("tag")
        if isinstance(tag, str):
            values[tag] = row.get("value")
    return values


def _open_order_status(order: Any, *, fallback_broker_order_id: str) -> BrokerOrderStatus:
    broker_order_id = (
        _optional_string(_get_value(order, "broker_order_id"))
        or _optional_string(_get_nested(order, "order", "orderId"))
        or _optional_string(_get_value(order, "orderId"))
        or fallback_broker_order_id
    )
    client_order_id = (
        _optional_string(_get_value(order, "client_order_id"))
        or _optional_string(_get_nested(order, "order", "orderRef"))
        or _optional_string(_get_value(order, "orderRef"))
    )
    status = (
        _optional_string(_get_value(order, "status"))
        or _optional_string(_get_nested(order, "orderStatus", "status"))
        or "UNKNOWN"
    )
    filled = _first_number(
        _get_value(order, "filled_quantity"),
        _get_nested(order, "orderStatus", "filled"),
        _get_value(order, "filled"),
    )
    remaining = _first_number(
        _get_value(order, "remaining_quantity"),
        _get_nested(order, "orderStatus", "remaining"),
        _get_value(order, "remaining"),
    )
    avg_fill_price = _first_float(
        _get_value(order, "avg_fill_price"),
        _get_nested(order, "orderStatus", "avgFillPrice"),
        _get_value(order, "avgFillPrice"),
    )
    return BrokerOrderStatus(
        broker_order_id=broker_order_id,
        client_order_id=client_order_id,
        status=status,
        filled_quantity=filled,
        remaining_quantity=remaining,
        avg_fill_price=avg_fill_price,
        raw=_json_safe_dict(
            {
                "broker_order_id": broker_order_id,
                "client_order_id": client_order_id,
                "status": status,
                "filled_quantity": filled,
                "remaining_quantity": remaining,
                "avg_fill_price": avg_fill_price,
                "contract": _known_attrs(
                    _get_value(order, "contract"),
                    ("symbol", "secType", "lastTradeDateOrContractMonth"),
                ),
            }
        ),
    )


def _broker_position(position: Any) -> BrokerPosition:
    contract = _get_value(position, "contract")
    symbol = (
        _optional_string(_get_value(position, "symbol"))
        or _optional_string(_get_value(contract, "symbol"))
        or ""
    )
    quantity = _first_number(_get_value(position, "quantity"), _get_value(position, "position"))
    avg_price = _first_float(_get_value(position, "avg_price"), _get_value(position, "avgCost"))
    return BrokerPosition(
        symbol=symbol,
        quantity=quantity if quantity is not None else 0,
        avg_price=avg_price,
        raw=_json_safe_dict(
            {
                "symbol": symbol,
                "quantity": quantity,
                "avg_price": avg_price,
                "contract": _known_attrs(contract, ("symbol", "secType")),
                "account": _get_value(position, "account"),
            }
        ),
    )


def _known_attrs(value: Any, names: tuple[str, ...]) -> dict[str, Any] | None:
    if value is None:
        return None
    return _json_safe_dict({name: _get_value(value, name) for name in names})


def _first_number(*values: Any) -> float | int | None:
    for value in values:
        number = _optional_number(value)
        if number is not None:
            return number
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        number = _optional_float(value)
        if number is not None:
            return number
    return None
