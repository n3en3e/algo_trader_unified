"""Paper broker adapter interface for Stage 4D.

Implementing adapters must serialize proprietary broker objects before returning
them through any ``raw`` field. Raw payloads may contain only JSON-safe
primitive values, lists, and dictionaries.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable


class BrokerRawValueError(ValueError):
    """Raised when a broker result raw payload is not JSON-safe."""


def _is_json_safe_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_safe_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_safe_value(item)
            for key, item in value.items()
        )
    return False


def assert_json_safe_raw(raw: dict[str, Any] | None) -> None:
    """Require raw broker payloads to contain only JSON-safe primitives.

    Proprietary broker objects, response handles, decimal values, datetime
    values, contracts, and order objects must be serialized by implementing
    classes before they are exposed through ``raw``.
    """

    if raw is None:
        return
    if not isinstance(raw, dict):
        raise BrokerRawValueError("broker raw payload must be a dict or None")
    if not _is_json_safe_value(raw):
        raise BrokerRawValueError("broker raw payload must contain only JSON-safe values")
    try:
        json.dumps(raw, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise BrokerRawValueError("broker raw payload must be JSON-serializable") from exc


@dataclass(frozen=True)
class BrokerSubmitResult:
    accepted: bool
    dry_run: bool
    broker_order_id: str | None
    client_order_id: str | None
    reason: str | None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        assert_json_safe_raw(self.raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrokerCancelResult:
    cancelled: bool
    dry_run: bool
    broker_order_id: str | None
    reason: str | None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        assert_json_safe_raw(self.raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrokerOrderStatus:
    broker_order_id: str
    client_order_id: str | None
    status: str
    filled_quantity: float | int | None
    remaining_quantity: float | int | None
    avg_fill_price: float | None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        assert_json_safe_raw(self.raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    quantity: float | int
    avg_price: float | None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        assert_json_safe_raw(self.raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrokerAccountSnapshot:
    net_liquidation: float | None
    available_funds: float | None
    buying_power: float | None
    timestamp: str | None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        assert_json_safe_raw(self.raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class BrokerAdapter(Protocol):
    """Protocol for inert paper-execution adapter implementations.

    Implementing classes must expose an explicit ``mode`` and must serialize
    proprietary broker objects before returning them through any result ``raw``
    field. This interface does not authorize live execution or external broker
    calls by itself.
    """

    mode: str

    def submit_order_intent(self, intent: dict[str, Any]) -> BrokerSubmitResult:
        ...

    def cancel_order(
        self,
        broker_order_id: str,
        *,
        reason: str | None = None,
    ) -> BrokerCancelResult:
        ...

    def get_order_status(self, broker_order_id: str) -> BrokerOrderStatus:
        ...

    def list_open_orders(self) -> list[BrokerOrderStatus]:
        ...

    def list_positions(self) -> list[BrokerPosition]:
        ...

    def get_account_snapshot(self) -> BrokerAccountSnapshot:
        ...
