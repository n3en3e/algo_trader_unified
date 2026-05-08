"""Stage 4D-5 IBKR paper request mapping contract.

This module builds JSON-safe IBKR-shaped request plans only. It does not import
IBKR libraries, instantiate broker clients, qualify contracts, or submit orders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from algo_trader_unified.core.paper_broker_adapter import (
    BrokerOrderRequest,
    sanitize_json_safe,
)


PAPER_TRADING_MODE = "PAPER"
IBKR_PAPER_PORT = 4004
IBKR_LIVE_PORT = 4002
SUPPORTED_ASSET_TYPES = frozenset({"OPTION", "STOCK", "FUTURE", "CASH"})
SUPPORTED_TIME_IN_FORCE = frozenset({"DAY", "GTC"})


@dataclass(frozen=True)
class IbkrPaperConfig:
    host: str
    port: int
    client_id: int
    account_id: str | None
    trading_mode: str
    readonly: bool


@dataclass(frozen=True)
class IbkrPaperOrderPlan:
    client_order_id: str
    strategy_id: str
    symbol: str
    asset_type: str
    action: str
    quantity: float | int
    order_type: str
    limit_price: float | None
    time_in_force: str
    account_id: str | None
    ibkr_contract_hint: dict[str, Any]
    ibkr_order_hint: dict[str, Any]
    dry_run: bool
    paper_only: bool
    ready_for_submission: bool
    blockers: list[str]
    warnings: list[str]


def validate_ibkr_paper_config(config: dict[str, Any]) -> IbkrPaperConfig:
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")

    trading_mode = config.get("trading_mode")
    if trading_mode != PAPER_TRADING_MODE:
        if trading_mode == "LIVE":
            raise ValueError("trading_mode LIVE is rejected for IBKR paper mapping")
        raise ValueError('trading_mode must be exactly "PAPER"')

    host = config.get("host")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")

    port = config.get("port")
    if isinstance(port, bool) or not isinstance(port, int):
        raise ValueError("port must be an int")
    if port == IBKR_LIVE_PORT:
        raise ValueError("port 4002 is rejected as likely live")
    if port != IBKR_PAPER_PORT:
        raise ValueError("port must be 4004 for IBKR paper mapping")

    client_id = config.get("client_id")
    if isinstance(client_id, bool) or not isinstance(client_id, int) or client_id <= 0:
        raise ValueError("client_id must be a positive int")

    readonly = config.get("readonly")
    if readonly is not False:
        raise ValueError("readonly must be False for paper submission planning")

    account_id = config.get("account_id")
    if account_id is not None:
        if not isinstance(account_id, str) or not account_id.strip():
            raise ValueError("account_id must be a non-empty string when provided")
        account_id = account_id.strip()

    return IbkrPaperConfig(
        host=host.strip(),
        port=port,
        client_id=client_id,
        account_id=account_id,
        trading_mode=trading_mode,
        readonly=readonly,
    )


def build_ibkr_paper_order_plan(
    request: BrokerOrderRequest,
    *,
    config: IbkrPaperConfig,
) -> IbkrPaperOrderPlan:
    blockers: list[str] = []
    warnings: list[str] = []

    action = _map_side(request.side, blockers)
    order_type = _map_order_type(request.order_type, blockers)
    quantity = _map_quantity(request.quantity, blockers)
    limit_price = _map_limit_price(request.order_type, request.limit_price, blockers)
    time_in_force = _map_time_in_force(request.time_in_force, blockers)
    asset_type = _normalize_text(request.asset_type)

    contract_hint = _build_contract_hint(request, asset_type, blockers)
    order_hint = _build_order_hint(
        action=action,
        quantity=quantity if _is_positive_number(quantity) else None,
        order_type=order_type,
        limit_price=limit_price,
        time_in_force=time_in_force,
        account_id=config.account_id,
    )

    contract_hint = _json_safe_dict(contract_hint)
    order_hint = _json_safe_dict(order_hint)

    return IbkrPaperOrderPlan(
        client_order_id=request.client_order_id,
        strategy_id=request.strategy_id,
        symbol=request.symbol,
        asset_type=asset_type,
        action=action or "",
        quantity=quantity,
        order_type=order_type or "",
        limit_price=limit_price,
        time_in_force=time_in_force or "",
        account_id=config.account_id,
        ibkr_contract_hint=contract_hint,
        ibkr_order_hint=order_hint,
        dry_run=True,
        paper_only=True,
        ready_for_submission=not blockers,
        blockers=blockers,
        warnings=warnings,
    )


def _map_side(side: str, blockers: list[str]) -> str | None:
    normalized = _normalize_text(side)
    if normalized in {"BUY", "SELL"}:
        return normalized
    blockers.append(f"unsupported side: {normalized}")
    return None


def _map_order_type(order_type: str, blockers: list[str]) -> str | None:
    normalized = _normalize_text(order_type)
    if normalized == "LIMIT":
        return "LMT"
    if normalized == "MARKET":
        return "MKT"
    blockers.append(f"unsupported order_type: {normalized}")
    return None


def _map_quantity(quantity: float | int, blockers: list[str]) -> float | int:
    if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
        blockers.append("quantity must be positive numeric")
        return 0
    if not math.isfinite(float(quantity)) or quantity <= 0:
        blockers.append("quantity must be positive numeric")
    return quantity


def _map_limit_price(
    request_order_type: str,
    request_limit_price: float | None,
    blockers: list[str],
) -> float | None:
    normalized_order_type = _normalize_text(request_order_type)
    if normalized_order_type == "LIMIT":
        if not _is_positive_number(request_limit_price):
            blockers.append("LIMIT orders require positive limit_price")
            return None
        return float(request_limit_price)
    if normalized_order_type == "MARKET" and request_limit_price is not None:
        blockers.append("MARKET orders must not include limit_price")
    return None


def _map_time_in_force(
    time_in_force: str | None,
    blockers: list[str],
) -> str | None:
    if time_in_force is None:
        return "DAY"
    normalized = _normalize_text(time_in_force)
    if normalized not in SUPPORTED_TIME_IN_FORCE:
        blockers.append("time_in_force must be one of ['DAY', 'GTC']")
        return None
    return normalized


def _build_contract_hint(
    request: BrokerOrderRequest,
    asset_type: str,
    blockers: list[str],
) -> dict[str, Any]:
    contract_hint: dict[str, Any] = {
        "symbol": request.symbol,
        "asset_type": asset_type,
    }

    if asset_type == "OPTION":
        contract_hint["secType"] = "OPT"
        _add_option_contract_hints(contract_hint, request.metadata, blockers)
    elif asset_type == "STOCK":
        contract_hint["secType"] = "STK"
    elif asset_type == "FUTURE":
        contract_hint["secType"] = "FUT"
    elif asset_type == "CASH":
        contract_hint["secType"] = "CASH"
    else:
        blockers.append(f"unsupported asset_type: {asset_type}")

    metadata = sanitize_json_safe(request.metadata)
    if isinstance(metadata, dict):
        contract_hint["metadata"] = metadata

    return contract_hint


def _add_option_contract_hints(
    contract_hint: dict[str, Any],
    metadata: dict[str, Any] | None,
    blockers: list[str],
) -> None:
    metadata = metadata if isinstance(metadata, dict) else {}
    expiry = metadata.get("expiry")
    strike = metadata.get("strike")
    right = metadata.get("right")

    if isinstance(expiry, str) and expiry.strip():
        contract_hint["expiry"] = expiry.strip()
    else:
        blockers.append("OPTION contract hint requires metadata.expiry")

    if _is_positive_number(strike):
        contract_hint["strike"] = float(strike)
    else:
        blockers.append("OPTION contract hint requires numeric metadata.strike")

    if isinstance(right, str) and right.strip():
        contract_hint["right"] = right.strip()
    else:
        blockers.append("OPTION contract hint requires metadata.right")


def _build_order_hint(
    *,
    action: str | None,
    quantity: float | int | None,
    order_type: str | None,
    limit_price: float | None,
    time_in_force: str | None,
    account_id: str | None,
) -> dict[str, Any]:
    order_hint: dict[str, Any] = {}
    if quantity is not None:
        order_hint["totalQuantity"] = quantity
    if action is not None:
        order_hint["action"] = action
    if order_type is not None:
        order_hint["orderType"] = order_type
    if time_in_force is not None:
        order_hint["tif"] = time_in_force
    if limit_price is not None:
        order_hint["lmtPrice"] = limit_price
    if account_id is not None:
        order_hint["account"] = account_id
    return order_hint


def _is_positive_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value > 0
    )


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return str(value).strip().upper()
    return value.strip().upper()


def _json_safe_dict(value: dict[str, Any]) -> dict[str, Any]:
    safe_value = sanitize_json_safe(value)
    if isinstance(safe_value, dict):
        return safe_value
    return {"value": safe_value}
