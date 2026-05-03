"""Position lifecycle helpers."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import uuid4

from algo_trader_unified.core.validation import validate_numeric_field


REQUIRED_FILLED_INTENT_FIELDS = (
    "source_signal_event_id",
    "order_intent_created_event_id",
    "order_submitted_event_id",
    "order_confirmed_event_id",
    "fill_confirmed_event_id",
    "simulated_order_id",
    "fill_id",
    "fill_price",
    "fill_quantity",
    "order_ref",
)

REQUIRED_FILLED_CLOSE_INTENT_FIELDS = (
    "close_intent_id",
    "position_id",
    "strategy_id",
    "sleeve_id",
    "symbol",
    "execution_mode",
    "close_intent_created_event_id",
    "close_order_submitted_event_id",
    "close_order_confirmed_event_id",
    "close_fill_confirmed_event_id",
    "position_opened_event_id",
    "source_signal_event_id",
    "fill_confirmed_event_id",
    "simulated_close_order_id",
    "close_fill_id",
    "close_fill_price",
    "close_fill_quantity",
    "quantity",
    "entry_price",
)


def _require_filled_intent(state_store, intent_id: str) -> dict[str, Any]:
    intent = state_store.get_order_intent(intent_id)
    if intent is None:
        raise KeyError(f"order intent {intent_id!r} does not exist")
    if intent.get("status") != "filled":
        raise ValueError(
            f"order intent {intent_id!r} status is {intent.get('status')!r}, not 'filled'"
        )
    return intent


def _require_string(intent: dict[str, Any], field: str, intent_id: str) -> str:
    value = intent.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"order intent {intent_id!r} is missing {field}")
    return value


def _context_or_empty(intent: dict[str, Any], field: str, intent_id: str) -> dict[str, Any]:
    if field not in intent or intent.get(field) is None:
        return {}
    value = intent.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"order intent {intent_id!r} {field} must be a dict")
    return value


def _position_id_for_intent(intent: dict[str, Any]) -> str:
    return f"{intent['strategy_id']}:{intent['intent_id']}:open"


def _require_filled_close_intent(state_store, close_intent_id: str) -> dict[str, Any]:
    close_intent = state_store.get_close_intent(close_intent_id)
    if close_intent is None:
        raise KeyError(f"close intent {close_intent_id!r} does not exist")
    if close_intent.get("status") != "filled":
        raise ValueError(
            f"close intent {close_intent_id!r} status is {close_intent.get('status')!r}, not 'filled'"
        )
    return close_intent


def _require_close_string(close_intent: dict[str, Any], field: str) -> str:
    close_intent_id = close_intent.get("close_intent_id", "<unknown>")
    value = close_intent.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"close intent {close_intent_id!r} is missing {field}")
    return value


def _validate_filled_close_intent(
    close_intent: dict[str, Any],
) -> tuple[int | float, int | float, int | float, int | float]:
    close_intent_id = close_intent.get("close_intent_id", "<unknown>")
    if "dry_run" not in close_intent:
        raise ValueError(
            f"close intent {close_intent_id!r} is missing close_intent.dry_run; Phase 3P requires dry_run=True"
        )
    if close_intent["dry_run"] is not True:
        raise ValueError(
            f"close intent {close_intent_id!r} has dry_run={close_intent['dry_run']!r}; Phase 3P requires dry_run=True"
        )
    for field in REQUIRED_FILLED_CLOSE_INTENT_FIELDS:
        if field in {"entry_price", "quantity", "close_fill_price", "close_fill_quantity"}:
            continue
        _require_close_string(close_intent, field)
    entry_price = validate_numeric_field(
        "entry_price",
        close_intent.get("entry_price"),
        minimum=0,
        allow_equal=True,
        allow_int=True,
    )
    quantity = validate_numeric_field(
        "quantity",
        close_intent.get("quantity"),
        minimum=0,
        allow_equal=False,
        allow_int=True,
    )
    close_fill_price = validate_numeric_field(
        "close_fill_price",
        close_intent.get("close_fill_price"),
        minimum=0,
        allow_equal=True,
        allow_int=True,
    )
    close_fill_quantity = validate_numeric_field(
        "close_fill_quantity",
        close_intent.get("close_fill_quantity"),
        minimum=0,
        allow_equal=False,
        allow_int=True,
    )
    return entry_price, quantity, close_fill_price, close_fill_quantity


def _validate_close_position(
    state_store,
    close_intent: dict[str, Any],
    *,
    entry_price: int | float,
    quantity: int | float,
    close_fill_quantity: int | float,
) -> dict[str, Any]:
    close_intent_id = close_intent["close_intent_id"]
    position_id = close_intent["position_id"]
    position = state_store.get_position(position_id)
    if position is None:
        raise KeyError(f"position {position_id!r} does not exist")
    if position.get("status") != "open":
        raise ValueError(
            f"position {position_id!r} status is {position.get('status')!r}, not 'open'"
        )
    if position.get("active_close_intent_id") != close_intent_id:
        raise ValueError(
            f"position {position_id!r} active_close_intent_id is {position.get('active_close_intent_id')!r}, not {close_intent_id!r}"
        )
    for field in ("strategy_id", "symbol"):
        if position.get(field) != close_intent.get(field):
            raise ValueError(
                f"position {position_id!r} {field} is {position.get(field)!r}, not close intent {close_intent.get(field)!r}"
            )
    position_quantity = validate_numeric_field(
        "quantity",
        position.get("quantity"),
        minimum=0,
        allow_equal=False,
        allow_int=True,
    )
    position_entry_price = validate_numeric_field(
        "entry_price",
        position.get("entry_price"),
        minimum=0,
        allow_equal=True,
        allow_int=True,
    )
    if position_quantity != close_fill_quantity:
        raise ValueError(
            f"position {position_id!r} quantity {position_quantity!r} does not equal close_fill_quantity {close_fill_quantity!r}"
        )
    if position_quantity != quantity:
        raise ValueError(
            f"position {position_id!r} quantity {position_quantity!r} does not equal close intent quantity {quantity!r}"
        )
    if position_entry_price != entry_price:
        raise ValueError(
            f"position {position_id!r} entry_price {position_entry_price!r} does not equal close intent entry_price {entry_price!r}"
        )
    for field in (
        "opened_at",
        "position_opened_event_id",
        "source_signal_event_id",
        "fill_confirmed_event_id",
        "simulated_order_id",
        "fill_id",
    ):
        value = position.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"position {position_id!r} is missing {field}")
    return position


def _placeholder_realized_pnl(
    *,
    entry_price: int | float,
    close_fill_price: int | float,
    quantity: int | float,
) -> int | float:
    """Dry-run placeholder only; this is not strategy-realistic P&L."""
    return (close_fill_price - entry_price) * quantity


def _numeric_not_decimal(value: Any) -> Any:
    return float(value) if isinstance(value, Decimal) else value


def open_position_from_filled_intent(
    *,
    state_store,
    ledger,
    intent_id: str,
    opened_at: str,
) -> dict[str, Any]:
    intent = _require_filled_intent(state_store, intent_id)
    strategy_id = _require_string(intent, "strategy_id", intent_id)
    with state_store.get_strategy_lock(strategy_id):
        intent = _require_filled_intent(state_store, intent_id)
        if intent.get("dry_run") is not True:
            if "dry_run" not in intent:
                raise ValueError(
                    f"order intent {intent_id!r} is missing intent.dry_run; Phase 3K requires dry_run=True"
                )
            raise ValueError(
                f"order intent {intent_id!r} has dry_run={intent.get('dry_run')!r}; Phase 3K requires dry_run=True"
            )
        for field in REQUIRED_FILLED_INTENT_FIELDS:
            _require_string(intent, field, intent_id) if field not in {
                "fill_price",
                "fill_quantity",
            } else None

        entry_price = validate_numeric_field(
            "fill_price",
            intent.get("fill_price"),
            minimum=0,
            allow_equal=True,
            allow_int=False,
        )
        quantity = validate_numeric_field(
            "fill_quantity",
            intent.get("fill_quantity"),
            minimum=0,
            allow_equal=False,
            allow_int=True,
        )
        if state_store.get_open_position(strategy_id) is not None:
            raise ValueError(f"open position already exists for strategy_id={strategy_id!r}")

        position_id = _position_id_for_intent(intent)
        sizing_context = _context_or_empty(intent, "sizing_context", intent_id)
        risk_context = _context_or_empty(intent, "risk_context", intent_id)
        position_payload = {
            "position_id": position_id,
            "intent_id": intent_id,
            "strategy_id": strategy_id,
            "sleeve_id": _require_string(intent, "sleeve_id", intent_id),
            "symbol": _require_string(intent, "symbol", intent_id),
            "status": "open",
            "execution_mode": _require_string(intent, "execution_mode", intent_id),
            "dry_run": True,
            "opened_at": opened_at,
            "updated_at": opened_at,
            "source_signal_event_id": intent["source_signal_event_id"],
            "order_intent_created_event_id": intent["order_intent_created_event_id"],
            "order_submitted_event_id": intent["order_submitted_event_id"],
            "order_confirmed_event_id": intent["order_confirmed_event_id"],
            "fill_confirmed_event_id": intent["fill_confirmed_event_id"],
            "order_ref": intent["order_ref"],
            "simulated_order_id": intent["simulated_order_id"],
            "fill_id": intent["fill_id"],
            "entry_price": entry_price,
            "quantity": quantity,
            "action": "open",
            "sizing_context": sizing_context,
            "risk_context": risk_context,
            "event_detail": "POSITION_OPENED",
        }
        event_id = str(
            ledger.append(
                event_type="POSITION_OPENED",
                strategy_id=strategy_id,
                execution_mode=position_payload["execution_mode"],
                source_module="core.positions",
                position_id=position_id,
                opportunity_id=None,
                payload=position_payload,
            )
        )
        position_record = dict(position_payload)
        position_record["position_opened_event_id"] = event_id
        position_record.pop("event_detail", None)
        created = state_store.create_open_position(position_record)
        state_store.mark_intent_position_opened(
            intent_id,
            position_id=position_id,
            position_opened_event_id=event_id,
            opened_at=opened_at,
        )
        return created


def close_position_from_filled_intent(
    *,
    state_store,
    ledger,
    close_intent_id: str,
    closed_at: str,
) -> dict[str, Any]:
    close_intent = _require_filled_close_intent(state_store, close_intent_id)
    strategy_id = _require_close_string(close_intent, "strategy_id")
    with state_store.get_strategy_lock(strategy_id):
        close_intent = _require_filled_close_intent(state_store, close_intent_id)
        entry_price, quantity, close_fill_price, close_fill_quantity = (
            _validate_filled_close_intent(close_intent)
        )
        position = _validate_close_position(
            state_store,
            close_intent,
            entry_price=entry_price,
            quantity=quantity,
            close_fill_quantity=close_fill_quantity,
        )
        realized_pnl = _placeholder_realized_pnl(
            entry_price=entry_price,
            close_fill_price=close_fill_price,
            quantity=quantity,
        )
        position_closed_event_id = f"evt_{uuid4().hex}"
        position_id = close_intent["position_id"]
        payload = {
            "position_id": position_id,
            "close_intent_id": close_intent_id,
            "strategy_id": close_intent["strategy_id"],
            "sleeve_id": close_intent["sleeve_id"],
            "symbol": close_intent["symbol"],
            "execution_mode": close_intent["execution_mode"],
            "dry_run": True,
            "opened_at": position["opened_at"],
            "closed_at": closed_at,
            "position_opened_event_id": position["position_opened_event_id"],
            "position_closed_event_id": position_closed_event_id,
            "close_intent_created_event_id": close_intent[
                "close_intent_created_event_id"
            ],
            "close_order_submitted_event_id": close_intent[
                "close_order_submitted_event_id"
            ],
            "close_order_confirmed_event_id": close_intent[
                "close_order_confirmed_event_id"
            ],
            "close_fill_confirmed_event_id": close_intent[
                "close_fill_confirmed_event_id"
            ],
            "source_signal_event_id": close_intent["source_signal_event_id"],
            "fill_confirmed_event_id": close_intent["fill_confirmed_event_id"],
            "simulated_order_id": position["simulated_order_id"],
            "simulated_close_order_id": close_intent["simulated_close_order_id"],
            "fill_id": position["fill_id"],
            "close_fill_id": close_intent["close_fill_id"],
            "entry_price": _numeric_not_decimal(entry_price),
            "quantity": _numeric_not_decimal(quantity),
            "close_fill_price": _numeric_not_decimal(close_fill_price),
            "close_fill_quantity": _numeric_not_decimal(close_fill_quantity),
            "realized_pnl": _numeric_not_decimal(realized_pnl),
            "action": "close",
            "status": "closed",
            "event_detail": "POSITION_CLOSED",
        }
        event_id = str(
            ledger.append(
                event_type="POSITION_CLOSED",
                strategy_id=close_intent["strategy_id"],
                execution_mode=close_intent["execution_mode"],
                source_module="core.positions",
                position_id=position_id,
                opportunity_id=None,
                payload=payload,
                event_id=position_closed_event_id,
                expected_ledger="execution_ledger.jsonl",
            )
        )
        closed_position = state_store.close_position(
            position_id,
            closed_at=closed_at,
            position_closed_event_id=event_id,
            close_intent_id=close_intent_id,
            close_fill_confirmed_event_id=close_intent[
                "close_fill_confirmed_event_id"
            ],
            close_fill_price=close_fill_price,
            close_fill_quantity=close_fill_quantity,
            realized_pnl=realized_pnl,
            dry_run=True,
        )
        state_store.mark_close_intent_position_closed(
            close_intent_id,
            position_closed_event_id=event_id,
            closed_at=closed_at,
        )
        return closed_position
