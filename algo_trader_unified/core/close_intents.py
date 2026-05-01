"""Strategy-agnostic close-intent lifecycle helpers."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from algo_trader_unified.core.validation import validate_numeric_field


REQUIRED_OPEN_POSITION_FIELDS = (
    "position_id",
    "strategy_id",
    "sleeve_id",
    "symbol",
    "execution_mode",
    "position_opened_event_id",
    "source_signal_event_id",
    "fill_confirmed_event_id",
    "quantity",
    "entry_price",
)


def _require_open_position(state_store, position_id: str) -> dict[str, Any]:
    position = state_store.get_position(position_id)
    if position is None:
        raise KeyError(f"position {position_id!r} does not exist")
    if position.get("status") != "open":
        raise ValueError(
            f"position {position_id!r} status is {position.get('status')!r}, not 'open'"
        )
    return position


def _require_string(position: dict[str, Any], field: str, position_id: str) -> str:
    value = position.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"position {position_id!r} is missing {field}")
    return value


def _validate_open_position(position: dict[str, Any], position_id: str) -> tuple[int | float, int | float]:
    if position.get("dry_run") is not True:
        if "dry_run" not in position:
            raise ValueError(
                f"position {position_id!r} is missing position.dry_run; Phase 3N requires dry_run=True"
            )
        raise ValueError(
            f"position {position_id!r} has dry_run={position.get('dry_run')!r}; Phase 3N requires dry_run=True"
        )
    for field in REQUIRED_OPEN_POSITION_FIELDS:
        if field in {"quantity", "entry_price"}:
            continue
        _require_string(position, field, position_id)
    quantity = validate_numeric_field(
        "quantity",
        position.get("quantity"),
        minimum=0,
        allow_equal=False,
        allow_int=True,
    )
    entry_price = validate_numeric_field(
        "entry_price",
        position.get("entry_price"),
        minimum=0,
        allow_equal=True,
        allow_int=True,
    )
    return quantity, entry_price


def _close_intent_id_for_position(position: dict[str, Any]) -> str:
    return f"{position['strategy_id']}:{position['position_id']}:close"


def create_close_intent_from_position(
    *,
    state_store,
    ledger,
    position_id: str,
    created_at: str,
    close_reason: str = "manual",
    requested_by: str = "operator",
) -> dict[str, Any]:
    position = _require_open_position(state_store, position_id)
    strategy_id = _require_string(position, "strategy_id", position_id)
    with state_store.get_strategy_lock(strategy_id):
        position = _require_open_position(state_store, position_id)
        quantity, entry_price = _validate_open_position(position, position_id)
        if state_store.get_active_close_intent(position_id) is not None:
            raise ValueError(f"active close intent already exists for position_id={position_id!r}")

        close_intent_id = _close_intent_id_for_position(position)
        event_id = f"evt_{uuid4().hex}"
        close_payload = {
            "close_intent_id": close_intent_id,
            "position_id": position_id,
            "strategy_id": strategy_id,
            "sleeve_id": position["sleeve_id"],
            "symbol": position["symbol"],
            "execution_mode": position["execution_mode"],
            "dry_run": True,
            "created_at": created_at,
            "close_reason": close_reason,
            "requested_by": requested_by,
            "position_opened_event_id": position["position_opened_event_id"],
            "source_signal_event_id": position["source_signal_event_id"],
            "fill_confirmed_event_id": position["fill_confirmed_event_id"],
            "close_intent_created_event_id": event_id,
            "quantity": quantity,
            "entry_price": entry_price,
            "action": "close",
            "status": "created",
            "event_detail": "CLOSE_INTENT_CREATED",
        }
        ledger.append(
            event_type="CLOSE_INTENT_CREATED",
            strategy_id=strategy_id,
            execution_mode=close_payload["execution_mode"],
            source_module="core.close_intents",
            position_id=position_id,
            opportunity_id=None,
            payload=close_payload,
            event_id=event_id,
            expected_ledger="order_ledger.jsonl",
        )
        close_record = dict(close_payload)
        close_record["updated_at"] = created_at
        close_record.pop("event_detail", None)
        created = state_store.create_close_intent(close_record)
        state_store.mark_position_close_intent_created(
            position_id,
            close_intent_id=close_intent_id,
            close_intent_created_event_id=event_id,
            close_intent_created_at=created_at,
        )
        return created
