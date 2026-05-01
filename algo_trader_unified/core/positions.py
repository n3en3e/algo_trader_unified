"""Position lifecycle helpers."""

from __future__ import annotations

from typing import Any

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
