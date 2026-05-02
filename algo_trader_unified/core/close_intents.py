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

REQUIRED_CLOSE_INTENT_FIELDS = (
    "close_intent_id",
    "position_id",
    "strategy_id",
    "sleeve_id",
    "symbol",
    "execution_mode",
    "close_intent_created_event_id",
    "position_opened_event_id",
    "fill_confirmed_event_id",
    "quantity",
    "entry_price",
    "close_reason",
    "requested_by",
)

CLOSE_SUBMIT_EVENT_TYPE = "CLOSE_" + "ORDER_" + "SUBMITTED"
CLOSE_CONFIRM_EVENT_TYPE = "CLOSE_" + "ORDER_" + "CONFIRMED"

REQUIRED_SUBMITTED_CLOSE_INTENT_FIELDS = REQUIRED_CLOSE_INTENT_FIELDS + (
    "close_order_submitted_event_id",
    "close_order_ref",
    "simulated_close_order_id",
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


def _require_created_close_intent(state_store, close_intent_id: str) -> dict[str, Any]:
    close_intent = state_store.get_close_intent(close_intent_id)
    if close_intent is None:
        raise KeyError(f"close intent {close_intent_id!r} does not exist")
    if close_intent.get("status") != "created":
        raise ValueError(
            f"close intent {close_intent_id!r} status is {close_intent.get('status')!r}, not 'created'"
        )
    return close_intent


def _require_submitted_close_intent(state_store, close_intent_id: str) -> dict[str, Any]:
    close_intent = state_store.get_close_intent(close_intent_id)
    if close_intent is None:
        raise KeyError(f"close intent {close_intent_id!r} does not exist")
    if close_intent.get("status") != "submitted":
        raise ValueError(
            f"close intent {close_intent_id!r} status is {close_intent.get('status')!r}, not 'submitted'"
        )
    return close_intent


def _require_close_intent_string(close_intent: dict[str, Any], field: str) -> str:
    close_intent_id = close_intent.get("close_intent_id", "<unknown>")
    value = close_intent.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"close intent {close_intent_id!r} is missing {field}")
    return value


def _validate_close_intent(close_intent: dict[str, Any]) -> tuple[int | float, int | float]:
    close_intent_id = close_intent.get("close_intent_id", "<unknown>")
    if "dry_run" not in close_intent:
        raise ValueError(
            f"close intent {close_intent_id!r} is missing close_intent.dry_run; Phase 3O-1 requires dry_run=True"
        )
    if close_intent["dry_run"] is not True:
        raise ValueError(
            f"close intent {close_intent_id!r} has dry_run={close_intent['dry_run']!r}; Phase 3O-1 requires dry_run=True"
        )
    for field in REQUIRED_CLOSE_INTENT_FIELDS:
        if field in {"quantity", "entry_price"}:
            continue
        _require_close_intent_string(close_intent, field)
    quantity = validate_numeric_field(
        "quantity",
        close_intent.get("quantity"),
        minimum=0,
        allow_equal=False,
        allow_int=True,
    )
    entry_price = validate_numeric_field(
        "entry_price",
        close_intent.get("entry_price"),
        minimum=0,
        allow_equal=True,
        allow_int=True,
    )
    return quantity, entry_price


def _validate_submitted_close_intent(close_intent: dict[str, Any]) -> tuple[int | float, int | float]:
    close_intent_id = close_intent.get("close_intent_id", "<unknown>")
    if "dry_run" not in close_intent:
        raise ValueError(
            f"close intent {close_intent_id!r} is missing close_intent.dry_run; Phase 3O-2 requires dry_run=True"
        )
    if close_intent["dry_run"] is not True:
        raise ValueError(
            f"close intent {close_intent_id!r} has dry_run={close_intent['dry_run']!r}; Phase 3O-2 requires dry_run=True"
        )
    for field in REQUIRED_SUBMITTED_CLOSE_INTENT_FIELDS:
        if field in {"quantity", "entry_price"}:
            continue
        _require_close_intent_string(close_intent, field)
    quantity = validate_numeric_field(
        "quantity",
        close_intent.get("quantity"),
        minimum=0,
        allow_equal=False,
        allow_int=True,
    )
    entry_price = validate_numeric_field(
        "entry_price",
        close_intent.get("entry_price"),
        minimum=0,
        allow_equal=True,
        allow_int=True,
    )
    return quantity, entry_price


def _validate_associated_position(state_store, close_intent: dict[str, Any]) -> dict[str, Any]:
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
    return position


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


def submit_close_intent(
    *,
    state_store,
    ledger,
    execution_adapter,
    close_intent_id: str,
    submitted_at: str,
) -> dict[str, Any]:
    close_intent = _require_created_close_intent(state_store, close_intent_id)
    strategy_id = _require_close_intent_string(close_intent, "strategy_id")
    with state_store.get_strategy_lock(strategy_id):
        close_intent = _require_created_close_intent(state_store, close_intent_id)
        quantity, entry_price = _validate_close_intent(close_intent)
        _validate_associated_position(state_store, close_intent)

        submission = execution_adapter.submit_close_intent(
            close_intent,
            submitted_at=submitted_at,
        )
        if submission.get("status") != "submitted":
            raise ValueError(
                f"close intent {close_intent_id!r} dry-run status is {submission.get('status')!r}, not 'submitted'"
            )

        event_id = str(
            ledger.append(
                event_type=CLOSE_SUBMIT_EVENT_TYPE,
                strategy_id=close_intent["strategy_id"],
                execution_mode=close_intent["execution_mode"],
                source_module="core.close_intents",
                position_id=close_intent["position_id"],
                opportunity_id=None,
                payload={
                    "close_intent_id": close_intent_id,
                    "position_id": close_intent["position_id"],
                    "strategy_id": close_intent["strategy_id"],
                    "sleeve_id": close_intent["sleeve_id"],
                    "symbol": close_intent["symbol"],
                    "execution_mode": close_intent["execution_mode"],
                    "dry_run": True,
                    "close_order_ref": submission["close_order_ref"],
                    "simulated_close_order_id": submission["simulated_close_order_id"],
                    "submitted_at": submission["submitted_at"],
                    "close_reason": close_intent["close_reason"],
                    "requested_by": close_intent["requested_by"],
                    "close_intent_created_event_id": close_intent[
                        "close_intent_created_event_id"
                    ],
                    "position_opened_event_id": close_intent["position_opened_event_id"],
                    "fill_confirmed_event_id": close_intent["fill_confirmed_event_id"],
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "action": "close",
                    "status": "submitted",
                    "event_detail": CLOSE_SUBMIT_EVENT_TYPE,
                },
                expected_ledger="order_ledger.jsonl",
            )
        )
        return state_store.submit_close_intent(
            close_intent_id,
            submitted_at=submitted_at,
            close_order_submitted_event_id=event_id,
            simulated_close_order_id=submission["simulated_close_order_id"],
            close_order_ref=submission["close_order_ref"],
            dry_run=True,
        )


def confirm_close_order(
    *,
    state_store,
    ledger,
    execution_adapter,
    close_intent_id: str,
    confirmed_at: str,
) -> dict[str, Any]:
    close_intent = _require_submitted_close_intent(state_store, close_intent_id)
    strategy_id = _require_close_intent_string(close_intent, "strategy_id")
    with state_store.get_strategy_lock(strategy_id):
        close_intent = _require_submitted_close_intent(state_store, close_intent_id)
        quantity, entry_price = _validate_submitted_close_intent(close_intent)
        _validate_associated_position(state_store, close_intent)

        status = execution_adapter.check_close_order_status(
            close_intent=close_intent,
            simulated_close_order_id=close_intent["simulated_close_order_id"],
            checked_at=confirmed_at,
        )
        if status.get("status") != "confirmed":
            raise ValueError(
                f"close intent {close_intent_id!r} dry-run status is {status.get('status')!r}, not 'confirmed'"
            )
        if status.get("simulated_close_order_id") != close_intent["simulated_close_order_id"]:
            raise ValueError(
                f"close intent {close_intent_id!r} dry-run simulated_close_order_id mismatch"
            )

        event_id = str(
            ledger.append(
                event_type=CLOSE_CONFIRM_EVENT_TYPE,
                strategy_id=close_intent["strategy_id"],
                execution_mode=close_intent["execution_mode"],
                source_module="core.close_intents",
                position_id=close_intent["position_id"],
                opportunity_id=None,
                payload={
                    "close_intent_id": close_intent_id,
                    "position_id": close_intent["position_id"],
                    "strategy_id": close_intent["strategy_id"],
                    "sleeve_id": close_intent["sleeve_id"],
                    "symbol": close_intent["symbol"],
                    "execution_mode": close_intent["execution_mode"],
                    "dry_run": True,
                    "close_order_ref": close_intent["close_order_ref"],
                    "simulated_close_order_id": status["simulated_close_order_id"],
                    "confirmed_at": status["confirmed_at"],
                    "checked_at": status["checked_at"],
                    "close_reason": close_intent["close_reason"],
                    "requested_by": close_intent["requested_by"],
                    "close_intent_created_event_id": close_intent[
                        "close_intent_created_event_id"
                    ],
                    "close_order_submitted_event_id": close_intent[
                        "close_order_submitted_event_id"
                    ],
                    "position_opened_event_id": close_intent["position_opened_event_id"],
                    "fill_confirmed_event_id": close_intent["fill_confirmed_event_id"],
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "action": "close",
                    "status": "confirmed",
                    "event_detail": CLOSE_CONFIRM_EVENT_TYPE,
                },
                expected_ledger="order_ledger.jsonl",
            )
        )
        return state_store.confirm_close_order(
            close_intent_id,
            confirmed_at=status["confirmed_at"],
            close_order_confirmed_event_id=event_id,
            simulated_close_order_id=status["simulated_close_order_id"],
            dry_run=True,
        )
