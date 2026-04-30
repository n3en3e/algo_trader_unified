"""Strategy-agnostic order-intent lifecycle helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


NEW_YORK = ZoneInfo("America/New_York")


class OrderIntentTimestampError(ValueError):
    """Raised when an order intent timestamp cannot be interpreted."""


def _parse_created_at(intent: dict[str, Any]) -> datetime:
    created_at = intent.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise OrderIntentTimestampError("order intent created_at is required")
    try:
        parsed = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise OrderIntentTimestampError(
            f"order intent created_at is invalid: {created_at!r}"
        ) from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=NEW_YORK)
    return parsed


def order_intent_age_minutes(intent: dict[str, Any], *, now: datetime) -> float:
    if now.tzinfo is None:
        now = now.replace(tzinfo=NEW_YORK)
    created_at = _parse_created_at(intent)
    return (now.astimezone(created_at.tzinfo) - created_at).total_seconds() / 60.0


def is_order_intent_stale(
    intent: dict[str, Any],
    *,
    now: datetime,
    ttl_minutes: int,
) -> bool:
    return order_intent_age_minutes(intent, now=now) > float(ttl_minutes)


def _require_created_intent(state_store, intent_id: str) -> dict[str, Any]:
    intent = state_store.get_order_intent(intent_id)
    if intent is None:
        raise KeyError(f"order intent {intent_id!r} does not exist")
    if intent.get("status") != "created":
        raise ValueError(
            f"order intent {intent_id!r} status is {intent.get('status')!r}, not 'created'"
        )
    return intent


def _require_submitted_intent(state_store, intent_id: str) -> dict[str, Any]:
    intent = state_store.get_order_intent(intent_id)
    if intent is None:
        raise KeyError(f"order intent {intent_id!r} does not exist")
    if intent.get("status") != "submitted":
        raise ValueError(
            f"order intent {intent_id!r} status is {intent.get('status')!r}, not 'submitted'"
        )
    return intent


def expire_order_intent(
    *,
    state_store,
    ledger,
    intent_id: str,
    expired_at: str,
    expire_reason: str,
) -> dict[str, Any]:
    intent = _require_created_intent(state_store, intent_id)
    age_minutes = order_intent_age_minutes(
        intent,
        now=datetime.fromisoformat(expired_at),
    )
    event_id = str(
        ledger.append(
            event_type="ORDER_INTENT_EXPIRED",
            strategy_id=intent["strategy_id"],
            execution_mode=intent["execution_mode"],
            source_module="core.order_intents",
            payload={
                "intent_id": intent_id,
                "strategy_id": intent["strategy_id"],
                "sleeve_id": intent.get("sleeve_id"),
                "symbol": intent.get("symbol"),
                "execution_mode": intent["execution_mode"],
                "prior_status": "created",
                "new_status": "expired",
                "expired_at": expired_at,
                "expire_reason": expire_reason,
                "created_at": intent.get("created_at"),
                "age_minutes": age_minutes,
                "event_detail": "ORDER_INTENT_EXPIRED",
            },
        )
    )
    return state_store.expire_order_intent(
        intent_id,
        expired_at=expired_at,
        expire_reason=expire_reason,
        expired_event_id=event_id,
    )


def cancel_order_intent(
    *,
    state_store,
    ledger,
    intent_id: str,
    cancelled_at: str,
    cancel_reason: str,
) -> dict[str, Any]:
    intent = _require_created_intent(state_store, intent_id)
    event_id = str(
        ledger.append(
            event_type="ORDER_INTENT_CANCELLED",
            strategy_id=intent["strategy_id"],
            execution_mode=intent["execution_mode"],
            source_module="core.order_intents",
            payload={
                "intent_id": intent_id,
                "strategy_id": intent["strategy_id"],
                "sleeve_id": intent.get("sleeve_id"),
                "symbol": intent.get("symbol"),
                "execution_mode": intent["execution_mode"],
                "prior_status": "created",
                "new_status": "cancelled",
                "cancelled_at": cancelled_at,
                "cancel_reason": cancel_reason,
                "created_at": intent.get("created_at"),
                "event_detail": "ORDER_INTENT_CANCELLED",
            },
        )
    )
    return state_store.cancel_order_intent(
        intent_id,
        cancelled_at=cancelled_at,
        cancel_reason=cancel_reason,
        cancelled_event_id=event_id,
    )


def submit_order_intent(
    *,
    state_store,
    ledger,
    execution_adapter,
    intent_id: str,
    submitted_at: str,
) -> dict[str, Any]:
    intent = _require_created_intent(state_store, intent_id)
    if "dry_run" not in intent:
        raise ValueError(
            f"order intent {intent_id!r} is missing intent.dry_run; Phase 3G requires dry_run=True"
        )
    if intent["dry_run"] is not True:
        raise ValueError(
            f"order intent {intent_id!r} has dry_run={intent['dry_run']!r}; Phase 3G requires dry_run=True"
        )

    submission = execution_adapter.submit_order_intent(
        intent,
        submitted_at=submitted_at,
    )
    event_id = str(
        ledger.append(
            event_type="ORDER_SUBMITTED",
            strategy_id=intent["strategy_id"],
            execution_mode=intent["execution_mode"],
            source_module="core.order_intents",
            payload={
                "intent_id": intent_id,
                "strategy_id": intent["strategy_id"],
                "sleeve_id": intent.get("sleeve_id"),
                "symbol": intent.get("symbol"),
                "execution_mode": intent["execution_mode"],
                "order_ref": intent.get("order_ref"),
                "source_signal_event_id": intent.get("source_signal_event_id"),
                "order_intent_created_event_id": intent.get(
                    "order_intent_created_event_id"
                ),
                "submitted_at": submitted_at,
                "dry_run": True,
                "simulated_order_id": submission["simulated_order_id"],
                "action": "open",
                "event_detail": "ORDER_SUBMITTED",
            },
        )
    )
    return state_store.submit_order_intent(
        intent_id,
        submitted_at=submitted_at,
        order_submitted_event_id=event_id,
        simulated_order_id=submission["simulated_order_id"],
        dry_run=True,
    )


def confirm_order_intent(
    *,
    state_store,
    ledger,
    execution_adapter,
    intent_id: str,
    confirmed_at: str,
) -> dict[str, Any]:
    intent = _require_submitted_intent(state_store, intent_id)
    if "dry_run" not in intent:
        raise ValueError(
            f"order intent {intent_id!r} is missing intent.dry_run; Phase 3I requires dry_run=True"
        )
    if intent["dry_run"] is not True:
        raise ValueError(
            f"order intent {intent_id!r} has dry_run={intent['dry_run']!r}; Phase 3I requires dry_run=True"
        )
    simulated_order_id = intent.get("simulated_order_id")
    if not isinstance(simulated_order_id, str) or not simulated_order_id:
        raise ValueError(
            f"order intent {intent_id!r} is missing simulated_order_id; Phase 3I requires simulated_order_id"
        )

    status = execution_adapter.check_order_status(
        simulated_order_id=simulated_order_id,
        intent=intent,
        checked_at=confirmed_at,
    )
    if status.get("status") != "confirmed":
        raise ValueError(
            f"order intent {intent_id!r} dry-run status is {status.get('status')!r}, not 'confirmed'"
        )
    event_id = str(
        ledger.append(
            event_type="ORDER_CONFIRMED",
            strategy_id=intent["strategy_id"],
            execution_mode=intent["execution_mode"],
            source_module="core.order_intents",
            payload={
                "intent_id": intent_id,
                "strategy_id": intent["strategy_id"],
                "sleeve_id": intent.get("sleeve_id"),
                "symbol": intent.get("symbol"),
                "execution_mode": intent["execution_mode"],
                "order_ref": intent.get("order_ref"),
                "source_signal_event_id": intent.get("source_signal_event_id"),
                "order_intent_created_event_id": intent.get(
                    "order_intent_created_event_id"
                ),
                "order_submitted_event_id": intent.get("order_submitted_event_id"),
                "confirmed_at": status["confirmed_at"],
                "checked_at": status["checked_at"],
                "dry_run": True,
                "simulated_order_id": simulated_order_id,
                "action": "open",
                "event_detail": "ORDER_CONFIRMED",
            },
        )
    )
    return state_store.confirm_order_intent(
        intent_id,
        confirmed_at=status["confirmed_at"],
        order_confirmed_event_id=event_id,
        simulated_order_id=simulated_order_id,
        dry_run=True,
    )
