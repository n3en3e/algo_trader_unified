"""Dry-run scheduler job for position transitions from already-filled intents."""

from __future__ import annotations

from typing import Any

from algo_trader_unified.core.positions import (
    close_position_from_filled_intent,
    open_position_from_filled_intent,
)


def _sort_key(record: dict[str, Any], id_field: str) -> tuple[str, str]:
    filled_at = record.get("filled_at")
    record_id = record.get(id_field)
    return (
        filled_at if isinstance(filled_at, str) else "",
        record_id if isinstance(record_id, str) else "",
    )


def _skip_entry(
    *,
    intent_type: str,
    id_field: str,
    record: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "intent_type": intent_type,
        id_field: record.get(id_field),
        "strategy_id": record.get("strategy_id"),
        "status": record.get("status"),
        "reason": reason,
    }


def _opened_position_entry(position: dict[str, Any]) -> dict[str, Any]:
    return {
        "position_id": position.get("position_id"),
        "intent_id": position.get("intent_id"),
        "strategy_id": position.get("strategy_id"),
        "status": position.get("status"),
        "opened_at": position.get("opened_at"),
        "position_opened_event_id": position.get("position_opened_event_id"),
    }


def _closed_position_entry(position: dict[str, Any]) -> dict[str, Any]:
    return {
        "position_id": position.get("position_id"),
        "close_intent_id": position.get("close_intent_id"),
        "strategy_id": position.get("strategy_id"),
        "status": position.get("status"),
        "closed_at": position.get("closed_at"),
        "position_closed_event_id": position.get("position_closed_event_id"),
    }


def run_position_transitions_job(
    *,
    state_store,
    ledger,
    now,
    strategy_id: str | None = None,
    include_open_intents: bool = True,
    include_close_intents: bool = True,
) -> dict:
    result = {
        "dry_run": True,
        "positions_opened_count": 0,
        "positions_closed_count": 0,
        "skipped_order_intents_count": 0,
        "skipped_close_intents_count": 0,
        "errors_count": 0,
        "positions_opened": [],
        "positions_closed": [],
        "skipped": [],
        "errors": [],
    }

    order_intents = []
    close_intents = []
    if include_open_intents:
        order_intents = sorted(
            state_store.list_order_intents(strategy_id=strategy_id),
            key=lambda intent: _sort_key(intent, "intent_id"),
        )
    if include_close_intents:
        close_intents = sorted(
            state_store.list_close_intents(strategy_id=strategy_id),
            key=lambda intent: _sort_key(intent, "close_intent_id"),
        )

    for intent in order_intents:
        intent_id = intent.get("intent_id")
        if intent.get("status") != "filled":
            result["skipped_order_intents_count"] += 1
            result["skipped"].append(
                _skip_entry(
                    intent_type="order_intent",
                    id_field="intent_id",
                    record=intent,
                    reason="status_not_filled",
                )
            )
            continue
        try:
            current = state_store.get_order_intent(intent_id)
            if current is None or current.get("status") != "filled":
                result["skipped_order_intents_count"] += 1
                result["skipped"].append(
                    _skip_entry(
                        intent_type="order_intent",
                        id_field="intent_id",
                        record=current or intent,
                        reason="status_not_filled",
                    )
                )
                continue
            position = open_position_from_filled_intent(
                state_store=state_store,
                ledger=ledger,
                intent_id=intent_id,
                opened_at=now,
            )
        except Exception as exc:
            result["errors_count"] += 1
            result["errors"].append(
                {
                    "intent_type": "order_intent",
                    "intent_id": intent_id,
                    "error": str(exc),
                }
            )
            continue
        result["positions_opened_count"] += 1
        result["positions_opened"].append(_opened_position_entry(position))

    for close_intent in close_intents:
        close_intent_id = close_intent.get("close_intent_id")
        if close_intent.get("status") != "filled":
            result["skipped_close_intents_count"] += 1
            result["skipped"].append(
                _skip_entry(
                    intent_type="close_intent",
                    id_field="close_intent_id",
                    record=close_intent,
                    reason="status_not_filled",
                )
            )
            continue
        try:
            current = state_store.get_close_intent(close_intent_id)
            if current is None or current.get("status") != "filled":
                result["skipped_close_intents_count"] += 1
                result["skipped"].append(
                    _skip_entry(
                        intent_type="close_intent",
                        id_field="close_intent_id",
                        record=current or close_intent,
                        reason="status_not_filled",
                    )
                )
                continue
            position = close_position_from_filled_intent(
                state_store=state_store,
                ledger=ledger,
                close_intent_id=close_intent_id,
                closed_at=now,
            )
        except Exception as exc:
            result["errors_count"] += 1
            result["errors"].append(
                {
                    "intent_type": "close_intent",
                    "close_intent_id": close_intent_id,
                    "error": str(exc),
                }
            )
            continue
        result["positions_closed_count"] += 1
        result["positions_closed"].append(_closed_position_entry(position))

    return result
