"""Dry-run scheduler job for fill-confirming already-confirmed intents."""

from __future__ import annotations

from typing import Any

from algo_trader_unified.core.close_intents import confirm_close_fill
from algo_trader_unified.core.execution import DryRunExecutionAdapter
from algo_trader_unified.core.order_intents import confirm_fill


def _sort_key(record: dict[str, Any], id_field: str) -> tuple[str, str]:
    confirmed_at = record.get("confirmed_at")
    record_id = record.get(id_field)
    return (
        confirmed_at if isinstance(confirmed_at, str) else "",
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


def _filled_order_entry(intent: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent_id": intent.get("intent_id"),
        "strategy_id": intent.get("strategy_id"),
        "status": intent.get("status"),
        "confirmed_at": intent.get("confirmed_at"),
        "filled_at": intent.get("filled_at"),
        "simulated_order_id": intent.get("simulated_order_id"),
        "fill_id": intent.get("fill_id"),
    }


def _filled_close_entry(intent: dict[str, Any]) -> dict[str, Any]:
    return {
        "close_intent_id": intent.get("close_intent_id"),
        "position_id": intent.get("position_id"),
        "strategy_id": intent.get("strategy_id"),
        "status": intent.get("status"),
        "confirmed_at": intent.get("confirmed_at"),
        "filled_at": intent.get("filled_at"),
        "simulated_close_order_id": intent.get("simulated_close_order_id"),
        "close_fill_id": intent.get("close_fill_id"),
    }


def run_intent_fill_confirmation_job(
    *,
    state_store,
    ledger,
    now,
    execution_adapter=None,
    strategy_id: str | None = None,
    include_open_intents: bool = True,
    include_close_intents: bool = True,
) -> dict:
    adapter = execution_adapter if execution_adapter is not None else DryRunExecutionAdapter()
    result = {
        "dry_run": True,
        "filled_order_intents_count": 0,
        "filled_close_intents_count": 0,
        "skipped_order_intents_count": 0,
        "skipped_close_intents_count": 0,
        "errors_count": 0,
        "filled_order_intents": [],
        "filled_close_intents": [],
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
        if intent.get("status") != "confirmed":
            result["skipped_order_intents_count"] += 1
            result["skipped"].append(
                _skip_entry(
                    intent_type="order_intent",
                    id_field="intent_id",
                    record=intent,
                    reason="status_not_confirmed",
                )
            )
            continue
        try:
            current = state_store.get_order_intent(intent_id)
            if current is None or current.get("status") != "confirmed":
                result["skipped_order_intents_count"] += 1
                result["skipped"].append(
                    _skip_entry(
                        intent_type="order_intent",
                        id_field="intent_id",
                        record=current or intent,
                        reason="status_not_confirmed",
                    )
                )
                continue
            filled = confirm_fill(
                state_store=state_store,
                ledger=ledger,
                execution_adapter=adapter,
                intent_id=intent_id,
                filled_at=now,
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
        result["filled_order_intents_count"] += 1
        result["filled_order_intents"].append(_filled_order_entry(filled))

    for close_intent in close_intents:
        close_intent_id = close_intent.get("close_intent_id")
        if close_intent.get("status") != "confirmed":
            result["skipped_close_intents_count"] += 1
            result["skipped"].append(
                _skip_entry(
                    intent_type="close_intent",
                    id_field="close_intent_id",
                    record=close_intent,
                    reason="status_not_confirmed",
                )
            )
            continue
        try:
            current = state_store.get_close_intent(close_intent_id)
            if current is None or current.get("status") != "confirmed":
                result["skipped_close_intents_count"] += 1
                result["skipped"].append(
                    _skip_entry(
                        intent_type="close_intent",
                        id_field="close_intent_id",
                        record=current or close_intent,
                        reason="status_not_confirmed",
                    )
                )
                continue
            filled = confirm_close_fill(
                state_store=state_store,
                ledger=ledger,
                execution_adapter=adapter,
                close_intent_id=close_intent_id,
                filled_at=now,
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
        result["filled_close_intents_count"] += 1
        result["filled_close_intents"].append(_filled_close_entry(filled))

    return result
