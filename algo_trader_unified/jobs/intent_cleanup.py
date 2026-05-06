"""Intent-level dry-run cleanup jobs for Stage 4B scheduling."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from algo_trader_unified.config.risk import ORDER_INTENT_TTL_MINUTES
from algo_trader_unified.core.order_intents import (
    cancel_order_intent,
    expire_order_intent,
    is_order_intent_stale,
)


def _sort_key(record: dict[str, Any], id_field: str) -> tuple[str, str]:
    created_at = record.get("created_at")
    record_id = record.get(id_field)
    return (
        created_at if isinstance(created_at, str) else "",
        record_id if isinstance(record_id, str) else "",
    )


def _skip_entry(*, record: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "intent_id": record.get("intent_id"),
        "strategy_id": record.get("strategy_id"),
        "status": record.get("status"),
        "reason": reason,
    }


def run_intent_expiry_job(
    *,
    state_store,
    ledger,
    now: str,
    ttl_minutes: int = ORDER_INTENT_TTL_MINUTES,
) -> dict[str, Any]:
    current_time = datetime.fromisoformat(now)
    result: dict[str, Any] = {
        "dry_run": True,
        "expired_order_intents_count": 0,
        "skipped_order_intents_count": 0,
        "errors_count": 0,
        "expired_order_intents": [],
        "skipped": [],
        "errors": [],
    }

    order_intents = sorted(
        state_store.list_order_intents(),
        key=lambda intent: _sort_key(intent, "intent_id"),
    )
    for intent in order_intents:
        intent_id = intent.get("intent_id")
        if intent.get("status") != "created":
            result["skipped_order_intents_count"] += 1
            result["skipped"].append(
                _skip_entry(record=intent, reason="status_not_created")
            )
            continue
        try:
            current = state_store.get_order_intent(intent_id)
            if current is None or current.get("status") != "created":
                result["skipped_order_intents_count"] += 1
                result["skipped"].append(
                    _skip_entry(record=current or intent, reason="status_not_created")
                )
                continue
            if not is_order_intent_stale(
                current,
                now=current_time,
                ttl_minutes=ttl_minutes,
            ):
                result["skipped_order_intents_count"] += 1
                result["skipped"].append(
                    _skip_entry(record=current, reason="ttl_not_expired")
                )
                continue
            expired = expire_order_intent(
                state_store=state_store,
                ledger=ledger,
                intent_id=intent_id,
                expired_at=now,
                expire_reason="ttl_expired",
            )
        except Exception as exc:
            result["errors_count"] += 1
            result["errors"].append({"intent_id": intent_id, "error": str(exc)})
            continue
        result["expired_order_intents_count"] += 1
        result["expired_order_intents"].append(
            {
                "intent_id": expired.get("intent_id"),
                "strategy_id": expired.get("strategy_id"),
                "status": expired.get("status"),
                "expired_at": expired.get("expired_at"),
            }
        )
    return result


def run_eod_intent_cleanup_job(
    *,
    state_store,
    ledger,
    now: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dry_run": True,
        "cancelled_order_intents_count": 0,
        "skipped_order_intents_count": 0,
        "errors_count": 0,
        "cancelled_order_intents": [],
        "skipped": [],
        "errors": [],
    }

    order_intents = sorted(
        state_store.list_order_intents(),
        key=lambda intent: _sort_key(intent, "intent_id"),
    )
    for intent in order_intents:
        intent_id = intent.get("intent_id")
        if intent.get("status") != "created":
            result["skipped_order_intents_count"] += 1
            result["skipped"].append(
                _skip_entry(record=intent, reason="status_not_created")
            )
            continue
        try:
            current = state_store.get_order_intent(intent_id)
            if current is None or current.get("status") != "created":
                result["skipped_order_intents_count"] += 1
                result["skipped"].append(
                    _skip_entry(record=current or intent, reason="status_not_created")
                )
                continue
            cancelled = cancel_order_intent(
                state_store=state_store,
                ledger=ledger,
                intent_id=intent_id,
                cancelled_at=now,
                cancel_reason="eod_intent_cleanup",
            )
        except Exception as exc:
            result["errors_count"] += 1
            result["errors"].append({"intent_id": intent_id, "error": str(exc)})
            continue
        result["cancelled_order_intents_count"] += 1
        result["cancelled_order_intents"].append(
            {
                "intent_id": cancelled.get("intent_id"),
                "strategy_id": cancelled.get("strategy_id"),
                "status": cancelled.get("status"),
                "cancelled_at": cancelled.get("cancelled_at"),
            }
        )
    return result
