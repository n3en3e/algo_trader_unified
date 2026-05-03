"""Management signal scan helpers.

Phase 3Q only evaluates existing open positions and starts the close lifecycle
by creating a close intent when an injected management signal asks for it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from algo_trader_unified.core.close_intents import create_close_intent_from_position
from algo_trader_unified.core.state_store import ACTIVE_CLOSE_INTENT_STATUSES


@dataclass(frozen=True)
class ManagementSignalResult:
    should_close: bool
    close_reason: str | None = None
    requested_by: str = "management"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ManagementSignalProvider = Callable[..., ManagementSignalResult | dict[str, Any]]


def default_management_signal_provider(*, position: dict[str, Any], now: str) -> ManagementSignalResult:
    return ManagementSignalResult(should_close=False)


def _empty_result() -> dict[str, Any]:
    return {
        "evaluated_count": 0,
        "close_intents_created_count": 0,
        "skipped_active_close_intent_count": 0,
        "no_action_count": 0,
        "errors_count": 0,
        "created_close_intents": [],
        "skipped": [],
        "no_action": [],
        "errors": [],
    }


def _position_ref(position: dict[str, Any]) -> dict[str, Any]:
    return {
        "position_id": position.get("position_id"),
        "strategy_id": position.get("strategy_id"),
        "status": position.get("status"),
    }


def _error_entry(position: dict[str, Any], stage: str, exc: Exception) -> dict[str, Any]:
    entry = _position_ref(position)
    entry.update(
        {
            "stage": stage,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    )
    return entry


def _active_intent_from_link(state_store, position: dict[str, Any]) -> dict[str, Any] | None:
    close_intent_id = position.get("active_close_intent_id")
    if not close_intent_id:
        return None
    close_intent = state_store.get_close_intent(str(close_intent_id))
    if close_intent is None:
        return {"close_intent_id": str(close_intent_id), "status": "unknown"}
    if close_intent.get("status") in ACTIVE_CLOSE_INTENT_STATUSES:
        return close_intent
    return None


def _skip_entry(position: dict[str, Any], close_intent: dict[str, Any]) -> dict[str, Any]:
    entry = _position_ref(position)
    entry.update(
        {
            "reason": "active_close_intent",
            "close_intent_id": close_intent.get("close_intent_id"),
            "close_intent_status": close_intent.get("status"),
        }
    )
    return entry


def _normalize_signal_result(raw_result: ManagementSignalResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_result, ManagementSignalResult):
        result = raw_result.to_dict()
    elif isinstance(raw_result, dict):
        result = dict(raw_result)
    else:
        result = {
            "should_close": getattr(raw_result, "should_close", None),
            "close_reason": getattr(raw_result, "close_reason", None),
            "requested_by": getattr(raw_result, "requested_by", "management"),
            "details": getattr(raw_result, "details", {}),
        }
        if result["close_reason"] is None and hasattr(raw_result, "reason"):
            result["close_reason"] = getattr(raw_result, "reason")

    if "should_close" not in result:
        raise ValueError("management signal result is missing should_close")
    if not isinstance(result["should_close"], bool):
        raise ValueError("management signal result should_close must be bool")

    close_reason = result.get("close_reason")
    if not isinstance(close_reason, str) or not close_reason.strip():
        close_reason = "management_signal"
    else:
        close_reason = close_reason.strip()

    requested_by = result.get("requested_by")
    if not isinstance(requested_by, str) or not requested_by.strip():
        requested_by = "management"
    else:
        requested_by = requested_by.strip()

    details = result.get("details")
    if not isinstance(details, dict):
        details = {}

    return {
        "should_close": result["should_close"],
        "close_reason": close_reason,
        "requested_by": requested_by,
        "details": details,
    }


def run_management_scan(
    *,
    state_store,
    ledger,
    strategy_id: str | None = None,
    management_signal_provider: ManagementSignalProvider = default_management_signal_provider,
    now: str,
) -> dict[str, Any]:
    result = _empty_result()
    positions = state_store.list_positions(strategy_id=strategy_id, status="open")

    for position in positions:
        position_id = position.get("position_id")
        linked_active = _active_intent_from_link(state_store, position)
        active_close_intent = linked_active or state_store.get_active_close_intent(str(position_id))
        if active_close_intent is not None:
            result["skipped_active_close_intent_count"] += 1
            result["skipped"].append(_skip_entry(position, active_close_intent))
            continue

        try:
            raw_signal = management_signal_provider(position=position, now=now)
            signal = _normalize_signal_result(raw_signal)
        except Exception as exc:
            result["errors_count"] += 1
            result["errors"].append(_error_entry(position, "management_signal_provider", exc))
            continue

        result["evaluated_count"] += 1
        if not signal["should_close"]:
            result["no_action_count"] += 1
            result["no_action"].append(_position_ref(position))
            continue

        try:
            close_intent = create_close_intent_from_position(
                state_store=state_store,
                ledger=ledger,
                position_id=str(position_id),
                created_at=now,
                close_reason=signal["close_reason"],
                requested_by=signal["requested_by"],
            )
        except Exception as exc:
            result["errors_count"] += 1
            result["errors"].append(_error_entry(position, "close_intent_creation", exc))
            continue

        result["close_intents_created_count"] += 1
        result["created_close_intents"].append(close_intent)

    return result
