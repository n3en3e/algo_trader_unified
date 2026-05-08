"""Read-only Stage 4D-4 paper adapter compatibility reporting."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from algo_trader_unified.core.paper_broker_adapter import build_broker_order_request


_DO_NOT_DO_YET = [
    "Do not wire adapter into lifecycle jobs until all current intents translate cleanly.",
]
_MISSING_FIELD_REASONS = {
    "strategy_id is required": "strategy_id",
    "symbol or underlying is required": "symbol_or_underlying",
    "quantity must be positive numeric": "quantity",
    "order_type is required": "order_type",
    "intent_id is required for deterministic client_order_id": "intent_id",
    "limit_price must be positive numeric": "limit_price",
}


def build_paper_adapter_compatibility_report(
    *,
    order_intents: list[dict],
    strategy_ids: list[str],
    now_provider: Callable[[], datetime] | None = None,
) -> dict:
    """Build a pure list-in/dict-out compatibility report for dry-run intents."""

    now = _current_time(now_provider)
    errors: list[str] = []
    warnings: list[str] = []
    strategy_id_list = [str(item) for item in strategy_ids]
    per_strategy = {
        strategy_id: _empty_strategy_report() for strategy_id in strategy_id_list
    }
    fallback_strategy = "__missing_strategy_id__"
    client_order_ids: list[str] = []
    translated_occurrences: list[dict[str, Any]] = []
    invalid_reasons = Counter()
    unsupported_order_types: set[str] = set()
    unsupported_sides: set[str] = set()
    missing_required_fields = Counter()
    deterministic_client_order_ids = True

    for index, intent in enumerate(list(order_intents)):
        strategy_key = _strategy_key(intent, fallback_strategy)
        per_strategy.setdefault(strategy_key, _empty_strategy_report())
        strategy_report = per_strategy[strategy_key]
        strategy_report["intents_seen"] += 1

        request, reason = _translate_intent(intent)
        if request is None:
            _record_invalid(
                strategy_report=strategy_report,
                invalid_reasons=invalid_reasons,
                reason=reason,
                intent=intent,
                index=index,
            )
            _classify_invalid_intent(
                intent,
                reason,
                unsupported_order_types=unsupported_order_types,
                unsupported_sides=unsupported_sides,
                missing_required_fields=missing_required_fields,
            )
            continue

        second, second_reason = _translate_intent(intent)
        if second is None or second.client_order_id != request.client_order_id:
            deterministic_client_order_ids = False
            reason = (
                second_reason
                if second is None
                else "ValueError: client_order_id changed during deterministic check"
            )
            _record_invalid(
                strategy_report=strategy_report,
                invalid_reasons=invalid_reasons,
                reason=reason,
                intent=intent,
                index=index,
            )
            _classify_invalid_intent(
                intent,
                reason,
                unsupported_order_types=unsupported_order_types,
                unsupported_sides=unsupported_sides,
                missing_required_fields=missing_required_fields,
            )
            continue

        client_order_ids.append(request.client_order_id)
        translated_occurrences.append(
            {
                "client_order_id": request.client_order_id,
                "strategy_report": strategy_report,
                "intent": intent,
                "index": index,
            }
        )

    duplicates = _duplicates(client_order_ids)
    duplicate_keys = set(duplicates)
    if duplicates:
        deterministic_client_order_ids = False

    for occurrence in translated_occurrences:
        strategy_report = occurrence["strategy_report"]
        client_order_id = occurrence["client_order_id"]
        if client_order_id in duplicate_keys:
            _record_invalid(
                strategy_report=strategy_report,
                invalid_reasons=invalid_reasons,
                reason="duplicate client_order_id",
                intent=occurrence["intent"],
                index=occurrence["index"],
            )
            continue
        strategy_report["intents_valid"] += 1
        strategy_report["sample_valid_client_order_ids"].append(client_order_id)

    for strategy_report in per_strategy.values():
        strategy_report["sample_valid_client_order_ids"] = sorted(
            set(strategy_report["sample_valid_client_order_ids"])
        )[:5]
        strategy_report["sample_invalid_intent_ids"] = sorted(
            set(strategy_report["sample_invalid_intent_ids"])
        )[:5]
        strategy_report["invalid_reasons"] = dict(
            sorted(strategy_report["invalid_reasons"].items())
        )
        strategy_report["compatible"] = (
            strategy_report["intents_seen"] > 0
            and strategy_report["intents_invalid"] == 0
        )

    total_seen = sum(item["intents_seen"] for item in per_strategy.values())
    total_valid = sum(item["intents_valid"] for item in per_strategy.values())
    total_invalid = sum(item["intents_invalid"] for item in per_strategy.values())
    all_compatible = (
        bool(total_seen)
        and total_invalid == 0
        and not duplicates
        and deterministic_client_order_ids
    )
    recommendations = _recommendations(
        total_seen=total_seen,
        invalid_reasons=invalid_reasons,
        duplicates=duplicates,
    )

    report = {
        "dry_run": True,
        "paper_adapter_compatibility_report": True,
        "generated_at": now.isoformat(),
        "strategy_ids": strategy_id_list,
        "inputs": {
            "order_intents_count": len(order_intents),
            "strategies_checked": len(strategy_id_list),
        },
        "per_strategy": dict(sorted(per_strategy.items())),
        "aggregate": {
            "total_intents_seen": total_seen,
            "total_intents_valid": total_valid,
            "total_intents_invalid": total_invalid,
            "compatibility_rate": (total_valid / total_seen) if total_seen else 0.0,
            "all_intents_compatible": all_compatible,
            "dominant_invalid_reason": _dominant_reason(invalid_reasons),
        },
        "validation": {
            "deterministic_client_order_ids": deterministic_client_order_ids,
            "duplicate_client_order_ids": duplicates,
            "unsupported_order_types": sorted(unsupported_order_types),
            "unsupported_sides": sorted(unsupported_sides),
            "missing_required_fields": dict(sorted(missing_required_fields.items())),
        },
        "recommendations": recommendations,
        "safety": {
            "broker_calls_enabled": False,
            "market_data_enabled": False,
            "external_fetch_enabled": False,
            "paper_live_orders_enabled": False,
            "live_orders_enabled": False,
            "scheduler_changes_enabled": False,
        },
        "success": True,
        "errors": errors,
        "warnings": warnings,
    }
    return _json_safe(report)


def _translate_intent(intent: Any) -> tuple[Any | None, str]:
    try:
        return build_broker_order_request(intent), ""
    except Exception as exc:  # noqa: BLE001 - report must never fail on one intent.
        return None, _stable_exception_reason(exc)


def _stable_exception_reason(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    if len(message) > 160:
        message = message[:157] + "..."
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _empty_strategy_report() -> dict[str, Any]:
    return {
        "intents_seen": 0,
        "intents_valid": 0,
        "intents_invalid": 0,
        "invalid_reasons": defaultdict(int),
        "sample_valid_client_order_ids": [],
        "sample_invalid_intent_ids": [],
        "compatible": False,
    }


def _strategy_key(intent: Any, fallback: str) -> str:
    if not isinstance(intent, dict):
        return fallback
    value = intent.get("strategy_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _record_invalid(
    *,
    strategy_report: dict[str, Any],
    invalid_reasons: Counter,
    reason: str,
    intent: Any,
    index: int,
) -> None:
    strategy_report["intents_invalid"] += 1
    strategy_report["invalid_reasons"][reason] += 1
    invalid_reasons[reason] += 1
    strategy_report["sample_invalid_intent_ids"].append(_intent_sample_id(intent, index))


def _intent_sample_id(intent: Any, index: int) -> str:
    if isinstance(intent, dict):
        value = intent.get("intent_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"intent_index_{index}"


def _classify_invalid_intent(
    intent: Any,
    reason: str,
    *,
    unsupported_order_types: set[str],
    unsupported_sides: set[str],
    missing_required_fields: Counter,
) -> None:
    for message, field in _MISSING_FIELD_REASONS.items():
        if message in reason:
            missing_required_fields[field] += 1
    if not isinstance(intent, dict):
        return

    order_type = intent.get("order_type")
    if isinstance(order_type, str) and "order_type must be one of" in reason:
        unsupported_order_types.add(order_type.strip().upper())

    side = intent.get("side")
    if isinstance(side, str) and "side must be one of" in reason:
        unsupported_sides.add(side.strip().upper())


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _dominant_reason(reasons: Counter) -> str | None:
    if not reasons:
        return None
    highest = max(reasons.values())
    return sorted(reason for reason, count in reasons.items() if count == highest)[0]


def _recommendations(
    *,
    total_seen: int,
    invalid_reasons: Counter,
    duplicates: list[str],
) -> dict[str, list[str]]:
    steps: list[str] = []
    if total_seen == 0:
        steps.append("Collect more dry-run intents before paper adapter compatibility can be assessed.")
    if any("strategy_id is required" in reason for reason in invalid_reasons):
        steps.append("Fix missing strategy_id in generated intents before paper adapter wiring.")
    if any("order_type" in reason for reason in invalid_reasons):
        steps.append("Fix unsupported order_type before paper adapter wiring.")
    if duplicates:
        steps.append("Investigate duplicate client_order_id generation before paper adapter wiring.")
    if not steps:
        steps.append("Do not wire adapter into lifecycle jobs until all current intents translate cleanly.")
    return {
        "ordered_next_steps": steps,
        "do_not_do_yet": _DO_NOT_DO_YET[:],
    }


def _current_time(now_provider: Callable[[], datetime] | None) -> datetime:
    value = now_provider() if now_provider is not None else datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value
