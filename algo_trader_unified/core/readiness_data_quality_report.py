"""Read-only readiness and local data-quality diagnosis report."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from algo_trader_unified.core.strategy_realism_report import classify_skip_reason


NY_TZ = ZoneInfo("America/New_York")
_CATEGORIES = (
    "data_problem",
    "halt_or_safety_problem",
    "readiness_problem",
    "strategy_filter_problem",
    "unknown",
)


def build_readiness_data_quality_report(
    *,
    strategy_realism_report: dict,
    readiness_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    snapshots_dir=None,
    halt_state_path=None,
    now_provider: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe diagnosis report without mutating local trading state."""

    del snapshots_dir, halt_state_path

    now = _current_time(now_provider)
    errors: list[str] = []
    if not isinstance(strategy_realism_report, dict):
        strategy_realism_report = {}
        errors.append("strategy realism report is unavailable")

    strategy_ids = _strategy_ids(strategy_realism_report)
    per_strategy_source = strategy_realism_report.get("per_strategy")
    if not isinstance(per_strategy_source, dict):
        per_strategy_source = {}
        errors.append("strategy realism report missing per_strategy evidence")

    readiness_source = _readiness_source(strategy_realism_report, readiness_snapshot)
    account_snapshot_fresh = _first_present(
        readiness_source.get("account_snapshot_fresh"),
        _mapping_get(state_snapshot, "account_snapshot_fresh"),
    )
    nlv_valid = _first_present(
        readiness_source.get("nlv_valid"),
        _mapping_get(state_snapshot, "nlv_valid"),
    )
    halt_active = readiness_source.get("halt_active")

    per_strategy: dict[str, dict[str, Any]] = {}
    category_counts: Counter[str] = Counter()
    skip_counts: Counter[str] = Counter()

    for strategy_id in strategy_ids:
        source = per_strategy_source.get(strategy_id)
        if not isinstance(source, dict):
            source = {}
        top_reason = _top_skip_reason(source)
        if top_reason is not None:
            skip_counts[top_reason] += 1
        category = _likely_category(
            top_reason=top_reason,
            source=source,
            account_snapshot_fresh=account_snapshot_fresh,
            nlv_valid=nlv_valid,
            halt_active=halt_active,
        )
        category_counts[category] += 1
        per_strategy[strategy_id] = {
            "top_skip_reason": top_reason,
            "likely_blocker_category": category,
            "readiness_available": _bool_or_none(source.get("readiness_available")),
            "readiness_passed": _bool_or_none(source.get("readiness_passed")),
            "dirty_state": _bool_or_none(source.get("dirty_state")),
            "calendar_expired": _bool_or_none(source.get("calendar_expired")),
            "iv_baseline_available": _bool_or_none(source.get("iv_baseline_available")),
            "account_snapshot_fresh": _bool_or_none(account_snapshot_fresh),
            "nlv_valid": _bool_or_none(nlv_valid),
            "halt_active": _bool_or_none(halt_active),
            "diagnosis": _diagnosis(category, top_reason),
            "recommended_next_check": _recommended_next_check(category, top_reason),
        }

    dominant_category = _top_counter_key(category_counts)
    dominant_reason = _dominant_skip_reason(strategy_realism_report, skip_counts)
    missing_readiness = _missing_readiness_ids(strategy_ids, per_strategy, readiness_source)
    stale_or_missing_inputs = _stale_or_missing_inputs(
        strategy_ids=strategy_ids,
        per_strategy=per_strategy,
        account_snapshot_fresh=account_snapshot_fresh,
        nlv_valid=nlv_valid,
        missing_readiness=missing_readiness,
    )
    report_success = strategy_realism_report.get("success") is True and not errors
    if strategy_realism_report.get("success") is not True:
        errors.append("strategy realism report was unsuccessful")
    for message in strategy_realism_report.get("errors") or []:
        errors.append(f"strategy realism report: {message}")

    report = {
        "dry_run": True,
        "readiness_data_quality_report": True,
        "generated_at": now.isoformat(),
        "strategy_ids": strategy_ids,
        "inputs": {
            "used_strategy_realism_report": True,
            "strategy_realism_success": strategy_realism_report.get("success"),
            "session_date": strategy_realism_report.get("session_date"),
        },
        "per_strategy": per_strategy,
        "aggregate": {
            "dominant_blocker_category": dominant_category,
            "dominant_skip_reason": dominant_reason,
            "affected_strategy_count": len(strategy_ids),
            "readiness_problem_count": category_counts["readiness_problem"],
            "data_problem_count": category_counts["data_problem"],
            "strategy_filter_problem_count": category_counts["strategy_filter_problem"],
            "halt_or_safety_problem_count": category_counts["halt_or_safety_problem"],
            "unknown_problem_count": category_counts["unknown"],
        },
        "data_quality": {
            "account_snapshot_fresh": _bool_or_none(account_snapshot_fresh),
            "nlv_valid": _bool_or_none(nlv_valid),
            "iv_baseline_available_by_strategy": {
                strategy_id: per_strategy[strategy_id]["iv_baseline_available"]
                for strategy_id in strategy_ids
            },
            "calendar_expired_by_strategy": {
                strategy_id: per_strategy[strategy_id]["calendar_expired"]
                for strategy_id in strategy_ids
            },
            "missing_readiness_strategy_ids": missing_readiness,
            "stale_or_missing_inputs": stale_or_missing_inputs,
        },
        "safety": {
            "broker_calls_enabled": False,
            "market_data_enabled": False,
            "paper_live_orders_enabled": False,
            "strategy_changes_enabled": False,
            "lifecycle_changes_enabled": False,
        },
        "recommendations": {
            "ordered_next_steps": _ordered_next_steps(
                dominant_category=dominant_category,
                categories=category_counts,
                stale_or_missing_inputs=stale_or_missing_inputs,
            )
        },
        "success": report_success,
        "errors": errors,
    }
    return _json_safe(report)


def _current_time(now_provider: Callable[[], datetime] | None) -> datetime:
    now = now_provider() if now_provider is not None else datetime.now(NY_TZ)
    if not isinstance(now, datetime):
        raise ValueError("now_provider must return a datetime")
    if now.tzinfo is None:
        return now.replace(tzinfo=NY_TZ)
    return now.astimezone(NY_TZ)


def _strategy_ids(report: dict[str, Any]) -> list[str]:
    value = report.get("strategy_ids")
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    per_strategy = report.get("per_strategy")
    if isinstance(per_strategy, dict):
        return sorted(str(item) for item in per_strategy)
    return []


def _readiness_source(report: dict[str, Any], readiness_snapshot: dict | None) -> dict[str, Any]:
    readiness = report.get("readiness")
    merged = readiness if isinstance(readiness, dict) else {}
    if isinstance(readiness_snapshot, dict):
        merged = {**merged, **readiness_snapshot}
    return merged


def _mapping_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _top_skip_reason(source: dict[str, Any]) -> str | None:
    reasons = source.get("skip_reasons")
    if isinstance(reasons, dict) and reasons:
        return _top_counter_key(Counter({str(key): int(value) for key, value in reasons.items()}))
    reason = source.get("top_skip_reason")
    return str(reason) if reason else None


def _top_counter_key(counts: Counter[str] | dict[str, int]) -> str | None:
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))[0][0]


def _dominant_skip_reason(report: dict[str, Any], fallback_counts: Counter[str]) -> str | None:
    aggregate = report.get("aggregate")
    if isinstance(aggregate, dict) and isinstance(aggregate.get("skip_reasons"), dict):
        return _top_counter_key(
            Counter({str(key): int(value) for key, value in aggregate["skip_reasons"].items()})
        )
    return _top_counter_key(fallback_counts)


def _likely_category(
    *,
    top_reason: str | None,
    source: dict[str, Any],
    account_snapshot_fresh: Any,
    nlv_valid: Any,
    halt_active: Any,
) -> str:
    if top_reason:
        return classify_skip_reason(top_reason)
    if halt_active is True:
        return "halt_or_safety_problem"
    if source.get("readiness_available") is False or source.get("readiness_passed") is False:
        return "readiness_problem"
    if (
        account_snapshot_fresh is False
        or nlv_valid is False
        or source.get("calendar_expired") is True
        or source.get("iv_baseline_available") is False
    ):
        return "data_problem"
    return "unknown"


def _diagnosis(category: str, reason: str | None) -> str:
    prefix = f"Top skip reason {reason}. " if reason else ""
    if category == "readiness_problem":
        return prefix + "Investigate readiness cadence, staleness, and StateStore availability before strategy changes."
    if category == "data_problem":
        return prefix + "Investigate missing or stale local inputs such as IV rank, VIX, account snapshot, and calendar/session state."
    if category == "strategy_filter_problem":
        return prefix + "Strategy filters are currently the dominant cause; confirm data and readiness quality before considering threshold tuning."
    if category == "halt_or_safety_problem":
        return prefix + "Review halt and reconciliation safety gates before any strategy work."
    return prefix + "Improve or complete reason-code logging before interpreting the skip distribution."


def _recommended_next_check(category: str, reason: str | None) -> str:
    text = (reason or "").upper()
    if category == "readiness_problem":
        return "Fix readiness provider cadence before tuning strategy filters."
    if category == "data_problem":
        if "VIX" in text:
            return "Investigate VIX source availability before changing VIX gate."
        if "IV" in text:
            return "Investigate IV rank source freshness before changing IV rank thresholds."
        return "Investigate local account, IV, VIX, and calendar inputs before tuning strategy filters."
    if category == "strategy_filter_problem":
        return "Confirm data and readiness quality before interpreting strategy filter skips."
    if category == "halt_or_safety_problem":
        return "Resolve NEEDS_RECONCILIATION before enabling lifecycle cadence."
    return "Improve reason-code logging before interpreting skip distribution."


def _missing_readiness_ids(
    strategy_ids: list[str],
    per_strategy: dict[str, dict[str, Any]],
    readiness: dict[str, Any],
) -> list[str]:
    missing = {
        strategy_id
        for strategy_id in strategy_ids
        if per_strategy[strategy_id].get("readiness_available") is False
    }
    explicit = readiness.get("missing_readiness_strategy_ids")
    if isinstance(explicit, list):
        missing.update(str(item) for item in explicit)
    return sorted(missing)


def _stale_or_missing_inputs(
    *,
    strategy_ids: list[str],
    per_strategy: dict[str, dict[str, Any]],
    account_snapshot_fresh: Any,
    nlv_valid: Any,
    missing_readiness: list[str],
) -> list[str]:
    inputs: set[str] = set()
    if account_snapshot_fresh is not True:
        inputs.add("account_snapshot")
    if nlv_valid is not True:
        inputs.add("nlv")
    if missing_readiness:
        inputs.add("readiness")
    for strategy_id in strategy_ids:
        if per_strategy[strategy_id].get("iv_baseline_available") is not True:
            inputs.add(f"{strategy_id}:iv_baseline")
        if per_strategy[strategy_id].get("calendar_expired") is not False:
            inputs.add(f"{strategy_id}:calendar")
    return sorted(inputs)


def _ordered_next_steps(
    *,
    dominant_category: str | None,
    categories: Counter[str],
    stale_or_missing_inputs: list[str],
) -> list[str]:
    candidates: list[str] = []
    ordered_categories = []
    if dominant_category is not None:
        ordered_categories.append(dominant_category)
    ordered_categories.extend(
        category
        for category in _CATEGORIES
        if category != dominant_category and categories.get(category, 0) > 0
    )
    for category in ordered_categories:
        if category == "readiness_problem":
            candidates.append("Fix readiness provider cadence before tuning strategy filters.")
        elif category == "data_problem":
            candidates.append("Investigate local data freshness before changing strategy filters.")
        elif category == "strategy_filter_problem":
            candidates.append(
                "Confirm data and readiness quality before interpreting strategy filter skips."
            )
        elif category == "halt_or_safety_problem":
            candidates.append("Resolve NEEDS_RECONCILIATION before enabling lifecycle cadence.")
        elif category == "unknown":
            candidates.append("Improve reason-code logging before interpreting skip distribution.")
    if any("iv" in item.lower() for item in stale_or_missing_inputs):
        candidates.append("Investigate IV rank source freshness before changing IV rank thresholds.")
    return _dedupe(candidates)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, tuple):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True)
