"""Read-only strategy quality decision support for S01/S02."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from algo_trader_unified.core.skip_reasons import UNKNOWN_SKIP_REASON
from algo_trader_unified.core.strategy_realism_report import classify_skip_reason


NY_TZ = ZoneInfo("America/New_York")

READY_FOR_STRATEGY_TUNING = "READY_FOR_STRATEGY_TUNING"
BLOCKED_BY_READINESS = "BLOCKED_BY_READINESS"
BLOCKED_BY_LOCAL_DATA = "BLOCKED_BY_LOCAL_DATA"
BLOCKED_BY_SAFETY_OR_HALT = "BLOCKED_BY_SAFETY_OR_HALT"
NEEDS_MORE_OBSERVABILITY = "NEEDS_MORE_OBSERVABILITY"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

_DECISION_PRIORITY = (
    BLOCKED_BY_SAFETY_OR_HALT,
    BLOCKED_BY_READINESS,
    BLOCKED_BY_LOCAL_DATA,
    NEEDS_MORE_OBSERVABILITY,
    READY_FOR_STRATEGY_TUNING,
    INSUFFICIENT_EVIDENCE,
)
_LOCAL_DATA_BLOCKERS = {
    "account_snapshot",
    "account_snapshot_invalid",
    "account_snapshot_stale",
    "iv_rank",
    "market_calendar",
    "readiness_snapshot",
    "vix",
}
_DO_NOT_DO_YET = [
    "Do not change strategy thresholds from this report.",
    "Do not change position sizing from this report.",
    "Do not change execution, broker, paper, or live order behavior from this report.",
    "Do not change lifecycle or readiness staleness windows from this report.",
]


def build_strategy_quality_decision_report(
    *,
    strategy_realism_report: dict | None,
    readiness_data_quality_report: dict | None,
    local_input_audit_report: dict | None,
    strategy_ids,
    now_provider: Callable[[], datetime] | None = None,
) -> dict:
    """Combine pre-built 4C reports into a pure read-only decision report."""

    now = _current_time(now_provider)
    ids = [str(strategy_id) for strategy_id in (strategy_ids or [])]
    errors: list[str] = []
    warnings: list[str] = []
    input_status = {
        "strategy_realism": _input_status(
            strategy_realism_report, "strategy_realism_report"
        ),
        "readiness_data_quality": _input_status(
            readiness_data_quality_report, "readiness_data_quality_report"
        ),
        "local_input_audit": _input_status(
            local_input_audit_report, "local_input_audit_report"
        ),
    }
    for name, status in sorted(input_status.items()):
        if not status["present"]:
            errors.append(f"{name} input report is missing or malformed")
        elif status["success"] is not True:
            errors.append(f"{name} input report was unsuccessful")
    valid_inputs = all(status["success"] is True for status in input_status.values())

    realism = strategy_realism_report if isinstance(strategy_realism_report, dict) else {}
    readiness = (
        readiness_data_quality_report
        if isinstance(readiness_data_quality_report, dict)
        else {}
    )
    local = local_input_audit_report if isinstance(local_input_audit_report, dict) else {}

    if not ids:
        ids = _strategy_ids(realism, readiness, local)
    if not ids:
        warnings.append("strategy_ids unavailable")

    per_strategy = {}
    category_counts: Counter[str] = Counter()
    skip_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()

    for strategy_id in ids:
        item = _per_strategy_decision(
            strategy_id=strategy_id,
            valid_inputs=valid_inputs,
            realism=realism,
            readiness=readiness,
            local=local,
        )
        per_strategy[strategy_id] = item
        decision_counts[item["decision"]] += 1
        category = item.get("likely_blocker_category")
        if category:
            category_counts[str(category)] += 1
        reason = item.get("top_skip_reason")
        if reason:
            skip_counts[str(reason)] += 1

    safety_available = _clear_safety_blocker(realism, readiness, local, ids)
    overall_decision = _overall_decision(
        valid_inputs=valid_inputs,
        safety_available=safety_available,
        decisions=[item["decision"] for item in per_strategy.values()],
    )
    recommendations = _recommendations(overall_decision, per_strategy)
    report = {
        "dry_run": True,
        "strategy_quality_decision_report": True,
        "generated_at": now.isoformat(),
        "strategy_ids": ids,
        "inputs": {
            "strategy_realism_success": input_status["strategy_realism"]["success"],
            "readiness_data_quality_success": input_status["readiness_data_quality"][
                "success"
            ],
            "local_input_audit_success": input_status["local_input_audit"]["success"],
            "session_date": _session_date(realism, readiness, local),
        },
        "per_strategy": per_strategy,
        "aggregate": {
            "overall_decision": overall_decision,
            "dominant_blocker_category": _top_counter_key(category_counts),
            "dominant_skip_reason": _dominant_skip_reason(realism, readiness, skip_counts),
            "strategies_ready_for_tuning": _ids_with_decision(
                per_strategy, READY_FOR_STRATEGY_TUNING
            ),
            "strategies_blocked_by_readiness": _ids_with_decision(
                per_strategy, BLOCKED_BY_READINESS
            ),
            "strategies_blocked_by_data": _ids_with_decision(
                per_strategy, BLOCKED_BY_LOCAL_DATA
            ),
            "strategies_blocked_by_safety": _ids_with_decision(
                per_strategy, BLOCKED_BY_SAFETY_OR_HALT
            ),
            "strategies_need_more_observability": _ids_with_decision(
                per_strategy, NEEDS_MORE_OBSERVABILITY
            ),
        },
        "evidence": {
            "skip_reason_summary": _skip_reason_summary(realism),
            "input_issue_summary": _input_issue_summary(local),
            "readiness_issue_summary": _readiness_issue_summary(readiness),
        },
        "recommendations": recommendations,
        "safety": {
            "broker_calls_enabled": False,
            "market_data_enabled": False,
            "external_fetch_enabled": False,
            "paper_live_orders_enabled": False,
            "strategy_changes_enabled": False,
            "lifecycle_changes_enabled": False,
        },
        "success": valid_inputs and not errors,
        "errors": errors,
        "warnings": warnings,
    }
    return _json_safe(report)


def _per_strategy_decision(
    *,
    strategy_id: str,
    valid_inputs: bool,
    realism: dict[str, Any],
    readiness: dict[str, Any],
    local: dict[str, Any],
) -> dict[str, Any]:
    realism_item = _mapping_get(realism.get("per_strategy"), strategy_id)
    readiness_item = _mapping_get(readiness.get("per_strategy"), strategy_id)
    local_item = _mapping_get(local.get("per_strategy"), strategy_id)
    signals_generated = _int_value(realism_item.get("signals_generated"))
    signals_skipped = _int_value(realism_item.get("signals_skipped"))
    top_reason = _top_skip_reason(realism_item, readiness_item)
    likely_category = _likely_blocker_category(top_reason, readiness_item)
    input_blockers = _input_blockers(local_item)
    readiness_status = _readiness_status(realism_item, readiness_item, local_item)
    local_input_status = _local_input_status(input_blockers, local)

    if not valid_inputs:
        decision = INSUFFICIENT_EVIDENCE
    elif _strategy_safety_blocked(realism_item, readiness_item, local_item):
        decision = BLOCKED_BY_SAFETY_OR_HALT
    elif _strategy_readiness_blocked(realism_item, readiness_item, local_item):
        decision = BLOCKED_BY_READINESS
    elif _strategy_local_data_blocked(input_blockers, local):
        decision = BLOCKED_BY_LOCAL_DATA
    elif _needs_more_observability(
        signals_generated=signals_generated,
        signals_skipped=signals_skipped,
        top_reason=top_reason,
        likely_category=likely_category,
    ):
        decision = NEEDS_MORE_OBSERVABILITY
    elif likely_category == "strategy_filter_problem":
        decision = READY_FOR_STRATEGY_TUNING
    else:
        decision = INSUFFICIENT_EVIDENCE

    return {
        "signals_generated": signals_generated,
        "signals_skipped": signals_skipped,
        "top_skip_reason": top_reason,
        "likely_blocker_category": _decision_category(decision, likely_category),
        "input_blockers": input_blockers,
        "readiness_status": readiness_status,
        "local_input_status": local_input_status,
        "decision": decision,
        "decision_reason": _decision_reason(decision, strategy_id, top_reason),
        "recommended_next_step": _recommended_next_step(decision, strategy_id),
    }


def _input_status(report: Any, marker: str) -> dict[str, Any]:
    present = isinstance(report, dict) and bool(report) and report.get(marker) is True
    return {"present": present, "success": report.get("success") if present else None}


def _strategy_ids(*reports: dict[str, Any]) -> list[str]:
    ids: set[str] = set()
    for report in reports:
        value = report.get("strategy_ids")
        if isinstance(value, (list, tuple)):
            ids.update(str(item) for item in value)
        per_strategy = report.get("per_strategy")
        if isinstance(per_strategy, dict):
            ids.update(str(item) for item in per_strategy)
    return sorted(ids)


def _mapping_get(mapping: Any, key: str) -> dict[str, Any]:
    if isinstance(mapping, dict) and isinstance(mapping.get(key), dict):
        return mapping[key]
    return {}


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _top_skip_reason(*items: dict[str, Any]) -> str | None:
    counts: Counter[str] = Counter()
    for item in items:
        reasons = item.get("skip_reasons")
        if isinstance(reasons, dict):
            for reason, count in reasons.items():
                counts[str(reason)] += _int_value(count)
    if counts:
        return _top_counter_key(counts)
    for item in items:
        reason = item.get("top_skip_reason")
        if reason:
            return str(reason)
    return None


def _likely_blocker_category(reason: str | None, readiness_item: dict[str, Any]) -> str:
    category = readiness_item.get("likely_blocker_category")
    if category:
        return str(category)
    if reason:
        return classify_skip_reason(reason)
    return "unknown"


def _input_blockers(local_item: dict[str, Any]) -> list[str]:
    blockers = local_item.get("input_blockers")
    if not isinstance(blockers, list):
        return []
    return sorted(str(item) for item in blockers)


def _readiness_status(
    realism_item: dict[str, Any],
    readiness_item: dict[str, Any],
    local_item: dict[str, Any],
) -> str:
    if _strategy_safety_blocked(realism_item, readiness_item, local_item):
        return "halt_or_safety_blocked"
    if realism_item.get("readiness_available") is False or local_item.get(
        "readiness_available"
    ) is False:
        return "missing"
    if realism_item.get("dirty_state") is True or readiness_item.get("dirty_state") is True:
        return "dirty"
    if realism_item.get("calendar_expired") is True or local_item.get(
        "calendar_expired"
    ) is True:
        return "calendar_expired"
    if realism_item.get("readiness_passed") is False or local_item.get(
        "readiness_passed"
    ) is False:
        return "failed"
    if realism_item.get("readiness_passed") is True or local_item.get(
        "readiness_passed"
    ) is True:
        return "ok"
    return "unknown"


def _local_input_status(
    input_blockers: list[str],
    local: dict[str, Any],
) -> str:
    if input_blockers:
        return "blocked"
    aggregate = local.get("aggregate")
    if isinstance(aggregate, dict):
        if any(_int_value(aggregate.get(key)) > 0 for key in (
            "blocking_input_count",
            "stale_input_count",
            "missing_input_count",
        )):
            return "blocked"
    return "ok"


def _strategy_safety_blocked(
    realism_item: dict[str, Any],
    readiness_item: dict[str, Any],
    local_item: dict[str, Any],
) -> bool:
    return (
        readiness_item.get("likely_blocker_category") == "halt_or_safety_problem"
        or realism_item.get("halt_active") is True
        or readiness_item.get("halt_active") is True
        or "halt_or_safety_block" in _input_blockers(local_item)
    )


def _strategy_readiness_blocked(
    realism_item: dict[str, Any],
    readiness_item: dict[str, Any],
    local_item: dict[str, Any],
) -> bool:
    return _readiness_status(realism_item, readiness_item, local_item) in {
        "missing",
        "dirty",
        "calendar_expired",
        "failed",
    } or readiness_item.get("likely_blocker_category") == "readiness_problem"


def _strategy_local_data_blocked(input_blockers: list[str], local: dict[str, Any]) -> bool:
    if any(blocker in _LOCAL_DATA_BLOCKERS for blocker in input_blockers):
        return True
    aggregate = local.get("aggregate")
    return isinstance(aggregate, dict) and any(
        _int_value(aggregate.get(key)) > 0
        for key in ("blocking_input_count", "stale_input_count", "missing_input_count")
    )


def _needs_more_observability(
    *,
    signals_generated: int,
    signals_skipped: int,
    top_reason: str | None,
    likely_category: str,
) -> bool:
    if signals_generated + signals_skipped <= 0:
        return True
    if not top_reason:
        return True
    text = top_reason.upper()
    return likely_category == "unknown" or text == UNKNOWN_SKIP_REASON or "UNKNOWN" in text


def _decision_category(decision: str, likely_category: str) -> str:
    if decision == BLOCKED_BY_READINESS:
        return "readiness_problem"
    if decision == BLOCKED_BY_LOCAL_DATA:
        return "data_problem"
    if decision == BLOCKED_BY_SAFETY_OR_HALT:
        return "halt_or_safety_problem"
    if decision == READY_FOR_STRATEGY_TUNING:
        return "strategy_filter_problem"
    if decision == NEEDS_MORE_OBSERVABILITY:
        return "unknown"
    return likely_category or "unknown"


def _overall_decision(
    *,
    valid_inputs: bool,
    safety_available: bool,
    decisions: list[str],
) -> str:
    if not valid_inputs:
        return BLOCKED_BY_SAFETY_OR_HALT if safety_available else INSUFFICIENT_EVIDENCE
    if not decisions:
        return INSUFFICIENT_EVIDENCE
    if BLOCKED_BY_SAFETY_OR_HALT in decisions:
        return BLOCKED_BY_SAFETY_OR_HALT
    if all(decision == READY_FOR_STRATEGY_TUNING for decision in decisions):
        return READY_FOR_STRATEGY_TUNING
    for decision in _DECISION_PRIORITY:
        if decision in decisions and decision != READY_FOR_STRATEGY_TUNING:
            return decision
    return INSUFFICIENT_EVIDENCE


def _clear_safety_blocker(
    realism: dict[str, Any],
    readiness: dict[str, Any],
    local: dict[str, Any],
    strategy_ids: list[str],
) -> bool:
    readiness_source = realism.get("readiness")
    if isinstance(readiness_source, dict) and readiness_source.get("halt_active") is True:
        return True
    for strategy_id in strategy_ids:
        if _strategy_safety_blocked(
            _mapping_get(realism.get("per_strategy"), strategy_id),
            _mapping_get(readiness.get("per_strategy"), strategy_id),
            _mapping_get(local.get("per_strategy"), strategy_id),
        ):
            return True
    return False


def _ids_with_decision(per_strategy: dict[str, dict[str, Any]], decision: str) -> list[str]:
    return sorted(
        strategy_id
        for strategy_id, item in per_strategy.items()
        if item.get("decision") == decision
    )


def _dominant_skip_reason(
    realism: dict[str, Any],
    readiness: dict[str, Any],
    fallback_counts: Counter[str],
) -> str | None:
    for report in (realism, readiness):
        aggregate = report.get("aggregate")
        if isinstance(aggregate, dict) and isinstance(aggregate.get("skip_reasons"), dict):
            counts = Counter(
                {
                    str(reason): _int_value(count)
                    for reason, count in aggregate["skip_reasons"].items()
                }
            )
            return _top_counter_key(counts)
        if isinstance(aggregate, dict) and aggregate.get("dominant_skip_reason"):
            return str(aggregate.get("dominant_skip_reason"))
    return _top_counter_key(fallback_counts)


def _skip_reason_summary(realism: dict[str, Any]) -> dict[str, Any]:
    aggregate = realism.get("aggregate")
    if not isinstance(aggregate, dict):
        return {"skip_reasons": {}, "dominant_skip_reason": None}
    reasons = aggregate.get("skip_reasons") if isinstance(aggregate.get("skip_reasons"), dict) else {}
    return {
        "skip_reasons": {str(key): _int_value(value) for key, value in sorted(reasons.items())},
        "dominant_skip_reason": _top_counter_key(
            Counter({str(key): _int_value(value) for key, value in reasons.items()})
        ),
    }


def _input_issue_summary(local: dict[str, Any]) -> dict[str, Any]:
    aggregate = local.get("aggregate")
    if not isinstance(aggregate, dict):
        return {}
    return {
        "dominant_input_issue": aggregate.get("dominant_input_issue"),
        "blocking_input_count": _int_value(aggregate.get("blocking_input_count")),
        "stale_input_count": _int_value(aggregate.get("stale_input_count")),
        "missing_input_count": _int_value(aggregate.get("missing_input_count")),
    }


def _readiness_issue_summary(readiness: dict[str, Any]) -> dict[str, Any]:
    aggregate = readiness.get("aggregate")
    data_quality = readiness.get("data_quality")
    return {
        "dominant_blocker_category": aggregate.get("dominant_blocker_category")
        if isinstance(aggregate, dict)
        else None,
        "dominant_skip_reason": aggregate.get("dominant_skip_reason")
        if isinstance(aggregate, dict)
        else None,
        "stale_or_missing_inputs": sorted(data_quality.get("stale_or_missing_inputs") or [])
        if isinstance(data_quality, dict)
        else [],
        "missing_readiness_strategy_ids": sorted(
            data_quality.get("missing_readiness_strategy_ids") or []
        )
        if isinstance(data_quality, dict)
        else [],
    }


def _recommendations(
    overall_decision: str,
    per_strategy: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    steps: list[str] = []
    if overall_decision == BLOCKED_BY_SAFETY_OR_HALT:
        steps.append("Resolve halt or reconciliation safety state before strategy tuning.")
    if any(item["decision"] == BLOCKED_BY_READINESS for item in per_strategy.values()):
        steps.append("Fix readiness cadence before strategy tuning.")
    if any(item["decision"] == BLOCKED_BY_LOCAL_DATA for item in per_strategy.values()):
        steps.append("Fix IV/VIX local input availability before changing thresholds.")
    if any(item["decision"] == NEEDS_MORE_OBSERVABILITY for item in per_strategy.values()):
        steps.append("Improve skip_reason logging before interpreting strategy quality.")
    for strategy_id in sorted(per_strategy):
        item = per_strategy[strategy_id]
        if item["decision"] == READY_FOR_STRATEGY_TUNING:
            steps.append(
                f"{strategy_id} appears ready for strategy-filter review; do not change sizing/execution."
            )
        elif item["decision"] == BLOCKED_BY_LOCAL_DATA:
            steps.append(
                f"{strategy_id} is not ready for tuning because local data inputs are unavailable."
            )
    if not steps:
        steps.append("Collect complete 4C report evidence before deciding on strategy tuning.")
    return {
        "ordered_next_steps": _dedupe(steps),
        "do_not_do_yet": list(_DO_NOT_DO_YET),
    }


def _decision_reason(decision: str, strategy_id: str, top_reason: str | None) -> str:
    prefix = f"{strategy_id}: "
    if decision == READY_FOR_STRATEGY_TUNING:
        return prefix + "readiness and local inputs are clean while strategy-filter skips dominate."
    if decision == BLOCKED_BY_READINESS:
        return prefix + "readiness is missing, failed, dirty, or calendar-expired."
    if decision == BLOCKED_BY_LOCAL_DATA:
        return prefix + "local IV/VIX/calendar/account input evidence is missing, stale, or invalid."
    if decision == BLOCKED_BY_SAFETY_OR_HALT:
        return prefix + "halt, reconciliation, or safety evidence is active."
    if decision == NEEDS_MORE_OBSERVABILITY:
        return prefix + "skip reason evidence is missing, unknown, or too sparse."
    if top_reason:
        return prefix + f"evidence is not strong enough to classify beyond {top_reason}."
    return prefix + "required 4C evidence is incomplete or unsuccessful."


def _recommended_next_step(decision: str, strategy_id: str) -> str:
    if decision == READY_FOR_STRATEGY_TUNING:
        return f"{strategy_id} appears ready for strategy-filter review; do not change sizing/execution."
    if decision == BLOCKED_BY_READINESS:
        return "Fix readiness cadence before strategy tuning."
    if decision == BLOCKED_BY_LOCAL_DATA:
        return "Fix IV/VIX local input availability before changing thresholds."
    if decision == BLOCKED_BY_SAFETY_OR_HALT:
        return "Resolve halt or reconciliation safety state before strategy tuning."
    if decision == NEEDS_MORE_OBSERVABILITY:
        return "Improve skip_reason logging before interpreting strategy quality."
    return "Collect complete 4C report evidence before deciding on strategy tuning."


def _session_date(*reports: dict[str, Any]) -> Any:
    for report in reports:
        value = report.get("session_date")
        if value is not None:
            return value
        inputs = report.get("inputs")
        if isinstance(inputs, dict) and inputs.get("session_date") is not None:
            return inputs.get("session_date")
    return None


def _top_counter_key(counts: Counter[str] | dict[str, int]) -> str | None:
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))[0][0]


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _current_time(now_provider: Callable[[], datetime] | None) -> datetime:
    now = now_provider() if now_provider is not None else datetime.now(NY_TZ)
    if not isinstance(now, datetime):
        raise ValueError("now_provider must return a datetime")
    return now.astimezone(NY_TZ) if now.tzinfo is not None else now.replace(tzinfo=NY_TZ)


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
