"""Read-only digest-driven strategy realism baseline reporting."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.halt_state_utils import halt_is_active
from algo_trader_unified.core.skip_reasons import UNKNOWN_SKIP_REASON


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_STRATEGY_IDS = (S01_VOL_BASELINE, S02_VOL_ENHANCED)
_ACTIVE_INTENT_STATUSES = {"created", "submitted", "confirmed", "filled"}
_SNAPSHOT_TIMESTAMP_FIELDS = ("timestamp", "captured_at", "snapshot_at", "generated_at")


def build_strategy_realism_report(
    *,
    ledger_reader,
    state_store,
    readiness_provider,
    snapshots_dir,
    halt_state_path,
    strategy_ids,
    session_date=None,
    now_provider: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe local report without mutating trading state."""

    now = _current_time(now_provider)
    session = _session_date(session_date, now)
    ids = tuple(strategy_ids or DEFAULT_STRATEGY_IDS)
    errors: list[str] = []
    critical_errors: list[str] = []

    per_strategy = {
        strategy_id: {
            "signals_generated": 0,
            "signals_skipped": 0,
            "skip_reasons": {},
            "top_skip_reason": None,
            "readiness_available": False,
            "readiness_passed": None,
            "dirty_state": None,
            "calendar_expired": None,
            "iv_baseline_available": None,
            "active_intents_count": None,
            "open_positions_count": None,
        }
        for strategy_id in ids
    }
    aggregate_generated = 0
    aggregate_skipped = 0
    aggregate_skip_reasons: Counter[str] = Counter()

    for event in _iter_session_events(ledger_reader, session, errors, critical_errors):
        event_type = event.get("event_type")
        if event_type not in {"SIGNAL_GENERATED", "SIGNAL_SKIPPED"}:
            continue
        strategy_id = _strategy_id(event)
        if strategy_id not in per_strategy:
            continue
        if event_type == "SIGNAL_GENERATED":
            per_strategy[strategy_id]["signals_generated"] += 1
            aggregate_generated += 1
        elif event_type == "SIGNAL_SKIPPED":
            reason = _skip_reason(event)
            per_strategy[strategy_id]["signals_skipped"] += 1
            per_strategy[strategy_id].setdefault("skip_reasons", {})
            per_strategy[strategy_id]["skip_reasons"][reason] = (
                per_strategy[strategy_id]["skip_reasons"].get(reason, 0) + 1
            )
            aggregate_skipped += 1
            aggregate_skip_reasons[reason] += 1

    readiness_snapshot = _call_readiness_provider(readiness_provider, errors)
    readiness_records = _readiness_records(state_store, ids, errors)
    counts_by_strategy = _state_counts_by_strategy(state_store, ids, errors)
    for strategy_id in ids:
        record = readiness_records.get(strategy_id)
        provider_values = _provider_values(readiness_snapshot, strategy_id)
        per_strategy[strategy_id].update(provider_values)
        per_strategy[strategy_id]["readiness_available"] = isinstance(record, dict) or bool(
            provider_values
        )
        if isinstance(record, dict):
            per_strategy[strategy_id]["readiness_passed"] = _readiness_passed(record)
            for key in ("dirty_state", "calendar_expired", "iv_baseline_available"):
                if record.get(key) is not None:
                    per_strategy[strategy_id][key] = record.get(key)
        per_strategy[strategy_id]["active_intents_count"] = counts_by_strategy.get(
            strategy_id, {}
        ).get("active_intents_count")
        per_strategy[strategy_id]["open_positions_count"] = counts_by_strategy.get(
            strategy_id, {}
        ).get("open_positions_count")
        per_strategy[strategy_id]["skip_reasons"] = dict(
            sorted(per_strategy[strategy_id]["skip_reasons"].items())
        )
        per_strategy[strategy_id]["top_skip_reason"] = _top_reason(
            per_strategy[strategy_id]["skip_reasons"]
        )

    aggregate = {
        "total_signals_generated": aggregate_generated,
        "total_signals_skipped": aggregate_skipped,
        "skip_reasons": dict(sorted(aggregate_skip_reasons.items())),
        "top_skip_reason": _top_reason(aggregate_skip_reasons),
        "generated_to_skipped_ratio": _ratio(aggregate_generated, aggregate_skipped),
    }
    halt_state = _load_halt_state(Path(halt_state_path), errors, critical_errors)
    readiness = {
        "account_snapshot_fresh": _account_snapshot_fresh(
            readiness_snapshot,
            Path(snapshots_dir),
            now,
            errors,
        ),
        "nlv_valid": _snapshot_attr(readiness_snapshot, "nlv_valid"),
        "halt_active": _halt_active(halt_state, readiness_snapshot, ids),
        "missing_readiness_strategy_ids": [
            strategy_id
            for strategy_id in ids
            if per_strategy[strategy_id]["readiness_available"] is False
        ],
    }

    diagnostics = _diagnostics(ids, per_strategy, aggregate, readiness)
    report = {
        "dry_run": True,
        "strategy_realism_report": True,
        "generated_at": now.isoformat(),
        "session_date": session.isoformat(),
        "strategy_ids": list(ids),
        "per_strategy": per_strategy,
        "aggregate": aggregate,
        "readiness": readiness,
        "diagnostics": diagnostics,
        "safety": {
            "broker_calls_enabled": False,
            "market_data_enabled": False,
            "paper_live_orders_enabled": False,
            "lifecycle_changes_enabled": False,
        },
        "success": not critical_errors,
        "errors": errors,
    }
    return _json_safe(report)


def classify_skip_reason(reason: Any) -> str:
    text = str(reason or UNKNOWN_SKIP_REASON).upper()
    if text == UNKNOWN_SKIP_REASON or "UNKNOWN" in text:
        return "unknown"
    if (
        "READINESS" in text
        or "STATESTORE_UNREADABLE" in text
        or "STATE_STORE_UNREADABLE" in text
    ):
        return "readiness_problem"
    if (
        "HALT" in text
        or "NEEDS_RECONCILIATION" in text
        or "UNKNOWN_BROKER_EXPOSURE" in text
        or "SAFETY" in text
    ):
        return "halt_or_safety_problem"
    if (
        "ACCOUNT_SNAPSHOT_STALE" in text
        or "MISSING" in text
        or "STALE" in text
        or "MARKET_CALENDAR" in text
        or "SESSION_DATA" in text
        or "CALENDAR" in text
    ):
        return "data_problem"
    if (
        "SKIP_IV" in text
        or "SKIP_VIX" in text
        or "DELTA" in text
        or "CREDIT" in text
        or "SPREAD" in text
        or "LIQUIDITY" in text
        or "EXISTING_POSITION" in text
        or "ORDER_INTENT" in text
        or "ALREADY_SIGNALED" in text
        or "BLACKOUT_DATE" in text
        or "ORDERREF" in text
        or "LOCK_STARVATION" in text
    ):
        return "strategy_filter_problem"
    if "NLV_DEGRADED" in text:
        return "data_problem"
    return "unknown"


def _current_time(now_provider: Callable[[], datetime] | None) -> datetime:
    now = now_provider() if now_provider is not None else datetime.now(NY_TZ)
    if not isinstance(now, datetime):
        raise ValueError("now_provider must return a datetime")
    return _to_ny(now)


def _session_date(value: Any, now: datetime) -> date:
    if value is None:
        return now.date()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return _to_ny(value).date()
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError("session_date must be a date, datetime, ISO date, or None")


def _iter_session_events(
    ledger_reader,
    session: date,
    errors: list[str],
    critical_errors: list[str],
) -> Iterable[dict[str, Any]]:
    paths = _ledger_paths(ledger_reader)
    if not paths:
        message = "ledger paths unavailable from ledger_reader"
        errors.append(message)
        critical_errors.append(message)
        return
    for path in paths:
        yield from _iter_session_events_from_path(path, session, errors, critical_errors)


def _iter_session_events_from_path(
    path: Path,
    session: date,
    errors: list[str],
    critical_errors: list[str],
) -> Iterable[dict[str, Any]]:
    try:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    message = f"invalid ledger JSON in {path} line {line_number}: {exc.msg}"
                    errors.append(message)
                    critical_errors.append(message)
                    continue
                if isinstance(event, dict) and _event_matches_session(
                    event, session, errors
                ):
                    yield event
    except OSError as exc:
        message = f"ledger unavailable for {path}: {exc}"
        errors.append(message)
        critical_errors.append(message)


def _ledger_paths(ledger_reader) -> list[Path]:
    paths = []
    for name in ("execution_ledger_path", "order_ledger_path", "execution_path", "order_path"):
        value = getattr(ledger_reader, name, None)
        if value is not None:
            paths.append(Path(value))
    return paths


def _event_matches_session(event: dict[str, Any], session: date, errors: list[str]) -> bool:
    timestamp = event.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        errors.append("ledger event missing timestamp")
        return False
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"ledger event malformed timestamp: {timestamp}")
        return False
    return _to_ny(parsed).date() == session


def _to_ny(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=NY_TZ)
    return value.astimezone(NY_TZ)


def _strategy_id(event: dict[str, Any]) -> str | None:
    strategy_id = event.get("strategy_id")
    if isinstance(strategy_id, str) and strategy_id:
        return strategy_id
    payload = event.get("payload")
    if isinstance(payload, dict):
        payload_strategy_id = payload.get("strategy_id")
        if isinstance(payload_strategy_id, str) and payload_strategy_id:
            return payload_strategy_id
    return None


def _skip_reason(event: dict[str, Any]) -> str:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return UNKNOWN_SKIP_REASON
    reason = payload.get("skip_reason")
    return str(reason) if reason else UNKNOWN_SKIP_REASON


def _top_reason(counts: Counter[str] | dict[str, int]) -> str | None:
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))[0][0]


def _ratio(generated: int, skipped: int) -> float | None:
    if skipped <= 0:
        return None
    return generated / skipped


def _call_readiness_provider(readiness_provider, errors: list[str]) -> Any:
    if readiness_provider is None:
        return None
    try:
        return readiness_provider()
    except Exception as exc:
        errors.append(f"readiness provider unavailable: {exc}")
        return None


def _readiness_records(
    state_store,
    strategy_ids: tuple[str, ...],
    errors: list[str],
) -> dict[str, dict[str, Any] | None]:
    records: dict[str, dict[str, Any] | None] = {}
    for strategy_id in strategy_ids:
        try:
            record = state_store.get_readiness(strategy_id)
        except Exception as exc:
            errors.append(f"readiness unavailable for {strategy_id}: {exc}")
            record = None
        records[strategy_id] = record if isinstance(record, dict) else None
    return records


def _readiness_passed(record: dict[str, Any]) -> bool | None:
    if "ready_for_entries" in record:
        return bool(record.get("ready_for_entries"))
    reason = record.get("reason")
    if reason is not None:
        return False
    return None


def _provider_values(snapshot: Any, strategy_id: str) -> dict[str, Any]:
    if snapshot is None:
        return {}
    return {
        "dirty_state": _mapping_attr(snapshot, "dirty_state_by_strategy", strategy_id),
        "calendar_expired": _mapping_attr(snapshot, "calendar_expired_by_strategy", strategy_id),
        "iv_baseline_available": _mapping_attr(
            snapshot,
            "iv_baseline_available_by_strategy",
            strategy_id,
        ),
    }


def _mapping_attr(snapshot: Any, name: str, strategy_id: str) -> Any:
    mapping = _snapshot_attr(snapshot, name)
    if isinstance(mapping, dict):
        return mapping.get(strategy_id)
    return None


def _snapshot_attr(snapshot: Any, name: str) -> Any:
    if snapshot is None:
        return None
    if isinstance(snapshot, dict):
        return snapshot.get(name)
    return getattr(snapshot, name, None)


def _state_counts_by_strategy(
    state_store,
    strategy_ids: tuple[str, ...],
    errors: list[str],
) -> dict[str, dict[str, int | None]]:
    counts = {
        strategy_id: {"active_intents_count": None, "open_positions_count": None}
        for strategy_id in strategy_ids
    }
    try:
        positions = _safe_list(state_store, "list_positions")
        for strategy_id in strategy_ids:
            counts[strategy_id]["open_positions_count"] = sum(
                1
                for position in positions
                if isinstance(position, dict)
                and position.get("strategy_id") == strategy_id
                and position.get("status") == "open"
            )
    except Exception as exc:
        errors.append(f"position counts unavailable: {exc}")
    try:
        intents = _safe_list(state_store, "list_order_intents") + _safe_list(
            state_store,
            "list_close_intents",
        )
        for strategy_id in strategy_ids:
            counts[strategy_id]["active_intents_count"] = sum(
                1
                for intent in intents
                if isinstance(intent, dict)
                and intent.get("strategy_id") == strategy_id
                and intent.get("status") in _ACTIVE_INTENT_STATUSES
            )
    except Exception as exc:
        errors.append(f"intent counts unavailable: {exc}")
    return counts


def _safe_list(target, method_name: str) -> list[dict[str, Any]]:
    method = getattr(target, method_name, None)
    if not callable(method):
        return []
    value = method()
    return value if isinstance(value, list) else []


def _load_halt_state(
    path: Path,
    errors: list[str],
    critical_errors: list[str],
) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        message = f"halt_state unavailable: {exc}"
        errors.append(message)
        critical_errors.append(message)
        return None
    if not isinstance(payload, dict):
        message = "halt_state is not a JSON object"
        errors.append(message)
        critical_errors.append(message)
        return None
    return payload


def _halt_active(
    halt_state: dict[str, Any] | None,
    readiness_snapshot: Any,
    strategy_ids: tuple[str, ...],
) -> bool:
    if halt_is_active(halt_state):
        return True
    halted = _snapshot_attr(readiness_snapshot, "halt_active_by_strategy")
    if isinstance(halted, dict):
        return any(bool(halted.get(strategy_id)) for strategy_id in strategy_ids)
    return False


def _account_snapshot_fresh(
    readiness_snapshot: Any,
    snapshots_dir: Path,
    now: datetime,
    errors: list[str],
) -> bool | None:
    from_provider = _snapshot_attr(readiness_snapshot, "account_snapshot_fresh")
    if from_provider is not None:
        return bool(from_provider)
    latest = _latest_snapshot_file(snapshots_dir, errors)
    if latest is None:
        return None
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"account snapshot unavailable: {exc}")
        return None
    timestamp = _snapshot_timestamp(payload)
    if timestamp is None:
        try:
            timestamp = datetime.fromtimestamp(latest.stat().st_mtime, timezone.utc)
        except OSError as exc:
            errors.append(f"account snapshot mtime unavailable: {exc}")
            return None
    return (_to_ny(now) - _to_ny(timestamp)).total_seconds() <= 15 * 60


def _latest_snapshot_file(snapshots_dir: Path, errors: list[str]) -> Path | None:
    try:
        if not snapshots_dir.exists():
            return None
        candidates = [
            path
            for path in snapshots_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".json"
        ]
    except OSError as exc:
        errors.append(f"snapshots unavailable: {exc}")
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _snapshot_timestamp(payload: Any) -> datetime | None:
    if not isinstance(payload, dict):
        return None
    for field in _SNAPSHOT_TIMESTAMP_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            continue
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
    return None


def _diagnostics(
    strategy_ids: tuple[str, ...],
    per_strategy: dict[str, dict[str, Any]],
    aggregate: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    top_reason = aggregate.get("top_skip_reason")
    category = classify_skip_reason(top_reason) if top_reason is not None else "unknown"
    notes: list[str] = []
    if readiness.get("halt_active") is True:
        notes.append("Active halt is present; investigate halt state before strategy tuning.")
    if top_reason is None:
        notes.append("No skipped signals found for the selected session.")
    else:
        notes.append(_note_for_top_reason(top_reason, category))
    for strategy_id in strategy_ids:
        summary = per_strategy[strategy_id]
        if summary["signals_generated"] == 0 and summary["signals_skipped"] == 0:
            notes.append(f"{strategy_id} has no signal evidence for this session.")
        elif summary["signals_generated"] == 0:
            notes.append(
                f"{strategy_id} has zero generated signals; inspect its top skip reason before tuning."
            )
    missing = readiness.get("missing_readiness_strategy_ids") or []
    if missing:
        notes.append(
            "Readiness is missing for: " + ", ".join(str(strategy_id) for strategy_id in missing)
        )
    return {
        "likely_blocker_category": category,
        "notes": notes,
    }


def _note_for_top_reason(reason: str, category: str) -> str:
    if category == "readiness_problem":
        return f"Top blocker is {reason}; investigate readiness cadence/staleness before tuning strategy."
    if category == "data_problem":
        return f"Top blocker is {reason}; investigate local data freshness/availability before changing thresholds."
    if category == "strategy_filter_problem":
        return f"Top blocker is {reason}; review filter evidence before any strategy threshold change."
    if category == "halt_or_safety_problem":
        return f"Top blocker is {reason}; resolve halt or reconciliation safety state before strategy work."
    return f"Top blocker is {reason}; improve skip reason logging before strategy tuning."


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
