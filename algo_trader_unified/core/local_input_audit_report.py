"""Read-only local IV/VIX/calendar/readiness input audit report."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo


NY_TZ = ZoneInfo("America/New_York")
_TIMESTAMP_FIELDS = ("timestamp", "captured_at", "snapshot_at", "generated_at", "as_of", "created_at")
_ACCOUNT_SNAPSHOT_EXCLUDES = {
    "iv.json",
    "iv_rank.json",
    "iv_snapshot.json",
    "vix.json",
    "vix_snapshot.json",
    "market_calendar.json",
    "calendar.json",
}


def build_local_input_audit_report(
    *,
    strategy_ids,
    state_store,
    readiness_provider,
    snapshots_dir,
    halt_state_path,
    iv_store=None,
    vix_snapshot_path=None,
    market_calendar_path=None,
    now_provider: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe local input audit without mutating trading state."""

    now = _current_time(now_provider)
    ids = [str(strategy_id) for strategy_id in strategy_ids]
    errors: list[str] = []
    warnings: list[str] = []

    readiness_snapshot = _call_readiness_provider(readiness_provider, warnings)
    readiness_records = _readiness_records(state_store, ids, warnings)
    halt_by_strategy = _halt_active_by_strategy(Path(halt_state_path), ids, warnings)

    iv_rank = _iv_rank_report(iv_store, ids, warnings)
    vix = _single_snapshot_report(
        vix_snapshot_path,
        name="VIX",
        value_fields=("vix", "value", "close", "last"),
        warnings=warnings,
    )
    market_calendar = _calendar_report(
        market_calendar_path,
        ids,
        readiness_snapshot,
        readiness_records,
        warnings,
    )
    account_snapshot = _account_snapshot_report(
        Path(snapshots_dir),
        readiness_snapshot,
        warnings,
    )
    readiness = _readiness_snapshot_report(
        ids,
        readiness_snapshot,
        readiness_records,
        halt_by_strategy,
    )

    per_strategy = _per_strategy_report(
        ids=ids,
        readiness_records=readiness_records,
        readiness_snapshot=readiness_snapshot,
        iv_rank=iv_rank,
        market_calendar=market_calendar,
        account_snapshot=account_snapshot,
        readiness=readiness,
    )
    aggregate = _aggregate_report(iv_rank, vix, market_calendar, account_snapshot, readiness)

    report = {
        "dry_run": True,
        "local_input_audit_report": True,
        "generated_at": now.isoformat(),
        "strategy_ids": ids,
        "inputs_checked": {
            "iv_rank": True,
            "vix": True,
            "market_calendar": True,
            "account_snapshot": True,
            "readiness_snapshot": True,
        },
        "per_strategy": per_strategy,
        "iv_rank": iv_rank,
        "vix": vix,
        "market_calendar": market_calendar,
        "account_snapshot": account_snapshot,
        "readiness_snapshot": readiness,
        "aggregate": aggregate,
        "recommendations": {
            "ordered_next_steps": _ordered_next_steps(
                iv_rank=iv_rank,
                vix=vix,
                market_calendar=market_calendar,
                account_snapshot=account_snapshot,
                readiness=readiness,
            )
        },
        "safety": {
            "broker_calls_enabled": False,
            "market_data_enabled": False,
            "external_fetch_enabled": False,
            "paper_live_orders_enabled": False,
            "strategy_changes_enabled": False,
        },
        "success": not errors,
        "errors": errors,
        "warnings": sorted(set(warnings)),
    }
    return _json_safe(report)


def _current_time(now_provider: Callable[[], datetime] | None) -> datetime:
    now = now_provider() if now_provider is not None else datetime.now(NY_TZ)
    if not isinstance(now, datetime):
        raise ValueError("now_provider must return a datetime")
    if now.tzinfo is None:
        return now.replace(tzinfo=NY_TZ)
    return now.astimezone(NY_TZ)


def _call_readiness_provider(readiness_provider, warnings: list[str]) -> Any:
    if readiness_provider is None:
        warnings.append("readiness provider unavailable")
        return None
    try:
        return readiness_provider()
    except Exception as exc:
        warnings.append(f"readiness provider unavailable: {exc}")
        return None


def _readiness_records(
    state_store,
    strategy_ids: list[str],
    warnings: list[str],
) -> dict[str, dict[str, Any] | None]:
    records: dict[str, dict[str, Any] | None] = {}
    for strategy_id in strategy_ids:
        try:
            record = state_store.get_readiness(strategy_id)
        except Exception as exc:
            warnings.append(f"readiness unavailable for {strategy_id}: {exc}")
            record = None
        records[strategy_id] = record if isinstance(record, dict) else None
    return records


def _iv_rank_report(iv_store: Any, strategy_ids: list[str], warnings: list[str]) -> dict[str, Any]:
    base = {
        "available_by_strategy": {strategy_id: None for strategy_id in strategy_ids},
        "latest_timestamp_by_strategy": {strategy_id: None for strategy_id in strategy_ids},
        "stale_by_strategy": {strategy_id: None for strategy_id in strategy_ids},
        "source_description": "local iv_store" if iv_store is not None else "no local IV rank source configured",
        "status": "unavailable",
    }
    if iv_store is None:
        warnings.append("IV rank unavailable locally")
        return base

    payload, status, warning = _coerce_payload(iv_store, "IV rank")
    if warning:
        warnings.append(warning)
    if status != "ok":
        base["status"] = status
        return base
    if not isinstance(payload, dict):
        warnings.append("IV rank source is not a JSON object")
        base["status"] = "invalid"
        return base

    source = str(iv_store) if isinstance(iv_store, (str, Path)) else "local iv_store"
    available = _strategy_mapping(payload, strategy_ids, "available_by_strategy", "available")
    timestamps = _timestamp_mapping(payload, strategy_ids, "latest_timestamp_by_strategy")
    stale = _strategy_mapping(payload, strategy_ids, "stale_by_strategy", "stale")
    if all(value is None for value in available.values()):
        baseline = _strategy_mapping(
            payload,
            strategy_ids,
            "iv_baseline_available_by_strategy",
            "iv_baseline_available",
        )
        available = baseline
    for strategy_id in strategy_ids:
        if available[strategy_id] is None:
            available[strategy_id] = timestamps[strategy_id] is not None
    parsed_timestamps: dict[str, str | None] = {}
    invalid_timestamp = False
    for strategy_id, value in timestamps.items():
        parsed, warning = _parse_timestamp(value)
        if warning:
            warnings.append(f"IV rank timestamp malformed for {strategy_id}: {value}")
            invalid_timestamp = True
        parsed_timestamps[strategy_id] = parsed.isoformat() if parsed is not None else None

    base.update(
        {
            "available_by_strategy": {key: _bool_or_none(value) for key, value in available.items()},
            "latest_timestamp_by_strategy": parsed_timestamps,
            "stale_by_strategy": {key: _bool_or_none(value) for key, value in stale.items()},
            "source_description": source,
            "status": _status_from_strategy_maps(available, stale, invalid_timestamp),
        }
    )
    return base


def _single_snapshot_report(
    path_value: Any,
    *,
    name: str,
    value_fields: tuple[str, ...],
    warnings: list[str],
) -> dict[str, Any]:
    base = {
        "available": False,
        "latest_timestamp": None,
        "stale": None,
        "source_description": f"no local {name} snapshot path configured",
        "status": "unavailable",
    }
    if path_value is None:
        warnings.append(f"{name} input unavailable locally")
        return base
    path = Path(path_value)
    base["source_description"] = str(path)
    payload, status, warning = _load_json_file(path, name)
    if warning:
        warnings.append(warning)
    if status != "ok":
        base["status"] = status
        return base
    if not isinstance(payload, dict):
        warnings.append(f"{name} snapshot is not a JSON object")
        base["status"] = "invalid"
        return base
    timestamp, invalid_timestamp = _payload_timestamp(payload, warnings, name)
    available = payload.get("available")
    if available is None:
        available = any(payload.get(field) is not None for field in value_fields)
    stale = _bool_or_none(payload.get("stale"))
    base.update(
        {
            "available": bool(available),
            "latest_timestamp": timestamp.isoformat() if timestamp is not None else None,
            "stale": stale,
            "status": _status_from_available(bool(available), stale, invalid_timestamp),
        }
    )
    return base


def _calendar_report(
    path_value: Any,
    strategy_ids: list[str],
    readiness_snapshot: Any,
    readiness_records: dict[str, dict[str, Any] | None],
    warnings: list[str],
) -> dict[str, Any]:
    expired_from_readiness = {
        strategy_id: _first_present(
            _mapping_attr(readiness_snapshot, "calendar_expired_by_strategy", strategy_id),
            _record_value(readiness_records.get(strategy_id), "calendar_expired"),
        )
        for strategy_id in strategy_ids
    }
    base = {
        "available": any(value is not None for value in expired_from_readiness.values()),
        "session_available": None,
        "calendar_expired_by_strategy": {
            strategy_id: _bool_or_none(expired_from_readiness[strategy_id])
            for strategy_id in strategy_ids
        },
        "source_description": "readiness snapshot/state readiness fields",
        "status": "ok" if any(value is not None for value in expired_from_readiness.values()) else "unavailable",
    }
    if path_value is None:
        if base["status"] == "unavailable":
            warnings.append("market calendar unavailable locally")
        return base

    path = Path(path_value)
    payload, status, warning = _load_json_file(path, "market calendar")
    if warning:
        warnings.append(warning)
    base["source_description"] = str(path)
    if status != "ok":
        base["available"] = False
        base["status"] = status
        return base
    if not isinstance(payload, dict):
        warnings.append("market calendar snapshot is not a JSON object")
        base["available"] = False
        base["status"] = "invalid"
        return base
    parsed_map = _strategy_mapping(
        payload,
        strategy_ids,
        "calendar_expired_by_strategy",
        "calendar_expired",
    )
    for strategy_id, value in parsed_map.items():
        if value is not None:
            base["calendar_expired_by_strategy"][strategy_id] = _bool_or_none(value)
    session_available = payload.get("session_available")
    base["session_available"] = _bool_or_none(session_available)
    base["available"] = session_available is not False
    if any(value is True for value in base["calendar_expired_by_strategy"].values()):
        base["status"] = "stale"
    elif base["available"]:
        base["status"] = "ok"
    else:
        base["status"] = "missing"
    return base


def _account_snapshot_report(
    snapshots_dir: Path,
    readiness_snapshot: Any,
    warnings: list[str],
) -> dict[str, Any]:
    fresh = _snapshot_attr(readiness_snapshot, "account_snapshot_fresh")
    nlv_valid = _snapshot_attr(readiness_snapshot, "nlv_valid")
    base = {
        "available": False,
        "account_snapshot_fresh": _bool_or_none(fresh),
        "nlv_valid": _bool_or_none(nlv_valid),
        "latest_timestamp": None,
        "status": "unavailable",
    }
    latest = _latest_account_snapshot_file(snapshots_dir, warnings)
    if latest is None:
        if fresh is not None or nlv_valid is not None:
            base["available"] = True
            base["status"] = _account_status(base["account_snapshot_fresh"], base["nlv_valid"], False)
        else:
            warnings.append("account snapshot unavailable locally")
            base["status"] = "missing"
        return base
    payload, status, warning = _load_json_file(latest, "account snapshot")
    if warning:
        warnings.append(warning)
    if status != "ok":
        base["status"] = status
        return base
    if not isinstance(payload, dict):
        warnings.append("account snapshot is not a JSON object")
        base["status"] = "invalid"
        return base
    timestamp, invalid_timestamp = _payload_timestamp(payload, warnings, "account snapshot")
    base["available"] = True
    base["latest_timestamp"] = timestamp.isoformat() if timestamp is not None else None
    if base["account_snapshot_fresh"] is None:
        base["account_snapshot_fresh"] = _bool_or_none(payload.get("account_snapshot_fresh"))
    if base["account_snapshot_fresh"] is None and payload.get("stale") is not None:
        base["account_snapshot_fresh"] = not bool(payload.get("stale"))
    if base["nlv_valid"] is None:
        base["nlv_valid"] = _nlv_valid(payload)
    if invalid_timestamp:
        base["status"] = "invalid"
    else:
        base["status"] = _account_status(base["account_snapshot_fresh"], base["nlv_valid"], False)
    return base


def _readiness_snapshot_report(
    strategy_ids: list[str],
    readiness_snapshot: Any,
    readiness_records: dict[str, dict[str, Any] | None],
    halt_by_strategy: dict[str, bool],
) -> dict[str, Any]:
    missing = sorted(
        strategy_id
        for strategy_id in strategy_ids
        if readiness_records.get(strategy_id) is None
        and not _has_provider_readiness(readiness_snapshot, strategy_id)
    )
    dirty = {
        strategy_id: _first_present(
            _mapping_attr(readiness_snapshot, "dirty_state_by_strategy", strategy_id),
            _record_value(readiness_records.get(strategy_id), "dirty_state"),
        )
        for strategy_id in strategy_ids
    }
    halted = {
        strategy_id: _first_present(
            _mapping_attr(readiness_snapshot, "halt_active_by_strategy", strategy_id),
            _record_value(readiness_records.get(strategy_id), "halt_active"),
            halt_by_strategy.get(strategy_id),
        )
        for strategy_id in strategy_ids
    }
    failed = any(_readiness_passed(readiness_records.get(strategy_id)) is False for strategy_id in strategy_ids)
    return {
        "available": not missing,
        "missing_readiness_strategy_ids": missing,
        "dirty_state_by_strategy": {key: _bool_or_none(value) for key, value in dirty.items()},
        "halt_active_by_strategy": {key: _bool_or_none(value) for key, value in halted.items()},
        "status": "missing" if missing else "failed" if failed else "ok",
    }


def _per_strategy_report(
    *,
    ids: list[str],
    readiness_records: dict[str, dict[str, Any] | None],
    readiness_snapshot: Any,
    iv_rank: dict[str, Any],
    market_calendar: dict[str, Any],
    account_snapshot: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    per_strategy: dict[str, dict[str, Any]] = {}
    iv_available = iv_rank.get("available_by_strategy") if isinstance(iv_rank.get("available_by_strategy"), dict) else {}
    calendar_expired = (
        market_calendar.get("calendar_expired_by_strategy")
        if isinstance(market_calendar.get("calendar_expired_by_strategy"), dict)
        else {}
    )
    for strategy_id in ids:
        record = readiness_records.get(strategy_id)
        readiness_available = strategy_id not in readiness.get("missing_readiness_strategy_ids", [])
        readiness_passed = _readiness_passed(record)
        iv_baseline_available = _first_present(
            _record_value(record, "iv_baseline_available"),
            _mapping_attr(readiness_snapshot, "iv_baseline_available_by_strategy", strategy_id),
            iv_available.get(strategy_id),
        )
        blockers = _input_blockers(
            strategy_id=strategy_id,
            readiness_available=readiness_available,
            readiness_passed=readiness_passed,
            iv_available=iv_available.get(strategy_id),
            calendar_expired=calendar_expired.get(strategy_id),
            account_snapshot=account_snapshot,
            readiness=readiness,
        )
        per_strategy[strategy_id] = {
            "readiness_available": readiness_available,
            "readiness_passed": readiness_passed,
            "iv_baseline_available": _bool_or_none(iv_baseline_available),
            "calendar_expired": _bool_or_none(calendar_expired.get(strategy_id)),
            "input_blockers": blockers,
            "likely_input_issue": sorted(blockers)[0] if blockers else None,
        }
    return per_strategy


def _input_blockers(
    *,
    strategy_id: str,
    readiness_available: bool,
    readiness_passed: bool | None,
    iv_available: Any,
    calendar_expired: Any,
    account_snapshot: dict[str, Any],
    readiness: dict[str, Any],
) -> list[str]:
    blockers: set[str] = set()
    if iv_available is not True:
        blockers.add("iv_rank")
    if calendar_expired is True:
        blockers.add("market_calendar")
    if account_snapshot.get("available") is not True:
        blockers.add("account_snapshot")
    if account_snapshot.get("account_snapshot_fresh") is False:
        blockers.add("account_snapshot_stale")
    if account_snapshot.get("nlv_valid") is False:
        blockers.add("account_snapshot_invalid")
    if not readiness_available:
        blockers.add("readiness_snapshot")
    if readiness_passed is False:
        blockers.add("readiness_failure")
    halted = readiness.get("halt_active_by_strategy")
    if isinstance(halted, dict) and halted.get(strategy_id) is True:
        blockers.add("halt_or_safety_block")
    return sorted(blockers)


def _aggregate_report(*sections: dict[str, Any]) -> dict[str, Any]:
    counts = Counter()
    for section in sections:
        status = section.get("status")
        if status == "missing":
            counts["missing"] += 1
        elif status == "stale":
            counts["stale"] += 1
        elif status in {"invalid", "failed"}:
            counts["blocking"] += 1
        elif status == "unavailable":
            counts["missing"] += 1
    issues = Counter()
    names = ("iv_rank", "vix", "market_calendar", "account_snapshot", "readiness_snapshot")
    for name, section in zip(names, sections):
        if section.get("status") not in {"ok", None}:
            issues[name] += 1
    return {
        "blocking_input_count": counts["blocking"],
        "stale_input_count": counts["stale"],
        "missing_input_count": counts["missing"],
        "dominant_input_issue": _top_counter_key(issues),
    }


def _ordered_next_steps(
    *,
    iv_rank: dict[str, Any],
    vix: dict[str, Any],
    market_calendar: dict[str, Any],
    account_snapshot: dict[str, Any],
    readiness: dict[str, Any],
) -> list[str]:
    candidates: list[str] = []
    if iv_rank.get("status") in {"missing", "unavailable", "invalid", "stale"}:
        candidates.append("IV rank unavailable locally; inspect IV store capture before tuning IV thresholds.")
    if vix.get("status") in {"missing", "unavailable", "invalid", "stale"}:
        candidates.append("VIX input unavailable locally; verify local VIX snapshot source before changing VIX gate.")
    if market_calendar.get("status") in {"missing", "unavailable", "invalid", "stale"}:
        candidates.append("Calendar expired for S01/S02; refresh local market calendar before interpreting skips.")
    if account_snapshot.get("status") in {"missing", "unavailable", "invalid", "stale"}:
        candidates.append("Account snapshot stale; fix snapshot cadence before evaluating signal quality.")
    for strategy_id in readiness.get("missing_readiness_strategy_ids") or []:
        candidates.append(f"Readiness missing for {strategy_id}; inspect market-open readiness output.")
    return _dedupe(candidates)


def _coerce_payload(source: Any, name: str) -> tuple[Any, str, str | None]:
    if isinstance(source, dict):
        return source, "ok", None
    if isinstance(source, (str, Path)):
        return _load_json_file(Path(source), name)
    for method_name in ("snapshot", "read_snapshot", "latest_snapshot", "get_snapshot"):
        method = getattr(source, method_name, None)
        if callable(method):
            try:
                return method(), "ok", None
            except Exception as exc:
                return None, "unavailable", f"{name} source unavailable: {exc}"
    return None, "unavailable", f"{name} source unavailable locally"


def _load_json_file(path: Path, name: str) -> tuple[Any, str, str | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, "missing", f"{name} file missing: {path}"
    except OSError as exc:
        return None, "unavailable", f"{name} file unavailable: {exc}"
    try:
        return json.loads(text), "ok", None
    except json.JSONDecodeError as exc:
        return None, "invalid", f"{name} JSON invalid: {exc.msg}"


def _strategy_mapping(
    payload: dict[str, Any],
    strategy_ids: list[str],
    mapping_key: str,
    record_key: str,
) -> dict[str, Any]:
    result = {strategy_id: None for strategy_id in strategy_ids}
    mapping = payload.get(mapping_key)
    if isinstance(mapping, dict):
        for strategy_id in strategy_ids:
            result[strategy_id] = mapping.get(strategy_id)
        return result
    strategies = payload.get("strategies")
    if isinstance(strategies, dict):
        for strategy_id in strategy_ids:
            record = strategies.get(strategy_id)
            if isinstance(record, dict):
                result[strategy_id] = record.get(record_key)
        return result
    for strategy_id in strategy_ids:
        record = payload.get(strategy_id)
        if isinstance(record, dict):
            result[strategy_id] = record.get(record_key)
    return result


def _timestamp_mapping(
    payload: dict[str, Any],
    strategy_ids: list[str],
    mapping_key: str,
) -> dict[str, Any]:
    result = {strategy_id: None for strategy_id in strategy_ids}
    mapping = payload.get(mapping_key)
    if isinstance(mapping, dict):
        for strategy_id in strategy_ids:
            result[strategy_id] = mapping.get(strategy_id)
        return result
    strategies = payload.get("strategies")
    if isinstance(strategies, dict):
        for strategy_id in strategy_ids:
            record = strategies.get(strategy_id)
            if isinstance(record, dict):
                result[strategy_id] = _first_timestamp_value(record)
        return result
    for strategy_id in strategy_ids:
        record = payload.get(strategy_id)
        if isinstance(record, dict):
            result[strategy_id] = _first_timestamp_value(record)
    shared = _first_timestamp_value(payload)
    if shared is not None:
        for strategy_id in strategy_ids:
            result[strategy_id] = result[strategy_id] or shared
    return result


def _first_timestamp_value(payload: dict[str, Any]) -> Any:
    for field in _TIMESTAMP_FIELDS:
        value = payload.get(field)
        if value is not None:
            return value
    return None


def _payload_timestamp(
    payload: dict[str, Any],
    warnings: list[str],
    name: str,
) -> tuple[datetime | None, bool]:
    value = _first_timestamp_value(payload)
    parsed, warning = _parse_timestamp(value)
    if warning:
        warnings.append(f"{name} timestamp malformed: {value}")
    return parsed, warning is not None


def _parse_timestamp(value: Any) -> tuple[datetime | None, str | None]:
    if value is None:
        return None, None
    if not isinstance(value, str) or not value:
        return None, "malformed timestamp"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None, "malformed timestamp"
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=NY_TZ), None
    return parsed.astimezone(NY_TZ), None


def _latest_account_snapshot_file(snapshots_dir: Path, warnings: list[str]) -> Path | None:
    try:
        candidates = [
            path
            for path in snapshots_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".json"
            and path.name not in _ACCOUNT_SNAPSHOT_EXCLUDES
        ]
    except FileNotFoundError:
        return None
    except OSError as exc:
        warnings.append(f"snapshots directory unavailable: {exc}")
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _halt_active_by_strategy(
    halt_state_path: Path,
    strategy_ids: list[str],
    warnings: list[str],
) -> dict[str, bool]:
    try:
        payload = json.loads(halt_state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {strategy_id: False for strategy_id in strategy_ids}
    except json.JSONDecodeError as exc:
        warnings.append(f"halt_state JSON invalid: {exc.msg}")
        return {strategy_id: True for strategy_id in strategy_ids}
    except OSError as exc:
        warnings.append(f"halt_state unavailable: {exc}")
        return {strategy_id: True for strategy_id in strategy_ids}
    if not isinstance(payload, dict) or payload.get("resumed") is True:
        return {strategy_id: False for strategy_id in strategy_ids}
    if payload.get("tier") not in {"soft", "hard"}:
        return {strategy_id: False for strategy_id in strategy_ids}
    if payload.get("scope") == "strategy":
        halted_id = payload.get("id") or payload.get("scope_id")
        return {strategy_id: strategy_id == halted_id for strategy_id in strategy_ids}
    return {strategy_id: True for strategy_id in strategy_ids}


def _status_from_strategy_maps(
    available: dict[str, Any],
    stale: dict[str, Any],
    invalid_timestamp: bool,
) -> str:
    if invalid_timestamp:
        return "invalid"
    if any(value is True for value in stale.values()):
        return "stale"
    if any(value is True for value in available.values()):
        return "ok"
    if all(value is False for value in available.values()):
        return "missing"
    return "unavailable"


def _status_from_available(available: bool, stale: bool | None, invalid_timestamp: bool) -> str:
    if invalid_timestamp:
        return "invalid"
    if not available:
        return "missing"
    if stale is True:
        return "stale"
    return "ok"


def _account_status(fresh: bool | None, nlv_valid: bool | None, invalid: bool) -> str:
    if invalid or nlv_valid is False:
        return "invalid"
    if fresh is None:
        return "unavailable"
    if fresh is False:
        return "stale"
    return "ok"


def _nlv_valid(payload: dict[str, Any]) -> bool | None:
    for key in ("nlv_valid", "net_liquidation_valid"):
        if key in payload:
            return bool(payload.get(key))
    for key in ("nlv", "net_liquidation", "net_liquidation_value"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False
    return None


def _readiness_passed(record: dict[str, Any] | None) -> bool | None:
    if not isinstance(record, dict):
        return None
    if "ready_for_entries" in record:
        return bool(record.get("ready_for_entries"))
    if record.get("reason") is not None:
        return False
    return None


def _has_provider_readiness(snapshot: Any, strategy_id: str) -> bool:
    for name in (
        "dirty_state_by_strategy",
        "halt_active_by_strategy",
        "calendar_expired_by_strategy",
        "iv_baseline_available_by_strategy",
    ):
        mapping = _snapshot_attr(snapshot, name)
        if isinstance(mapping, dict) and strategy_id in mapping:
            return True
    return False


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


def _record_value(record: dict[str, Any] | None, key: str) -> Any:
    if isinstance(record, dict):
        return record.get(key)
    return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _top_counter_key(counts: Counter[str]) -> str | None:
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))[0][0]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


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
