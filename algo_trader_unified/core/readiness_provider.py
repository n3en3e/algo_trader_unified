"""Local-only HealthSnapshot provider for readiness checks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from algo_trader_unified.jobs.readiness import HealthSnapshot


_SNAPSHOT_TIMESTAMP_FIELDS = (
    "timestamp",
    "captured_at",
    "snapshot_at",
    "generated_at",
    "as_of",
    "created_at",
)


class DefaultHealthSnapshotProvider:
    """Build a HealthSnapshot from local state and local snapshot files only."""

    def __init__(
        self,
        *,
        state_store: Any,
        halt_state_path: str | Path,
        snapshots_dir: str | Path,
        max_staleness_minutes: int | float,
        strategy_ids: list[str],
    ) -> None:
        self.state_store = state_store
        self.halt_state_path = Path(halt_state_path)
        self.snapshots_dir = Path(snapshots_dir)
        self.max_staleness_minutes = max_staleness_minutes
        self.strategy_ids = tuple(strategy_ids)

    def __call__(self) -> HealthSnapshot:
        account_snapshot_fresh = _latest_snapshot_is_fresh(
            self.snapshots_dir,
            self.max_staleness_minutes,
        )
        halt_active_by_strategy = _halt_active_by_strategy(
            self.halt_state_path,
            self.strategy_ids,
        )
        readiness = _readiness_by_strategy(self.state_store, self.strategy_ids)
        state_store_readable = readiness is not None

        return HealthSnapshot(
            account_snapshot_fresh=account_snapshot_fresh,
            nlv_valid=account_snapshot_fresh,
            state_store_readable=state_store_readable,
            halt_active_by_strategy=halt_active_by_strategy,
            dirty_state_by_strategy={
                strategy_id: _readiness_bool(
                    readiness,
                    strategy_id,
                    "dirty_state",
                    default=True,
                )
                for strategy_id in self.strategy_ids
            },
            unknown_broker_exposure_by_strategy={
                strategy_id: _readiness_bool(
                    readiness,
                    strategy_id,
                    "unknown_broker_exposure",
                    default=True,
                )
                for strategy_id in self.strategy_ids
            },
            calendar_expired_by_strategy={
                strategy_id: _readiness_bool(
                    readiness,
                    strategy_id,
                    "calendar_expired",
                    default=True,
                )
                for strategy_id in self.strategy_ids
            },
            iv_baseline_available_by_strategy={
                strategy_id: _readiness_value(
                    readiness,
                    strategy_id,
                    "iv_baseline_available",
                    default=False,
                )
                for strategy_id in self.strategy_ids
            },
        )


def _readiness_by_strategy(
    state_store: Any,
    strategy_ids: tuple[str, ...],
) -> dict[str, dict[str, Any] | None] | None:
    records: dict[str, dict[str, Any] | None] = {}
    try:
        for strategy_id in strategy_ids:
            record = state_store.get_readiness(strategy_id)
            records[strategy_id] = record if isinstance(record, dict) else None
    except Exception:
        return None
    return records


def _readiness_bool(
    readiness: dict[str, dict[str, Any] | None] | None,
    strategy_id: str,
    key: str,
    *,
    default: bool,
) -> bool:
    value = _readiness_value(readiness, strategy_id, key, default=default)
    return bool(value)


def _readiness_value(
    readiness: dict[str, dict[str, Any] | None] | None,
    strategy_id: str,
    key: str,
    *,
    default: Any,
) -> Any:
    if readiness is None:
        return default
    record = readiness.get(strategy_id)
    if not isinstance(record, dict):
        return default
    return record.get(key, default)


def _halt_active_by_strategy(
    halt_state_path: Path,
    strategy_ids: tuple[str, ...],
) -> dict[str, bool]:
    try:
        if not halt_state_path.exists():
            return {strategy_id: False for strategy_id in strategy_ids}
        payload = json.loads(halt_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {strategy_id: True for strategy_id in strategy_ids}

    if not isinstance(payload, dict) or not _halt_is_active(payload):
        return {strategy_id: False for strategy_id in strategy_ids}

    scope = payload.get("scope")
    if scope == "account":
        return {strategy_id: True for strategy_id in strategy_ids}
    if scope == "strategy":
        halted_id = payload.get("id") or payload.get("scope_id")
        return {strategy_id: strategy_id == halted_id for strategy_id in strategy_ids}
    return {strategy_id: True for strategy_id in strategy_ids}


def _halt_is_active(payload: dict[str, Any]) -> bool:
    if payload.get("resumed") is True:
        return False
    return payload.get("tier") in {"soft", "hard"}


def _latest_snapshot_is_fresh(
    snapshots_dir: Path,
    max_staleness_minutes: int | float,
) -> bool:
    latest = _latest_snapshot_file(snapshots_dir)
    if latest is None:
        return False

    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    timestamp = _timestamp_from_payload(payload)
    if timestamp is None:
        timestamp = datetime.fromtimestamp(latest.stat().st_mtime, timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - timestamp).total_seconds()
    return age_seconds <= float(max_staleness_minutes) * 60


def _latest_snapshot_file(snapshots_dir: Path) -> Path | None:
    try:
        candidates = [
            path
            for path in snapshots_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".json"
        ]
    except OSError:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _timestamp_from_payload(payload: Any) -> datetime | None:
    if not isinstance(payload, dict):
        return None
    for field in _SNAPSHOT_TIMESTAMP_FIELDS:
        value = payload.get(field)
        parsed = _parse_iso_datetime(value)
        if parsed is not None:
            return parsed
    return None


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
