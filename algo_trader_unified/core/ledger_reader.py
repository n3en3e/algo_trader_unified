"""Read-only JSONL ledger queries."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from algo_trader_unified.core.ledger_paths import (
    EXECUTION_LEDGER_RELATIVE_PATH,
    ORDER_LEDGER_RELATIVE_PATH,
)


class LedgerReadError(RuntimeError):
    """Raised when a ledger file cannot be safely read."""


class LedgerReader:
    """Read-only query surface for execution and order ledger files."""

    def __init__(
        self,
        *,
        execution_ledger_path: str | Path,
        order_ledger_path: str | Path,
    ) -> None:
        self.execution_ledger_path = Path(execution_ledger_path)
        self.order_ledger_path = Path(order_ledger_path)

    @classmethod
    def from_root(cls, root_dir: str | Path = ".") -> "LedgerReader":
        root = Path(root_dir)
        return cls(
            execution_ledger_path=root / EXECUTION_LEDGER_RELATIVE_PATH,
            order_ledger_path=root / ORDER_LEDGER_RELATIVE_PATH,
        )

    def read_events(self, ledger_name: str | None = None) -> list[dict]:
        events: list[dict] = []
        for path in self._paths_for_ledger_name(ledger_name):
            events.extend(self._read_jsonl(path))
        return events

    def read_today(
        self,
        strategy_id: str,
        event_type: str,
        now,
        timezone: str = "America/New_York",
    ) -> list[dict]:
        tz = ZoneInfo(timezone)
        today = self._to_local_datetime(now, tz).date()
        matches = []
        for event in self.read_events("execution"):
            if event.get("event_type") != event_type:
                continue
            if event.get("strategy_id") != strategy_id:
                continue
            timestamp = event.get("timestamp")
            if not isinstance(timestamp, str):
                continue
            event_datetime = self._parse_event_timestamp(timestamp, tz)
            if event_datetime.date() == today:
                matches.append(event)
        return matches

    def _paths_for_ledger_name(self, ledger_name: str | None) -> tuple[Path, ...]:
        if ledger_name is None:
            return (self.execution_ledger_path, self.order_ledger_path)
        normalized = ledger_name.lower()
        if normalized in {"execution", "execution_ledger", "execution_ledger.jsonl"}:
            return (self.execution_ledger_path,)
        if normalized in {"order", "order_ledger", "order_ledger.jsonl"}:
            return (self.order_ledger_path,)
        raise LedgerReadError(f"Unknown ledger_name: {ledger_name}")

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        events: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise LedgerReadError(
                        f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
                    ) from exc
                if isinstance(event, dict):
                    events.append(event)
        return events

    def _parse_event_timestamp(self, timestamp: str, tz: ZoneInfo) -> datetime:
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise LedgerReadError(f"Invalid ledger timestamp: {timestamp}") from exc
        return self._to_local_datetime(parsed, tz)

    def _to_local_datetime(self, value, tz: ZoneInfo) -> datetime:
        if not isinstance(value, datetime):
            raise LedgerReadError("now must be a datetime")
        if value.tzinfo is None:
            return value.replace(tzinfo=tz)
        return value.astimezone(tz)
