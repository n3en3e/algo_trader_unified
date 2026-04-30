"""Append-only ledger writer."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from algo_trader_unified.core.ledger_paths import (
    EXECUTION_LEDGER_FILENAME,
    LEDGER_DIR_RELATIVE_PATH,
    ORDER_LEDGER_FILENAME,
)


ORDER_LEDGER_EVENTS = {
    "ORDER_INTENT_CREATED",
    "ORDER_INTENT_EXPIRED",
    "ORDER_INTENT_CANCELLED",
    "ORDER_SUBMITTED",
    "ORDER_CONFIRMED",
    "ORDER_CANCEL_REQUESTED",
    "ORDER_CANCEL_CONFIRMED",
    "FILL_CONFIRMED",
}

EXECUTION_LEDGER_EVENTS = {
    "SIGNAL_GENERATED",
    "SIGNAL_SKIPPED",
    "OPPORTUNITY_SCORED",
    "ALERT_SENT",
    "MANUAL_STATUS_UPDATED",
    "POSITION_OPENED",
    "POSITION_ADJUSTED",
    "POSITION_CLOSED",
    "POSITION_ADOPTED_FROM_LEGACY",
    "RECONCILIATION_FAILED",
    "HALT_TRIGGERED",
    "HALT_RESUMED",
}

KNOWN_EVENT_TYPES = ORDER_LEDGER_EVENTS | EXECUTION_LEDGER_EVENTS


class LedgerInitError(RuntimeError):
    """Raised when ledger files cannot be initialized."""


class LedgerValidationError(ValueError):
    """Raised when a ledger event envelope is invalid."""


class LedgerEventId(str):
    """String event id with legacy ``.event_id`` access."""

    @property
    def event_id(self) -> str:
        return str(self)


@dataclass(frozen=True)
class LedgerEvent:
    event_id: str
    event_type: str
    timestamp: str
    strategy_id: str
    execution_mode: str
    source_module: str
    position_id: str | None
    opportunity_id: str | None
    payload: dict[str, Any] = field(default_factory=dict)


class LedgerAppender:
    """Append-only JSONL appender with one lock per ledger file."""

    def __init__(self, root_dir: str | Path = ".") -> None:
        self.root_dir = Path(root_dir)
        self.ledger_dir = self.root_dir / LEDGER_DIR_RELATIVE_PATH
        self.order_path = self.ledger_dir / ORDER_LEDGER_FILENAME
        self.execution_path = self.ledger_dir / EXECUTION_LEDGER_FILENAME
        self._locks = {
            self.order_path: threading.Lock(),
            self.execution_path: threading.Lock(),
        }
        self._initialize()

    def _initialize(self) -> None:
        try:
            self.ledger_dir.mkdir(parents=True, exist_ok=True)
            for path in (self.order_path, self.execution_path):
                with path.open("a", encoding="utf-8"):
                    pass
        except OSError as exc:
            raise LedgerInitError(f"Unable to initialize ledger at {self.ledger_dir}: {exc}") from exc

    def path_for_event_type(self, event_type: str) -> Path:
        if event_type in ORDER_LEDGER_EVENTS:
            return self.order_path
        if event_type in EXECUTION_LEDGER_EVENTS:
            return self.execution_path
        raise LedgerValidationError(f"Unknown ledger event_type: {event_type}")

    def append(
        self,
        *,
        event_type: str,
        strategy_id: str,
        execution_mode: str,
        source_module: str,
        payload: dict[str, Any],
        position_id: str | None = None,
        opportunity_id: str | None = None,
        event_id: str | None = None,
        timestamp: str | None = None,
        expected_ledger: str | None = None,
    ) -> LedgerEventId:
        path = self.path_for_event_type(event_type)
        if expected_ledger is not None and path.name != expected_ledger:
            raise LedgerValidationError(
                f"{event_type} routes to {path.name}, not {expected_ledger}"
            )
        event = LedgerEvent(
            event_id=event_id or f"evt_{uuid4().hex}",
            event_type=event_type,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            strategy_id=strategy_id,
            execution_mode=execution_mode,
            source_module=source_module,
            position_id=position_id,
            opportunity_id=opportunity_id,
            payload=payload,
        )
        self._validate_event(event)
        line = json.dumps(asdict(event), separators=(",", ":"), sort_keys=True)
        with self._locks[path]:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return LedgerEventId(event.event_id)

    def _validate_event(self, event: LedgerEvent) -> None:
        if event.event_type not in KNOWN_EVENT_TYPES:
            raise LedgerValidationError(f"Unknown ledger event_type: {event.event_type}")
        required_string_fields = {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "strategy_id": event.strategy_id,
            "execution_mode": event.execution_mode,
            "source_module": event.source_module,
        }
        for name, value in required_string_fields.items():
            if not isinstance(value, str) or not value:
                raise LedgerValidationError(f"Ledger field {name} is required")
        if event.position_id is not None and not isinstance(event.position_id, str):
            raise LedgerValidationError("position_id must be a string or None")
        if event.opportunity_id is not None and not isinstance(event.opportunity_id, str):
            raise LedgerValidationError("opportunity_id must be a string or None")
        if not isinstance(event.payload, dict):
            raise LedgerValidationError("payload must be a dict")
