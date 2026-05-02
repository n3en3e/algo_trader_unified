"""JSON materialized StateStore with atomic writes.

Always acquire strategy_state_locks[strategy_id] before state_store_write_lock
when both are needed. Never reverse this order.

strategy_state_locks uses threading.RLock. This is intentional — VolOrderManager
holds the strategy lock while calling engine lifecycle methods such as
mark_position_pending_close and record_close, which re-enter the same lock. Any
new code that holds this lock must be safe for RLock re-entrancy. Do not change
to threading.Lock without auditing all nested lock acquisition paths.
"""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

from algo_trader_unified.config.portfolio import STRATEGY_IDS, S02_VOL_ENHANCED
from algo_trader_unified.core.validation import validate_numeric_field


CURRENT_SCHEMA_VERSION = 1
S02_LEGACY_READINESS_FIELDS = {
    "standard_strangle_clean_days": 0,
    "last_clean_day_date": None,
    "last_reconciliation_check": None,
    "0dte_jobs_registered": False,
}


class StateStoreCorruptError(RuntimeError):
    """Raised when StateStore cannot be trusted."""


class OrderIntentTransitionError(ValueError):
    """Raised when an order intent lifecycle transition is invalid."""


class CloseIntentTransitionError(ValueError):
    """Raised when a close intent lifecycle transition is invalid."""


ACTIVE_CLOSE_INTENT_STATUSES = {"created", "submitted"}


class PositionBook(dict):
    """Dict-backed position collection with legacy list-style test compatibility."""

    def append(self, position: dict[str, Any]) -> None:
        if not isinstance(position, dict):
            raise TypeError("position must be a dict")
        position_id = position.get("position_id") or f"legacy_position_{len(self)}"
        self[str(position_id)] = position

    def __iter__(self):
        # Intentionally iterate over position records for backwards
        # compatibility with legacy list-like position collections. Use
        # .keys() or .items() when position IDs are required.
        return iter(self.values())

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            actual_key = list(self.keys())[key]
            return super().__setitem__(actual_key, value)
        return super().__setitem__(key, value)


def _normalize_positions_collection(value: Any) -> PositionBook:
    positions = PositionBook()
    if isinstance(value, dict):
        for position_id, position in value.items():
            if isinstance(position, dict):
                positions[str(position_id)] = position
        return positions
    if isinstance(value, list):
        for index, position in enumerate(value):
            if not isinstance(position, dict):
                continue
            position_id = position.get("position_id") or f"legacy_position_{index}"
            positions[str(position_id)] = position
    return positions


def _position_values(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [position for position in value.values() if isinstance(position, dict)]
    if isinstance(value, list):
        return [position for position in value if isinstance(position, dict)]
    return []


def _fresh_state() -> dict[str, Any]:
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "positions": PositionBook(),
        "opportunities": [],
        "orders": [],
        "order_intents": {},
        "close_intents": {},
        "fills": [],
        "strategy_snapshots": [],
        "account_snapshots": [],
        "reconciliation_snapshots": [],
        "halt_state": None,
        "readiness": {
            "strategies": {
                S02_VOL_ENHANCED: deepcopy(S02_LEGACY_READINESS_FIELDS),
            },
        },
    }


class StateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._write_lock = threading.Lock()
        self.strategy_state_locks = {
            strategy_id: threading.RLock() for strategy_id in STRATEGY_IDS
        }
        if self.path.exists():
            self.state = self._load_existing()
        else:
            self.state = _fresh_state()
            self.save()

    @property
    def state_store_write_lock(self) -> threading.Lock:
        return self._write_lock

    def get_strategy_lock(self, strategy_id: str) -> threading.Lock:
        """Return a strategy lock.

        Always acquire strategy_state_locks[strategy_id] before
        state_store_write_lock when both are needed. Never reverse this order.
        """
        if strategy_id not in self.strategy_state_locks:
            self.strategy_state_locks[strategy_id] = threading.RLock()
        return self.strategy_state_locks[strategy_id]

    def _load_existing(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StateStoreCorruptError(f"StateStore JSON is corrupt: {exc}") from exc
        found = payload.get("schema_version")
        if found != CURRENT_SCHEMA_VERSION:
            raise StateStoreCorruptError(
                f"StateStore schema_version mismatch: found {found!r}, "
                f"expected {CURRENT_SCHEMA_VERSION!r}"
            )
        self._normalize_readiness(payload)
        payload.setdefault("order_intents", {})
        payload.setdefault("close_intents", {})
        payload["positions"] = _normalize_positions_collection(payload.get("positions", {}))
        return payload

    @staticmethod
    def _normalize_readiness(payload: dict[str, Any]) -> None:
        readiness = payload.setdefault("readiness", {})
        strategies = readiness.setdefault("strategies", {})
        existing = strategies.setdefault(S02_VOL_ENHANCED, {})
        legacy_top_level = readiness.get(S02_VOL_ENHANCED, {})
        for key, default in S02_LEGACY_READINESS_FIELDS.items():
            if key not in existing:
                if isinstance(legacy_top_level, dict) and key in legacy_top_level:
                    existing[key] = legacy_top_level[key]
                else:
                    existing[key] = default
        readiness.pop(S02_VOL_ENHANCED, None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        with self._write_lock:
            payload = deepcopy(self.state)
            payload["schema_version"] = CURRENT_SCHEMA_VERSION
            encoded = json.dumps(payload, indent=2, sort_keys=True)
            tmp_path.write_text(encoded, encoding="utf-8")
            try:
                os.replace(tmp_path, self.path)
            except OSError:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
                raise

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.state.get("schema_version"),
            "positions": len(_position_values(self.state.get("positions", {}))),
            "opportunities": len(self.state.get("opportunities", [])),
            "orders": len(self.state.get("orders", [])),
            "fills": len(self.state.get("fills", [])),
            "halt_state": self.state.get("halt_state"),
            "latest_reconciliation": self.latest_reconciliation(),
        }

    def latest_reconciliation(self) -> dict[str, Any] | None:
        snapshots = self.state.get("reconciliation_snapshots", [])
        if not snapshots:
            return None
        return snapshots[-1]

    def get_readiness(self, strategy_id: str) -> dict[str, Any] | None:
        readiness = self.state.setdefault("readiness", {})
        strategies = readiness.setdefault("strategies", {})
        if strategy_id in strategies:
            return deepcopy(strategies[strategy_id])
        return None

    def update_readiness(
        self,
        strategy_id: str,
        readiness_status: dict[str, Any],
    ) -> None:
        with self.get_strategy_lock(strategy_id):
            readiness = self.state.setdefault("readiness", {})
            strategies = readiness.setdefault("strategies", {})
            existing = deepcopy(strategies.get(strategy_id, {}))
            if strategy_id == S02_VOL_ENHANCED:
                for key, default in S02_LEGACY_READINESS_FIELDS.items():
                    existing.setdefault(key, default)
            existing.update(deepcopy(readiness_status))
            strategies[strategy_id] = existing
            self.save()

    def create_order_intent(self, intent_record: dict[str, Any]) -> dict[str, Any]:
        strategy_id = intent_record.get("strategy_id")
        intent_id = intent_record.get("intent_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise ValueError("order intent strategy_id is required")
        if not isinstance(intent_id, str) or not intent_id:
            raise ValueError("order intent intent_id is required")
        record = deepcopy(intent_record)
        if "dry_run" not in record:
            record["dry_run"] = record.get("execution_mode") != "live_enabled"
        with self.get_strategy_lock(strategy_id):
            intents = self.state.setdefault("order_intents", {})
            intents[intent_id] = record
            self.save()
            return deepcopy(intents[intent_id])

    def _transition_created_order_intent(
        self,
        intent_id: str,
        *,
        new_status: str,
        updated_at: str,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        intents = self.state.setdefault("order_intents", {})
        intent = intents.get(intent_id)
        if intent is None:
            raise KeyError(f"order intent {intent_id!r} does not exist")
        if not isinstance(intent, dict):
            raise OrderIntentTransitionError(f"order intent {intent_id!r} is malformed")
        strategy_id = intent.get("strategy_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise OrderIntentTransitionError(f"order intent {intent_id!r} has no strategy_id")
        with self.get_strategy_lock(strategy_id):
            current = intents.get(intent_id)
            if current is None:
                raise KeyError(f"order intent {intent_id!r} does not exist")
            if current.get("status") != "created":
                raise OrderIntentTransitionError(
                    f"order intent {intent_id!r} status is {current.get('status')!r}, not 'created'"
                )
            updated = deepcopy(current)
            updated.update(deepcopy(fields))
            updated["status"] = new_status
            updated["updated_at"] = updated_at
            intents[intent_id] = updated
            self.save()
            return deepcopy(updated)

    def _transition_submitted_order_intent(
        self,
        intent_id: str,
        *,
        new_status: str,
        updated_at: str,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        intents = self.state.setdefault("order_intents", {})
        intent = intents.get(intent_id)
        if intent is None:
            raise KeyError(f"order intent {intent_id!r} does not exist")
        if not isinstance(intent, dict):
            raise OrderIntentTransitionError(f"order intent {intent_id!r} is malformed")
        strategy_id = intent.get("strategy_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise OrderIntentTransitionError(f"order intent {intent_id!r} has no strategy_id")
        with self.get_strategy_lock(strategy_id):
            current = intents.get(intent_id)
            if current is None:
                raise KeyError(f"order intent {intent_id!r} does not exist")
            if current.get("status") != "submitted":
                raise OrderIntentTransitionError(
                    f"order intent {intent_id!r} status is {current.get('status')!r}, not 'submitted'"
                )
            updated = deepcopy(current)
            updated.update(deepcopy(fields))
            updated["status"] = new_status
            updated["updated_at"] = updated_at
            intents[intent_id] = updated
            self.save()
            return deepcopy(updated)

    def _transition_confirmed_order_intent(
        self,
        intent_id: str,
        *,
        new_status: str,
        updated_at: str,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        intents = self.state.setdefault("order_intents", {})
        intent = intents.get(intent_id)
        if intent is None:
            raise KeyError(f"order intent {intent_id!r} does not exist")
        if not isinstance(intent, dict):
            raise OrderIntentTransitionError(f"order intent {intent_id!r} is malformed")
        strategy_id = intent.get("strategy_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise OrderIntentTransitionError(f"order intent {intent_id!r} has no strategy_id")
        with self.get_strategy_lock(strategy_id):
            current = intents.get(intent_id)
            if current is None:
                raise KeyError(f"order intent {intent_id!r} does not exist")
            if current.get("status") != "confirmed":
                raise OrderIntentTransitionError(
                    f"order intent {intent_id!r} status is {current.get('status')!r}, not 'confirmed'"
                )
            updated = deepcopy(current)
            updated.update(deepcopy(fields))
            updated["status"] = new_status
            updated["updated_at"] = updated_at
            intents[intent_id] = updated
            self.save()
            return deepcopy(updated)

    def expire_order_intent(
        self,
        intent_id: str,
        *,
        expired_at: str,
        expire_reason: str,
        expired_event_id: str,
    ) -> dict[str, Any]:
        return self._transition_created_order_intent(
            intent_id,
            new_status="expired",
            updated_at=expired_at,
            fields={
                "expired_at": expired_at,
                "expire_reason": expire_reason,
                "expired_event_id": expired_event_id,
            },
        )

    def cancel_order_intent(
        self,
        intent_id: str,
        *,
        cancelled_at: str,
        cancel_reason: str,
        cancelled_event_id: str,
    ) -> dict[str, Any]:
        return self._transition_created_order_intent(
            intent_id,
            new_status="cancelled",
            updated_at=cancelled_at,
            fields={
                "cancelled_at": cancelled_at,
                "cancel_reason": cancel_reason,
                "cancelled_event_id": cancelled_event_id,
            },
        )

    def submit_order_intent(
        self,
        intent_id: str,
        *,
        submitted_at: str,
        order_submitted_event_id: str,
        simulated_order_id: str,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        return self._transition_created_order_intent(
            intent_id,
            new_status="submitted",
            updated_at=submitted_at,
            fields={
                "submitted_at": submitted_at,
                "order_submitted_event_id": order_submitted_event_id,
                "simulated_order_id": simulated_order_id,
                "dry_run": dry_run,
            },
        )

    def confirm_order_intent(
        self,
        intent_id: str,
        *,
        confirmed_at: str,
        order_confirmed_event_id: str,
        simulated_order_id: str,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        return self._transition_submitted_order_intent(
            intent_id,
            new_status="confirmed",
            updated_at=confirmed_at,
            fields={
                "confirmed_at": confirmed_at,
                "order_confirmed_event_id": order_confirmed_event_id,
                "simulated_order_id": simulated_order_id,
                "dry_run": dry_run,
            },
        )

    def fill_order_intent(
        self,
        intent_id: str,
        *,
        filled_at: str,
        fill_confirmed_event_id: str,
        simulated_order_id: str,
        fill_id: str,
        fill_price: float | Decimal,
        fill_quantity: int | float | Decimal,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(fill_id, str) or not fill_id:
            raise OrderIntentTransitionError("fill_id must be a non-empty string")
        validated_price = validate_numeric_field(
            "fill_price",
            fill_price,
            minimum=0,
            allow_equal=True,
            allow_int=False,
        )
        validated_quantity = validate_numeric_field(
            "fill_quantity",
            fill_quantity,
            minimum=0,
            allow_equal=False,
            allow_int=True,
        )
        return self._transition_confirmed_order_intent(
            intent_id,
            new_status="filled",
            updated_at=filled_at,
            fields={
                "filled_at": filled_at,
                "fill_confirmed_event_id": fill_confirmed_event_id,
                "simulated_order_id": simulated_order_id,
                "fill_id": fill_id,
                "fill_price": validated_price,
                "fill_quantity": validated_quantity,
                "dry_run": dry_run,
            },
        )

    def create_open_position(self, position_record: dict[str, Any]) -> dict[str, Any]:
        strategy_id = position_record.get("strategy_id")
        position_id = position_record.get("position_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise ValueError("position strategy_id is required")
        if not isinstance(position_id, str) or not position_id:
            raise ValueError("position position_id is required")
        if position_record.get("status") != "open":
            raise ValueError("position status must be 'open'")
        entry_price = validate_numeric_field(
            "entry_price",
            position_record.get("entry_price"),
            minimum=0,
            allow_equal=True,
            allow_int=False,
        )
        quantity = validate_numeric_field(
            "quantity",
            position_record.get("quantity"),
            minimum=0,
            allow_equal=False,
            allow_int=True,
        )
        with self.get_strategy_lock(strategy_id):
            positions = _normalize_positions_collection(self.state.get("positions", {}))
            for position in positions.values():
                if (
                    isinstance(position, dict)
                    and position.get("strategy_id") == strategy_id
                    and position.get("status") == "open"
                ):
                    raise ValueError(f"open position already exists for strategy_id={strategy_id!r}")
            record = deepcopy(position_record)
            record["entry_price"] = entry_price
            record["quantity"] = quantity
            positions[position_id] = record
            self.state["positions"] = positions
            self.save()
            return deepcopy(record)

    def get_position(self, position_id: str) -> dict[str, Any] | None:
        positions = _normalize_positions_collection(self.state.get("positions", {}))
        position = positions.get(position_id)
        if position is None:
            return None
        return deepcopy(position)

    def get_open_position(self, strategy_id: str, symbol: str | None = None) -> dict[str, Any] | None:
        with self.get_strategy_lock(strategy_id):
            for position in _position_values(self.state.get("positions", {})):
                if position.get("strategy_id") != strategy_id:
                    continue
                if position.get("status") != "open":
                    continue
                if symbol is not None and position.get("symbol") != symbol:
                    continue
                return deepcopy(position)
        return None

    def list_positions(
        self,
        strategy_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        records = []
        for position in _position_values(self.state.get("positions", {})):
            if strategy_id is not None and position.get("strategy_id") != strategy_id:
                continue
            if status is not None and position.get("status") != status:
                continue
            records.append(deepcopy(position))
        return records

    def create_close_intent(self, close_intent_record: dict[str, Any]) -> dict[str, Any]:
        strategy_id = close_intent_record.get("strategy_id")
        close_intent_id = close_intent_record.get("close_intent_id")
        position_id = close_intent_record.get("position_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise ValueError("close intent strategy_id is required")
        if not isinstance(close_intent_id, str) or not close_intent_id:
            raise ValueError("close intent close_intent_id is required")
        if not isinstance(position_id, str) or not position_id:
            raise ValueError("close intent position_id is required")
        if close_intent_record.get("status") != "created":
            raise ValueError("close intent status must be 'created'")
        if close_intent_record.get("action") != "close":
            raise ValueError("close intent action must be 'close'")
        if close_intent_record.get("dry_run") is not True:
            raise ValueError("close intent dry_run must be True")

        required_string_fields = (
            "sleeve_id",
            "symbol",
            "execution_mode",
            "created_at",
            "updated_at",
            "close_reason",
            "requested_by",
            "position_opened_event_id",
            "source_signal_event_id",
            "fill_confirmed_event_id",
            "close_intent_created_event_id",
        )
        for field in required_string_fields:
            value = close_intent_record.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"close intent {field} is required")

        quantity = validate_numeric_field(
            "quantity",
            close_intent_record.get("quantity"),
            minimum=0,
            allow_equal=False,
            allow_int=True,
        )
        entry_price = validate_numeric_field(
            "entry_price",
            close_intent_record.get("entry_price"),
            minimum=0,
            allow_equal=True,
            allow_int=True,
        )

        forbidden_fields = ("target_price", "limit_price", "order_type")
        for field in forbidden_fields:
            if field in close_intent_record:
                raise ValueError(f"close intent must not include {field}")

        with self.get_strategy_lock(strategy_id):
            active = self.get_active_close_intent(position_id)
            if active is not None:
                raise ValueError(
                    f"active close intent already exists for position_id={position_id!r}"
                )
            close_intents = self.state.setdefault("close_intents", {})
            if close_intent_id in close_intents:
                raise ValueError(f"close intent {close_intent_id!r} already exists")
            record = deepcopy(close_intent_record)
            record["quantity"] = quantity
            record["entry_price"] = entry_price
            close_intents[close_intent_id] = record
            self.save()
            return deepcopy(record)

    def get_close_intent(self, close_intent_id: str) -> dict[str, Any] | None:
        intent = self.state.setdefault("close_intents", {}).get(close_intent_id)
        if intent is None:
            return None
        return deepcopy(intent)

    def get_active_close_intent(self, position_id: str) -> dict[str, Any] | None:
        for intent in self.state.setdefault("close_intents", {}).values():
            if not isinstance(intent, dict):
                continue
            if intent.get("position_id") != position_id:
                continue
            if intent.get("status") in ACTIVE_CLOSE_INTENT_STATUSES:
                return deepcopy(intent)
        return None

    def submit_close_intent(
        self,
        close_intent_id: str,
        *,
        submitted_at: str,
        close_order_submitted_event_id: str,
        simulated_close_order_id: str,
        close_order_ref: str,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        close_intents = self.state.setdefault("close_intents", {})
        intent = close_intents.get(close_intent_id)
        if intent is None:
            raise KeyError(f"close intent {close_intent_id!r} does not exist")
        if not isinstance(intent, dict):
            raise CloseIntentTransitionError(
                f"close intent {close_intent_id!r} is malformed"
            )
        strategy_id = intent.get("strategy_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise CloseIntentTransitionError(
                f"close intent {close_intent_id!r} has no strategy_id"
            )
        with self.get_strategy_lock(strategy_id):
            current = close_intents.get(close_intent_id)
            if current is None:
                raise KeyError(f"close intent {close_intent_id!r} does not exist")
            if current.get("status") != "created":
                raise CloseIntentTransitionError(
                    f"close intent {close_intent_id!r} status is {current.get('status')!r}, not 'created'"
                )
            updated = deepcopy(current)
            updated.update(
                {
                    "status": "submitted",
                    "submitted_at": submitted_at,
                    "updated_at": submitted_at,
                    "close_order_submitted_event_id": close_order_submitted_event_id,
                    "simulated_close_order_id": simulated_close_order_id,
                    "close_order_ref": close_order_ref,
                    "dry_run": dry_run,
                }
            )
            close_intents[close_intent_id] = updated
            self.save()
            return deepcopy(updated)

    def list_close_intents(
        self,
        strategy_id: str | None = None,
        status: str | None = None,
        position_id: str | None = None,
    ) -> list[dict[str, Any]]:
        records = []
        for intent in self.state.setdefault("close_intents", {}).values():
            if not isinstance(intent, dict):
                continue
            if strategy_id is not None and intent.get("strategy_id") != strategy_id:
                continue
            if status is not None and intent.get("status") != status:
                continue
            if position_id is not None and intent.get("position_id") != position_id:
                continue
            records.append(deepcopy(intent))
        return records

    def mark_position_close_intent_created(
        self,
        position_id: str,
        *,
        close_intent_id: str,
        close_intent_created_event_id: str,
        close_intent_created_at: str,
    ) -> dict[str, Any]:
        positions = _normalize_positions_collection(self.state.get("positions", {}))
        position = positions.get(position_id)
        if position is None:
            raise KeyError(f"position {position_id!r} does not exist")
        if position.get("status") != "open":
            raise ValueError(
                f"position {position_id!r} status is {position.get('status')!r}, not 'open'"
            )
        existing = position.get("active_close_intent_id")
        if existing:
            active = self.get_close_intent(str(existing))
            if active is not None and active.get("status") in ACTIVE_CLOSE_INTENT_STATUSES:
                raise ValueError(
                    f"active close intent already exists for position_id={position_id!r}"
                )
        strategy_id = position.get("strategy_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise ValueError(f"position {position_id!r} has no strategy_id")
        with self.get_strategy_lock(strategy_id):
            current = positions.get(position_id)
            if current is None:
                raise KeyError(f"position {position_id!r} does not exist")
            if current.get("status") != "open":
                raise ValueError(
                    f"position {position_id!r} status is {current.get('status')!r}, not 'open'"
                )
            existing = current.get("active_close_intent_id")
            if existing:
                active = self.get_close_intent(str(existing))
                if active is not None and active.get("status") in ACTIVE_CLOSE_INTENT_STATUSES:
                    raise ValueError(
                        f"active close intent already exists for position_id={position_id!r}"
                    )
            updated = deepcopy(current)
            updated["active_close_intent_id"] = close_intent_id
            updated["close_intent_created_event_id"] = close_intent_created_event_id
            updated["close_intent_created_at"] = close_intent_created_at
            updated["updated_at"] = close_intent_created_at
            positions[position_id] = updated
            self.state["positions"] = positions
            self.save()
            return deepcopy(updated)

    def mark_intent_position_opened(
        self,
        intent_id: str,
        *,
        position_id: str,
        position_opened_event_id: str,
        opened_at: str,
    ) -> dict[str, Any]:
        intents = self.state.setdefault("order_intents", {})
        intent = intents.get(intent_id)
        if intent is None:
            raise KeyError(f"order intent {intent_id!r} does not exist")
        if not isinstance(intent, dict):
            raise OrderIntentTransitionError(f"order intent {intent_id!r} is malformed")
        strategy_id = intent.get("strategy_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise OrderIntentTransitionError(f"order intent {intent_id!r} has no strategy_id")
        with self.get_strategy_lock(strategy_id):
            current = intents.get(intent_id)
            if current is None:
                raise KeyError(f"order intent {intent_id!r} does not exist")
            if current.get("status") != "filled":
                raise OrderIntentTransitionError(
                    f"order intent {intent_id!r} status is {current.get('status')!r}, not 'filled'"
                )
            updated = deepcopy(current)
            updated.update(
                {
                    "status": "position_opened",
                    "position_id": position_id,
                    "position_opened_event_id": position_opened_event_id,
                    "position_opened_at": opened_at,
                    "updated_at": opened_at,
                }
            )
            intents[intent_id] = updated
            self.save()
            return deepcopy(updated)

    def get_order_intent(self, intent_id: str) -> dict[str, Any] | None:
        intent = self.state.setdefault("order_intents", {}).get(intent_id)
        if intent is None:
            return None
        return deepcopy(intent)

    def get_active_order_intent(self, strategy_id: str) -> dict[str, Any] | None:
        active_statuses = {"created"}
        with self.get_strategy_lock(strategy_id):
            for intent in self.state.setdefault("order_intents", {}).values():
                if not isinstance(intent, dict):
                    continue
                if intent.get("strategy_id") != strategy_id:
                    continue
                if intent.get("status") in active_statuses:
                    return deepcopy(intent)
        return None

    def list_order_intents(self, strategy_id: str | None = None) -> list[dict[str, Any]]:
        intents = self.state.setdefault("order_intents", {}).values()
        records = [
            deepcopy(intent)
            for intent in intents
            if isinstance(intent, dict)
            and (strategy_id is None or intent.get("strategy_id") == strategy_id)
        ]
        return records

    def get_all_readiness(self) -> dict[str, Any]:
        return deepcopy(self.state.setdefault("readiness", {}))

    def bot_attributed_exposure(self) -> dict[str, float]:
        exposure: dict[str, float] = {}
        for position in _position_values(self.state.get("positions", {})):
            status = position.get("status")
            if status not in {"pending_open", "open", "pending_close", "partial_fill_error", "NEEDS_RECONCILIATION"}:
                continue
            for leg in position.get("legs", []):
                if not isinstance(leg, dict):
                    continue
                key = leg.get("conId") or leg.get("symbol")
                if key is None:
                    continue
                sec_type = leg.get("secType", "")
                exposure_key = f"{sec_type}:{key}"
                exposure[exposure_key] = exposure.get(exposure_key, 0.0) + float(
                    leg.get("signed_qty", 0) or 0
                )
        return exposure
