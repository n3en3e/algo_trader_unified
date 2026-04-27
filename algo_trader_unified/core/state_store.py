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
from pathlib import Path
from typing import Any

from algo_trader_unified.config.portfolio import STRATEGY_IDS, S02_VOL_ENHANCED


CURRENT_SCHEMA_VERSION = 1


class StateStoreCorruptError(RuntimeError):
    """Raised when StateStore cannot be trusted."""


def _fresh_state() -> dict[str, Any]:
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "positions": [],
        "opportunities": [],
        "orders": [],
        "fills": [],
        "strategy_snapshots": [],
        "account_snapshots": [],
        "reconciliation_snapshots": [],
        "halt_state": None,
        "readiness": {
            S02_VOL_ENHANCED: {
                "standard_strangle_clean_days": 0,
                "last_clean_day_date": None,
                "last_reconciliation_check": None,
                "0dte_jobs_registered": False,
            }
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
        payload.setdefault("readiness", {})
        payload["readiness"].setdefault(
            S02_VOL_ENHANCED,
            _fresh_state()["readiness"][S02_VOL_ENHANCED],
        )
        return payload

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
            "positions": len(self.state.get("positions", [])),
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

    def bot_attributed_exposure(self) -> dict[str, float]:
        exposure: dict[str, float] = {}
        for position in self.state.get("positions", []):
            if not isinstance(position, dict):
                continue
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
