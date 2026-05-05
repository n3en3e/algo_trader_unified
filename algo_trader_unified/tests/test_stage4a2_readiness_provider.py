from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.readiness_provider import DefaultHealthSnapshotProvider
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.jobs.readiness import HealthSnapshot


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class ReadinessProviderCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_store = StateStore(self.root / "data/state/portfolio_state.json")
        self.halt_state_path = self.root / "data/state/halt_state.json"
        self.snapshots_dir = self.root / "data/snapshots"
        self.strategy_ids = [S01_VOL_BASELINE, S02_VOL_ENHANCED]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def provider(self, **overrides) -> DefaultHealthSnapshotProvider:
        kwargs = {
            "state_store": self.state_store,
            "halt_state_path": self.halt_state_path,
            "snapshots_dir": self.snapshots_dir,
            "max_staleness_minutes": 15,
            "strategy_ids": self.strategy_ids,
        }
        kwargs.update(overrides)
        return DefaultHealthSnapshotProvider(**kwargs)

    def write_snapshot(self, payload: dict | str, *, mtime: datetime | None = None) -> Path:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshots_dir / "account_snapshot.json"
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")
        if mtime is not None:
            epoch = mtime.timestamp()
            os.utime(path, (epoch, epoch))
        return path

    def set_readiness(
        self,
        strategy_id: str,
        *,
        dirty_state: bool = False,
        unknown_broker_exposure: bool = False,
        calendar_expired: bool = False,
        iv_baseline_available: bool | None = True,
    ) -> None:
        strategies = self.state_store.state.setdefault("readiness", {}).setdefault(
            "strategies",
            {},
        )
        existing = dict(strategies.get(strategy_id, {}))
        existing.update(
            {
                "strategy_id": strategy_id,
                "dirty_state": dirty_state,
                "unknown_broker_exposure": unknown_broker_exposure,
                "calendar_expired": calendar_expired,
                "iv_baseline_available": iv_baseline_available,
            }
        )
        strategies[strategy_id] = existing


class ConstructorAndBoundaryTests(ReadinessProviderCase):
    def test_constructor_requires_explicit_dependencies(self) -> None:
        with self.assertRaises(TypeError):
            DefaultHealthSnapshotProvider()

    def test_provider_returns_existing_health_snapshot_type(self) -> None:
        snapshot = self.provider()()
        self.assertIsInstance(snapshot, HealthSnapshot)

    def test_provider_is_not_wired_into_daemon_entrypoint(self) -> None:
        source = (PACKAGE_ROOT / "tools/daemon.py").read_text(encoding="utf-8")
        self.assertNotIn("DefaultHealthSnapshotProvider", source)
        self.assertNotIn("readiness_provider", source)

    def test_provider_source_has_no_broker_or_live_boundaries(self) -> None:
        source = (PACKAGE_ROOT / "core/readiness_provider.py").read_text(encoding="utf-8")
        forbidden = (
            "ib_insync",
            "reqMktData",
            "placeOrder",
            "core.broker",
            "systemd",
            "scheduler.start",
            "LedgerAppender",
            ".jsonl",
        )
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_provider_does_not_mutate_statestore_or_write_files(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        before_state = json.loads(json.dumps(self.state_store.state))
        before_paths = {
            path.relative_to(self.root): path.stat().st_mtime_ns
            for path in self.root.rglob("*")
            if path.is_file()
        }
        with mock.patch.object(self.state_store, "save", side_effect=AssertionError):
            self.provider()()
        after_paths = {
            path.relative_to(self.root): path.stat().st_mtime_ns
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before_state, json.loads(json.dumps(self.state_store.state)))
        self.assertEqual(before_paths, after_paths)


class SnapshotFreshnessTests(ReadinessProviderCase):
    def test_missing_snapshots_dir_sets_account_fields_false(self) -> None:
        snapshot = self.provider()()
        self.assertFalse(snapshot.account_snapshot_fresh)
        self.assertFalse(snapshot.nlv_valid)

    def test_empty_snapshots_dir_sets_account_fields_false(self) -> None:
        self.snapshots_dir.mkdir(parents=True)
        snapshot = self.provider()()
        self.assertFalse(snapshot.account_snapshot_fresh)
        self.assertFalse(snapshot.nlv_valid)

    def test_fresh_snapshot_iso_timestamp_sets_account_fields_true(self) -> None:
        self.write_snapshot({"timestamp": datetime.now(timezone.utc).isoformat()})
        snapshot = self.provider()()
        self.assertTrue(snapshot.account_snapshot_fresh)
        self.assertTrue(snapshot.nlv_valid)

    def test_stale_snapshot_iso_timestamp_sets_account_fields_false(self) -> None:
        stale = datetime.now(timezone.utc) - timedelta(minutes=20)
        self.write_snapshot({"timestamp": stale.isoformat()})
        snapshot = self.provider()()
        self.assertFalse(snapshot.account_snapshot_fresh)
        self.assertFalse(snapshot.nlv_valid)

    def test_missing_timestamp_falls_back_to_os_mtime(self) -> None:
        fresh_mtime = datetime.now(timezone.utc) - timedelta(minutes=1)
        self.write_snapshot({"nlv": 100000}, mtime=fresh_mtime)
        snapshot = self.provider()()
        self.assertTrue(snapshot.account_snapshot_fresh)
        self.assertTrue(snapshot.nlv_valid)

    def test_unparseable_timestamp_falls_back_to_os_mtime(self) -> None:
        fresh_mtime = datetime.now(timezone.utc) - timedelta(minutes=1)
        self.write_snapshot({"timestamp": "not-a-timestamp"}, mtime=fresh_mtime)
        snapshot = self.provider()()
        self.assertTrue(snapshot.account_snapshot_fresh)
        self.assertTrue(snapshot.nlv_valid)

    def test_malformed_snapshot_json_does_not_crash_and_fails_closed(self) -> None:
        self.write_snapshot("{not json")
        snapshot = self.provider()()
        self.assertFalse(snapshot.account_snapshot_fresh)
        self.assertFalse(snapshot.nlv_valid)


class HaltStateTests(ReadinessProviderCase):
    def test_missing_halt_state_means_no_active_halt(self) -> None:
        snapshot = self.provider()()
        self.assertEqual(
            snapshot.halt_active_by_strategy,
            {S01_VOL_BASELINE: False, S02_VOL_ENHANCED: False},
        )

    def test_account_halt_applies_to_all_strategies(self) -> None:
        self.halt_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.halt_state_path.write_text(
            json.dumps({"scope": "account", "tier": "hard", "reason": "operator"}),
            encoding="utf-8",
        )
        snapshot = self.provider()()
        self.assertEqual(
            snapshot.halt_active_by_strategy,
            {S01_VOL_BASELINE: True, S02_VOL_ENHANCED: True},
        )

    def test_strategy_halt_applies_only_to_affected_strategy(self) -> None:
        self.halt_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.halt_state_path.write_text(
            json.dumps(
                {
                    "scope": "strategy",
                    "id": S02_VOL_ENHANCED,
                    "tier": "soft",
                    "reason": "operator",
                }
            ),
            encoding="utf-8",
        )
        snapshot = self.provider()()
        self.assertEqual(
            snapshot.halt_active_by_strategy,
            {S01_VOL_BASELINE: False, S02_VOL_ENHANCED: True},
        )

    def test_malformed_halt_state_fails_closed_without_crashing(self) -> None:
        self.halt_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.halt_state_path.write_text("{bad json", encoding="utf-8")
        snapshot = self.provider()()
        self.assertEqual(
            snapshot.halt_active_by_strategy,
            {S01_VOL_BASELINE: True, S02_VOL_ENHANCED: True},
        )


class PerStrategyReadinessTests(ReadinessProviderCase):
    def test_output_contains_every_injected_strategy_id(self) -> None:
        snapshot = self.provider()()
        for field in (
            snapshot.halt_active_by_strategy,
            snapshot.dirty_state_by_strategy,
            snapshot.unknown_broker_exposure_by_strategy,
            snapshot.calendar_expired_by_strategy,
            snapshot.iv_baseline_available_by_strategy,
        ):
            self.assertEqual(set(field), {S01_VOL_BASELINE, S02_VOL_ENHANCED})

    def test_provider_does_not_derive_strategy_ids_from_statestore_keys(self) -> None:
        self.set_readiness("EXTRA_STRATEGY")
        snapshot = self.provider(strategy_ids=[S01_VOL_BASELINE])()
        self.assertEqual(set(snapshot.dirty_state_by_strategy), {S01_VOL_BASELINE})
        self.assertNotIn("EXTRA_STRATEGY", snapshot.dirty_state_by_strategy)

    def test_readiness_fields_reflect_statestore_records(self) -> None:
        self.set_readiness(
            S01_VOL_BASELINE,
            dirty_state=True,
            unknown_broker_exposure=False,
            calendar_expired=True,
            iv_baseline_available=False,
        )
        self.set_readiness(
            S02_VOL_ENHANCED,
            dirty_state=False,
            unknown_broker_exposure=True,
            calendar_expired=False,
            iv_baseline_available=True,
        )
        snapshot = self.provider()()
        self.assertTrue(snapshot.dirty_state_by_strategy[S01_VOL_BASELINE])
        self.assertFalse(snapshot.dirty_state_by_strategy[S02_VOL_ENHANCED])
        self.assertFalse(snapshot.unknown_broker_exposure_by_strategy[S01_VOL_BASELINE])
        self.assertTrue(snapshot.unknown_broker_exposure_by_strategy[S02_VOL_ENHANCED])
        self.assertTrue(snapshot.calendar_expired_by_strategy[S01_VOL_BASELINE])
        self.assertFalse(snapshot.calendar_expired_by_strategy[S02_VOL_ENHANCED])
        self.assertFalse(snapshot.iv_baseline_available_by_strategy[S01_VOL_BASELINE])
        self.assertTrue(snapshot.iv_baseline_available_by_strategy[S02_VOL_ENHANCED])

    def test_account_snapshot_fields_are_account_level_bools(self) -> None:
        snapshot = self.provider()()
        self.assertIsInstance(snapshot.account_snapshot_fresh, bool)
        self.assertIsInstance(snapshot.nlv_valid, bool)


if __name__ == "__main__":
    unittest.main()
