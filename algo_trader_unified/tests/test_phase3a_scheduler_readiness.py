from __future__ import annotations

import builtins
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.scheduler import (
    DEFAULT_COALESCE,
    DEFAULT_MAX_INSTANCES,
    JOB_DAILY_DIGEST,
    JOB_KEEPALIVE,
    JOB_MARKET_OPEN_SCAN,
    JOB_RISK_MONITOR,
    JOB_S01_VOL_SCAN,
    JOB_S02_VOL_SCAN,
    JOB_SPECS,
    JOB_WEEKLY_DIGEST,
)
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.readiness import ReadinessManager, ReadinessStatus
from algo_trader_unified.core.scheduler import (
    JobNotFoundError,
    MissingSchedulerDependencyError,
    UnifiedScheduler,
)
from algo_trader_unified.core.skip_reasons import (
    SKIP_ACCOUNT_SNAPSHOT_STALE,
    SKIP_BLACKOUT_DATE,
    SKIP_HALTED,
    SKIP_IV_BASELINE_MISSING,
    SKIP_NEEDS_RECONCILIATION,
    SKIP_NLV_DEGRADED,
    SKIP_STATESTORE_UNREADABLE,
    SKIP_UNKNOWN_BROKER_EXPOSURE,
)
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.jobs.readiness import (
    HealthSnapshot,
    all_clear_health_snapshot,
    market_open_scan,
)
from algo_trader_unified.strategies.vol import signals


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent


def source_path(*parts: str) -> Path:
    return PACKAGE_ROOT.joinpath(*parts)


class TmpCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_store = StateStore(self.root / "data/state/portfolio_state.json")
        self.ledger = LedgerAppender(self.root)
        self.manager = ReadinessManager(self.state_store, self.ledger)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def execution_events(self) -> list[dict]:
        text = (self.root / "data/ledger/execution_ledger.jsonl").read_text()
        return [json.loads(line) for line in text.splitlines() if line]

    def order_ledger_text(self) -> str:
        return (self.root / "data/ledger/order_ledger.jsonl").read_text()


class SchedulerConfigTests(unittest.TestCase):
    def test_expected_job_specs_present(self) -> None:
        self.assertEqual(
            {
                JOB_KEEPALIVE,
                JOB_RISK_MONITOR,
                JOB_MARKET_OPEN_SCAN,
                JOB_DAILY_DIGEST,
                JOB_WEEKLY_DIGEST,
                JOB_S01_VOL_SCAN,
                JOB_S02_VOL_SCAN,
            },
            set(JOB_SPECS),
        )

    def test_scheduler_defaults_and_disabled_stubs(self) -> None:
        for spec in JOB_SPECS.values():
            self.assertEqual(spec.max_instances, DEFAULT_MAX_INSTANCES)
            self.assertEqual(spec.coalesce, DEFAULT_COALESCE)
        self.assertFalse(JOB_SPECS[JOB_S01_VOL_SCAN].enabled)
        self.assertFalse(JOB_SPECS[JOB_S02_VOL_SCAN].enabled)
        self.assertFalse(any("0dte" in job_id.lower() for job_id in JOB_SPECS))


class SkipReasonTests(unittest.TestCase):
    def test_import_safe_skip_reasons(self) -> None:
        module = importlib.import_module("algo_trader_unified.core.skip_reasons")
        for name in (
            "SKIP_NLV_DEGRADED",
            "SKIP_UNKNOWN_BROKER_EXPOSURE",
            "SKIP_ACCOUNT_SNAPSHOT_STALE",
            "SKIP_STATESTORE_UNREADABLE",
            "SKIP_IV_BASELINE_MISSING",
        ):
            self.assertTrue(hasattr(module, name))

    def test_readiness_imports_skip_reasons_from_core(self) -> None:
        source = source_path("jobs", "readiness.py").read_text()
        self.assertIn("algo_trader_unified.core.skip_reasons import", source)
        for value in (
            SKIP_NLV_DEGRADED,
            SKIP_UNKNOWN_BROKER_EXPOSURE,
            SKIP_ACCOUNT_SNAPSHOT_STALE,
            SKIP_STATESTORE_UNREADABLE,
            SKIP_IV_BASELINE_MISSING,
        ):
            self.assertNotIn(f'"{value}"', source)
            self.assertNotIn(f"'{value}'", source)

    def test_vol_signals_imports_canonical_skip_reasons(self) -> None:
        core = importlib.import_module("algo_trader_unified.core.skip_reasons")
        for name in (
            "SKIP_HALTED",
            "SKIP_EXISTING_POSITION",
            "SKIP_BLACKOUT_DATE",
            "SKIP_VIX_GATE",
            "SKIP_IV_RANK_BELOW_MIN",
            "SKIP_NEEDS_RECONCILIATION",
            "SKIP_ORDERREF_MISSING",
        ):
            self.assertIs(getattr(signals, name), getattr(core, name))

    def test_vol_signals_does_not_define_skip_literals(self) -> None:
        source = source_path("strategies", "vol", "signals.py").read_text()
        for value in (
            SKIP_HALTED,
            "SKIP_EXISTING_POSITION",
            SKIP_BLACKOUT_DATE,
            "SKIP_VIX_GATE",
            "SKIP_IV_RANK_BELOW_MIN",
            SKIP_NEEDS_RECONCILIATION,
            "SKIP_ORDERREF_MISSING",
        ):
            self.assertNotIn(f'{value} = "{value}"', source)
            self.assertNotIn(f"{value} = '{value}'", source)


class JobsPackageTests(unittest.TestCase):
    def test_jobs_package_imports(self) -> None:
        self.assertTrue(source_path("jobs", "__init__.py").exists())
        importlib.import_module("algo_trader_unified.jobs.readiness")


class LazyAPSchedulerTests(TmpCase):
    def test_import_scheduler_without_apscheduler(self) -> None:
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith("apscheduler"):
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        sys.modules.pop("algo_trader_unified.core.scheduler", None)
        with mock.patch("builtins.__import__", guarded_import):
            module = importlib.import_module("algo_trader_unified.core.scheduler")
            scheduler = module.UnifiedScheduler(
                state_store=self.state_store,
                ledger=self.ledger,
                readiness_manager=self.manager,
            )
            self.assertGreater(len(scheduler.list_job_specs()), 0)
            with self.assertRaises(module.MissingSchedulerDependencyError):
                scheduler.build_scheduler()

    def test_start_not_called_on_import(self) -> None:
        with mock.patch.object(UnifiedScheduler, "start") as start:
            importlib.reload(sys.modules["algo_trader_unified.core.scheduler"])
        start.assert_not_called()


class ReadinessManagerTests(TmpCase):
    def test_fresh_store_preserves_s02_readiness_fields(self) -> None:
        readiness = self.state_store.get_readiness(S02_VOL_ENHANCED)
        for key in (
            "standard_strangle_clean_days",
            "last_clean_day_date",
            "last_reconciliation_check",
            "0dte_jobs_registered",
        ):
            self.assertIn(key, readiness)

    def test_update_and_get_readiness(self) -> None:
        status = ReadinessStatus(
            strategy_id=S01_VOL_BASELINE,
            ready_for_entries=False,
            reason=SKIP_NLV_DEGRADED,
            checked_at="2026-04-27T13:35:00+00:00",
            dirty_state=False,
            unknown_broker_exposure=False,
            nlv_degraded=True,
            halt_active=False,
            calendar_expired=False,
            iv_baseline_available=True,
        )
        with mock.patch.object(self.state_store, "save", wraps=self.state_store.save) as save:
            self.manager.update_readiness(status)
        save.assert_called_once()
        found = self.manager.get_readiness(S01_VOL_BASELINE)
        self.assertEqual(found, status)
        self.assertEqual(
            self.state_store.get_readiness(S02_VOL_ENHANCED)["0dte_jobs_registered"],
            False,
        )

    def test_s02_legacy_fields_preserved_on_update(self) -> None:
        entry = self.state_store.state["readiness"]["strategies"][S02_VOL_ENHANCED]
        entry.update(
            {
                "standard_strangle_clean_days": 3,
                "last_clean_day_date": "2026-04-24",
                "last_reconciliation_check": "2026-04-24T20:00:00+00:00",
                "0dte_jobs_registered": True,
            }
        )
        status = ReadinessStatus(
            strategy_id=S02_VOL_ENHANCED,
            ready_for_entries=True,
            reason=None,
            checked_at="2026-04-27T13:35:00+00:00",
            dirty_state=False,
            unknown_broker_exposure=False,
            nlv_degraded=False,
            halt_active=False,
            calendar_expired=False,
            iv_baseline_available=True,
        )
        self.state_store.update_readiness(S02_VOL_ENHANCED, asdict(status))
        readiness = self.state_store.get_readiness(S02_VOL_ENHANCED)
        self.assertEqual(readiness["standard_strangle_clean_days"], 3)
        self.assertEqual(readiness["last_clean_day_date"], "2026-04-24")
        self.assertEqual(readiness["last_reconciliation_check"], "2026-04-24T20:00:00+00:00")
        self.assertTrue(readiness["0dte_jobs_registered"])
        self.assertTrue(readiness["ready_for_entries"])

    def test_legacy_top_level_s02_readiness_migrates_to_strategies(self) -> None:
        path = self.root / "legacy_state.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
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
                            "standard_strangle_clean_days": 5,
                            "last_clean_day_date": "2026-04-23",
                            "last_reconciliation_check": "2026-04-23T20:00:00+00:00",
                            "0dte_jobs_registered": True,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        store = StateStore(path)
        readiness = store.get_all_readiness()
        self.assertIn("strategies", readiness)
        self.assertIn(S02_VOL_ENHANCED, readiness["strategies"])
        self.assertEqual(
            store.get_readiness(S02_VOL_ENHANCED)["standard_strangle_clean_days"],
            5,
        )

    def test_get_all_readiness_exposes_canonical_strategies_shape(self) -> None:
        readiness = self.state_store.get_all_readiness()
        self.assertIn("strategies", readiness)
        self.assertIn(S02_VOL_ENHANCED, readiness["strategies"])


class MarketOpenScanTests(TmpCase):
    def snapshot(self, **overrides) -> HealthSnapshot:
        base = asdict(all_clear_health_snapshot((S01_VOL_BASELINE, S02_VOL_ENHANCED)))
        base.update(overrides)
        return HealthSnapshot(**base)

    def run_scan(self, snapshot: HealthSnapshot):
        return market_open_scan(
            readiness_manager=self.manager,
            current_time=datetime(2026, 4, 27, 13, 35, tzinfo=timezone.utc),
            strategy_ids=(S01_VOL_BASELINE, S02_VOL_ENHANCED),
            health_snapshot=snapshot,
        )

    def assert_skip(self, snapshot: HealthSnapshot, expected_reason: str) -> None:
        result = self.run_scan(snapshot)
        self.assertFalse(result.statuses[S01_VOL_BASELINE].ready_for_entries)
        self.assertEqual(result.statuses[S01_VOL_BASELINE].reason, expected_reason)
        events = [
            event
            for event in self.execution_events()
            if event["strategy_id"] == S01_VOL_BASELINE
        ]
        self.assertEqual(events[-1]["event_type"], "SIGNAL_SKIPPED")
        self.assertEqual(events[-1]["payload"]["skip_reason"], expected_reason)
        self.assertEqual(self.order_ledger_text(), "")

    def test_all_clear(self) -> None:
        result = self.run_scan(self.snapshot())
        self.assertTrue(result.statuses[S01_VOL_BASELINE].ready_for_entries)
        self.assertTrue(result.statuses[S02_VOL_ENHANCED].ready_for_entries)
        self.assertEqual(self.execution_events(), [])
        self.assertIsNotNone(self.state_store.get_readiness(S01_VOL_BASELINE))
        s02_readiness = self.state_store.get_readiness(S02_VOL_ENHANCED)
        self.assertIn("standard_strangle_clean_days", s02_readiness)
        self.assertIn("last_clean_day_date", s02_readiness)
        self.assertIn("last_reconciliation_check", s02_readiness)
        self.assertIn("0dte_jobs_registered", s02_readiness)

    def test_s02_legacy_fields_preserved_after_market_open_scan(self) -> None:
        self.state_store.state["readiness"]["strategies"][S02_VOL_ENHANCED][
            "standard_strangle_clean_days"
        ] = 3
        self.run_scan(self.snapshot())
        readiness = self.state_store.get_readiness(S02_VOL_ENHANCED)
        self.assertEqual(readiness["standard_strangle_clean_days"], 3)

    def test_each_failure_reason(self) -> None:
        cases = [
            ({"nlv_valid": False}, SKIP_NLV_DEGRADED),
            ({"dirty_state_by_strategy": {S01_VOL_BASELINE: True}}, SKIP_NEEDS_RECONCILIATION),
            (
                {"unknown_broker_exposure_by_strategy": {S01_VOL_BASELINE: True}},
                SKIP_UNKNOWN_BROKER_EXPOSURE,
            ),
            ({"halt_active_by_strategy": {S01_VOL_BASELINE: True}}, SKIP_HALTED),
            ({"calendar_expired_by_strategy": {S01_VOL_BASELINE: True}}, SKIP_BLACKOUT_DATE),
            (
                {"iv_baseline_available_by_strategy": {S01_VOL_BASELINE: False}},
                SKIP_IV_BASELINE_MISSING,
            ),
            ({"account_snapshot_fresh": False}, SKIP_ACCOUNT_SNAPSHOT_STALE),
            ({"state_store_readable": False}, SKIP_STATESTORE_UNREADABLE),
        ]
        for overrides, reason in cases:
            with self.subTest(reason=reason):
                self.assert_skip(self.snapshot(**overrides), reason)

    def test_multiple_failures_priority_and_failed_checks(self) -> None:
        snapshot = self.snapshot(
            nlv_valid=False,
            dirty_state_by_strategy={S01_VOL_BASELINE: True},
            unknown_broker_exposure_by_strategy={S01_VOL_BASELINE: True},
            halt_active_by_strategy={S01_VOL_BASELINE: True},
            calendar_expired_by_strategy={S01_VOL_BASELINE: True},
            iv_baseline_available_by_strategy={S01_VOL_BASELINE: False},
            account_snapshot_fresh=False,
            state_store_readable=False,
        )
        self.assert_skip(snapshot, SKIP_NLV_DEGRADED)
        failed = [
            event
            for event in self.execution_events()
            if event["strategy_id"] == S01_VOL_BASELINE
        ][-1]["payload"]["failed_checks"]
        self.assertEqual(
            failed,
            [
                "nlv_valid",
                "dirty_state",
                "unknown_broker_exposure",
                "halt_active",
                "calendar_expired",
                "iv_baseline_available",
                "account_snapshot_fresh",
                "state_store_readable",
            ],
        )

    def test_no_broker_or_order_calls(self) -> None:
        broker = mock.Mock()
        self.run_scan(self.snapshot(nlv_valid=False))
        broker.submit_order.assert_not_called()
        broker.placeOrder.assert_not_called()
        broker.cancelOrder.assert_not_called()
        self.assertEqual(self.order_ledger_text(), "")


class UnifiedSchedulerRunOnceTests(TmpCase):
    def test_market_open_scan_uses_injected_degraded_snapshot(self) -> None:
        scheduler = UnifiedScheduler(
            state_store=self.state_store,
            ledger=self.ledger,
            readiness_manager=self.manager,
            health_snapshot_provider=lambda: HealthSnapshot(
                account_snapshot_fresh=True,
                nlv_valid=False,
                state_store_readable=True,
                halt_active_by_strategy={},
                dirty_state_by_strategy={},
                unknown_broker_exposure_by_strategy={},
                calendar_expired_by_strategy={},
                iv_baseline_available_by_strategy={S01_VOL_BASELINE: True, S02_VOL_ENHANCED: True},
            ),
        )
        result = scheduler.run_job_once(JOB_MARKET_OPEN_SCAN)
        self.assertEqual(result.status, "skipped")
        self.assertIn("SIGNAL_SKIPPED", (self.root / "data/ledger/execution_ledger.jsonl").read_text())

    def test_stub_and_unknown_jobs(self) -> None:
        scheduler = UnifiedScheduler(
            state_store=self.state_store,
            ledger=self.ledger,
            readiness_manager=self.manager,
        )
        result = scheduler.run_job_once(JOB_S01_VOL_SCAN)
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.detail, "stub_not_implemented")
        with self.assertRaises(JobNotFoundError):
            scheduler.run_job_once("missing")

    def test_source_has_no_forbidden_lifecycle_calls(self) -> None:
        source = source_path("core", "scheduler.py").read_text()
        for forbidden in (
            "create_pending_position(",
            "execute_close(",
            "confirm_close_fill(",
            "broker.submit_order(",
            "placeOrder(",
            "cancelOrder(",
        ):
            self.assertNotIn(forbidden, source)


class CLITests(TmpCase):
    def run_module(self, module: str, *args: str) -> subprocess.CompletedProcess[str]:
        env = {
            **dict(os.environ),
            "PYTHONPATH": str(REPO_ROOT),
        }
        return subprocess.run(
            [sys.executable, "-m", module, *args],
            cwd=self.root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_scheduler_status_read_only(self) -> None:
        before_state = self.state_store.path.read_text()
        before_exec = (self.root / "data/ledger/execution_ledger.jsonl").read_text()
        result = self.run_module("algo_trader_unified.tools.scheduler_status")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(JOB_MARKET_OPEN_SCAN, result.stdout)
        self.assertEqual(before_state, self.state_store.path.read_text())
        self.assertEqual(before_exec, (self.root / "data/ledger/execution_ledger.jsonl").read_text())

    def test_run_market_open_scan_dry_run(self) -> None:
        result = self.run_module("algo_trader_unified.tools.run_market_open_scan")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("submit_order", result.stdout + result.stderr)


class RegressionGuards(unittest.TestCase):
    def test_no_direct_ledger_or_state_writes_in_phase3a_modules(self) -> None:
        for module_path in [
            source_path("core", "scheduler.py"),
            source_path("core", "readiness.py"),
            source_path("jobs", "readiness.py"),
        ]:
            source = module_path.read_text()
            self.assertNotIn(".jsonl", source, module_path)
            self.assertNotIn("write_text(", source, module_path)
            self.assertNotIn("open(", source, module_path)

    def test_s02_paper_only_and_no_commodity_strategy(self) -> None:
        from algo_trader_unified.config.variants import S02_CONFIG

        self.assertEqual(S02_CONFIG.execution_mode, "paper_only")
        all_sources = "\n".join(
            path.read_text()
            for path in PACKAGE_ROOT.rglob("*.py")
            if "__pycache__" not in str(path)
        )
        self.assertNotIn("commodity" + "_vrp", all_sources.lower())
