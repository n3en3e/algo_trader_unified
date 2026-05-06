from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.strategy_realism_report import (
    UNKNOWN_SKIP_REASON,
    build_strategy_realism_report,
    classify_skip_reason,
)
from algo_trader_unified.jobs.readiness import HealthSnapshot
from algo_trader_unified.tools import strategy_realism_report as tool


NY = ZoneInfo("America/New_York")
_DEFAULT_READINESS = object()


def event(event_type, timestamp, strategy_id=S01_VOL_BASELINE, payload=None):
    return {
        "event_id": f"evt_{event_type}_{timestamp}_{strategy_id}",
        "event_type": event_type,
        "timestamp": timestamp,
        "strategy_id": strategy_id,
        "execution_mode": "paper_only",
        "source_module": "test",
        "position_id": None,
        "opportunity_id": None,
        "payload": payload or {},
    }


class FakeLedgerReader:
    def __init__(self, root: Path) -> None:
        self.execution_ledger_path = root / "data/ledger/execution_ledger.jsonl"
        self.order_ledger_path = root / "data/ledger/order_ledger.jsonl"

    def read_events(self):
        raise AssertionError("report must stream path-backed ledger files")


class NoPathLedgerReader:
    def read_events(self):
        raise AssertionError("report must not fall back to read_events")


class FakeStateStore:
    def __init__(self) -> None:
        self.saved = False
        self.readiness = {
            S01_VOL_BASELINE: {
                "ready_for_entries": False,
                "reason": "SKIP_READINESS_NOT_EVALUATED",
                "dirty_state": True,
                "calendar_expired": False,
                "iv_baseline_available": True,
            },
            S02_VOL_ENHANCED: {
                "ready_for_entries": True,
                "reason": None,
                "dirty_state": False,
                "calendar_expired": False,
                "iv_baseline_available": True,
            },
        }
        self.positions = [
            {"position_id": "p1", "strategy_id": S01_VOL_BASELINE, "status": "open"},
            {"position_id": "p2", "strategy_id": S02_VOL_ENHANCED, "status": "closed"},
        ]
        self.order_intents = [
            {"intent_id": "i1", "strategy_id": S01_VOL_BASELINE, "status": "created"},
            {"intent_id": "i2", "strategy_id": S02_VOL_ENHANCED, "status": "submitted"},
            {"intent_id": "i3", "strategy_id": S02_VOL_ENHANCED, "status": "cancelled"},
        ]
        self.close_intents = [
            {"close_intent_id": "c1", "strategy_id": S02_VOL_ENHANCED, "status": "filled"},
        ]

    def get_readiness(self, strategy_id):
        return deepcopy(self.readiness.get(strategy_id))

    def list_positions(self):
        return deepcopy(self.positions)

    def list_order_intents(self):
        return deepcopy(self.order_intents)

    def list_close_intents(self):
        return deepcopy(self.close_intents)

    def save(self):
        self.saved = True
        raise AssertionError("strategy realism report must not save StateStore")


def health_snapshot(**overrides):
    base = HealthSnapshot(
        account_snapshot_fresh=True,
        nlv_valid=True,
        state_store_readable=True,
        halt_active_by_strategy={
            S01_VOL_BASELINE: False,
            S02_VOL_ENHANCED: False,
        },
        dirty_state_by_strategy={
            S01_VOL_BASELINE: False,
            S02_VOL_ENHANCED: False,
        },
        unknown_broker_exposure_by_strategy={
            S01_VOL_BASELINE: False,
            S02_VOL_ENHANCED: False,
        },
        calendar_expired_by_strategy={
            S01_VOL_BASELINE: False,
            S02_VOL_ENHANCED: False,
        },
        iv_baseline_available_by_strategy={
            S01_VOL_BASELINE: True,
            S02_VOL_ENHANCED: True,
        },
    )
    values = base.__dict__.copy()
    values.update(overrides)
    return HealthSnapshot(**values)


def write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in events),
        encoding="utf-8",
    )


class StrategyRealismReportTests(unittest.TestCase):
    def build(self, root: Path, state_store=None, readiness=_DEFAULT_READINESS, session=None):
        snapshot = health_snapshot() if readiness is _DEFAULT_READINESS else readiness
        return build_strategy_realism_report(
            ledger_reader=FakeLedgerReader(root),
            state_store=state_store or FakeStateStore(),
            readiness_provider=lambda: snapshot,
            snapshots_dir=root / "data/snapshots",
            halt_state_path=root / "data/state/halt_state.json",
            strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
            session_date=session or date(2026, 5, 5),
            now_provider=lambda: datetime(2026, 5, 5, 17, 0, tzinfo=NY),
        )

    def test_report_counts_required_fields_and_read_only_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(
                root / "data/ledger/execution_ledger.jsonl",
                [
                    event("SIGNAL_GENERATED", "2026-05-05T13:40:00+00:00", S01_VOL_BASELINE),
                    event(
                        "SIGNAL_SKIPPED",
                        "2026-05-05T14:00:00+00:00",
                        S01_VOL_BASELINE,
                        {"skip_reason": "SKIP_IV_RANK_BELOW_MIN: 12% < 15%"},
                    ),
                    event(
                        "SIGNAL_SKIPPED",
                        "2026-05-05T15:00:00+00:00",
                        S02_VOL_ENHANCED,
                        {"skip_reason": "SKIP_VIX_GATE: VIX 31 > threshold 28"},
                    ),
                    event("SIGNAL_SKIPPED", "2026-05-05T15:01:00+00:00", S02_VOL_ENHANCED),
                    event("ORDER_INTENT_CREATED", "2026-05-05T15:02:00+00:00", S02_VOL_ENHANCED),
                    event("SIGNAL_GENERATED", "2026-05-04T15:00:00+00:00", S02_VOL_ENHANCED),
                ],
            )
            write_jsonl(root / "data/ledger/order_ledger.jsonl", [])
            store = FakeStateStore()
            before = deepcopy(store.__dict__)

            report = self.build(root, state_store=store)

            self.assertTrue(report["dry_run"])
            self.assertTrue(report["strategy_realism_report"])
            self.assertTrue(report["success"])
            self.assertEqual(report["session_date"], "2026-05-05")
            self.assertEqual(
                report["strategy_ids"],
                [S01_VOL_BASELINE, S02_VOL_ENHANCED],
            )
            self.assertEqual(report["aggregate"]["total_signals_generated"], 1)
            self.assertEqual(report["aggregate"]["total_signals_skipped"], 3)
            self.assertEqual(
                report["aggregate"]["skip_reasons"][UNKNOWN_SKIP_REASON],
                1,
            )
            self.assertEqual(report["per_strategy"][S01_VOL_BASELINE]["signals_generated"], 1)
            self.assertEqual(report["per_strategy"][S01_VOL_BASELINE]["signals_skipped"], 1)
            self.assertEqual(report["per_strategy"][S02_VOL_ENHANCED]["signals_generated"], 0)
            self.assertEqual(report["per_strategy"][S02_VOL_ENHANCED]["signals_skipped"], 2)
            self.assertEqual(
                report["per_strategy"][S02_VOL_ENHANCED]["top_skip_reason"],
                "SKIP_VIX_GATE: VIX 31 > threshold 28",
            )
            self.assertEqual(report["per_strategy"][S01_VOL_BASELINE]["readiness_passed"], False)
            self.assertEqual(report["per_strategy"][S01_VOL_BASELINE]["dirty_state"], True)
            self.assertEqual(report["per_strategy"][S02_VOL_ENHANCED]["active_intents_count"], 2)
            self.assertEqual(report["per_strategy"][S01_VOL_BASELINE]["open_positions_count"], 1)
            self.assertEqual(report["readiness"]["account_snapshot_fresh"], True)
            self.assertEqual(report["readiness"]["nlv_valid"], True)
            self.assertEqual(report["readiness"]["halt_active"], False)
            self.assertEqual(
                report["safety"],
                {
                    "broker_calls_enabled": False,
                    "market_data_enabled": False,
                    "paper_live_orders_enabled": False,
                    "lifecycle_changes_enabled": False,
                },
            )
            self.assertEqual(before, store.__dict__)

    def test_missing_and_malformed_timestamps_do_not_crash_or_fail_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(
                root / "data/ledger/execution_ledger.jsonl",
                [
                    event("SIGNAL_GENERATED", "not-a-time", S01_VOL_BASELINE),
                    {**event("SIGNAL_GENERATED", "2026-05-05T15:00:00+00:00"), "timestamp": ""},
                    event("SIGNAL_GENERATED", "2026-05-05T15:00:00+00:00", S01_VOL_BASELINE),
                ],
            )
            report = self.build(root)
            self.assertTrue(report["success"])
            self.assertEqual(report["aggregate"]["total_signals_generated"], 1)
            self.assertTrue(
                any("timestamp" in message for message in report["errors"]),
                report["errors"],
            )

    def test_ny_session_filter_uses_zoneinfo_dst_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(
                root / "data/ledger/execution_ledger.jsonl",
                [
                    event("SIGNAL_GENERATED", "2026-03-08T04:30:00+00:00", S01_VOL_BASELINE),
                    event("SIGNAL_GENERATED", "2026-03-08T06:30:00+00:00", S01_VOL_BASELINE),
                ],
            )
            report = self.build(root, session=date(2026, 3, 8))
            self.assertEqual(report["aggregate"]["total_signals_generated"], 1)

    def test_no_events_missing_readiness_and_missing_snapshots_are_graceful(self) -> None:
        class MissingReadinessStore(FakeStateStore):
            def get_readiness(self, strategy_id):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_strategy_realism_report(
                ledger_reader=NoPathLedgerReader(),
                state_store=MissingReadinessStore(),
                readiness_provider=lambda: None,
                snapshots_dir=root / "data/snapshots",
                halt_state_path=root / "data/state/halt_state.json",
                strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
                session_date=date(2026, 5, 5),
                now_provider=lambda: datetime(2026, 5, 5, 17, 0, tzinfo=NY),
            )
            self.assertFalse(report["success"])
            self.assertIn("ledger paths unavailable from ledger_reader", report["errors"])
            self.assertEqual(report["aggregate"]["total_signals_generated"], 0)
            self.assertEqual(report["aggregate"]["total_signals_skipped"], 0)
            self.assertEqual(report["aggregate"]["skip_reasons"], {})
            self.assertEqual(
                report["readiness"]["missing_readiness_strategy_ids"],
                [S01_VOL_BASELINE, S02_VOL_ENHANCED],
            )
            self.assertIsNone(report["readiness"]["account_snapshot_fresh"])

    def test_halt_state_sets_halt_active_and_classifies_safety(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(
                root / "data/ledger/execution_ledger.jsonl",
                [
                    event(
                        "SIGNAL_SKIPPED",
                        "2026-05-05T15:00:00+00:00",
                        S01_VOL_BASELINE,
                        {"skip_reason": "SKIP_NEEDS_RECONCILIATION"},
                    ),
                ],
            )
            halt_path = root / "data/state/halt_state.json"
            halt_path.parent.mkdir(parents=True, exist_ok=True)
            halt_path.write_text(json.dumps({"scope": "account", "tier": "hard"}), encoding="utf-8")
            report = self.build(root)
            self.assertTrue(report["readiness"]["halt_active"])
            self.assertEqual(
                report["diagnostics"]["likely_blocker_category"],
                "halt_or_safety_problem",
            )

    def test_critical_local_read_failure_sets_success_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data/ledger/execution_ledger.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not-json}\n", encoding="utf-8")
            report = self.build(root)
            self.assertFalse(report["success"])
            self.assertTrue(any("invalid ledger JSON" in message for message in report["errors"]))

    def test_missing_ledger_paths_never_calls_read_events_and_fails_safely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_strategy_realism_report(
                ledger_reader=NoPathLedgerReader(),
                state_store=FakeStateStore(),
                readiness_provider=lambda: health_snapshot(),
                snapshots_dir=root / "data/snapshots",
                halt_state_path=root / "data/state/halt_state.json",
                strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
                session_date=date(2026, 5, 5),
                now_provider=lambda: datetime(2026, 5, 5, 17, 0, tzinfo=NY),
            )
            self.assertFalse(report["success"])
            self.assertIn("ledger paths unavailable from ledger_reader", report["errors"])

    def test_path_backed_report_streams_without_calling_read_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(
                root / "data/ledger/execution_ledger.jsonl",
                [
                    event("SIGNAL_GENERATED", "2026-05-05T15:00:00+00:00", S01_VOL_BASELINE),
                ],
            )
            report = self.build(root)
            self.assertTrue(report["success"])
            self.assertEqual(report["aggregate"]["total_signals_generated"], 1)

    def test_snapshot_mtime_fallback_uses_utc_aware_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots_dir = root / "data/snapshots"
            snapshots_dir.mkdir(parents=True)
            snapshot_path = snapshots_dir / "account.json"
            snapshot_path.write_text(json.dumps({"account": "local"}), encoding="utf-8")

            def report_with_mtime(mtime_utc: datetime) -> dict:
                mtime = mtime_utc.timestamp()
                os.utime(snapshot_path, (mtime, mtime))
                return build_strategy_realism_report(
                    ledger_reader=FakeLedgerReader(root),
                    state_store=FakeStateStore(),
                    readiness_provider=lambda: None,
                    snapshots_dir=snapshots_dir,
                    halt_state_path=root / "data/state/halt_state.json",
                    strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
                    session_date=date(2026, 5, 5),
                    now_provider=lambda: datetime(2026, 5, 5, 17, 0, tzinfo=NY),
                )

            fresh_report = report_with_mtime(
                datetime(2026, 5, 5, 20, 55, tzinfo=timezone.utc)
            )
            stale_report = report_with_mtime(
                datetime(2026, 5, 5, 20, 40, tzinfo=timezone.utc)
            )

            self.assertTrue(fresh_report["success"])
            self.assertTrue(fresh_report["readiness"]["account_snapshot_fresh"])
            self.assertTrue(stale_report["success"])
            self.assertFalse(stale_report["readiness"]["account_snapshot_fresh"])


class SkipReasonClassificationTests(unittest.TestCase):
    def test_substring_based_classification_handles_dynamic_context(self) -> None:
        self.assertEqual(
            classify_skip_reason("SKIP_READINESS_NOT_EVALUATED: cadence stale"),
            "readiness_problem",
        )
        self.assertEqual(
            classify_skip_reason("SKIP_STATESTORE_UNREADABLE"),
            "readiness_problem",
        )
        self.assertEqual(
            classify_skip_reason("SKIP_IV_BASELINE_MISSING"),
            "data_problem",
        )
        self.assertEqual(
            classify_skip_reason("missing VIX snapshot"),
            "data_problem",
        )
        self.assertEqual(
            classify_skip_reason("SKIP_IV_RANK_BELOW_MIN: 12% < 15%"),
            "strategy_filter_problem",
        )
        self.assertEqual(
            classify_skip_reason("SKIP_VIX_GATE: VIX 31 > threshold 28"),
            "strategy_filter_problem",
        )
        self.assertEqual(
            classify_skip_reason("delta filter rejected candidate"),
            "strategy_filter_problem",
        )
        self.assertEqual(
            classify_skip_reason("SKIP_HALTED"),
            "halt_or_safety_problem",
        )
        self.assertEqual(
            classify_skip_reason("UNKNOWN_SKIP_REASON"),
            "unknown",
        )


class StrategyRealismCliTests(unittest.TestCase):
    def test_missing_dry_run_only_exits_before_any_loader_or_factory(self) -> None:
        calls = []

        def factory(name):
            def _inner(*args, **kwargs):
                calls.append(name)
                return mock.Mock()

            return _inner

        err = io.StringIO()
        with redirect_stderr(err):
            code = tool.run_strategy_realism_report(
                [],
                state_store_factory=factory("state"),
                ledger_reader_factory=factory("ledger"),
                readiness_provider_factory=factory("readiness"),
                report_builder=factory("report"),
            )
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("--dry-run-only", err.getvalue())

    def test_cli_report_path_loads_only_local_readers_and_does_not_start_scheduler(self) -> None:
        calls = []
        report = {"success": True, "dry_run": True, "strategy_realism_report": True}

        def state_factory(path):
            calls.append("state")
            return FakeStateStore()

        def ledger_factory(root):
            calls.append("ledger")
            return mock.Mock()

        def readiness_factory(**kwargs):
            calls.append("readiness")
            return lambda: health_snapshot()

        def report_builder(**kwargs):
            calls.append("report")
            return report

        out = io.StringIO()
        with redirect_stdout(out):
            code = tool.run_strategy_realism_report(
                ["--dry-run-only"],
                state_store_factory=state_factory,
                ledger_reader_factory=ledger_factory,
                readiness_provider_factory=readiness_factory,
                report_builder=report_builder,
            )
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["state", "ledger", "readiness", "report"])
        self.assertEqual(json.loads(out.getvalue()), report)

    def test_default_cli_missing_state_does_not_create_statestore_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                code = tool.run_strategy_realism_report(
                    ["--dry-run-only", "--root", str(root)],
                )
            self.assertEqual(code, 0)
            self.assertFalse((root / "data/state/portfolio_state.json").exists())
            self.assertTrue(json.loads(stdout.getvalue())["success"])

    def test_report_modules_do_not_import_broker_market_data_systemd_or_scheduler(self) -> None:
        core_source = Path("algo_trader_unified/core/strategy_realism_report.py").read_text(
            encoding="utf-8"
        )
        tool_source = Path("algo_trader_unified/tools/strategy_realism_report.py").read_text(
            encoding="utf-8"
        )
        combined = core_source + "\n" + tool_source
        for forbidden in (
            "ib_insync",
            "reqMktData",
            "placeOrder",
            "IBKR",
            "systemd",
            "UnifiedScheduler",
            "scheduler_cadence",
        ):
            self.assertNotIn(forbidden, combined)


if __name__ == "__main__":
    unittest.main()
