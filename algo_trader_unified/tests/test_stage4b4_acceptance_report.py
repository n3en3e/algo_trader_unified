from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.scheduler import (
    JOB_DAILY_DIGEST,
    JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS,
    JOB_DRY_RUN_CONFIRM_FILLS,
    JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS,
    JOB_DRY_RUN_SUBMIT_PENDING_INTENTS,
    JOB_EOD_REVIEW,
    JOB_HEARTBEAT,
    JOB_KEEPALIVE,
    JOB_MARKET_OPEN_SCAN,
    JOB_RISK_MONITOR,
    JOB_S01_VOL_SCAN,
    JOB_S02_VOL_SCAN,
)
from algo_trader_unified.core.acceptance_report import build_dry_run_acceptance_report
from algo_trader_unified.core.halt_state_utils import halt_is_active
from algo_trader_unified.core.ledger_reader import LedgerReader
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.jobs.readiness import all_clear_health_snapshot
from algo_trader_unified.tools import daemon


EXPECTED_4A = [
    JOB_KEEPALIVE,
    JOB_RISK_MONITOR,
    JOB_HEARTBEAT,
    JOB_MARKET_OPEN_SCAN,
    JOB_S01_VOL_SCAN,
    JOB_S02_VOL_SCAN,
    JOB_EOD_REVIEW,
    JOB_DAILY_DIGEST,
]

EXPECTED_4B = [
    JOB_DRY_RUN_SUBMIT_PENDING_INTENTS,
    JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS,
    JOB_DRY_RUN_CONFIRM_FILLS,
    JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS,
]


class HaltStateUtilsTests(unittest.TestCase):
    def test_halt_is_active_preserves_existing_daemon_report_semantics(self) -> None:
        self.assertFalse(halt_is_active(None))
        self.assertFalse(halt_is_active({}))
        self.assertFalse(halt_is_active({"tier": "hard", "resumed": True}))
        self.assertFalse(halt_is_active({"tier": "info"}))
        self.assertTrue(halt_is_active({"tier": "soft"}))
        self.assertTrue(halt_is_active({"tier": "hard"}))


class FakeLedger:
    def __init__(self) -> None:
        self.events = []

    def append(self, **kwargs):
        self.events.append(kwargs)
        return "evt_fake"


class AcceptanceReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.snapshots_dir = self.root / "data/snapshots"
        self.halt_state_path = self.root / "data/state/halt_state.json"
        self.state_store = StateStore(self.root / "data/state/portfolio_state.json")
        self.now = datetime(2026, 5, 5, 14, 0, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def readiness_provider(self):
        return all_clear_health_snapshot([S01_VOL_BASELINE, S02_VOL_ENHANCED])

    def report(self):
        return build_dry_run_acceptance_report(
            state_store=self.state_store,
            ledger_reader=LedgerReader.from_root(self.root),
            readiness_provider=self.readiness_provider,
            snapshots_dir=self.snapshots_dir,
            halt_state_path=self.halt_state_path,
            scheduler_builder=__import__(
                "algo_trader_unified.core.scheduler_cadence",
                fromlist=["build_scheduler"],
            ).build_scheduler,
            now_provider=lambda: self.now,
        )

    def test_report_includes_required_dry_run_sections_and_is_json_safe(self) -> None:
        self.snapshots_dir.mkdir(parents=True)
        (self.snapshots_dir / "account.json").write_text(
            json.dumps({"timestamp": (self.now - timedelta(minutes=5)).isoformat()}),
            encoding="utf-8",
        )
        (self.snapshots_dir / "digest_2026-05-05.txt").write_text(
            "digest\n",
            encoding="utf-8",
        )
        report = self.report()

        self.assertTrue(report["dry_run"])
        self.assertTrue(report["acceptance_report"])
        self.assertEqual(report["expected_4a_jobs"], EXPECTED_4A)
        self.assertEqual(report["expected_4b_lifecycle_jobs"], EXPECTED_4B)
        self.assertEqual(report["scheduler"]["jobs_registered_without_triggers"], [])
        self.assertEqual(report["scheduler"]["observation_jobs_registered"], EXPECTED_4A)
        self.assertEqual(
            report["scheduler"]["lifecycle_jobs_registered"],
            [*EXPECTED_4A, *EXPECTED_4B],
        )
        self.assertFalse(report["scheduler"]["lifecycle_pipeline_enabled_by_default"])
        self.assertTrue(report["startup_gate"]["dry_run_only_required"])
        self.assertFalse(report["startup_gate"]["halt_active"])
        self.assertEqual(report["state"]["open_positions_count"], 0)
        self.assertEqual(report["state"]["active_intents_count"], 0)
        self.assertTrue(report["snapshots"]["account_snapshot_fresh"])
        self.assertTrue(report["snapshots"]["latest_digest_path"].endswith("digest_2026-05-05.txt"))
        self.assertEqual(
            report["safety"],
            {
                "broker_calls_enabled": False,
                "market_data_enabled": False,
                "systemd_enabled": False,
                "paper_live_orders_enabled": False,
            },
        )
        self.assertTrue(report["success"])
        json.dumps(report)

    def test_report_observes_halt_without_writing_ledger_events(self) -> None:
        self.halt_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.halt_state_path.write_text(
            json.dumps({"scope": "account", "tier": "hard", "reason": "operator"}),
            encoding="utf-8",
        )
        ledger = FakeLedger()
        report = build_dry_run_acceptance_report(
            state_store=self.state_store,
            ledger_reader=LedgerReader.from_root(self.root),
            readiness_provider=self.readiness_provider,
            snapshots_dir=self.snapshots_dir,
            halt_state_path=self.halt_state_path,
            scheduler_builder=__import__(
                "algo_trader_unified.core.scheduler_cadence",
                fromlist=["build_scheduler"],
            ).build_scheduler,
            now_provider=lambda: self.now,
        )
        self.assertTrue(report["startup_gate"]["halt_active"])
        self.assertEqual(ledger.events, [])

    def test_report_uses_ledger_size_instead_of_full_count(self) -> None:
        ledger_dir = self.root / "data/ledger"
        ledger_dir.mkdir(parents=True)
        (ledger_dir / "order_ledger.jsonl").write_text(
            json.dumps({"timestamp": "2026-05-05T13:00:00+00:00"}) + "\n",
            encoding="utf-8",
        )
        (ledger_dir / "execution_ledger.jsonl").write_text("", encoding="utf-8")

        report = self.report()

        self.assertEqual(report["ledger"]["total_event_count"], "unavailable")
        self.assertGreater(report["ledger"]["ledger_size_bytes"], 0)
        self.assertEqual(
            report["ledger"]["latest_event_timestamp"],
            "2026-05-05T13:00:00+00:00",
        )


class Spy:
    def __init__(self, name, calls, result=None):
        self.name = name
        self.calls = calls
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append(self.name)
        return self.result


class FakeDiagnosticProvider:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, state_store, halt_state):
        self.calls.append("diagnostics")
        return self

    def run(self):
        return daemon.DiagnosticResult(True)


class DaemonAcceptanceCliTests(unittest.TestCase):
    def run_daemon(self, argv, *, calls, report=None):
        state_store = object()
        ledger_reader = object()
        readiness_provider = object()
        report = report or {
            "dry_run": True,
            "acceptance_report": True,
            "startup_gate": {},
            "errors": [],
            "success": True,
        }

        def readiness_provider_factory(**kwargs):
            calls.append("readiness_provider")
            self.assertIs(kwargs["state_store"], state_store)
            return readiness_provider

        def acceptance_report_builder(**kwargs):
            calls.append("acceptance_report")
            self.assertIs(kwargs["state_store"], state_store)
            self.assertIs(kwargs["ledger_reader"], ledger_reader)
            self.assertIs(kwargs["readiness_provider"], readiness_provider)
            return report

        return daemon.run_daemon(
            argv,
            env_loader=Spy("env", calls),
            config_loader=Spy("config", calls),
            ledger_factory=Spy("ledger", calls),
            state_store_factory=Spy("statestore", calls, state_store),
            ledger_reader_factory=Spy("ledger_reader", calls, ledger_reader),
            halt_loader=Spy("halt", calls, None),
            diagnostic_provider=FakeDiagnosticProvider(calls),
            scheduler_factory=Spy("scheduler", calls),
            readiness_provider_factory=readiness_provider_factory,
            acceptance_report_builder=acceptance_report_builder,
        )

    def test_acceptance_report_requires_dry_run_before_any_loader_or_factory(self) -> None:
        calls = []
        err = io.StringIO()
        with redirect_stderr(err):
            code = self.run_daemon(["--acceptance-report"], calls=calls)
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("--dry-run-only", err.getvalue())

    def test_acceptance_report_uses_read_only_path_and_does_not_start_scheduler(self) -> None:
        calls = []
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = self.run_daemon(
                ["--dry-run-only", "--acceptance-report"],
                calls=calls,
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            calls,
            [
                "statestore",
                "ledger_reader",
                "halt",
                "readiness_provider",
                "acceptance_report",
                "diagnostics",
            ],
        )
        self.assertNotIn("env", calls)
        self.assertNotIn("config", calls)
        self.assertNotIn("ledger", calls)
        self.assertNotIn("scheduler", calls)
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["acceptance_report"])

    def test_acceptance_report_failure_returns_nonzero(self) -> None:
        calls = []
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = self.run_daemon(
                ["--dry-run-only", "--acceptance-report"],
                calls=calls,
                report={
                    "dry_run": True,
                    "acceptance_report": True,
                    "startup_gate": {},
                    "errors": ["boom"],
                    "success": False,
                },
            )
        self.assertEqual(code, 1)
        self.assertIn("acceptance_report", calls)

    def test_acceptance_report_rejects_other_bounded_modes_before_loaders(self) -> None:
        calls = []
        err = io.StringIO()
        with redirect_stderr(err):
            code = self.run_daemon(
                ["--dry-run-only", "--acceptance-report", "--smoke-cycles", "1"],
                calls=calls,
            )
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("acceptance report", err.getvalue())


if __name__ == "__main__":
    unittest.main()
