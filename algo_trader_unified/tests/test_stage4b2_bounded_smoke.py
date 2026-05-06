from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.scheduler import (
    JOB_DAILY_DIGEST,
    JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS,
    JOB_DRY_RUN_CONFIRM_FILLS,
    JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS,
    JOB_DRY_RUN_EOD_INTENT_CLEANUP,
    JOB_DRY_RUN_EXPIRE_INTENTS,
    JOB_DRY_RUN_SUBMIT_PENDING_INTENTS,
    JOB_EOD_REVIEW,
    JOB_HEARTBEAT,
    JOB_KEEPALIVE,
    JOB_MARKET_OPEN_SCAN,
    JOB_RISK_MONITOR,
    JOB_S01_VOL_SCAN,
    JOB_S02_VOL_SCAN,
)
from algo_trader_unified.core import scheduler_cadence
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.jobs.readiness import all_clear_health_snapshot
from algo_trader_unified.tools import daemon


STAGE4A_SMOKE_ORDER = [
    JOB_KEEPALIVE,
    JOB_RISK_MONITOR,
    JOB_HEARTBEAT,
    JOB_MARKET_OPEN_SCAN,
    JOB_S01_VOL_SCAN,
    JOB_S02_VOL_SCAN,
    JOB_EOD_REVIEW,
    JOB_DAILY_DIGEST,
]

STAGE4B2_LIFECYCLE_ORDER = [
    JOB_DRY_RUN_SUBMIT_PENDING_INTENTS,
    JOB_DRY_RUN_EXPIRE_INTENTS,
    JOB_DRY_RUN_EOD_INTENT_CLEANUP,
]


class FakeLedger:
    def __init__(self) -> None:
        self.events = []

    def append(self, **kwargs):
        self.events.append(kwargs)
        return f"evt_{len(self.events)}"


class AdvancingNow:
    def __init__(self) -> None:
        self.current = datetime(2026, 5, 5, 13, 35, tzinfo=timezone.utc)
        self.calls = 0

    def __call__(self) -> datetime:
        value = self.current + timedelta(seconds=self.calls)
        self.calls += 1
        return value


class SmokeCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.snapshots_dir = self.root / "data/snapshots"
        self.halt_state_path = self.root / "data/state/halt_state.json"
        self.state_store = StateStore(self.root / "data/state/portfolio_state.json")
        self.ledger = FakeLedger()
        self.now = AdvancingNow()
        self.sleep_calls = []

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def ready_provider(self):
        return all_clear_health_snapshot([S01_VOL_BASELINE, S02_VOL_ENHANCED])

    def run_smoke(self, *, cycles=1, include_lifecycle_pipeline=False):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return scheduler_cadence.run_bounded_dry_run_smoke(
                state_store=self.state_store,
                ledger=self.ledger,
                readiness_provider=self.ready_provider,
                snapshots_dir=self.snapshots_dir,
                halt_state_path=self.halt_state_path,
                cycles=cycles,
                include_lifecycle_pipeline=include_lifecycle_pipeline,
                now_provider=self.now,
                sleep_fn=self.sleep_calls.append,
            )


class BoundedSmokeRunnerTests(SmokeCase):
    def test_cycles_one_without_lifecycle_runs_only_stage4a_jobs(self) -> None:
        summary = self.run_smoke()
        self.assertTrue(summary["success"])
        self.assertIs(summary["dry_run"], True)
        self.assertEqual(summary["cycles_requested"], 1)
        self.assertEqual(summary["cycles_completed"], 1)
        self.assertFalse(summary["include_lifecycle_pipeline"])
        self.assertEqual(list(summary["jobs_run"]), STAGE4A_SMOKE_ORDER)
        self.assertEqual(summary["jobs_run"], {job_id: 1 for job_id in STAGE4A_SMOKE_ORDER})
        self.assertEqual([item["job_id"] for item in summary["job_results"]], STAGE4A_SMOKE_ORDER)
        json.dumps(summary)

    def test_lifecycle_flag_adds_only_intent_level_stage4b_jobs_before_eod_and_digest(self) -> None:
        summary = self.run_smoke(include_lifecycle_pipeline=True)
        expected = [
            JOB_KEEPALIVE,
            JOB_RISK_MONITOR,
            JOB_HEARTBEAT,
            JOB_MARKET_OPEN_SCAN,
            JOB_S01_VOL_SCAN,
            JOB_S02_VOL_SCAN,
            *STAGE4B2_LIFECYCLE_ORDER,
            JOB_EOD_REVIEW,
            JOB_DAILY_DIGEST,
        ]
        self.assertTrue(summary["success"])
        self.assertEqual([item["job_id"] for item in summary["job_results"]], expected)
        self.assertEqual(summary["jobs_run"], {job_id: 1 for job_id in expected})
        forbidden = {
            JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS,
            JOB_DRY_RUN_CONFIRM_FILLS,
            JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS,
        }
        self.assertFalse(forbidden & set(summary["jobs_run"]))
        for job_id in STAGE4B2_LIFECYCLE_ORDER:
            result = next(item for item in summary["job_results"] if item["job_id"] == job_id)
            self.assertIs(result["result"]["dry_run"], True)

    def test_multi_cycle_uses_injected_sleep_and_advancing_now_provider(self) -> None:
        summary = self.run_smoke(cycles=2)
        self.assertTrue(summary["success"])
        self.assertEqual(summary["cycles_completed"], 2)
        self.assertEqual(self.sleep_calls, [0])
        self.assertGreaterEqual(self.now.calls, 4)
        self.assertEqual(summary["jobs_run"][JOB_KEEPALIVE], 2)
        self.assertEqual(summary["jobs_run"][JOB_DAILY_DIGEST], 2)

    def test_no_work_lifecycle_jobs_noop_without_crashing(self) -> None:
        summary = self.run_smoke(include_lifecycle_pipeline=True)
        self.assertEqual(summary["errors"], [])
        lifecycle_results = [
            item["result"]
            for item in summary["job_results"]
            if item["job_id"] in STAGE4B2_LIFECYCLE_ORDER
        ]
        self.assertEqual(len(lifecycle_results), 3)
        self.assertTrue(all(result["dry_run"] is True for result in lifecycle_results))

    def test_unexpected_job_error_records_compact_error_and_stops(self) -> None:
        with mock.patch.object(
            scheduler_cadence,
            "run_eod_review",
            side_effect=RuntimeError("boom"),
        ):
            summary = self.run_smoke(cycles=2)
        self.assertFalse(summary["success"])
        self.assertEqual(summary["cycles_completed"], 0)
        self.assertEqual(summary["errors"], [
            {
                "cycle": 1,
                "job_id": JOB_EOD_REVIEW,
                "error_type": "RuntimeError",
                "message": "boom",
            }
        ])
        self.assertEqual(summary["jobs_run"][JOB_EOD_REVIEW], 0)
        self.assertEqual(summary["jobs_run"][JOB_DAILY_DIGEST], 0)

    def test_cycles_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive finite integer"):
            self.run_smoke(cycles=0)

    def test_smoke_does_not_mutate_trigger_globals_or_import_live_boundaries(self) -> None:
        source = Path("algo_trader_unified/core/scheduler_cadence.py").read_text(encoding="utf-8")
        forbidden = (
            "ENABLE_SCHEDULER_TRIGGERS",
            "ib_insync",
            "reqMktData",
            "placeOrder",
            "core.broker",
            "systemd",
            "BlockingScheduler",
        )
        for token in forbidden:
            self.assertNotIn(token, source)


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


class DaemonSmokeCliTests(unittest.TestCase):
    def run_daemon(self, argv, *, calls, smoke_summary=None):
        state_store = mock.Mock()
        ledger = FakeLedger()
        readiness_provider = object()
        smoke_summary = smoke_summary or {"dry_run": True, "success": True}

        def readiness_provider_factory(**kwargs):
            calls.append("readiness_provider")
            self.assertIs(kwargs["state_store"], state_store)
            return readiness_provider

        def smoke_runner(**kwargs):
            calls.append("smoke")
            self.assertIs(kwargs["state_store"], state_store)
            self.assertIs(kwargs["ledger"], ledger)
            self.assertIs(kwargs["readiness_provider"], readiness_provider)
            return smoke_summary

        return daemon.run_daemon(
            argv,
            env_loader=Spy("env", calls),
            config_loader=Spy("config", calls),
            ledger_factory=Spy("ledger", calls, ledger),
            state_store_factory=Spy("statestore", calls, state_store),
            halt_loader=Spy("halt", calls, None),
            diagnostic_provider=FakeDiagnosticProvider(calls),
            scheduler_factory=Spy("scheduler", calls),
            readiness_provider_factory=readiness_provider_factory,
            smoke_runner=smoke_runner,
        )

    def test_smoke_requires_dry_run_before_any_loader_or_factory(self) -> None:
        calls = []
        err = io.StringIO()
        with redirect_stderr(err):
            code = self.run_daemon(["--smoke-cycles", "1"], calls=calls)
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("--dry-run-only", err.getvalue())

    def test_non_positive_smoke_cycles_exit_before_any_loader_or_factory(self) -> None:
        calls = []
        err = io.StringIO()
        with redirect_stderr(err):
            code = self.run_daemon(["--dry-run-only", "--smoke-cycles", "0"], calls=calls)
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("--smoke-cycles", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_smoke_cli_runs_bounded_path_and_not_scheduler_start(self) -> None:
        calls = []
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = self.run_daemon(
                [
                    "--dry-run-only",
                    "--smoke-cycles",
                    "1",
                    "--enable-lifecycle-pipeline",
                ],
                calls=calls,
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            calls,
            [
                "env",
                "config",
                "ledger",
                "statestore",
                "halt",
                "diagnostics",
                "readiness_provider",
                "smoke",
            ],
        )
        self.assertNotIn("scheduler", calls)
        self.assertTrue(json.loads(out.getvalue())["dry_run"])

    def test_smoke_cli_returns_nonzero_when_summary_fails(self) -> None:
        calls = []
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = self.run_daemon(
                ["--dry-run-only", "--smoke-cycles", "1"],
                calls=calls,
                smoke_summary={"dry_run": True, "success": False},
            )
        self.assertEqual(code, 1)
        self.assertIn("smoke", calls)


if __name__ == "__main__":
    unittest.main()
