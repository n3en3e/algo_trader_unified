from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
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


STAGE4A_JOB_IDS = [
    JOB_KEEPALIVE,
    JOB_RISK_MONITOR,
    JOB_HEARTBEAT,
    JOB_MARKET_OPEN_SCAN,
    JOB_S01_VOL_SCAN,
    JOB_S02_VOL_SCAN,
    JOB_EOD_REVIEW,
    JOB_DAILY_DIGEST,
]

STAGE4B_LIFECYCLE_JOB_IDS = [
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


class FakeScheduler:
    def __init__(self, *, start_exc=None, shutdown_exc=None) -> None:
        self.jobs = []
        self.start_calls = 0
        self.shutdown_calls = []
        self.start_exc = start_exc
        self.shutdown_exc = shutdown_exc

    def add_job(self, func, **kwargs) -> None:
        self.jobs.append({"func": func, **kwargs})

    def start(self) -> None:
        self.start_calls += 1
        if self.start_exc is not None:
            raise self.start_exc

    def shutdown(self, wait=True) -> None:
        self.shutdown_calls.append({"wait": wait})
        if self.shutdown_exc is not None:
            raise self.shutdown_exc


class FakeClock:
    def __init__(self) -> None:
        self.current = 0.0
        self.sleep_calls = []

    def monotonic(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.current += seconds


class ForegroundCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.snapshots_dir = self.root / "data/snapshots"
        self.halt_state_path = self.root / "data/state/halt_state.json"
        self.state_store = StateStore(self.root / "data/state/portfolio_state.json")
        self.ledger = FakeLedger()
        self.scheduler = FakeScheduler()
        self.clock = FakeClock()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def readiness_provider(self):
        return all_clear_health_snapshot([S01_VOL_BASELINE, S02_VOL_ENHANCED])

    def run_foreground(
        self,
        *,
        runtime_seconds=0.01,
        enable_triggers=True,
        include_lifecycle_pipeline=False,
        scheduler=None,
        sleep_fn=None,
    ):
        return scheduler_cadence.run_bounded_foreground_scheduler(
            state_store=self.state_store,
            ledger=self.ledger,
            readiness_provider=self.readiness_provider,
            snapshots_dir=self.snapshots_dir,
            halt_state_path=self.halt_state_path,
            runtime_seconds=runtime_seconds,
            enable_triggers=enable_triggers,
            include_lifecycle_pipeline=include_lifecycle_pipeline,
            scheduler_factory=lambda: scheduler or self.scheduler,
            now_provider=lambda: datetime(2026, 5, 5, 13, 35, tzinfo=timezone.utc),
            sleep_fn=sleep_fn or self.clock.sleep,
            monotonic_fn=self.clock.monotonic,
        )


class ForegroundRunnerTests(ForegroundCase):
    def test_runtime_seconds_must_be_positive_and_finite(self) -> None:
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive finite number"):
                    self.run_foreground(runtime_seconds=value)

    def test_bounded_run_starts_and_shutdowns_scheduler_once(self) -> None:
        summary = self.run_foreground()
        self.assertTrue(summary["success"])
        self.assertIs(summary["dry_run"], True)
        self.assertIs(summary["foreground_run"], True)
        self.assertTrue(summary["scheduler_started"])
        self.assertTrue(summary["scheduler_shutdown"])
        self.assertFalse(summary["interrupted"])
        self.assertEqual(self.scheduler.start_calls, 1)
        self.assertEqual(self.scheduler.shutdown_calls, [{"wait": True}])
        self.assertGreaterEqual(summary["elapsed_seconds"], 0.01)
        self.assertEqual(self.clock.sleep_calls, [0.01])
        json.dumps(summary)

    def test_summary_includes_required_observability_fields(self) -> None:
        summary = self.run_foreground()
        self.assertTrue(
            {
                "dry_run",
                "foreground_run",
                "runtime_seconds_requested",
                "elapsed_seconds",
                "enable_triggers",
                "include_lifecycle_pipeline",
                "scheduler_started",
                "scheduler_shutdown",
                "jobs_registered",
                "interrupted",
                "errors",
                "success",
            }.issubset(summary)
        )
        self.assertIs(summary["dry_run"], True)
        self.assertIs(summary["foreground_run"], True)
        self.assertIsInstance(summary["errors"], list)
        json.dumps(summary)

    def test_enable_triggers_false_registers_zero_jobs(self) -> None:
        summary = self.run_foreground(enable_triggers=False)
        self.assertEqual(summary["jobs_registered"], [])

    def test_enable_triggers_true_without_lifecycle_registers_stage4a_jobs(self) -> None:
        summary = self.run_foreground(enable_triggers=True, include_lifecycle_pipeline=False)
        self.assertEqual(summary["jobs_registered"], STAGE4A_JOB_IDS)

    def test_enable_lifecycle_pipeline_registers_stage4a_plus_intent_level_lifecycle_jobs(self) -> None:
        summary = self.run_foreground(enable_triggers=True, include_lifecycle_pipeline=True)
        self.assertEqual(
            summary["jobs_registered"],
            [*STAGE4A_JOB_IDS, *STAGE4B_LIFECYCLE_JOB_IDS],
        )
        forbidden = {
            JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS,
            JOB_DRY_RUN_CONFIRM_FILLS,
            JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS,
        }
        self.assertFalse(forbidden & set(summary["jobs_registered"]))

    def test_keyboard_interrupt_during_wait_gracefully_shutdowns(self) -> None:
        def interrupting_sleep(seconds):
            self.clock.sleep_calls.append(seconds)
            raise KeyboardInterrupt

        summary = self.run_foreground(runtime_seconds=5, sleep_fn=interrupting_sleep)
        self.assertTrue(summary["success"])
        self.assertTrue(summary["interrupted"])
        self.assertEqual(self.scheduler.start_calls, 1)
        self.assertEqual(self.scheduler.shutdown_calls, [{"wait": True}])
        self.assertEqual(summary["errors"], [])

    def test_scheduler_start_failure_returns_unsuccessful_summary(self) -> None:
        scheduler = FakeScheduler(start_exc=RuntimeError("start failed"))
        summary = self.run_foreground(scheduler=scheduler)
        self.assertFalse(summary["success"])
        self.assertFalse(summary["scheduler_started"])
        self.assertFalse(summary["scheduler_shutdown"])
        self.assertEqual(scheduler.start_calls, 1)
        self.assertEqual(scheduler.shutdown_calls, [])
        self.assertEqual(summary["errors"][0]["phase"], "start")

    def test_scheduler_shutdown_failure_returns_unsuccessful_summary(self) -> None:
        scheduler = FakeScheduler(shutdown_exc=RuntimeError("shutdown failed"))
        summary = self.run_foreground(scheduler=scheduler)
        self.assertFalse(summary["success"])
        self.assertTrue(summary["scheduler_started"])
        self.assertFalse(summary["scheduler_shutdown"])
        self.assertEqual(scheduler.shutdown_calls, [{"wait": True}])
        self.assertEqual(
            summary["errors"],
            [
                {
                    "phase": "shutdown",
                    "error_type": "RuntimeError",
                    "message": "shutdown failed",
                }
            ],
        )

    def test_foreground_source_avoids_blocking_scheduler_trap_and_live_boundaries(self) -> None:
        source = Path("algo_trader_unified/core/scheduler_cadence.py").read_text(
            encoding="utf-8"
        )
        forbidden = (
            "BlockingScheduler",
            "ib_insync",
            "reqMktData",
            "placeOrder",
            "core.broker",
            "systemd",
            "ENABLE_SCHEDULER_TRIGGERS",
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


class DaemonForegroundCliTests(unittest.TestCase):
    def run_daemon(self, argv, *, calls, foreground_summary=None):
        state_store = mock.Mock()
        ledger = FakeLedger()
        readiness_provider = object()
        foreground_summary = foreground_summary or {
            "dry_run": True,
            "foreground_run": True,
            "success": True,
        }

        def readiness_provider_factory(**kwargs):
            calls.append("readiness_provider")
            self.assertIs(kwargs["state_store"], state_store)
            return readiness_provider

        def foreground_runner(**kwargs):
            calls.append("foreground")
            self.assertIs(kwargs["state_store"], state_store)
            self.assertIs(kwargs["ledger"], ledger)
            self.assertIs(kwargs["readiness_provider"], readiness_provider)
            return foreground_summary

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
            foreground_runner=foreground_runner,
        )

    def test_foreground_requires_dry_run_before_any_loader_or_factory(self) -> None:
        calls = []
        err = io.StringIO()
        with redirect_stderr(err):
            code = self.run_daemon(["--foreground-runtime-seconds", "1"], calls=calls)
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("--dry-run-only", err.getvalue())

    def test_non_positive_foreground_runtime_exits_before_loaders(self) -> None:
        calls = []
        err = io.StringIO()
        with redirect_stderr(err):
            code = self.run_daemon(
                ["--dry-run-only", "--foreground-runtime-seconds", "0"],
                calls=calls,
            )
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("--foreground-runtime-seconds", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_foreground_cli_rejects_concurrent_smoke_mode_before_loaders(self) -> None:
        calls = []
        err = io.StringIO()
        with redirect_stderr(err):
            code = self.run_daemon(
                [
                    "--dry-run-only",
                    "--smoke-cycles",
                    "1",
                    "--foreground-runtime-seconds",
                    "1",
                ],
                calls=calls,
            )
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("choose either", err.getvalue())

    def test_foreground_cli_runs_bounded_path_and_not_default_scheduler(self) -> None:
        calls = []
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = self.run_daemon(
                [
                    "--dry-run-only",
                    "--foreground-runtime-seconds",
                    "1",
                    "--enable-triggers",
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
                "foreground",
            ],
        )
        self.assertNotIn("scheduler", calls)
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["dry_run"])
        self.assertTrue(payload["foreground_run"])

    def test_foreground_cli_returns_nonzero_when_summary_fails(self) -> None:
        calls = []
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = self.run_daemon(
                ["--dry-run-only", "--foreground-runtime-seconds", "1"],
                calls=calls,
                foreground_summary={
                    "dry_run": True,
                    "foreground_run": True,
                    "success": False,
                },
            )
        self.assertEqual(code, 1)
        self.assertIn("foreground", calls)

    def test_existing_signal_shutdown_path_still_uses_wait_true(self) -> None:
        scheduler = mock.Mock()
        made_threads = []

        class FakeThread:
            def __init__(self, target, kwargs, daemon):
                self.target = target
                self.kwargs = kwargs
                self.daemon = daemon
                made_threads.append(self)

            def start(self):
                self.target(**self.kwargs)

            def join(self, timeout):
                self.timeout = timeout

            def is_alive(self):
                return False

        code = daemon._shutdown_scheduler(scheduler, thread_factory=FakeThread)
        self.assertEqual(code, 0)
        scheduler.shutdown.assert_called_once_with(wait=True)
        self.assertTrue(made_threads[0].daemon)


if __name__ == "__main__":
    unittest.main()
