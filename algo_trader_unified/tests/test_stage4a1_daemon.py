from __future__ import annotations

import io
import signal
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.scheduler import SCHEDULER_SHUTDOWN_TIMEOUT_SEC
from algo_trader_unified.core.state_store import StateStore, StateStoreCorruptError
from algo_trader_unified.tools import daemon


class Spy:
    def __init__(self, name, calls, result=None, exc=None):
        self.name = name
        self.calls = calls
        self.result = result
        self.exc = exc

    def __call__(self, *args, **kwargs):
        self.calls.append(self.name)
        if self.exc is not None:
            raise self.exc
        return self.result


class FakeProvider:
    def __init__(self, calls, result):
        self.calls = calls
        self.result = result

    def __call__(self, state_store, halt_state):
        self.calls.append("diagnostics")
        return self

    def run(self):
        return self.result


class FakeLedger:
    def __init__(self, calls=None):
        self.calls = calls if calls is not None else []
        self.events = []

    def append(self, **kwargs):
        self.calls.append("ledger.append")
        self.events.append(kwargs)
        return "evt_fake"


class FakeScheduler:
    def __init__(self, calls):
        self.calls = calls
        self.shutdown_calls = []

    def start(self):
        self.calls.append("scheduler.start")

    def wait_until_stopped(self):
        self.calls.append("scheduler.wait")

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)


class DaemonStartupTests(unittest.TestCase):
    def test_missing_dry_run_only_exits_before_any_loader_or_factory(self) -> None:
        calls = []
        err = io.StringIO()
        with redirect_stderr(err):
            code = daemon.run_daemon(
                [],
                env_loader=Spy("env", calls),
                config_loader=Spy("config", calls),
                ledger_factory=Spy("ledger", calls),
                state_store_factory=Spy("statestore", calls),
                halt_loader=Spy("halt", calls),
                diagnostic_provider=Spy("diagnostics", calls),
                scheduler_factory=Spy("scheduler", calls),
            )
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("--dry-run-only", err.getvalue())

    def test_startup_gate_order_is_strict(self) -> None:
        calls = []
        ledger = FakeLedger(calls)
        scheduler = FakeScheduler(calls)

        def scheduler_factory(**kwargs):
            calls.append("scheduler")
            return scheduler

        code = daemon.run_daemon(
            ["--dry-run-only"],
            env_loader=Spy("env", calls),
            config_loader=Spy("config", calls),
            ledger_factory=Spy("ledger", calls, ledger),
            state_store_factory=Spy("statestore", calls, object()),
            halt_loader=Spy("halt", calls, None),
            diagnostic_provider=FakeProvider(calls, daemon.DiagnosticResult(True)),
            scheduler_factory=scheduler_factory,
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
                "scheduler",
                "scheduler.start",
                "scheduler.wait",
            ],
        )

    def test_corrupt_statestore_exits_and_scheduler_is_not_created(self) -> None:
        calls = []
        err = io.StringIO()
        with redirect_stderr(err):
            code = daemon.run_daemon(
                ["--dry-run-only"],
                env_loader=Spy("env", calls),
                config_loader=Spy("config", calls),
                ledger_factory=Spy("ledger", calls, FakeLedger()),
                state_store_factory=Spy(
                    "statestore",
                    calls,
                    exc=StateStoreCorruptError("StateStore JSON is corrupt"),
                ),
                halt_loader=Spy("halt", calls, None),
                diagnostic_provider=FakeProvider(calls, daemon.DiagnosticResult(True)),
                scheduler_factory=Spy("scheduler", calls),
            )
        self.assertEqual(code, 1)
        self.assertNotIn("scheduler", calls)
        self.assertIn("StateStore JSON is corrupt", err.getvalue())

    def test_active_halt_blocks_startup_and_does_not_emit_halt_triggered(self) -> None:
        calls = []
        ledger = FakeLedger(calls)
        halt_state = {
            "scope": "strategy",
            "id": "S01_VOL_BASELINE",
            "tier": "hard",
            "reason": "operator_halt",
            "halt_event_id": "evt_halt",
        }
        err = io.StringIO()
        with redirect_stderr(err):
            code = daemon.run_daemon(
                ["--dry-run-only"],
                env_loader=Spy("env", calls),
                config_loader=Spy("config", calls),
                ledger_factory=Spy("ledger", calls, ledger),
                state_store_factory=Spy("statestore", calls, object()),
                halt_loader=Spy("halt", calls, halt_state),
                diagnostic_provider=FakeProvider(calls, daemon.DiagnosticResult(True)),
                scheduler_factory=Spy("scheduler", calls),
            )
        self.assertEqual(code, 1)
        self.assertNotIn("diagnostics", calls)
        self.assertNotIn("scheduler", calls)
        self.assertEqual(len(ledger.events), 1)
        self.assertEqual(ledger.events[0]["event_type"], "MANUAL_STATUS_UPDATED")
        self.assertNotEqual(ledger.events[0]["event_type"], "HALT_TRIGGERED")
        self.assertEqual(
            ledger.events[0]["payload"]["event_detail"],
            "STARTUP_BLOCKED_HALT_ACTIVE",
        )

    def test_injected_diagnostic_failure_exits_and_scheduler_is_not_created(self) -> None:
        calls = []
        err = io.StringIO()
        with redirect_stderr(err):
            code = daemon.run_daemon(
                ["--dry-run-only"],
                env_loader=Spy("env", calls),
                config_loader=Spy("config", calls),
                ledger_factory=Spy("ledger", calls, FakeLedger()),
                state_store_factory=Spy("statestore", calls, object()),
                halt_loader=Spy("halt", calls, None),
                diagnostic_provider=FakeProvider(
                    calls,
                    daemon.DiagnosticResult(False, "local state dirty"),
                ),
                scheduler_factory=Spy("scheduler", calls),
            )
        self.assertEqual(code, 1)
        self.assertNotIn("scheduler", calls)
        self.assertIn("local state dirty", err.getvalue())


class DefaultDiagnosticProviderTests(unittest.TestCase):
    def test_default_provider_all_clear_for_clean_local_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(Path(tmp) / "data/state/portfolio_state.json")
            result = daemon.StartupDiagnosticProvider(store, None).run()
        self.assertTrue(result.passed)
        self.assertIsNone(result.reason)

    def test_default_provider_detects_schema_mismatch(self) -> None:
        store = mock.Mock()
        store.state = {"schema_version": 999}
        result = daemon.StartupDiagnosticProvider(store, None).run()
        self.assertFalse(result.passed)
        self.assertIn("schema_version mismatch", result.reason)

    def test_default_provider_detects_halt_state_conflict(self) -> None:
        store = mock.Mock()
        store.state = {"schema_version": 1, "halt_state": {"tier": "hard"}}
        result = daemon.StartupDiagnosticProvider(store, None).run()
        self.assertFalse(result.passed)
        self.assertIn("conflicts", result.reason)

    def test_default_provider_detects_needs_reconciliation_records(self) -> None:
        store = mock.Mock()
        store.state = {
            "schema_version": 1,
            "halt_state": None,
            "positions": {
                "pos_1": {"position_id": "pos_1", "status": "NEEDS_RECONCILIATION"}
            },
        }
        result = daemon.StartupDiagnosticProvider(store, None).run()
        self.assertFalse(result.passed)
        self.assertIn("NEEDS_RECONCILIATION", result.reason)


class DaemonShutdownTests(unittest.TestCase):
    def test_shutdown_uses_daemon_thread_join_timeout_and_wait_true(self) -> None:
        scheduler = mock.Mock()
        made_threads = []

        class FakeThread:
            def __init__(self, target, kwargs, daemon):
                self.target = target
                self.kwargs = kwargs
                self.daemon = daemon
                self.join_timeout = None
                made_threads.append(self)

            def start(self):
                self.target(**self.kwargs)

            def join(self, timeout):
                self.join_timeout = timeout

            def is_alive(self):
                return False

        code = daemon._shutdown_scheduler(scheduler, thread_factory=FakeThread)
        self.assertEqual(code, 0)
        scheduler.shutdown.assert_called_once_with(wait=True)
        self.assertTrue(made_threads[0].daemon)
        self.assertEqual(made_threads[0].join_timeout, SCHEDULER_SHUTDOWN_TIMEOUT_SEC)

    def test_shutdown_timeout_exits_1_and_logs_warning(self) -> None:
        scheduler = mock.Mock()

        class StuckThread:
            def __init__(self, target, kwargs, daemon):
                self.join_timeout = None

            def start(self):
                return None

            def join(self, timeout):
                self.join_timeout = timeout

            def is_alive(self):
                return True

        err = io.StringIO()
        with redirect_stderr(err):
            code = daemon._shutdown_scheduler(scheduler, thread_factory=StuckThread)
        self.assertEqual(code, 1)
        self.assertIn("WARNING", err.getvalue())
        scheduler.shutdown.assert_not_called()

    def test_sigint_and_sigterm_handlers_shutdown_scheduler(self) -> None:
        scheduler = mock.Mock()
        handlers = {}

        def fake_signal(signum, handler):
            handlers[signum] = handler

        with mock.patch.object(signal, "signal", side_effect=fake_signal):
            with mock.patch.object(daemon, "_shutdown_scheduler", return_value=0) as shutdown:
                daemon._install_shutdown_handlers(scheduler)
                with self.assertRaises(SystemExit) as int_exit:
                    handlers[signal.SIGINT](signal.SIGINT, None)
                with self.assertRaises(SystemExit) as term_exit:
                    handlers[signal.SIGTERM](signal.SIGTERM, None)
        self.assertEqual(int_exit.exception.code, 0)
        self.assertEqual(term_exit.exception.code, 0)
        self.assertEqual(shutdown.call_count, 2)
        shutdown.assert_called_with(scheduler)


if __name__ == "__main__":
    unittest.main()
