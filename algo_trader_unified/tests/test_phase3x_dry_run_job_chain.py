from __future__ import annotations

import contextlib
import io
import json
import py_compile
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.scheduler import (
    JOB_INTENT_CONFIRMATION,
    JOB_INTENT_FILL_CONFIRMATION,
    JOB_INTENT_SUBMISSION,
    JOB_POSITION_TRANSITIONS,
    JOB_S01_MANAGEMENT_SCAN,
    JOB_S01_VOL_SCAN,
    JOB_S02_MANAGEMENT_SCAN,
)
from algo_trader_unified.core.execution import DryRunExecutionAdapter
from algo_trader_unified.core.job_chain import run_dry_run_job_chain
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.ledger_reader import LedgerReader
from algo_trader_unified.core.management import ManagementSignalResult
from algo_trader_unified.core.readiness import ReadinessManager, ReadinessStatus
from algo_trader_unified.core.scheduler import UnifiedScheduler
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.strategies.vol.signals import VolSignalInput
from algo_trader_unified.tools.run_dry_run_chain import main as run_chain_main


NOW = datetime(2026, 5, 4, 14, 0, tzinfo=timezone.utc)
CHAIN_NOW = "2026-05-04T14:00:00+00:00"
ORDER_LIFECYCLE = [
    "ORDER_INTENT_CREATED",
    "ORDER_SUBMITTED",
    "ORDER_CONFIRMED",
    "FILL_CONFIRMED",
    "CLOSE_INTENT_CREATED",
    "CLOSE_ORDER_SUBMITTED",
    "CLOSE_ORDER_CONFIRMED",
    "CLOSE_FILL_CONFIRMED",
]
EXECUTION_LIFECYCLE = ["POSITION_OPENED", "POSITION_CLOSED"]


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "algo_trader_unified").is_dir() and (parent / ".git").exists():
            return parent
    raise RuntimeError("repository root not found")


class Phase3XCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_store = StateStore(self.root / "data/state/portfolio_state.json")
        self.ledger = LedgerAppender(self.root)
        self.readiness_manager = ReadinessManager(self.state_store, self.ledger)
        self.scheduler = UnifiedScheduler(
            state_store=self.state_store,
            ledger=self.ledger,
            readiness_manager=self.readiness_manager,
        )
        self.adapter = DryRunExecutionAdapter()
        self.set_readiness(S01_VOL_BASELINE)
        self.set_readiness(S02_VOL_ENHANCED)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def set_readiness(self, strategy_id: str) -> None:
        self.readiness_manager.update_readiness(
            ReadinessStatus(
                strategy_id=strategy_id,
                ready_for_entries=True,
                reason=None,
                checked_at="2026-05-04T13:35:00+00:00",
                dirty_state=False,
                unknown_broker_exposure=False,
                nlv_degraded=False,
                halt_active=False,
                calendar_expired=False,
                iv_baseline_available=True,
            )
        )

    def signal_context(self) -> VolSignalInput:
        return VolSignalInput(
            symbol="XSP",
            current_date=date(2026, 5, 4),
            vix=18.0,
            iv_rank=45.0,
            target_dte=45,
            blackout_dates=(),
            order_ref_candidate=f"{S01_VOL_BASELINE}|PHASE3X|OPEN",
        )

    def close_signal(self, *, position: dict, now: str) -> ManagementSignalResult:
        return ManagementSignalResult(
            should_close=True,
            close_reason="phase3x",
            requested_by="phase3x",
        )

    def no_close_signal(self, *, position: dict, now: str) -> ManagementSignalResult:
        return ManagementSignalResult(should_close=False)

    def order_events(self) -> list[dict]:
        return LedgerReader.from_root(self.root).read_events("order")

    def execution_events(self) -> list[dict]:
        return LedgerReader.from_root(self.root).read_events("execution")

    def order_event_types(self) -> list[str]:
        return [event["event_type"] for event in self.order_events()]

    def execution_event_types(self) -> list[str]:
        return [event["event_type"] for event in self.execution_events()]

    def lifecycle_order_event_types(self) -> list[str]:
        return [event for event in self.order_event_types() if event in ORDER_LIFECYCLE]

    def lifecycle_execution_event_types(self) -> list[str]:
        return [event for event in self.execution_event_types() if event in EXECUTION_LIFECYCLE]

    def run_chain(self, **kwargs) -> dict:
        return run_dry_run_job_chain(
            scheduler=self.scheduler,
            state_store=self.state_store,
            ledger=self.ledger,
            now=kwargs.pop("now", NOW),
            execution_adapter=kwargs.pop("execution_adapter", self.adapter),
            **kwargs,
        )


class CoreDryRunJobChainTests(Phase3XCase):
    def test_default_noop_behavior_runs_safe_jobs_without_mutation_or_ledger_writes(self) -> None:
        before = deepcopy(self.state_store.state)
        result = self.run_chain()
        self.assertIs(result["dry_run"], True)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            [step["job_id"] for step in result["steps_run"]],
            [
                JOB_S01_MANAGEMENT_SCAN,
                JOB_S02_MANAGEMENT_SCAN,
                JOB_INTENT_SUBMISSION,
                JOB_INTENT_CONFIRMATION,
                JOB_INTENT_FILL_CONFIRMATION,
                JOB_POSITION_TRANSITIONS,
            ],
        )
        self.assertEqual(result["steps_skipped"][0]["step"], "entry_scan")
        self.assertEqual(result["steps_skipped"][0]["reason"], "missing_signal_context_provider")
        self.assertEqual(self.state_store.state, before)
        self.assertEqual(self.order_events(), [])
        self.assertEqual(self.execution_events(), [])

    def test_full_entry_lifecycle_with_injected_signal_provider(self) -> None:
        result = self.run_chain(
            strategy_id=S01_VOL_BASELINE,
            signal_context_provider=self.signal_context,
            management_signal_provider=self.no_close_signal,
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["summary"]["entry_scan_runs"], 1)
        self.assertEqual(self.lifecycle_order_event_types(), ORDER_LIFECYCLE[:4])
        self.assertEqual(self.lifecycle_execution_event_types(), ["POSITION_OPENED"])
        self.assertEqual(self.state_store.list_order_intents()[0]["status"], "position_opened")
        self.assertEqual(self.state_store.list_positions(status="open")[0]["status"], "open")
        self.assertEqual(self.state_store.list_close_intents(), [])
        self.assertNotIn("POSITION_CLOSED", self.execution_event_types())

    def test_full_entry_and_exit_lifecycle_with_two_chain_runs(self) -> None:
        first = self.run_chain(
            strategy_id=S01_VOL_BASELINE,
            signal_context_provider=self.signal_context,
            management_signal_provider=self.no_close_signal,
        )
        self.assertEqual(first["status"], "completed")
        second = self.run_chain(
            strategy_id=S01_VOL_BASELINE,
            include_entry_scan=False,
            management_signal_provider=self.close_signal,
        )
        self.assertEqual(second["status"], "completed")
        self.assertEqual(self.lifecycle_order_event_types(), ORDER_LIFECYCLE)
        self.assertEqual(self.lifecycle_execution_event_types(), EXECUTION_LIFECYCLE)
        self.assertEqual(self.state_store.list_positions()[0]["status"], "closed")
        self.assertEqual(self.state_store.list_close_intents()[0]["status"], "position_closed")
        self.assertIsNone(self.state_store.get_open_position(S01_VOL_BASELINE, "XSP"))

    def test_include_flags_skip_selected_steps(self) -> None:
        no_mutation = self.run_chain(
            include_entry_scan=False,
            include_management_scan=False,
            include_submission=False,
            include_confirmation=False,
            include_fill_confirmation=False,
            include_position_transitions=False,
        )
        self.assertEqual(no_mutation["status"], "completed")
        self.assertEqual(no_mutation["steps_run"], [])
        self.assertEqual(len(no_mutation["steps_skipped"]), 6)
        self.assertEqual(self.order_events(), [])
        self.assertEqual(self.execution_events(), [])

        no_submit = self.run_chain(
            strategy_id=S01_VOL_BASELINE,
            signal_context_provider=self.signal_context,
            include_management_scan=False,
            include_submission=False,
        )
        self.assertEqual(no_submit["summary"]["submission_runs"], 0)
        self.assertEqual(self.state_store.list_order_intents()[0]["status"], "created")
        self.assertNotIn("ORDER_SUBMITTED", self.order_event_types())

    def test_skip_position_transitions_leaves_filled_intents_stranded(self) -> None:
        result = self.run_chain(
            strategy_id=S01_VOL_BASELINE,
            signal_context_provider=self.signal_context,
            include_management_scan=False,
            include_position_transitions=False,
        )
        self.assertEqual(result["summary"]["position_transition_runs"], 0)
        self.assertEqual(self.state_store.list_order_intents()[0]["status"], "filled")
        self.assertNotIn("POSITION_OPENED", self.execution_event_types())

    def test_strategy_filtering_and_unknown_strategy(self) -> None:
        unknown = self.run_chain(strategy_id="UNKNOWN", include_entry_scan=True)
        self.assertEqual(unknown["status"], "completed")
        self.assertEqual(unknown["summary"]["entry_scan_runs"], 0)
        self.assertEqual(unknown["summary"]["management_scan_runs"], 0)
        self.assertEqual(self.order_events(), [])

        self.run_chain(
            strategy_id=S01_VOL_BASELINE,
            signal_context_provider=self.signal_context,
            include_management_scan=False,
        )
        self.assertEqual(self.state_store.list_order_intents()[0]["strategy_id"], S01_VOL_BASELINE)
        self.assertEqual(len(self.state_store.list_order_intents(strategy_id=S02_VOL_ENHANCED)), 0)

    def test_step_error_records_and_later_steps_continue(self) -> None:
        class FailingScheduler:
            def __init__(self) -> None:
                self.calls = []

            def run_job_once(self, job_id: str, **kwargs):
                self.calls.append(job_id)
                if job_id == JOB_INTENT_CONFIRMATION:
                    raise ValueError("boom")
                return {"dry_run": True, "errors_count": 0, "status": "ok"}

        fake_scheduler = FailingScheduler()
        result = run_dry_run_job_chain(
            scheduler=fake_scheduler,
            state_store=self.state_store,
            ledger=self.ledger,
            now=CHAIN_NOW,
            include_entry_scan=False,
            include_management_scan=False,
        )
        self.assertEqual(result["status"], "completed_with_errors")
        self.assertEqual(result["errors_count"], 1)
        self.assertEqual(result["errors"][0]["job_id"], JOB_INTENT_CONFIRMATION)
        json.dumps(result)
        self.assertIn(JOB_POSITION_TRANSITIONS, fake_scheduler.calls)


class RunDryRunChainCliTests(unittest.TestCase):
    def test_cli_default_json_is_strict_json_and_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = run_chain_main(["--root-dir", tmp, "--json"])
            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertIs(payload["dry_run"], True)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(LedgerReader.from_root(tmp).read_events("order"), [])
            self.assertEqual(LedgerReader.from_root(tmp).read_events("execution"), [])

    def test_cli_skip_flags_and_z_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = run_chain_main(
                    [
                        "--root-dir",
                        tmp,
                        "--json",
                        "--dry-run",
                        "--now",
                        "2026-05-04T16:00:00Z",
                        "--skip-entry-scan",
                        "--skip-management-scan",
                        "--skip-submission",
                        "--skip-confirmation",
                        "--skip-fill-confirmation",
                        "--skip-position-transitions",
                    ]
                )
            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["steps_run"], [])
            self.assertEqual(len(payload["steps_skipped"]), 6)

    def test_cli_invalid_now_exits_nonzero_without_traceback(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "algo_trader_unified.tools.run_dry_run_chain",
                "--now",
                "not-a-time",
                "--json",
            ],
            cwd=repo_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertNotIn("Traceback", completed.stderr)


class Phase3XSourceSafetyTests(unittest.TestCase):
    def test_source_safety_scans(self) -> None:
        paths = [
            Path("algo_trader_unified/core/job_chain.py"),
            Path("algo_trader_unified/tools/run_dry_run_chain.py"),
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for forbidden in (
            "ib_" + "insync",
            "yf" + "inance",
            "requ" + "ests",
            ".sta" + "rt(",
            "place" + "Order",
            "cancel" + "Order",
            "broker.submit" + "_order",
            "LedgerAppender.append",
            ".jsonl",
            "submit" + "_order_intent(",
            "submit" + "_close_intent(",
            "confirm" + "_order_intent(",
            "confirm" + "_close_order(",
            "confirm" + "_fill(",
            "confirm" + "_close_fill(",
            "open" + "_position_from_filled_intent(",
            "close" + "_position_from_filled_intent(",
            "target" + "_price",
            "limit" + "_price",
            "order" + "_type",
            "time" + "_in_force",
            "side",
            "direction",
            "multiplier",
            "legs",
            "except:",
        ):
            self.assertNotIn(forbidden, combined)
        self.assertFalse(list(Path(".").glob("*.service")))

    def test_compile_and_runtime_safety_regressions(self) -> None:
        package_files = [
            path
            for path in Path("algo_trader_unified").rglob("*.py")
            if "__pycache__" not in path.parts
        ]
        for path in package_files:
            py_compile.compile(str(path), doraise=True)
        self.assertTrue(Path(".gitignore").read_text(encoding="utf-8").find("data/") >= 0)
        joined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in package_files
            if "tests" not in path.parts
        )
        self.assertNotIn("commodity" + "_vrp", joined.lower())
