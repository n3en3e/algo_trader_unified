from __future__ import annotations

import py_compile
import tempfile
import unittest
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE
from algo_trader_unified.config.scheduler import (
    JOB_INTENT_CONFIRMATION,
    JOB_INTENT_FILL_CONFIRMATION,
    JOB_INTENT_SUBMISSION,
    JOB_POSITION_TRANSITIONS,
    JOB_S01_MANAGEMENT_SCAN,
    JOB_S01_VOL_SCAN,
)
from algo_trader_unified.core.execution import DryRunExecutionAdapter
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.ledger_reader import LedgerReader
from algo_trader_unified.core.management import ManagementSignalResult
from algo_trader_unified.core.readiness import ReadinessManager, ReadinessStatus
from algo_trader_unified.core.scheduler import UnifiedScheduler
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.strategies.vol.signals import VolSignalInput
from algo_trader_unified.tools.system_status import build_summary


ORDER_LIFECYCLE_EVENTS = [
    "ORDER_INTENT_CREATED",
    "ORDER_SUBMITTED",
    "ORDER_CONFIRMED",
    "FILL_CONFIRMED",
    "CLOSE_INTENT_CREATED",
    "CLOSE_ORDER_SUBMITTED",
    "CLOSE_ORDER_CONFIRMED",
    "CLOSE_FILL_CONFIRMED",
]
EXECUTION_LIFECYCLE_EVENTS = ["POSITION_OPENED", "POSITION_CLOSED"]
FORBIDDEN_EVENTS = {
    "OPPORTUNITY_" + "IDENTIFIED",
    "OPPORTUNITY_" + "SCORED",
    "SIGNAL_" + "SIZED",
    "POSITION_" + "ADJUSTED",
}


class Phase3WE2EDryRunLifecycleTests(unittest.TestCase):
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
        self.set_readiness()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def set_readiness(self) -> None:
        self.readiness_manager.update_readiness(
            ReadinessStatus(
                strategy_id=S01_VOL_BASELINE,
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

    def order_events(self) -> list[dict]:
        return LedgerReader.from_root(self.root).read_events("order")

    def execution_events(self) -> list[dict]:
        return LedgerReader.from_root(self.root).read_events("execution")

    def order_event_types(self) -> list[str]:
        return [event["event_type"] for event in self.order_events()]

    def execution_event_types(self) -> list[str]:
        return [event["event_type"] for event in self.execution_events()]

    def lifecycle_execution_event_types(self) -> list[str]:
        return [
            event_type
            for event_type in self.execution_event_types()
            if event_type in EXECUTION_LIFECYCLE_EVENTS
        ]

    def lifecycle_order_event_types(self) -> list[str]:
        return [
            event_type
            for event_type in self.order_event_types()
            if event_type in ORDER_LIFECYCLE_EVENTS
        ]

    def assert_no_forbidden_events(self) -> None:
        event_types = set(self.order_event_types() + self.execution_event_types())
        self.assertTrue(FORBIDDEN_EVENTS.isdisjoint(event_types))

    def signal_context(self) -> VolSignalInput:
        return VolSignalInput(
            symbol="XSP",
            current_date=date(2026, 5, 4),
            vix=18.0,
            iv_rank=45.0,
            target_dte=45,
            blackout_dates=(),
            order_ref_candidate=f"{S01_VOL_BASELINE}|PHASE3W|OPEN",
        )

    def close_signal(self, *, position: dict, now: str) -> ManagementSignalResult:
        return ManagementSignalResult(
            should_close=True,
            close_reason="phase3w_smoke",
            requested_by="phase3w",
            details={"position_id": position.get("position_id"), "now": now},
        )

    def run_entry_lifecycle(self) -> None:
        scan = self.scheduler.run_job_once(
            JOB_S01_VOL_SCAN,
            state_store=self.state_store,
            ledger=self.ledger,
            readiness_manager=self.readiness_manager,
            current_time=datetime(2026, 5, 4, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=self.signal_context,
        )
        self.assertEqual(scan.detail, "order_intent_created")
        self.assertEqual(self.lifecycle_order_event_types(), ["ORDER_INTENT_CREATED"])

        submitted = self.scheduler.run_job_once(
            JOB_INTENT_SUBMISSION,
            state_store=self.state_store,
            ledger=self.ledger,
            now="2026-05-04T14:00:00+00:00",
            execution_adapter=self.adapter,
        )
        self.assertEqual(submitted["submitted_order_intents_count"], 1)
        self.assertEqual(
            self.lifecycle_order_event_types(),
            ["ORDER_INTENT_CREATED", "ORDER_SUBMITTED"],
        )

        confirmed = self.scheduler.run_job_once(
            JOB_INTENT_CONFIRMATION,
            state_store=self.state_store,
            ledger=self.ledger,
            now="2026-05-04T14:01:00+00:00",
            execution_adapter=self.adapter,
        )
        self.assertEqual(confirmed["confirmed_order_intents_count"], 1)
        self.assertEqual(
            self.lifecycle_order_event_types(),
            ["ORDER_INTENT_CREATED", "ORDER_SUBMITTED", "ORDER_CONFIRMED"],
        )

        filled = self.scheduler.run_job_once(
            JOB_INTENT_FILL_CONFIRMATION,
            state_store=self.state_store,
            ledger=self.ledger,
            now="2026-05-04T14:02:00+00:00",
            execution_adapter=self.adapter,
        )
        self.assertEqual(filled["filled_order_intents_count"], 1)
        self.assertEqual(
            self.lifecycle_order_event_types(),
            [
                "ORDER_INTENT_CREATED",
                "ORDER_SUBMITTED",
                "ORDER_CONFIRMED",
                "FILL_CONFIRMED",
            ],
        )

        opened = self.scheduler.run_job_once(
            JOB_POSITION_TRANSITIONS,
            state_store=self.state_store,
            ledger=self.ledger,
            now="2026-05-04T14:03:00+00:00",
        )
        self.assertEqual(opened["positions_opened_count"], 1)
        self.assertEqual(self.lifecycle_execution_event_types(), ["POSITION_OPENED"])
        self.assertEqual(len(self.state_store.list_positions(status="open")), 1)
        order_intent = self.state_store.list_order_intents()[0]
        self.assertEqual(order_intent["status"], "position_opened")

    def run_exit_lifecycle(self) -> None:
        management = self.scheduler.run_job_once(
            JOB_S01_MANAGEMENT_SCAN,
            state_store=self.state_store,
            ledger=self.ledger,
            now="2026-05-04T15:40:00+00:00",
            management_signal_provider=self.close_signal,
        )
        self.assertEqual(management["close_intents_created_count"], 1)
        self.assertEqual(
            self.lifecycle_order_event_types(),
            ORDER_LIFECYCLE_EVENTS[:5],
        )

        submitted = self.scheduler.run_job_once(
            JOB_INTENT_SUBMISSION,
            state_store=self.state_store,
            ledger=self.ledger,
            now="2026-05-04T15:41:00+00:00",
            execution_adapter=self.adapter,
        )
        self.assertEqual(submitted["submitted_close_intents_count"], 1)
        self.assertEqual(self.lifecycle_order_event_types(), ORDER_LIFECYCLE_EVENTS[:6])

        confirmed = self.scheduler.run_job_once(
            JOB_INTENT_CONFIRMATION,
            state_store=self.state_store,
            ledger=self.ledger,
            now="2026-05-04T15:42:00+00:00",
            execution_adapter=self.adapter,
        )
        self.assertEqual(confirmed["confirmed_close_intents_count"], 1)
        self.assertEqual(self.lifecycle_order_event_types(), ORDER_LIFECYCLE_EVENTS[:7])

        filled = self.scheduler.run_job_once(
            JOB_INTENT_FILL_CONFIRMATION,
            state_store=self.state_store,
            ledger=self.ledger,
            now="2026-05-04T15:43:00+00:00",
            execution_adapter=self.adapter,
        )
        self.assertEqual(filled["filled_close_intents_count"], 1)
        self.assertEqual(self.lifecycle_order_event_types(), ORDER_LIFECYCLE_EVENTS)

        closed = self.scheduler.run_job_once(
            JOB_POSITION_TRANSITIONS,
            state_store=self.state_store,
            ledger=self.ledger,
            now="2026-05-04T15:44:00+00:00",
        )
        self.assertEqual(closed["positions_closed_count"], 1)
        self.assertEqual(self.lifecycle_execution_event_types(), EXECUTION_LIFECYCLE_EVENTS)
        position = self.state_store.list_positions()[0]
        close_intent = self.state_store.list_close_intents()[0]
        self.assertEqual(position["status"], "closed")
        self.assertEqual(close_intent["status"], "position_closed")
        self.assertIsNone(self.state_store.get_open_position(S01_VOL_BASELINE, "XSP"))

    def assert_final_state(self) -> None:
        order_intents = self.state_store.list_order_intents()
        positions = self.state_store.list_positions()
        close_intents = self.state_store.list_close_intents()

        self.assertEqual(len(order_intents), 1)
        self.assertEqual(order_intents[0]["status"], "position_opened")
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["status"], "closed")
        self.assertTrue(positions[0].get("position_closed_event_id"))
        self.assertIsInstance(positions[0].get("realized_pnl"), (int, float))
        self.assertIsNone(positions[0].get("active_close_intent_id"))
        self.assertEqual(len(close_intents), 1)
        self.assertEqual(close_intents[0]["status"], "position_closed")
        self.assertTrue(close_intents[0].get("position_closed_event_id"))
        self.assertIsNone(self.state_store.get_open_position(S01_VOL_BASELINE, "XSP"))

        summary = build_summary(
            order_intents,
            positions,
            close_intents,
            strategy_id=S01_VOL_BASELINE,
        )
        self.assertEqual(summary["open_positions_count"], 0)
        self.assertEqual(summary["closed_positions_count"], 1)
        self.assertEqual(summary["position_closed_close_intents_count"], 1)
        self.assertEqual(summary["total_positions_count"], 1)
        self.assertEqual(summary["total_order_intents_count"], 1)
        self.assertEqual(summary["total_close_intents_count"], 1)

    def test_full_dry_run_lifecycle_reaches_closed_position_with_exact_sequences(self) -> None:
        self.run_entry_lifecycle()
        self.run_exit_lifecycle()

        self.assertEqual(self.lifecycle_order_event_types(), ORDER_LIFECYCLE_EVENTS)
        self.assertEqual(self.lifecycle_execution_event_types(), EXECUTION_LIFECYCLE_EVENTS)
        self.assert_no_forbidden_events()
        all_event_types = self.order_event_types() + self.execution_event_types()
        self.assertIn("SIGNAL_GENERATED", all_event_types)
        self.assertNotIn("SIGNAL_SKIPPED", all_event_types)
        self.assert_final_state()

    def test_repeat_scheduler_jobs_are_noops_after_full_lifecycle(self) -> None:
        self.run_entry_lifecycle()
        self.run_exit_lifecycle()
        before_state = deepcopy(self.state_store.state)
        before_order_events = self.order_events()
        before_execution_events = self.execution_events()

        submission = self.scheduler.run_job_once(
            JOB_INTENT_SUBMISSION,
            state_store=self.state_store,
            ledger=self.ledger,
            now="2026-05-04T16:00:00+00:00",
            execution_adapter=self.adapter,
        )
        confirmation = self.scheduler.run_job_once(
            JOB_INTENT_CONFIRMATION,
            state_store=self.state_store,
            ledger=self.ledger,
            now="2026-05-04T16:01:00+00:00",
            execution_adapter=self.adapter,
        )
        fill = self.scheduler.run_job_once(
            JOB_INTENT_FILL_CONFIRMATION,
            state_store=self.state_store,
            ledger=self.ledger,
            now="2026-05-04T16:02:00+00:00",
            execution_adapter=self.adapter,
        )
        position = self.scheduler.run_job_once(
            JOB_POSITION_TRANSITIONS,
            state_store=self.state_store,
            ledger=self.ledger,
            now="2026-05-04T16:03:00+00:00",
        )
        management = self.scheduler.run_job_once(
            JOB_S01_MANAGEMENT_SCAN,
            state_store=self.state_store,
            ledger=self.ledger,
            now="2026-05-04T16:04:00+00:00",
            management_signal_provider=self.close_signal,
        )

        self.assertEqual(submission["submitted_order_intents_count"], 0)
        self.assertEqual(submission["submitted_close_intents_count"], 0)
        self.assertEqual(confirmation["confirmed_order_intents_count"], 0)
        self.assertEqual(confirmation["confirmed_close_intents_count"], 0)
        self.assertEqual(fill["filled_order_intents_count"], 0)
        self.assertEqual(fill["filled_close_intents_count"], 0)
        self.assertEqual(position["positions_opened_count"], 0)
        self.assertEqual(position["positions_closed_count"], 0)
        self.assertGreaterEqual(submission["skipped_order_intents_count"], 1)
        self.assertGreaterEqual(submission["skipped_close_intents_count"], 1)
        self.assertEqual(management["close_intents_created_count"], 0)
        self.assertEqual(management["evaluated_count"], 0)
        self.assertEqual(self.state_store.state, before_state)
        self.assertEqual(self.order_events(), before_order_events)
        self.assertEqual(self.execution_events(), before_execution_events)
        self.assert_final_state()


class Phase3WSafetyTests(unittest.TestCase):
    def test_new_phase3w_sources_do_not_add_live_or_schema_paths(self) -> None:
        paths = [
            Path("algo_trader_unified/tests/test_phase3w_e2e_dry_run_lifecycle.py"),
            Path("docs/dry_run_lifecycle_runbook.md"),
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for forbidden in (
            "ib_" + "insync",
            "yf" + "inance",
            "requ" + "ests",
            ".sta" + "rt()",
            "place" + "Order",
            "cancel" + "Order",
            "target" + "_price",
            "limit" + "_price",
            "order" + "_type",
            "time" + "_in_force",
            "contract " + "multiplier",
            "option " + "legs",
            "spread " + "legs",
            "OPPORTUNITY_" + "IDENTIFIED",
            "OPPORTUNITY_" + "SCORED",
            "SIGNAL_" + "SIZED",
        ):
            self.assertNotIn(forbidden, combined)
        self.assertNotIn("write" + "_text", combined)
        self.assertNotIn("POSITION_" + "ADJUSTED", combined)
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
