from __future__ import annotations

import inspect
import json
import py_compile
import tempfile
import unittest
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.scheduler import JOB_INTENT_SUBMISSION, JOB_SPECS
from algo_trader_unified.core.close_intents import (
    confirm_close_fill,
    confirm_close_order,
    create_close_intent_from_position,
    submit_close_intent,
)
from algo_trader_unified.core.execution import DryRunExecutionAdapter
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.ledger_reader import LedgerReader
from algo_trader_unified.core.order_intents import (
    confirm_fill,
    confirm_order_intent,
    submit_order_intent,
)
from algo_trader_unified.core.positions import (
    close_position_from_filled_intent,
    open_position_from_filled_intent,
)
from algo_trader_unified.core.readiness import ReadinessManager, ReadinessStatus
from algo_trader_unified.core.scheduler import JobNotFoundError, UnifiedScheduler
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.jobs.submission import run_intent_submission_job
from algo_trader_unified.jobs.vol import run_s01_vol_scan
from algo_trader_unified.strategies.vol.signals import VolSignalInput


NOW = "2026-04-30T15:59:00+00:00"


class RecordingAdapter(DryRunExecutionAdapter):
    def __init__(self) -> None:
        self.calls = []

    def submit_order_intent(self, intent, *, submitted_at):
        self.calls.append(("order", intent["intent_id"]))
        return super().submit_order_intent(intent, submitted_at=submitted_at)

    def submit_close_intent(self, close_intent, *, submitted_at):
        self.calls.append(("close", close_intent["close_intent_id"]))
        return super().submit_close_intent(close_intent, submitted_at=submitted_at)


class Phase3SCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_store = StateStore(self.root / "data/state/portfolio_state.json")
        self.ledger = LedgerAppender(self.root)
        self.readiness_manager = ReadinessManager(self.state_store, self.ledger)
        self.order_path = self.root / "data/ledger/order_ledger.jsonl"
        self.execution_path = self.root / "data/ledger/execution_ledger.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def order_events(self) -> list[dict]:
        return LedgerReader.from_root(self.root).read_events("order")

    def execution_events(self) -> list[dict]:
        return LedgerReader.from_root(self.root).read_events("execution")

    def all_events(self) -> list[dict]:
        return self.order_events() + self.execution_events()

    def create_order_intent(
        self,
        intent_id: str,
        *,
        strategy_id: str = S01_VOL_BASELINE,
        status: str = "created",
        created_at: str = "2026-04-30T14:00:00+00:00",
        dry_run: bool | None = True,
    ) -> dict:
        record = {
            "intent_id": intent_id,
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "execution_mode": "paper_only",
            "status": status,
            "source_signal_event_id": f"evt_signal_{intent_id}",
            "order_intent_created_event_id": f"evt_created_{intent_id}",
            "order_ref": f"{strategy_id}|{intent_id}|OPEN",
            "created_at": created_at,
            "updated_at": created_at,
            "sizing_context": {},
            "risk_context": {},
            "signal_payload_snapshot": {},
        }
        if dry_run is not None:
            record["dry_run"] = dry_run
        return self.state_store.create_order_intent(record)

    def position_record(
        self,
        strategy_id: str,
        position_id: str,
        *,
        status: str = "open",
        quantity: object = 2,
        entry_price: object = 0.75,
    ) -> dict:
        return {
            "position_id": position_id,
            "intent_id": f"{position_id}:intent",
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "status": status,
            "execution_mode": "paper_only",
            "dry_run": True,
            "opened_at": "2026-04-30T14:07:00+00:00",
            "updated_at": "2026-04-30T14:07:00+00:00",
            "source_signal_event_id": "evt_signal",
            "order_intent_created_event_id": "evt_created",
            "order_submitted_event_id": "evt_submitted",
            "order_confirmed_event_id": "evt_confirmed",
            "fill_confirmed_event_id": "evt_fill",
            "position_opened_event_id": "evt_opened",
            "order_ref": f"{strategy_id}|{position_id}|OPEN",
            "simulated_order_id": f"sim:{position_id}",
            "fill_id": f"fill:{position_id}",
            "entry_price": entry_price,
            "quantity": quantity,
            "action": "open",
        }

    def create_position(self, strategy_id: str, position_id: str, **kwargs) -> dict:
        record = self.position_record(strategy_id, position_id, **kwargs)
        self.state_store.state["positions"][position_id] = deepcopy(record)
        self.state_store.save()
        return record

    def create_close_intent(
        self,
        close_intent_id: str,
        *,
        strategy_id: str = S01_VOL_BASELINE,
        position_id: str | None = None,
        status: str = "created",
        created_at: str = "2026-04-30T14:30:00+00:00",
        quantity: object = 2,
        entry_price: object = 0.75,
    ) -> dict:
        position_id = position_id or f"position:{close_intent_id}"
        if self.state_store.get_position(position_id) is None:
            self.create_position(
                strategy_id,
                position_id,
                quantity=quantity,
                entry_price=entry_price,
            )
        record = {
            "close_intent_id": close_intent_id,
            "position_id": position_id,
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "status": status,
            "execution_mode": "paper_only",
            "dry_run": True,
            "created_at": created_at,
            "updated_at": created_at,
            "close_reason": "management",
            "requested_by": "phase3s",
            "position_opened_event_id": "evt_opened",
            "source_signal_event_id": "evt_signal",
            "fill_confirmed_event_id": "evt_fill",
            "close_intent_created_event_id": f"evt_created_{close_intent_id}",
            "quantity": quantity,
            "entry_price": entry_price,
            "action": "close",
        }
        self.state_store.state["close_intents"][close_intent_id] = record
        self.state_store.state["positions"][position_id][
            "active_close_intent_id"
        ] = close_intent_id
        self.state_store.save()
        return deepcopy(record)

    def run_job(self, **kwargs) -> dict:
        return run_intent_submission_job(
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
            **kwargs,
        )


class SchedulerSubmissionConfigTests(unittest.TestCase):
    def test_intent_submission_job_spec_is_registered_and_only_submission_job_added(self) -> None:
        self.assertEqual(JOB_INTENT_SUBMISSION, "intent_submission")
        self.assertIn(JOB_INTENT_SUBMISSION, JOB_SPECS)
        self.assertTrue(JOB_SPECS[JOB_INTENT_SUBMISSION].enabled)
        self.assertIn("Dry-run", JOB_SPECS[JOB_INTENT_SUBMISSION].description)
        forbidden_fragments = (
            "confirm",
            "fill",
            "position_open",
            "position_close",
            "close_position",
        )
        added_jobs = set(JOB_SPECS) - {
            "keepalive",
            "risk_monitor",
            "market_open_scan",
            "daily_digest",
            "weekly_digest",
            "s01_vol_scan_stub",
            "s02_vol_scan_stub",
            "s01_management_scan",
            "s02_management_scan",
            JOB_INTENT_SUBMISSION,
        }
        self.assertEqual(added_jobs, set())
        for fragment in forbidden_fragments:
            self.assertFalse(any(fragment in job_id for job_id in JOB_SPECS))

    def test_no_live_scheduler_start_behavior(self) -> None:
        source = Path("algo_trader_unified/core/scheduler.py").read_text(encoding="utf-8")
        self.assertEqual(source.count(".start()"), 1)
        self.assertNotIn("BlockingScheduler", source)


class IntentSubmissionJobTests(Phase3SCase):
    def test_no_intents_returns_zeroes_without_mutation_or_writes(self) -> None:
        before = deepcopy(self.state_store.state)
        result = self.run_job()
        self.assertIs(result["dry_run"], True)
        for key in (
            "submitted_order_intents_count",
            "submitted_close_intents_count",
            "skipped_order_intents_count",
            "skipped_close_intents_count",
            "errors_count",
        ):
            self.assertEqual(result[key], 0)
        self.assertEqual(self.state_store.state, before)
        self.assertEqual(self.all_events(), [])

    def test_open_side_order_intent_submission_uses_helper_only(self) -> None:
        self.create_order_intent("s01:open")
        result = self.run_job()
        self.assertEqual(result["submitted_order_intents_count"], 1)
        self.assertEqual(self.state_store.get_order_intent("s01:open")["status"], "submitted")
        self.assertEqual([event["event_type"] for event in self.all_events()], ["ORDER_SUBMITTED"])
        self.assertTrue({"ORDER_CONFIRMED", "FILL_CONFIRMED", "POSITION_OPENED"}.isdisjoint({event["event_type"] for event in self.all_events()}))

    def test_close_intent_submission_uses_helper_only(self) -> None:
        self.create_close_intent("s01:close")
        result = self.run_job()
        self.assertEqual(result["submitted_close_intents_count"], 1)
        self.assertEqual(
            self.state_store.get_close_intent("s01:close")["status"],
            "submitted",
        )
        self.assertEqual([event["event_type"] for event in self.all_events()], ["CLOSE_ORDER_SUBMITTED"])
        self.assertTrue({"CLOSE_ORDER_CONFIRMED", "CLOSE_FILL_CONFIRMED", "POSITION_CLOSED"}.isdisjoint({event["event_type"] for event in self.all_events()}))

    def test_mixed_batch_submits_open_side_before_close_side_deterministically(self) -> None:
        adapter = RecordingAdapter()
        self.create_close_intent("s01:close:b", created_at="2026-04-30T14:10:00+00:00")
        self.create_order_intent("s01:open:b", created_at="2026-04-30T14:05:00+00:00")
        self.create_close_intent("s01:close:a", created_at="2026-04-30T14:00:00+00:00")
        self.create_order_intent("s01:open:a", created_at="2026-04-30T14:00:00+00:00")
        result = self.run_job(execution_adapter=adapter)
        self.assertEqual(result["submitted_order_intents_count"], 2)
        self.assertEqual(result["submitted_close_intents_count"], 2)
        self.assertEqual(
            adapter.calls,
            [
                ("order", "s01:open:a"),
                ("order", "s01:open:b"),
                ("close", "s01:close:a"),
                ("close", "s01:close:b"),
            ],
        )
        self.assertEqual(
            [event["event_type"] for event in self.all_events()],
            [
                "ORDER_SUBMITTED",
                "ORDER_SUBMITTED",
                "CLOSE_ORDER_SUBMITTED",
                "CLOSE_ORDER_SUBMITTED",
            ],
        )

    def test_strategy_filtering(self) -> None:
        self.create_order_intent("s01:open", strategy_id=S01_VOL_BASELINE)
        self.create_close_intent("s01:close", strategy_id=S01_VOL_BASELINE)
        self.create_order_intent("s02:open", strategy_id=S02_VOL_ENHANCED)
        self.create_close_intent("s02:close", strategy_id=S02_VOL_ENHANCED)

        s01 = self.run_job(strategy_id=S01_VOL_BASELINE)
        self.assertEqual(s01["submitted_order_intents_count"], 1)
        self.assertEqual(s01["submitted_close_intents_count"], 1)
        self.assertEqual(self.state_store.get_order_intent("s02:open")["status"], "created")
        self.assertEqual(self.state_store.get_close_intent("s02:close")["status"], "created")

        s02 = self.run_job(strategy_id=S02_VOL_ENHANCED)
        self.assertEqual(s02["submitted_order_intents_count"], 1)
        self.assertEqual(s02["submitted_close_intents_count"], 1)

        unknown = self.run_job(strategy_id="UNKNOWN")
        self.assertEqual(unknown["submitted_order_intents_count"], 0)
        self.assertEqual(unknown["submitted_close_intents_count"], 0)
        self.assertEqual(unknown["errors_count"], 0)

    def test_include_flags(self) -> None:
        self.create_order_intent("open:skip")
        self.create_close_intent("close:submit")
        close_only = self.run_job(include_open_intents=False)
        self.assertEqual(close_only["submitted_order_intents_count"], 0)
        self.assertEqual(close_only["submitted_close_intents_count"], 1)
        self.assertEqual(self.state_store.get_order_intent("open:skip")["status"], "created")

        self.create_order_intent("open:submit")
        self.create_close_intent("close:skip")
        open_only = self.run_job(include_close_intents=False)
        self.assertEqual(open_only["submitted_order_intents_count"], 2)
        self.assertEqual(open_only["submitted_close_intents_count"], 0)
        self.assertEqual(self.state_store.get_close_intent("close:skip")["status"], "created")

        before = deepcopy(self.state_store.state)
        before_events = self.all_events()
        neither = self.run_job(include_open_intents=False, include_close_intents=False)
        self.assertEqual(neither["submitted_order_intents_count"], 0)
        self.assertEqual(neither["submitted_close_intents_count"], 0)
        self.assertEqual(neither["skipped_order_intents_count"], 0)
        self.assertEqual(neither["skipped_close_intents_count"], 0)
        self.assertEqual(self.state_store.state, before)
        self.assertEqual(self.all_events(), before_events)

    def test_skip_non_created_intents(self) -> None:
        for status in ("submitted", "confirmed", "filled", "position_opened"):
            self.create_order_intent(f"open:{status}", status=status)
        for status in ("submitted", "confirmed", "filled", "position_closed"):
            self.create_close_intent(f"close:{status}", status=status)
        result = self.run_job()
        self.assertEqual(result["submitted_order_intents_count"], 0)
        self.assertEqual(result["submitted_close_intents_count"], 0)
        self.assertEqual(result["skipped_order_intents_count"], 4)
        self.assertEqual(result["skipped_close_intents_count"], 4)
        self.assertEqual(self.all_events(), [])
        self.assertEqual({entry["reason"] for entry in result["skipped"]}, {"status_not_created"})

    def test_per_intent_order_error_records_and_continues(self) -> None:
        self.create_order_intent("open:bad", dry_run=False)
        self.create_order_intent("open:ok")
        result = self.run_job()
        self.assertEqual(result["submitted_order_intents_count"], 1)
        self.assertEqual(result["errors_count"], 1)
        self.assertEqual(result["errors"][0]["intent_type"], "order_intent")
        json.dumps(result)
        self.assertEqual(self.state_store.get_order_intent("open:ok")["status"], "submitted")
        self.assertEqual(self.state_store.get_order_intent("open:bad")["status"], "created")

    def test_per_intent_close_error_records_and_continues(self) -> None:
        self.create_close_intent("close:bad", quantity="bad")
        self.create_close_intent("close:ok")
        result = self.run_job()
        self.assertEqual(result["submitted_close_intents_count"], 1)
        self.assertEqual(result["errors_count"], 1)
        self.assertEqual(result["errors"][0]["intent_type"], "close_intent")
        json.dumps(result)
        self.assertEqual(self.state_store.get_close_intent("close:ok")["status"], "submitted")
        self.assertEqual(self.state_store.get_close_intent("close:bad")["status"], "created")


class UnifiedSchedulerSubmissionRunOnceTests(Phase3SCase):
    def scheduler(self) -> UnifiedScheduler:
        return UnifiedScheduler(
            state_store=self.state_store,
            ledger=self.ledger,
            readiness_manager=self.readiness_manager,
        )

    def test_run_job_once_calls_submission_wrapper_with_injected_kwargs(self) -> None:
        expected = {"dry_run": True, "sentinel": "ok"}
        wrapper = mock.Mock(return_value=expected)
        adapter = DryRunExecutionAdapter()
        scheduler = self.scheduler()
        with mock.patch.dict(
            scheduler.run_job_once.__globals__,
            {"run_intent_submission_job": wrapper},
        ):
            result = scheduler.run_job_once(
                JOB_INTENT_SUBMISSION,
                state_store=self.state_store,
                ledger=self.ledger,
                now=NOW,
                execution_adapter=adapter,
                strategy_id=S01_VOL_BASELINE,
                include_open_intents=False,
                include_close_intents=True,
            )
        self.assertIs(result, expected)
        wrapper.assert_called_once_with(
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
            execution_adapter=adapter,
            strategy_id=S01_VOL_BASELINE,
            include_open_intents=False,
            include_close_intents=True,
        )

    def test_run_job_once_submits_created_intents_and_unknown_job_unchanged(self) -> None:
        self.create_order_intent("s01:open")
        result = self.scheduler().run_job_once(
            JOB_INTENT_SUBMISSION,
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
        )
        self.assertEqual(result["submitted_order_intents_count"], 1)
        with self.assertRaises(JobNotFoundError):
            self.scheduler().run_job_once("missing")


class SubmissionSafetyAndRegressionTests(Phase3SCase):
    def test_source_safety_scans(self) -> None:
        source = Path("algo_trader_unified/jobs/submission.py").read_text(encoding="utf-8")
        for forbidden in (
            "ib_insync",
            "yfinance",
            "requests",
            "broker.submit_order",
            "placeOrder",
            "cancelOrder",
            "LedgerAppender.append",
            "state_store.submit_order_intent",
            "state_store.submit_close_intent",
            ".jsonl",
            "confirm_order_intent",
            "confirm_fill",
            "open_position_from_filled_intent",
            "confirm_close_order",
            "confirm_close_fill",
            "close_position_from_filled_intent",
            "except:",
            ".start()",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("submit_order_intent(", source)
        self.assertIn("submit_close_intent(", source)

    def test_lifecycle_path_regression_events_remain_single_step(self) -> None:
        self.set_readiness()
        signal_input = VolSignalInput(
            symbol="XSP",
            current_date=date(2026, 4, 30),
            vix=18.0,
            iv_rank=45.0,
            target_dte=45,
            blackout_dates=(),
            order_ref_candidate=f"{S01_VOL_BASELINE}|P0430XSP|OPEN",
        )
        run_s01_vol_scan(
            readiness_manager=self.readiness_manager,
            state_store=self.state_store,
            ledger=self.ledger,
            current_time=datetime(2026, 4, 30, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: signal_input,
        )
        self.assertEqual([event["event_type"] for event in self.order_events()], ["ORDER_INTENT_CREATED"])
        self.assertNotIn("ORDER_SUBMITTED", {event["event_type"] for event in self.all_events()})

        intent_id = self.state_store.list_order_intents()[0]["intent_id"]
        submit_order_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            intent_id=intent_id,
            submitted_at="2026-04-30T14:00:00+00:00",
        )
        self.assertEqual(self.order_events()[-1]["event_type"], "ORDER_SUBMITTED")
        confirm_order_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            intent_id=intent_id,
            confirmed_at="2026-04-30T14:01:00+00:00",
        )
        self.assertEqual(self.order_events()[-1]["event_type"], "ORDER_CONFIRMED")
        confirm_fill(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            intent_id=intent_id,
            filled_at="2026-04-30T14:02:00+00:00",
        )
        self.assertEqual(self.order_events()[-1]["event_type"], "FILL_CONFIRMED")
        open_position_from_filled_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            intent_id=intent_id,
            opened_at="2026-04-30T14:03:00+00:00",
        )
        self.assertEqual(self.execution_events()[-1]["event_type"], "POSITION_OPENED")

        position_id = self.state_store.list_positions()[0]["position_id"]
        create_close_intent_from_position(
            state_store=self.state_store,
            ledger=self.ledger,
            position_id=position_id,
            created_at="2026-04-30T15:40:00+00:00",
        )
        self.assertEqual(self.order_events()[-1]["event_type"], "CLOSE_INTENT_CREATED")
        close_intent_id = self.state_store.list_close_intents()[0]["close_intent_id"]
        submit_close_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=close_intent_id,
            submitted_at="2026-04-30T15:41:00+00:00",
        )
        self.assertEqual(self.order_events()[-1]["event_type"], "CLOSE_ORDER_SUBMITTED")
        confirm_close_order(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=close_intent_id,
            confirmed_at="2026-04-30T15:42:00+00:00",
        )
        self.assertEqual(self.order_events()[-1]["event_type"], "CLOSE_ORDER_CONFIRMED")
        confirm_close_fill(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=close_intent_id,
            filled_at="2026-04-30T15:43:00+00:00",
        )
        self.assertEqual(self.order_events()[-1]["event_type"], "CLOSE_FILL_CONFIRMED")
        close_position_from_filled_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            close_intent_id=close_intent_id,
            closed_at="2026-04-30T15:44:00+00:00",
        )
        self.assertEqual(self.execution_events()[-1]["event_type"], "POSITION_CLOSED")

    def test_run_management_scan_still_creates_close_intent_only(self) -> None:
        from algo_trader_unified.core.management import ManagementSignalResult, run_management_scan

        self.create_position(S01_VOL_BASELINE, "position:management")
        result = run_management_scan(
            state_store=self.state_store,
            ledger=self.ledger,
            strategy_id=S01_VOL_BASELINE,
            now=NOW,
            management_signal_provider=lambda position, now: ManagementSignalResult(True),
        )
        self.assertEqual(result["close_intents_created_count"], 1)
        self.assertEqual([event["event_type"] for event in self.all_events()], ["CLOSE_INTENT_CREATED"])

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
        self.assertNotIn("commodity_vrp", joined.lower())

    def set_readiness(self) -> None:
        self.readiness_manager.update_readiness(
            ReadinessStatus(
                strategy_id=S01_VOL_BASELINE,
                ready_for_entries=True,
                reason=None,
                checked_at="2026-04-30T13:35:00+00:00",
                dirty_state=False,
                unknown_broker_exposure=False,
                nlv_degraded=False,
                halt_active=False,
                calendar_expired=False,
                iv_baseline_available=True,
            )
        )

    def test_no_dry_run_parameter_added_to_submission_job(self) -> None:
        self.assertNotIn("dry_run", inspect.signature(run_intent_submission_job).parameters)
