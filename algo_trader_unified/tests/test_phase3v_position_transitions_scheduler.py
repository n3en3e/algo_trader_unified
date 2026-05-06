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
from algo_trader_unified.config.scheduler import JOB_POSITION_TRANSITIONS, JOB_SPECS
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
from algo_trader_unified.jobs.confirmation import run_intent_confirmation_job
from algo_trader_unified.jobs.fill_confirmation import run_intent_fill_confirmation_job
from algo_trader_unified.jobs.management import run_management_scan_job
from algo_trader_unified.jobs.position_transitions import run_position_transitions_job
from algo_trader_unified.jobs.submission import run_intent_submission_job
from algo_trader_unified.jobs.vol import run_s01_vol_scan
from algo_trader_unified.strategies.vol.signals import VolSignalInput


NOW = "2026-05-04T16:02:00+00:00"


class Phase3VCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_store = StateStore(self.root / "data/state/portfolio_state.json")
        self.ledger = LedgerAppender(self.root)
        self.readiness_manager = ReadinessManager(self.state_store, self.ledger)

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
        status: str = "filled",
        filled_at: str | None = "2026-05-04T16:01:00+00:00",
        dry_run: bool | None = True,
        fill_price: object = 0.75,
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
            "order_submitted_event_id": f"evt_submitted_{intent_id}",
            "order_confirmed_event_id": f"evt_confirmed_{intent_id}",
            "fill_confirmed_event_id": f"evt_fill_{intent_id}",
            "order_ref": f"{strategy_id}|{intent_id}|OPEN",
            "created_at": "2026-05-04T15:40:00+00:00",
            "submitted_at": "2026-05-04T15:55:00+00:00",
            "confirmed_at": "2026-05-04T16:00:00+00:00",
            "updated_at": filled_at or "2026-05-04T16:00:00+00:00",
            "simulated_order_id": f"sim_{intent_id}",
            "fill_id": f"fill_{intent_id}",
            "fill_price": fill_price,
            "fill_quantity": 1,
            "sizing_context": {},
            "risk_context": {},
            "signal_payload_snapshot": {},
        }
        if filled_at is not None:
            record["filled_at"] = filled_at
        if dry_run is not None:
            record["dry_run"] = dry_run
        self.state_store.state["order_intents"][intent_id] = record
        self.state_store.save()
        return deepcopy(record)

    def position_record(self, strategy_id: str, position_id: str) -> dict:
        return {
            "position_id": position_id,
            "intent_id": f"{position_id}:intent",
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "status": "open",
            "execution_mode": "paper_only",
            "dry_run": True,
            "opened_at": "2026-05-04T15:00:00+00:00",
            "updated_at": "2026-05-04T15:00:00+00:00",
            "source_signal_event_id": "evt_signal",
            "order_intent_created_event_id": "evt_created",
            "order_submitted_event_id": "evt_submitted",
            "order_confirmed_event_id": "evt_confirmed",
            "fill_confirmed_event_id": "evt_fill",
            "position_opened_event_id": "evt_opened",
            "order_ref": f"{strategy_id}|{position_id}|OPEN",
            "simulated_order_id": f"sim:{position_id}",
            "fill_id": f"fill:{position_id}",
            "entry_price": 0.75,
            "quantity": 2,
            "action": "open",
        }

    def create_close_intent(
        self,
        close_intent_id: str,
        *,
        strategy_id: str = S01_VOL_BASELINE,
        position_id: str | None = None,
        status: str = "filled",
        filled_at: str | None = "2026-05-04T16:01:30+00:00",
        quantity: object = 2,
        position_status: str = "open",
    ) -> dict:
        position_id = position_id or f"position:{close_intent_id}"
        if self.state_store.get_position(position_id) is None:
            position = self.position_record(strategy_id, position_id)
            position["status"] = position_status
            self.state_store.state["positions"][position_id] = position
        record = {
            "close_intent_id": close_intent_id,
            "position_id": position_id,
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "status": status,
            "execution_mode": "paper_only",
            "dry_run": True,
            "created_at": "2026-05-04T15:45:00+00:00",
            "submitted_at": "2026-05-04T15:56:00+00:00",
            "confirmed_at": "2026-05-04T16:00:30+00:00",
            "updated_at": filled_at or "2026-05-04T16:00:30+00:00",
            "close_reason": "management",
            "requested_by": "phase3v",
            "position_opened_event_id": "evt_opened",
            "source_signal_event_id": "evt_signal",
            "fill_confirmed_event_id": "evt_fill",
            "close_intent_created_event_id": f"evt_created_{close_intent_id}",
            "close_order_submitted_event_id": f"evt_submitted_{close_intent_id}",
            "close_order_confirmed_event_id": f"evt_confirmed_{close_intent_id}",
            "close_fill_confirmed_event_id": f"evt_close_fill_{close_intent_id}",
            "close_order_ref": f"{strategy_id}:{position_id}:{close_intent_id}:close",
            "simulated_close_order_id": f"sim_close_{close_intent_id}",
            "close_fill_id": f"close_fill_{close_intent_id}",
            "close_fill_price": 0.85,
            "close_fill_quantity": quantity,
            "quantity": quantity,
            "entry_price": 0.75,
            "action": "close",
        }
        if filled_at is not None:
            record["filled_at"] = filled_at
        self.state_store.state["close_intents"][close_intent_id] = record
        self.state_store.state["positions"][position_id][
            "active_close_intent_id"
        ] = close_intent_id
        self.state_store.save()
        return deepcopy(record)

    def run_job(self, **kwargs) -> dict:
        return run_position_transitions_job(
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
            **kwargs,
        )

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


class SchedulerPositionTransitionConfigTests(unittest.TestCase):
    def test_position_transition_job_spec_is_registered_without_new_order_jobs(self) -> None:
        self.assertEqual(JOB_POSITION_TRANSITIONS, "position_transitions")
        self.assertIn(JOB_POSITION_TRANSITIONS, JOB_SPECS)
        self.assertTrue(JOB_SPECS[JOB_POSITION_TRANSITIONS].enabled)
        self.assertIn("Dry-run", JOB_SPECS[JOB_POSITION_TRANSITIONS].description)
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
            "intent_submission",
            "intent_confirmation",
            "intent_fill_confirmation",
            JOB_POSITION_TRANSITIONS,
            "dry_run_submit_pending_intents",
            "dry_run_expire_intents",
            "dry_run_eod_intent_cleanup",
            "dry_run_confirm_submitted_orders",
            "dry_run_confirm_fills",
            "dry_run_apply_position_transitions",
        }
        self.assertEqual(added_jobs, set())

    def test_no_live_scheduler_start_behavior(self) -> None:
        source = Path("algo_trader_unified/core/scheduler.py").read_text(encoding="utf-8")
        self.assertEqual(source.count(".start()"), 1)
        self.assertNotIn("BlockingScheduler", source)


class PositionTransitionJobTests(Phase3VCase):
    def test_no_filled_intents_returns_zeroes_without_mutation_or_writes(self) -> None:
        before = deepcopy(self.state_store.state)
        result = self.run_job()
        self.assertIs(result["dry_run"], True)
        for key in (
            "positions_opened_count",
            "positions_closed_count",
            "skipped_order_intents_count",
            "skipped_close_intents_count",
            "errors_count",
        ):
            self.assertEqual(result[key], 0)
        self.assertEqual(self.state_store.state, before)
        self.assertEqual(self.all_events(), [])

    def test_open_side_position_transition_uses_helper_only(self) -> None:
        self.create_order_intent("s01:open")
        result = self.run_job()
        intent = self.state_store.get_order_intent("s01:open")
        self.assertEqual(result["positions_opened_count"], 1)
        self.assertEqual(intent["status"], "position_opened")
        self.assertEqual([event["event_type"] for event in self.all_events()], ["POSITION_OPENED"])
        self.assertTrue(
            {"ORDER_SUBMITTED", "ORDER_CONFIRMED", "FILL_CONFIRMED"}.isdisjoint(
                {event["event_type"] for event in self.all_events()}
            )
        )

    def test_close_side_position_transition_uses_helper_only(self) -> None:
        self.create_close_intent("s01:close")
        result = self.run_job()
        close_intent = self.state_store.get_close_intent("s01:close")
        position = self.state_store.get_position("position:s01:close")
        self.assertEqual(result["positions_closed_count"], 1)
        self.assertEqual(close_intent["status"], "position_closed")
        self.assertEqual(position["status"], "closed")
        self.assertEqual([event["event_type"] for event in self.all_events()], ["POSITION_CLOSED"])
        self.assertTrue(
            {
                "CLOSE_ORDER_SUBMITTED",
                "CLOSE_ORDER_CONFIRMED",
                "CLOSE_FILL_CONFIRMED",
            }.isdisjoint({event["event_type"] for event in self.all_events()})
        )

    def test_mixed_batch_transitions_open_side_before_close_side_deterministically(self) -> None:
        calls = []

        def fake_open(*, state_store, ledger, intent_id, opened_at):
            calls.append(("order", intent_id))
            return {
                "position_id": f"position:{intent_id}",
                "intent_id": intent_id,
                "strategy_id": "mock",
                "status": "open",
                "opened_at": opened_at,
            }

        def fake_close(*, state_store, ledger, close_intent_id, closed_at):
            calls.append(("close", close_intent_id))
            return {
                "position_id": f"position:{close_intent_id}",
                "close_intent_id": close_intent_id,
                "strategy_id": "mock",
                "status": "closed",
                "closed_at": closed_at,
            }

        self.create_close_intent("close:b", strategy_id="CLOSE_B", filled_at="2026-05-04T16:01:40+00:00")
        self.create_order_intent("open:b", strategy_id="OPEN_B", filled_at="2026-05-04T16:01:10+00:00")
        self.create_close_intent("close:a", strategy_id="CLOSE_A", filled_at="2026-05-04T16:01:30+00:00")
        self.create_order_intent("open:missing", strategy_id="OPEN_MISSING", filled_at=None)
        self.create_order_intent("open:a", strategy_id="OPEN_A", filled_at="2026-05-04T16:01:10+00:00")
        with mock.patch.dict(
            run_position_transitions_job.__globals__,
            {
                "open_position_from_filled_intent": fake_open,
                "close_position_from_filled_intent": fake_close,
            },
        ):
            result = self.run_job()
        self.assertEqual(result["positions_opened_count"], 3)
        self.assertEqual(result["positions_closed_count"], 2)
        self.assertEqual(
            calls,
            [
                ("order", "open:missing"),
                ("order", "open:a"),
                ("order", "open:b"),
                ("close", "close:a"),
                ("close", "close:b"),
            ],
        )

    def test_strategy_filtering(self) -> None:
        self.create_order_intent("s01:open", strategy_id=S01_VOL_BASELINE)
        self.create_close_intent("s01:close", strategy_id=S01_VOL_BASELINE)
        self.create_order_intent("s02:open", strategy_id=S02_VOL_ENHANCED)
        self.create_close_intent("s02:close", strategy_id=S02_VOL_ENHANCED)

        s01 = self.run_job(strategy_id=S01_VOL_BASELINE)
        self.assertEqual(s01["positions_opened_count"], 0)
        self.assertEqual(s01["positions_closed_count"], 1)
        self.assertEqual(self.state_store.get_order_intent("s02:open")["status"], "filled")
        self.assertEqual(self.state_store.get_close_intent("s02:close")["status"], "filled")

        s02 = self.run_job(strategy_id=S02_VOL_ENHANCED)
        self.assertEqual(s02["positions_opened_count"], 0)
        self.assertEqual(s02["positions_closed_count"], 1)

        unknown = self.run_job(strategy_id="UNKNOWN")
        self.assertEqual(unknown["positions_opened_count"], 0)
        self.assertEqual(unknown["positions_closed_count"], 0)
        self.assertEqual(unknown["errors_count"], 0)

    def test_include_flags(self) -> None:
        self.create_order_intent("open:skip", strategy_id="OPEN_SKIP")
        self.create_close_intent("close:transition", strategy_id="CLOSE_ONLY")
        close_only = self.run_job(include_open_intents=False)
        self.assertEqual(close_only["positions_opened_count"], 0)
        self.assertEqual(close_only["positions_closed_count"], 1)
        self.assertEqual(self.state_store.get_order_intent("open:skip")["status"], "filled")

        self.create_order_intent("open:transition", strategy_id="OPEN_ONLY")
        self.create_close_intent("close:skip", strategy_id="CLOSE_SKIP")
        open_only = self.run_job(include_close_intents=False)
        self.assertEqual(open_only["positions_opened_count"], 2)
        self.assertEqual(open_only["positions_closed_count"], 0)
        self.assertEqual(self.state_store.get_close_intent("close:skip")["status"], "filled")

        before = deepcopy(self.state_store.state)
        before_events = self.all_events()
        neither = self.run_job(include_open_intents=False, include_close_intents=False)
        self.assertEqual(neither["positions_opened_count"], 0)
        self.assertEqual(neither["positions_closed_count"], 0)
        self.assertEqual(neither["skipped_order_intents_count"], 0)
        self.assertEqual(neither["skipped_close_intents_count"], 0)
        self.assertEqual(self.state_store.state, before)
        self.assertEqual(self.all_events(), before_events)

    def test_skip_non_filled_intents(self) -> None:
        for status in ("created", "submitted", "confirmed", "position_opened"):
            self.create_order_intent(f"open:{status}", strategy_id=f"OPEN_{status}", status=status)
        for status in ("created", "submitted", "confirmed", "position_closed"):
            self.create_close_intent(f"close:{status}", strategy_id=f"CLOSE_{status}", status=status)
        result = self.run_job()
        self.assertEqual(result["positions_opened_count"], 0)
        self.assertEqual(result["positions_closed_count"], 0)
        self.assertEqual(result["skipped_order_intents_count"], 4)
        self.assertEqual(result["skipped_close_intents_count"], 4)
        self.assertEqual(self.all_events(), [])
        self.assertEqual({entry["reason"] for entry in result["skipped"]}, {"status_not_filled"})

    def test_per_intent_order_error_records_and_continues(self) -> None:
        self.create_order_intent("open:bad", strategy_id="OPEN_BAD", dry_run=False)
        self.create_order_intent("open:ok", strategy_id="OPEN_OK")
        result = self.run_job()
        self.assertEqual(result["positions_opened_count"], 1)
        self.assertEqual(result["errors_count"], 1)
        self.assertEqual(result["errors"][0]["intent_type"], "order_intent")
        json.dumps(result)
        self.assertEqual(self.state_store.get_order_intent("open:ok")["status"], "position_opened")
        self.assertEqual(self.state_store.get_order_intent("open:bad")["status"], "filled")

    def test_per_intent_close_error_records_and_continues(self) -> None:
        self.create_close_intent("close:bad", strategy_id="CLOSE_BAD", quantity="bad")
        self.create_close_intent("close:ok", strategy_id="CLOSE_OK")
        result = self.run_job()
        self.assertEqual(result["positions_closed_count"], 1)
        self.assertEqual(result["errors_count"], 1)
        self.assertEqual(result["errors"][0]["intent_type"], "close_intent")
        json.dumps(result)
        self.assertEqual(self.state_store.get_close_intent("close:ok")["status"], "position_closed")
        self.assertEqual(self.state_store.get_close_intent("close:bad")["status"], "filled")


class UnifiedSchedulerPositionTransitionRunOnceTests(Phase3VCase):
    def scheduler(self) -> UnifiedScheduler:
        return UnifiedScheduler(
            state_store=self.state_store,
            ledger=self.ledger,
            readiness_manager=self.readiness_manager,
        )

    def test_run_job_once_calls_position_wrapper_with_injected_kwargs(self) -> None:
        expected = {"dry_run": True, "sentinel": "ok"}
        wrapper = mock.Mock(return_value=expected)
        scheduler = self.scheduler()
        with mock.patch.dict(
            scheduler.run_job_once.__globals__,
            {"run_position_transitions_job": wrapper},
        ):
            result = scheduler.run_job_once(
                JOB_POSITION_TRANSITIONS,
                state_store=self.state_store,
                ledger=self.ledger,
                now=NOW,
                strategy_id=S01_VOL_BASELINE,
                include_open_intents=False,
                include_close_intents=True,
            )
        self.assertIs(result, expected)
        wrapper.assert_called_once_with(
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
            strategy_id=S01_VOL_BASELINE,
            include_open_intents=False,
            include_close_intents=True,
        )

    def test_run_job_once_transitions_filled_intents_and_unknown_job_unchanged(self) -> None:
        self.create_order_intent("s01:open")
        result = self.scheduler().run_job_once(
            JOB_POSITION_TRANSITIONS,
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
        )
        self.assertEqual(result["positions_opened_count"], 1)
        with self.assertRaises(JobNotFoundError):
            self.scheduler().run_job_once("missing")


class PositionTransitionSafetyAndRegressionTests(Phase3VCase):
    def test_source_safety_scans(self) -> None:
        source = Path("algo_trader_unified/jobs/position_transitions.py").read_text(encoding="utf-8")
        for forbidden in (
            "ib_insync",
            "yfinance",
            "requests",
            "broker.submit_order",
            "placeOrder",
            "cancelOrder",
            "LedgerAppender.append",
            "state_store.create_open_position",
            "state_store.close_position",
            "state_store.mark_intent_position_opened",
            "state_store.mark_close_intent_position_closed",
            ".jsonl",
            "submit_order_intent(",
            "submit_close_intent(",
            "confirm_order_intent(",
            "confirm_close_order(",
            "confirm_fill(",
            "confirm_close_fill(",
            "except:",
            ".start()",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("open_position_from_filled_intent(", source)
        self.assertIn("close_position_from_filled_intent(", source)

    def test_lifecycle_path_regression_events_remain_single_step(self) -> None:
        self.set_readiness()
        signal_input = VolSignalInput(
            symbol="XSP",
            current_date=date(2026, 5, 4),
            vix=18.0,
            iv_rank=45.0,
            target_dte=45,
            blackout_dates=(),
            order_ref_candidate=f"{S01_VOL_BASELINE}|P0504XSP|OPEN",
        )
        run_s01_vol_scan(
            readiness_manager=self.readiness_manager,
            state_store=self.state_store,
            ledger=self.ledger,
            current_time=datetime(2026, 5, 4, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: signal_input,
        )
        self.assertEqual([event["event_type"] for event in self.order_events()], ["ORDER_INTENT_CREATED"])

        intent_id = self.state_store.list_order_intents()[0]["intent_id"]
        submit_order_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            intent_id=intent_id,
            submitted_at="2026-05-04T14:00:00+00:00",
        )
        self.assertEqual(self.order_events()[-1]["event_type"], "ORDER_SUBMITTED")
        confirm_order_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            intent_id=intent_id,
            confirmed_at="2026-05-04T14:01:00+00:00",
        )
        self.assertEqual(self.order_events()[-1]["event_type"], "ORDER_CONFIRMED")
        confirm_fill(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            intent_id=intent_id,
            filled_at="2026-05-04T14:02:00+00:00",
        )
        self.assertEqual(self.order_events()[-1]["event_type"], "FILL_CONFIRMED")

        before_fill_job_events = len(self.order_events())
        fill_job = run_intent_fill_confirmation_job(
            state_store=self.state_store,
            ledger=self.ledger,
            now="2026-05-04T14:02:30+00:00",
        )
        self.assertEqual(fill_job["filled_order_intents_count"], 0)
        self.assertEqual(len(self.order_events()), before_fill_job_events)

        open_position_from_filled_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            intent_id=intent_id,
            opened_at="2026-05-04T14:03:00+00:00",
        )
        self.assertEqual(self.execution_events()[-1]["event_type"], "POSITION_OPENED")

        position_id = self.state_store.list_positions()[0]["position_id"]
        create_close_intent_from_position(
            state_store=self.state_store,
            ledger=self.ledger,
            position_id=position_id,
            created_at="2026-05-04T15:40:00+00:00",
        )
        self.assertEqual(self.order_events()[-1]["event_type"], "CLOSE_INTENT_CREATED")
        close_intent_id = self.state_store.list_close_intents()[0]["close_intent_id"]
        submit_close_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=close_intent_id,
            submitted_at="2026-05-04T15:41:00+00:00",
        )
        self.assertEqual(self.order_events()[-1]["event_type"], "CLOSE_ORDER_SUBMITTED")
        confirm_close_order(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=close_intent_id,
            confirmed_at="2026-05-04T15:42:00+00:00",
        )
        self.assertEqual(self.order_events()[-1]["event_type"], "CLOSE_ORDER_CONFIRMED")
        confirm_close_fill(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=close_intent_id,
            filled_at="2026-05-04T15:43:00+00:00",
        )
        self.assertEqual(self.order_events()[-1]["event_type"], "CLOSE_FILL_CONFIRMED")
        close_position_from_filled_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            close_intent_id=close_intent_id,
            closed_at="2026-05-04T15:44:00+00:00",
        )
        self.assertEqual(self.execution_events()[-1]["event_type"], "POSITION_CLOSED")

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

    def test_no_dry_run_parameter_added_to_position_transition_job(self) -> None:
        self.assertNotIn("dry_run", inspect.signature(run_position_transitions_job).parameters)
