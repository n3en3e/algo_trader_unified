from __future__ import annotations

import inspect
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.scheduler import (
    JOB_S01_MANAGEMENT_SCAN,
    JOB_S02_MANAGEMENT_SCAN,
    JOB_SPECS,
)
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.ledger_reader import LedgerReader
from algo_trader_unified.core.management import ManagementSignalResult
from algo_trader_unified.core.readiness import ReadinessManager
from algo_trader_unified.core.scheduler import JobNotFoundError, UnifiedScheduler
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.jobs import management as management_job


NOW = "2026-04-29T15:50:00+00:00"


class Phase3RCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_store = StateStore(self.root / "data/state/portfolio_state.json")
        self.ledger = LedgerAppender(self.root)
        self.readiness_manager = ReadinessManager(self.state_store, self.ledger)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def scheduler(self) -> UnifiedScheduler:
        return UnifiedScheduler(
            state_store=self.state_store,
            ledger=self.ledger,
            readiness_manager=self.readiness_manager,
        )

    def reader(self) -> LedgerReader:
        return LedgerReader.from_root(self.root)

    def order_events(self) -> list[dict]:
        return self.reader().read_events("order")

    def execution_events(self) -> list[dict]:
        return self.reader().read_events("execution")

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
            "opened_at": "2026-04-29T14:07:00+00:00",
            "updated_at": "2026-04-29T14:07:00+00:00",
            "source_signal_event_id": "evt_signal",
            "order_intent_created_event_id": "evt_created",
            "order_submitted_event_id": "evt_submitted",
            "order_confirmed_event_id": "evt_confirmed",
            "fill_confirmed_event_id": "evt_fill",
            "position_opened_event_id": "evt_opened",
            "order_ref": f"{strategy_id}|P0429XSP|OPEN",
            "simulated_order_id": f"sim:{position_id}",
            "fill_id": f"fill:{position_id}",
            "entry_price": 0.75,
            "quantity": 2,
            "action": "open",
        }

    def create_position(self, strategy_id: str, position_id: str) -> dict:
        record = self.position_record(strategy_id, position_id)
        self.state_store.state["positions"][position_id] = deepcopy(record)
        self.state_store.save()
        return record

    def attach_close_intent(self, strategy_id: str, position_id: str) -> None:
        close_intent_id = f"{strategy_id}:{position_id}:close"
        self.state_store.state["close_intents"][close_intent_id] = {
            "close_intent_id": close_intent_id,
            "position_id": position_id,
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "status": "created",
            "execution_mode": "paper_only",
            "dry_run": True,
            "created_at": NOW,
            "updated_at": NOW,
            "close_reason": "management",
            "requested_by": "management",
            "position_opened_event_id": "evt_opened",
            "source_signal_event_id": "evt_signal",
            "fill_confirmed_event_id": "evt_fill",
            "close_intent_created_event_id": f"evt_{close_intent_id}",
            "quantity": 2,
            "entry_price": 0.75,
            "action": "close",
        }
        self.state_store.state["positions"][position_id][
            "active_close_intent_id"
        ] = close_intent_id
        self.state_store.save()


class SchedulerManagementConfigTests(unittest.TestCase):
    def test_management_job_specs_are_registered_and_dry_run_only(self) -> None:
        self.assertEqual(JOB_S01_MANAGEMENT_SCAN, "s01_management_scan")
        self.assertEqual(JOB_S02_MANAGEMENT_SCAN, "s02_management_scan")
        self.assertIn(JOB_S01_MANAGEMENT_SCAN, JOB_SPECS)
        self.assertIn(JOB_S02_MANAGEMENT_SCAN, JOB_SPECS)
        self.assertTrue(JOB_SPECS[JOB_S01_MANAGEMENT_SCAN].enabled)
        self.assertTrue(JOB_SPECS[JOB_S02_MANAGEMENT_SCAN].enabled)
        self.assertIn("Dry-run", JOB_SPECS[JOB_S01_MANAGEMENT_SCAN].description)
        self.assertIn("Dry-run", JOB_SPECS[JOB_S02_MANAGEMENT_SCAN].description)
        self.assertFalse(any("0dte" in job_id.lower() for job_id in JOB_SPECS))


class ManagementJobWrapperTests(Phase3RCase):
    def test_wrapper_calls_core_runner_and_returns_result_unchanged(self) -> None:
        provider = mock.Mock(return_value=ManagementSignalResult(False))
        expected = {"dry_run": True, "sentinel": object()}
        with mock.patch(
            "algo_trader_unified.jobs.management.run_management_scan",
            return_value=expected,
        ) as run_scan:
            result = management_job.run_management_scan_job(
                strategy_id=S01_VOL_BASELINE,
                state_store=self.state_store,
                ledger=self.ledger,
                now=NOW,
                management_signal_provider=provider,
            )

        self.assertIs(result, expected)
        run_scan.assert_called_once_with(
            state_store=self.state_store,
            ledger=self.ledger,
            strategy_id=S01_VOL_BASELINE,
            management_signal_provider=provider,
            now=NOW,
        )
        source = Path("algo_trader_unified/jobs/management.py").read_text(encoding="utf-8")
        self.assertNotIn('result["dry_run"]', source)
        self.assertNotIn("result['dry_run']", source)

    def test_default_provider_no_action_has_no_writes_or_state_mutation(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01")
        before_state = deepcopy(self.state_store.state)

        result = management_job.run_management_scan_job(
            strategy_id=S01_VOL_BASELINE,
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
        )

        self.assertIs(result["dry_run"], True)
        self.assertEqual(result["no_action_count"], 1)
        self.assertEqual(result["close_intents_created_count"], 0)
        self.assertEqual(self.state_store.state, before_state)
        self.assertEqual(self.order_events(), [])
        self.assertEqual(self.execution_events(), [])


class UnifiedSchedulerManagementRunOnceTests(Phase3RCase):
    def test_run_job_once_calls_management_wrapper_with_injected_kwargs(self) -> None:
        provider = mock.Mock(return_value=ManagementSignalResult(False))
        expected = {"dry_run": True, "close_intents_created_count": 0}
        scheduler = self.scheduler()
        wrapper = mock.Mock(return_value=expected)

        with mock.patch.dict(
            scheduler.run_job_once.__globals__,
            {"run_management_scan_job": wrapper},
        ):
            result = scheduler.run_job_once(
                JOB_S01_MANAGEMENT_SCAN,
                state_store=self.state_store,
                ledger=self.ledger,
                now=NOW,
                management_signal_provider=provider,
            )

        self.assertIs(result, expected)
        wrapper.assert_called_once_with(
            strategy_id=S01_VOL_BASELINE,
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
            management_signal_provider=provider,
        )

    def test_duplicate_strategy_id_kwarg_is_removed_and_routed_strategy_wins(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01")
        self.create_position(S02_VOL_ENHANCED, "position:s02")

        result = self.scheduler().run_job_once(
            JOB_S01_MANAGEMENT_SCAN,
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
            strategy_id=S02_VOL_ENHANCED,
            management_signal_provider=lambda position, now: ManagementSignalResult(False),
        )

        self.assertEqual(result["evaluated_count"], 1)
        self.assertEqual(result["no_action"][0]["strategy_id"], S01_VOL_BASELINE)

    def test_default_provider_through_scheduler_is_no_action(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01")
        before_state = deepcopy(self.state_store.state)

        result = self.scheduler().run_job_once(
            JOB_S01_MANAGEMENT_SCAN,
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
        )

        self.assertIs(result["dry_run"], True)
        self.assertEqual(result["close_intents_created_count"], 0)
        self.assertEqual(result["no_action_count"], 1)
        self.assertEqual(self.state_store.state, before_state)
        self.assertEqual(self.order_events(), [])
        self.assertEqual(self.execution_events(), [])

    def test_injected_no_action_provider_has_no_writes_or_state_mutation(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01")
        before_state = deepcopy(self.state_store.state)

        result = self.scheduler().run_job_once(
            JOB_S01_MANAGEMENT_SCAN,
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
            management_signal_provider=lambda position, now: ManagementSignalResult(False),
        )

        self.assertEqual(result["no_action_count"], 1)
        self.assertEqual(self.state_store.state, before_state)
        self.assertEqual(self.order_events(), [])
        self.assertEqual(self.execution_events(), [])

    def test_injected_close_provider_creates_close_intent_only(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01")

        result = self.scheduler().run_job_once(
            JOB_S01_MANAGEMENT_SCAN,
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
            management_signal_provider=lambda position, now: ManagementSignalResult(
                True,
                close_reason="scheduler test",
                requested_by="phase3r",
            ),
        )

        self.assertEqual(result["close_intents_created_count"], 1)
        position = self.state_store.get_position("position:s01")
        self.assertEqual(position["status"], "open")
        self.assertIn("active_close_intent_id", position)
        events = self.order_events() + self.execution_events()
        self.assertEqual([event["event_type"] for event in events], ["CLOSE_INTENT_CREATED"])
        self.assertTrue(
            {
                "CLOSE_ORDER_SUBMITTED",
                "CLOSE_ORDER_CONFIRMED",
                "CLOSE_FILL_CONFIRMED",
                "POSITION_CLOSED",
                "POSITION_ADJUSTED",
            }.isdisjoint({event["event_type"] for event in events})
        )

    def test_s01_and_s02_management_jobs_route_to_their_own_positions(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01")
        self.create_position(S02_VOL_ENHANCED, "position:s02")
        calls = []

        def provider(position, now):
            calls.append(position["position_id"])
            return ManagementSignalResult(False)

        s01 = self.scheduler().run_job_once(
            JOB_S01_MANAGEMENT_SCAN,
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
            management_signal_provider=provider,
        )
        s02 = self.scheduler().run_job_once(
            JOB_S02_MANAGEMENT_SCAN,
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
            management_signal_provider=provider,
        )

        self.assertEqual(s01["evaluated_count"], 1)
        self.assertEqual(s02["evaluated_count"], 1)
        self.assertEqual(calls, ["position:s01", "position:s02"])
        self.assertEqual(self.order_events(), [])
        self.assertEqual(self.execution_events(), [])

    def test_active_close_intent_is_skipped_and_provider_not_called(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01")
        self.attach_close_intent(S01_VOL_BASELINE, "position:s01")
        provider = mock.Mock(return_value=ManagementSignalResult(True))

        result = self.scheduler().run_job_once(
            JOB_S01_MANAGEMENT_SCAN,
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
            management_signal_provider=provider,
        )

        provider.assert_not_called()
        self.assertEqual(result["skipped_active_close_intent_count"], 1)
        self.assertEqual(result["close_intents_created_count"], 0)
        self.assertEqual(self.order_events(), [])

    def test_provider_error_is_captured_and_other_positions_continue(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:error")
        self.create_position(S01_VOL_BASELINE, "position:ok")

        def provider(position, now):
            if position["position_id"] == "position:error":
                raise RuntimeError("provider failed")
            return ManagementSignalResult(False)

        result = self.scheduler().run_job_once(
            JOB_S01_MANAGEMENT_SCAN,
            state_store=self.state_store,
            ledger=self.ledger,
            now=NOW,
            management_signal_provider=provider,
        )

        self.assertEqual(result["errors_count"], 1)
        self.assertEqual(result["no_action_count"], 1)
        self.assertEqual(result["errors"][0]["stage"], "management_signal_provider")
        self.assertEqual(self.order_events(), [])
        self.assertEqual(self.execution_events(), [])

    def test_missing_infrastructure_dependencies_propagate_clearly(self) -> None:
        with self.assertRaises(TypeError):
            self.scheduler().run_job_once(
                JOB_S01_MANAGEMENT_SCAN,
                state_store=self.state_store,
                ledger=self.ledger,
            )

    def test_unknown_job_still_raises(self) -> None:
        with self.assertRaises(JobNotFoundError):
            self.scheduler().run_job_once("missing")


class SchedulerManagementSafetyTests(unittest.TestCase):
    def test_source_safety_scans(self) -> None:
        sources = {
            "scheduler": Path("algo_trader_unified/core/scheduler.py").read_text(encoding="utf-8"),
            "job": Path("algo_trader_unified/jobs/management.py").read_text(encoding="utf-8"),
            "config": Path("algo_trader_unified/config/scheduler.py").read_text(encoding="utf-8"),
        }
        job_source = sources["job"]
        self.assertNotIn(".jsonl", job_source)
        self.assertNotIn("StateStore(", job_source)
        for forbidden in (
            "ib_insync",
            "yfinance",
            "requests",
            "broker.submit_order",
            "placeOrder",
            "cancelOrder",
            "submit_close_intent(",
            "confirm_close_order(",
            "confirm_close_fill(",
            "close_position_from_filled_intent(",
            "except:",
        ):
            for source in sources.values():
                self.assertNotIn(forbidden, source)
        scheduler_source = sources["scheduler"]
        self.assertEqual(scheduler_source.count(".start()"), 1)
        self.assertNotIn("BlockingScheduler", scheduler_source)
        self.assertNotIn("systemd", "\n".join(sources.values()).lower())
        self.assertNotIn("0DTE", "\n".join(sources.values()))

    def test_run_management_scan_signature_still_has_no_dry_run_parameter(self) -> None:
        from algo_trader_unified.core.management import run_management_scan

        self.assertNotIn("dry_run", inspect.signature(run_management_scan).parameters)
