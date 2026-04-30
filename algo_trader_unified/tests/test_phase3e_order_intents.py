from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.scheduler import JOB_S01_VOL_SCAN, JOB_S02_VOL_SCAN
from algo_trader_unified.core.ledger import KNOWN_EVENT_TYPES, LedgerAppender
from algo_trader_unified.core.ledger_reader import LedgerReader
from algo_trader_unified.core.readiness import ReadinessManager, ReadinessStatus
from algo_trader_unified.core.scheduler import UnifiedScheduler
from algo_trader_unified.core.skip_reasons import SKIP_ACTIVE_ORDER_INTENT
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.jobs.vol import run_s01_vol_scan, run_s02_vol_scan
from algo_trader_unified.strategies.vol.signals import SignalResult, VolSignalInput


class TmpCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_store = StateStore(self.root / "data/state/portfolio_state.json")
        self.ledger = LedgerAppender(self.root)
        self.manager = ReadinessManager(self.state_store, self.ledger)
        self.execution_path = self.root / "data/ledger/execution_ledger.jsonl"
        self.order_path = self.root / "data/ledger/order_ledger.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def set_readiness(self, strategy_id: str) -> None:
        self.manager.update_readiness(
            ReadinessStatus(
                strategy_id=strategy_id,
                ready_for_entries=True,
                reason=None,
                checked_at="2026-04-27T13:35:00+00:00",
                dirty_state=False,
                unknown_broker_exposure=False,
                nlv_degraded=False,
                halt_active=False,
                calendar_expired=False,
                iv_baseline_available=True,
            )
        )

    def reader(self) -> LedgerReader:
        return LedgerReader(
            execution_ledger_path=self.execution_path,
            order_ledger_path=self.order_path,
        )

    def events(self, path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def clean_input(self, strategy_id: str, order_ref: str | None = None) -> VolSignalInput:
        return VolSignalInput(
            symbol="XSP",
            current_date=date(2026, 4, 27),
            vix=18.0,
            iv_rank=45.0,
            target_dte=45,
            blackout_dates=(),
            order_ref_candidate=order_ref or f"{strategy_id}|P0427XSP|OPEN",
        )

    def create_intent(self, strategy_id: str, intent_id: str | None = None) -> dict:
        now = "2026-04-27T13:40:00+00:00"
        record = {
            "intent_id": intent_id or f"{strategy_id}:manual",
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "execution_mode": "paper_only",
            "status": "created",
            "source_signal_event_id": None,
            "order_intent_created_event_id": "evt_manual",
            "order_ref": "same-order-ref",
            "created_at": now,
            "updated_at": now,
            "sizing_context": {},
            "risk_context": {},
            "signal_payload_snapshot": {},
        }
        return self.state_store.create_order_intent(record)


class StateStoreOrderIntentTests(TmpCase):
    def test_fresh_and_legacy_state_include_order_intents(self) -> None:
        self.assertEqual(self.state_store.state["order_intents"], {})
        legacy_path = self.root / "data/state/legacy_state.json"
        payload = {
            "schema_version": 1,
            "positions": [],
            "opportunities": [],
            "orders": [],
            "fills": [],
            "strategy_snapshots": [],
            "account_snapshots": [],
            "reconciliation_snapshots": [],
            "halt_state": None,
            "readiness": {"strategies": {}},
        }
        legacy_path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = StateStore(legacy_path)
        self.assertEqual(loaded.state["order_intents"], {})

    def test_create_get_active_and_list_order_intents(self) -> None:
        s01 = self.create_intent(S01_VOL_BASELINE, "s01:intent")
        self.create_intent(S02_VOL_ENHANCED, "s02:intent")
        self.assertEqual(self.state_store.get_order_intent("s01:intent"), s01)
        self.assertEqual(
            self.state_store.get_active_order_intent(S01_VOL_BASELINE)["intent_id"],
            "s01:intent",
        )
        self.assertEqual(
            self.state_store.get_active_order_intent(S02_VOL_ENHANCED)["intent_id"],
            "s02:intent",
        )
        self.assertEqual(len(self.state_store.list_order_intents()), 2)
        self.assertEqual(len(self.state_store.list_order_intents(S01_VOL_BASELINE)), 1)
        self.assertNotIn("dry_run", self.state_store.get_order_intent("s01:intent"))


class LedgerOrderIntentTests(TmpCase):
    def test_order_intent_created_is_known_and_routes_to_order_ledger(self) -> None:
        self.assertIn("ORDER_INTENT_CREATED", KNOWN_EVENT_TYPES)
        event_id = self.ledger.append(
            event_type="ORDER_INTENT_CREATED",
            strategy_id=S01_VOL_BASELINE,
            execution_mode="paper_only",
            source_module="test",
            payload={
                "intent_id": "intent",
                "strategy_id": S01_VOL_BASELINE,
                "sleeve_id": "VOL",
                "symbol": "XSP",
                "execution_mode": "paper_only",
                "source_signal_event_id": "evt_signal",
                "order_ref": "ref",
                "intent_status": "created",
                "created_at": "2026-04-27T13:40:00+00:00",
                "event_detail": "ORDER_INTENT_CREATED",
                "sizing_context": {},
                "risk_context": {},
                "signal_payload_snapshot": {},
                "dry_run": True,
            },
        )
        self.assertIsInstance(event_id, str)
        self.assertIn(event_id, self.order_path.read_text(encoding="utf-8"))
        self.assertNotIn(event_id, self.execution_path.read_text(encoding="utf-8"))


class VolOrderIntentCreationTests(TmpCase):
    def test_s01_clean_signal_creates_order_intent_and_state_record(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        broker = mock.Mock()
        broker.submit_order = mock.Mock()
        result = run_s01_vol_scan(
            readiness_manager=self.manager,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=broker,
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: self.clean_input(S01_VOL_BASELINE),
            ledger_reader=self.reader(),
        )
        execution_events = self.events(self.execution_path)
        order_events = self.events(self.order_path)
        self.assertEqual(result.detail, "order_intent_created")
        self.assertIsNotNone(result.order_intent_id)
        self.assertIsNotNone(result.order_intent_created_event_id)
        self.assertEqual([event["event_type"] for event in execution_events], ["SIGNAL_GENERATED"])
        self.assertEqual([event["event_type"] for event in order_events], ["ORDER_INTENT_CREATED"])
        self.assertEqual(execution_events[0]["source_module"], "jobs.vol")
        self.assertTrue(order_events[0]["payload"]["dry_run"])
        self.assertEqual(
            order_events[0]["payload"]["source_signal_event_id"],
            execution_events[0]["event_id"],
        )
        intent = self.state_store.get_order_intent(result.order_intent_id)
        self.assertEqual(intent["order_intent_created_event_id"], order_events[0]["event_id"])
        self.assertEqual(intent["source_signal_event_id"], execution_events[0]["event_id"])
        self.assertTrue(intent["dry_run"])
        self.assertNotIn("ORDER_SUBMITTED", self.order_path.read_text(encoding="utf-8"))
        self.assertNotIn("POSITION_", self.execution_path.read_text(encoding="utf-8"))
        broker.submit_order.assert_not_called()

    def test_s02_clean_signal_creates_isolated_order_intent(self) -> None:
        self.set_readiness(S02_VOL_ENHANCED)
        result = run_s02_vol_scan(
            readiness_manager=self.manager,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=mock.Mock(),
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: self.clean_input(S02_VOL_ENHANCED),
            ledger_reader=self.reader(),
        )
        intent = self.state_store.get_order_intent(result.order_intent_id)
        self.assertEqual(intent["strategy_id"], S02_VOL_ENHANCED)
        self.assertTrue(intent["dry_run"])
        self.assertEqual(self.events(self.execution_path)[0]["source_module"], "jobs.vol")
        self.assertEqual(result.detail, "order_intent_created")

    def test_active_intent_precheck_skips_before_provider_and_engine(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        self.create_intent(S01_VOL_BASELINE)
        provider = mock.Mock(return_value=self.clean_input(S01_VOL_BASELINE))
        with mock.patch("algo_trader_unified.jobs.vol.VolSellingEngine") as engine_class:
            result = run_s01_vol_scan(
                readiness_manager=self.manager,
                state_store=self.state_store,
                ledger=self.ledger,
                broker=mock.Mock(),
                current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
                signal_context_provider=provider,
                ledger_reader=self.reader(),
            )
        self.assertEqual(result.detail, "active_order_intent")
        provider.assert_not_called()
        engine_class.assert_not_called()
        events = self.events(self.execution_path)
        self.assertEqual(events[-1]["event_type"], "SIGNAL_SKIPPED")
        self.assertEqual(events[-1]["payload"]["skip_reason"], SKIP_ACTIVE_ORDER_INTENT)
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")

    def test_final_recheck_blocks_concurrent_active_intent(self) -> None:
        self.set_readiness(S02_VOL_ENHANCED)
        active = {
            "intent_id": "s02:concurrent",
            "strategy_id": S02_VOL_ENHANCED,
            "status": "created",
        }
        with mock.patch.object(
            self.state_store,
            "get_active_order_intent",
            side_effect=[None, active],
        ):
            result = run_s02_vol_scan(
                readiness_manager=self.manager,
                state_store=self.state_store,
                ledger=self.ledger,
                broker=mock.Mock(),
                current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
                signal_context_provider=lambda: self.clean_input(S02_VOL_ENHANCED),
                ledger_reader=self.reader(),
            )
        self.assertEqual(result.detail, "active_order_intent")
        self.assertEqual([event["event_type"] for event in self.events(self.execution_path)], [
            "SIGNAL_GENERATED",
            "SIGNAL_SKIPPED",
        ])
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")

    def test_s01_and_s02_active_intents_are_isolated_and_ids_differ(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        self.set_readiness(S02_VOL_ENHANCED)
        self.create_intent(S01_VOL_BASELINE, "S01_VOL_BASELINE:same-ref")
        s02 = run_s02_vol_scan(
            readiness_manager=self.manager,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=mock.Mock(),
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: self.clean_input(S02_VOL_ENHANCED, "same-ref"),
            ledger_reader=self.reader(),
        )
        self.assertEqual(s02.detail, "order_intent_created")
        self.assertNotEqual(s02.order_intent_id, "S01_VOL_BASELINE:same-ref")
        self.assertEqual(
            self.state_store.get_order_intent(s02.order_intent_id)["strategy_id"],
            S02_VOL_ENHANCED,
        )


class LockSpy:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.depth = 0

    @property
    def held(self) -> bool:
        return self.depth > 0

    def __enter__(self):
        self._lock.acquire()
        self.depth += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.depth -= 1
        self._lock.release()


class LockingTests(TmpCase):
    def test_provider_and_signal_evaluation_happen_outside_strategy_lock(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        spy = LockSpy()
        self.state_store.strategy_state_locks[S01_VOL_BASELINE] = spy
        observed = {"precheck": False, "provider": False, "engine": False, "final": False}
        original_get_active = self.state_store.get_active_order_intent

        def get_active(strategy_id: str):
            self.assertTrue(spy.held)
            if observed["precheck"]:
                observed["final"] = True
            else:
                observed["precheck"] = True
            return original_get_active(strategy_id)

        def provider() -> VolSignalInput:
            observed["provider"] = True
            self.assertFalse(spy.held)
            return self.clean_input(S01_VOL_BASELINE)

        engine = mock.Mock()

        def evaluate(signal_input, log_to_ledger=True) -> SignalResult:
            observed["engine"] = True
            self.assertFalse(spy.held)
            return SignalResult(
                should_enter=True,
                skip_reason=None,
                skip_detail=None,
                sizing_context={"capital": 90000.0, "allocation_pct": 0.225},
                risk_context={"execution_mode": "paper_only", "strategy_id": S01_VOL_BASELINE},
            )

        engine.generate_standard_strangle_signal.side_effect = evaluate
        with mock.patch.object(self.state_store, "get_active_order_intent", side_effect=get_active):
            result = run_s01_vol_scan(
                readiness_manager=self.manager,
                state_store=self.state_store,
                ledger=self.ledger,
                broker=mock.Mock(),
                current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
                signal_context_provider=provider,
                engine=engine,
                ledger_reader=self.reader(),
            )
        self.assertEqual(result.detail, "order_intent_created")
        self.assertEqual(observed, {
            "precheck": True,
            "provider": True,
            "engine": True,
            "final": True,
        })


class SchedulerOrderIntentTests(TmpCase):
    def test_run_job_once_creates_s01_and_s02_order_intents(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        self.set_readiness(S02_VOL_ENHANCED)
        scheduler = UnifiedScheduler(
            state_store=self.state_store,
            ledger=self.ledger,
            readiness_manager=self.manager,
        )
        s01 = scheduler.run_job_once(
            JOB_S01_VOL_SCAN,
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: self.clean_input(S01_VOL_BASELINE),
            ledger_reader=self.reader(),
            broker=mock.Mock(),
        )
        s02 = scheduler.run_job_once(
            JOB_S02_VOL_SCAN,
            current_time=datetime(2026, 4, 28, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: self.clean_input(S02_VOL_ENHANCED),
            ledger_reader=self.reader(),
            broker=mock.Mock(),
        )
        self.assertEqual(s01.detail, "order_intent_created")
        self.assertEqual(s02.detail, "order_intent_created")
        self.assertEqual(len(self.state_store.list_order_intents()), 2)


class FailureOrderTests(TmpCase):
    def test_order_intent_ledger_append_failure_does_not_mutate_state(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        real_append = self.ledger.append

        def append(**kwargs):
            if kwargs["event_type"] == "ORDER_INTENT_CREATED":
                raise RuntimeError("order ledger down")
            return real_append(**kwargs)

        with mock.patch.object(self.ledger, "append", side_effect=append):
            with self.assertRaisesRegex(RuntimeError, "order ledger down"):
                run_s01_vol_scan(
                    readiness_manager=self.manager,
                    state_store=self.state_store,
                    ledger=self.ledger,
                    broker=mock.Mock(),
                    current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
                    signal_context_provider=lambda: self.clean_input(S01_VOL_BASELINE),
                    ledger_reader=self.reader(),
                )
        self.assertEqual(self.state_store.state["order_intents"], {})

    def test_state_store_create_failure_propagates_after_order_intent_event(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        with mock.patch.object(
            self.state_store,
            "create_order_intent",
            side_effect=RuntimeError("state save failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "state save failed"):
                run_s01_vol_scan(
                    readiness_manager=self.manager,
                    state_store=self.state_store,
                    ledger=self.ledger,
                    broker=mock.Mock(),
                    current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
                    signal_context_provider=lambda: self.clean_input(S01_VOL_BASELINE),
                    ledger_reader=self.reader(),
                )
        self.assertIn("ORDER_INTENT_CREATED", self.order_path.read_text(encoding="utf-8"))
        self.assertNotIn("ORDER_SUBMITTED", self.order_path.read_text(encoding="utf-8"))


class SourceHygieneTests(unittest.TestCase):
    def test_signal_generated_detail_helper_is_not_duplicated_in_jobs_and_engine(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        jobs_source = (package_root / "jobs/vol.py").read_text(encoding="utf-8")
        engine_source = (package_root / "strategies/vol/engine.py").read_text(encoding="utf-8")
        signals_source = (package_root / "strategies/vol/signals.py").read_text(encoding="utf-8")
        self.assertNotIn("def _signal_generated_detail", jobs_source)
        self.assertNotIn("def _signal_generated_detail", engine_source)
        self.assertEqual(signals_source.count("def signal_generated_detail"), 1)
