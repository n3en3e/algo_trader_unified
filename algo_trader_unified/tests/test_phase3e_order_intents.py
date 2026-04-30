from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.risk import ORDER_INTENT_TTL_MINUTES
from algo_trader_unified.config.scheduler import JOB_S01_VOL_SCAN, JOB_S02_VOL_SCAN
from algo_trader_unified.core.ledger import KNOWN_EVENT_TYPES, LedgerAppender
from algo_trader_unified.core.ledger_reader import LedgerReader
from algo_trader_unified.core.order_intents import (
    OrderIntentTimestampError,
    cancel_order_intent,
    expire_order_intent,
    is_order_intent_stale,
)
from algo_trader_unified.core.readiness import ReadinessManager, ReadinessStatus
from algo_trader_unified.core.scheduler import UnifiedScheduler
from algo_trader_unified.core.skip_reasons import (
    SKIP_ACTIVE_ORDER_INTENT,
    SKIP_STALE_ORDER_INTENT,
)
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.jobs.vol import run_s01_vol_scan, run_s02_vol_scan
from algo_trader_unified.strategies.base import Phase2ARiskManagerStub
from algo_trader_unified.strategies.vol.engine import VolSellingEngine
from algo_trader_unified.config.variants import S01_CONFIG
from algo_trader_unified.config.variants import S02_CONFIG
from algo_trader_unified.strategies.vol.signals import (
    SignalResult,
    VolSignalInput,
    signal_generated_detail,
)


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

    def create_intent(
        self,
        strategy_id: str,
        intent_id: str | None = None,
        *,
        created_at: str = "2026-04-27T13:40:00+00:00",
        status: str = "created",
        dry_run: bool | None = None,
    ) -> dict:
        record = {
            "intent_id": intent_id or f"{strategy_id}:manual",
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "execution_mode": "paper_only",
            "status": status,
            "source_signal_event_id": None,
            "order_intent_created_event_id": "evt_manual",
            "order_ref": "same-order-ref",
            "created_at": created_at,
            "updated_at": created_at,
            "sizing_context": {},
            "risk_context": {},
            "signal_payload_snapshot": {},
        }
        if dry_run is not None:
            record["dry_run"] = dry_run
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
        self.assertTrue(self.state_store.get_order_intent("s01:intent")["dry_run"])

    def test_legacy_intent_without_dry_run_remains_readable(self) -> None:
        legacy = self.create_intent(S01_VOL_BASELINE, "legacy:intent")
        del legacy["dry_run"]
        self.state_store.state["order_intents"]["legacy:intent"] = legacy
        self.state_store.save()
        self.assertNotIn("dry_run", self.state_store.get_order_intent("legacy:intent"))
        self.assertEqual(
            self.state_store.get_active_order_intent(S01_VOL_BASELINE)["intent_id"],
            "legacy:intent",
        )
        self.assertEqual(self.state_store.list_order_intents()[0]["intent_id"], "legacy:intent")

    def test_expire_and_cancel_transitions_created_intents(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "s01:expire")
        expired = self.state_store.expire_order_intent(
            "s01:expire",
            expired_at="2026-04-27T14:20:00+00:00",
            expire_reason="ttl_expired",
            expired_event_id="evt_expired",
        )
        self.assertEqual(expired["status"], "expired")
        self.assertEqual(expired["expired_event_id"], "evt_expired")
        self.assertEqual(expired["updated_at"], "2026-04-27T14:20:00+00:00")
        self.assertIsNone(self.state_store.get_active_order_intent(S01_VOL_BASELINE))
        self.assertEqual(self.state_store.get_order_intent("s01:expire")["status"], "expired")

        self.create_intent(S02_VOL_ENHANCED, "s02:cancel")
        cancelled = self.state_store.cancel_order_intent(
            "s02:cancel",
            cancelled_at="2026-04-27T14:21:00+00:00",
            cancel_reason="operator_cancelled",
            cancelled_event_id="evt_cancelled",
        )
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(cancelled["cancelled_event_id"], "evt_cancelled")
        self.assertIsNone(self.state_store.get_active_order_intent(S02_VOL_ENHANCED))
        self.assertEqual(len(self.state_store.list_order_intents()), 2)

    def test_invalid_transitions_and_missing_intents_raise(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "s01:cancelled")
        self.state_store.cancel_order_intent(
            "s01:cancelled",
            cancelled_at="2026-04-27T14:21:00+00:00",
            cancel_reason="operator_cancelled",
            cancelled_event_id="evt_cancelled",
        )
        with self.assertRaisesRegex(ValueError, "not 'created'"):
            self.state_store.expire_order_intent(
                "s01:cancelled",
                expired_at="2026-04-27T14:22:00+00:00",
                expire_reason="ttl_expired",
                expired_event_id="evt_expired",
            )

        self.create_intent(S02_VOL_ENHANCED, "s02:expired")
        self.state_store.expire_order_intent(
            "s02:expired",
            expired_at="2026-04-27T14:22:00+00:00",
            expire_reason="ttl_expired",
            expired_event_id="evt_expired",
        )
        with self.assertRaisesRegex(ValueError, "not 'created'"):
            self.state_store.cancel_order_intent(
                "s02:expired",
                cancelled_at="2026-04-27T14:23:00+00:00",
                cancel_reason="operator_cancelled",
                cancelled_event_id="evt_cancelled",
            )
        with self.assertRaisesRegex(KeyError, "does not exist"):
            self.state_store.expire_order_intent(
                "missing",
                expired_at="2026-04-27T14:22:00+00:00",
                expire_reason="ttl_expired",
                expired_event_id="evt_expired",
            )
        with self.assertRaisesRegex(KeyError, "does not exist"):
            self.state_store.cancel_order_intent(
                "missing",
                cancelled_at="2026-04-27T14:23:00+00:00",
                cancel_reason="operator_cancelled",
                cancelled_event_id="evt_cancelled",
            )


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

    def test_expired_and_cancelled_are_known_and_route_to_order_ledger(self) -> None:
        self.assertIn("ORDER_INTENT_EXPIRED", KNOWN_EVENT_TYPES)
        self.assertIn("ORDER_INTENT_CANCELLED", KNOWN_EVENT_TYPES)
        expired_id = self.ledger.append(
            event_type="ORDER_INTENT_EXPIRED",
            strategy_id=S01_VOL_BASELINE,
            execution_mode="paper_only",
            source_module="test",
            payload={
                "intent_id": "intent",
                "strategy_id": S01_VOL_BASELINE,
                "sleeve_id": "VOL",
                "symbol": "XSP",
                "execution_mode": "paper_only",
                "prior_status": "created",
                "new_status": "expired",
                "expired_at": "2026-04-27T14:20:00+00:00",
                "expire_reason": "ttl_expired",
                "created_at": "2026-04-27T13:40:00+00:00",
                "age_minutes": 40.0,
                "event_detail": "ORDER_INTENT_EXPIRED",
            },
        )
        cancelled_id = self.ledger.append(
            event_type="ORDER_INTENT_CANCELLED",
            strategy_id=S02_VOL_ENHANCED,
            execution_mode="paper_only",
            source_module="test",
            payload={
                "intent_id": "intent",
                "strategy_id": S02_VOL_ENHANCED,
                "sleeve_id": "VOL",
                "symbol": "XSP",
                "execution_mode": "paper_only",
                "prior_status": "created",
                "new_status": "cancelled",
                "cancelled_at": "2026-04-27T14:20:00+00:00",
                "cancel_reason": "operator_cancelled",
                "created_at": "2026-04-27T13:40:00+00:00",
                "event_detail": "ORDER_INTENT_CANCELLED",
            },
        )
        order_text = self.order_path.read_text(encoding="utf-8")
        exec_text = self.execution_path.read_text(encoding="utf-8")
        self.assertIn(expired_id, order_text)
        self.assertIn(cancelled_id, order_text)
        self.assertNotIn(expired_id, exec_text)
        self.assertNotIn(cancelled_id, exec_text)
        self.assertNotIn("ORDER_SUBMITTED", order_text)


class EngineImmutabilityTests(TmpCase):
    def test_vol_selling_engine_frozen_true_incompatibility_is_documented(self) -> None:
        engine = VolSellingEngine(
            config=S01_CONFIG,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=mock.Mock(),
            risk_manager=Phase2ARiskManagerStub(),
        )
        with mock.patch.object(engine, "record_close") as record_close:
            record_close.assert_not_called()
        source = Path(VolSellingEngine.__module__.replace(".", "/") + ".py")
        engine_source = (Path(__file__).resolve().parents[2] / source).read_text(
            encoding="utf-8"
        )
        self.assertIn("FrozenInstanceError", engine_source)


class OrderIntentHelperTests(TmpCase):
    def test_expire_order_intent_writes_event_then_updates_state(self) -> None:
        original = self.create_intent(S01_VOL_BASELINE, "s01:stale")
        updated = expire_order_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            intent_id="s01:stale",
            expired_at="2026-04-27T14:20:00+00:00",
            expire_reason="ttl_expired",
        )
        self.assertEqual(updated["status"], "expired")
        self.assertEqual(updated["expire_reason"], "ttl_expired")
        event = self.events(self.order_path)[-1]
        self.assertEqual(event["event_type"], "ORDER_INTENT_EXPIRED")
        payload = event["payload"]
        self.assertEqual(payload["intent_id"], original["intent_id"])
        self.assertEqual(payload["strategy_id"], original["strategy_id"])
        self.assertEqual(payload["sleeve_id"], original["sleeve_id"])
        self.assertEqual(payload["symbol"], original["symbol"])
        self.assertEqual(payload["execution_mode"], original["execution_mode"])
        self.assertEqual(payload["prior_status"], "created")
        self.assertEqual(payload["new_status"], "expired")
        self.assertEqual(payload["expired_at"], "2026-04-27T14:20:00+00:00")
        self.assertEqual(payload["expire_reason"], "ttl_expired")
        self.assertEqual(payload["created_at"], original["created_at"])
        self.assertEqual(payload["age_minutes"], 40.0)
        self.assertEqual(payload["event_detail"], "ORDER_INTENT_EXPIRED")
        self.assertEqual(updated["expired_event_id"], event["event_id"])
        self.assertNotIn("ORDER_SUBMITTED", self.order_path.read_text(encoding="utf-8"))

    def test_cancel_order_intent_writes_event_then_updates_state(self) -> None:
        original = self.create_intent(S02_VOL_ENHANCED, "s02:cancel")
        updated = cancel_order_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            intent_id="s02:cancel",
            cancelled_at="2026-04-27T14:20:00+00:00",
            cancel_reason="operator_cancelled",
        )
        self.assertEqual(updated["status"], "cancelled")
        self.assertEqual(updated["cancel_reason"], "operator_cancelled")
        event = self.events(self.order_path)[-1]
        self.assertEqual(event["event_type"], "ORDER_INTENT_CANCELLED")
        payload = event["payload"]
        self.assertEqual(payload["intent_id"], original["intent_id"])
        self.assertEqual(payload["strategy_id"], original["strategy_id"])
        self.assertEqual(payload["sleeve_id"], original["sleeve_id"])
        self.assertEqual(payload["symbol"], original["symbol"])
        self.assertEqual(payload["execution_mode"], original["execution_mode"])
        self.assertEqual(payload["prior_status"], "created")
        self.assertEqual(payload["new_status"], "cancelled")
        self.assertEqual(payload["cancelled_at"], "2026-04-27T14:20:00+00:00")
        self.assertEqual(payload["cancel_reason"], "operator_cancelled")
        self.assertEqual(payload["created_at"], original["created_at"])
        self.assertEqual(payload["event_detail"], "ORDER_INTENT_CANCELLED")
        self.assertEqual(updated["cancelled_event_id"], event["event_id"])
        self.assertIsNone(self.state_store.get_active_order_intent(S02_VOL_ENHANCED))
        self.assertNotIn("ORDER_SUBMITTED", self.order_path.read_text(encoding="utf-8"))

    def test_helper_ledger_append_failure_leaves_state_unchanged(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "s01:stale")
        self.create_intent(S02_VOL_ENHANCED, "s02:cancel")
        with mock.patch.object(self.ledger, "append", side_effect=RuntimeError("ledger down")):
            with self.assertRaisesRegex(RuntimeError, "ledger down"):
                expire_order_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    intent_id="s01:stale",
                    expired_at="2026-04-27T14:20:00+00:00",
                    expire_reason="ttl_expired",
                )
            with self.assertRaisesRegex(RuntimeError, "ledger down"):
                cancel_order_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    intent_id="s02:cancel",
                    cancelled_at="2026-04-27T14:20:00+00:00",
                    cancel_reason="operator_cancelled",
                )
        self.assertEqual(self.state_store.get_order_intent("s01:stale")["status"], "created")
        self.assertEqual(self.state_store.get_order_intent("s02:cancel")["status"], "created")

    def test_helper_state_failure_after_ledger_append_propagates(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "s01:stale")
        with mock.patch.object(
            self.state_store,
            "expire_order_intent",
            side_effect=RuntimeError("state failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "state failed"):
                expire_order_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    intent_id="s01:stale",
                    expired_at="2026-04-27T14:20:00+00:00",
                    expire_reason="ttl_expired",
                )
        self.assertIn("ORDER_INTENT_EXPIRED", self.order_path.read_text(encoding="utf-8"))

        self.create_intent(S02_VOL_ENHANCED, "s02:cancel")
        with mock.patch.object(
            self.state_store,
            "cancel_order_intent",
            side_effect=RuntimeError("state failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "state failed"):
                cancel_order_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    intent_id="s02:cancel",
                    cancelled_at="2026-04-27T14:20:00+00:00",
                    cancel_reason="operator_cancelled",
                )
        self.assertIn("ORDER_INTENT_CANCELLED", self.order_path.read_text(encoding="utf-8"))

    def test_cancel_order_intent_rejects_non_created_intents(self) -> None:
        self.create_intent(S02_VOL_ENHANCED, "s02:expired")
        expire_order_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            intent_id="s02:expired",
            expired_at="2026-04-27T14:20:00+00:00",
            expire_reason="ttl_expired",
        )
        with self.assertRaisesRegex(ValueError, "not 'created'"):
            cancel_order_intent(
                state_store=self.state_store,
                ledger=self.ledger,
                intent_id="s02:expired",
                cancelled_at="2026-04-27T14:21:00+00:00",
                cancel_reason="operator_cancelled",
            )


class OrderIntentTtlTests(unittest.TestCase):
    def test_ttl_boundary_is_strict(self) -> None:
        now = datetime(2026, 4, 27, 14, 10, tzinfo=timezone.utc)
        self.assertFalse(is_order_intent_stale(
            {"created_at": (now - timedelta(minutes=29)).isoformat()},
            now=now,
            ttl_minutes=30,
        ))
        self.assertFalse(is_order_intent_stale(
            {"created_at": (now - timedelta(minutes=30)).isoformat()},
            now=now,
            ttl_minutes=30,
        ))
        self.assertTrue(is_order_intent_stale(
            {"created_at": (now - timedelta(minutes=31)).isoformat()},
            now=now,
            ttl_minutes=30,
        ))

    def test_timezone_aware_and_naive_created_at(self) -> None:
        now = datetime(2026, 4, 27, 14, 40, tzinfo=timezone.utc)
        self.assertTrue(is_order_intent_stale(
            {"created_at": "2026-04-27T10:00:00-04:00"},
            now=now,
            ttl_minutes=30,
        ))
        self.assertTrue(is_order_intent_stale(
            {"created_at": "2026-04-27T10:00:00"},
            now=now,
            ttl_minutes=30,
        ))

    def test_missing_or_invalid_created_at_raises(self) -> None:
        with self.assertRaises(OrderIntentTimestampError):
            is_order_intent_stale({}, now=datetime.now(timezone.utc), ttl_minutes=30)
        with self.assertRaises(OrderIntentTimestampError):
            is_order_intent_stale(
                {"created_at": "not-a-time"},
                now=datetime.now(timezone.utc),
                ttl_minutes=30,
            )


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

    def test_s02_active_intent_precheck_skips_before_provider_and_engine(self) -> None:
        self.set_readiness(S02_VOL_ENHANCED)
        self.create_intent(S02_VOL_ENHANCED)
        provider = mock.Mock(return_value=self.clean_input(S02_VOL_ENHANCED))
        with mock.patch("algo_trader_unified.jobs.vol.VolSellingEngine") as engine_class:
            result = run_s02_vol_scan(
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
        self.assertEqual(events[-1]["payload"]["skip_reason"], SKIP_ACTIVE_ORDER_INTENT)
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")

    def test_final_recheck_blocks_concurrent_active_intent(self) -> None:
        self.set_readiness(S02_VOL_ENHANCED)
        active = {
            "intent_id": "s02:concurrent",
            "strategy_id": S02_VOL_ENHANCED,
            "status": "created",
            "created_at": "2026-04-27T13:40:00+00:00",
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
        self.assertEqual(
            self.events(self.execution_path)[-1]["payload"]["skip_reason"],
            SKIP_ACTIVE_ORDER_INTENT,
        )
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")

    def test_s01_final_recheck_blocks_concurrent_active_intent(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        active = {
            "intent_id": "s01:concurrent",
            "strategy_id": S01_VOL_BASELINE,
            "status": "created",
            "created_at": "2026-04-27T13:40:00+00:00",
        }
        with mock.patch.object(
            self.state_store,
            "get_active_order_intent",
            side_effect=[None, active],
        ):
            result = run_s01_vol_scan(
                readiness_manager=self.manager,
                state_store=self.state_store,
                ledger=self.ledger,
                broker=mock.Mock(),
                current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
                signal_context_provider=lambda: self.clean_input(S01_VOL_BASELINE),
                ledger_reader=self.reader(),
            )
        self.assertEqual(result.detail, "active_order_intent")
        self.assertEqual([event["event_type"] for event in self.events(self.execution_path)], [
            "SIGNAL_GENERATED",
            "SIGNAL_SKIPPED",
        ])
        self.assertEqual(
            self.events(self.execution_path)[-1]["payload"]["skip_reason"],
            SKIP_ACTIVE_ORDER_INTENT,
        )
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

    def test_s01_stale_intent_expires_skips_and_stops(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        broker = mock.Mock()
        self.create_intent(
            S01_VOL_BASELINE,
            "s01:stale",
            created_at="2026-04-27T13:00:00+00:00",
        )
        provider = mock.Mock(return_value=self.clean_input(S01_VOL_BASELINE))
        with mock.patch("algo_trader_unified.jobs.vol.VolSellingEngine") as engine_class:
            result = run_s01_vol_scan(
                readiness_manager=self.manager,
                state_store=self.state_store,
                ledger=self.ledger,
                broker=broker,
                current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
                signal_context_provider=provider,
                ledger_reader=self.reader(),
            )
        self.assertEqual(result.detail, "stale_order_intent_expired")
        self.assertEqual(self.state_store.get_order_intent("s01:stale")["status"], "expired")
        order_events = self.events(self.order_path)
        exec_events = self.events(self.execution_path)
        self.assertEqual([event["event_type"] for event in order_events], ["ORDER_INTENT_EXPIRED"])
        self.assertEqual(exec_events[-1]["event_type"], "SIGNAL_SKIPPED")
        self.assertEqual(exec_events[-1]["payload"]["skip_reason"], SKIP_STALE_ORDER_INTENT)
        self.assertEqual(exec_events[-1]["payload"]["ttl_minutes"], ORDER_INTENT_TTL_MINUTES)
        self.assertEqual(exec_events[-1]["payload"]["expired_event_id"], order_events[0]["event_id"])
        provider.assert_not_called()
        engine_class.assert_not_called()
        broker.submit_order.assert_not_called()
        self.assertNotIn("SIGNAL_GENERATED", self.execution_path.read_text(encoding="utf-8"))
        self.assertNotIn("ORDER_INTENT_CREATED", self.order_path.read_text(encoding="utf-8"))
        self.assertNotIn("ORDER_SUBMITTED", self.order_path.read_text(encoding="utf-8"))

    def test_s02_stale_intent_expires_skips_and_stops(self) -> None:
        self.set_readiness(S02_VOL_ENHANCED)
        broker = mock.Mock()
        self.create_intent(
            S02_VOL_ENHANCED,
            "s02:stale",
            created_at="2026-04-27T13:00:00+00:00",
        )
        provider = mock.Mock(return_value=self.clean_input(S02_VOL_ENHANCED))
        result = run_s02_vol_scan(
            readiness_manager=self.manager,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=broker,
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=provider,
            ledger_reader=self.reader(),
        )
        self.assertEqual(result.detail, "stale_order_intent_expired")
        self.assertEqual(self.state_store.get_order_intent("s02:stale")["status"], "expired")
        self.assertEqual(len(self.state_store.list_order_intents(S02_VOL_ENHANCED)), 1)
        self.assertEqual(self.events(self.order_path)[0]["event_type"], "ORDER_INTENT_EXPIRED")
        provider.assert_not_called()
        broker.submit_order.assert_not_called()
        self.assertNotIn("SIGNAL_GENERATED", self.execution_path.read_text(encoding="utf-8"))
        self.assertNotIn("ORDER_INTENT_CREATED", self.order_path.read_text(encoding="utf-8"))
        self.assertNotIn("ORDER_SUBMITTED", self.order_path.read_text(encoding="utf-8"))

    def test_non_stale_intent_still_blocks_without_expiring(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        self.create_intent(
            S01_VOL_BASELINE,
            "s01:active",
            created_at="2026-04-27T13:10:00+00:00",
        )
        result = run_s01_vol_scan(
            readiness_manager=self.manager,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=mock.Mock(),
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=mock.Mock(return_value=self.clean_input(S01_VOL_BASELINE)),
            ledger_reader=self.reader(),
        )
        self.assertEqual(result.detail, "active_order_intent")
        self.assertEqual(self.state_store.get_order_intent("s01:active")["status"], "created")
        self.assertNotIn("ORDER_INTENT_EXPIRED", self.order_path.read_text(encoding="utf-8"))
        self.assertEqual(
            self.events(self.execution_path)[-1]["payload"]["skip_reason"],
            SKIP_ACTIVE_ORDER_INTENT,
        )

    def test_expired_and_cancelled_intents_do_not_block_future_scans(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        self.set_readiness(S02_VOL_ENHANCED)
        self.create_intent(S01_VOL_BASELINE, "s01:expired", status="expired")
        self.create_intent(S02_VOL_ENHANCED, "s02:cancelled", status="cancelled")
        s01 = run_s01_vol_scan(
            readiness_manager=self.manager,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=mock.Mock(),
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: self.clean_input(S01_VOL_BASELINE, "fresh-s01"),
            ledger_reader=self.reader(),
        )
        s02 = run_s02_vol_scan(
            readiness_manager=self.manager,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=mock.Mock(),
            current_time=datetime(2026, 4, 28, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: self.clean_input(S02_VOL_ENHANCED, "fresh-s02"),
            ledger_reader=self.reader(),
        )
        self.assertEqual(s01.detail, "order_intent_created")
        self.assertEqual(s02.detail, "order_intent_created")
        statuses = {intent["status"] for intent in self.state_store.list_order_intents()}
        self.assertIn("expired", statuses)
        self.assertIn("cancelled", statuses)
        self.assertIn("created", statuses)

    def test_stale_intents_are_strategy_isolated(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        self.set_readiness(S02_VOL_ENHANCED)
        self.create_intent(
            S01_VOL_BASELINE,
            "s01:stale",
            created_at="2026-04-27T13:00:00+00:00",
        )
        s02 = run_s02_vol_scan(
            readiness_manager=self.manager,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=mock.Mock(),
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: self.clean_input(S02_VOL_ENHANCED),
            ledger_reader=self.reader(),
        )
        self.assertEqual(s02.detail, "order_intent_created")
        self.assertEqual(self.state_store.get_order_intent("s01:stale")["status"], "created")

        self.state_store.state["order_intents"] = {}
        self.state_store.save()
        self.create_intent(
            S02_VOL_ENHANCED,
            "s02:stale",
            created_at="2026-04-27T13:00:00+00:00",
        )
        s01 = run_s01_vol_scan(
            readiness_manager=self.manager,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=mock.Mock(),
            current_time=datetime(2026, 4, 28, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: self.clean_input(S01_VOL_BASELINE),
            ledger_reader=self.reader(),
        )
        self.assertEqual(s01.detail, "order_intent_created")
        self.assertEqual(self.state_store.get_order_intent("s02:stale")["status"], "created")


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

    def test_stale_detect_expire_and_skip_happen_while_strategy_lock_is_held(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        self.create_intent(
            S01_VOL_BASELINE,
            "s01:stale",
            created_at="2026-04-27T13:00:00+00:00",
        )
        spy = LockSpy()
        self.state_store.strategy_state_locks[S01_VOL_BASELINE] = spy
        provider = mock.Mock(return_value=self.clean_input(S01_VOL_BASELINE))
        real_append = self.ledger.append
        observed = {"expired": False, "skipped": False}

        def append(**kwargs):
            if kwargs["event_type"] == "ORDER_INTENT_EXPIRED":
                observed["expired"] = True
                self.assertTrue(spy.held)
            if kwargs["event_type"] == "SIGNAL_SKIPPED":
                observed["skipped"] = True
                self.assertTrue(spy.held)
                self.assertTrue(observed["expired"])
            return real_append(**kwargs)

        with mock.patch.object(self.ledger, "append", side_effect=append):
            result = run_s01_vol_scan(
                readiness_manager=self.manager,
                state_store=self.state_store,
                ledger=self.ledger,
                broker=mock.Mock(),
                current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
                signal_context_provider=provider,
                ledger_reader=self.reader(),
            )
        self.assertEqual(result.detail, "stale_order_intent_expired")
        self.assertEqual(observed, {"expired": True, "skipped": True})
        provider.assert_not_called()


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

    def test_run_job_once_expires_stale_s01_and_s02_intents_without_signal_eval(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        self.set_readiness(S02_VOL_ENHANCED)
        self.create_intent(
            S01_VOL_BASELINE,
            "s01:stale",
            created_at="2026-04-27T13:00:00+00:00",
        )
        self.create_intent(
            S02_VOL_ENHANCED,
            "s02:stale",
            created_at="2026-04-28T13:00:00+00:00",
        )
        scheduler = UnifiedScheduler(
            state_store=self.state_store,
            ledger=self.ledger,
            readiness_manager=self.manager,
        )
        s01_provider = mock.Mock(return_value=self.clean_input(S01_VOL_BASELINE))
        s02_provider = mock.Mock(return_value=self.clean_input(S02_VOL_ENHANCED))
        s01 = scheduler.run_job_once(
            JOB_S01_VOL_SCAN,
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=s01_provider,
            ledger_reader=self.reader(),
            broker=mock.Mock(),
        )
        s02 = scheduler.run_job_once(
            JOB_S02_VOL_SCAN,
            current_time=datetime(2026, 4, 28, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=s02_provider,
            ledger_reader=self.reader(),
            broker=mock.Mock(),
        )
        self.assertEqual(s01.detail, "stale_order_intent_expired")
        self.assertEqual(s02.detail, "stale_order_intent_expired")
        self.assertEqual(self.state_store.get_order_intent("s01:stale")["status"], "expired")
        self.assertEqual(self.state_store.get_order_intent("s02:stale")["status"], "expired")
        s01_provider.assert_not_called()
        s02_provider.assert_not_called()
        self.assertNotIn("SIGNAL_GENERATED", self.execution_path.read_text(encoding="utf-8"))
        self.assertNotIn("ORDER_INTENT_CREATED", self.order_path.read_text(encoding="utf-8"))


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
        self.assertEqual(signal_generated_detail(S01_CONFIG), "S01_VOL_SIGNAL_GENERATED")
        self.assertEqual(signal_generated_detail(S02_CONFIG), "S02_VOL_SIGNAL_GENERATED")

    def test_stale_skip_reason_is_imported_not_hardcoded_in_jobs(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        jobs_source = (package_root / "jobs/vol.py").read_text(encoding="utf-8")
        self.assertEqual(SKIP_STALE_ORDER_INTENT, "SKIP_STALE_ORDER_INTENT")
        self.assertIn("SKIP_STALE_ORDER_INTENT", jobs_source)
        self.assertNotIn('"SKIP_STALE_ORDER_INTENT"', jobs_source)
        self.assertNotIn("'SKIP_STALE_ORDER_INTENT'", jobs_source)
