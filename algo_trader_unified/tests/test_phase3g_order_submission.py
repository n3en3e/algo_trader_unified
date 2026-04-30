from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.execution import DryRunExecutionAdapter
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.ledger_reader import LedgerReader
from algo_trader_unified.core.order_intents import submit_order_intent
from algo_trader_unified.core.readiness import ReadinessManager, ReadinessStatus
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.jobs.vol import run_s01_vol_scan, run_s02_vol_scan
from algo_trader_unified.strategies.vol.signals import VolSignalInput


SUBMITTED_AT = "2026-04-27T14:00:00+00:00"
FORBIDDEN_ORDER_ID_FIELD = "broker" + "_order_id"


class Phase3GCase(unittest.TestCase):
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

    def create_intent(
        self,
        strategy_id: str = S01_VOL_BASELINE,
        intent_id: str | None = None,
        *,
        status: str = "created",
        dry_run: bool | None = True,
    ) -> dict:
        record = {
            "intent_id": intent_id or f"{strategy_id}:intent",
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "execution_mode": "paper_only",
            "status": status,
            "source_signal_event_id": "evt_signal",
            "order_intent_created_event_id": "evt_created",
            "order_ref": f"{strategy_id}|P0427XSP|OPEN",
            "created_at": "2026-04-27T13:40:00+00:00",
            "updated_at": "2026-04-27T13:40:00+00:00",
            "sizing_context": {},
            "risk_context": {},
            "signal_payload_snapshot": {},
        }
        if dry_run is not None:
            record["dry_run"] = dry_run
        return self.state_store.create_order_intent(record)

    def order_events(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.order_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

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

    def clean_input(self, strategy_id: str) -> VolSignalInput:
        return VolSignalInput(
            symbol="XSP",
            current_date=date(2026, 4, 27),
            vix=18.0,
            iv_rank=45.0,
            target_dte=45,
            blackout_dates=(),
            order_ref_candidate=f"{strategy_id}|P0427XSP|OPEN",
        )

    def reader(self) -> LedgerReader:
        return LedgerReader(
            execution_ledger_path=self.execution_path,
            order_ledger_path=self.order_path,
        )


class StateStoreSubmitTransitionTests(Phase3GCase):
    def test_submit_order_intent_transitions_created_to_submitted(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "s01:submit")
        submitted = self.state_store.submit_order_intent(
            "s01:submit",
            submitted_at=SUBMITTED_AT,
            order_submitted_event_id="evt_submitted",
            simulated_order_id="sim_123",
        )
        self.assertEqual(submitted["status"], "submitted")
        self.assertEqual(submitted["submitted_at"], SUBMITTED_AT)
        self.assertEqual(submitted["updated_at"], SUBMITTED_AT)
        self.assertEqual(submitted["order_submitted_event_id"], "evt_submitted")
        self.assertEqual(submitted["simulated_order_id"], "sim_123")
        self.assertTrue(submitted["dry_run"])
        self.assertNotIn(FORBIDDEN_ORDER_ID_FIELD, submitted)
        self.assertIsNone(self.state_store.get_active_order_intent(S01_VOL_BASELINE))

    def test_submit_order_intent_rejects_missing_and_non_created(self) -> None:
        with self.assertRaisesRegex(KeyError, "does not exist"):
            self.state_store.submit_order_intent(
                "missing",
                submitted_at=SUBMITTED_AT,
                order_submitted_event_id="evt_submitted",
                simulated_order_id="sim_missing",
            )
        for status in ("expired", "cancelled", "submitted"):
            intent_id = f"s01:{status}"
            self.create_intent(S01_VOL_BASELINE, intent_id)
            if status == "expired":
                self.state_store.expire_order_intent(
                    intent_id,
                    expired_at=SUBMITTED_AT,
                    expire_reason="ttl_expired",
                    expired_event_id="evt_expired",
                )
            elif status == "cancelled":
                self.state_store.cancel_order_intent(
                    intent_id,
                    cancelled_at=SUBMITTED_AT,
                    cancel_reason="operator_cancelled",
                    cancelled_event_id="evt_cancelled",
                )
            else:
                self.state_store.submit_order_intent(
                    intent_id,
                    submitted_at=SUBMITTED_AT,
                    order_submitted_event_id="evt_submitted",
                    simulated_order_id="sim_once",
                )
            with self.assertRaisesRegex(ValueError, "not 'created'"):
                self.state_store.submit_order_intent(
                    intent_id,
                    submitted_at=SUBMITTED_AT,
                    order_submitted_event_id="evt_submitted_again",
                    simulated_order_id="sim_twice",
                )

    def test_create_order_intent_populates_dry_run_for_s01_and_s02(self) -> None:
        s01 = self.create_intent(S01_VOL_BASELINE, "s01:dry")
        s02 = self.create_intent(S02_VOL_ENHANCED, "s02:dry")
        self.assertTrue(s01["dry_run"])
        self.assertTrue(s02["dry_run"])


class DryRunExecutionAdapterTests(Phase3GCase):
    def test_submit_order_intent_returns_pure_dry_run_result(self) -> None:
        intent = self.create_intent(S01_VOL_BASELINE, "s01:adapter")
        state_before = deepcopy(self.state_store.state)
        adapter = DryRunExecutionAdapter()
        result = adapter.submit_order_intent(intent, submitted_at=SUBMITTED_AT)
        self.assertTrue(result["dry_run"])
        self.assertIn("simulated_order_id", result)
        self.assertNotIn(FORBIDDEN_ORDER_ID_FIELD, result)
        self.assertEqual(result["intent_id"], intent["intent_id"])
        self.assertEqual(result["strategy_id"], intent["strategy_id"])
        self.assertEqual(result["symbol"], intent["symbol"])
        self.assertEqual(result["order_ref"], intent["order_ref"])
        self.assertEqual(result["action"], "open")
        self.assertEqual(result["status"], "submitted")
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")
        self.assertEqual(self.state_store.state, state_before)


class SubmitOrderIntentHelperTests(Phase3GCase):
    def assert_successful_submit(self, strategy_id: str, intent_id: str) -> dict:
        self.create_intent(strategy_id, intent_id)
        updated = submit_order_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            intent_id=intent_id,
            submitted_at=SUBMITTED_AT,
        )
        events = self.order_events()
        self.assertEqual([event["event_type"] for event in events], ["ORDER_SUBMITTED"])
        payload = events[0]["payload"]
        for field in (
            "intent_id",
            "strategy_id",
            "sleeve_id",
            "symbol",
            "execution_mode",
            "order_ref",
            "source_signal_event_id",
            "order_intent_created_event_id",
            "submitted_at",
            "dry_run",
            "simulated_order_id",
            "action",
            "event_detail",
        ):
            self.assertIn(field, payload)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["event_detail"], "ORDER_SUBMITTED")
        self.assertEqual(payload["action"], "open")
        self.assertNotIn(FORBIDDEN_ORDER_ID_FIELD, payload)
        self.assertEqual(updated["order_submitted_event_id"], events[0]["event_id"])
        self.assertEqual(updated["simulated_order_id"], payload["simulated_order_id"])
        self.assertNotIn(FORBIDDEN_ORDER_ID_FIELD, updated)
        self.assertIsNone(self.state_store.get_active_order_intent(strategy_id))
        ledger_text = self.order_path.read_text(encoding="utf-8")
        self.assertNotIn("ORDER_CONFIRMED", ledger_text)
        self.assertNotIn("FILL_CONFIRMED", ledger_text)
        self.assertNotIn("POSITION_", ledger_text)
        return updated

    def test_created_s01_intent_submits_successfully(self) -> None:
        self.assert_successful_submit(S01_VOL_BASELINE, "s01:helper")

    def test_created_s02_intent_submits_successfully(self) -> None:
        self.assert_successful_submit(S02_VOL_ENHANCED, "s02:helper")

    def test_missing_dry_run_raises_clear_error(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "s01:legacy")
        legacy = self.state_store.state["order_intents"]["s01:legacy"]
        del legacy["dry_run"]
        self.state_store.save()
        with self.assertRaisesRegex(ValueError, "intent.dry_run"):
            submit_order_intent(
                state_store=self.state_store,
                ledger=self.ledger,
                execution_adapter=DryRunExecutionAdapter(),
                intent_id="s01:legacy",
                submitted_at=SUBMITTED_AT,
            )
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")
        self.assertEqual(
            self.state_store.get_order_intent("s01:legacy")["status"],
            "created",
        )

    def test_no_broker_calls(self) -> None:
        broker = mock.Mock()
        broker.submit_order = mock.Mock()
        broker.placeOrder = mock.Mock()
        broker.cancelOrder = mock.Mock()
        self.create_intent(S01_VOL_BASELINE, "s01:no-broker")
        submit_order_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            intent_id="s01:no-broker",
            submitted_at=SUBMITTED_AT,
        )
        broker.submit_order.assert_not_called()
        broker.placeOrder.assert_not_called()
        broker.cancelOrder.assert_not_called()


class SubmitOrderIntentFailureOrderTests(Phase3GCase):
    def test_adapter_failure_writes_nothing_and_leaves_state_unchanged(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "s01:adapter-fail")
        before = deepcopy(self.state_store.state)
        adapter = mock.Mock()
        adapter.submit_order_intent.side_effect = RuntimeError("adapter failed")
        with self.assertRaisesRegex(RuntimeError, "adapter failed"):
            submit_order_intent(
                state_store=self.state_store,
                ledger=self.ledger,
                execution_adapter=adapter,
                intent_id="s01:adapter-fail",
                submitted_at=SUBMITTED_AT,
            )
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")
        self.assertEqual(self.state_store.state, before)

    def test_ledger_failure_leaves_state_unchanged(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "s01:ledger-fail")
        before = deepcopy(self.state_store.state)
        ledger = mock.Mock()
        ledger.append.side_effect = RuntimeError("ledger failed")
        with self.assertRaisesRegex(RuntimeError, "ledger failed"):
            submit_order_intent(
                state_store=self.state_store,
                ledger=ledger,
                execution_adapter=DryRunExecutionAdapter(),
                intent_id="s01:ledger-fail",
                submitted_at=SUBMITTED_AT,
            )
        self.assertEqual(self.state_store.state, before)

    def test_state_store_failure_propagates_after_order_submitted(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "s01:state-fail")
        with mock.patch.object(
            self.state_store,
            "submit_order_intent",
            side_effect=RuntimeError("state failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "state failed"):
                submit_order_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    execution_adapter=DryRunExecutionAdapter(),
                    intent_id="s01:state-fail",
                    submitted_at=SUBMITTED_AT,
                )
        text = self.order_path.read_text(encoding="utf-8")
        self.assertIn("ORDER_SUBMITTED", text)
        self.assertNotIn("ORDER_CONFIRMED", text)
        self.assertNotIn("FILL_CONFIRMED", text)


class ScanPathRegressionTests(Phase3GCase):
    def test_s01_scan_creates_intent_only_and_does_not_submit(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        broker = mock.Mock()
        result = run_s01_vol_scan(
            readiness_manager=self.manager,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=broker,
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: self.clean_input(S01_VOL_BASELINE),
            ledger_reader=self.reader(),
        )
        self.assertEqual(result.detail, "order_intent_created")
        self.assertEqual(
            [event["event_type"] for event in self.order_events()],
            ["ORDER_INTENT_CREATED"],
        )
        broker.submit_order.assert_not_called()

    def test_s02_scan_creates_intent_only_and_does_not_submit(self) -> None:
        self.set_readiness(S02_VOL_ENHANCED)
        broker = mock.Mock()
        result = run_s02_vol_scan(
            readiness_manager=self.manager,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=broker,
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: self.clean_input(S02_VOL_ENHANCED),
            ledger_reader=self.reader(),
        )
        self.assertEqual(result.detail, "order_intent_created")
        self.assertNotIn("ORDER_SUBMITTED", self.order_path.read_text(encoding="utf-8"))
        broker.submit_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
