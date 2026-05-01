from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import json
import tempfile
import unittest
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.execution import DryRunExecutionAdapter
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.ledger_reader import LedgerReader
from algo_trader_unified.core.order_intents import (
    confirm_fill,
    confirm_order_intent,
    submit_order_intent,
)
from algo_trader_unified.core.readiness import ReadinessManager, ReadinessStatus
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.jobs.vol import run_s01_vol_scan
from algo_trader_unified.strategies.vol.signals import VolSignalInput
from algo_trader_unified.tools import confirm_fill as fill_tool
from algo_trader_unified.tools import confirm_order_intent as confirm_tool
from algo_trader_unified.tools import submit_order_intent as submit_tool


SUBMITTED_AT = "2026-04-27T14:00:00+00:00"
CONFIRMED_AT = "2026-04-27T14:05:00+00:00"
FILLED_AT = "2026-04-27T14:06:00+00:00"
FORBIDDEN_ORDER_ID_FIELD = "broker" + "_order_id"


def assert_numeric_price(testcase: unittest.TestCase, value) -> None:
    testcase.assertIsInstance(value, (float, Decimal))
    testcase.assertNotIsInstance(value, str)
    testcase.assertGreaterEqual(value, 0)


def assert_numeric_quantity(testcase: unittest.TestCase, value) -> None:
    testcase.assertIsInstance(value, (int, float, Decimal))
    testcase.assertNotIsInstance(value, (bool, str))
    testcase.assertGreater(value, 0)


class Phase3JCase(unittest.TestCase):
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
        dry_run: bool | None = True,
    ) -> dict:
        record = {
            "intent_id": intent_id or f"{strategy_id}:intent",
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "execution_mode": "paper_only",
            "status": "created",
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

    def create_submitted_intent(self, strategy_id: str, intent_id: str) -> dict:
        created = self.create_intent(strategy_id, intent_id)
        return self.state_store.submit_order_intent(
            created["intent_id"],
            submitted_at=SUBMITTED_AT,
            order_submitted_event_id="evt_submitted",
            simulated_order_id=f"sim_{intent_id.replace(':', '_')}",
        )

    def create_confirmed_intent(self, strategy_id: str, intent_id: str) -> dict:
        submitted = self.create_submitted_intent(strategy_id, intent_id)
        return self.state_store.confirm_order_intent(
            submitted["intent_id"],
            confirmed_at=CONFIRMED_AT,
            order_confirmed_event_id="evt_confirmed",
            simulated_order_id=submitted["simulated_order_id"],
        )

    def submit_with_helper(self, strategy_id: str, intent_id: str) -> dict:
        self.create_intent(strategy_id, intent_id)
        return submit_order_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            intent_id=intent_id,
            submitted_at=SUBMITTED_AT,
        )

    def confirm_with_helper(self, strategy_id: str, intent_id: str) -> dict:
        self.submit_with_helper(strategy_id, intent_id)
        return confirm_order_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            intent_id=intent_id,
            confirmed_at=CONFIRMED_AT,
        )

    def order_events(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.order_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def run_fill_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = fill_tool.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def load_state_store(self) -> StateStore:
        return StateStore(self.root / "data/state/portfolio_state.json")

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


class StateStoreFillTransitionTests(Phase3JCase):
    def test_fill_order_intent_transitions_confirmed_to_filled(self) -> None:
        self.create_confirmed_intent(S01_VOL_BASELINE, "s01:fill")
        filled = self.state_store.fill_order_intent(
            "s01:fill",
            filled_at=FILLED_AT,
            fill_confirmed_event_id="evt_fill",
            simulated_order_id="sim_123",
            fill_id="fill_123",
            fill_price=1.25,
            fill_quantity=1,
        )
        self.assertEqual(filled["status"], "filled")
        self.assertEqual(filled["filled_at"], FILLED_AT)
        self.assertEqual(filled["updated_at"], FILLED_AT)
        self.assertEqual(filled["fill_confirmed_event_id"], "evt_fill")
        self.assertEqual(filled["simulated_order_id"], "sim_123")
        self.assertEqual(filled["fill_id"], "fill_123")
        assert_numeric_price(self, filled["fill_price"])
        assert_numeric_quantity(self, filled["fill_quantity"])
        self.assertTrue(filled["dry_run"])
        self.assertNotIn(FORBIDDEN_ORDER_ID_FIELD, filled)
        self.assertIsNone(self.state_store.get_active_order_intent(S01_VOL_BASELINE))

    def test_fill_order_intent_rejects_missing_non_confirmed_and_invalid_fills(self) -> None:
        with self.assertRaisesRegex(KeyError, "does not exist"):
            self.state_store.fill_order_intent(
                "missing",
                filled_at=FILLED_AT,
                fill_confirmed_event_id="evt_fill",
                simulated_order_id="sim_missing",
                fill_id="fill_missing",
                fill_price=1.25,
                fill_quantity=1,
            )
        self.create_intent(S01_VOL_BASELINE, "s01:created")
        self.create_submitted_intent(S01_VOL_BASELINE, "s01:submitted")
        self.create_intent(S01_VOL_BASELINE, "s01:expired")
        self.state_store.expire_order_intent(
            "s01:expired",
            expired_at=FILLED_AT,
            expire_reason="ttl_expired",
            expired_event_id="evt_expired",
        )
        self.create_intent(S01_VOL_BASELINE, "s01:cancelled")
        self.state_store.cancel_order_intent(
            "s01:cancelled",
            cancelled_at=FILLED_AT,
            cancel_reason="operator_cancelled",
            cancelled_event_id="evt_cancelled",
        )
        self.create_confirmed_intent(S01_VOL_BASELINE, "s01:already")
        self.state_store.fill_order_intent(
            "s01:already",
            filled_at=FILLED_AT,
            fill_confirmed_event_id="evt_fill",
            simulated_order_id="sim_already",
            fill_id="fill_already",
            fill_price=1.25,
            fill_quantity=1,
        )
        for intent_id in ("s01:created", "s01:submitted", "s01:expired", "s01:cancelled", "s01:already"):
            with self.assertRaisesRegex(ValueError, "not 'confirmed'"):
                self.state_store.fill_order_intent(
                    intent_id,
                    filled_at=FILLED_AT,
                    fill_confirmed_event_id="evt_fill_again",
                    simulated_order_id="sim_again",
                    fill_id="fill_again",
                    fill_price=1.25,
                    fill_quantity=1,
                )
        self.create_confirmed_intent(S01_VOL_BASELINE, "s01:bad-price")
        with self.assertRaisesRegex(ValueError, "fill_price"):
            self.state_store.fill_order_intent(
                "s01:bad-price",
                filled_at=FILLED_AT,
                fill_confirmed_event_id="evt_fill",
                simulated_order_id="sim_bad",
                fill_id="fill_bad",
                fill_price="1.25",
                fill_quantity=1,
            )
        self.create_confirmed_intent(S01_VOL_BASELINE, "s01:bad-qty")
        with self.assertRaisesRegex(ValueError, "fill_quantity"):
            self.state_store.fill_order_intent(
                "s01:bad-qty",
                filled_at=FILLED_AT,
                fill_confirmed_event_id="evt_fill",
                simulated_order_id="sim_bad",
                fill_id="fill_bad",
                fill_price=1.25,
                fill_quantity=0,
            )


class DryRunExecutionAdapterFillTests(Phase3JCase):
    def test_check_for_fills_returns_pure_filled_result(self) -> None:
        intent = self.create_confirmed_intent(S01_VOL_BASELINE, "s01:adapter")
        state_before = deepcopy(self.state_store.state)
        result = DryRunExecutionAdapter().check_for_fills(
            simulated_order_id=intent["simulated_order_id"],
            intent=intent,
            checked_at=FILLED_AT,
        )
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["status"], "filled")
        self.assertEqual(result["simulated_order_id"], intent["simulated_order_id"])
        self.assertIsInstance(result["fill_id"], str)
        assert_numeric_price(self, result["fill_price"])
        assert_numeric_quantity(self, result["fill_quantity"])
        self.assertNotIn(FORBIDDEN_ORDER_ID_FIELD, result)
        self.assertEqual(result["intent_id"], intent["intent_id"])
        self.assertEqual(result["strategy_id"], intent["strategy_id"])
        self.assertEqual(result["symbol"], intent["symbol"])
        self.assertEqual(result["order_ref"], intent["order_ref"])
        self.assertEqual(result["action"], "open")
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")
        self.assertEqual(self.state_store.state, state_before)


class ConfirmFillHelperTests(Phase3JCase):
    def assert_successful_fill(self, strategy_id: str, intent_id: str) -> dict:
        confirmed = self.create_confirmed_intent(strategy_id, intent_id)
        updated = confirm_fill(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            intent_id=intent_id,
            filled_at=FILLED_AT,
        )
        events = self.order_events()
        self.assertEqual([event["event_type"] for event in events], ["FILL_CONFIRMED"])
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
            "order_submitted_event_id",
            "order_confirmed_event_id",
            "filled_at",
            "checked_at",
            "dry_run",
            "simulated_order_id",
            "fill_id",
            "fill_price",
            "fill_quantity",
            "action",
            "event_detail",
        ):
            self.assertIn(field, payload)
        self.assertEqual(payload["simulated_order_id"], confirmed["simulated_order_id"])
        self.assertNotIn(FORBIDDEN_ORDER_ID_FIELD, payload)
        assert_numeric_price(self, payload["fill_price"])
        assert_numeric_quantity(self, payload["fill_quantity"])
        self.assertEqual(updated["fill_confirmed_event_id"], events[0]["event_id"])
        self.assertEqual(updated["simulated_order_id"], confirmed["simulated_order_id"])
        self.assertIn("fill_id", updated)
        assert_numeric_price(self, updated["fill_price"])
        assert_numeric_quantity(self, updated["fill_quantity"])
        self.assertEqual(updated["status"], "filled")
        self.assertTrue(updated["dry_run"])
        self.assertNotIn("POSITION_", self.order_path.read_text(encoding="utf-8"))
        return updated

    def test_confirmed_s01_and_s02_intents_fill_successfully(self) -> None:
        self.assert_successful_fill(S01_VOL_BASELINE, "s01:helper")
        self.order_path.write_text("", encoding="utf-8")
        self.assert_successful_fill(S02_VOL_ENHANCED, "s02:helper")

    def test_missing_required_intent_fields_raise_clear_errors(self) -> None:
        self.create_confirmed_intent(S01_VOL_BASELINE, "s01:no-dry")
        del self.state_store.state["order_intents"]["s01:no-dry"]["dry_run"]
        self.state_store.save()
        with self.assertRaisesRegex(ValueError, "intent.dry_run"):
            confirm_fill(
                state_store=self.state_store,
                ledger=self.ledger,
                execution_adapter=DryRunExecutionAdapter(),
                intent_id="s01:no-dry",
                filled_at=FILLED_AT,
            )

        self.create_confirmed_intent(S01_VOL_BASELINE, "s01:no-sim")
        del self.state_store.state["order_intents"]["s01:no-sim"]["simulated_order_id"]
        self.state_store.save()
        with self.assertRaisesRegex(ValueError, "simulated_order_id"):
            confirm_fill(
                state_store=self.state_store,
                ledger=self.ledger,
                execution_adapter=DryRunExecutionAdapter(),
                intent_id="s01:no-sim",
                filled_at=FILLED_AT,
            )

        self.create_confirmed_intent(S01_VOL_BASELINE, "s01:no-confirm")
        del self.state_store.state["order_intents"]["s01:no-confirm"]["order_confirmed_event_id"]
        self.state_store.save()
        with self.assertRaisesRegex(ValueError, "order_confirmed_event_id"):
            confirm_fill(
                state_store=self.state_store,
                ledger=self.ledger,
                execution_adapter=DryRunExecutionAdapter(),
                intent_id="s01:no-confirm",
                filled_at=FILLED_AT,
            )
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")

    def test_adapter_non_filled_and_invalid_fill_fields_write_nothing(self) -> None:
        scenarios = [
            ("s01:pending", {"status": "confirmed"}, "not 'filled'"),
            ("s01:zero-qty", {"status": "filled", "fill_id": "f", "fill_price": 1.25, "fill_quantity": 0}, "fill_quantity"),
            ("s01:string-qty", {"status": "filled", "fill_id": "f", "fill_price": 1.25, "fill_quantity": "1"}, "fill_quantity"),
            ("s01:neg-price", {"status": "filled", "fill_id": "f", "fill_price": -0.01, "fill_quantity": 1}, "fill_price"),
            ("s01:string-price", {"status": "filled", "fill_id": "f", "fill_price": "1.25", "fill_quantity": 1}, "fill_price"),
        ]
        for intent_id, fill, expected in scenarios:
            self.create_confirmed_intent(S01_VOL_BASELINE, intent_id)
            before = deepcopy(self.state_store.state)
            adapter = mock.Mock()
            adapter.check_for_fills.return_value = fill
            with self.assertRaisesRegex(ValueError, expected):
                confirm_fill(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    execution_adapter=adapter,
                    intent_id=intent_id,
                    filled_at=FILLED_AT,
                )
            self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")
            self.assertEqual(self.state_store.state, before)

    def test_no_broker_calls(self) -> None:
        broker = mock.Mock()
        broker.submit_order = mock.Mock()
        broker.placeOrder = mock.Mock()
        broker.cancelOrder = mock.Mock()
        self.assert_successful_fill(S01_VOL_BASELINE, "s01:no-broker")
        broker.submit_order.assert_not_called()
        broker.placeOrder.assert_not_called()
        broker.cancelOrder.assert_not_called()


class ConfirmFillFailureOrderTests(Phase3JCase):
    def test_adapter_failure_writes_nothing_and_leaves_state_unchanged(self) -> None:
        self.create_confirmed_intent(S01_VOL_BASELINE, "s01:adapter-fail")
        before = deepcopy(self.state_store.state)
        adapter = mock.Mock()
        adapter.check_for_fills.side_effect = RuntimeError("adapter failed")
        with self.assertRaisesRegex(RuntimeError, "adapter failed"):
            confirm_fill(
                state_store=self.state_store,
                ledger=self.ledger,
                execution_adapter=adapter,
                intent_id="s01:adapter-fail",
                filled_at=FILLED_AT,
            )
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")
        self.assertEqual(self.state_store.state, before)

    def test_ledger_failure_leaves_state_unchanged(self) -> None:
        self.create_confirmed_intent(S01_VOL_BASELINE, "s01:ledger-fail")
        before = deepcopy(self.state_store.state)
        ledger = mock.Mock()
        ledger.append.side_effect = RuntimeError("ledger failed")
        with self.assertRaisesRegex(RuntimeError, "ledger failed"):
            confirm_fill(
                state_store=self.state_store,
                ledger=ledger,
                execution_adapter=DryRunExecutionAdapter(),
                intent_id="s01:ledger-fail",
                filled_at=FILLED_AT,
            )
        self.assertEqual(self.state_store.state, before)

    def test_state_store_failure_propagates_after_fill_confirmed(self) -> None:
        self.create_confirmed_intent(S01_VOL_BASELINE, "s01:state-fail")
        with mock.patch.object(
            self.state_store,
            "fill_order_intent",
            side_effect=RuntimeError("state failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "state failed"):
                confirm_fill(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    execution_adapter=DryRunExecutionAdapter(),
                    intent_id="s01:state-fail",
                    filled_at=FILLED_AT,
                )
        text = self.order_path.read_text(encoding="utf-8")
        self.assertIn("FILL_CONFIRMED", text)
        self.assertNotIn("POSITION_", text)


class ConfirmFillCliTests(Phase3JCase):
    def assert_cli_fills_confirmed_intent(self, strategy_id: str, intent_id: str) -> None:
        self.create_confirmed_intent(strategy_id, intent_id)
        code, stdout, stderr = self.run_fill_cli(
            [
                "--root-dir",
                str(self.root),
                "--intent-id",
                intent_id,
                "--filled-at",
                FILLED_AT,
            ]
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertIn(intent_id, stdout)
        self.assertIn("filled", stdout)
        self.assertIn("simulated_order_id", stdout)
        self.assertIn("fill_id", stdout)
        self.assertIn("fill_confirmed_event_id", stdout)
        stored = self.load_state_store().get_order_intent(intent_id)
        self.assertEqual(stored["status"], "filled")
        self.assertIn("FILL_CONFIRMED", self.order_path.read_text(encoding="utf-8"))
        self.assertNotIn("POSITION_", self.order_path.read_text(encoding="utf-8"))

    def test_cli_fills_s01_and_s02(self) -> None:
        self.assert_cli_fills_confirmed_intent(S01_VOL_BASELINE, "s01:cli")
        self.order_path.write_text("", encoding="utf-8")
        self.assert_cli_fills_confirmed_intent(S02_VOL_ENHANCED, "s02:cli")

    def test_json_output_is_strict_json_only(self) -> None:
        intent_id = "s01:json"
        self.create_confirmed_intent(S01_VOL_BASELINE, intent_id)
        code, stdout, stderr = self.run_fill_cli(
            [
                "--root-dir",
                str(self.root),
                "--intent-id",
                intent_id,
                "--filled-at",
                FILLED_AT,
                "--json",
            ]
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(stdout.strip(), json.dumps(payload, separators=(",", ":"), sort_keys=True))
        self.assertEqual(
            set(payload),
            {
                "intent_id",
                "status",
                "simulated_order_id",
                "fill_id",
                "fill_confirmed_event_id",
                "dry_run",
            },
        )
        self.assertEqual(payload["intent_id"], intent_id)
        self.assertEqual(payload["status"], "filled")
        self.assertIs(payload["dry_run"], True)
        self.assertIsInstance(payload["dry_run"], bool)

    def test_cli_error_paths_write_no_fill(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.run_fill_cli(["--root-dir", str(self.root)])
        self.assertNotEqual(caught.exception.code, 0)

        self.create_intent(S01_VOL_BASELINE, "s01:created")
        self.create_submitted_intent(S01_VOL_BASELINE, "s01:submitted")
        self.create_intent(S01_VOL_BASELINE, "s01:expired")
        self.state_store.expire_order_intent(
            "s01:expired",
            expired_at=FILLED_AT,
            expire_reason="ttl_expired",
            expired_event_id="evt_expired",
        )
        self.create_intent(S01_VOL_BASELINE, "s01:cancelled")
        self.state_store.cancel_order_intent(
            "s01:cancelled",
            cancelled_at=FILLED_AT,
            cancel_reason="operator_cancelled",
            cancelled_event_id="evt_cancelled",
        )
        self.create_confirmed_intent(S01_VOL_BASELINE, "s01:filled")
        self.state_store.fill_order_intent(
            "s01:filled",
            filled_at=FILLED_AT,
            fill_confirmed_event_id="evt_fill",
            simulated_order_id="sim_filled",
            fill_id="fill_filled",
            fill_price=1.25,
            fill_quantity=1,
        )
        self.create_confirmed_intent(S01_VOL_BASELINE, "s01:no-dry")
        del self.state_store.state["order_intents"]["s01:no-dry"]["dry_run"]
        self.create_confirmed_intent(S01_VOL_BASELINE, "s01:no-sim")
        del self.state_store.state["order_intents"]["s01:no-sim"]["simulated_order_id"]
        self.create_confirmed_intent(S01_VOL_BASELINE, "s01:no-confirm")
        del self.state_store.state["order_intents"]["s01:no-confirm"]["order_confirmed_event_id"]
        self.state_store.save()

        cases = [
            ("s01:created", "not 'confirmed'"),
            ("s01:submitted", "not 'confirmed'"),
            ("s01:expired", "not 'confirmed'"),
            ("s01:cancelled", "not 'confirmed'"),
            ("s01:filled", "not 'confirmed'"),
            ("s01:no-dry", "intent.dry_run"),
            ("s01:no-sim", "simulated_order_id"),
            ("s01:no-confirm", "order_confirmed_event_id"),
            ("missing:intent", "missing:intent"),
        ]
        for intent_id, expected in cases:
            code, stdout, stderr = self.run_fill_cli(
                [
                    "--root-dir",
                    str(self.root),
                    "--intent-id",
                    intent_id,
                    "--filled-at",
                    FILLED_AT,
                ]
            )
            self.assertNotEqual(code, 0)
            self.assertEqual(stdout, "")
            self.assertIn(expected, stderr)
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")

    def test_invalid_filled_at_exits_nonzero_without_mutation(self) -> None:
        intent_id = "s01:bad-time"
        self.create_confirmed_intent(S01_VOL_BASELINE, intent_id)
        before = self.load_state_store().get_order_intent(intent_id)
        with self.assertRaises(SystemExit):
            self.run_fill_cli(
                [
                    "--root-dir",
                    str(self.root),
                    "--intent-id",
                    intent_id,
                    "--filled-at",
                    "not-a-time",
                ]
            )
        code, stdout, stderr = self.run_fill_cli(
            [
                "--root-dir",
                str(self.root),
                "--intent-id",
                intent_id,
                "--filled-at",
                "not-a-time",
            ]
        )
        self.assertNotEqual(code, 0)
        self.assertIn("Error", stderr)
        self.assertEqual(self.load_state_store().get_order_intent(intent_id), before)
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")


class Phase3JRegressionAndSafetyTests(Phase3JCase):
    def test_prior_paths_do_not_auto_fill_or_create_positions(self) -> None:
        self.set_readiness(S01_VOL_BASELINE)
        scan = run_s01_vol_scan(
            readiness_manager=self.manager,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=mock.Mock(),
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: self.clean_input(S01_VOL_BASELINE),
            ledger_reader=self.reader(),
        )
        self.assertEqual(scan.detail, "order_intent_created")
        self.assertEqual([event["event_type"] for event in self.order_events()], ["ORDER_INTENT_CREATED"])

        self.order_path.write_text("", encoding="utf-8")
        self.submit_with_helper(S01_VOL_BASELINE, "s01:submit-only")
        self.assertEqual([event["event_type"] for event in self.order_events()], ["ORDER_SUBMITTED"])

        self.order_path.write_text("", encoding="utf-8")
        self.confirm_with_helper(S01_VOL_BASELINE, "s01:confirm-only")
        self.assertEqual([event["event_type"] for event in self.order_events()], ["ORDER_SUBMITTED", "ORDER_CONFIRMED"])
        self.assertNotIn("FILL_CONFIRMED", self.order_path.read_text(encoding="utf-8"))

        self.order_path.write_text("", encoding="utf-8")
        self._run_submit_cli_for_created_intent("s01:submit-cli")
        self.assertNotIn("ORDER_CONFIRMED", self.order_path.read_text(encoding="utf-8"))
        self.assertNotIn("FILL_CONFIRMED", self.order_path.read_text(encoding="utf-8"))

        self.order_path.write_text("", encoding="utf-8")
        self._run_confirm_cli_for_submitted_intent("s01:confirm-cli")
        self.assertNotIn("FILL_CONFIRMED", self.order_path.read_text(encoding="utf-8"))

    def _run_submit_cli_for_created_intent(self, intent_id: str) -> tuple[int, str, str]:
        self.create_intent(S01_VOL_BASELINE, intent_id)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = submit_tool.main(
                [
                    "--root-dir",
                    str(self.root),
                    "--intent-id",
                    intent_id,
                    "--submitted-at",
                    SUBMITTED_AT,
                ]
            )
        return code, stdout.getvalue(), stderr.getvalue()

    def _run_confirm_cli_for_submitted_intent(self, intent_id: str) -> tuple[int, str, str]:
        self.create_submitted_intent(S01_VOL_BASELINE, intent_id)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = confirm_tool.main(
                [
                    "--root-dir",
                    str(self.root),
                    "--intent-id",
                    intent_id,
                    "--confirmed-at",
                    CONFIRMED_AT,
                ]
            )
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_source_stays_dry_run_and_helper_only(self) -> None:
        source = inspect.getsource(fill_tool)
        helper_source = inspect.getsource(confirm_fill)
        self.assertIn("argparse.ArgumentParser", source)
        self.assertIn("DryRunExecutionAdapter", source)
        self.assertIn("confirm_fill(", source)
        self.assertNotIn("ib_insync", source)
        self.assertNotIn("yfinance", source)
        self.assertNotIn("requests", source)
        self.assertNotIn("broker.submit_order", source)
        self.assertNotIn("placeOrder", source)
        self.assertNotIn("cancelOrder", source)
        self.assertNotIn("ledger.append", source)
        self.assertNotIn("StateStore.fill_order_intent", source)
        self.assertNotIn("scheduler", source.lower())
        self.assertNotIn("POSITION_", source)
        self.assertNotIn("except:", source)
        self.assertNotIn("except:", helper_source)

    def test_import_has_no_output_and_default_time_is_aware(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            importlib.reload(fill_tool)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

        intent_id = "s01:auto-time"
        self.create_confirmed_intent(S01_VOL_BASELINE, intent_id)
        code, stdout_text, stderr_text = self.run_fill_cli(
            ["--root-dir", str(self.root), "--intent-id", intent_id, "--json"]
        )
        self.assertEqual(code, 0, stderr_text)
        self.assertEqual(json.loads(stdout_text)["status"], "filled")
        parsed = datetime.fromisoformat(
            self.load_state_store().get_order_intent(intent_id)["filled_at"]
        )
        self.assertIsNotNone(parsed.tzinfo)


if __name__ == "__main__":
    unittest.main()