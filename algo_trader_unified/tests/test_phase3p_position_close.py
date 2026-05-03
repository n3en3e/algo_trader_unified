from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import json
import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.close_intents import (
    confirm_close_fill,
    confirm_close_order,
    create_close_intent_from_position,
    submit_close_intent,
)
from algo_trader_unified.core.execution import DryRunExecutionAdapter
from algo_trader_unified.core.ledger import KNOWN_EVENT_TYPES, LedgerAppender
from algo_trader_unified.core.positions import close_position_from_filled_intent
from algo_trader_unified.core.state_store import ACTIVE_CLOSE_INTENT_STATUSES, StateStore
from algo_trader_unified.tools import close_position as close_tool
from algo_trader_unified.tools import list_positions as list_positions_tool
from algo_trader_unified.tools import system_status


CREATED_AT = "2026-04-27T15:10:00+00:00"
SUBMITTED_AT = "2026-04-27T15:20:00+00:00"
CONFIRMED_AT = "2026-04-27T15:25:00+00:00"
FILLED_AT = "2026-04-27T15:30:00+00:00"
CLOSED_AT = "2026-04-27T15:35:00+00:00"
BROKER_ORDER_ID = "broker" + "_order_id"


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


class Phase3PCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_path = self.root / "data/state/portfolio_state.json"
        self.order_path = self.root / "data/ledger/order_ledger.jsonl"
        self.execution_path = self.root / "data/ledger/execution_ledger.jsonl"
        self.state_store = StateStore(self.state_path)
        self.ledger = LedgerAppender(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def position_record(
        self,
        strategy_id: str = S01_VOL_BASELINE,
        position_id: str = "position:1",
        *,
        symbol: str = "XSP",
        status: str = "open",
        quantity: object = 2,
        entry_price: object = 0.75,
    ) -> dict:
        return {
            "position_id": position_id,
            "intent_id": f"{position_id}:intent",
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": symbol,
            "status": status,
            "execution_mode": "paper_only",
            "dry_run": True,
            "opened_at": "2026-04-27T14:07:00+00:00",
            "updated_at": "2026-04-27T14:07:00+00:00",
            "source_signal_event_id": "evt_signal",
            "order_intent_created_event_id": "evt_created",
            "order_submitted_event_id": "evt_submitted",
            "order_confirmed_event_id": "evt_confirmed",
            "fill_confirmed_event_id": "evt_fill",
            "position_opened_event_id": "evt_opened",
            "order_ref": f"{strategy_id}|P0427XSP|OPEN",
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

    def create_filled_close_intent(self, strategy_id: str, position_id: str) -> dict:
        self.create_position(strategy_id, position_id)
        created = create_close_intent_from_position(
            state_store=self.state_store,
            ledger=self.ledger,
            position_id=position_id,
            created_at=CREATED_AT,
        )
        submitted = submit_close_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=created["close_intent_id"],
            submitted_at=SUBMITTED_AT,
        )
        confirmed = confirm_close_order(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=submitted["close_intent_id"],
            confirmed_at=CONFIRMED_AT,
        )
        return confirm_close_fill(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=confirmed["close_intent_id"],
            filled_at=FILLED_AT,
        )

    def order_events(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.order_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def execution_events(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.execution_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = close_tool.main(["--root-dir", str(self.root), *argv])
        return code, stdout.getvalue(), stderr.getvalue()


class PositionClosedEventAndStateTests(Phase3PCase):
    def test_position_closed_routes_to_execution_and_payload_is_scaffold_only(self) -> None:
        self.assertIn("POSITION_CLOSED", KNOWN_EVENT_TYPES)
        self.assertEqual(
            self.ledger.path_for_event_type("POSITION_CLOSED").name,
            "execution_ledger.jsonl",
        )
        intent = self.create_filled_close_intent(S01_VOL_BASELINE, "position:event")
        closed = close_position_from_filled_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            close_intent_id=intent["close_intent_id"],
            closed_at=CLOSED_AT,
        )

        self.assertEqual(
            [event["event_type"] for event in self.order_events()],
            [
                "CLOSE_INTENT_CREATED",
                "CLOSE_ORDER_SUBMITTED",
                "CLOSE_ORDER_CONFIRMED",
                "CLOSE_FILL_CONFIRMED",
            ],
        )
        self.assertEqual([event["event_type"] for event in self.execution_events()], ["POSITION_CLOSED"])
        payload = self.execution_events()[0]["payload"]
        for field in (
            "position_id",
            "close_intent_id",
            "strategy_id",
            "sleeve_id",
            "symbol",
            "execution_mode",
            "dry_run",
            "opened_at",
            "closed_at",
            "position_opened_event_id",
            "position_closed_event_id",
            "close_intent_created_event_id",
            "close_order_submitted_event_id",
            "close_order_confirmed_event_id",
            "close_fill_confirmed_event_id",
            "source_signal_event_id",
            "fill_confirmed_event_id",
            "simulated_order_id",
            "simulated_close_order_id",
            "fill_id",
            "close_fill_id",
            "entry_price",
            "quantity",
            "close_fill_price",
            "close_fill_quantity",
            "realized_pnl",
            "action",
            "status",
            "event_detail",
        ):
            self.assertIn(field, payload)
        self.assertIs(payload["dry_run"], True)
        for field in ("entry_price", "quantity", "close_fill_price", "close_fill_quantity", "realized_pnl"):
            self.assertIsInstance(payload[field], (int, float))
            self.assertNotIsInstance(payload[field], str)
        self.assertGreaterEqual(payload["entry_price"], 0)
        self.assertGreater(payload["quantity"], 0)
        self.assertGreaterEqual(payload["close_fill_price"], 0)
        self.assertGreater(payload["close_fill_quantity"], 0)
        self.assertEqual(payload["close_fill_quantity"], payload["quantity"])
        self.assertEqual(payload["status"], "closed")
        self.assertEqual(payload["action"], "close")
        self.assertEqual(payload["event_detail"], "POSITION_CLOSED")
        self.assertEqual(closed["position_closed_event_id"], payload["position_closed_event_id"])
        for forbidden in (
            BROKER_ORDER_ID,
            "target_price",
            "limit_price",
            "order_type",
            "time_in_force",
            "side",
            "direction",
            "multiplier",
            "legs",
        ):
            self.assertNotIn(forbidden, payload)
        self.assertNotIn("POSITION_ADJUSTED", [event["event_type"] for event in self.execution_events()])

    def test_state_store_position_and_close_intent_final_transitions(self) -> None:
        intent = self.create_filled_close_intent(S01_VOL_BASELINE, "position:state")
        closed = self.state_store.close_position(
            "position:state",
            closed_at=CLOSED_AT,
            position_closed_event_id="evt_closed",
            close_intent_id=intent["close_intent_id"],
            close_fill_confirmed_event_id=intent["close_fill_confirmed_event_id"],
            close_fill_price=1.25,
            close_fill_quantity=intent["quantity"],
            realized_pnl=1.0,
        )
        self.assertEqual(closed["status"], "closed")
        self.assertEqual(closed["closed_at"], CLOSED_AT)
        self.assertEqual(closed["updated_at"], CLOSED_AT)
        self.assertEqual(closed["position_closed_event_id"], "evt_closed")
        self.assertEqual(closed["close_intent_id"], intent["close_intent_id"])
        self.assertEqual(closed["close_fill_confirmed_event_id"], intent["close_fill_confirmed_event_id"])
        self.assertEqual(closed["close_fill_price"], 1.25)
        self.assertEqual(closed["close_fill_quantity"], intent["quantity"])
        self.assertEqual(closed["realized_pnl"], 1.0)
        self.assertTrue(closed["dry_run"])
        self.assertIsNone(closed.get("active_close_intent_id"))
        self.assertIsNone(self.state_store.get_open_position(S01_VOL_BASELINE))
        self.assertEqual(self.state_store.list_positions(status="closed"), [closed])

        final_intent = self.state_store.mark_close_intent_position_closed(
            intent["close_intent_id"],
            position_closed_event_id="evt_closed",
            closed_at=CLOSED_AT,
        )
        self.assertEqual(final_intent["status"], "position_closed")
        self.assertEqual(final_intent["position_closed_event_id"], "evt_closed")
        self.assertEqual(final_intent["position_closed_at"], CLOSED_AT)
        self.assertEqual(final_intent["updated_at"], CLOSED_AT)
        self.assertIsNone(self.state_store.get_active_close_intent("position:state"))
        self.assertEqual(ACTIVE_CLOSE_INTENT_STATUSES, {"created", "submitted", "confirmed", "filled"})

    def test_state_store_transition_rejections(self) -> None:
        intent = self.create_filled_close_intent(S01_VOL_BASELINE, "position:reject")
        with self.assertRaisesRegex(KeyError, "does not exist"):
            self.state_store.close_position(
                "missing",
                closed_at=CLOSED_AT,
                position_closed_event_id="evt",
                close_intent_id=intent["close_intent_id"],
                close_fill_confirmed_event_id=intent["close_fill_confirmed_event_id"],
                close_fill_price=1.0,
                close_fill_quantity=intent["quantity"],
                realized_pnl=0,
            )
        for field, value, pattern in (
            ("status", "closed", "not 'open'"),
            ("active_close_intent_id", "other", "active_close_intent_id"),
        ):
            intent = self.create_filled_close_intent(S02_VOL_ENHANCED, f"position:{field}")
            self.state_store.state["positions"][intent["position_id"]][field] = value
            self.state_store.save()
            with self.assertRaisesRegex(ValueError, pattern):
                self.state_store.close_position(
                    intent["position_id"],
                    closed_at=CLOSED_AT,
                    position_closed_event_id="evt",
                    close_intent_id=intent["close_intent_id"],
                    close_fill_confirmed_event_id=intent["close_fill_confirmed_event_id"],
                    close_fill_price=1.0,
                    close_fill_quantity=intent["quantity"],
                    realized_pnl=0,
                )
        intent = self.create_filled_close_intent(S01_VOL_BASELINE, "position:qty-mismatch")
        with self.assertRaisesRegex(ValueError, "close_fill_quantity"):
            self.state_store.close_position(
                intent["position_id"],
                closed_at=CLOSED_AT,
                position_closed_event_id="evt",
                close_intent_id=intent["close_intent_id"],
                close_fill_confirmed_event_id=intent["close_fill_confirmed_event_id"],
                close_fill_price=1.0,
                close_fill_quantity=intent["quantity"] + 1,
                realized_pnl=0,
            )

        with self.assertRaisesRegex(KeyError, "does not exist"):
            self.state_store.mark_close_intent_position_closed(
                "missing",
                position_closed_event_id="evt",
                closed_at=CLOSED_AT,
            )
        self.create_position(S01_VOL_BASELINE, "position:created-nonfilled")
        created = create_close_intent_from_position(
            state_store=self.state_store,
            ledger=self.ledger,
            position_id="position:created-nonfilled",
            created_at="2026-04-27T15:40:00+00:00",
        )
        with self.assertRaisesRegex(ValueError, "not 'filled'"):
            self.state_store.mark_close_intent_position_closed(
                created["close_intent_id"],
                position_closed_event_id="evt",
                closed_at=CLOSED_AT,
            )
        intent = self.create_filled_close_intent(S02_VOL_ENHANCED, "position:missing-fill-event")
        self.state_store.state["close_intents"][intent["close_intent_id"]].pop("close_fill_confirmed_event_id")
        self.state_store.save()
        with self.assertRaisesRegex(ValueError, "close_fill_confirmed_event_id"):
            self.state_store.mark_close_intent_position_closed(
                intent["close_intent_id"],
                position_closed_event_id="evt",
                closed_at=CLOSED_AT,
            )


class ClosePositionHelperTests(Phase3PCase):
    def assert_successful_close(self, strategy_id: str, position_id: str) -> dict:
        intent = self.create_filled_close_intent(strategy_id, position_id)
        closed = close_position_from_filled_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            close_intent_id=intent["close_intent_id"],
            closed_at=CLOSED_AT,
        )
        self.assertEqual(closed["status"], "closed")
        self.assertIsNone(closed.get("active_close_intent_id"))
        self.assertEqual(
            self.state_store.get_close_intent(intent["close_intent_id"])["status"],
            "position_closed",
        )
        self.assertEqual([event["event_type"] for event in self.execution_events()][-1], "POSITION_CLOSED")
        return closed

    def test_s01_and_s02_filled_close_intents_close_positions(self) -> None:
        self.assert_successful_close(S01_VOL_BASELINE, "position:s01")
        self.assert_successful_close(S02_VOL_ENHANCED, "position:s02")

    def test_dry_run_placeholder_realized_pnl_formula_is_mechanical_only(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:pnl", quantity=3, entry_price=10)
        intent = self.create_filled_close_intent(S01_VOL_BASELINE, "position:pnl")
        self.state_store.state["close_intents"][intent["close_intent_id"]].update(
            entry_price=10,
            quantity=3,
            close_fill_price=12,
            close_fill_quantity=3,
        )
        self.state_store.state["positions"]["position:pnl"].update(entry_price=10, quantity=3)
        self.state_store.save()

        closed = close_position_from_filled_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            close_intent_id=intent["close_intent_id"],
            closed_at=CLOSED_AT,
        )

        self.assertEqual(closed["realized_pnl"], 6)
        payload = self.execution_events()[-1]["payload"]
        self.assertEqual(payload["realized_pnl"], 6)
        source = inspect.getsource(importlib.import_module("algo_trader_unified.core.positions"))
        for forbidden in ("side", "direction", "multiplier", "legs"):
            self.assertNotIn(f"{forbidden}_pnl", source)

    def test_validation_failures_happen_before_ledger_or_state_mutation(self) -> None:
        base_intent = self.create_filled_close_intent(S01_VOL_BASELINE, "position:base")
        with self.assertRaisesRegex(KeyError, "does not exist"):
            close_position_from_filled_intent(
                state_store=self.state_store,
                ledger=self.ledger,
                close_intent_id="missing",
                closed_at=CLOSED_AT,
            )
        self.assertEqual(self.execution_events(), [])

        for missing in (
            "dry_run",
            "close_intent_created_event_id",
            "close_order_submitted_event_id",
            "close_order_confirmed_event_id",
            "close_fill_confirmed_event_id",
            "position_opened_event_id",
            "source_signal_event_id",
            "fill_confirmed_event_id",
            "simulated_close_order_id",
            "close_fill_id",
            "close_fill_price",
            "close_fill_quantity",
            "quantity",
            "entry_price",
        ):
            intent = self.create_filled_close_intent(S02_VOL_ENHANCED, f"position:missing:{missing}")
            before_exec = self.execution_events()
            self.state_store.state["close_intents"][intent["close_intent_id"]].pop(missing)
            self.state_store.save()
            with self.assertRaises((ValueError, KeyError)):
                close_position_from_filled_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    close_intent_id=intent["close_intent_id"],
                    closed_at=CLOSED_AT,
                )
            self.assertEqual(self.execution_events(), before_exec)
            self.assertEqual(self.state_store.get_close_intent(intent["close_intent_id"])["status"], "filled")

        cases = [
            ("position-missing", lambda intent: self.state_store.state["positions"].pop(intent["position_id"])),
            ("not-open", lambda intent: self.state_store.state["positions"][intent["position_id"]].update(status="closed")),
            ("wrong-link", lambda intent: self.state_store.state["positions"][intent["position_id"]].update(active_close_intent_id="other")),
            ("strategy-mismatch", lambda intent: self.state_store.state["positions"][intent["position_id"]].update(strategy_id=S02_VOL_ENHANCED)),
            ("symbol-mismatch", lambda intent: self.state_store.state["positions"][intent["position_id"]].update(symbol="SPX")),
            ("quantity-mismatch", lambda intent: self.state_store.state["positions"][intent["position_id"]].update(quantity=intent["quantity"] + 1)),
            ("entry-price-mismatch", lambda intent: self.state_store.state["positions"][intent["position_id"]].update(entry_price=intent["entry_price"] + 1)),
        ]
        for name, mutate in cases:
            intent = self.create_filled_close_intent(S01_VOL_BASELINE, f"position:{name}")
            before_exec = self.execution_events()
            mutate(intent)
            self.state_store.save()
            with self.assertRaises((KeyError, ValueError)):
                close_position_from_filled_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    close_intent_id=intent["close_intent_id"],
                    closed_at=CLOSED_AT,
                )
            self.assertEqual(self.execution_events(), before_exec)
            self.assertEqual(self.state_store.get_close_intent(intent["close_intent_id"])["status"], "filled")

        self.assertEqual(self.state_store.get_close_intent(base_intent["close_intent_id"])["status"], "filled")

    def test_failure_ordering_after_validation(self) -> None:
        intent = self.create_filled_close_intent(S01_VOL_BASELINE, "position:ledger-fail")
        ledger = mock.Mock()
        ledger.append.side_effect = RuntimeError("ledger failed")
        with self.assertRaisesRegex(RuntimeError, "ledger failed"):
            close_position_from_filled_intent(
                state_store=self.state_store,
                ledger=ledger,
                close_intent_id=intent["close_intent_id"],
                closed_at=CLOSED_AT,
            )
        self.assertEqual(self.state_store.get_position(intent["position_id"])["status"], "open")
        self.assertEqual(self.state_store.get_close_intent(intent["close_intent_id"])["status"], "filled")

        intent = self.create_filled_close_intent(S02_VOL_ENHANCED, "position:position-state-fail")
        with mock.patch.object(self.state_store, "close_position", side_effect=RuntimeError("position failed")):
            with self.assertRaisesRegex(RuntimeError, "position failed"):
                close_position_from_filled_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    close_intent_id=intent["close_intent_id"],
                    closed_at=CLOSED_AT,
                )
        self.assertEqual(self.execution_events()[-1]["event_type"], "POSITION_CLOSED")
        self.assertEqual(self.state_store.get_position(intent["position_id"])["status"], "open")
        self.assertEqual(self.state_store.get_close_intent(intent["close_intent_id"])["status"], "filled")

        intent = self.create_filled_close_intent(S01_VOL_BASELINE, "position:intent-state-fail")
        with mock.patch.object(
            self.state_store,
            "mark_close_intent_position_closed",
            side_effect=RuntimeError("intent failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "intent failed"):
                close_position_from_filled_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    close_intent_id=intent["close_intent_id"],
                    closed_at=CLOSED_AT,
                )
        self.assertEqual(self.execution_events()[-1]["event_type"], "POSITION_CLOSED")
        self.assertEqual(self.state_store.get_position(intent["position_id"])["status"], "closed")
        self.assertEqual(self.state_store.get_close_intent(intent["close_intent_id"])["status"], "filled")

    def test_helper_holds_strategy_lock_through_ledger_and_state_transitions(self) -> None:
        intent = self.create_filled_close_intent(S01_VOL_BASELINE, "position:lock")
        spy = LockSpy()
        self.state_store.strategy_state_locks[S01_VOL_BASELINE] = spy
        real_append = self.ledger.append
        real_close_position = self.state_store.close_position
        real_mark_intent = self.state_store.mark_close_intent_position_closed

        def append_with_lock_check(**kwargs):
            self.assertTrue(spy.held)
            return real_append(**kwargs)

        def close_position_with_lock_check(*args, **kwargs):
            self.assertTrue(spy.held)
            return real_close_position(*args, **kwargs)

        def mark_intent_with_lock_check(*args, **kwargs):
            self.assertTrue(spy.held)
            return real_mark_intent(*args, **kwargs)

        with mock.patch.object(self.ledger, "append", side_effect=append_with_lock_check):
            with mock.patch.object(self.state_store, "close_position", side_effect=close_position_with_lock_check):
                with mock.patch.object(
                    self.state_store,
                    "mark_close_intent_position_closed",
                    side_effect=mark_intent_with_lock_check,
                ):
                    close_position_from_filled_intent(
                        state_store=self.state_store,
                        ledger=self.ledger,
                        close_intent_id=intent["close_intent_id"],
                        closed_at=CLOSED_AT,
                    )


class ClosePositionCliTests(Phase3PCase):
    def test_cli_success_human_json_z_suffix_omitted_timestamp_and_dry_run(self) -> None:
        intent = self.create_filled_close_intent(S01_VOL_BASELINE, "position:cli:s01")
        code, stdout, stderr = self.run_cli(
            ["--close-intent-id", intent["close_intent_id"], "--closed-at", "2026-04-27T15:35:00Z"]
        )
        self.assertEqual(code, 0, stderr)
        for expected in ("position_id", "close_intent_id", "closed", "position_closed_event_id", "realized_pnl"):
            self.assertIn(expected, stdout)
        updated = StateStore(self.state_path).get_position("position:cli:s01")
        self.assertEqual(updated["closed_at"], CLOSED_AT)

        intent = self.create_filled_close_intent(S02_VOL_ENHANCED, "position:cli:s02")
        code, stdout, stderr = self.run_cli(
            ["--close-intent-id", intent["close_intent_id"], "--json", "--dry-run"]
        )
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["position_id"], "position:cli:s02")
        self.assertEqual(payload["close_intent_id"], intent["close_intent_id"])
        self.assertEqual(payload["status"], "closed")
        self.assertIn("position_closed_event_id", payload)
        self.assertIn("realized_pnl", payload)
        self.assertIs(payload["dry_run"], True)
        self.assertEqual(stdout, json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        self.assertIsNotNone(StateStore(self.state_path).get_position("position:cli:s02")["closed_at"])

    def test_cli_errors_are_concise_and_do_not_write_second_close(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                close_tool.main([])

        intent = self.create_filled_close_intent(S01_VOL_BASELINE, "position:cli:error")
        code, _, stderr = self.run_cli(["--close-intent-id", intent["close_intent_id"]])
        self.assertEqual(code, 0, stderr)
        before_exec = self.execution_events()
        code, _, stderr = self.run_cli(["--close-intent-id", intent["close_intent_id"]])
        self.assertNotEqual(code, 0)
        self.assertIn("not 'filled'", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertEqual(self.execution_events(), before_exec)

        code, _, stderr = self.run_cli(["--close-intent-id", "missing"])
        self.assertNotEqual(code, 0)
        self.assertIn("does not exist", stderr)
        self.assertNotIn("Traceback", stderr)

        intent = self.create_filled_close_intent(S02_VOL_ENHANCED, "position:cli:missing-dry-run")
        self.state_store.state["close_intents"][intent["close_intent_id"]].pop("dry_run")
        self.state_store.save()
        before_exec = self.execution_events()
        code, _, stderr = self.run_cli(["--close-intent-id", intent["close_intent_id"]])
        self.assertNotEqual(code, 0)
        self.assertIn("dry_run", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertEqual(self.execution_events(), before_exec)

        intent = self.create_filled_close_intent(S01_VOL_BASELINE, "position:cli:mismatch")
        self.state_store.state["positions"][intent["position_id"]].update(symbol="SPX")
        self.state_store.save()
        before_exec = self.execution_events()
        code, _, stderr = self.run_cli(["--close-intent-id", intent["close_intent_id"]])
        self.assertNotEqual(code, 0)
        self.assertIn("symbol", stderr)
        self.assertEqual(self.execution_events(), before_exec)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                close_tool.main(
                    [
                        "--root-dir",
                        str(self.root),
                        "--close-intent-id",
                        intent["close_intent_id"],
                        "--closed-at",
                        "not-a-timestamp",
                    ]
                )


class SystemStatusAndPathRegressionTests(Phase3PCase):
    def test_status_and_list_positions_include_closed_records(self) -> None:
        intent = self.create_filled_close_intent(S01_VOL_BASELINE, "position:status:s01")
        close_position_from_filled_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            close_intent_id=intent["close_intent_id"],
            closed_at=CLOSED_AT,
        )
        self.create_position(S02_VOL_ENHANCED, "position:status:s02")

        summary = system_status.build_summary(
            [],
            StateStore(self.state_path).list_positions(),
            StateStore(self.state_path).list_close_intents(),
        )
        self.assertEqual(summary["open_positions_count"], 1)
        self.assertEqual(summary["closed_positions_count"], 1)
        self.assertEqual(summary["position_closed_close_intents_count"], 1)
        self.assertEqual(summary["total_positions_count"], 2)
        self.assertEqual(summary["total_close_intents_count"], 1)

        s01_summary = system_status.build_summary(
            [],
            StateStore(self.state_path).list_positions(),
            StateStore(self.state_path).list_close_intents(),
            strategy_id=S01_VOL_BASELINE,
        )
        self.assertEqual(s01_summary["open_positions_count"], 0)
        self.assertEqual(s01_summary["closed_positions_count"], 1)

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = list_positions_tool.main(
                ["--root-dir", str(self.root), "--status", "closed", "--json"]
            )
        self.assertEqual(code, 0, stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["positions"][0]["status"], "closed")

    def test_close_path_writes_only_position_closed_after_existing_close_path(self) -> None:
        intent = self.create_filled_close_intent(S01_VOL_BASELINE, "position:path")
        before_order_events = [event["event_type"] for event in self.order_events()]
        before_execution_events = [event["event_type"] for event in self.execution_events()]
        close_position_from_filled_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            close_intent_id=intent["close_intent_id"],
            closed_at=CLOSED_AT,
        )
        self.assertEqual(
            before_order_events,
            [
                "CLOSE_INTENT_CREATED",
                "CLOSE_ORDER_SUBMITTED",
                "CLOSE_ORDER_CONFIRMED",
                "CLOSE_FILL_CONFIRMED",
            ],
        )
        self.assertEqual(before_execution_events, [])
        self.assertEqual([event["event_type"] for event in self.execution_events()], ["POSITION_CLOSED"])
        self.assertNotIn("POSITION_CLOSED", [event["event_type"] for event in self.order_events()])


class Phase3PSafetySourceTests(unittest.TestCase):
    def test_close_position_sources_avoid_forbidden_surfaces(self) -> None:
        modules = [
            importlib.import_module("algo_trader_unified.core.positions"),
            importlib.import_module("algo_trader_unified.tools.close_position"),
        ]
        forbidden = [
            "ib_insync",
            "yfinance",
            "requests",
            "broker.submit_order",
            "placeOrder",
            "cancelOrder",
            "scheduler.start",
            "except:",
            "POSITION_ADJUSTED",
            "target_price",
            "limit_price",
            "order_type",
            "time_in_force",
            "broker_order_id",
            "side",
            "direction",
            "multiplier",
            "legs",
        ]
        for module in modules:
            source = inspect.getsource(module)
            for needle in forbidden:
                self.assertNotIn(needle, source)
