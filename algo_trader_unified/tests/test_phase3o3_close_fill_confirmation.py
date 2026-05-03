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
from algo_trader_unified.core.state_store import (
    ACTIVE_CLOSE_INTENT_STATUSES,
    StateStore,
)
from algo_trader_unified.tools import confirm_close_fill as fill_tool
from algo_trader_unified.tools import system_status


CREATED_AT = "2026-04-27T15:10:00+00:00"
SUBMITTED_AT = "2026-04-27T15:20:00+00:00"
CONFIRMED_AT = "2026-04-27T15:25:00+00:00"
FILLED_AT = "2026-04-27T15:30:00+00:00"
FORBIDDEN_BROKER_ID_FIELD = "broker" + "_order_id"


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


class Phase3O3Case(unittest.TestCase):
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
            "simulated_order_id": "sim",
            "fill_id": "fill",
            "entry_price": entry_price,
            "quantity": quantity,
            "action": "open",
        }

    def create_position(self, strategy_id: str, position_id: str, **kwargs) -> dict:
        record = self.position_record(strategy_id, position_id, **kwargs)
        self.state_store.state["positions"][position_id] = deepcopy(record)
        self.state_store.save()
        return record

    def create_confirmed_close_intent(self, strategy_id: str, position_id: str) -> dict:
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
        return confirm_close_order(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=submitted["close_intent_id"],
            confirmed_at=CONFIRMED_AT,
        )

    def order_events(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.order_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = fill_tool.main(["--root-dir", str(self.root), *argv])
        return code, stdout.getvalue(), stderr.getvalue()


class CloseFillConfirmedEventTests(Phase3O3Case):
    def test_event_routes_to_order_ledger_only_and_payload_is_scaffold_only(self) -> None:
        self.assertIn("CLOSE_FILL_CONFIRMED", KNOWN_EVENT_TYPES)
        self.assertEqual(
            self.ledger.path_for_event_type("CLOSE_FILL_CONFIRMED").name,
            "order_ledger.jsonl",
        )
        intent = self.create_confirmed_close_intent(S01_VOL_BASELINE, "position:event")
        updated = confirm_close_fill(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=intent["close_intent_id"],
            filled_at=FILLED_AT,
        )
        events = self.order_events()
        self.assertEqual(
            [event["event_type"] for event in events],
            [
                "CLOSE_INTENT_CREATED",
                "CLOSE_ORDER_SUBMITTED",
                "CLOSE_ORDER_CONFIRMED",
                "CLOSE_FILL_CONFIRMED",
            ],
        )
        self.assertEqual(self.execution_path.read_text(encoding="utf-8"), "")
        payload = events[-1]["payload"]
        for field in (
            "close_intent_id",
            "position_id",
            "strategy_id",
            "sleeve_id",
            "symbol",
            "execution_mode",
            "dry_run",
            "close_order_ref",
            "simulated_close_order_id",
            "close_fill_id",
            "close_fill_price",
            "close_fill_quantity",
            "filled_at",
            "checked_at",
            "close_reason",
            "requested_by",
            "close_intent_created_event_id",
            "close_order_submitted_event_id",
            "close_order_confirmed_event_id",
            "position_opened_event_id",
            "fill_confirmed_event_id",
            "quantity",
            "entry_price",
            "action",
            "status",
            "event_detail",
        ):
            self.assertIn(field, payload)
        self.assertIs(payload["dry_run"], True)
        self.assertIsInstance(payload["quantity"], (int, float))
        self.assertIsInstance(payload["entry_price"], (int, float))
        self.assertIsInstance(payload["close_fill_price"], (int, float))
        self.assertIsInstance(payload["close_fill_quantity"], (int, float))
        self.assertNotIsInstance(payload["close_fill_price"], str)
        self.assertNotIsInstance(payload["close_fill_quantity"], str)
        self.assertGreater(payload["quantity"], 0)
        self.assertGreaterEqual(payload["entry_price"], 0)
        self.assertGreaterEqual(payload["close_fill_price"], 0)
        self.assertGreater(payload["close_fill_quantity"], 0)
        self.assertEqual(payload["close_fill_quantity"], payload["quantity"])
        self.assertEqual(payload["status"], "filled")
        self.assertEqual(payload["action"], "close")
        self.assertEqual(payload["event_detail"], "CLOSE_FILL_CONFIRMED")
        self.assertEqual(updated["close_fill_confirmed_event_id"], events[-1]["event_id"])
        for forbidden in (
            FORBIDDEN_BROKER_ID_FIELD,
            "target_price",
            "limit_price",
            "order_type",
            "time_in_force",
        ):
            self.assertNotIn(forbidden, payload)
        self.assertNotIn("POSITION_CLOSED", [event["event_type"] for event in events])
        self.assertNotIn("POSITION_ADJUSTED", [event["event_type"] for event in events])


class StateStoreCloseFillTests(Phase3O3Case):
    def test_confirmed_close_intent_transitions_to_filled_without_closing_position(self) -> None:
        intent = self.create_confirmed_close_intent(S01_VOL_BASELINE, "position:state")
        updated = self.state_store.fill_close_intent(
            intent["close_intent_id"],
            filled_at=FILLED_AT,
            close_fill_confirmed_event_id="evt_close_fill",
            simulated_close_order_id=intent["simulated_close_order_id"],
            close_fill_id="close_fill_1",
            close_fill_price=0.75,
            close_fill_quantity=intent["quantity"],
        )
        self.assertEqual(updated["status"], "filled")
        self.assertEqual(updated["filled_at"], FILLED_AT)
        self.assertEqual(updated["updated_at"], FILLED_AT)
        self.assertEqual(updated["close_fill_confirmed_event_id"], "evt_close_fill")
        self.assertEqual(updated["simulated_close_order_id"], intent["simulated_close_order_id"])
        self.assertEqual(updated["close_fill_id"], "close_fill_1")
        self.assertEqual(updated["close_fill_price"], 0.75)
        self.assertEqual(updated["close_fill_quantity"], intent["quantity"])
        self.assertTrue(updated["dry_run"])
        position = self.state_store.get_position("position:state")
        self.assertEqual(position["status"], "open")
        self.assertEqual(position["active_close_intent_id"], intent["close_intent_id"])
        self.assertIn("filled", ACTIVE_CLOSE_INTENT_STATUSES)
        self.assertEqual(self.state_store.get_active_close_intent("position:state"), updated)

    def test_missing_non_confirmed_missing_fields_and_mismatches_are_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "does not exist"):
            self.state_store.fill_close_intent(
                "missing",
                filled_at=FILLED_AT,
                close_fill_confirmed_event_id="evt",
                simulated_close_order_id="sim",
                close_fill_id="fill",
                close_fill_price=1.0,
                close_fill_quantity=1,
            )
        intent = self.create_confirmed_close_intent(S01_VOL_BASELINE, "position:already")
        self.state_store.fill_close_intent(
            intent["close_intent_id"],
            filled_at=FILLED_AT,
            close_fill_confirmed_event_id="evt",
            simulated_close_order_id=intent["simulated_close_order_id"],
            close_fill_id="fill",
            close_fill_price=1.0,
            close_fill_quantity=intent["quantity"],
        )
        with self.assertRaisesRegex(ValueError, "not 'confirmed'"):
            self.state_store.fill_close_intent(
                intent["close_intent_id"],
                filled_at=FILLED_AT,
                close_fill_confirmed_event_id="evt2",
                simulated_close_order_id=intent["simulated_close_order_id"],
                close_fill_id="fill2",
                close_fill_price=1.0,
                close_fill_quantity=intent["quantity"],
            )
        for missing in ("close_order_submitted_event_id", "close_order_confirmed_event_id"):
            intent = self.create_confirmed_close_intent(S02_VOL_ENHANCED, f"position:missing:{missing}")
            self.state_store.state["close_intents"][intent["close_intent_id"]].pop(missing)
            self.state_store.save()
            with self.assertRaisesRegex(ValueError, missing):
                self.state_store.fill_close_intent(
                    intent["close_intent_id"],
                    filled_at=FILLED_AT,
                    close_fill_confirmed_event_id="evt",
                    simulated_close_order_id=intent["simulated_close_order_id"],
                    close_fill_id="fill",
                    close_fill_price=1.0,
                    close_fill_quantity=intent["quantity"],
                )
        intent = self.create_confirmed_close_intent(S01_VOL_BASELINE, "position:mismatch")
        with self.assertRaisesRegex(ValueError, "simulated_close_order_id"):
            self.state_store.fill_close_intent(
                intent["close_intent_id"],
                filled_at=FILLED_AT,
                close_fill_confirmed_event_id="evt",
                simulated_close_order_id="different",
                close_fill_id="fill",
                close_fill_price=1.0,
                close_fill_quantity=intent["quantity"],
            )
        with self.assertRaisesRegex(ValueError, "close_fill_quantity"):
            self.state_store.fill_close_intent(
                intent["close_intent_id"],
                filled_at=FILLED_AT,
                close_fill_confirmed_event_id="evt",
                simulated_close_order_id=intent["simulated_close_order_id"],
                close_fill_id="fill",
                close_fill_price=1.0,
                close_fill_quantity=intent["quantity"] + 1,
            )


class DryRunCloseFillAdapterTests(Phase3O3Case):
    def test_check_close_fills_is_pure_and_uses_simulated_close_id(self) -> None:
        intent = self.create_confirmed_close_intent(S01_VOL_BASELINE, "position:adapter")
        state_before = deepcopy(self.state_store.state)
        result = DryRunExecutionAdapter().check_close_fills(
            close_intent=intent,
            simulated_close_order_id=intent["simulated_close_order_id"],
            checked_at=FILLED_AT,
        )
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["status"], "filled")
        self.assertEqual(result["action"], "close")
        self.assertEqual(result["simulated_close_order_id"], intent["simulated_close_order_id"])
        self.assertEqual(result["close_fill_quantity"], intent["quantity"])
        self.assertEqual(result["close_fill_price"], intent["entry_price"])
        self.assertIsInstance(result["close_fill_id"], str)
        self.assertNotIn(FORBIDDEN_BROKER_ID_FIELD, result)
        for forbidden in ("target_price", "limit_price", "order_type", "time_in_force"):
            self.assertNotIn(forbidden, result)
        self.assertEqual(
            [event["event_type"] for event in self.order_events()],
            ["CLOSE_INTENT_CREATED", "CLOSE_ORDER_SUBMITTED", "CLOSE_ORDER_CONFIRMED"],
        )
        self.assertEqual(self.state_store.state, state_before)


class ConfirmCloseFillHelperTests(Phase3O3Case):
    def assert_successful_fill(self, strategy_id: str, position_id: str) -> dict:
        intent = self.create_confirmed_close_intent(strategy_id, position_id)
        updated = confirm_close_fill(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=intent["close_intent_id"],
            filled_at=FILLED_AT,
        )
        self.assertEqual(updated["status"], "filled")
        self.assertEqual(updated["close_fill_confirmed_event_id"], self.order_events()[-1]["event_id"])
        position = self.state_store.get_position(position_id)
        self.assertEqual(position["status"], "open")
        self.assertEqual(position["active_close_intent_id"], intent["close_intent_id"])
        self.assertEqual(self.order_events()[-1]["event_type"], "CLOSE_FILL_CONFIRMED")
        return updated

    def test_s01_and_s02_confirmed_close_intents_fill_successfully(self) -> None:
        self.assert_successful_fill(S01_VOL_BASELINE, "position:s01")
        self.assert_successful_fill(S02_VOL_ENHANCED, "position:s02")

    def test_validations_happen_before_adapter_or_ledger(self) -> None:
        for missing in (
            "dry_run",
            "close_intent_created_event_id",
            "close_order_submitted_event_id",
            "close_order_confirmed_event_id",
            "position_opened_event_id",
            "fill_confirmed_event_id",
            "close_order_ref",
            "simulated_close_order_id",
            "quantity",
            "entry_price",
        ):
            intent = self.create_confirmed_close_intent(S01_VOL_BASELINE, f"position:missing:{missing}")
            self.state_store.state["close_intents"][intent["close_intent_id"]].pop(missing)
            self.state_store.save()
            adapter = mock.Mock()
            before_events = self.order_events()
            with self.assertRaises((ValueError, KeyError)):
                confirm_close_fill(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    execution_adapter=adapter,
                    close_intent_id=intent["close_intent_id"],
                    filled_at=FILLED_AT,
                )
            adapter.check_close_fills.assert_not_called()
            self.assertEqual(self.order_events(), before_events)

    def test_position_validation_failures_happen_before_adapter_or_ledger(self) -> None:
        cases = [
            ("missing", lambda intent: self.state_store.state["positions"].pop(intent["position_id"])),
            ("not-open", lambda intent: self.state_store.state["positions"][intent["position_id"]].update(status="closed")),
            ("wrong-link", lambda intent: self.state_store.state["positions"][intent["position_id"]].update(active_close_intent_id="other")),
            ("strategy-mismatch", lambda intent: self.state_store.state["positions"][intent["position_id"]].update(strategy_id=S02_VOL_ENHANCED)),
            ("symbol-mismatch", lambda intent: self.state_store.state["positions"][intent["position_id"]].update(symbol="SPX")),
        ]
        for name, mutate in cases:
            intent = self.create_confirmed_close_intent(S01_VOL_BASELINE, f"position:{name}")
            before_events = self.order_events()
            mutate(intent)
            self.state_store.save()
            adapter = mock.Mock()
            with self.assertRaises((KeyError, ValueError)):
                confirm_close_fill(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    execution_adapter=adapter,
                    close_intent_id=intent["close_intent_id"],
                    filled_at=FILLED_AT,
                )
            adapter.check_close_fills.assert_not_called()
            self.assertEqual(self.order_events(), before_events)
            self.assertEqual(
                self.state_store.get_close_intent(intent["close_intent_id"])["status"],
                "confirmed",
            )

    def test_adapter_ledger_and_state_failure_ordering(self) -> None:
        intent = self.create_confirmed_close_intent(S01_VOL_BASELINE, "position:adapter-fail")
        adapter = mock.Mock()
        adapter.check_close_fills.side_effect = RuntimeError("adapter failed")
        with self.assertRaisesRegex(RuntimeError, "adapter failed"):
            confirm_close_fill(
                state_store=self.state_store,
                ledger=self.ledger,
                execution_adapter=adapter,
                close_intent_id=intent["close_intent_id"],
                filled_at=FILLED_AT,
            )
        self.assertEqual(self.order_events()[-1]["event_type"], "CLOSE_ORDER_CONFIRMED")
        self.assertEqual(self.state_store.get_close_intent(intent["close_intent_id"])["status"], "confirmed")

        intent = self.create_confirmed_close_intent(S02_VOL_ENHANCED, "position:non-filled")
        adapter.check_close_fills.side_effect = None
        adapter.check_close_fills.return_value = {"status": "pending"}
        with self.assertRaisesRegex(ValueError, "not 'filled'"):
            confirm_close_fill(
                state_store=self.state_store,
                ledger=self.ledger,
                execution_adapter=adapter,
                close_intent_id=intent["close_intent_id"],
                filled_at=FILLED_AT,
            )
        self.assertEqual(self.order_events()[-1]["event_type"], "CLOSE_ORDER_CONFIRMED")

        intent = self.create_confirmed_close_intent(S01_VOL_BASELINE, "position:id-mismatch")
        adapter.check_close_fills.return_value = {
            "status": "filled",
            "simulated_close_order_id": "different",
        }
        with self.assertRaisesRegex(ValueError, "mismatch"):
            confirm_close_fill(
                state_store=self.state_store,
                ledger=self.ledger,
                execution_adapter=adapter,
                close_intent_id=intent["close_intent_id"],
                filled_at=FILLED_AT,
            )

        for field, value in (("close_fill_price", "1.0"), ("close_fill_quantity", 0), ("close_fill_quantity", intent["quantity"] + 1)):
            intent = self.create_confirmed_close_intent(S02_VOL_ENHANCED, f"position:bad:{field}:{value}")
            adapter.check_close_fills.return_value = {
                "status": "filled",
                "simulated_close_order_id": intent["simulated_close_order_id"],
                "close_fill_id": "fill",
                "close_fill_price": 1.0,
                "close_fill_quantity": intent["quantity"],
                "filled_at": FILLED_AT,
                "checked_at": FILLED_AT,
            }
            adapter.check_close_fills.return_value[field] = value
            with self.assertRaisesRegex(ValueError, field):
                confirm_close_fill(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    execution_adapter=adapter,
                    close_intent_id=intent["close_intent_id"],
                    filled_at=FILLED_AT,
                )
            self.assertEqual(self.state_store.get_close_intent(intent["close_intent_id"])["status"], "confirmed")

        intent = self.create_confirmed_close_intent(S01_VOL_BASELINE, "position:ledger-fail")
        ledger = mock.Mock()
        ledger.append.side_effect = RuntimeError("ledger failed")
        with self.assertRaisesRegex(RuntimeError, "ledger failed"):
            confirm_close_fill(
                state_store=self.state_store,
                ledger=ledger,
                execution_adapter=DryRunExecutionAdapter(),
                close_intent_id=intent["close_intent_id"],
                filled_at=FILLED_AT,
            )
        self.assertEqual(self.state_store.get_close_intent(intent["close_intent_id"])["status"], "confirmed")

        intent = self.create_confirmed_close_intent(S02_VOL_ENHANCED, "position:state-fail")
        with mock.patch.object(
            self.state_store,
            "fill_close_intent",
            side_effect=RuntimeError("state failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "state failed"):
                confirm_close_fill(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    execution_adapter=DryRunExecutionAdapter(),
                    close_intent_id=intent["close_intent_id"],
                    filled_at=FILLED_AT,
                )
        self.assertEqual(self.order_events()[-1]["event_type"], "CLOSE_FILL_CONFIRMED")
        self.assertEqual(self.state_store.get_close_intent(intent["close_intent_id"])["status"], "confirmed")

    def test_filled_close_intent_blocks_second_close_intent_for_position(self) -> None:
        updated = self.assert_successful_fill(S01_VOL_BASELINE, "position:dupe")
        before_events = self.order_events()
        before_intents = self.state_store.list_close_intents(position_id="position:dupe")
        with self.assertRaisesRegex(ValueError, "active close intent already exists"):
            create_close_intent_from_position(
                state_store=self.state_store,
                ledger=self.ledger,
                position_id="position:dupe",
                created_at="2026-04-27T15:35:00+00:00",
            )
        self.assertEqual(self.order_events(), before_events)
        self.assertEqual(before_intents, [updated])
        self.assertEqual(self.state_store.list_close_intents(position_id="position:dupe"), [updated])

    def test_helper_holds_strategy_lock_through_adapter_ledger_and_state_transition(self) -> None:
        intent = self.create_confirmed_close_intent(S01_VOL_BASELINE, "position:lock")
        spy = LockSpy()
        self.state_store.strategy_state_locks[S01_VOL_BASELINE] = spy
        adapter = mock.Mock(wraps=DryRunExecutionAdapter())
        real_append = self.ledger.append
        real_fill = self.state_store.fill_close_intent

        def append_with_lock_check(**kwargs):
            self.assertTrue(spy.held)
            return real_append(**kwargs)

        def fill_with_lock_check(*args, **kwargs):
            self.assertTrue(spy.held)
            return real_fill(*args, **kwargs)

        adapter.check_close_fills.side_effect = lambda *args, **kwargs: (
            self.assertTrue(spy.held)
            or DryRunExecutionAdapter().check_close_fills(*args, **kwargs)
        )
        with mock.patch.object(self.ledger, "append", side_effect=append_with_lock_check):
            with mock.patch.object(
                self.state_store,
                "fill_close_intent",
                side_effect=fill_with_lock_check,
            ):
                confirm_close_fill(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    execution_adapter=adapter,
                    close_intent_id=intent["close_intent_id"],
                    filled_at=FILLED_AT,
                )


class ConfirmCloseFillCliTests(Phase3O3Case):
    def test_cli_success_human_json_z_suffix_omitted_timestamp_and_dry_run(self) -> None:
        intent = self.create_confirmed_close_intent(S01_VOL_BASELINE, "position:cli:s01")
        code, stdout, stderr = self.run_cli(
            ["--close-intent-id", intent["close_intent_id"], "--filled-at", "2026-04-27T15:30:00Z"]
        )
        self.assertEqual(code, 0, stderr)
        self.assertIn("close_intent_id", stdout)
        self.assertIn("position_id", stdout)
        self.assertIn("filled", stdout)
        self.assertIn("close_fill_confirmed_event_id", stdout)
        updated = StateStore(self.state_path).get_close_intent(intent["close_intent_id"])
        self.assertEqual(updated["filled_at"], FILLED_AT)

        intent = self.create_confirmed_close_intent(S02_VOL_ENHANCED, "position:cli:s02")
        code, stdout, stderr = self.run_cli(
            ["--close-intent-id", intent["close_intent_id"], "--json", "--dry-run"]
        )
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["close_intent_id"], intent["close_intent_id"])
        self.assertEqual(payload["position_id"], "position:cli:s02")
        self.assertEqual(payload["status"], "filled")
        self.assertEqual(payload["simulated_close_order_id"], intent["simulated_close_order_id"])
        self.assertIn("close_fill_id", payload)
        self.assertIn("close_fill_confirmed_event_id", payload)
        self.assertIs(payload["dry_run"], True)
        self.assertIsNotNone(StateStore(self.state_path).get_close_intent(intent["close_intent_id"])["filled_at"])

    def test_cli_errors_are_concise_and_do_not_write_second_fill(self) -> None:
        intent = self.create_confirmed_close_intent(S01_VOL_BASELINE, "position:cli:error")
        code, _, stderr = self.run_cli(["--close-intent-id", intent["close_intent_id"]])
        self.assertEqual(code, 0, stderr)
        before_events = self.order_events()
        code, _, stderr = self.run_cli(["--close-intent-id", intent["close_intent_id"]])
        self.assertNotEqual(code, 0)
        self.assertIn("not 'confirmed'", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertEqual(self.order_events(), before_events)

        code, _, stderr = self.run_cli(["--close-intent-id", "missing"])
        self.assertNotEqual(code, 0)
        self.assertIn("does not exist", stderr)
        self.assertNotIn("Traceback", stderr)

        intent = self.create_confirmed_close_intent(S02_VOL_ENHANCED, "position:cli:bad-field")
        self.state_store.state["close_intents"][intent["close_intent_id"]].pop("dry_run")
        self.state_store.save()
        code, _, stderr = self.run_cli(["--close-intent-id", intent["close_intent_id"]])
        self.assertNotEqual(code, 0)
        self.assertIn("dry_run", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertEqual(
            StateStore(self.state_path).get_close_intent(intent["close_intent_id"])["status"],
            "confirmed",
        )

        intent = self.create_confirmed_close_intent(S01_VOL_BASELINE, "position:cli:bad-time")
        with self.assertRaises(SystemExit):
            self.run_cli(["--close-intent-id", intent["close_intent_id"], "--filled-at", "not-a-time"])
        self.assertEqual(
            StateStore(self.state_path).get_close_intent(intent["close_intent_id"])["status"],
            "confirmed",
        )
        self.assertNotEqual(self.order_events()[-1]["event_type"], "CLOSE_FILL_CONFIRMED")


class SystemStatusFilledCloseIntentTests(Phase3O3Case):
    def test_filled_close_intents_are_counted_and_filtered(self) -> None:
        empty = system_status.build_summary([], [], [])
        self.assertEqual(empty["close_intent_counts_by_status"]["filled"], 0)
        s01 = self.create_confirmed_close_intent(S01_VOL_BASELINE, "position:status:s01")
        s02 = self.create_confirmed_close_intent(S02_VOL_ENHANCED, "position:status:s02")
        confirm_close_fill(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=s01["close_intent_id"],
            filled_at=FILLED_AT,
        )
        summary = system_status.build_summary([], [], self.state_store.list_close_intents())
        self.assertEqual(
            summary["close_intent_counts_by_status"],
            {"created": 0, "submitted": 0, "confirmed": 1, "filled": 1, "position_closed": 0},
        )
        self.assertEqual(summary["confirmed_close_intents_count"], 1)
        self.assertEqual(summary["filled_close_intents_count"], 1)
        self.assertEqual(summary["total_close_intents_count"], 2)
        filtered = system_status.build_summary(
            [],
            [],
            self.state_store.list_close_intents(),
            strategy_id=S01_VOL_BASELINE,
        )
        self.assertEqual(
            filtered["close_intent_counts_by_status"],
            {"created": 0, "submitted": 0, "confirmed": 0, "filled": 1, "position_closed": 0},
        )
        self.assertEqual(s02["status"], "confirmed")


class CloseFillSafetyTests(unittest.TestCase):
    def test_source_avoids_forbidden_live_future_and_write_surfaces(self) -> None:
        modules = (
            importlib.import_module("algo_trader_unified.core.close_intents"),
            importlib.import_module("algo_trader_unified.core.execution"),
            importlib.import_module("algo_trader_unified.tools.confirm_close_fill"),
        )
        sources = "\n".join(inspect.getsource(module) for module in modules)
        self.assertIn("confirm_close_fill", sources)
        self.assertIn("check_close_fills", sources)
        self.assertNotIn("LedgerAppender.append", inspect.getsource(modules[2]))
        for needle in (
            "ib_insync",
            "yfinance",
            "requests",
            "broker.submit_order",
            "placeOrder",
            "cancelOrder",
            "scheduler.start",
            "except:",
            "POSITION_ADJUSTED",
            "POSITION_CLOSED",
            FORBIDDEN_BROKER_ID_FIELD,
            "target_price",
            "limit_price",
            "order_type",
            "time_in_force",
        ):
            self.assertNotIn(needle, sources)


if __name__ == "__main__":
    unittest.main()
