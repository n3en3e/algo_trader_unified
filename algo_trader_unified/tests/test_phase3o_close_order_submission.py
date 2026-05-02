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
    create_close_intent_from_position,
    submit_close_intent,
)
from algo_trader_unified.core.execution import DryRunExecutionAdapter
from algo_trader_unified.core.ledger import KNOWN_EVENT_TYPES, LedgerAppender
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.tools import submit_close_intent as submit_tool
from algo_trader_unified.tools import system_status


CREATED_AT = "2026-04-27T15:10:00+00:00"
SUBMITTED_AT = "2026-04-27T15:20:00+00:00"
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


class Phase3OCase(unittest.TestCase):
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

    def create_close_intent(self, strategy_id: str, position_id: str) -> dict:
        self.create_position(strategy_id, position_id)
        return create_close_intent_from_position(
            state_store=self.state_store,
            ledger=self.ledger,
            position_id=position_id,
            created_at=CREATED_AT,
        )

    def order_events(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.order_path.read_text(encoding="utf-8").splitlines()
            if line
        ]


class CloseOrderSubmittedEventTests(Phase3OCase):
    def test_event_routes_to_order_ledger_only_and_payload_is_scaffold_only(self) -> None:
        self.assertIn("CLOSE_ORDER_SUBMITTED", KNOWN_EVENT_TYPES)
        self.assertEqual(
            self.ledger.path_for_event_type("CLOSE_ORDER_SUBMITTED").name,
            "order_ledger.jsonl",
        )
        intent = self.create_close_intent(S01_VOL_BASELINE, "position:event")
        submit_close_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=intent["close_intent_id"],
            submitted_at=SUBMITTED_AT,
        )
        events = self.order_events()
        self.assertEqual(
            [event["event_type"] for event in events],
            ["CLOSE_INTENT_CREATED", "CLOSE_ORDER_SUBMITTED"],
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
            "submitted_at",
            "close_reason",
            "requested_by",
            "close_intent_created_event_id",
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
        self.assertGreater(payload["quantity"], 0)
        self.assertGreaterEqual(payload["entry_price"], 0)
        self.assertEqual(payload["action"], "close")
        self.assertEqual(payload["status"], "submitted")
        self.assertEqual(payload["event_detail"], "CLOSE_ORDER_SUBMITTED")
        self.assertIn("simulated_close_order_id", payload)
        for forbidden in (
            FORBIDDEN_BROKER_ID_FIELD,
            "target_price",
            "limit_price",
            "order_type",
            "time_in_force",
        ):
            self.assertNotIn(forbidden, payload)
        self.assertNotIn("CLOSE_ORDER_CONFIRMED", self.order_path.read_text())
        self.assertNotIn("CLOSE_FILL_CONFIRMED", self.order_path.read_text())
        self.assertNotIn("POSITION_CLOSED", self.order_path.read_text())
        self.assertNotIn("POSITION_ADJUSTED", self.order_path.read_text())


class StateStoreCloseSubmitTests(Phase3OCase):
    def test_created_close_intent_transitions_to_submitted_without_closing_position(self) -> None:
        intent = self.create_close_intent(S01_VOL_BASELINE, "position:state")
        updated = self.state_store.submit_close_intent(
            intent["close_intent_id"],
            submitted_at=SUBMITTED_AT,
            close_order_submitted_event_id="evt_close_submit",
            simulated_close_order_id="sim_close_123",
            close_order_ref="ref-close",
        )
        self.assertEqual(updated["status"], "submitted")
        self.assertEqual(updated["submitted_at"], SUBMITTED_AT)
        self.assertEqual(updated["updated_at"], SUBMITTED_AT)
        self.assertEqual(updated["close_order_submitted_event_id"], "evt_close_submit")
        self.assertEqual(updated["simulated_close_order_id"], "sim_close_123")
        self.assertEqual(updated["close_order_ref"], "ref-close")
        self.assertTrue(updated["dry_run"])
        position = self.state_store.get_position("position:state")
        self.assertEqual(position["status"], "open")
        self.assertEqual(position["active_close_intent_id"], intent["close_intent_id"])

    def test_missing_and_non_created_close_intents_are_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "does not exist"):
            self.state_store.submit_close_intent(
                "missing",
                submitted_at=SUBMITTED_AT,
                close_order_submitted_event_id="evt",
                simulated_close_order_id="sim",
                close_order_ref="ref",
            )
        intent = self.create_close_intent(S01_VOL_BASELINE, "position:once")
        self.state_store.submit_close_intent(
            intent["close_intent_id"],
            submitted_at=SUBMITTED_AT,
            close_order_submitted_event_id="evt",
            simulated_close_order_id="sim",
            close_order_ref="ref",
        )
        with self.assertRaisesRegex(ValueError, "not 'created'"):
            self.state_store.submit_close_intent(
                intent["close_intent_id"],
                submitted_at=SUBMITTED_AT,
                close_order_submitted_event_id="evt2",
                simulated_close_order_id="sim2",
                close_order_ref="ref2",
            )


class DryRunCloseAdapterTests(Phase3OCase):
    def test_submit_close_intent_is_pure_and_uses_simulated_close_id(self) -> None:
        intent = self.create_close_intent(S01_VOL_BASELINE, "position:adapter")
        state_before = deepcopy(self.state_store.state)
        result = DryRunExecutionAdapter().submit_close_intent(
            intent,
            submitted_at=SUBMITTED_AT,
        )
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["close_intent_id"], intent["close_intent_id"])
        self.assertEqual(result["position_id"], intent["position_id"])
        self.assertEqual(result["strategy_id"], intent["strategy_id"])
        self.assertEqual(result["symbol"], intent["symbol"])
        self.assertEqual(result["submitted_at"], SUBMITTED_AT)
        self.assertEqual(result["status"], "submitted")
        self.assertEqual(result["action"], "close")
        self.assertIn("simulated_close_order_id", result)
        self.assertNotIn(FORBIDDEN_BROKER_ID_FIELD, result)
        for forbidden in ("target_price", "limit_price", "order_type", "time_in_force"):
            self.assertNotIn(forbidden, result)
        self.assertEqual([event["event_type"] for event in self.order_events()], ["CLOSE_INTENT_CREATED"])
        self.assertEqual(self.state_store.state, state_before)


class SubmitCloseIntentHelperTests(Phase3OCase):
    def assert_successful_submit(self, strategy_id: str, position_id: str) -> dict:
        intent = self.create_close_intent(strategy_id, position_id)
        updated = submit_close_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=intent["close_intent_id"],
            submitted_at=SUBMITTED_AT,
        )
        self.assertEqual(updated["status"], "submitted")
        self.assertEqual(updated["close_order_submitted_event_id"], self.order_events()[-1]["event_id"])
        self.assertEqual(self.state_store.get_position(position_id)["status"], "open")
        self.assertEqual(
            [event["event_type"] for event in self.order_events()[-2:]],
            ["CLOSE_INTENT_CREATED", "CLOSE_ORDER_SUBMITTED"],
        )
        return updated

    def test_s01_and_s02_created_close_intents_submit_successfully(self) -> None:
        self.assert_successful_submit(S01_VOL_BASELINE, "position:s01")
        self.ledger = LedgerAppender(self.root)
        self.assert_successful_submit(S02_VOL_ENHANCED, "position:s02")

    def test_submitted_close_intent_blocks_second_close_intent_for_position(self) -> None:
        intent = self.create_close_intent(S01_VOL_BASELINE, "position:submitted-dupe")
        updated = submit_close_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=intent["close_intent_id"],
            submitted_at=SUBMITTED_AT,
        )
        before_events = self.order_events()
        before_intent = self.state_store.get_close_intent(intent["close_intent_id"])

        with self.assertRaisesRegex(ValueError, "active close intent already exists"):
            create_close_intent_from_position(
                state_store=self.state_store,
                ledger=self.ledger,
                position_id="position:submitted-dupe",
                created_at="2026-04-27T15:30:00+00:00",
            )

        self.assertEqual(self.order_events(), before_events)
        self.assertEqual(
            self.state_store.list_close_intents(position_id="position:submitted-dupe"),
            [before_intent],
        )
        self.assertEqual(before_intent, updated)

    def test_required_fields_and_dry_run_are_validated_before_adapter(self) -> None:
        for missing in (
            "dry_run",
            "close_intent_created_event_id",
            "position_opened_event_id",
            "fill_confirmed_event_id",
            "quantity",
            "entry_price",
        ):
            intent = self.create_close_intent(S01_VOL_BASELINE, f"position:missing:{missing}")
            self.state_store.state["close_intents"][intent["close_intent_id"]].pop(missing)
            self.state_store.save()
            adapter = mock.Mock()
            with self.assertRaises((ValueError, KeyError)):
                submit_close_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    execution_adapter=adapter,
                    close_intent_id=intent["close_intent_id"],
                    submitted_at=SUBMITTED_AT,
                )
            adapter.submit_close_intent.assert_not_called()

    def test_position_validation_failures_happen_before_adapter_or_ledger(self) -> None:
        cases = [
            ("missing", lambda intent: self.state_store.state["positions"].pop(intent["position_id"])),
            ("not-open", lambda intent: self.state_store.state["positions"][intent["position_id"]].update(status="closed")),
            ("wrong-link", lambda intent: self.state_store.state["positions"][intent["position_id"]].update(active_close_intent_id="other")),
            ("strategy-mismatch", lambda intent: self.state_store.state["positions"][intent["position_id"]].update(strategy_id=S02_VOL_ENHANCED)),
            ("symbol-mismatch", lambda intent: self.state_store.state["positions"][intent["position_id"]].update(symbol="SPX")),
        ]
        for name, mutate in cases:
            intent = self.create_close_intent(S01_VOL_BASELINE, f"position:{name}")
            before_events = self.order_events()
            mutate(intent)
            self.state_store.save()
            adapter = mock.Mock()
            with self.assertRaises((KeyError, ValueError)):
                submit_close_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    execution_adapter=adapter,
                    close_intent_id=intent["close_intent_id"],
                    submitted_at=SUBMITTED_AT,
                )
            adapter.submit_close_intent.assert_not_called()
            self.assertEqual(self.order_events(), before_events)
            self.assertEqual(
                self.state_store.get_close_intent(intent["close_intent_id"])["status"],
                "created",
            )

    def test_adapter_and_ledger_failures_leave_close_intent_created(self) -> None:
        intent = self.create_close_intent(S01_VOL_BASELINE, "position:adapter-fail")
        adapter = mock.Mock()
        adapter.submit_close_intent.side_effect = RuntimeError("adapter failed")
        with self.assertRaisesRegex(RuntimeError, "adapter failed"):
            submit_close_intent(
                state_store=self.state_store,
                ledger=self.ledger,
                execution_adapter=adapter,
                close_intent_id=intent["close_intent_id"],
                submitted_at=SUBMITTED_AT,
            )
        self.assertEqual([event["event_type"] for event in self.order_events()], ["CLOSE_INTENT_CREATED"])
        self.assertEqual(self.state_store.get_close_intent(intent["close_intent_id"])["status"], "created")

        intent = self.create_close_intent(S02_VOL_ENHANCED, "position:ledger-fail")
        ledger = mock.Mock()
        ledger.append.side_effect = RuntimeError("ledger failed")
        with self.assertRaisesRegex(RuntimeError, "ledger failed"):
            submit_close_intent(
                state_store=self.state_store,
                ledger=ledger,
                execution_adapter=DryRunExecutionAdapter(),
                close_intent_id=intent["close_intent_id"],
                submitted_at=SUBMITTED_AT,
            )
        self.assertEqual(self.state_store.get_close_intent(intent["close_intent_id"])["status"], "created")

    def test_adapter_non_submitted_and_state_failure_ordering(self) -> None:
        intent = self.create_close_intent(S01_VOL_BASELINE, "position:non-submitted")
        adapter = mock.Mock()
        adapter.submit_close_intent.return_value = {"status": "pending"}
        with self.assertRaisesRegex(ValueError, "not 'submitted'"):
            submit_close_intent(
                state_store=self.state_store,
                ledger=self.ledger,
                execution_adapter=adapter,
                close_intent_id=intent["close_intent_id"],
                submitted_at=SUBMITTED_AT,
            )
        self.assertEqual([event["event_type"] for event in self.order_events()], ["CLOSE_INTENT_CREATED"])

        intent = self.create_close_intent(S02_VOL_ENHANCED, "position:state-fail")
        with mock.patch.object(
            self.state_store,
            "submit_close_intent",
            side_effect=RuntimeError("state failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "state failed"):
                submit_close_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    execution_adapter=DryRunExecutionAdapter(),
                    close_intent_id=intent["close_intent_id"],
                    submitted_at=SUBMITTED_AT,
                )
        self.assertEqual(self.order_events()[-1]["event_type"], "CLOSE_ORDER_SUBMITTED")
        self.assertEqual(self.state_store.get_close_intent(intent["close_intent_id"])["status"], "created")

    def test_helper_holds_strategy_lock_through_adapter_ledger_and_state_transition(self) -> None:
        intent = self.create_close_intent(S01_VOL_BASELINE, "position:lock")
        spy = LockSpy()
        self.state_store.strategy_state_locks[S01_VOL_BASELINE] = spy
        adapter = mock.Mock(wraps=DryRunExecutionAdapter())
        real_append = self.ledger.append
        real_submit = self.state_store.submit_close_intent

        def append_with_lock_check(**kwargs):
            self.assertTrue(spy.held)
            return real_append(**kwargs)

        def submit_with_lock_check(*args, **kwargs):
            self.assertTrue(spy.held)
            return real_submit(*args, **kwargs)

        adapter.submit_close_intent.side_effect = lambda *args, **kwargs: (
            self.assertTrue(spy.held) or DryRunExecutionAdapter().submit_close_intent(*args, **kwargs)
        )
        with mock.patch.object(self.ledger, "append", side_effect=append_with_lock_check):
            with mock.patch.object(self.state_store, "submit_close_intent", side_effect=submit_with_lock_check):
                submit_close_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    execution_adapter=adapter,
                    close_intent_id=intent["close_intent_id"],
                    submitted_at=SUBMITTED_AT,
                )


class SubmitCloseIntentCliTests(Phase3OCase):
    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = submit_tool.main(["--root-dir", str(self.root), *argv])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_success_human_and_json(self) -> None:
        intent = self.create_close_intent(S01_VOL_BASELINE, "position:cli:s01")
        code, stdout, stderr = self.run_cli(
            ["--close-intent-id", intent["close_intent_id"], "--submitted-at", "2026-04-27T15:20:00Z"]
        )
        self.assertEqual(code, 0, stderr)
        self.assertIn("close_intent_id", stdout)
        self.assertIn("position_id", stdout)
        self.assertIn("submitted", stdout)
        self.assertIn("close_order_submitted_event_id", stdout)
        updated = StateStore(self.state_path).get_close_intent(intent["close_intent_id"])
        self.assertEqual(updated["submitted_at"], SUBMITTED_AT)

        intent = self.create_close_intent(S02_VOL_ENHANCED, "position:cli:s02")
        code, stdout, stderr = self.run_cli(
            ["--close-intent-id", intent["close_intent_id"], "--json", "--dry-run"]
        )
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["close_intent_id"], intent["close_intent_id"])
        self.assertEqual(payload["position_id"], "position:cli:s02")
        self.assertEqual(payload["status"], "submitted")
        self.assertIn("simulated_close_order_id", payload)
        self.assertIn("close_order_submitted_event_id", payload)
        self.assertIs(payload["dry_run"], True)

    def test_cli_errors_do_not_write_second_submission(self) -> None:
        intent = self.create_close_intent(S01_VOL_BASELINE, "position:cli:error")
        code, _, stderr = self.run_cli(["--close-intent-id", intent["close_intent_id"]])
        self.assertEqual(code, 0, stderr)
        before_events = self.order_events()
        code, _, stderr = self.run_cli(["--close-intent-id", intent["close_intent_id"]])
        self.assertNotEqual(code, 0)
        self.assertIn("not 'created'", stderr)
        self.assertEqual(self.order_events(), before_events)

        intent = self.create_close_intent(S02_VOL_ENHANCED, "position:cli:bad-time")
        with self.assertRaises(SystemExit):
            self.run_cli(["--close-intent-id", intent["close_intent_id"], "--submitted-at", "not-a-time"])
        self.assertEqual(
            StateStore(self.state_path).get_close_intent(intent["close_intent_id"])["status"],
            "created",
        )


class SystemStatusSubmittedCloseIntentTests(Phase3OCase):
    def test_submitted_close_intents_are_counted_and_filtered(self) -> None:
        s01 = self.create_close_intent(S01_VOL_BASELINE, "position:status:s01")
        s02 = self.create_close_intent(S02_VOL_ENHANCED, "position:status:s02")
        submit_close_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=s01["close_intent_id"],
            submitted_at=SUBMITTED_AT,
        )
        summary = system_status.build_summary(
            [],
            [],
            self.state_store.list_close_intents(),
        )
        self.assertEqual(
            summary["close_intent_counts_by_status"],
            {"created": 1, "submitted": 1, "confirmed": 0, "filled": 0},
        )
        self.assertEqual(summary["created_close_intents_count"], 1)
        self.assertEqual(summary["submitted_close_intents_count"], 1)
        self.assertEqual(summary["confirmed_close_intents_count"], 0)
        self.assertEqual(summary["total_close_intents_count"], 2)
        filtered = system_status.build_summary(
            [],
            [],
            self.state_store.list_close_intents(),
            strategy_id=S01_VOL_BASELINE,
        )
        self.assertEqual(
            filtered["close_intent_counts_by_status"],
            {"created": 0, "submitted": 1, "confirmed": 0, "filled": 0},
        )
        self.assertEqual(s02["status"], "created")


class CloseSubmissionSafetyTests(unittest.TestCase):
    def test_source_avoids_forbidden_live_and_future_surfaces(self) -> None:
        modules = (
            importlib.import_module("algo_trader_unified.core.close_intents"),
            importlib.import_module("algo_trader_unified.core.execution"),
            importlib.import_module("algo_trader_unified.tools.submit_close_intent"),
        )
        sources = "\n".join(inspect.getsource(module) for module in modules)
        self.assertIn("submit_close_intent", sources)
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
            "CLOSE_FILL_CONFIRMED",
            "target_price",
            "limit_price",
            "order_type",
            "time_in_force",
        ):
            self.assertNotIn(needle, sources)


if __name__ == "__main__":
    unittest.main()
