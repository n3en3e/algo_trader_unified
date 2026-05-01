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
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.ledger import KNOWN_EVENT_TYPES, LedgerAppender
from algo_trader_unified.core.ledger_reader import LedgerReader
from algo_trader_unified.core.order_intents import (
    confirm_fill,
    confirm_order_intent,
    submit_order_intent,
)
from algo_trader_unified.core.positions import open_position_from_filled_intent
from algo_trader_unified.core.readiness import ReadinessManager, ReadinessStatus
from algo_trader_unified.core.state_store import PositionBook, StateStore
from algo_trader_unified.core.validation import validate_numeric_field
from algo_trader_unified.jobs.vol import run_s01_vol_scan
from algo_trader_unified.strategies.vol.signals import VolSignalInput
from algo_trader_unified.tools import confirm_fill as fill_tool
from algo_trader_unified.tools import confirm_order_intent as confirm_tool
from algo_trader_unified.tools import open_position_from_intent as open_tool
from algo_trader_unified.tools import submit_order_intent as submit_tool


SUBMITTED_AT = "2026-04-27T14:00:00+00:00"
CONFIRMED_AT = "2026-04-27T14:05:00+00:00"
FILLED_AT = "2026-04-27T14:06:00+00:00"
OPENED_AT = "2026-04-27T14:07:00+00:00"


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


class Phase3KCase(unittest.TestCase):
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

    def create_intent(self, strategy_id: str, intent_id: str) -> dict:
        return self.state_store.create_order_intent(
            {
                "intent_id": intent_id,
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
                "sizing_context": {"target": 1},
                "risk_context": {"risk": "dry"},
                "signal_payload_snapshot": {},
                "dry_run": True,
            }
        )

    def create_filled_intent(self, strategy_id: str, intent_id: str) -> dict:
        self.create_intent(strategy_id, intent_id)
        submitted = self.state_store.submit_order_intent(
            intent_id,
            submitted_at=SUBMITTED_AT,
            order_submitted_event_id="evt_submitted",
            simulated_order_id=f"sim_{strategy_id}_{intent_id}",
        )
        confirmed = self.state_store.confirm_order_intent(
            intent_id,
            confirmed_at=CONFIRMED_AT,
            order_confirmed_event_id="evt_confirmed",
            simulated_order_id=submitted["simulated_order_id"],
        )
        return self.state_store.fill_order_intent(
            intent_id,
            filled_at=FILLED_AT,
            fill_confirmed_event_id="evt_fill",
            simulated_order_id=confirmed["simulated_order_id"],
            fill_id=f"fill_{strategy_id}_{intent_id}",
            fill_price=0.0,
            fill_quantity=1,
        )

    def position_record(self, strategy_id: str = S01_VOL_BASELINE, position_id: str = "pos:1") -> dict:
        return {
            "position_id": position_id,
            "intent_id": "intent:1",
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "status": "open",
            "execution_mode": "paper_only",
            "dry_run": True,
            "opened_at": OPENED_AT,
            "updated_at": OPENED_AT,
            "source_signal_event_id": "evt_signal",
            "order_intent_created_event_id": "evt_created",
            "order_submitted_event_id": "evt_submitted",
            "order_confirmed_event_id": "evt_confirmed",
            "fill_confirmed_event_id": "evt_fill",
            "position_opened_event_id": "evt_opened",
            "order_ref": "ref",
            "simulated_order_id": "sim",
            "fill_id": "fill",
            "entry_price": 0.0,
            "quantity": 1,
            "action": "open",
            "sizing_context": {},
            "risk_context": {},
        }

    def execution_events(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.execution_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def run_open_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = open_tool.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

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


class ValidationPreworkTests(Phase3KCase):
    def test_validation_module_is_canonical(self) -> None:
        self.assertEqual(validate_numeric_field("quantity", 1, minimum=0, allow_equal=False, allow_int=True), 1)
        self.assertEqual(validate_numeric_field("price", 0.0, minimum=0, allow_equal=True, allow_int=False), 0.0)
        state_source = inspect.getsource(importlib.import_module("algo_trader_unified.core.state_store"))
        intents_source = inspect.getsource(importlib.import_module("algo_trader_unified.core.order_intents"))
        self.assertIn("from algo_trader_unified.core.validation import validate_numeric_field", state_source)
        self.assertIn("from algo_trader_unified.core.validation import validate_numeric_field", intents_source)
        self.assertNotIn("from algo_trader_unified.core.state_store import _validate_numeric_field", intents_source)
        self.assertNotIn("def _validate_numeric_field", state_source)


class StateStorePositionTests(Phase3KCase):
    def test_fresh_and_legacy_state_have_positions_mapping(self) -> None:
        self.assertEqual(self.state_store.state["positions"], {})
        legacy_path = self.root / "legacy_state.json"
        legacy_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "opportunities": [],
                    "orders": [],
                    "order_intents": {},
                    "fills": [],
                    "strategy_snapshots": [],
                    "account_snapshots": [],
                    "reconciliation_snapshots": [],
                    "halt_state": None,
                    "readiness": {"strategies": {}},
                }
            ),
            encoding="utf-8",
        )
        loaded = StateStore(legacy_path)
        self.assertEqual(loaded.state["positions"], {})

    def test_create_get_and_list_open_position(self) -> None:
        record = self.position_record()
        created = self.state_store.create_open_position(record)
        self.assertEqual(created["status"], "open")
        self.assertEqual(self.state_store.get_position("pos:1")["position_id"], "pos:1")
        self.assertEqual(self.state_store.get_open_position(S01_VOL_BASELINE)["position_id"], "pos:1")
        self.assertIsNone(self.state_store.get_open_position(S02_VOL_ENHANCED))
        self.assertEqual(len(self.state_store.list_positions(S01_VOL_BASELINE)), 1)
        self.assertEqual(len(self.state_store.list_positions(status="open")), 1)
        self.assertEqual(self.state_store.list_positions(status="closed"), [])
        with self.assertRaisesRegex(ValueError, "open position already exists"):
            self.state_store.create_open_position(self.position_record(position_id="pos:2"))

    def test_get_position_is_read_pure(self) -> None:
        self.state_store.create_open_position(self.position_record())
        before = deepcopy(self.state_store.state)
        self.assertEqual(self.state_store.get_position("pos:1")["position_id"], "pos:1")
        self.assertEqual(self.state_store.state, before)

    def test_get_missing_position_is_read_pure(self) -> None:
        self.state_store.create_open_position(self.position_record())
        before = deepcopy(self.state_store.state)
        self.assertIsNone(self.state_store.get_position("missing"))
        self.assertEqual(self.state_store.state, before)

    def test_position_book_iteration_is_documented_legacy_behavior(self) -> None:
        book = PositionBook()
        book["pos:1"] = {"position_id": "pos:1"}
        book["pos:2"] = {"position_id": "pos:2"}
        self.assertEqual(
            [position["position_id"] for position in list(book)],
            ["pos:1", "pos:2"],
        )
        self.assertEqual(list(book.keys()), ["pos:1", "pos:2"])
        encoded = json.loads(json.dumps(book, sort_keys=True))
        self.assertEqual(set(encoded), {"pos:1", "pos:2"})

    def test_position_numeric_validation(self) -> None:
        cases = [
            ("entry_price", "0.0"),
            ("entry_price", True),
            ("entry_price", 0),
            ("quantity", "1"),
            ("quantity", True),
            ("quantity", 0),
        ]
        for field, value in cases:
            record = self.position_record(position_id=f"pos:{field}:{value}")
            record[field] = value
            with self.assertRaisesRegex(ValueError, field):
                self.state_store.create_open_position(record)


class IntentPositionTransitionTests(Phase3KCase):
    def test_mark_intent_position_opened_transitions_filled_intent(self) -> None:
        self.create_filled_intent(S01_VOL_BASELINE, "s01:filled")
        updated = self.state_store.mark_intent_position_opened(
            "s01:filled",
            position_id="pos:1",
            position_opened_event_id="evt_opened",
            opened_at=OPENED_AT,
        )
        self.assertEqual(updated["status"], "position_opened")
        self.assertEqual(updated["position_id"], "pos:1")
        self.assertEqual(updated["position_opened_event_id"], "evt_opened")
        self.assertEqual(updated["position_opened_at"], OPENED_AT)
        self.assertIsNone(self.state_store.get_active_order_intent(S01_VOL_BASELINE))

    def test_mark_intent_position_opened_rejects_non_filled_statuses(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "s01:created")
        self.create_intent(S01_VOL_BASELINE, "s01:submitted")
        self.state_store.submit_order_intent(
            "s01:submitted",
            submitted_at=SUBMITTED_AT,
            order_submitted_event_id="evt_sub",
            simulated_order_id="sim_sub",
        )
        self.create_intent(S01_VOL_BASELINE, "s01:confirmed")
        self.state_store.submit_order_intent("s01:confirmed", submitted_at=SUBMITTED_AT, order_submitted_event_id="evt_sub", simulated_order_id="sim")
        self.state_store.confirm_order_intent("s01:confirmed", confirmed_at=CONFIRMED_AT, order_confirmed_event_id="evt_conf", simulated_order_id="sim")
        self.create_intent(S01_VOL_BASELINE, "s01:expired")
        self.state_store.expire_order_intent("s01:expired", expired_at=OPENED_AT, expire_reason="ttl", expired_event_id="evt_exp")
        self.create_intent(S01_VOL_BASELINE, "s01:cancelled")
        self.state_store.cancel_order_intent("s01:cancelled", cancelled_at=OPENED_AT, cancel_reason="operator", cancelled_event_id="evt_cancel")
        self.create_filled_intent(S01_VOL_BASELINE, "s01:opened")
        self.state_store.mark_intent_position_opened(
            "s01:opened",
            position_id="pos:opened",
            position_opened_event_id="evt_opened",
            opened_at=OPENED_AT,
        )
        with self.assertRaisesRegex(KeyError, "does not exist"):
            self.state_store.mark_intent_position_opened(
                "missing",
                position_id="pos",
                position_opened_event_id="evt",
                opened_at=OPENED_AT,
            )
        for intent_id in ("s01:created", "s01:submitted", "s01:confirmed", "s01:expired", "s01:cancelled", "s01:opened"):
            with self.assertRaisesRegex(ValueError, "not 'filled'"):
                self.state_store.mark_intent_position_opened(
                    intent_id,
                    position_id="pos",
                    position_opened_event_id="evt",
                    opened_at=OPENED_AT,
                )


class OpenPositionHelperTests(Phase3KCase):
    def assert_successful_open(self, strategy_id: str, intent_id: str) -> dict:
        filled = self.create_filled_intent(strategy_id, intent_id)
        position = open_position_from_filled_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            intent_id=intent_id,
            opened_at=OPENED_AT,
        )
        events = self.execution_events()
        self.assertEqual([event["event_type"] for event in events], ["POSITION_OPENED"])
        payload = events[0]["payload"]
        for field in (
            "position_id",
            "intent_id",
            "strategy_id",
            "sleeve_id",
            "symbol",
            "status",
            "execution_mode",
            "dry_run",
            "opened_at",
            "source_signal_event_id",
            "order_intent_created_event_id",
            "order_submitted_event_id",
            "order_confirmed_event_id",
            "fill_confirmed_event_id",
            "order_ref",
            "simulated_order_id",
            "fill_id",
            "entry_price",
            "quantity",
            "action",
            "event_detail",
            "sizing_context",
            "risk_context",
        ):
            self.assertIn(field, payload)
        self.assertIs(payload["dry_run"], True)
        self.assertIsInstance(payload["entry_price"], float)
        self.assertIsInstance(payload["quantity"], (int, float))
        self.assertEqual(position["position_opened_event_id"], events[0]["event_id"])
        self.assertEqual(position["order_ref"], filled["order_ref"])
        self.assertEqual(position["entry_price"], filled["fill_price"])
        self.assertEqual(position["quantity"], filled["fill_quantity"])
        self.assertEqual(position["position_id"], f"{strategy_id}:{intent_id}:open")
        self.assertEqual(self.state_store.get_order_intent(intent_id)["status"], "position_opened")
        self.assertEqual(self.state_store.get_order_intent(intent_id)["position_id"], position["position_id"])
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")
        self.assertNotIn("POSITION_ADJUSTED", self.execution_path.read_text(encoding="utf-8"))
        self.assertNotIn("POSITION_CLOSED", self.execution_path.read_text(encoding="utf-8"))
        return position

    def test_filled_s01_and_s02_intents_open_positions(self) -> None:
        s01 = self.assert_successful_open(S01_VOL_BASELINE, "same")
        self.execution_path.write_text("", encoding="utf-8")
        s02 = self.assert_successful_open(S02_VOL_ENHANCED, "same")
        self.assertNotEqual(s01["position_id"], s02["position_id"])

    def test_missing_optional_context_defaults_to_empty_dict(self) -> None:
        self.create_filled_intent(S01_VOL_BASELINE, "s01:no-context")
        intent = self.state_store.state["order_intents"]["s01:no-context"]
        del intent["sizing_context"]
        del intent["risk_context"]
        position = open_position_from_filled_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            intent_id="s01:no-context",
            opened_at=OPENED_AT,
        )
        self.assertEqual(position["sizing_context"], {})
        self.assertEqual(position["risk_context"], {})
        self.assertEqual(self.execution_events()[0]["payload"]["sizing_context"], {})
        self.assertEqual(self.execution_events()[0]["payload"]["risk_context"], {})

    def test_missing_required_fields_and_invalid_numbers_raise(self) -> None:
        required = (
            "dry_run",
            "source_signal_event_id",
            "order_intent_created_event_id",
            "order_submitted_event_id",
            "order_confirmed_event_id",
            "fill_confirmed_event_id",
            "simulated_order_id",
            "fill_id",
            "fill_price",
            "fill_quantity",
            "order_ref",
        )
        for field in required:
            intent_id = f"s01:missing:{field}"
            self.create_filled_intent(S01_VOL_BASELINE, intent_id)
            del self.state_store.state["order_intents"][intent_id][field]
            with self.assertRaisesRegex(ValueError, field if field != "dry_run" else "intent.dry_run"):
                open_position_from_filled_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    intent_id=intent_id,
                    opened_at=OPENED_AT,
                )
            self.assertEqual(self.execution_path.read_text(encoding="utf-8"), "")
            self.assertIsNone(self.state_store.get_position(f"{S01_VOL_BASELINE}:{intent_id}:open"))
            self.assertEqual(self.state_store.get_order_intent(intent_id)["status"], "filled")
        invalids = [("fill_price", "0.0"), ("fill_quantity", "1"), ("fill_price", True), ("fill_quantity", True)]
        for field, value in invalids:
            intent_id = f"s01:invalid:{field}:{value}"
            self.create_filled_intent(S01_VOL_BASELINE, intent_id)
            self.state_store.state["order_intents"][intent_id][field] = value
            with self.assertRaisesRegex(ValueError, field):
                open_position_from_filled_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    intent_id=intent_id,
                    opened_at=OPENED_AT,
                )
            self.assertEqual(self.execution_path.read_text(encoding="utf-8"), "")
            self.assertIsNone(self.state_store.get_position(f"{S01_VOL_BASELINE}:{intent_id}:open"))
            self.assertEqual(self.state_store.get_order_intent(intent_id)["status"], "filled")

    def test_existing_open_position_blocks_open(self) -> None:
        self.create_filled_intent(S01_VOL_BASELINE, "s01:first")
        open_position_from_filled_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            intent_id="s01:first",
            opened_at=OPENED_AT,
        )
        self.create_filled_intent(S01_VOL_BASELINE, "s01:second")
        with self.assertRaisesRegex(ValueError, "open position already exists"):
            open_position_from_filled_intent(
                state_store=self.state_store,
                ledger=self.ledger,
                intent_id="s01:second",
                opened_at=OPENED_AT,
            )

    def test_failure_ordering(self) -> None:
        self.create_filled_intent(S01_VOL_BASELINE, "s01:ledger-fail")
        before = deepcopy(self.state_store.state)
        ledger = mock.Mock()
        ledger.append.side_effect = RuntimeError("ledger failed")
        with self.assertRaisesRegex(RuntimeError, "ledger failed"):
            open_position_from_filled_intent(
                state_store=self.state_store,
                ledger=ledger,
                intent_id="s01:ledger-fail",
                opened_at=OPENED_AT,
            )
        self.assertEqual(self.state_store.state, before)

        self.create_filled_intent(S02_VOL_ENHANCED, "s02:create-fail")
        with mock.patch.object(self.state_store, "create_open_position", side_effect=RuntimeError("create failed")):
            with self.assertRaisesRegex(RuntimeError, "create failed"):
                open_position_from_filled_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    intent_id="s02:create-fail",
                    opened_at=OPENED_AT,
                )
        self.assertIn("POSITION_OPENED", self.execution_path.read_text(encoding="utf-8"))
        self.assertEqual(self.state_store.get_order_intent("s02:create-fail")["status"], "filled")

        self.create_filled_intent(S02_VOL_ENHANCED, "s02:mark-fail")
        with mock.patch.object(self.state_store, "mark_intent_position_opened", side_effect=RuntimeError("mark failed")):
            with self.assertRaisesRegex(RuntimeError, "mark failed"):
                open_position_from_filled_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    intent_id="s02:mark-fail",
                    opened_at=OPENED_AT,
                )
        self.assertIsNotNone(self.state_store.get_position(f"{S02_VOL_ENHANCED}:s02:mark-fail:open"))
        self.assertNotIn("POSITION_ADJUSTED", self.execution_path.read_text(encoding="utf-8"))
        self.assertNotIn("POSITION_CLOSED", self.execution_path.read_text(encoding="utf-8"))

    def test_open_position_sequence_runs_under_strategy_lock(self) -> None:
        self.create_filled_intent(S01_VOL_BASELINE, "s01:lock")
        spy = LockSpy()
        self.state_store.strategy_state_locks[S01_VOL_BASELINE] = spy
        observed = {"ledger": False, "create": False}
        real_append = self.ledger.append
        real_create = self.state_store.create_open_position

        def append(**kwargs):
            observed["ledger"] = True
            self.assertTrue(spy.held)
            return real_append(**kwargs)

        def create(position_record):
            observed["create"] = True
            self.assertTrue(spy.held)
            return real_create(position_record)

        with mock.patch.object(self.ledger, "append", side_effect=append):
            with mock.patch.object(self.state_store, "create_open_position", side_effect=create):
                open_position_from_filled_intent(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    intent_id="s01:lock",
                    opened_at=OPENED_AT,
                )
        self.assertEqual(observed, {"ledger": True, "create": True})


class OpenPositionCliTests(Phase3KCase):
    def assert_cli_opens(self, strategy_id: str, intent_id: str) -> None:
        self.create_filled_intent(strategy_id, intent_id)
        code, stdout, stderr = self.run_open_cli(
            [
                "--root-dir",
                str(self.root),
                "--intent-id",
                intent_id,
                "--opened-at",
                OPENED_AT,
            ]
        )
        self.assertEqual(code, 0, stderr)
        self.assertIn(intent_id, stdout)
        self.assertIn("position_id", stdout)
        self.assertIn("open", stdout)
        self.assertIn("position_opened_event_id", stdout)
        self.assertEqual(self.state_store.get_order_intent(intent_id)["status"], "filled")
        loaded = StateStore(self.root / "data/state/portfolio_state.json")
        self.assertEqual(loaded.get_order_intent(intent_id)["status"], "position_opened")
        self.assertEqual(loaded.get_open_position(strategy_id)["status"], "open")
        self.assertIn("POSITION_OPENED", self.execution_path.read_text(encoding="utf-8"))

    def test_cli_opens_s01_and_s02(self) -> None:
        self.assert_cli_opens(S01_VOL_BASELINE, "s01:cli")
        self.execution_path.write_text("", encoding="utf-8")
        self.assert_cli_opens(S02_VOL_ENHANCED, "s02:cli")

    def test_cli_json_and_z_timestamp(self) -> None:
        intent_id = "s01:json"
        self.create_filled_intent(S01_VOL_BASELINE, intent_id)
        code, stdout, stderr = self.run_open_cli(
            [
                "--root-dir",
                str(self.root),
                "--intent-id",
                intent_id,
                "--opened-at",
                "2026-04-27T14:07:00Z",
                "--json",
            ]
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(stdout, json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        self.assertEqual(set(payload), {"intent_id", "position_id", "status", "position_opened_event_id", "dry_run"})
        self.assertEqual(payload["status"], "open")
        self.assertIs(payload["dry_run"], True)
        loaded = StateStore(self.root / "data/state/portfolio_state.json")
        self.assertEqual(loaded.get_position(payload["position_id"])["opened_at"], "2026-04-27T14:07:00+00:00")

    def test_cli_error_paths(self) -> None:
        with self.assertRaises(SystemExit):
            self.run_open_cli(["--root-dir", str(self.root)])
        self.create_intent(S01_VOL_BASELINE, "s01:created")
        self.create_intent(S01_VOL_BASELINE, "s01:submitted")
        self.state_store.submit_order_intent(
            "s01:submitted",
            submitted_at=SUBMITTED_AT,
            order_submitted_event_id="evt_submitted",
            simulated_order_id="sim_submitted",
        )
        self.create_intent(S01_VOL_BASELINE, "s01:confirmed")
        self.state_store.submit_order_intent(
            "s01:confirmed",
            submitted_at=SUBMITTED_AT,
            order_submitted_event_id="evt_submitted",
            simulated_order_id="sim_confirmed",
        )
        self.state_store.confirm_order_intent(
            "s01:confirmed",
            confirmed_at=CONFIRMED_AT,
            order_confirmed_event_id="evt_confirmed",
            simulated_order_id="sim_confirmed",
        )
        self.create_intent(S01_VOL_BASELINE, "s01:expired")
        self.state_store.expire_order_intent(
            "s01:expired",
            expired_at=OPENED_AT,
            expire_reason="ttl_expired",
            expired_event_id="evt_expired",
        )
        self.create_intent(S01_VOL_BASELINE, "s01:cancelled")
        self.state_store.cancel_order_intent(
            "s01:cancelled",
            cancelled_at=OPENED_AT,
            cancel_reason="operator_cancelled",
            cancelled_event_id="evt_cancelled",
        )
        self.create_filled_intent(S01_VOL_BASELINE, "s01:missing-ref")
        del self.state_store.state["order_intents"]["s01:missing-ref"]["order_ref"]
        for field in ("dry_run", "fill_confirmed_event_id", "fill_id", "fill_price", "fill_quantity"):
            intent_id = f"s01:missing:{field}"
            self.create_filled_intent(S01_VOL_BASELINE, intent_id)
            del self.state_store.state["order_intents"][intent_id][field]
        self.state_store.save()
        cases = [
            ("missing:intent", "missing:intent"),
            ("s01:created", "not 'filled'"),
            ("s01:submitted", "not 'filled'"),
            ("s01:confirmed", "not 'filled'"),
            ("s01:expired", "not 'filled'"),
            ("s01:cancelled", "not 'filled'"),
            ("s01:missing-ref", "order_ref"),
            ("s01:missing:dry_run", "intent.dry_run"),
            ("s01:missing:fill_confirmed_event_id", "fill_confirmed_event_id"),
            ("s01:missing:fill_id", "fill_id"),
            ("s01:missing:fill_price", "fill_price"),
            ("s01:missing:fill_quantity", "fill_quantity"),
        ]
        for intent_id, expected in cases:
            before = deepcopy(StateStore(self.root / "data/state/portfolio_state.json").state)
            code, stdout, stderr = self.run_open_cli(
                ["--root-dir", str(self.root), "--intent-id", intent_id, "--opened-at", OPENED_AT]
            )
            self.assertNotEqual(code, 0)
            self.assertEqual(stdout, "")
            self.assertIn(expected, stderr)
            self.assertEqual(StateStore(self.root / "data/state/portfolio_state.json").state, before)
        self.assertEqual(self.execution_path.read_text(encoding="utf-8"), "")

        self.create_filled_intent(S01_VOL_BASELINE, "s01:already-opened")
        opened = open_position_from_filled_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            intent_id="s01:already-opened",
            opened_at=OPENED_AT,
        )
        self.assertEqual(len(self.execution_events()), 1)
        code, stdout, stderr = self.run_open_cli(
            ["--root-dir", str(self.root), "--intent-id", "s01:already-opened", "--opened-at", OPENED_AT]
        )
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("not 'filled'", stderr)
        self.assertEqual(len(self.execution_events()), 1)
        self.assertEqual(self.state_store.get_position(opened["position_id"])["status"], "open")

        other_root = Path(self.tmp.name) / "existing-open"
        other_store = StateStore(other_root / "data/state/portfolio_state.json")
        other_ledger = LedgerAppender(other_root)
        other_case_position = self.position_record(position_id="pos:existing")
        other_store.create_open_position(other_case_position)
        self.state_store = other_store
        self.ledger = other_ledger
        self.execution_path = other_root / "data/ledger/execution_ledger.jsonl"
        self.order_path = other_root / "data/ledger/order_ledger.jsonl"
        self.create_filled_intent(S01_VOL_BASELINE, "s01:blocked")
        code, stdout, stderr = self.run_open_cli(
            ["--root-dir", str(other_root), "--intent-id", "s01:blocked", "--opened-at", OPENED_AT]
        )
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("open position already exists", stderr)
        self.assertEqual(self.execution_path.read_text(encoding="utf-8"), "")

        self.create_filled_intent(S01_VOL_BASELINE, "s01:bad-time")
        before = deepcopy(self.state_store.state)
        with self.assertRaises(SystemExit):
            self.run_open_cli(
                ["--root-dir", str(other_root), "--intent-id", "s01:bad-time", "--opened-at", "not-a-time"]
            )
        self.assertEqual(self.state_store.state, before)


class Phase3KRegressionSafetyTests(Phase3KCase):
    def test_prior_paths_do_not_auto_open_positions(self) -> None:
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
        self.assertNotIn("POSITION_OPENED", self.execution_path.read_text(encoding="utf-8"))
        self.submit_with_tool("s01:submit-cli")
        self.assertNotIn("POSITION_OPENED", self.execution_path.read_text(encoding="utf-8"))
        self.confirm_with_tool("s01:confirm-cli")
        self.assertNotIn("POSITION_OPENED", self.execution_path.read_text(encoding="utf-8"))
        self.fill_with_tool("s01:fill-cli")
        self.assertNotIn("POSITION_OPENED", self.execution_path.read_text(encoding="utf-8"))

    def submit_with_tool(self, intent_id: str) -> None:
        self.create_intent(S01_VOL_BASELINE, intent_id)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            submit_tool.main(["--root-dir", str(self.root), "--intent-id", intent_id, "--submitted-at", SUBMITTED_AT])

    def confirm_with_tool(self, intent_id: str) -> None:
        self.create_submitted_for_tool(intent_id)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            confirm_tool.main(["--root-dir", str(self.root), "--intent-id", intent_id, "--confirmed-at", CONFIRMED_AT])

    def fill_with_tool(self, intent_id: str) -> None:
        self.create_confirmed_for_tool(intent_id)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fill_tool.main(["--root-dir", str(self.root), "--intent-id", intent_id, "--filled-at", FILLED_AT])

    def create_submitted_for_tool(self, intent_id: str) -> None:
        self.create_intent(S01_VOL_BASELINE, intent_id)
        self.state_store.submit_order_intent(intent_id, submitted_at=SUBMITTED_AT, order_submitted_event_id="evt", simulated_order_id="sim")
        self.state_store.save()

    def create_confirmed_for_tool(self, intent_id: str) -> None:
        self.create_submitted_for_tool(intent_id)
        self.state_store.confirm_order_intent(intent_id, confirmed_at=CONFIRMED_AT, order_confirmed_event_id="evt_conf", simulated_order_id="sim")
        self.state_store.save()

    def test_open_position_cli_source_is_safe(self) -> None:
        source = inspect.getsource(open_tool)
        helper_source = inspect.getsource(open_position_from_filled_intent)
        self.assertIn("argparse.ArgumentParser", source)
        self.assertIn("open_position_from_filled_intent", source)
        self.assertNotIn("ib_insync", source)
        self.assertNotIn("yfinance", source)
        self.assertNotIn("requests", source)
        self.assertNotIn("broker.submit_order", source)
        self.assertNotIn("placeOrder", source)
        self.assertNotIn("cancelOrder", source)
        self.assertNotIn("ledger.append", source)
        self.assertNotIn("create_open_position(", source)
        self.assertNotIn("scheduler", source.lower())
        self.assertNotIn("except:", source)
        self.assertNotIn("except:", helper_source)
        self.assertIn("POSITION_OPENED", KNOWN_EVENT_TYPES)
        self.assertEqual(self.ledger.path_for_event_type("POSITION_OPENED"), self.execution_path)


if __name__ == "__main__":
    unittest.main()
