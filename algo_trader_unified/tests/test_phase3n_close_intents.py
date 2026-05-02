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
from datetime import datetime
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.close_intents import create_close_intent_from_position
from algo_trader_unified.core.ledger import KNOWN_EVENT_TYPES, LedgerAppender
from algo_trader_unified.core.ledger_reader import LedgerReader
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.tools import create_close_intent as close_tool
from algo_trader_unified.tools import system_status

CREATED_AT = "2026-04-27T15:10:00+00:00"


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


class Phase3NCase(unittest.TestCase):
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

    def reader(self) -> LedgerReader:
        return LedgerReader(
            execution_ledger_path=self.execution_path,
            order_ledger_path=self.order_path,
        )

    def position_record(
        self,
        strategy_id: str = S01_VOL_BASELINE,
        position_id: str = "position:1",
        *,
        status: str = "open",
        dry_run: object = True,
        quantity: object = 2,
        entry_price: object = 0.75,
    ) -> dict:
        return {
            "position_id": position_id,
            "intent_id": f"{position_id}:intent",
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "status": status,
            "execution_mode": "paper_only",
            "dry_run": dry_run,
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

    def create_position(
        self,
        strategy_id: str = S01_VOL_BASELINE,
        position_id: str = "position:1",
        **kwargs,
    ) -> dict:
        record = self.position_record(strategy_id, position_id, **kwargs)
        self.state_store.state["positions"][position_id] = deepcopy(record)
        self.state_store.save()
        return record

    def close_record(
        self,
        strategy_id: str = S01_VOL_BASELINE,
        position_id: str = "position:1",
        close_intent_id: str | None = None,
        *,
        status: str = "created",
        quantity: object = 2,
        entry_price: object = 0.75,
    ) -> dict:
        close_intent_id = close_intent_id or f"{strategy_id}:{position_id}:close"
        return {
            "close_intent_id": close_intent_id,
            "position_id": position_id,
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "status": status,
            "execution_mode": "paper_only",
            "dry_run": True,
            "created_at": CREATED_AT,
            "updated_at": CREATED_AT,
            "close_reason": "manual",
            "requested_by": "operator",
            "position_opened_event_id": "evt_opened",
            "source_signal_event_id": "evt_signal",
            "fill_confirmed_event_id": "evt_fill",
            "close_intent_created_event_id": f"evt_{close_intent_id}",
            "quantity": quantity,
            "entry_price": entry_price,
            "action": "close",
        }

    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = close_tool.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()


class CloseIntentLedgerRoutingTests(Phase3NCase):
    def test_close_intent_event_routes_to_order_ledger_only(self) -> None:
        self.assertIn("CLOSE_INTENT_CREATED", KNOWN_EVENT_TYPES)
        self.assertEqual(
            self.ledger.path_for_event_type("CLOSE_INTENT_CREATED").name,
            "order_ledger.jsonl",
        )

        self.create_position()
        created = create_close_intent_from_position(
            state_store=self.state_store,
            ledger=self.ledger,
            position_id="position:1",
            created_at=CREATED_AT,
        )

        order_events = self.reader().read_events("order")
        execution_events = self.reader().read_events("execution")
        self.assertEqual([event["event_type"] for event in order_events], ["CLOSE_INTENT_CREATED"])
        self.assertEqual(execution_events, [])
        event = order_events[0]
        payload = event["payload"]
        for field in (
            "close_intent_id",
            "position_id",
            "strategy_id",
            "sleeve_id",
            "symbol",
            "execution_mode",
            "dry_run",
            "created_at",
            "close_reason",
            "requested_by",
            "position_opened_event_id",
            "source_signal_event_id",
            "fill_confirmed_event_id",
            "quantity",
            "entry_price",
            "action",
            "status",
            "event_detail",
        ):
            self.assertIn(field, payload)
        self.assertIs(payload["dry_run"], True)
        self.assertIs(type(payload["quantity"]), int)
        self.assertIs(type(payload["entry_price"]), float)
        self.assertGreater(payload["quantity"], 0)
        self.assertGreaterEqual(payload["entry_price"], 0)
        self.assertEqual(payload["action"], "close")
        self.assertEqual(payload["status"], "created")
        self.assertEqual(payload["event_detail"], "CLOSE_INTENT_CREATED")
        self.assertEqual(created["close_intent_created_event_id"], event["event_id"])
        self.assertNotIn("target_price", payload)
        self.assertNotIn("limit_price", payload)
        self.assertNotIn("order_type", payload)
        ledger_text = self.order_path.read_text(encoding="utf-8") + self.execution_path.read_text(encoding="utf-8")
        for forbidden in (
            "ORDER_SUBMITTED",
            "ORDER_CONFIRMED",
            "FILL_CONFIRMED",
            "POSITION_ADJUSTED",
            "POSITION_CLOSED",
        ):
            self.assertNotIn(forbidden, ledger_text)


class StateStoreCloseIntentTests(Phase3NCase):
    def test_fresh_and_legacy_state_include_close_intents_mapping(self) -> None:
        self.assertEqual(self.state_store.state["close_intents"], {})
        legacy_path = self.root / "legacy_state.json"
        legacy_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "positions": {},
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
        self.assertEqual(loaded.state["close_intents"], {})

    def test_create_get_active_and_list_close_intents(self) -> None:
        s01 = self.state_store.create_close_intent(
            self.close_record(S01_VOL_BASELINE, "position:s01")
        )
        s02 = self.state_store.create_close_intent(
            self.close_record(S02_VOL_ENHANCED, "position:s02")
        )
        self.assertEqual(self.state_store.get_close_intent(s01["close_intent_id"]), s01)
        self.assertEqual(self.state_store.get_active_close_intent("position:s01"), s01)
        self.assertEqual(
            self.state_store.list_close_intents(strategy_id=S02_VOL_ENHANCED),
            [s02],
        )
        self.assertEqual(self.state_store.list_close_intents(status="created"), [s01, s02])
        self.assertEqual(self.state_store.list_close_intents(position_id="position:s01"), [s01])
        for field in (
            "close_intent_id",
            "position_id",
            "strategy_id",
            "sleeve_id",
            "symbol",
            "status",
            "execution_mode",
            "dry_run",
            "created_at",
            "updated_at",
            "close_reason",
            "requested_by",
            "position_opened_event_id",
            "source_signal_event_id",
            "fill_confirmed_event_id",
            "close_intent_created_event_id",
            "quantity",
            "entry_price",
            "action",
        ):
            self.assertIn(field, s01)
        self.assertIs(type(s01["quantity"]), int)
        self.assertIs(type(s01["entry_price"]), float)
        self.assertNotIn("target_price", s01)
        self.assertNotIn("limit_price", s01)
        self.assertNotIn("order_type", s01)

    def test_list_close_intents_with_combined_filters(self) -> None:
        s01_created = self.state_store.create_close_intent(
            self.close_record(S01_VOL_BASELINE, "position:s01-created", status="created")
        )
        self.state_store.create_close_intent(
            self.close_record(S02_VOL_ENHANCED, "position:s02-created", status="created")
        )
        submitted_base = self.close_record(S02_VOL_ENHANCED, "position:s02-submitted")
        submitted_id = submitted_base["close_intent_id"]
        self.state_store.state.setdefault("close_intents", {})[submitted_id] = dict(
            submitted_base, status="submitted"
        )
        self.state_store.save()
        s02_submitted = self.state_store.get_close_intent(submitted_id)

        # Matching combined filter
        self.assertEqual(
            self.state_store.list_close_intents(
                strategy_id=S01_VOL_BASELINE, status="created"
            ),
            [s01_created],
        )

        # Non-matching combined filter
        self.assertEqual(
            self.state_store.list_close_intents(
                strategy_id=S01_VOL_BASELINE, status="submitted"
            ),
            [],
        )

        # Another matching combined filter
        self.assertEqual(
            self.state_store.list_close_intents(
                strategy_id=S02_VOL_ENHANCED, status="submitted"
            ),
            [s02_submitted],
        )

    def test_duplicate_active_close_intent_is_rejected(self) -> None:
        self.state_store.create_close_intent(self.close_record(position_id="position:dupe"))
        with self.assertRaisesRegex(ValueError, "active close intent already exists"):
            self.state_store.create_close_intent(
                self.close_record(
                    position_id="position:dupe",
                    close_intent_id="other:close",
                )
            )

    def test_submitted_close_intent_is_active_and_blocks_direct_duplicate(self) -> None:
        submitted = self.close_record(
            position_id="position:submitted-active",
            close_intent_id="submitted:close",
        )
        self.state_store.state.setdefault("close_intents", {})[
            submitted["close_intent_id"]
        ] = dict(submitted, status="submitted")
        self.state_store.save()

        active = self.state_store.get_active_close_intent("position:submitted-active")
        self.assertIsNotNone(active)
        self.assertEqual(active["close_intent_id"], "submitted:close")
        self.assertEqual(active["status"], "submitted")
        with self.assertRaisesRegex(ValueError, "active close intent already exists"):
            self.state_store.create_close_intent(
                self.close_record(
                    position_id="position:submitted-active",
                    close_intent_id="second:close",
                )
            )

    def test_close_intent_numeric_validation_rejects_bool_and_string(self) -> None:
        for field in ("quantity", "entry_price"):
            for value in (True, "1"):
                record = self.close_record(position_id=f"position:{field}:{value!r}")
                record[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    self.state_store.create_close_intent(record)

    def test_position_linkage_marks_open_position_without_closing(self) -> None:
        original = self.create_position(position_id="position:link")
        marked = self.state_store.mark_position_close_intent_created(
            "position:link",
            close_intent_id="close:1",
            close_intent_created_event_id="evt_close",
            close_intent_created_at=CREATED_AT,
        )
        self.assertEqual(marked["status"], "open")
        self.assertEqual(marked["active_close_intent_id"], "close:1")
        self.assertEqual(marked["close_intent_created_event_id"], "evt_close")
        self.assertEqual(marked["close_intent_created_at"], CREATED_AT)
        self.assertEqual(marked["position_opened_event_id"], original["position_opened_event_id"])

    def test_position_linkage_rejects_missing_non_open_and_active_created(self) -> None:
        with self.assertRaisesRegex(KeyError, "does not exist"):
            self.state_store.mark_position_close_intent_created(
                "missing",
                close_intent_id="close",
                close_intent_created_event_id="evt_close",
                close_intent_created_at=CREATED_AT,
            )

        self.create_position(position_id="position:closed", status="closed")
        with self.assertRaisesRegex(ValueError, "not 'open'"):
            self.state_store.mark_position_close_intent_created(
                "position:closed",
                close_intent_id="close",
                close_intent_created_event_id="evt_close",
                close_intent_created_at=CREATED_AT,
            )

        self.create_position(position_id="position:active")
        self.state_store.create_close_intent(
            self.close_record(position_id="position:active", close_intent_id="close:active")
        )
        self.state_store.state["positions"]["position:active"]["active_close_intent_id"] = "close:active"
        self.state_store.save()
        with self.assertRaisesRegex(ValueError, "active close intent already exists"):
            self.state_store.mark_position_close_intent_created(
                "position:active",
                close_intent_id="close:2",
                close_intent_created_event_id="evt_close_2",
                close_intent_created_at=CREATED_AT,
            )


class CloseIntentHelperTests(Phase3NCase):
    def assert_helper_creates_close_intent(self, strategy_id: str, position_id: str) -> dict:
        self.create_position(strategy_id, position_id)
        created = create_close_intent_from_position(
            state_store=self.state_store,
            ledger=self.ledger,
            position_id=position_id,
            created_at=CREATED_AT,
        )
        self.assertEqual(created["close_intent_id"], f"{strategy_id}:{position_id}:close")
        self.assertEqual(created["position_id"], position_id)
        self.assertEqual(created["status"], "created")
        self.assertEqual(created["close_reason"], "manual")
        self.assertEqual(created["requested_by"], "operator")
        self.assertEqual(created["quantity"], 2)
        self.assertEqual(created["entry_price"], 0.75)
        self.assertEqual(created["position_opened_event_id"], "evt_opened")
        self.assertEqual(created["source_signal_event_id"], "evt_signal")
        self.assertEqual(created["fill_confirmed_event_id"], "evt_fill")
        self.assertNotIn("target_price", created)
        self.assertNotIn("limit_price", created)
        self.assertNotIn("order_type", created)
        self.assertEqual(self.state_store.get_close_intent(created["close_intent_id"]), created)
        position = self.state_store.get_position(position_id)
        self.assertEqual(position["status"], "open")
        self.assertEqual(position["active_close_intent_id"], created["close_intent_id"])
        order_events = self.reader().read_events("order")
        self.assertEqual(order_events[-1]["event_type"], "CLOSE_INTENT_CREATED")
        self.assertEqual(self.reader().read_events("execution"), [])
        return created

    def test_helper_creates_for_s01_and_s02_with_disambiguated_ids(self) -> None:
        s01 = self.assert_helper_creates_close_intent(S01_VOL_BASELINE, "same-position")
        self.state_store = StateStore(self.state_path)
        self.ledger = LedgerAppender(self.root)
        self.create_position(S02_VOL_ENHANCED, "same-position-2")
        s02 = create_close_intent_from_position(
            state_store=self.state_store,
            ledger=self.ledger,
            position_id="same-position-2",
            created_at=CREATED_AT,
            close_reason="risk",
            requested_by="tester",
        )
        self.assertNotEqual(s01["close_intent_id"], s02["close_intent_id"])
        self.assertEqual(s02["close_reason"], "risk")
        self.assertEqual(s02["requested_by"], "tester")

    def test_helper_rejects_missing_required_fields_and_invalid_numbers(self) -> None:
        cases = [
            ("dry_run", None, "dry_run"),
            ("position_opened_event_id", None, "position_opened_event_id"),
            ("source_signal_event_id", None, "source_signal_event_id"),
            ("fill_confirmed_event_id", None, "fill_confirmed_event_id"),
            ("quantity", None, "quantity"),
            ("entry_price", None, "entry_price"),
            ("quantity", "2", "quantity"),
            ("entry_price", "0.75", "entry_price"),
        ]
        for field, value, pattern in cases:
            position_id = f"position:bad:{field}:{value!r}"
            record = self.position_record(position_id=position_id)
            if value is None:
                record.pop(field)
            else:
                record[field] = value
            self.state_store.state["positions"][position_id] = record
            self.state_store.save()
            with self.assertRaisesRegex((ValueError, KeyError), pattern):
                create_close_intent_from_position(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    position_id=position_id,
                    created_at=CREATED_AT,
                )

    def test_existing_active_close_intent_blocks_another(self) -> None:
        self.assert_helper_creates_close_intent(S01_VOL_BASELINE, "position:dupe")
        with self.assertRaisesRegex(ValueError, "active close intent already exists"):
            create_close_intent_from_position(
                state_store=self.state_store,
                ledger=self.ledger,
                position_id="position:dupe",
                created_at=CREATED_AT,
            )

    def test_ledger_append_failure_leaves_state_unchanged(self) -> None:
        self.create_position(position_id="position:ledger-fail")
        before = deepcopy(StateStore(self.state_path).state)
        with mock.patch.object(self.ledger, "append", side_effect=RuntimeError("ledger failed")):
            with self.assertRaisesRegex(RuntimeError, "ledger failed"):
                create_close_intent_from_position(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    position_id="position:ledger-fail",
                    created_at=CREATED_AT,
                )
        self.assertEqual(deepcopy(StateStore(self.state_path).state), before)

    def test_create_failure_after_ledger_append_leaves_position_unlinked(self) -> None:
        self.create_position(position_id="position:create-fail")
        with mock.patch.object(
            self.state_store,
            "create_close_intent",
            side_effect=RuntimeError("create failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "create failed"):
                create_close_intent_from_position(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    position_id="position:create-fail",
                    created_at=CREATED_AT,
                )
        order_event_types = [event["event_type"] for event in self.reader().read_events("order")]
        self.assertEqual(order_event_types[-1], "CLOSE_INTENT_CREATED")
        position = self.state_store.get_position("position:create-fail")
        self.assertNotIn("active_close_intent_id", position)

    def test_position_link_failure_after_close_intent_creation_propagates(self) -> None:
        self.create_position(position_id="position:link-fail")
        with mock.patch.object(
            self.state_store,
            "mark_position_close_intent_created",
            side_effect=RuntimeError("link failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "link failed"):
                create_close_intent_from_position(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    position_id="position:link-fail",
                    created_at=CREATED_AT,
                )
        order_event_types = [event["event_type"] for event in self.reader().read_events("order")]
        self.assertEqual(order_event_types[-1], "CLOSE_INTENT_CREATED")
        self.assertEqual(len(self.state_store.list_close_intents(position_id="position:link-fail")), 1)
        position = self.state_store.get_position("position:link-fail")
        self.assertNotIn("active_close_intent_id", position)
        ledger_text = self.order_path.read_text(encoding="utf-8") + self.execution_path.read_text(encoding="utf-8")
        self.assertNotIn("POSITION_ADJUSTED", ledger_text)
        self.assertNotIn("POSITION_CLOSED", ledger_text)

    def test_helper_sequence_runs_under_strategy_lock(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:lock")
        spy = LockSpy()
        self.state_store.strategy_state_locks[S01_VOL_BASELINE] = spy
        observed = {"ledger": False, "create": False}
        real_append = self.ledger.append
        real_create = self.state_store.create_close_intent

        def append(**kwargs):
            observed["ledger"] = True
            self.assertTrue(spy.held)
            return real_append(**kwargs)

        def create(record):
            observed["create"] = True
            self.assertTrue(spy.held)
            return real_create(record)

        with mock.patch.object(self.ledger, "append", side_effect=append):
            with mock.patch.object(self.state_store, "create_close_intent", side_effect=create):
                create_close_intent_from_position(
                    state_store=self.state_store,
                    ledger=self.ledger,
                    position_id="position:lock",
                    created_at=CREATED_AT,
                )
        self.assertEqual(observed, {"ledger": True, "create": True})


class CloseIntentCliTests(Phase3NCase):
    def assert_cli_creates_close_intent(self, strategy_id: str, position_id: str) -> None:
        self.create_position(strategy_id, position_id)
        code, stdout, stderr = self.run_cli(
            [
                "--root-dir",
                str(self.root),
                "--position-id",
                position_id,
                "--created-at",
                CREATED_AT,
            ]
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertIn("close_intent_id", stdout)
        self.assertIn(position_id, stdout)
        self.assertIn("created", stdout)
        self.assertIn("close_intent_created_event_id", stdout)
        loaded = StateStore(self.state_path)
        close_intents = loaded.list_close_intents(position_id=position_id)
        self.assertEqual(len(close_intents), 1)
        self.assertEqual(close_intents[0]["status"], "created")
        position = loaded.get_position(position_id)
        self.assertEqual(position["status"], "open")
        self.assertEqual(position["active_close_intent_id"], close_intents[0]["close_intent_id"])
        order_event_types = [event["event_type"] for event in self.reader().read_events("order")]
        self.assertEqual(order_event_types[-1], "CLOSE_INTENT_CREATED")
        self.assertNotIn("POSITION_ADJUSTED", self.execution_path.read_text(encoding="utf-8"))
        self.assertNotIn("POSITION_CLOSED", self.execution_path.read_text(encoding="utf-8"))

    def test_cli_success_for_s01_and_s02(self) -> None:
        self.assert_cli_creates_close_intent(S01_VOL_BASELINE, "position:cli:s01")
        self.state_store = StateStore(self.state_path)
        self.assert_cli_creates_close_intent(S02_VOL_ENHANCED, "position:cli:s02")

    def test_cli_json_output_is_strict_and_z_timestamp_is_normalized(self) -> None:
        self.create_position(position_id="position:json")
        code, stdout, stderr = self.run_cli(
            [
                "--root-dir",
                str(self.root),
                "--position-id",
                "position:json",
                "--created-at",
                "2026-04-27T15:10:00Z",
                "--json",
            ]
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(
            set(payload),
            {
                "close_intent_id",
                "position_id",
                "status",
                "close_intent_created_event_id",
                "dry_run",
            },
        )
        self.assertEqual(payload["position_id"], "position:json")
        self.assertEqual(payload["status"], "created")
        self.assertIs(payload["dry_run"], True)
        self.assertEqual(stdout, json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        close_intent = StateStore(self.state_path).get_close_intent(payload["close_intent_id"])
        self.assertEqual(close_intent["created_at"], CREATED_AT)
        self.assertEqual(close_intent["close_reason"], "manual")
        self.assertEqual(close_intent["requested_by"], "operator")

    def test_cli_omitted_created_at_is_auto_generated_and_aware(self) -> None:
        position_id = "position:cli:auto-time"
        self.create_position(S01_VOL_BASELINE, position_id)
        code, stdout, stderr = self.run_cli(
            [
                "--root-dir",
                str(self.root),
                "--position-id",
                position_id,
                "--json",
            ]
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["status"], "created")

        order_events = self.reader().read_events("order")
        self.assertEqual([e["event_type"] for e in order_events], ["CLOSE_INTENT_CREATED"])

        loaded = StateStore(self.state_path)
        close_intent = loaded.get_close_intent(payload["close_intent_id"])
        self.assertIn("created_at", close_intent)
        created_at = datetime.fromisoformat(close_intent["created_at"])
        self.assertIsNotNone(created_at.tzinfo)

        position = loaded.get_position(position_id)
        self.assertEqual(position["status"], "open")
        self.assertIn("close_intent_created_at", position)
        linked_at = datetime.fromisoformat(position["close_intent_created_at"])
        self.assertIsNotNone(linked_at.tzinfo)

    def test_cli_required_and_error_paths(self) -> None:
        with self.assertRaises(SystemExit):
            self.run_cli(["--root-dir", str(self.root)])

        code, stdout, stderr = self.run_cli(
            ["--root-dir", str(self.root), "--position-id", "missing", "--created-at", CREATED_AT]
        )
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("missing", stderr)

        self.create_position(position_id="position:not-open", status="closed")
        code, stdout, stderr = self.run_cli(
            [
                "--root-dir",
                str(self.root),
                "--position-id",
                "position:not-open",
                "--created-at",
                CREATED_AT,
            ]
        )
        self.assertNotEqual(code, 0)
        self.assertEqual(self.reader().read_events("order"), [])

        self.create_position(position_id="position:bad-time")
        with self.assertRaises(SystemExit):
            self.run_cli(
                [
                    "--root-dir",
                    str(self.root),
                    "--position-id",
                    "position:bad-time",
                    "--created-at",
                    "not-a-time",
                ]
            )
        self.assertEqual(self.reader().read_events("order"), [])


class SystemStatusCloseIntentTests(Phase3NCase):
    def run_status(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = system_status.main(["--root-dir", str(self.root), *argv])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_status_counts_close_intents_and_filters_by_strategy(self) -> None:
        code, stdout, stderr = self.run_status(["--json"])
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(
            payload["close_intent_counts_by_status"],
            {"created": 0, "submitted": 0, "confirmed": 0},
        )
        self.assertEqual(payload["created_close_intents_count"], 0)
        self.assertEqual(payload["submitted_close_intents_count"], 0)
        self.assertEqual(payload["confirmed_close_intents_count"], 0)
        self.assertEqual(payload["total_close_intents_count"], 0)

        self.create_position(S01_VOL_BASELINE, "position:status:s01")
        create_close_intent_from_position(
            state_store=self.state_store,
            ledger=self.ledger,
            position_id="position:status:s01",
            created_at=CREATED_AT,
        )
        self.create_position(S02_VOL_ENHANCED, "position:status:s02")
        create_close_intent_from_position(
            state_store=self.state_store,
            ledger=self.ledger,
            position_id="position:status:s02",
            created_at=CREATED_AT,
        )

        code, stdout, stderr = self.run_status(["--json"])
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(
            payload["close_intent_counts_by_status"],
            {"created": 2, "submitted": 0, "confirmed": 0},
        )
        self.assertEqual(payload["created_close_intents_count"], 2)
        self.assertEqual(payload["submitted_close_intents_count"], 0)
        self.assertEqual(payload["confirmed_close_intents_count"], 0)
        self.assertEqual(payload["total_close_intents_count"], 2)

        code, stdout, stderr = self.run_status(["--json", "--strategy-id", S01_VOL_BASELINE])
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(
            payload["close_intent_counts_by_status"],
            {"created": 1, "submitted": 0, "confirmed": 0},
        )
        self.assertEqual(payload["created_close_intents_count"], 1)
        self.assertEqual(payload["submitted_close_intents_count"], 0)
        self.assertEqual(payload["confirmed_close_intents_count"], 0)
        self.assertEqual(payload["total_close_intents_count"], 1)

    def test_status_remains_read_only(self) -> None:
        self.create_position(position_id="position:readonly")
        create_close_intent_from_position(
            state_store=self.state_store,
            ledger=self.ledger,
            position_id="position:readonly",
            created_at=CREATED_AT,
        )
        before = self.state_path.read_text(encoding="utf-8")
        code, stdout, stderr = self.run_status(["--json"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        json.loads(stdout)
        self.assertEqual(self.state_path.read_text(encoding="utf-8"), before)


class CloseIntentSafetyTests(unittest.TestCase):
    def test_close_intent_cli_uses_helper_and_avoids_forbidden_surfaces(self) -> None:
        cli_source = inspect.getsource(importlib.import_module("algo_trader_unified.tools.create_close_intent"))
        helper_source = inspect.getsource(importlib.import_module("algo_trader_unified.core.close_intents"))
        self.assertIn("create_close_intent_from_position", cli_source)
        self.assertNotIn("LedgerAppender.append", cli_source)
        for source in (cli_source, helper_source):
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
                "ORDER_SUBMITTED",
                "target_price",
                "limit_price",
                "order_type",
            ):
                self.assertNotIn(needle, source)
