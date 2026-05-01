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
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.tools._formatting import compact_table
from algo_trader_unified.tools import list_order_intents as intents_tool
from algo_trader_unified.tools import list_positions as positions_tool


class FormattingHelperTests(unittest.TestCase):
    def test_compact_table_empty_records_is_safe(self) -> None:
        self.assertEqual(compact_table([]), "")
        self.assertEqual(compact_table([], ["intent_id", "status"]), "intent_id  status\n---------  ------")

    def test_compact_table_non_empty_output_stays_stable(self) -> None:
        output = compact_table(
            [
                {"intent_id": "intent:1", "status": "created"},
                {"intent_id": "intent:22", "status": "filled"},
            ],
            ["intent_id", "status"],
        )
        self.assertEqual(
            output,
            "\n".join(
                [
                    "intent_id  status ",
                    "---------  -------",
                    "intent:1   created",
                    "intent:22  filled ",
                ]
            ),
        )


class Phase3LOperatorToolCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_store = StateStore(self.root / "data/state/portfolio_state.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def load_state_store(self) -> StateStore:
        return StateStore(self.root / "data/state/portfolio_state.json")

    def run_intents(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = intents_tool.main(["--root-dir", str(self.root), *argv])
        return code, stdout.getvalue(), stderr.getvalue()

    def run_positions(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = positions_tool.main(["--root-dir", str(self.root), *argv])
        return code, stdout.getvalue(), stderr.getvalue()

    def create_intent(
        self,
        strategy_id: str,
        intent_id: str,
        *,
        status: str = "created",
        symbol: str = "XSP",
        created_at: object = "2026-04-27T13:40:00+00:00",
        updated_at: object = "2026-04-27T13:40:00+00:00",
        order_ref: str | None = None,
        dry_run: bool = True,
    ) -> dict:
        record = {
            "intent_id": intent_id,
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": symbol,
            "execution_mode": "paper_only",
            "status": status,
            "source_signal_event_id": "evt_signal",
            "order_intent_created_event_id": "evt_created",
            "order_ref": order_ref or f"{strategy_id}|P0427{symbol}|OPEN",
            "created_at": created_at,
            "updated_at": updated_at,
            "sizing_context": {"contracts": 1},
            "risk_context": {},
            "signal_payload_snapshot": {},
            "dry_run": dry_run,
            "score": 4.5,
        }
        self.state_store.state["order_intents"][intent_id] = deepcopy(record)
        self.state_store.save()
        return record

    def create_position(
        self,
        strategy_id: str,
        position_id: str,
        *,
        status: str = "open",
        symbol: str = "XSP",
        opened_at: object = "2026-04-27T14:07:00+00:00",
        updated_at: object = "2026-04-27T14:07:00+00:00",
        quantity: int = 1,
        entry_price: float = 0.5,
        dry_run: bool = True,
    ) -> dict:
        record = {
            "position_id": position_id,
            "intent_id": f"{position_id}:intent",
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": symbol,
            "status": status,
            "execution_mode": "paper_only",
            "dry_run": dry_run,
            "opened_at": opened_at,
            "updated_at": updated_at,
            "entry_price": entry_price,
            "quantity": quantity,
            "order_ref": f"{strategy_id}|P0427{symbol}|OPEN",
            "simulated_order_id": f"sim:{position_id}",
        }
        self.state_store.state["positions"][position_id] = deepcopy(record)
        self.state_store.save()
        return record


class ListOrderIntentsTests(Phase3LOperatorToolCase):
    def test_empty_human_output(self) -> None:
        code, stdout, stderr = self.run_intents([])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, "No order intents found.\n")

    def test_empty_json_output(self) -> None:
        code, stdout, stderr = self.run_intents(["--json"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["order_intents"], [])
        self.assertEqual(payload["filters"], {"strategy_id": None, "status": None})

    def test_missing_state_file_does_not_create_state(self) -> None:
        missing_root = self.root / "missing"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = intents_tool.main(["--root-dir", str(missing_root)])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "No order intents found.\n")
        self.assertFalse((missing_root / "data/state/portfolio_state.json").exists())

    def test_lists_multiple_intents_human_and_json(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "intent:s01", status="created")
        self.create_intent(S02_VOL_ENHANCED, "intent:s02", status="submitted")
        code, stdout, stderr = self.run_intents([])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        for expected in ("intent:s01", "intent:s02", S01_VOL_BASELINE, S02_VOL_ENHANCED, "created", "submitted", "XSP"):
            self.assertIn(expected, stdout)

        code, stdout, stderr = self.run_intents(["--json"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["count"], 2)

    def test_filters_strategy_status_and_unknown(self) -> None:
        statuses = ["created", "submitted", "confirmed", "filled", "position_opened"]
        for index, status in enumerate(statuses):
            strategy_id = S01_VOL_BASELINE if index % 2 == 0 else S02_VOL_ENHANCED
            self.create_intent(strategy_id, f"intent:{status}", status=status)

        code, stdout, stderr = self.run_intents(["--json", "--strategy-id", S01_VOL_BASELINE])
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual({record["strategy_id"] for record in payload["order_intents"]}, {S01_VOL_BASELINE})

        code, stdout, stderr = self.run_intents(["--json", "--strategy-id", S02_VOL_ENHANCED])
        self.assertEqual(code, 0, stderr)
        self.assertEqual({record["strategy_id"] for record in json.loads(stdout)["order_intents"]}, {S02_VOL_ENHANCED})

        for status in statuses:
            code, stdout, stderr = self.run_intents(["--json", "--status", status])
            self.assertEqual(code, 0, stderr)
            records = json.loads(stdout)["order_intents"]
            self.assertEqual({record["status"] for record in records}, {status})

        code, stdout, stderr = self.run_intents(["--json", "--status", "missing"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["count"], 0)

    def test_unknown_strategy_filter_returns_empty_success(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "intent:s01", status="created")

        code, stdout, stderr = self.run_intents(["--json", "--strategy-id", "DOES_NOT_EXIST"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["order_intents"], [])

        code, stdout, stderr = self.run_intents(["--strategy-id", "DOES_NOT_EXIST"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, "No order intents found.\n")

    def test_combined_strategy_and_status_filters(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "intent:s01-created", status="created")
        self.create_intent(S01_VOL_BASELINE, "intent:s01-submitted", status="submitted")
        self.create_intent(S02_VOL_ENHANCED, "intent:s02-created", status="created")

        code, stdout, stderr = self.run_intents(
            ["--json", "--strategy-id", S01_VOL_BASELINE, "--status", "created"]
        )
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["order_intents"][0]["intent_id"], "intent:s01-created")

        code, stdout, stderr = self.run_intents(
            ["--json", "--strategy-id", S01_VOL_BASELINE, "--status", "filled"]
        )
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["order_intents"], [])

    def test_limit_zero_returns_empty_json(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "intent:s01", status="created")

        code, stdout, stderr = self.run_intents(["--json", "--limit", "0"])
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["order_intents"], [])

    def test_limit_sort_reverse_and_missing_sort_key(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "intent:b", created_at="2026-04-27T13:42:00+00:00")
        self.create_intent(S01_VOL_BASELINE, "intent:a", created_at="2026-04-27T13:41:00+00:00")
        self.create_intent(S01_VOL_BASELINE, "intent:missing")
        del self.state_store.state["order_intents"]["intent:missing"]["created_at"]
        self.state_store.save()

        code, stdout, stderr = self.run_intents(["--json", "--sort", "created_at", "--limit", "2"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual([record["intent_id"] for record in json.loads(stdout)["order_intents"]], ["intent:a", "intent:b"])

        code, stdout, stderr = self.run_intents(["--json", "--sort", "created_at", "--reverse"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["order_intents"][0]["intent_id"], "intent:missing")

    def test_invalid_limit_and_sort_fail_through_argparse(self) -> None:
        with self.assertRaises(SystemExit):
            self.run_intents(["--limit", "-1"])
        with self.assertRaises(SystemExit):
            self.run_intents(["--sort", "intent_id"])

    def test_json_strictness_and_read_purity(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "intent:json")
        before = deepcopy(self.load_state_store().state)
        code, stdout, stderr = self.run_intents(["--json"])
        after = deepcopy(self.load_state_store().state)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(stdout, json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        record = payload["order_intents"][0]
        self.assertIs(record["dry_run"], True)
        self.assertIsInstance(record["dry_run"], bool)
        self.assertIsInstance(record["score"], float)
        self.assertEqual(before, after)

    def test_json_serializes_native_dates_from_state_records(self) -> None:
        created_at = datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc)

        class FakeStateStore:
            def __init__(self, path: Path) -> None:
                self.path = path

            def list_order_intents(self, strategy_id: str | None = None) -> list[dict]:
                return [
                    {
                        "intent_id": "intent:date",
                        "strategy_id": S01_VOL_BASELINE,
                        "status": "created",
                        "created_at": created_at,
                        "dry_run": True,
                        "score": 4.5,
                    }
                ]

        with mock.patch.object(intents_tool, "StateStore", FakeStateStore):
            code, stdout, stderr = self.run_intents(["--json"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        record = json.loads(stdout)["order_intents"][0]
        self.assertEqual(record["created_at"], created_at.isoformat())
        self.assertIs(record["dry_run"], True)
        self.assertIsInstance(record["score"], float)


class ListPositionsTests(Phase3LOperatorToolCase):
    def test_empty_human_output(self) -> None:
        code, stdout, stderr = self.run_positions([])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, "No positions found.\n")

    def test_empty_json_output(self) -> None:
        code, stdout, stderr = self.run_positions(["--json"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["positions"], [])
        self.assertEqual(payload["filters"], {"strategy_id": None, "status": None, "symbol": None})

    def test_missing_state_file_does_not_create_state(self) -> None:
        missing_root = self.root / "missing"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = positions_tool.main(["--root-dir", str(missing_root)])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "No positions found.\n")
        self.assertFalse((missing_root / "data/state/portfolio_state.json").exists())

    def test_lists_multiple_positions_human_and_json(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01", quantity=2, entry_price=0.75)
        self.create_position(S02_VOL_ENHANCED, "position:s02", symbol="SPY", quantity=3, entry_price=1.25)
        code, stdout, stderr = self.run_positions([])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        for expected in ("position:s01", "position:s02", S01_VOL_BASELINE, S02_VOL_ENHANCED, "XSP", "SPY", "2", "0.75"):
            self.assertIn(expected, stdout)

        code, stdout, stderr = self.run_positions(["--json"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["count"], 2)

    def test_filters_strategy_status_symbol_and_unknown(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01", status="open", symbol="XSP")
        self.create_position(S02_VOL_ENHANCED, "position:s02", status="closed", symbol="SPY")

        code, stdout, stderr = self.run_positions(["--json", "--strategy-id", S01_VOL_BASELINE])
        self.assertEqual(code, 0, stderr)
        self.assertEqual({record["strategy_id"] for record in json.loads(stdout)["positions"]}, {S01_VOL_BASELINE})

        code, stdout, stderr = self.run_positions(["--json", "--strategy-id", S02_VOL_ENHANCED])
        self.assertEqual(code, 0, stderr)
        self.assertEqual({record["strategy_id"] for record in json.loads(stdout)["positions"]}, {S02_VOL_ENHANCED})

        code, stdout, stderr = self.run_positions(["--json", "--status", "open"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual({record["status"] for record in json.loads(stdout)["positions"]}, {"open"})

        code, stdout, stderr = self.run_positions(["--json", "--symbol", "SPY"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual({record["symbol"] for record in json.loads(stdout)["positions"]}, {"SPY"})

        code, stdout, stderr = self.run_positions(["--json", "--symbol", "QQQ"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["count"], 0)

    def test_unknown_strategy_filter_returns_empty_success(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01", status="open")

        code, stdout, stderr = self.run_positions(["--json", "--strategy-id", "DOES_NOT_EXIST"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["positions"], [])

        code, stdout, stderr = self.run_positions(["--strategy-id", "DOES_NOT_EXIST"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, "No positions found.\n")

    def test_combined_strategy_status_and_symbol_filters(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01-open-xsp", status="open", symbol="XSP")
        self.create_position(S01_VOL_BASELINE, "position:s01-closed-xsp", status="closed", symbol="XSP")
        self.create_position(S01_VOL_BASELINE, "position:s01-open-spy", status="open", symbol="SPY")
        self.create_position(S02_VOL_ENHANCED, "position:s02-open-xsp", status="open", symbol="XSP")

        code, stdout, stderr = self.run_positions(
            ["--json", "--strategy-id", S01_VOL_BASELINE, "--status", "open"]
        )
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["count"], 2)
        self.assertEqual(
            {record["position_id"] for record in payload["positions"]},
            {"position:s01-open-xsp", "position:s01-open-spy"},
        )

        code, stdout, stderr = self.run_positions(
            ["--json", "--strategy-id", S01_VOL_BASELINE, "--status", "pending_close"]
        )
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["positions"], [])

        code, stdout, stderr = self.run_positions(
            [
                "--json",
                "--strategy-id",
                S01_VOL_BASELINE,
                "--status",
                "open",
                "--symbol",
                "XSP",
            ]
        )
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["positions"][0]["position_id"], "position:s01-open-xsp")

    def test_limit_zero_returns_empty_json(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01", status="open")

        code, stdout, stderr = self.run_positions(["--json", "--limit", "0"])
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["positions"], [])

    def test_limit_sort_reverse_and_missing_sort_key(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:b", opened_at="2026-04-27T14:08:00+00:00")
        self.create_position(S01_VOL_BASELINE, "position:a", opened_at="2026-04-27T14:07:00+00:00")
        self.create_position(S01_VOL_BASELINE, "position:missing")
        del self.state_store.state["positions"]["position:missing"]["opened_at"]
        self.state_store.save()

        code, stdout, stderr = self.run_positions(["--json", "--sort", "opened_at", "--limit", "2"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual([record["position_id"] for record in json.loads(stdout)["positions"]], ["position:a", "position:b"])

        code, stdout, stderr = self.run_positions(["--json", "--sort", "opened_at", "--reverse"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["positions"][0]["position_id"], "position:missing")

    def test_invalid_limit_and_sort_fail_through_argparse(self) -> None:
        with self.assertRaises(SystemExit):
            self.run_positions(["--limit", "-1"])
        with self.assertRaises(SystemExit):
            self.run_positions(["--sort", "position_id"])

    def test_json_strictness_and_read_purity(self) -> None:
        self.create_position(
            S01_VOL_BASELINE,
            "position:json",
            quantity=2,
            entry_price=0.75,
        )
        before = deepcopy(self.load_state_store().state)
        code, stdout, stderr = self.run_positions(["--json"])
        after = deepcopy(self.load_state_store().state)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(stdout, json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        record = payload["positions"][0]
        self.assertIs(record["dry_run"], True)
        self.assertIsInstance(record["dry_run"], bool)
        self.assertEqual(record["quantity"], 2)
        self.assertIsInstance(record["quantity"], int)
        self.assertEqual(record["entry_price"], 0.75)
        self.assertIsInstance(record["entry_price"], float)
        self.assertEqual(before, after)

    def test_json_serializes_native_dates_from_state_records(self) -> None:
        opened_at = datetime(2026, 4, 27, 14, 7, tzinfo=timezone.utc)
        updated_at = date(2026, 4, 27)

        class FakeStateStore:
            def __init__(self, path: Path) -> None:
                self.path = path

            def list_positions(
                self,
                strategy_id: str | None = None,
                status: str | None = None,
            ) -> list[dict]:
                return [
                    {
                        "position_id": "position:date",
                        "strategy_id": S01_VOL_BASELINE,
                        "symbol": "XSP",
                        "status": "open",
                        "opened_at": opened_at,
                        "updated_at": updated_at,
                        "quantity": 2,
                        "entry_price": 0.75,
                        "dry_run": True,
                    }
                ]

        with mock.patch.object(positions_tool, "StateStore", FakeStateStore):
            code, stdout, stderr = self.run_positions(["--json"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        record = json.loads(stdout)["positions"][0]
        self.assertEqual(record["opened_at"], opened_at.isoformat())
        self.assertEqual(record["updated_at"], updated_at.isoformat())
        self.assertIs(record["dry_run"], True)
        self.assertEqual(record["quantity"], 2)
        self.assertEqual(record["entry_price"], 0.75)


class Phase3LOperatorToolSafetyTests(unittest.TestCase):
    def test_listing_tools_import_shared_helpers_without_private_duplicates(self) -> None:
        formatting = importlib.import_module("algo_trader_unified.tools._formatting")
        intents = importlib.import_module("algo_trader_unified.tools.list_order_intents")
        positions = importlib.import_module("algo_trader_unified.tools.list_positions")

        self.assertIs(intents.limit_arg, formatting.limit_arg)
        self.assertIs(positions.limit_arg, formatting.limit_arg)
        self.assertIs(intents.state_path, formatting.state_path)
        self.assertIs(positions.state_path, formatting.state_path)
        self.assertFalse(hasattr(intents, "_limit"))
        self.assertFalse(hasattr(positions, "_limit"))
        self.assertFalse(hasattr(intents, "_state_path"))
        self.assertFalse(hasattr(positions, "_state_path"))

    def test_new_tools_do_not_import_or_call_forbidden_surfaces(self) -> None:
        modules = [
            importlib.import_module("algo_trader_unified.tools._formatting"),
            importlib.import_module("algo_trader_unified.tools.list_order_intents"),
            importlib.import_module("algo_trader_unified.tools.list_positions"),
        ]
        forbidden = [
            "ib_insync",
            "yfinance",
            "requests",
            "broker.submit_order",
            "placeOrder",
            "cancelOrder",
            "scheduler.start",
            "LedgerAppender.append",
            "StateStore.save",
            "submit_order_intent",
            "confirm_order_intent",
            "confirm_fill",
            "open_position_from_filled_intent",
            "POSITION_ADJUSTED",
            "POSITION_CLOSED",
            "except:",
        ]
        for module in modules:
            source = inspect.getsource(module)
            for needle in forbidden:
                self.assertNotIn(needle, source)
