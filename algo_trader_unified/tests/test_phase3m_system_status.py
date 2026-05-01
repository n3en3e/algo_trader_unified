from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.tools import system_status


REQUIRED_SUMMARY_KEYS = {
    "order_intent_counts_by_status",
    "position_counts_by_status",
    "open_positions_count",
    "created_order_intents_count",
    "submitted_order_intents_count",
    "confirmed_order_intents_count",
    "filled_order_intents_count",
    "position_opened_intents_count",
    "unresolved_order_intents_count",
    "stranded_order_intents_count",
    "stranded_filled_intents_count",
    "stranded_submitted_intents_count",
    "stranded_confirmed_intents_count",
    "total_order_intents_count",
    "total_positions_count",
    "filters",
}
EXPECTED_ORDER_INTENT_STATUS_KEYS = {
    "created",
    "submitted",
    "confirmed",
    "filled",
    "position_opened",
    "expired",
    "cancelled",
}
COUNT_KEYS = {
    key
    for key in REQUIRED_SUMMARY_KEYS
    if key.endswith("_count")
}


class SystemStatusCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @property
    def state_path(self) -> Path:
        return self.root / "data/state/portfolio_state.json"

    def state_store(self) -> StateStore:
        return StateStore(self.state_path)

    def run_status(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = system_status.main(["--root-dir", str(self.root), *argv])
        return code, stdout.getvalue(), stderr.getvalue()

    def create_intent(
        self,
        strategy_id: str,
        intent_id: str,
        status: str,
    ) -> dict:
        state_store = self.state_store()
        record = {
            "intent_id": intent_id,
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "execution_mode": "paper_only",
            "status": status,
            "source_signal_event_id": "evt_signal",
            "order_intent_created_event_id": "evt_created",
            "order_ref": f"{strategy_id}|{intent_id}|OPEN",
            "created_at": "2026-04-27T13:40:00+00:00",
            "updated_at": "2026-04-27T13:40:00+00:00",
            "sizing_context": {"contracts": 1},
            "risk_context": {},
            "signal_payload_snapshot": {},
            "dry_run": True,
        }
        state_store.state["order_intents"][intent_id] = deepcopy(record)
        state_store.save()
        return record

    def create_position(
        self,
        strategy_id: str,
        position_id: str,
        status: str,
    ) -> dict:
        state_store = self.state_store()
        record = {
            "position_id": position_id,
            "intent_id": f"{position_id}:intent",
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "status": status,
            "execution_mode": "paper_only",
            "dry_run": True,
            "opened_at": "2026-04-27T14:07:00+00:00",
            "updated_at": "2026-04-27T14:07:00+00:00",
            "entry_price": 0.5,
            "quantity": 1,
            "order_ref": f"{strategy_id}|{position_id}|OPEN",
            "simulated_order_id": f"sim:{position_id}",
        }
        state_store.state["positions"][position_id] = deepcopy(record)
        state_store.save()
        return record


class SystemStatusEmptyTests(SystemStatusCase):
    def test_missing_state_file_reports_zero_counts_without_creating_state(self) -> None:
        code, stdout, stderr = self.run_status(["--json"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertFalse(self.state_path.exists())
        payload = json.loads(stdout)
        for key in COUNT_KEYS:
            self.assertEqual(payload[key], 0)
        self.assertTrue(
            EXPECTED_ORDER_INTENT_STATUS_KEYS.issubset(
                payload["order_intent_counts_by_status"]
            )
        )
        for status in EXPECTED_ORDER_INTENT_STATUS_KEYS:
            self.assertEqual(payload["order_intent_counts_by_status"][status], 0)
        self.assertEqual(payload["position_counts_by_status"]["open"], 0)

    def test_corrupt_state_file_returns_error_without_modifying_file(self) -> None:
        self.state_path.parent.mkdir(parents=True)
        corrupt_text = '{"key":'
        self.state_path.write_text(corrupt_text, encoding="utf-8")

        code, stdout, stderr = self.run_status(["--json"])

        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr)
        self.assertTrue(
            any(
                marker in stderr.lower()
                for marker in ("failed", "error", "corrupt", "invalid", "expecting")
            ),
            stderr,
        )
        self.assertEqual(self.state_path.read_text(encoding="utf-8"), corrupt_text)

    def test_empty_fresh_state_json_and_human_are_stable(self) -> None:
        self.state_store()

        code, stdout, stderr = self.run_status(["--json"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertTrue(REQUIRED_SUMMARY_KEYS.issubset(payload))
        for key in COUNT_KEYS:
            self.assertEqual(payload[key], 0)
            self.assertIs(type(payload[key]), int)
        self.assertTrue(
            EXPECTED_ORDER_INTENT_STATUS_KEYS.issubset(
                payload["order_intent_counts_by_status"]
            )
        )
        for status in EXPECTED_ORDER_INTENT_STATUS_KEYS:
            self.assertEqual(payload["order_intent_counts_by_status"][status], 0)
        self.assertEqual(payload["filters"], {"strategy_id": None})

        code, stdout, stderr = self.run_status([])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertEqual(
            stdout,
            "\n".join(
                [
                    "System status",
                    "Order intents:",
                    "  total: 0",
                    "  created: 0",
                    "  submitted: 0",
                    "  confirmed: 0",
                    "  filled: 0",
                    "  position_opened: 0",
                    "  unresolved: 0",
                    "Positions:",
                    "  total: 0",
                    "  open: 0",
                    "Stranded:",
                    "  total: 0",
                    "  submitted: 0",
                    "  confirmed: 0",
                    "  filled: 0",
                ]
            )
            + "\n",
        )


class SystemStatusSummaryTests(SystemStatusCase):
    def seed_mixed_lifecycle(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "intent:s01-created", "created")
        self.create_intent(S01_VOL_BASELINE, "intent:s01-submitted", "submitted")
        self.create_intent(S01_VOL_BASELINE, "intent:s01-confirmed", "confirmed")
        self.create_intent(S01_VOL_BASELINE, "intent:s01-filled", "filled")
        self.create_intent(S01_VOL_BASELINE, "intent:s01-opened", "position_opened")
        self.create_intent(S01_VOL_BASELINE, "intent:s01-expired", "expired")
        self.create_intent(S02_VOL_ENHANCED, "intent:s02-created", "created")
        self.create_intent(S02_VOL_ENHANCED, "intent:s02-opened", "position_opened")
        self.create_intent(S02_VOL_ENHANCED, "intent:s02-cancelled", "cancelled")
        self.create_position(S01_VOL_BASELINE, "position:s01-open", "open")
        self.create_position(S02_VOL_ENHANCED, "position:s02-open", "open")

    def test_mixed_lifecycle_summary_counts(self) -> None:
        self.seed_mixed_lifecycle()

        code, stdout, stderr = self.run_status(["--json"])
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(
            payload["order_intent_counts_by_status"],
            {
                "cancelled": 1,
                "confirmed": 1,
                "created": 2,
                "expired": 1,
                "filled": 1,
                "position_opened": 2,
                "submitted": 1,
            },
        )
        self.assertEqual(payload["position_counts_by_status"], {"open": 2})
        self.assertEqual(payload["created_order_intents_count"], 2)
        self.assertEqual(payload["submitted_order_intents_count"], 1)
        self.assertEqual(payload["confirmed_order_intents_count"], 1)
        self.assertEqual(payload["filled_order_intents_count"], 1)
        self.assertEqual(payload["position_opened_intents_count"], 2)
        self.assertEqual(payload["unresolved_order_intents_count"], 5)
        self.assertEqual(payload["stranded_order_intents_count"], 3)
        self.assertEqual(payload["stranded_submitted_intents_count"], 1)
        self.assertEqual(payload["stranded_confirmed_intents_count"], 1)
        self.assertEqual(payload["stranded_filled_intents_count"], 1)
        self.assertEqual(payload["open_positions_count"], 2)
        self.assertEqual(payload["total_order_intents_count"], 9)
        self.assertEqual(payload["total_positions_count"], 2)

    def test_strategy_filter_counts_and_unknown_strategy(self) -> None:
        self.seed_mixed_lifecycle()

        code, stdout, stderr = self.run_status(["--json", "--strategy-id", S01_VOL_BASELINE])
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["filters"], {"strategy_id": S01_VOL_BASELINE})
        self.assertEqual(payload["total_order_intents_count"], 6)
        self.assertEqual(payload["created_order_intents_count"], 1)
        self.assertEqual(payload["submitted_order_intents_count"], 1)
        self.assertEqual(payload["confirmed_order_intents_count"], 1)
        self.assertEqual(payload["filled_order_intents_count"], 1)
        self.assertEqual(payload["position_opened_intents_count"], 1)
        self.assertEqual(payload["open_positions_count"], 1)
        self.assertEqual(payload["unresolved_order_intents_count"], 4)
        self.assertEqual(payload["stranded_order_intents_count"], 3)

        code, stdout, stderr = self.run_status(["--json", "--strategy-id", S02_VOL_ENHANCED])
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["filters"], {"strategy_id": S02_VOL_ENHANCED})
        self.assertEqual(payload["total_order_intents_count"], 3)
        self.assertEqual(payload["created_order_intents_count"], 1)
        self.assertEqual(payload["position_opened_intents_count"], 1)
        self.assertEqual(payload["open_positions_count"], 1)
        self.assertEqual(payload["unresolved_order_intents_count"], 1)
        self.assertEqual(payload["stranded_order_intents_count"], 0)

        code, stdout, stderr = self.run_status(["--json", "--strategy-id", "DOES_NOT_EXIST"])
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["filters"], {"strategy_id": "DOES_NOT_EXIST"})
        for key in COUNT_KEYS:
            self.assertEqual(payload[key], 0)

    def test_human_output_contains_expected_labels_for_data_and_unknown_strategy(self) -> None:
        self.seed_mixed_lifecycle()

        code, stdout, stderr = self.run_status([])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        for expected in (
            "System status",
            "Order intents:",
            "Positions:",
            "Stranded:",
            "total",
            "created",
            "submitted",
            "confirmed",
            "filled",
            "position_opened",
            "unresolved",
            "open",
        ):
            self.assertIn(expected, stdout)

        code, stdout, stderr = self.run_status(["--strategy-id", "DOES_NOT_EXIST"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertIn("System status", stdout)
        self.assertIn("  total: 0", stdout)
        self.assertIn("  open: 0", stdout)

    def test_json_stdout_is_strict_and_unpolluted(self) -> None:
        self.seed_mixed_lifecycle()

        code, stdout, stderr = self.run_status(["--json"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertNotIn("System status", stdout)
        payload = json.loads(stdout)
        self.assertEqual(stdout, json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        self.assertTrue(REQUIRED_SUMMARY_KEYS.issubset(payload))
        for key in COUNT_KEYS:
            self.assertIs(type(payload[key]), int)
        self.assertIsNone(payload["filters"]["strategy_id"])


class SystemStatusReadPurityTests(SystemStatusCase):
    def test_tool_does_not_mutate_disk_state_or_call_save(self) -> None:
        self.create_intent(S01_VOL_BASELINE, "intent:s01-created", "created")
        self.create_position(S01_VOL_BASELINE, "position:s01-open", "open")
        before_text = self.state_path.read_text(encoding="utf-8")
        before_state = deepcopy(StateStore(self.state_path).state)

        with mock.patch.object(system_status.StateStore, "save", side_effect=AssertionError):
            code, stdout, stderr = self.run_status(["--json"])

        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["total_order_intents_count"], 1)
        self.assertEqual(self.state_path.read_text(encoding="utf-8"), before_text)
        self.assertEqual(deepcopy(StateStore(self.state_path).state), before_state)

    def test_tool_does_not_mutate_state_store_state(self) -> None:
        captured: dict[str, object] = {}

        class FakeStateStore:
            def __init__(self, path: Path) -> None:
                self.path = path
                self.state = {
                    "schema_version": 1,
                    "order_intents": {
                        "intent:created": {
                            "intent_id": "intent:created",
                            "strategy_id": S01_VOL_BASELINE,
                            "status": "created",
                        }
                    },
                    "positions": {
                        "position:open": {
                            "position_id": "position:open",
                            "strategy_id": S01_VOL_BASELINE,
                            "status": "open",
                        }
                    },
                }
                captured["instance"] = self

        self.state_path.parent.mkdir(parents=True)
        self.state_path.write_text('{"schema_version":1,"order_intents":{},"positions":{}}', encoding="utf-8")
        with mock.patch.object(system_status, "StateStore", FakeStateStore):
            code, stdout, stderr = self.run_status(["--json"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["created_order_intents_count"], 1)
        instance = captured["instance"]
        self.assertEqual(
            instance.state,
            {
                "schema_version": 1,
                "order_intents": {
                    "intent:created": {
                        "intent_id": "intent:created",
                        "strategy_id": S01_VOL_BASELINE,
                        "status": "created",
                    }
                },
                "positions": {
                    "position:open": {
                        "position_id": "position:open",
                        "strategy_id": S01_VOL_BASELINE,
                        "status": "open",
                    }
                },
            },
        )

    def test_missing_state_file_does_not_create_state(self) -> None:
        code, stdout, stderr = self.run_status([])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertIn("System status", stdout)
        self.assertFalse(self.state_path.exists())


class SystemStatusSafetyTests(unittest.TestCase):
    def test_system_status_source_avoids_forbidden_surfaces(self) -> None:
        module = importlib.import_module("algo_trader_unified.tools.system_status")
        source = inspect.getsource(module)
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
        for needle in forbidden:
            self.assertNotIn(needle, source)
