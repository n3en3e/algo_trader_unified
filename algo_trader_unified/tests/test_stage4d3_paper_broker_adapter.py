from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from dataclasses import asdict
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from algo_trader_unified.core.broker_adapter import (
    BrokerAccountSnapshot,
    BrokerAdapter,
    BrokerCancelResult,
    BrokerOrderStatus,
    BrokerPosition,
    BrokerSubmitResult,
)
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.paper_broker_adapter import (
    BrokerOrderRequest,
    PaperBrokerAdapter,
    build_broker_order_request,
    sanitize_json_safe,
)
from algo_trader_unified.core.paper_broker_contract import BrokerModeError, validate_broker_mode
from algo_trader_unified.core.state_store import StateStore


ROOT = Path(__file__).resolve().parents[1]
STAGE4D3_FILES = [
    ROOT / "core/paper_broker_adapter.py",
]


class UnsupportedThing:
    pass


class FakePaperClient:
    def __init__(self) -> None:
        self.submitted_requests: list[BrokerOrderRequest] = []
        self.cancel_calls: list[tuple[str, str | None]] = []
        self.raise_on: str | None = None
        self.open_orders_response: list[dict] | None = None
        self.positions_response: list[dict] | None = None

    def submit_order(self, request: BrokerOrderRequest) -> dict:
        self.submitted_requests.append(request)
        if self.raise_on == "submit_order":
            raise RuntimeError("submit failed")
        return {
            "accepted": True,
            "broker_order_id": "paper_order_1",
            "client_order_id": request.client_order_id,
            "reason": None,
            "decimal": Decimal("1.25"),
            "timestamp": datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
            "date": date(2026, 5, 6),
            "exception": ValueError("bad payload"),
            "unknown": UnsupportedThing(),
            "nested": [{"value": Decimal("2.5")}],
        }

    def cancel_order(self, broker_order_id: str, reason: str | None = None) -> dict:
        self.cancel_calls.append((broker_order_id, reason))
        if self.raise_on == "cancel_order":
            raise RuntimeError("cancel failed")
        return {
            "cancelled": True,
            "broker_order_id": broker_order_id,
            "reason": reason,
            "timestamp": datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        }

    def get_order_status(self, broker_order_id: str) -> dict:
        if self.raise_on == "get_order_status":
            raise RuntimeError("status failed")
        return {
            "broker_order_id": broker_order_id,
            "client_order_id": "intent_1",
            "status": "OPEN",
            "filled_quantity": 0,
            "remaining_quantity": 1,
            "avg_fill_price": Decimal("1.25"),
        }

    def list_open_orders(self) -> list[dict]:
        if self.raise_on == "list_open_orders":
            raise RuntimeError("list open orders failed")
        if self.open_orders_response is not None:
            return self.open_orders_response
        return [self.get_order_status("paper_order_1")]

    def list_positions(self) -> list[dict]:
        if self.raise_on == "list_positions":
            raise RuntimeError("list positions failed")
        if self.positions_response is not None:
            return self.positions_response
        return [
            {
                "symbol": "XSP",
                "quantity": Decimal("1"),
                "avg_price": Decimal("1.2"),
            }
        ]

    def get_account_snapshot(self) -> dict:
        if self.raise_on == "get_account_snapshot":
            raise RuntimeError("account failed")
        return {
            "net_liquidation": Decimal("100000.12"),
            "available_funds": Decimal("90000.00"),
            "buying_power": Decimal("180000.00"),
            "timestamp": datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        }


def valid_intent() -> dict:
    return {
        "intent_id": "intent_1",
        "strategy_id": "S01_VOL_BASELINE",
        "symbol": "XSP",
        "asset_type": "index_option",
        "side": "BUY",
        "quantity": 1,
        "order_type": "LIMIT",
        "limit_price": Decimal("1.25"),
        "time_in_force": "DAY",
        "metadata": {
            "timestamp": datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
            "decimal": Decimal("2.5"),
        },
    }


class PaperBrokerRequestValidationTests(unittest.TestCase):
    def test_build_broker_order_request_validates_required_fields(self) -> None:
        cases = [
            (None, "intent must be a dict"),
            ({}, "strategy_id is required"),
            ({**valid_intent(), "strategy_id": ""}, "strategy_id is required"),
            ({**valid_intent(), "symbol": "", "underlying": ""}, "symbol or underlying"),
            ({**valid_intent(), "side": "HOLD"}, "side must be one of"),
            ({**valid_intent(), "quantity": 0}, "quantity must be positive"),
            ({**valid_intent(), "quantity": -1}, "quantity must be positive"),
            ({**valid_intent(), "order_type": "STOP"}, "order_type must be one of"),
            ({**valid_intent(), "limit_price": None}, "limit_price must be positive"),
            (
                {**valid_intent(), "order_type": "MARKET", "limit_price": 1.25},
                "limit_price is only applicable for LIMIT orders",
            ),
            ({**valid_intent(), "intent_id": ""}, "intent_id is required"),
            ({**valid_intent(), "intent_id": "   "}, "intent_id is required"),
            ({**valid_intent(), "intent_id": 123}, "intent_id is required"),
            ({**valid_intent(), "metadata": object()}, "metadata must be a dict"),
        ]
        for intent, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_broker_order_request(intent)  # type: ignore[arg-type]

    def test_valid_intent_produces_deterministic_request_without_mutation(self) -> None:
        intent = valid_intent()
        before = repr(intent)

        first = build_broker_order_request(intent)
        second = build_broker_order_request(intent)

        self.assertEqual(first, second)
        self.assertEqual(first.client_order_id, "intent_1")
        self.assertEqual(first.intent_id, "intent_1")
        self.assertEqual(first.side, "BUY")
        self.assertEqual(first.order_type, "LIMIT")
        self.assertEqual(first.limit_price, 1.25)
        self.assertEqual(first.asset_type, "INDEX_OPTION")
        self.assertEqual(intent["metadata"]["decimal"], Decimal("2.5"))
        self.assertEqual(repr(intent), before)
        json.dumps(asdict(first), sort_keys=True)

    def test_underlying_can_supply_symbol_and_market_order_does_not_need_limit(self) -> None:
        intent = valid_intent()
        intent.pop("symbol")
        intent["underlying"] = "XSP"
        intent["order_type"] = "MARKET"
        intent.pop("limit_price")

        request = build_broker_order_request(intent)

        self.assertEqual(request.symbol, "XSP")
        self.assertIsNone(request.limit_price)

    def test_market_order_with_limit_price_fails_closed(self) -> None:
        intent = valid_intent()
        intent["order_type"] = "MARKET"
        intent["limit_price"] = Decimal("1.25")

        with self.assertRaisesRegex(
            ValueError,
            "limit_price is only applicable for LIMIT orders",
        ):
            build_broker_order_request(intent)

    def test_client_order_id_generation_source_contains_no_nondeterministic_calls(self) -> None:
        source = inspect.getsource(build_broker_order_request)
        forbidden_tokens = (
            "uuid." + "uuid4",
            "rand" + "om",
            "time" + "stamp",
            "now(",
        )
        for token in forbidden_tokens:
            with self.subTest(token=token):
                self.assertNotIn(token, source)


class PaperBrokerAdapterTests(unittest.TestCase):
    def test_adapter_satisfies_protocol_and_mode_gates_fail_closed(self) -> None:
        self.assertEqual(validate_broker_mode("PAPER"), "PAPER")
        self.assertEqual(validate_broker_mode("DRY_RUN"), "DRY_RUN")
        for mode in ("LIVE", "UNKNOWN"):
            with self.subTest(mode=mode):
                with self.assertRaises(BrokerModeError):
                    validate_broker_mode(mode)
                with self.assertRaises(BrokerModeError):
                    PaperBrokerAdapter(
                        mode=mode,
                        client=FakePaperClient(),
                        allow_live=True,
                    )

        self.assertIsInstance(
            PaperBrokerAdapter(mode="PAPER", client=FakePaperClient()),
            BrokerAdapter,
        )
        self.assertIsInstance(
            PaperBrokerAdapter(mode="DRY_RUN", client=FakePaperClient()),
            BrokerAdapter,
        )

    def test_invalid_submit_intents_return_rejected_result_without_client_call(self) -> None:
        invalid_intents = [
            None,
            {},
            {**valid_intent(), "strategy_id": ""},
            {**valid_intent(), "symbol": ""},
            {**valid_intent(), "side": "HOLD"},
            {**valid_intent(), "quantity": 0},
            {**valid_intent(), "limit_price": None},
            {**valid_intent(), "order_type": "MARKET", "limit_price": 1.25},
        ]
        for intent in invalid_intents:
            with self.subTest(intent=intent):
                client = FakePaperClient()
                result = PaperBrokerAdapter(mode="PAPER", client=client).submit_order_intent(
                    intent  # type: ignore[arg-type]
                )

                self.assertIsInstance(result, BrokerSubmitResult)
                self.assertFalse(result.accepted)
                self.assertFalse(result.dry_run)
                self.assertIsNotNone(result.reason)
                self.assertEqual(client.submitted_requests, [])
                json.dumps(result.to_dict(), sort_keys=True)

    def test_paper_submit_valid_intent_calls_fake_client_once_and_sanitizes_raw(self) -> None:
        client = FakePaperClient()
        adapter = PaperBrokerAdapter(mode="PAPER", client=client)

        result = adapter.submit_order_intent(valid_intent())

        self.assertTrue(result.accepted)
        self.assertFalse(result.dry_run)
        self.assertEqual(result.broker_order_id, "paper_order_1")
        self.assertEqual(result.client_order_id, "intent_1")
        self.assertEqual(len(client.submitted_requests), 1)
        self.assertEqual(client.submitted_requests[0].client_order_id, "intent_1")
        self.assertEqual(result.raw["decimal"], 1.25)
        self.assertEqual(result.raw["timestamp"], "2026-05-06T12:00:00+00:00")
        self.assertEqual(result.raw["date"], "2026-05-06")
        self.assertEqual(result.raw["exception"], "ValueError: bad payload")
        self.assertTrue(str(result.raw["unknown"]).endswith(".UnsupportedThing>"))
        json.dumps(result.to_dict(), sort_keys=True)

    def test_dry_run_submit_is_deterministic_and_does_not_call_client(self) -> None:
        client = FakePaperClient()
        adapter = PaperBrokerAdapter(mode="DRY_RUN", client=client)
        intent = valid_intent()

        first = adapter.submit_order_intent(intent)
        second = adapter.submit_order_intent(intent)

        self.assertEqual(first, second)
        self.assertTrue(first.accepted)
        self.assertTrue(first.dry_run)
        self.assertEqual(first.broker_order_id, "dry_run_intent_1")
        self.assertEqual(client.submitted_requests, [])
        json.dumps(first.to_dict(), sort_keys=True)

    def test_cancel_status_list_positions_and_account_convert_fake_responses(self) -> None:
        client = FakePaperClient()
        adapter = PaperBrokerAdapter(mode="PAPER", client=client)

        cancel = adapter.cancel_order("paper_order_1", reason="operator")
        status = adapter.get_order_status("paper_order_1")
        open_orders = adapter.list_open_orders()
        positions = adapter.list_positions()
        snapshot = adapter.get_account_snapshot()

        self.assertIsInstance(cancel, BrokerCancelResult)
        self.assertTrue(cancel.cancelled)
        self.assertFalse(cancel.dry_run)
        self.assertEqual(client.cancel_calls, [("paper_order_1", "operator")])
        self.assertIsInstance(status, BrokerOrderStatus)
        self.assertEqual(status.status, "OPEN")
        self.assertEqual(status.avg_fill_price, 1.25)
        self.assertIsInstance(open_orders[0], BrokerOrderStatus)
        self.assertIsInstance(positions[0], BrokerPosition)
        self.assertEqual(positions[0].avg_price, 1.2)
        self.assertIsInstance(snapshot, BrokerAccountSnapshot)
        self.assertEqual(snapshot.net_liquidation, 100000.12)
        self.assertEqual(snapshot.timestamp, "2026-05-06T12:00:00+00:00")
        for value in (cancel, status, open_orders[0], positions[0], snapshot):
            json.dumps(value.to_dict(), sort_keys=True)

    def test_successful_empty_list_responses_return_empty_lists(self) -> None:
        client = FakePaperClient()
        adapter = PaperBrokerAdapter(mode="PAPER", client=client)

        client.open_orders_response = []
        self.assertEqual(adapter.list_open_orders(), [])

        client.positions_response = []
        self.assertEqual(adapter.list_positions(), [])

    def test_list_open_orders_client_exception_is_visible(self) -> None:
        client = FakePaperClient()
        client.raise_on = "list_open_orders"
        adapter = PaperBrokerAdapter(mode="PAPER", client=client)

        with self.assertRaisesRegex(RuntimeError, "list open orders failed"):
            adapter.list_open_orders()

    def test_list_positions_client_exception_is_visible(self) -> None:
        client = FakePaperClient()
        client.raise_on = "list_positions"
        adapter = PaperBrokerAdapter(mode="PAPER", client=client)

        with self.assertRaisesRegex(RuntimeError, "list positions failed"):
            adapter.list_positions()

    def test_client_exceptions_convert_to_safe_failure_results(self) -> None:
        client = FakePaperClient()
        adapter = PaperBrokerAdapter(mode="PAPER", client=client)

        client.raise_on = "submit_order"
        submit = adapter.submit_order_intent(valid_intent())
        self.assertFalse(submit.accepted)
        self.assertEqual(submit.reason, "RuntimeError: submit failed")

        client.raise_on = "cancel_order"
        cancel = adapter.cancel_order("paper_order_1")
        self.assertFalse(cancel.cancelled)
        self.assertEqual(cancel.reason, "RuntimeError: cancel failed")

        client.raise_on = "get_order_status"
        status = adapter.get_order_status("paper_order_1")
        self.assertEqual(status.status, "ERROR")
        self.assertEqual(status.raw["error"], "RuntimeError: status failed")

        client.raise_on = "get_account_snapshot"
        snapshot = adapter.get_account_snapshot()
        self.assertEqual(snapshot.raw["error"], "RuntimeError: account failed")
        for value in (submit, cancel, status, snapshot):
            json.dumps(value.to_dict(), sort_keys=True)

    def test_adapter_does_not_mutate_state_store_or_ledger_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_store = StateStore(root / "data/state/portfolio_state.json")
            ledger = LedgerAppender(root)
            state_before = json.dumps(state_store.state, sort_keys=True)
            order_before = ledger.order_path.read_text(encoding="utf-8")
            execution_before = ledger.execution_path.read_text(encoding="utf-8")

            adapter = PaperBrokerAdapter(mode="PAPER", client=FakePaperClient())
            adapter.submit_order_intent(valid_intent())
            adapter.cancel_order("paper_order_1")
            adapter.get_order_status("paper_order_1")
            adapter.list_open_orders()
            adapter.list_positions()
            adapter.get_account_snapshot()

            self.assertEqual(json.dumps(state_store.state, sort_keys=True), state_before)
            self.assertEqual(ledger.order_path.read_text(encoding="utf-8"), order_before)
            self.assertEqual(
                ledger.execution_path.read_text(encoding="utf-8"),
                execution_before,
            )


class PaperBrokerRawSafetyTests(unittest.TestCase):
    def test_recursive_raw_sanitizer_converts_unsupported_values(self) -> None:
        unsafe = {
            "decimal": Decimal("1.25"),
            "datetime": datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
            "date": date(2026, 5, 6),
            "exception": RuntimeError("boom"),
            "unknown": UnsupportedThing(),
            "nested": [{"decimal": Decimal("2.5")}],
            12: "non-string key",
        }

        safe = sanitize_json_safe(unsafe)

        self.assertEqual(safe["decimal"], 1.25)
        self.assertEqual(safe["datetime"], "2026-05-06T12:00:00+00:00")
        self.assertEqual(safe["date"], "2026-05-06")
        self.assertEqual(safe["exception"], "RuntimeError: boom")
        self.assertEqual(safe["nested"][0]["decimal"], 2.5)
        self.assertEqual(safe["12"], "non-string key")
        json.dumps(safe, sort_keys=True)


class Stage4D3SafetyBoundaryTests(unittest.TestCase):
    def test_stage4d3_files_do_not_import_or_call_blocked_integrations(self) -> None:
        blocked_tokens = (
            "ib_" + "insync",
            "req" + "MktData",
            "place" + "Order",
            "y" + "finance",
            "requests",
            "url" + "lib",
            "system" + "ctl",
            "system" + "d",
            "uuid." + "uuid4",
            "rand" + "om",
        )
        for path in STAGE4D3_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_stage4d3_files_do_not_import_broker_client_symbols(self) -> None:
        forbidden_lines = (
            "from ib",
            "import ib",
            " IB(",
            "= IB",
        )
        for path in STAGE4D3_FILES:
            source = path.read_text(encoding="utf-8")
            for token in forbidden_lines:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_paper_adapter_is_not_wired_into_daemon_scheduler_or_lifecycle_jobs(self) -> None:
        paths = [
            ROOT / "tools/daemon.py",
            ROOT / "core/scheduler.py",
            ROOT / "core/scheduler_cadence.py",
            *sorted((ROOT / "jobs").glob("*.py")),
        ]
        for path in paths:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotIn("PaperBrokerAdapter", source)
                self.assertNotIn("paper_broker_adapter", source)


if __name__ == "__main__":
    unittest.main()
