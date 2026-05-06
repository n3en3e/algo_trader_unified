from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from algo_trader_unified.core import broker_adapter as broker_adapter_module
from algo_trader_unified.core.broker_adapter import (
    BrokerAccountSnapshot,
    BrokerAdapter,
    BrokerCancelResult,
    BrokerOrderStatus,
    BrokerPosition,
    BrokerRawValueError,
    BrokerSubmitResult,
)
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.paper_broker_contract import (
    BrokerModeError,
    NullBrokerAdapter,
    assert_broker_adapter,
    validate_broker_mode,
)
from algo_trader_unified.core.state_store import StateStore


ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = [
    ROOT / "core/broker_adapter.py",
    ROOT / "core/paper_broker_contract.py",
]


class BrokerModeContractTests(unittest.TestCase):
    def test_validate_broker_mode_allows_dry_run_and_paper(self) -> None:
        self.assertEqual(validate_broker_mode("DRY_RUN"), "DRY_RUN")
        self.assertEqual(validate_broker_mode("PAPER"), "PAPER")

    def test_validate_broker_mode_rejects_live_and_unknown_fail_closed(self) -> None:
        for mode in ("LIVE", "UNKNOWN", "dry_run", "", "Paper"):
            with self.subTest(mode=mode):
                with self.assertRaises(BrokerModeError):
                    validate_broker_mode(mode)

    def test_null_adapter_constructor_rejects_live_mode(self) -> None:
        with self.assertRaises(BrokerModeError):
            NullBrokerAdapter(mode="LIVE")


class BrokerResultShapeTests(unittest.TestCase):
    def assert_json_safe(self, value: object) -> None:
        json.dumps(asdict(value), sort_keys=True)

    def test_result_dataclasses_are_json_safe_with_populated_raw(self) -> None:
        raw = {
            "string": "value",
            "integer": 1,
            "float": 1.25,
            "boolean": True,
            "none": None,
            "list": ["nested", 2, False],
            "dict": {"child": "safe"},
        }
        values = [
            BrokerSubmitResult(
                accepted=True,
                dry_run=True,
                broker_order_id="order_1",
                client_order_id="client_1",
                reason=None,
                raw=raw,
            ),
            BrokerCancelResult(
                cancelled=True,
                dry_run=True,
                broker_order_id="order_1",
                reason="operator_cancelled",
                raw=raw,
            ),
            BrokerOrderStatus(
                broker_order_id="order_1",
                client_order_id="client_1",
                status="OPEN",
                filled_quantity=0,
                remaining_quantity=1,
                avg_fill_price=None,
                raw=raw,
            ),
            BrokerPosition(symbol="XSP", quantity=1, avg_price=1.2, raw=raw),
            BrokerAccountSnapshot(
                net_liquidation=1000.0,
                available_funds=900.0,
                buying_power=1800.0,
                timestamp="2026-05-06T12:00:00+00:00",
                raw=raw,
            ),
        ]
        for value in values:
            with self.subTest(type=type(value).__name__):
                self.assert_json_safe(value)

    def test_raw_rejects_non_serialized_objects(self) -> None:
        unsafe_values = [
            {"value": Decimal("1.2")},
            {"value": datetime(2026, 5, 6, tzinfo=timezone.utc)},
            {"value": object()},
            {123: "non-string key"},
        ]
        for raw in unsafe_values:
            with self.subTest(raw=raw):
                with self.assertRaises(BrokerRawValueError):
                    BrokerSubmitResult(
                        accepted=True,
                        dry_run=True,
                        broker_order_id="order_1",
                        client_order_id=None,
                        reason=None,
                        raw=raw,
                    )

    def test_interface_docstrings_require_serialized_raw_payloads(self) -> None:
        interface_doc = inspect.getdoc(BrokerAdapter) or ""
        raw_helper_doc = inspect.getdoc(broker_adapter_module.assert_json_safe_raw) or ""
        module_doc = inspect.getdoc(broker_adapter_module) or ""
        combined = " ".join([interface_doc, raw_helper_doc, module_doc]).lower()
        self.assertIn("serialize", combined)
        self.assertIn("proprietary broker objects", combined)
        self.assertIn("raw", combined)


class NullBrokerAdapterContractTests(unittest.TestCase):
    def test_null_adapter_satisfies_protocol(self) -> None:
        adapter = NullBrokerAdapter()
        self.assertIsInstance(adapter, BrokerAdapter)
        self.assertIs(assert_broker_adapter(adapter), adapter)

    def test_submit_order_intent_returns_deterministic_result(self) -> None:
        adapter = NullBrokerAdapter()
        intent = {"intent_id": "intent_1", "client_order_id": "client_custom", "symbol": "XSP"}

        first = adapter.submit_order_intent(intent)
        second = adapter.submit_order_intent(intent)

        self.assertEqual(first, second)
        self.assertIsInstance(first, BrokerSubmitResult)
        self.assertTrue(first.accepted)
        self.assertTrue(first.dry_run)
        self.assertEqual(first.broker_order_id, "null_order_0001")
        self.assertEqual(first.client_order_id, "client_custom")
        json.dumps(first.to_dict(), sort_keys=True)

    def test_submit_order_intent_can_reject_only_by_constructor_parameter(self) -> None:
        adapter = NullBrokerAdapter(accept_submissions=False)

        result = adapter.submit_order_intent({"intent_id": "intent_1"})

        self.assertFalse(result.accepted)
        self.assertTrue(result.dry_run)
        self.assertIsNone(result.broker_order_id)
        self.assertEqual(result.reason, "null_adapter_rejected")

    def test_cancel_order_returns_deterministic_result(self) -> None:
        adapter = NullBrokerAdapter(cancel_succeeds=True)

        result = adapter.cancel_order("order_1", reason="test_cancel")

        self.assertIsInstance(result, BrokerCancelResult)
        self.assertTrue(result.cancelled)
        self.assertTrue(result.dry_run)
        self.assertEqual(result.broker_order_id, "order_1")
        self.assertEqual(result.reason, "test_cancel")
        json.dumps(result.to_dict(), sort_keys=True)

    def test_get_order_status_returns_deterministic_status(self) -> None:
        adapter = NullBrokerAdapter()

        result = adapter.get_order_status("order_1")

        self.assertIsInstance(result, BrokerOrderStatus)
        self.assertEqual(result.broker_order_id, "order_1")
        self.assertEqual(result.status, "OPEN")
        self.assertEqual(result.filled_quantity, 0)
        json.dumps(result.to_dict(), sort_keys=True)

    def test_list_open_orders_positions_and_account_snapshot_shapes(self) -> None:
        adapter = NullBrokerAdapter()

        open_orders = adapter.list_open_orders()
        positions = adapter.list_positions()
        snapshot = adapter.get_account_snapshot()

        self.assertIsInstance(open_orders, list)
        self.assertIsInstance(open_orders[0], BrokerOrderStatus)
        self.assertIsInstance(positions, list)
        self.assertEqual(positions, [])
        self.assertIsInstance(snapshot, BrokerAccountSnapshot)
        json.dumps(snapshot.to_dict(), sort_keys=True)

    def test_null_adapter_does_not_mutate_state_store_or_ledger_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_store = StateStore(root / "data/state/portfolio_state.json")
            ledger = LedgerAppender(root)
            state_before = json.dumps(state_store.state, sort_keys=True)
            order_before = ledger.order_path.read_text(encoding="utf-8")
            execution_before = ledger.execution_path.read_text(encoding="utf-8")

            adapter = NullBrokerAdapter()
            adapter.submit_order_intent({"intent_id": "intent_1"})
            adapter.cancel_order("order_1")
            adapter.get_order_status("order_1")
            adapter.list_open_orders()
            adapter.list_positions()
            adapter.get_account_snapshot()

            self.assertEqual(json.dumps(state_store.state, sort_keys=True), state_before)
            self.assertEqual(ledger.order_path.read_text(encoding="utf-8"), order_before)
            self.assertEqual(
                ledger.execution_path.read_text(encoding="utf-8"),
                execution_before,
            )


class Stage4D1SafetyBoundaryTests(unittest.TestCase):
    def test_stage4d1_core_files_do_not_import_or_call_blocked_integrations(self) -> None:
        blocked_tokens = (
            "ib_" + "insync",
            "req" + "MktData",
            "place" + "Order",
            "y" + "finance",
            "requests",
            "url" + "lib",
            "system" + "ctl",
            "system" + "d",
        )
        for path in CORE_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_stage4d1_core_files_do_not_import_broker_client_symbols(self) -> None:
        for path in CORE_FILES:
            source = path.read_text(encoding="utf-8")
            forbidden_lines = (
                "from ib",
                "import ib",
                " IB(",
                "= IB",
            )
            for token in forbidden_lines:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
