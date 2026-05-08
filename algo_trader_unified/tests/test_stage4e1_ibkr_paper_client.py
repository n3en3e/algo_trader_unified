from __future__ import annotations

import importlib
import inspect
import json
import sys
import unittest
from dataclasses import asdict
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from algo_trader_unified.core.broker_adapter import (
    BrokerAccountSnapshot,
    BrokerOrderStatus,
    BrokerPosition,
)
from algo_trader_unified.core.ibkr_paper_client import (
    IbkrConnectionStatus,
    IbkrPaperClientError,
    IbkrPaperReadOnlyClient,
)
from algo_trader_unified.core.ibkr_paper_order_mapper import (
    IBKR_PAPER_PORT,
    IbkrPaperConfig,
    validate_ibkr_paper_config,
)


ROOT = Path(__file__).resolve().parents[1]
STAGE4E1_CORE_FILES = [
    ROOT / "core/ibkr_paper_client.py",
]
UNWIRED_RUNTIME_FILES = [
    ROOT / "tools/daemon.py",
    ROOT / "core/scheduler.py",
    ROOT / "core/scheduler_cadence.py",
    ROOT / "core/job_chain.py",
]


class UnsupportedThing:
    pass


def valid_config(**overrides: object) -> IbkrPaperConfig:
    config: dict[str, object] = {
        "host": "127.0.0.1",
        "port": IBKR_PAPER_PORT,
        "client_id": 7,
        "account_id": "DU1234567",
        "trading_mode": "PAPER",
        "readonly": False,
    }
    config.update(overrides)
    return validate_ibkr_paper_config(config)


class FakeIb:
    def __init__(self) -> None:
        self.connected = False
        self.connect_calls: list[tuple[str, int, int]] = []
        self.disconnect_calls = 0
        self.current_time_calls = 0
        self.account_summary_calls = 0
        self.open_orders_calls = 0
        self.positions_calls = 0
        self.socket_opened = False
        self.async_loop_started = False
        self.fail_connect = False
        self.fail_open_orders = False
        self.fail_positions = False
        self.fail_account_summary = False
        self.current_time = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)
        self.account_summary = [
            SimpleNamespace(
                account="DU1234567",
                tag="NetLiquidation",
                value=Decimal("100000.25"),
                currency="USD",
            ),
            SimpleNamespace(
                account="DU1234567",
                tag="AvailableFunds",
                value="90000.50",
                currency="USD",
            ),
            SimpleNamespace(
                account="DU1234567",
                tag="BuyingPower",
                value=200000,
                currency="USD",
            ),
            SimpleNamespace(
                account="DU1234567",
                tag="timestamp",
                value=date(2026, 5, 8),
                currency="",
            ),
        ]
        self.open_orders = [
            SimpleNamespace(
                order=SimpleNamespace(orderId=123, orderRef="client-123"),
                orderStatus=SimpleNamespace(
                    status="Submitted",
                    filled=0,
                    remaining=1,
                    avgFillPrice=0,
                ),
                contract=SimpleNamespace(
                    symbol="XSP",
                    secType="OPT",
                    lastTradeDateOrContractMonth="20260619",
                    proprietary=UnsupportedThing(),
                ),
            )
        ]
        self.positions_payload = [
            SimpleNamespace(
                account="DU1234567",
                contract=SimpleNamespace(symbol="XSP", secType="OPT"),
                position=2,
                avgCost=Decimal("1.25"),
            )
        ]

    def connect(self, host: str, port: int, clientId: int) -> bool:
        self.connect_calls.append((host, port, clientId))
        if self.fail_connect:
            raise RuntimeError("gateway down")
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    def isConnected(self) -> bool:
        return self.connected

    def reqCurrentTime(self) -> object:
        self.current_time_calls += 1
        return self.current_time

    def accountSummary(self) -> object:
        self.account_summary_calls += 1
        if self.fail_account_summary:
            raise RuntimeError("account unavailable")
        return self.account_summary

    def openOrders(self) -> object:
        self.open_orders_calls += 1
        if self.fail_open_orders:
            raise RuntimeError("orders unavailable")
        return self.open_orders

    def positions(self) -> object:
        self.positions_calls += 1
        if self.fail_positions:
            raise RuntimeError("positions unavailable")
        return self.positions_payload


class IbkrPaperReadOnlyClientConstructionTests(unittest.TestCase):
    def test_client_construction_accepts_valid_paper_4004_config(self) -> None:
        client = IbkrPaperReadOnlyClient(ib=FakeIb(), config=valid_config())

        self.assertEqual(client.config.trading_mode, "PAPER")
        self.assertEqual(client.config.port, 4004)

    def test_client_construction_rejects_live_config(self) -> None:
        config = IbkrPaperConfig(
            host="127.0.0.1",
            port=4004,
            client_id=7,
            account_id="DU1234567",
            trading_mode="LIVE",
            readonly=False,
        )

        with self.assertRaisesRegex(ValueError, "LIVE is rejected"):
            IbkrPaperReadOnlyClient(ib=FakeIb(), config=config)

    def test_client_construction_rejects_live_port(self) -> None:
        config = IbkrPaperConfig(
            host="127.0.0.1",
            port=4002,
            client_id=7,
            account_id="DU1234567",
            trading_mode="PAPER",
            readonly=False,
        )

        with self.assertRaisesRegex(ValueError, "4002 is rejected"):
            IbkrPaperReadOnlyClient(ib=FakeIb(), config=config)

    def test_client_construction_rejects_unknown_trading_mode(self) -> None:
        config = IbkrPaperConfig(
            host="127.0.0.1",
            port=4004,
            client_id=7,
            account_id="DU1234567",
            trading_mode="UNKNOWN",
            readonly=False,
        )

        with self.assertRaisesRegex(ValueError, 'trading_mode must be exactly "PAPER"'):
            IbkrPaperReadOnlyClient(ib=FakeIb(), config=config)


class IbkrPaperReadOnlyClientOperationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_ib = FakeIb()
        self.client = IbkrPaperReadOnlyClient(ib=self.fake_ib, config=valid_config())

    def test_connect_calls_fake_ib_with_host_port_and_client_id(self) -> None:
        status = self.client.connect()

        self.assertEqual(self.fake_ib.connect_calls, [("127.0.0.1", 4004, 7)])
        self.assertIsInstance(status, IbkrConnectionStatus)
        self.assertTrue(status.connected)
        self.assertTrue(status.paper_mode)
        self.assertIsNone(status.reason)
        json.dumps(asdict(status), sort_keys=True)

    def test_fake_connect_is_local_and_starts_no_socket_or_async_loop(self) -> None:
        self.client.connect()

        self.assertFalse(self.fake_ib.socket_opened)
        self.assertFalse(self.fake_ib.async_loop_started)

    def test_connect_failure_returns_visible_failed_status(self) -> None:
        self.fake_ib.fail_connect = True

        status = self.client.connect()

        self.assertFalse(status.connected)
        self.assertIn("RuntimeError: gateway down", status.reason or "")
        self.assertEqual(status.raw["operation"], "connect")

    def test_disconnect_calls_fake_ib_disconnect(self) -> None:
        self.client.connect()

        status = self.client.disconnect()

        self.assertEqual(self.fake_ib.disconnect_calls, 1)
        self.assertFalse(status.connected)

    def test_get_connection_status_reports_fake_ib_connected_state(self) -> None:
        self.assertFalse(self.client.get_connection_status().connected)

        self.client.connect()

        self.assertTrue(self.client.get_connection_status().connected)

    def test_get_current_time_calls_read_only_method_and_returns_json_safe_result(self) -> None:
        result = self.client.get_current_time()

        self.assertEqual(self.fake_ib.current_time_calls, 1)
        self.assertEqual(result["current_time"], "2026-05-08T12:00:00+00:00")
        json.dumps(result, sort_keys=True)

    def test_get_account_snapshot_converts_fake_account_summary(self) -> None:
        snapshot = self.client.get_account_snapshot()

        self.assertIsInstance(snapshot, BrokerAccountSnapshot)
        self.assertEqual(self.fake_ib.account_summary_calls, 1)
        self.assertEqual(snapshot.net_liquidation, 100000.25)
        self.assertEqual(snapshot.available_funds, 90000.50)
        self.assertEqual(snapshot.buying_power, 200000.0)
        self.assertEqual(snapshot.timestamp, "2026-05-08")
        json.dumps(asdict(snapshot), sort_keys=True)

    def test_get_account_snapshot_failure_returns_visible_failed_snapshot(self) -> None:
        self.fake_ib.fail_account_summary = True

        snapshot = self.client.get_account_snapshot()

        self.assertIsNone(snapshot.net_liquidation)
        self.assertIn("RuntimeError: account unavailable", snapshot.raw["error"])

    def test_list_open_orders_converts_fake_open_orders(self) -> None:
        orders = self.client.list_open_orders()

        self.assertEqual(self.fake_ib.open_orders_calls, 1)
        self.assertEqual(len(orders), 1)
        self.assertIsInstance(orders[0], BrokerOrderStatus)
        self.assertEqual(orders[0].broker_order_id, "123")
        self.assertEqual(orders[0].client_order_id, "client-123")
        self.assertEqual(orders[0].status, "Submitted")
        self.assertEqual(orders[0].filled_quantity, 0)
        self.assertEqual(orders[0].remaining_quantity, 1)
        json.dumps(asdict(orders[0]), sort_keys=True)

    def test_list_positions_converts_fake_positions(self) -> None:
        positions = self.client.list_positions()

        self.assertEqual(self.fake_ib.positions_calls, 1)
        self.assertEqual(len(positions), 1)
        self.assertIsInstance(positions[0], BrokerPosition)
        self.assertEqual(positions[0].symbol, "XSP")
        self.assertEqual(positions[0].quantity, 2)
        self.assertEqual(positions[0].avg_price, 1.25)
        json.dumps(asdict(positions[0]), sort_keys=True)

    def test_list_open_orders_failure_does_not_return_empty_list_silently(self) -> None:
        self.fake_ib.fail_open_orders = True

        with self.assertRaisesRegex(IbkrPaperClientError, "orders unavailable"):
            self.client.list_open_orders()

    def test_list_positions_failure_does_not_return_empty_list_silently(self) -> None:
        self.fake_ib.fail_positions = True

        with self.assertRaisesRegex(IbkrPaperClientError, "positions unavailable"):
            self.client.list_positions()

    def test_raw_fields_are_json_safe_and_do_not_leak_fake_ib_objects(self) -> None:
        order = self.client.list_open_orders()[0]
        position = self.client.list_positions()[0]
        payload = {"order": asdict(order), "position": asdict(position)}

        encoded = json.dumps(payload, sort_keys=True)

        self.assertNotIn("UnsupportedThing object", encoded)
        self.assertNotIn("SimpleNamespace(", encoded)
        self.assertEqual(order.raw["contract"]["symbol"], "XSP")
        self.assertEqual(position.raw["contract"]["symbol"], "XSP")


class IbkrPaperReadOnlyClientSafetyTests(unittest.TestCase):
    def test_public_client_methods_are_synchronous_functions(self) -> None:
        public_methods = [
            "connect",
            "disconnect",
            "get_connection_status",
            "get_current_time",
            "get_account_snapshot",
            "list_open_orders",
            "list_positions",
        ]

        for method_name in public_methods:
            with self.subTest(method_name=method_name):
                method = getattr(IbkrPaperReadOnlyClient, method_name)
                self.assertTrue(inspect.isfunction(method))
                self.assertFalse(inspect.iscoroutinefunction(method))

    def test_public_client_does_not_expose_execution_or_market_data_methods(self) -> None:
        disallowed = (
            "submit_order",
            "place_order",
            "cancel_order",
            "qualify_contract",
            "req_mkt_data",
        )

        for method_name in disallowed:
            with self.subTest(method_name=method_name):
                self.assertFalse(hasattr(IbkrPaperReadOnlyClient, method_name))

    def test_core_module_import_succeeds_without_ib_insync_installed(self) -> None:
        with mock.patch.dict(sys.modules, {"ib" + "_insync": None}):
            module = importlib.reload(
                importlib.import_module("algo_trader_unified.core.ibkr_paper_client")
            )

        self.assertTrue(hasattr(module, "IbkrPaperReadOnlyClient"))

    def test_stage4e1_core_files_do_not_import_or_call_blocked_integrations(self) -> None:
        blocked_tokens = (
            "place" + "Order",
            "req" + "MktData",
            "qualify" + "Contracts",
            "cancel" + "Order",
            "y" + "finance",
            "requests",
            "url" + "lib",
            "system" + "ctl",
            "system" + "d",
            "socket.create" + "_connection",
            "socket." + "socket",
            "asyncio." + "run",
            "asyncio." + "get_event_loop",
            "asyncio." + "new_event_loop",
            "ib" + "_insync",
        )
        for path in STAGE4E1_CORE_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_stage4e1_client_is_not_wired_into_daemon_scheduler_or_lifecycle(self) -> None:
        blocked_tokens = (
            "IbkrPaperReadOnly" + "Client",
            "ibkr_paper" + "_client",
        )
        for path in UNWIRED_RUNTIME_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)
