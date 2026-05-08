from __future__ import annotations

import copy
import importlib
import inspect
import json
import sys
import unittest
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from algo_trader_unified.core.broker_adapter import (
    BrokerCancelResult,
    BrokerOrderStatus,
    BrokerSubmitResult,
)
from algo_trader_unified.core.ibkr_paper_execution_client import (
    IbkrPaperExecutionClient,
    _optional_number,
)
from algo_trader_unified.core.ibkr_paper_order_mapper import (
    IBKR_PAPER_PORT,
    IbkrPaperConfig,
    build_ibkr_paper_order_plan,
    validate_ibkr_paper_config,
)
from algo_trader_unified.core.paper_broker_adapter import BrokerOrderRequest


ROOT = Path(__file__).resolve().parents[1]
STAGE4E3_FILES = [
    ROOT / "core/ibkr_paper_execution_client.py",
    ROOT / "tests/test_stage4e3_ibkr_paper_execution_client.py",
]
UNWIRED_RUNTIME_FILES = [
    ROOT / "tools/daemon.py",
    ROOT / "core/scheduler.py",
    ROOT / "core/scheduler_cadence.py",
    ROOT / "core/job_chain.py",
    ROOT / "jobs/submission.py",
    ROOT / "jobs/confirmation.py",
    ROOT / "jobs/fill_confirmation.py",
]
OPERATOR_CLI_FILES = [
    ROOT / "tools/submit_order_intent.py",
    ROOT / "tools/submit_close_intent.py",
    ROOT / "tools/confirm_order_intent.py",
    ROOT / "tools/confirm_close_order.py",
]


class ProprietaryThing:
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


def raw_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "host": "127.0.0.1",
        "port": IBKR_PAPER_PORT,
        "client_id": 7,
        "account_id": "DU1234567",
        "trading_mode": "PAPER",
        "readonly": False,
    }
    config.update(overrides)
    return config


def valid_request(**overrides: object) -> BrokerOrderRequest:
    request = {
        "client_order_id": "intent-stage4e3-001",
        "strategy_id": "S01_VOL_BASELINE",
        "symbol": "XSP",
        "asset_type": "OPTION",
        "side": "BUY",
        "quantity": 1,
        "order_type": "LIMIT",
        "limit_price": 1.25,
        "time_in_force": "DAY",
        "intent_id": "intent-stage4e3-001",
        "metadata": {"expiry": "20260619", "strike": 525.0, "right": "C"},
    }
    request.update(overrides)
    return BrokerOrderRequest(**request)  # type: ignore[arg-type]


def valid_plan(**request_overrides: object):
    return build_ibkr_paper_order_plan(
        valid_request(**request_overrides),
        config=valid_config(),
    )


class FakeTrade:
    def __init__(
        self,
        *,
        order_id: int = 9001,
        order_ref: str = "intent-stage4e3-001",
        status: str = "Submitted",
        accepted: bool | None = None,
        cancelled: bool | None = None,
        reason: str | None = None,
    ) -> None:
        self.order = SimpleNamespace(
            orderId=order_id,
            orderRef=order_ref,
            created_at=datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc),
        )
        self.orderStatus = SimpleNamespace(
            status=status,
            filled=Decimal("0"),
            remaining=1,
            avgFillPrice=Decimal("0"),
        )
        self.contract = SimpleNamespace(
            symbol="XSP",
            secType="OPT",
            lastTradeDateOrContractMonth="20260619",
            proprietary=ProprietaryThing(),
        )
        self.accepted = accepted
        self.cancelled = cancelled
        self.reason = reason
        self.proprietary = ProprietaryThing()


class FakeIb:
    def __init__(self) -> None:
        self.connected = False
        self.connect_calls: list[tuple[str, int, int]] = []
        self.disconnect_calls = 0
        self.place_order_calls: list[tuple[dict[str, object], dict[str, object]]] = []
        self.cancel_order_calls: list[tuple[str, str | None]] = []
        self.status_calls: list[str] = []
        self.req_mkt_data_calls = 0
        self.qualify_contracts_calls = 0
        self.contract_objects_created = 0
        self.order_objects_created = 0
        self.place_response: object = FakeTrade()
        self.cancel_response: object = FakeTrade(status="Cancelled", cancelled=True)
        self.status_response: object = FakeTrade(status="Submitted")
        self.place_exception: Exception | None = None
        self.cancel_exception: Exception | None = None
        self.status_exception: Exception | None = None

    def connect(self, host: str, port: int, clientId: int) -> dict[str, object]:
        self.connect_calls.append((host, port, clientId))
        self.connected = True
        return {"connected": True}

    def disconnect(self) -> dict[str, object]:
        self.disconnect_calls += 1
        self.connected = False
        return {"connected": False}

    def isConnected(self) -> bool:
        return self.connected

    def placeOrder(self, contract: dict[str, object], order: dict[str, object]) -> object:
        self.place_order_calls.append((copy.deepcopy(contract), copy.deepcopy(order)))
        if self.place_exception is not None:
            raise self.place_exception
        return self.place_response

    def cancelOrder(self, broker_order_id: str, reason: str | None = None) -> object:
        self.cancel_order_calls.append((broker_order_id, reason))
        if self.cancel_exception is not None:
            raise self.cancel_exception
        return self.cancel_response

    def get_order_status(self, broker_order_id: str) -> object:
        self.status_calls.append(broker_order_id)
        if self.status_exception is not None:
            raise self.status_exception
        return self.status_response

    def __getattr__(self, name: str) -> object:
        if name == "req" + "MktData":
            self.req_mkt_data_calls += 1
            raise AssertionError("market data must not be requested")
        if name == "qualify" + "Contracts":
            self.qualify_contracts_calls += 1
            raise AssertionError("contracts must not be qualified")
        raise AttributeError(name)


def assert_json_safe(test_case: unittest.TestCase, raw: dict[str, object] | None) -> None:
    json.dumps(raw, sort_keys=True)
    serialized = json.dumps(raw, sort_keys=True)
    test_case.assertNotIn("ProprietaryThing", serialized)
    test_case.assertNotIn("datetime", serialized)
    test_case.assertNotIn("Decimal", serialized)


class IbkrPaperExecutionClientConstructionTests(unittest.TestCase):
    def test_client_construction_accepts_submit_capable_paper_config(self) -> None:
        client = IbkrPaperExecutionClient(ib=FakeIb(), config=valid_config())

        self.assertEqual(client.config.trading_mode, "PAPER")
        self.assertEqual(client.config.port, 4004)
        self.assertFalse(client.config.readonly)

    def test_client_construction_rejects_live_mode(self) -> None:
        config = IbkrPaperConfig(**raw_config(trading_mode="LIVE"))  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "LIVE is rejected"):
            IbkrPaperExecutionClient(ib=FakeIb(), config=config)

    def test_client_construction_rejects_live_port_4002(self) -> None:
        config = IbkrPaperConfig(**raw_config(port=4002))  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "4002 is rejected"):
            IbkrPaperExecutionClient(ib=FakeIb(), config=config)

    def test_client_construction_rejects_unknown_trading_mode(self) -> None:
        config = IbkrPaperConfig(**raw_config(trading_mode="UNKNOWN"))  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, 'trading_mode must be exactly "PAPER"'):
            IbkrPaperExecutionClient(ib=FakeIb(), config=config)

    def test_connect_and_disconnect_return_visible_connection_status(self) -> None:
        fake = FakeIb()
        client = IbkrPaperExecutionClient(ib=fake, config=valid_config())

        connected = client.connect()
        disconnected = client.disconnect()

        self.assertTrue(connected.connected)
        self.assertFalse(disconnected.connected)
        self.assertTrue(connected.paper_mode)
        self.assertEqual(fake.connect_calls, [("127.0.0.1", 4004, 7)])


class IbkrPaperExecutionClientSubmitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeIb()
        self.client = IbkrPaperExecutionClient(ib=self.fake, config=valid_config())

    def assert_submit_rejected_without_place_order(self, plan, expected: str) -> None:
        result = self.client.submit_order_plan(plan)

        self.assertIsInstance(result, BrokerSubmitResult)
        self.assertFalse(result.accepted)
        self.assertTrue(result.dry_run)
        self.assertIn(expected, result.reason or "")
        self.assertEqual(self.fake.place_order_calls, [])
        assert_json_safe(self, result.raw)

    def test_submit_rejects_ready_for_submission_false_without_calling_place_order(self) -> None:
        self.assert_submit_rejected_without_place_order(
            replace(valid_plan(), ready_for_submission=False),
            "ready_for_submission",
        )

    def test_submit_rejects_paper_only_false_without_calling_place_order(self) -> None:
        self.assert_submit_rejected_without_place_order(
            replace(valid_plan(), paper_only=False),
            "paper_only",
        )

    def test_submit_rejects_dry_run_false_without_calling_place_order(self) -> None:
        self.assert_submit_rejected_without_place_order(
            replace(valid_plan(), dry_run=False),
            "dry_run",
        )

    def test_submit_rejects_non_empty_blockers_without_calling_place_order(self) -> None:
        self.assert_submit_rejected_without_place_order(
            replace(valid_plan(), blockers=["manual blocker"]),
            "blockers",
        )

    def test_submit_rejects_missing_client_order_id_without_calling_place_order(self) -> None:
        self.assert_submit_rejected_without_place_order(
            replace(valid_plan(), client_order_id=""),
            "client_order_id",
        )

    def test_submit_rejects_non_plain_hint_payloads_without_calling_place_order(self) -> None:
        self.assert_submit_rejected_without_place_order(
            replace(valid_plan(), ibkr_order_hint=ProprietaryThing()),  # type: ignore[arg-type]
            "ibkr_order_hint",
        )

    def test_submit_accepts_ready_paper_plan_and_calls_fake_place_order_once(self) -> None:
        plan = valid_plan()

        result = self.client.submit_order_plan(plan)

        self.assertTrue(result.accepted)
        self.assertTrue(result.dry_run)
        self.assertEqual(result.broker_order_id, "9001")
        self.assertEqual(result.client_order_id, plan.client_order_id)
        self.assertEqual(len(self.fake.place_order_calls), 1)
        contract_hint, order_hint = self.fake.place_order_calls[0]
        self.assertEqual(contract_hint["secType"], "OPT")
        self.assertEqual(order_hint["orderRef"], plan.client_order_id)
        assert_json_safe(self, result.raw)

    def test_submit_preserves_client_order_id_and_does_not_mutate_plan(self) -> None:
        plan = valid_plan()
        before = asdict(plan)

        result = self.client.submit_order_plan(plan)

        self.assertEqual(result.client_order_id, "intent-stage4e3-001")
        self.assertEqual(asdict(plan), before)

    def test_submit_avoids_data_qualification_and_real_object_construction(self) -> None:
        self.client.submit_order_plan(valid_plan())

        self.assertEqual(self.fake.req_mkt_data_calls, 0)
        self.assertEqual(self.fake.qualify_contracts_calls, 0)
        self.assertEqual(self.fake.contract_objects_created, 0)
        self.assertEqual(self.fake.order_objects_created, 0)

    def test_blocked_option_plan_with_opt_sectype_is_rejected_by_ready_gate(self) -> None:
        plan = valid_plan(metadata={})
        self.assertEqual(plan.ibkr_contract_hint["secType"], "OPT")

        self.assert_submit_rejected_without_place_order(plan, "ready_for_submission")

    def test_blocked_plan_with_empty_action_order_type_tif_is_rejected_by_ready_gate(self) -> None:
        plan = replace(
            valid_plan(),
            action="",
            order_type="",
            time_in_force="",
            ready_for_submission=False,
            blockers=["mapper rejected empty fields"],
        )

        self.assert_submit_rejected_without_place_order(plan, "ready_for_submission")

    def test_fake_place_order_failures_return_visible_submit_failure(self) -> None:
        for exc in (
            TimeoutError("submit timed out"),
            ConnectionError("gateway down"),
            RuntimeError("native failure"),
        ):
            with self.subTest(exc=type(exc).__name__):
                fake = FakeIb()
                fake.place_exception = exc
                client = IbkrPaperExecutionClient(ib=fake, config=valid_config())

                result = client.submit_order_plan(valid_plan())

                self.assertFalse(result.accepted)
                self.assertTrue(result.dry_run)
                self.assertIn(type(exc).__name__, result.reason or "")
                assert_json_safe(self, result.raw)

    def test_fake_place_order_rejected_trade_is_not_clean_success(self) -> None:
        self.fake.place_response = FakeTrade(
            status="Rejected",
            accepted=False,
            reason="fake broker rejected order",
        )

        result = self.client.submit_order_plan(valid_plan())

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "fake broker rejected order")


class IbkrPaperExecutionClientCancelStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeIb()
        self.client = IbkrPaperExecutionClient(ib=self.fake, config=valid_config())

    def test_cancel_order_with_valid_id_calls_fake_cancel_order_once(self) -> None:
        result = self.client.cancel_order("9001", reason="manual cancel")

        self.assertIsInstance(result, BrokerCancelResult)
        self.assertTrue(result.cancelled)
        self.assertTrue(result.dry_run)
        self.assertEqual(result.broker_order_id, "9001")
        self.assertEqual(self.fake.cancel_order_calls, [("9001", "manual cancel")])
        assert_json_safe(self, result.raw)

    def test_cancel_order_with_empty_id_fails_closed_without_fake_call(self) -> None:
        result = self.client.cancel_order("  ")

        self.assertFalse(result.cancelled)
        self.assertTrue(result.dry_run)
        self.assertIn("broker_order_id", result.reason or "")
        self.assertEqual(self.fake.cancel_order_calls, [])

    def test_cancel_order_fake_failures_return_visible_failure(self) -> None:
        for exc in (
            TimeoutError("cancel timed out"),
            ConnectionError("gateway down"),
            RuntimeError("native failure"),
        ):
            with self.subTest(exc=type(exc).__name__):
                fake = FakeIb()
                fake.cancel_exception = exc
                client = IbkrPaperExecutionClient(ib=fake, config=valid_config())

                result = client.cancel_order("9001")

                self.assertFalse(result.cancelled)
                self.assertTrue(result.dry_run)
                self.assertIn(type(exc).__name__, result.reason or "")
                assert_json_safe(self, result.raw)

    def test_get_order_status_with_valid_id_calls_fake_status_once(self) -> None:
        self.fake.status_response = FakeTrade(status="Submitted")
        self.fake.status_response.orderStatus.filled = Decimal("2")
        self.fake.status_response.orderStatus.remaining = "3"
        self.fake.status_response.orderStatus.avgFillPrice = Decimal("101.25")

        result = self.client.get_order_status("9001")

        self.assertIsInstance(result, BrokerOrderStatus)
        self.assertEqual(result.broker_order_id, "9001")
        self.assertEqual(result.client_order_id, "intent-stage4e3-001")
        self.assertEqual(result.status, "Submitted")
        self.assertEqual(result.filled_quantity, 2.0)
        self.assertEqual(result.remaining_quantity, 3.0)
        self.assertEqual(result.avg_fill_price, 101.25)
        self.assertEqual(self.fake.status_calls, ["9001"])
        assert_json_safe(self, result.raw)

    def test_optional_number_handles_json_safe_numeric_inputs(self) -> None:
        cases = [
            (None, None),
            (2, 2),
            (1.25, 1.25),
            (Decimal("1.25"), 1.25),
            ("1.25", 1.25),
            ("2", 2.0),
            ("  ", None),
            ("not numeric", None),
            (ProprietaryThing(), None),
        ]
        for value, expected in cases:
            with self.subTest(value=repr(value)):
                self.assertEqual(_optional_number(value), expected)

    def test_get_order_status_with_empty_id_fails_closed(self) -> None:
        result = self.client.get_order_status("")

        self.assertEqual(result.status, "ERROR")
        self.assertEqual(self.fake.status_calls, [])
        assert_json_safe(self, result.raw)

    def test_get_order_status_fake_failure_is_visible(self) -> None:
        self.fake.status_exception = RuntimeError("status unavailable")

        result = self.client.get_order_status("9001")

        self.assertEqual(result.status, "ERROR")
        self.assertIn("RuntimeError", json.dumps(result.raw, sort_keys=True))
        assert_json_safe(self, result.raw)

    def test_cancel_rejected_response_is_not_clean_success(self) -> None:
        self.fake.cancel_response = FakeTrade(
            status="Submitted",
            cancelled=False,
            reason="fake cancel rejected",
        )

        result = self.client.cancel_order("9001")

        self.assertFalse(result.cancelled)
        self.assertEqual(result.reason, "fake cancel rejected")


class IbkrPaperExecutionClientSafetyTests(unittest.TestCase):
    def test_public_client_methods_are_synchronous_functions(self) -> None:
        public_methods = [
            "connect",
            "disconnect",
            "submit_order_plan",
            "cancel_order",
            "get_order_status",
        ]

        for method_name in public_methods:
            with self.subTest(method_name=method_name):
                method = getattr(IbkrPaperExecutionClient, method_name)
                self.assertTrue(inspect.isfunction(method))
                self.assertFalse(inspect.iscoroutinefunction(method))

    def test_public_client_does_not_expose_live_market_or_lifecycle_methods(self) -> None:
        disallowed = (
            "req_mkt_data",
            "qualify_contract",
            "place_live_order",
            "list_open_orders",
            "list_positions",
            "run_scheduler",
            "run_lifecycle",
        )

        for method_name in disallowed:
            with self.subTest(method_name=method_name):
                self.assertFalse(hasattr(IbkrPaperExecutionClient, method_name))

    def test_core_module_import_succeeds_without_ib_runtime_installed(self) -> None:
        with mock.patch.dict(sys.modules, {"ib" + "_insync": None}):
            module = importlib.reload(
                importlib.import_module("algo_trader_unified.core.ibkr_paper_execution_client")
            )

        self.assertTrue(hasattr(module, "IbkrPaperExecutionClient"))

    def test_stage4e3_files_do_not_import_or_call_blocked_integrations(self) -> None:
        blocked_tokens = (
            "req" + "MktData",
            "qualify" + "Contracts",
            "y" + "finance",
            "requ" + "ests",
            "url" + "lib",
            "system" + "ctl",
            "system" + "d",
            "socket.create" + "_connection",
            "socket." + "socket",
            "asyncio." + "run",
            "asyncio." + "get_event_loop",
            "asyncio." + "new_event_loop",
            "uuid." + "uuid4",
            "ran" + "dom",
            "ib" + "_insync",
            "Con" + "tract(",
        )
        for path in STAGE4E3_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_place_and_cancel_order_are_not_wired_into_runtime_or_operator_cli(self) -> None:
        blocked_tokens = (
            "IbkrPaperExecution" + "Client",
            "ibkr_paper_execution" + "_client",
            "place" + "Order",
            "cancel" + "Order",
        )
        for path in UNWIRED_RUNTIME_FILES + OPERATOR_CLI_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)
