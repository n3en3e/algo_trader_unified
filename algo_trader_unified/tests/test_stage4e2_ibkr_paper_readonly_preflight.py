from __future__ import annotations

import argparse
import importlib
import inspect
import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from algo_trader_unified.core.broker_adapter import (
    BrokerAccountSnapshot,
    BrokerOrderStatus,
    BrokerPosition,
)
from algo_trader_unified.core.ibkr_paper_order_mapper import (
    IBKR_PAPER_PORT,
    validate_ibkr_paper_config,
    validate_ibkr_paper_readonly_config,
)
from algo_trader_unified.core.ibkr_paper_readonly_preflight import (
    build_ibkr_paper_readonly_preflight_report,
)
from algo_trader_unified.tools import ibkr_paper_readonly_preflight as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4E2_FILES = [
    ROOT / "core/ibkr_paper_readonly_preflight.py",
    ROOT / "tools/ibkr_paper_readonly_preflight.py",
]
UNWIRED_RUNTIME_FILES = [
    ROOT / "tools/daemon.py",
    ROOT / "core/scheduler.py",
    ROOT / "core/scheduler_cadence.py",
    ROOT / "core/job_chain.py",
]


def fixed_now() -> datetime:
    return datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


def valid_config(**overrides: object):
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


def valid_readonly_config(**overrides: object):
    config: dict[str, object] = {
        "host": "127.0.0.1",
        "port": IBKR_PAPER_PORT,
        "client_id": 7,
        "account_id": "DU1234567",
        "trading_mode": "PAPER",
        "readonly": True,
    }
    config.update(overrides)
    return validate_ibkr_paper_readonly_config(config)


@dataclass(frozen=True)
class FakeStatus:
    connected: bool
    paper_mode: bool = True
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "connected": self.connected,
            "paper_mode": self.paper_mode,
            "reason": self.reason,
        }


class FakeClient:
    def __init__(self) -> None:
        self.config = valid_readonly_config()
        self.calls: list[str] = []
        self.connected = False
        self.fail_connect: Exception | None = None
        self.connect_status = FakeStatus(True)
        self.status = FakeStatus(True)
        self.fail_current_time: Exception | None = None
        self.fail_account: Exception | None = None
        self.account_error_snapshot = False
        self.fail_open_orders: Exception | None = None
        self.fail_positions: Exception | None = None
        self.fail_disconnect: Exception | None = None
        self.socket_opened = False
        self.async_loop_started = False

    def connect(self) -> FakeStatus:
        self.calls.append("connect")
        if self.fail_connect:
            raise self.fail_connect
        self.connected = self.connect_status.connected
        return self.connect_status

    def get_connection_status(self) -> FakeStatus:
        self.calls.append("get_connection_status")
        return self.status

    def get_current_time(self) -> dict:
        self.calls.append("get_current_time")
        if self.fail_current_time:
            raise self.fail_current_time
        return {"current_time": "2026-05-08T12:00:00+00:00"}

    def get_account_snapshot(self) -> BrokerAccountSnapshot:
        self.calls.append("get_account_snapshot")
        if self.fail_account:
            raise self.fail_account
        if self.account_error_snapshot:
            return BrokerAccountSnapshot(
                net_liquidation=None,
                available_funds=None,
                buying_power=None,
                timestamp=None,
                raw={"operation": "accountSummary", "error": "RuntimeError: account unavailable"},
            )
        return BrokerAccountSnapshot(
            net_liquidation=100000.25,
            available_funds=90000.5,
            buying_power=200000.0,
            timestamp="2026-05-08T12:00:00+00:00",
            raw={"operation": "accountSummary"},
        )

    def list_open_orders(self) -> list[BrokerOrderStatus]:
        self.calls.append("list_open_orders")
        if self.fail_open_orders:
            raise self.fail_open_orders
        return [
            BrokerOrderStatus(
                broker_order_id="123",
                client_order_id="client-123",
                status="Submitted",
                filled_quantity=0,
                remaining_quantity=1,
                avg_fill_price=0,
                raw={"status": "Submitted"},
            )
        ]

    def list_positions(self) -> list[BrokerPosition]:
        self.calls.append("list_positions")
        if self.fail_positions:
            raise self.fail_positions
        return [
            BrokerPosition(
                symbol="XSP",
                quantity=2,
                avg_price=1.25,
                raw={"symbol": "XSP"},
            )
        ]

    def disconnect(self) -> FakeStatus:
        self.calls.append("disconnect")
        if self.fail_disconnect:
            raise self.fail_disconnect
        self.connected = False
        return FakeStatus(False)


class IbkrPaperReadonlyPreflightCoreTests(unittest.TestCase):
    def build(self, client: FakeClient) -> dict:
        return build_ibkr_paper_readonly_preflight_report(
            client=client,
            now_provider=fixed_now,
        )

    def test_success_report_calls_readonly_methods_in_safe_order(self) -> None:
        client = FakeClient()

        report = self.build(client)

        self.assertEqual(
            client.calls,
            [
                "connect",
                "get_connection_status",
                "get_current_time",
                "get_account_snapshot",
                "list_open_orders",
                "list_positions",
                "disconnect",
            ],
        )
        self.assertTrue(report["dry_run"])
        self.assertTrue(report["ibkr_paper_readonly_preflight"])
        self.assertTrue(report["success"])
        self.assertEqual(report["generated_at"], "2026-05-08T12:00:00+00:00")
        self.assertEqual(report["open_orders"]["count"], 1)
        self.assertEqual(report["positions"]["count"], 1)
        self.assertTrue(report["readonly_checks"]["disconnect_ok"])
        json.dumps(report, sort_keys=True)

    def test_failed_connect_does_not_attempt_read_calls(self) -> None:
        client = FakeClient()
        client.connect_status = FakeStatus(False, reason="gateway down")

        report = self.build(client)

        self.assertEqual(client.calls, ["connect", "disconnect"])
        self.assertFalse(report["success"])
        self.assertFalse(report["connection"]["connected"])
        self.assertIn("gateway down", report["connection"]["reason"])
        self.assertIn("connect failed", report["errors"][0])

    def test_connect_timeout_is_json_failure_and_does_not_hang(self) -> None:
        client = FakeClient()
        client.fail_connect = TimeoutError("connect timed out")

        report = self.build(client)

        self.assertEqual(client.calls, ["connect", "disconnect"])
        self.assertFalse(report["success"])
        self.assertIn("TimeoutError: connect timed out", json.dumps(report, sort_keys=True))

    def test_disconnected_status_after_successful_connect_stops_read_calls(self) -> None:
        client = FakeClient()
        client.status = FakeStatus(False, reason="status disconnected after connect")

        report = self.build(client)

        self.assertEqual(client.calls, ["connect", "get_connection_status", "disconnect"])
        self.assertFalse(report["success"])
        self.assertFalse(report["connection"]["connected"])
        self.assertEqual(report["connection"]["reason"], "status disconnected after connect")
        self.assertIn(
            "connection status failed: status disconnected after connect",
            report["errors"],
        )
        self.assertFalse(report["readonly_checks"]["account_snapshot_ok"])
        self.assertFalse(report["readonly_checks"]["open_orders_ok"])
        self.assertFalse(report["readonly_checks"]["positions_ok"])

    def test_current_time_failure_does_not_block_later_readonly_reads(self) -> None:
        client = FakeClient()
        client.fail_current_time = TimeoutError("current time timed out")

        report = self.build(client)

        self.assertEqual(
            client.calls,
            [
                "connect",
                "get_connection_status",
                "get_current_time",
                "get_account_snapshot",
                "list_open_orders",
                "list_positions",
                "disconnect",
            ],
        )
        self.assertFalse(report["success"])
        self.assertIn(
            "current_time failed: TimeoutError: current time timed out",
            report["errors"],
        )
        self.assertFalse(report["readonly_checks"]["current_time_ok"])
        self.assertTrue(report["readonly_checks"]["account_snapshot_ok"])
        self.assertTrue(report["readonly_checks"]["open_orders_ok"])
        self.assertTrue(report["readonly_checks"]["positions_ok"])
        self.assertTrue(report["readonly_checks"]["disconnect_ok"])

    def test_account_snapshot_exception_is_reported_not_crashed(self) -> None:
        client = FakeClient()
        client.fail_account = TimeoutError("account timed out")

        report = self.build(client)

        self.assertFalse(report["success"])
        self.assertFalse(report["readonly_checks"]["account_snapshot_ok"])
        self.assertFalse(report["account_snapshot"]["available"])
        self.assertIn("TimeoutError: account timed out", json.dumps(report, sort_keys=True))
        self.assertEqual(client.calls[-1], "disconnect")

    def test_account_snapshot_error_payload_is_reported_not_clean_success(self) -> None:
        client = FakeClient()
        client.account_error_snapshot = True

        report = self.build(client)

        self.assertFalse(report["success"])
        self.assertFalse(report["account_snapshot"]["available"])
        self.assertIn("RuntimeError: account unavailable", json.dumps(report, sort_keys=True))

    def test_open_orders_failure_is_not_converted_to_clean_empty_state(self) -> None:
        client = FakeClient()
        client.fail_open_orders = RuntimeError("orders unavailable")

        report = self.build(client)

        self.assertFalse(report["success"])
        self.assertFalse(report["open_orders"]["available"])
        self.assertEqual(report["open_orders"]["count"], 0)
        self.assertEqual(report["open_orders"]["failure_reason"], "RuntimeError: orders unavailable")
        self.assertFalse(report["readonly_checks"]["open_orders_ok"])

    def test_positions_failure_is_not_converted_to_clean_empty_state(self) -> None:
        client = FakeClient()
        client.fail_positions = RuntimeError("positions unavailable")

        report = self.build(client)

        self.assertFalse(report["success"])
        self.assertFalse(report["positions"]["available"])
        self.assertEqual(report["positions"]["failure_reason"], "RuntimeError: positions unavailable")
        self.assertFalse(report["readonly_checks"]["positions_ok"])

    def test_disconnect_failure_is_reported(self) -> None:
        client = FakeClient()
        client.fail_disconnect = RuntimeError("disconnect unavailable")

        report = self.build(client)

        self.assertFalse(report["success"])
        self.assertFalse(report["readonly_checks"]["disconnect_ok"])
        self.assertIn("RuntimeError: disconnect unavailable", json.dumps(report, sort_keys=True))

    def test_read_failure_and_disconnect_failure_are_both_visible(self) -> None:
        client = FakeClient()
        client.fail_open_orders = TimeoutError("orders timed out")
        client.fail_disconnect = RuntimeError("disconnect unavailable")

        report = self.build(client)

        self.assertFalse(report["success"])
        self.assertIn(
            "open_orders failed: TimeoutError: orders timed out",
            report["errors"],
        )
        self.assertIn(
            "disconnect failed: RuntimeError: disconnect unavailable",
            report["errors"],
        )
        self.assertNotIn(
            "disconnect failed: RuntimeError: disconnect unavailable",
            report["warnings"],
        )
        self.assertFalse(report["readonly_checks"]["disconnect_ok"])
        self.assertEqual(report["open_orders"]["failure_reason"], "TimeoutError: orders timed out")

    def test_required_fields_and_safety_flags_are_present(self) -> None:
        report = self.build(FakeClient())

        for key in (
            "dry_run",
            "ibkr_paper_readonly_preflight",
            "generated_at",
            "config",
            "connection",
            "readonly_checks",
            "account_snapshot",
            "open_orders",
            "positions",
            "safety",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        safety = report["safety"]
        self.assertTrue(report["config"]["readonly"])
        self.assertTrue(report["config"]["paper_config_valid"])
        self.assertTrue(safety["broker_calls_enabled"])
        self.assertFalse(safety["order_submission_enabled"])
        self.assertFalse(safety["cancel_enabled"])
        self.assertFalse(safety["market_data_enabled"])
        self.assertFalse(safety["contract_qualification_enabled"])
        self.assertFalse(safety["live_orders_enabled"])
        self.assertFalse(safety["scheduler_changes_enabled"])

    def test_recommendations_are_deterministic_and_non_binding(self) -> None:
        first = self.build(FakeClient())["recommendations"]
        second = self.build(FakeClient())["recommendations"]

        self.assertEqual(first, second)
        encoded = json.dumps(first, sort_keys=True)
        self.assertNotIn("place", encoded.lower())
        self.assertNotIn("live trading", encoded.lower())
        self.assertNotIn("threshold", encoded.lower())
        self.assertNotIn("sizing", encoded.lower())
        self.assertNotIn("cadence", encoded.lower())


class IbkrPaperReadonlyPreflightCliTests(unittest.TestCase):
    def test_readonly_preflight_config_accepts_readonly_true(self) -> None:
        config = valid_readonly_config()

        self.assertTrue(config.readonly)
        self.assertEqual(config.trading_mode, "PAPER")
        self.assertEqual(config.port, 4004)

    def test_readonly_preflight_config_rejects_readonly_false(self) -> None:
        with self.assertRaisesRegex(ValueError, "readonly must be True"):
            valid_readonly_config(readonly=False)

    def test_readonly_preflight_config_rejects_live_live_port_and_unknown_mode(self) -> None:
        rejected = (
            {"trading_mode": "LIVE"},
            {"port": 4002},
            {"trading_mode": "UNKNOWN"},
        )
        for override in rejected:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    valid_readonly_config(**override)

    def test_cli_requires_dry_run_before_config_or_client_load(self) -> None:
        calls = []

        def client_factory(**kwargs):
            calls.append(kwargs)
            raise AssertionError("factory must not be called")

        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = tool.run_ibkr_paper_readonly_preflight(
                ["--json", "--port", "4002"],
                client_factory=client_factory,
            )

        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertEqual(out.getvalue(), "")
        self.assertIn("--dry-run-only", err.getvalue())

    def test_cli_rejects_live_and_live_port_after_dry_run_validation(self) -> None:
        for argv in (
            ["--dry-run-only", "--trading-mode", "LIVE"],
            ["--dry-run-only", "--port", "4002"],
            ["--dry-run-only", "--trading-mode", "UNKNOWN"],
        ):
            with self.subTest(argv=argv):
                out = io.StringIO()
                err = io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = tool.run_ibkr_paper_readonly_preflight(
                        argv,
                        client_factory=lambda **kwargs: FakeClient(),
                    )
                self.assertEqual(code, 1)
                self.assertEqual(out.getvalue(), "")
                self.assertIn("ERROR:", err.getvalue())

    def test_cli_json_writes_strict_json_to_stdout_only(self) -> None:
        def client_factory(**kwargs):
            client = FakeClient()
            client.config = kwargs["config"]
            return client

        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = tool.run_ibkr_paper_readonly_preflight(
                ["--dry-run-only", "--json"],
                client_factory=client_factory,
                report_builder=lambda **kwargs: build_ibkr_paper_readonly_preflight_report(
                    now_provider=fixed_now,
                    **kwargs,
                ),
            )

        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["ibkr_paper_readonly_preflight"])
        self.assertTrue(payload["config"]["readonly"])

    def test_cli_human_readable_output_is_default(self) -> None:
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = tool.run_ibkr_paper_readonly_preflight(
                ["--dry-run-only"],
                client_factory=lambda **kwargs: FakeClient(),
                report_builder=lambda **kwargs: build_ibkr_paper_readonly_preflight_report(
                    now_provider=fixed_now,
                    **kwargs,
                ),
            )

        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        self.assertIn("IBKR paper read-only preflight", out.getvalue())
        with self.assertRaises(json.JSONDecodeError):
            json.loads(out.getvalue())

    def test_cli_flags_are_boolean_store_true_actions(self) -> None:
        actions = []
        original_add_argument = argparse.ArgumentParser.add_argument

        def spy_add_argument(self, *args, **kwargs):
            if "--dry-run-only" in args or "--json" in args:
                actions.append((args, kwargs.get("action")))
            return original_add_argument(self, *args, **kwargs)

        argparse.ArgumentParser.add_argument = spy_add_argument
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                tool.run_ibkr_paper_readonly_preflight(
                    ["--dry-run-only"],
                    client_factory=lambda **kwargs: FakeClient(),
                )
        finally:
            argparse.ArgumentParser.add_argument = original_add_argument

        self.assertIn((("--dry-run-only",), "store_true"), actions)
        self.assertIn((("--json",), "store_true"), actions)

    def test_cli_exposes_no_submit_cancel_market_data_or_qualification_actions(self) -> None:
        actions = []
        original_add_argument = argparse.ArgumentParser.add_argument

        def spy_add_argument(self, *args, **kwargs):
            actions.extend(str(arg) for arg in args)
            return original_add_argument(self, *args, **kwargs)

        argparse.ArgumentParser.add_argument = spy_add_argument
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                tool.run_ibkr_paper_readonly_preflight(
                    ["--dry-run-only"],
                    client_factory=lambda **kwargs: FakeClient(),
                )
        finally:
            argparse.ArgumentParser.add_argument = original_add_argument

        joined = " ".join(actions).lower()
        self.assertNotIn("submit", joined)
        self.assertNotIn("cancel", joined)
        self.assertNotIn("market", joined)
        self.assertNotIn("qualif", joined)


class Stage4E2SafetyBoundaryTests(unittest.TestCase):
    def test_core_module_import_succeeds_without_ib_insync_installed(self) -> None:
        with mock.patch.dict(sys.modules, {"ib" + "_insync": None}):
            module = importlib.reload(
                importlib.import_module("algo_trader_unified.core.ibkr_paper_readonly_preflight")
            )

        self.assertTrue(hasattr(module, "build_ibkr_paper_readonly_preflight_report"))

    def test_tool_module_import_succeeds_without_ib_insync_installed(self) -> None:
        with mock.patch.dict(sys.modules, {"ib" + "_insync": None}):
            module = importlib.reload(
                importlib.import_module("algo_trader_unified.tools.ibkr_paper_readonly_preflight")
            )

        self.assertTrue(hasattr(module, "run_ibkr_paper_readonly_preflight"))

    def test_public_report_path_is_synchronous(self) -> None:
        self.assertTrue(inspect.isfunction(build_ibkr_paper_readonly_preflight_report))
        self.assertFalse(inspect.iscoroutinefunction(build_ibkr_paper_readonly_preflight_report))

    def test_stage4e2_files_do_not_import_or_call_blocked_integrations(self) -> None:
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
        )
        for path in STAGE4E2_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_fake_client_opens_no_socket_or_async_loop(self) -> None:
        client = FakeClient()

        self.build_report(client)

        self.assertFalse(client.socket_opened)
        self.assertFalse(client.async_loop_started)

    def build_report(self, client: FakeClient) -> dict:
        return build_ibkr_paper_readonly_preflight_report(
            client=client,
            now_provider=fixed_now,
        )

    def test_stage4e2_client_is_not_wired_into_daemon_scheduler_or_lifecycle(self) -> None:
        blocked_tokens = (
            "ibkr_paper_readonly" + "_preflight",
            "IbkrPaperReadOnly" + "Client",
            "ibkr_paper" + "_client",
        )
        for path in UNWIRED_RUNTIME_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
