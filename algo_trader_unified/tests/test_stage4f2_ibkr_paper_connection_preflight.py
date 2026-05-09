from __future__ import annotations

import argparse
import importlib
import inspect
import io
import json
import sys
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from algo_trader_unified.core import ibkr_paper_connection_preflight
from algo_trader_unified.core.ibkr_paper_connection_preflight import (
    CLIENT_ID_RECOMMENDATION,
    build_ibkr_paper_connection_preflight_report,
)
from algo_trader_unified.core.ibkr_paper_order_mapper import (
    IBKR_PAPER_PORT,
    IbkrPaperConfig,
    validate_ibkr_paper_readonly_config,
)
from algo_trader_unified.tools import ibkr_paper_connection_preflight as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4F2_FILES = [
    ROOT / "core/ibkr_paper_connection_preflight.py",
    ROOT / "tools/ibkr_paper_connection_preflight.py",
]
UNWIRED_RUNTIME_FILES = [
    ROOT / "tools/daemon.py",
    ROOT / "core/scheduler.py",
    ROOT / "core/scheduler_cadence.py",
    ROOT / "core/job_chain.py",
]


def fixed_now() -> datetime:
    return datetime(2026, 5, 9, 13, 0, tzinfo=timezone.utc)


def raw_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "host": "127.0.0.1",
        "port": IBKR_PAPER_PORT,
        "client_id": 7,
        "account_id": "DU1234567",
        "trading_mode": "PAPER",
        "readonly": True,
    }
    config.update(overrides)
    return config


def valid_config(**overrides: object) -> IbkrPaperConfig:
    return validate_ibkr_paper_readonly_config(raw_config(**overrides))


@dataclass(frozen=True)
class FakeStatus:
    connected: bool
    paper_mode: bool = True
    reason: str | None = None


@dataclass(frozen=True)
class FakeSnapshot:
    net_liquidation: float | None = 100000.0
    available_funds: float | None = 90000.0
    buying_power: float | None = 180000.0
    timestamp: str | None = "2026-05-09T13:00:00+00:00"
    raw: dict | None = None


class FakeClient:
    def __init__(
        self,
        *,
        connect_status: FakeStatus | None = None,
        status: FakeStatus | None = None,
        fail_on: dict[str, BaseException] | None = None,
        snapshot: object | None = None,
        open_orders: object | None = None,
        positions: object | None = None,
        connect_sleep_seconds: float = 0.0,
    ) -> None:
        self.calls: list[str] = []
        self.connect_status = connect_status or FakeStatus(True)
        self.status = status or FakeStatus(True)
        self.fail_on = fail_on or {}
        self.snapshot = snapshot if snapshot is not None else FakeSnapshot()
        self.open_orders = open_orders if open_orders is not None else []
        self.positions = positions if positions is not None else []
        self.connect_sleep_seconds = connect_sleep_seconds

    def connect(self) -> FakeStatus:
        self.calls.append("connect")
        if self.connect_sleep_seconds:
            time.sleep(self.connect_sleep_seconds)
        if "connect" in self.fail_on:
            raise self.fail_on["connect"]
        return self.connect_status

    def get_connection_status(self) -> FakeStatus:
        self.calls.append("get_connection_status")
        if "get_connection_status" in self.fail_on:
            raise self.fail_on["get_connection_status"]
        return self.status

    def get_current_time(self) -> dict:
        self.calls.append("get_current_time")
        if "get_current_time" in self.fail_on:
            raise self.fail_on["get_current_time"]
        return {"current_time": "2026-05-09T13:00:00+00:00"}

    def get_account_snapshot(self) -> object:
        self.calls.append("get_account_snapshot")
        if "get_account_snapshot" in self.fail_on:
            raise self.fail_on["get_account_snapshot"]
        return self.snapshot

    def list_open_orders(self) -> object:
        self.calls.append("list_open_orders")
        if "list_open_orders" in self.fail_on:
            raise self.fail_on["list_open_orders"]
        return self.open_orders

    def list_positions(self) -> object:
        self.calls.append("list_positions")
        if "list_positions" in self.fail_on:
            raise self.fail_on["list_positions"]
        return self.positions

    def disconnect(self) -> FakeStatus:
        self.calls.append("disconnect")
        if "disconnect" in self.fail_on:
            raise self.fail_on["disconnect"]
        return FakeStatus(False)

    def __getattr__(self, name: str):
        forbidden = {
            "place" + "Order",
            "cancel" + "Order",
            "req" + "MktData",
            "qualify" + "Contracts",
        }
        if name in forbidden:
            raise AssertionError(f"forbidden method accessed: {name}")
        raise AttributeError(name)


class FactoryRecorder:
    def __init__(self, client: FakeClient | None = None) -> None:
        self.ib_calls = 0
        self.client_calls = 0
        self.ib = object()
        self.client = client or FakeClient()
        self.client_configs: list[IbkrPaperConfig] = []

    def ib_factory(self) -> object:
        self.ib_calls += 1
        return self.ib

    def client_factory(self, ib: object, config: IbkrPaperConfig) -> FakeClient:
        self.client_calls += 1
        self.client_configs.append(config)
        self.assert_ib = ib
        return self.client


class IbkrPaperConnectionPreflightCoreTests(unittest.TestCase):
    def build(
        self,
        config: dict[str, object] | IbkrPaperConfig | None = None,
        *,
        recorder: FactoryRecorder | None = None,
        allow_real_ibkr: bool = True,
        connect_timeout_seconds: float = 1.0,
    ) -> tuple[dict, FactoryRecorder]:
        recorder = recorder or FactoryRecorder()
        report = build_ibkr_paper_connection_preflight_report(
            config=config or raw_config(),
            ib_factory=recorder.ib_factory,
            client_factory=recorder.client_factory,
            allow_real_ibkr=allow_real_ibkr,
            connect_timeout_seconds=connect_timeout_seconds,
            now_provider=fixed_now,
        )
        return report, recorder

    def test_allow_real_ibkr_false_does_not_call_factories(self) -> None:
        report, recorder = self.build(allow_real_ibkr=False)

        self.assertEqual(recorder.ib_calls, 0)
        self.assertEqual(recorder.client_calls, 0)
        self.assertFalse(report["connection"]["attempted"])
        self.assertFalse(
            report["readiness_for_stage4f3"][
                "ready_to_build_manual_real_paper_submit_command"
            ]
        )

    def test_valid_paper_4004_with_allow_true_calls_factories_once(self) -> None:
        report, recorder = self.build()

        self.assertEqual(recorder.ib_calls, 1)
        self.assertEqual(recorder.client_calls, 1)
        self.assertEqual(recorder.client_configs, [valid_config()])
        self.assertTrue(report["connection"]["connected"])
        self.assertTrue(report["connection"]["disconnected"])
        self.assertTrue(
            report["readiness_for_stage4f3"][
                "ready_to_build_manual_real_paper_submit_command"
            ]
        )

    def test_live_config_rejects_before_factory_calls(self) -> None:
        report, recorder = self.build(raw_config(trading_mode="LIVE"))

        self.assertEqual(recorder.ib_calls, 0)
        self.assertEqual(recorder.client_calls, 0)
        self.assertFalse(report["config"]["paper_config_valid"])
        self.assertIn("LIVE is rejected", json.dumps(report, sort_keys=True))

    def test_port_4002_rejects_before_factory_calls(self) -> None:
        report, recorder = self.build(raw_config(port=4002))

        self.assertEqual(recorder.ib_calls, 0)
        self.assertEqual(recorder.client_calls, 0)
        self.assertIn("4002 is rejected", json.dumps(report, sort_keys=True))

    def test_ib_factory_exception_is_reported(self) -> None:
        def fail_ib_factory() -> object:
            raise RuntimeError("ib unavailable")

        report = build_ibkr_paper_connection_preflight_report(
            config=raw_config(),
            ib_factory=fail_ib_factory,
            client_factory=lambda ib, config: FakeClient(),
            allow_real_ibkr=True,
            now_provider=fixed_now,
        )

        self.assertIn("ib_factory failed: RuntimeError: ib unavailable", report["errors"])
        self.assertFalse(report["connection"]["attempted"])

    def test_client_factory_exception_is_reported(self) -> None:
        calls = {"client_factory": 0}

        def fail_client_factory(ib: object, config: IbkrPaperConfig) -> object:
            calls["client_factory"] += 1
            raise ValueError("client unavailable")

        report = build_ibkr_paper_connection_preflight_report(
            config=raw_config(),
            ib_factory=lambda: object(),
            client_factory=fail_client_factory,
            allow_real_ibkr=True,
            now_provider=fixed_now,
        )

        self.assertEqual(calls["client_factory"], 1)
        self.assertIn("client_factory failed: ValueError: client unavailable", report["errors"])
        self.assertFalse(report["connection"]["attempted"])

    def test_connect_failure_prevents_read_paths_and_disconnects(self) -> None:
        client = FakeClient(connect_status=FakeStatus(False, reason="rejected"))
        report, _ = self.build(recorder=FactoryRecorder(client))

        self.assertEqual(client.calls, ["connect", "disconnect"])
        self.assertIn("connect failed: rejected", report["errors"])
        self.assertFalse(report["readonly_checks"]["current_time_ok"])

    def test_connect_timeout_is_json_safe_and_prevents_reads_without_hanging(self) -> None:
        client = FakeClient(connect_sleep_seconds=0.2)
        started = time.monotonic()
        report, _ = self.build(
            recorder=FactoryRecorder(client),
            connect_timeout_seconds=0.01,
        )
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.15)
        self.assertEqual(client.calls, ["connect", "disconnect"])
        self.assertIn("TimeoutError", json.dumps(report, sort_keys=True))
        self.assertFalse(report["readonly_checks"]["account_snapshot_ok"])
        json.dumps(report, sort_keys=True)

    def test_successful_connect_performs_readonly_checks_only(self) -> None:
        client = FakeClient(open_orders=[{"id": 1}], positions=[{"symbol": "SPY"}])
        report, _ = self.build(recorder=FactoryRecorder(client))

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
        self.assertEqual(report["open_orders"]["count"], 1)
        self.assertEqual(report["positions"]["count"], 1)
        self.assertTrue(all(report["readonly_checks"].values()))

    def test_read_failures_are_reported_without_stopping_other_reads(self) -> None:
        client = FakeClient(
            fail_on={
                "get_current_time": RuntimeError("clock failed"),
                "list_open_orders": ConnectionError("orders failed"),
            }
        )
        report, _ = self.build(recorder=FactoryRecorder(client))

        self.assertIn("current_time failed: RuntimeError: clock failed", report["errors"])
        self.assertIn("open_orders failed: ConnectionError: orders failed", report["errors"])
        self.assertIn("list_positions", client.calls)
        self.assertIn("disconnect", client.calls)

    def test_read_failure_and_disconnect_failure_are_both_preserved(self) -> None:
        client = FakeClient(
            fail_on={
                "get_account_snapshot": RuntimeError("snapshot failed"),
                "disconnect": RuntimeError("disconnect failed"),
            }
        )
        report, _ = self.build(recorder=FactoryRecorder(client))

        self.assertIn(
            "account_snapshot failed: RuntimeError: snapshot failed",
            report["errors"],
        )
        self.assertIn(
            "disconnect failed: RuntimeError: disconnect failed",
            report["errors"],
        )
        self.assertFalse(report["connection"]["disconnected"])

    def test_timeout_error_from_read_path_is_reported(self) -> None:
        client = FakeClient(fail_on={"list_positions": TimeoutError("positions too slow")})
        report, _ = self.build(recorder=FactoryRecorder(client))

        self.assertIn(
            "positions failed: TimeoutError: positions too slow",
            report["errors"],
        )
        self.assertFalse(report["readonly_checks"]["positions_ok"])

    def test_connection_failure_recommends_client_id_zombie_check(self) -> None:
        client = FakeClient(connect_status=FakeStatus(False, reason="duplicate id"))
        report, _ = self.build(recorder=FactoryRecorder(client))

        self.assertIn(
            CLIENT_ID_RECOMMENDATION,
            report["recommendations"]["ordered_next_steps"],
        )

    def test_report_includes_required_fields_and_is_json_safe(self) -> None:
        report, _ = self.build()

        for key in (
            "dry_run",
            "ibkr_paper_connection_preflight",
            "generated_at",
            "config",
            "connection",
            "readonly_checks",
            "account_snapshot",
            "open_orders",
            "positions",
            "readiness_for_stage4f3",
            "safety",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report, sort_keys=True)

    def test_non_positive_timeout_blocks_before_factory_calls(self) -> None:
        report, recorder = self.build(connect_timeout_seconds=0)

        self.assertEqual(recorder.ib_calls, 0)
        self.assertIn("connect_timeout_seconds must be positive", report["errors"])


class IbkrPaperConnectionPreflightCliTests(unittest.TestCase):
    def test_cli_requires_dry_run_before_builder_runs(self) -> None:
        calls = []

        def report_builder(**kwargs):
            calls.append(kwargs)
            raise AssertionError("report builder must not run")

        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = tool.run_ibkr_paper_connection_preflight(
                ["--json", "--allow-real-ibkr"],
                report_builder=report_builder,
            )

        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertEqual(out.getvalue(), "")
        self.assertIn("--dry-run-only", err.getvalue())

    def test_cli_without_allow_real_ibkr_does_not_attempt_connection(self) -> None:
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = tool.run_ibkr_paper_connection_preflight(
                ["--dry-run-only", "--json"],
                report_builder=lambda **kwargs: build_ibkr_paper_connection_preflight_report(
                    now_provider=fixed_now,
                    **kwargs,
                ),
            )

        payload = json.loads(out.getvalue())
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(code, 0)
        self.assertFalse(payload["connection"]["allow_real_ibkr"])
        self.assertFalse(payload["connection"]["attempted"])

    def test_cli_json_stdout_is_strict_json_only(self) -> None:
        def fake_builder(**kwargs):
            recorder = FactoryRecorder()
            kwargs["ib_factory"] = recorder.ib_factory
            kwargs["client_factory"] = recorder.client_factory
            return build_ibkr_paper_connection_preflight_report(
                now_provider=fixed_now,
                **kwargs,
            )

        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = tool.run_ibkr_paper_connection_preflight(
                ["--dry-run-only", "--json", "--allow-real-ibkr"],
                report_builder=fake_builder,
            )

        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["ibkr_paper_connection_preflight"])

    def test_cli_rejects_non_positive_timeout_before_builder_runs(self) -> None:
        calls = []

        def report_builder(**kwargs):
            calls.append(kwargs)
            return {}

        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = tool.run_ibkr_paper_connection_preflight(
                ["--dry-run-only", "--connect-timeout-seconds", "0"],
                report_builder=report_builder,
            )

        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("--connect-timeout-seconds", err.getvalue())
        self.assertEqual(out.getvalue(), "")

    def test_cli_flags_are_boolean_store_true_actions(self) -> None:
        actions = []
        original_add_argument = argparse.ArgumentParser.add_argument

        def spy_add_argument(self, *args, **kwargs):
            if "--dry-run-only" in args or "--json" in args or "--allow-real-ibkr" in args:
                actions.append((args, kwargs.get("action")))
            return original_add_argument(self, *args, **kwargs)

        argparse.ArgumentParser.add_argument = spy_add_argument
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                tool.run_ibkr_paper_connection_preflight(
                    ["--dry-run-only"],
                    report_builder=lambda **kwargs: build_ibkr_paper_connection_preflight_report(
                        now_provider=fixed_now,
                        **kwargs,
                    ),
                )
        finally:
            argparse.ArgumentParser.add_argument = original_add_argument

        self.assertIn((("--dry-run-only",), "store_true"), actions)
        self.assertIn((("--json",), "store_true"), actions)
        self.assertIn((("--allow-real-ibkr",), "store_true"), actions)

    def test_cli_exposes_no_forbidden_actions(self) -> None:
        actions = []
        original_add_argument = argparse.ArgumentParser.add_argument

        def spy_add_argument(self, *args, **kwargs):
            actions.extend(str(arg) for arg in args)
            return original_add_argument(self, *args, **kwargs)

        argparse.ArgumentParser.add_argument = spy_add_argument
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                tool.run_ibkr_paper_connection_preflight(
                    ["--dry-run-only"],
                    report_builder=lambda **kwargs: build_ibkr_paper_connection_preflight_report(
                        now_provider=fixed_now,
                        **kwargs,
                    ),
                )
        finally:
            argparse.ArgumentParser.add_argument = original_add_argument

        joined = " ".join(actions).lower()
        self.assertNotIn("submit", joined)
        self.assertNotIn("cancel", joined)
        self.assertNotIn("market", joined)
        self.assertNotIn("qualif", joined)


class Stage4F2SafetyBoundaryTests(unittest.TestCase):
    def test_core_and_tool_import_without_ib_insync_installed(self) -> None:
        with mock.patch.dict(sys.modules, {"ib" + "_insync": None}):
            core_module = importlib.reload(
                importlib.import_module(
                    "algo_trader_unified.core.ibkr_paper_connection_preflight"
                )
            )
            tool_module = importlib.reload(
                importlib.import_module(
                    "algo_trader_unified.tools.ibkr_paper_connection_preflight"
                )
            )

        self.assertTrue(hasattr(core_module, "build_ibkr_paper_connection_preflight_report"))
        self.assertTrue(hasattr(tool_module, "run_ibkr_paper_connection_preflight"))

    def test_public_report_path_is_synchronous(self) -> None:
        self.assertTrue(inspect.isfunction(build_ibkr_paper_connection_preflight_report))
        self.assertFalse(inspect.iscoroutinefunction(build_ibkr_paper_connection_preflight_report))

    def test_stage4f2_files_do_not_call_blocked_integrations(self) -> None:
        blocked_tokens = (
            "place" + "Order(",
            "cancel" + "Order(",
            "submit_order" + "_plan",
            "req" + "MktData(",
            "qualify" + "Contracts(",
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
        )
        for path in STAGE4F2_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_stage4f2_is_not_wired_into_daemon_scheduler_or_lifecycle(self) -> None:
        blocked_tokens = (
            "ibkr_paper_connection_preflight",
            "build_ibkr_paper_connection_preflight_report",
            "run_ibkr_paper_connection_preflight",
        )
        for path in UNWIRED_RUNTIME_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_no_direct_ib_insync_import_in_stage4f2_files(self) -> None:
        for path in STAGE4F2_FILES:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("\nimport ib" + "_insync", source)
            self.assertNotIn("\nfrom ib" + "_insync import", source)


if __name__ == "__main__":
    unittest.main()
