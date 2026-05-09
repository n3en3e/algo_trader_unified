from __future__ import annotations

import argparse
import importlib
import inspect
import io
import json
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from algo_trader_unified.core import ibkr_paper_factory
from algo_trader_unified.core.ibkr_paper_factory import (
    build_ibkr_paper_factory_preflight_report,
    create_real_ibkr_paper_ib,
    validate_ibkr_paper_factory_config,
)
from algo_trader_unified.core.ibkr_paper_order_mapper import (
    IBKR_PAPER_PORT,
    IbkrPaperConfig,
    validate_ibkr_paper_readonly_config,
)
from algo_trader_unified.tools import ibkr_paper_factory_preflight as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4F1_FILES = [
    ROOT / "core/ibkr_paper_factory.py",
    ROOT / "tools/ibkr_paper_factory_preflight.py",
]
UNWIRED_RUNTIME_FILES = [
    ROOT / "tools/daemon.py",
    ROOT / "core/scheduler.py",
    ROOT / "core/scheduler_cadence.py",
    ROOT / "core/job_chain.py",
]


def fixed_now() -> datetime:
    return datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)


def valid_config(**overrides: object) -> IbkrPaperConfig:
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


class FakeIB:
    def __init__(self) -> None:
        self.connect_calls = 0
        self.sleep_calls = 0
        self.place_order_calls = 0
        self.cancel_order_calls = 0
        self.market_data_calls = 0
        self.qualification_calls = 0

    def connect(self, *args, **kwargs):
        self.connect_calls += 1
        raise AssertionError("factory must not connect")

    def sleep(self, *args, **kwargs):
        self.sleep_calls += 1
        raise AssertionError("factory must not sleep")

    def __getattr__(self, name: str):
        if name == "place" + "Order":
            self.place_order_calls += 1
            raise AssertionError("factory must not submit")
        if name == "cancel" + "Order":
            self.cancel_order_calls += 1
            raise AssertionError("factory must not cancel")
        if name == "req" + "MktData":
            self.market_data_calls += 1
            raise AssertionError("factory must not request market data")
        if name == "qualify" + "Contracts":
            self.qualification_calls += 1
            raise AssertionError("factory must not qualify contracts")
        raise AttributeError(name)


class FakeIbModule(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__()
        self.instances: list[FakeIB] = []

        def make_ib() -> FakeIB:
            instance = FakeIB()
            self.instances.append(instance)
            return instance

        self.IB = make_ib


class IbkrPaperFactoryPreflightCoreTests(unittest.TestCase):
    def build(self, config: dict[str, object] | IbkrPaperConfig, **kwargs) -> dict:
        return build_ibkr_paper_factory_preflight_report(
            config=config,
            import_probe=kwargs.pop("import_probe", lambda name: object()),
            now_provider=fixed_now,
            **kwargs,
        )

    def test_core_preflight_accepts_valid_paper_4004_config(self) -> None:
        report = self.build(valid_config())

        self.assertTrue(report["dry_run"])
        self.assertTrue(report["ibkr_paper_factory_preflight"])
        self.assertTrue(report["success"])
        self.assertEqual(report["generated_at"], "2026-05-09T12:00:00+00:00")
        self.assertTrue(report["config"]["paper_config_valid"])
        self.assertEqual(report["config"]["validation_reason"], "valid")
        self.assertTrue(
            report["readiness_for_stage4f2"][
                "ready_to_build_manual_real_paper_submit_command"
            ]
        )
        json.dumps(report, sort_keys=True)

    def test_core_preflight_rejects_live_config(self) -> None:
        report = self.build(raw_config(trading_mode="LIVE"))

        self.assertFalse(report["config"]["paper_config_valid"])
        self.assertFalse(
            report["readiness_for_stage4f2"][
                "ready_to_build_manual_real_paper_submit_command"
            ]
        )
        self.assertIn("LIVE is rejected", json.dumps(report, sort_keys=True))

    def test_core_preflight_rejects_port_4002(self) -> None:
        report = self.build(raw_config(port=4002))

        self.assertFalse(report["config"]["paper_config_valid"])
        self.assertIn("4002 is rejected", json.dumps(report, sort_keys=True))

    def test_core_preflight_rejects_unknown_trading_mode(self) -> None:
        report = self.build(raw_config(trading_mode="UNKNOWN"))

        self.assertFalse(report["config"]["paper_config_valid"])
        self.assertIn('trading_mode must be exactly "PAPER"', report["config"]["validation_reason"])

    def test_validation_wrapper_rejects_bad_host_client_account_and_readonly(self) -> None:
        bad_configs = (
            raw_config(host=" "),
            raw_config(client_id=0),
            raw_config(client_id=True),
            raw_config(account_id=" "),
            raw_config(readonly=False),
        )
        for config in bad_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    validate_ibkr_paper_factory_config(config)

    def test_core_preflight_handles_config_validation_exceptions_with_details(self) -> None:
        def fail_validation(config):
            raise RuntimeError("validator exploded")

        with mock.patch.object(
            ibkr_paper_factory,
            "validate_ibkr_paper_factory_config",
            side_effect=fail_validation,
        ):
            report = self.build(raw_config())

        self.assertTrue(report["success"])
        self.assertFalse(report["config"]["paper_config_valid"])
        self.assertIn(
            "validate_ibkr_paper_factory_config failed: RuntimeError: validator exploded",
            report["errors"],
        )

    def test_import_probe_exception_is_caught_as_warning(self) -> None:
        def fail_probe(name: str) -> object:
            raise ValueError("probe unavailable")

        report = self.build(raw_config(), import_probe=fail_probe)

        self.assertTrue(report["success"])
        self.assertEqual(report["factory"]["ib_insync_available"], False)
        self.assertIn(
            "ib_insync availability probe failed: ValueError: probe unavailable",
            report["warnings"],
        )

    def test_preflight_uses_injected_probe_without_importing_ib_module(self) -> None:
        calls: list[str] = []

        def probe(name: str) -> object:
            calls.append(name)
            return None

        with mock.patch.object(importlib, "import_module") as import_module:
            report = self.build(raw_config(), import_probe=probe)

        self.assertEqual(calls, ["ib_insync"])
        import_module.assert_not_called()
        self.assertFalse(report["factory"]["ib_insync_available"])
        self.assertTrue(report["import_safety"]["preflight_uses_find_spec_or_injected_probe"])

    def test_preflight_default_probe_uses_find_spec(self) -> None:
        with mock.patch.object(
            ibkr_paper_factory.importlib.util,
            "find_spec",
            return_value=object(),
        ) as find_spec:
            report = build_ibkr_paper_factory_preflight_report(
                config=raw_config(),
                now_provider=fixed_now,
            )

        find_spec.assert_called_once_with("ib_insync")
        self.assertTrue(report["factory"]["ib_insync_available"])

    def test_report_includes_required_top_level_fields(self) -> None:
        report = self.build(raw_config())

        for key in (
            "dry_run",
            "ibkr_paper_factory_preflight",
            "generated_at",
            "config",
            "factory",
            "import_safety",
            "safety",
            "readiness_for_stage4f2",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)

    def test_safety_flags_are_false_for_external_actions(self) -> None:
        report = self.build(raw_config())
        factory = report["factory"]
        safety = report["safety"]

        self.assertFalse(factory["would_import_ib_insync"])
        self.assertFalse(factory["would_connect"])
        self.assertFalse(factory["would_submit_orders"])
        self.assertFalse(factory["would_cancel_orders"])
        self.assertFalse(factory["would_request_market_data"])
        self.assertFalse(factory["would_qualify_contracts"])
        self.assertFalse(safety["real_ibkr_enabled"])
        self.assertFalse(safety["paper_order_submission_enabled"])
        self.assertFalse(safety["cancel_enabled"])
        self.assertFalse(safety["live_orders_enabled"])
        self.assertFalse(safety["market_data_enabled"])
        self.assertFalse(safety["contract_qualification_enabled"])
        self.assertFalse(safety["scheduler_changes_enabled"])
        self.assertFalse(safety["lifecycle_wiring_enabled"])

    def test_recommendations_are_deterministic_and_non_binding(self) -> None:
        first = self.build(raw_config())["recommendations"]
        second = self.build(raw_config())["recommendations"]

        self.assertEqual(first, second)
        encoded = json.dumps(first, sort_keys=True).lower()
        self.assertNotIn("live trading", encoded)
        self.assertNotIn("threshold", encoded)
        self.assertNotIn("sizing", encoded)
        self.assertNotIn("cadence", encoded)


class IbkrPaperFactoryGatedFactoryTests(unittest.TestCase):
    def test_factory_defaults_allow_real_ibkr_false_and_fails_before_import(self) -> None:
        with mock.patch.object(importlib, "import_module") as import_module:
            with self.assertRaisesRegex(PermissionError, "allow_real_ibkr=True"):
                create_real_ibkr_paper_ib(config=valid_config())

        import_module.assert_not_called()

    def test_factory_validates_config_before_import(self) -> None:
        bad_config = replace(valid_config(), trading_mode="LIVE")

        with mock.patch.object(importlib, "import_module") as import_module:
            with self.assertRaisesRegex(ValueError, "LIVE is rejected"):
                create_real_ibkr_paper_ib(config=bad_config, allow_real_ibkr=True)

        import_module.assert_not_called()

    def test_factory_rejects_live_port_before_import(self) -> None:
        bad_config = replace(valid_config(), port=4002)

        with mock.patch.object(importlib, "import_module") as import_module:
            with self.assertRaisesRegex(ValueError, "4002 is rejected"):
                create_real_ibkr_paper_ib(config=bad_config, allow_real_ibkr=True)

        import_module.assert_not_called()

    def test_factory_with_allow_true_only_instantiates_ib_and_returns_it(self) -> None:
        fake_module = FakeIbModule()

        with mock.patch.object(importlib, "import_module", return_value=fake_module) as import_module:
            ib = create_real_ibkr_paper_ib(config=valid_config(), allow_real_ibkr=True)

        import_module.assert_called_once_with("ib_insync")
        self.assertIs(ib, fake_module.instances[0])
        self.assertEqual(len(fake_module.instances), 1)
        self.assertEqual(ib.connect_calls, 0)
        self.assertEqual(ib.sleep_calls, 0)
        self.assertEqual(ib.place_order_calls, 0)
        self.assertEqual(ib.cancel_order_calls, 0)
        self.assertEqual(ib.market_data_calls, 0)
        self.assertEqual(ib.qualification_calls, 0)
        self.assertEqual(vars(ib), {
            "connect_calls": 0,
            "sleep_calls": 0,
            "place_order_calls": 0,
            "cancel_order_calls": 0,
            "market_data_calls": 0,
            "qualification_calls": 0,
        })


class IbkrPaperFactoryCliTests(unittest.TestCase):
    def test_cli_requires_dry_run_before_local_checks(self) -> None:
        calls = []

        def report_builder(**kwargs):
            calls.append(kwargs)
            raise AssertionError("report builder must not run")

        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = tool.run_ibkr_paper_factory_preflight(
                ["--json", "--port", "4002"],
                report_builder=report_builder,
            )

        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertEqual(out.getvalue(), "")
        self.assertIn("--dry-run-only", err.getvalue())

    def test_cli_json_writes_strict_json_to_stdout_only(self) -> None:
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = tool.run_ibkr_paper_factory_preflight(
                ["--dry-run-only", "--json"],
                report_builder=lambda **kwargs: build_ibkr_paper_factory_preflight_report(
                    now_provider=fixed_now,
                    import_probe=lambda name: None,
                    **kwargs,
                ),
            )

        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["ibkr_paper_factory_preflight"])
        self.assertTrue(payload["config"]["readonly"])

    def test_cli_human_readable_output_is_default(self) -> None:
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = tool.run_ibkr_paper_factory_preflight(
                ["--dry-run-only"],
                report_builder=lambda **kwargs: build_ibkr_paper_factory_preflight_report(
                    now_provider=fixed_now,
                    import_probe=lambda name: None,
                    **kwargs,
                ),
            )

        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        self.assertIn("IBKR paper factory preflight", out.getvalue())
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
                tool.run_ibkr_paper_factory_preflight(
                    ["--dry-run-only"],
                    report_builder=lambda **kwargs: build_ibkr_paper_factory_preflight_report(
                        import_probe=lambda name: None,
                        **kwargs,
                    ),
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
                tool.run_ibkr_paper_factory_preflight(
                    ["--dry-run-only"],
                    report_builder=lambda **kwargs: build_ibkr_paper_factory_preflight_report(
                        import_probe=lambda name: None,
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


class Stage4F1SafetyBoundaryTests(unittest.TestCase):
    def test_core_and_tool_modules_import_without_ib_insync_installed(self) -> None:
        with mock.patch.dict(sys.modules, {"ib" + "_insync": None}):
            core_module = importlib.reload(
                importlib.import_module("algo_trader_unified.core.ibkr_paper_factory")
            )
            tool_module = importlib.reload(
                importlib.import_module("algo_trader_unified.tools.ibkr_paper_factory_preflight")
            )

        self.assertTrue(hasattr(core_module, "build_ibkr_paper_factory_preflight_report"))
        self.assertTrue(hasattr(tool_module, "run_ibkr_paper_factory_preflight"))

    def test_public_report_path_is_synchronous(self) -> None:
        self.assertTrue(inspect.isfunction(build_ibkr_paper_factory_preflight_report))
        self.assertFalse(inspect.iscoroutinefunction(build_ibkr_paper_factory_preflight_report))

    def test_no_module_level_ib_insync_import_exists(self) -> None:
        source = (ROOT / "core/ibkr_paper_factory.py").read_text(encoding="utf-8")

        self.assertNotIn("\nimport ib" + "_insync", source)
        self.assertNotIn("\nfrom ib" + "_insync import", source)
        self.assertIn("if TYPE_CHECKING:", source)
        self.assertIn("    from ib" + "_insync import", source)

    def test_preflight_does_not_use_try_except_import_for_availability(self) -> None:
        source = inspect.getsource(build_ibkr_paper_factory_preflight_report)

        self.assertIn("importlib.util.find_spec", source)
        self.assertNotIn("import ib" + "_insync", source)
        self.assertNotIn("from ib" + "_insync import", source)

    def test_runtime_import_is_only_inside_explicitly_gated_factory(self) -> None:
        source = inspect.getsource(create_real_ibkr_paper_ib)

        self.assertIn("allow_real_ibkr is not True", source)
        self.assertIn("validate_ibkr_paper_factory_config", source)
        self.assertIn('import_module("ib' + '_insync")', source)

    def test_stage4f1_files_do_not_import_or_call_blocked_integrations(self) -> None:
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
        for path in STAGE4F1_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_factory_does_not_call_connect_sleep_or_attach_event_handlers(self) -> None:
        source = inspect.getsource(create_real_ibkr_paper_ib)

        self.assertNotIn(".connect(", source)
        self.assertNotIn(".sleep(", source)
        self.assertNotIn("+=", source)
        self.assertNotIn(".event", source.lower())

    def test_stage4f1_is_not_wired_into_daemon_scheduler_or_lifecycle(self) -> None:
        blocked_tokens = (
            "ibkr_paper_factory",
            "build_ibkr_paper_factory_preflight_report",
            "create_real_ibkr_paper_ib",
        )
        for path in UNWIRED_RUNTIME_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
