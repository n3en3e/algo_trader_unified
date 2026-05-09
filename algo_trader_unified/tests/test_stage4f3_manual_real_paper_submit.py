from __future__ import annotations

import copy
import inspect
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from algo_trader_unified.core.broker_adapter import BrokerSubmitResult
from algo_trader_unified.core.ibkr_paper_order_mapper import (
    IBKR_PAPER_PORT,
    IbkrPaperConfig,
)
from algo_trader_unified.core.manual_real_paper_submit import (
    REQUIRED_ACKNOWLEDGEMENTS,
    build_manual_real_paper_submit_report,
)
from algo_trader_unified.tools import manual_real_paper_submit as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4F3_FILES = [
    ROOT / "core/manual_real_paper_submit.py",
    ROOT / "tools/manual_real_paper_submit.py",
]
UNWIRED_RUNTIME_FILES = [
    ROOT / "tools/daemon.py",
    ROOT / "core/scheduler.py",
    ROOT / "core/scheduler_cadence.py",
    ROOT / "core/job_chain.py",
]


def fixed_now() -> datetime:
    return datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc)


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


def valid_ticket(**overrides: object) -> dict:
    report = {
        "dry_run": True,
        "paper_order_ticket_report": True,
        "generated_at": "2026-05-09T14:00:00+00:00",
        "intent": {
            "intent_id": "intent-stage4f3-001",
            "strategy_id": "S01_VOL_BASELINE",
            "symbol": "XSP",
            "side": "BUY",
            "quantity": 1,
            "order_type": "LIMIT",
            "valid": True,
            "validation_reason": None,
        },
        "broker_order_request": {
            "available": True,
            "client_order_id": "intent-stage4f3-001",
            "strategy_id": "S01_VOL_BASELINE",
            "symbol": "XSP",
            "asset_type": "OPTION",
            "side": "BUY",
            "quantity": 1,
            "order_type": "LIMIT",
            "limit_price": 1.25,
            "time_in_force": "DAY",
            "intent_id": "intent-stage4f3-001",
        },
        "ibkr_order_plan": {
            "available": True,
            "ready_for_submission": True,
            "client_order_id": "intent-stage4f3-001",
            "strategy_id": "S01_VOL_BASELINE",
            "symbol": "XSP",
            "asset_type": "OPTION",
            "action": "BUY",
            "quantity": 1,
            "order_type": "LMT",
            "limit_price": 1.25,
            "time_in_force": "DAY",
            "account_id": "DU1234567",
            "paper_only": True,
            "dry_run": True,
            "blockers": [],
            "warnings": [],
            "ibkr_contract_hint": {"symbol": "XSP", "secType": "OPT"},
            "ibkr_order_hint": {"action": "BUY", "totalQuantity": 1, "orderType": "LMT"},
        },
        "submit_gate": {
            "eligible_for_future_manual_submit": True,
            "reasons": [],
        },
        "safety": {
            "live_orders_enabled": False,
            "market_data_enabled": False,
            "contract_qualification_enabled": False,
            "scheduler_changes_enabled": False,
            "lifecycle_wiring_enabled": False,
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    report.update(overrides)
    return report


def valid_preflight(**overrides: object) -> dict:
    report = {
        "dry_run": True,
        "ibkr_paper_connection_preflight": True,
        "generated_at": "2026-05-09T14:00:00+00:00",
        "connection": {
            "allow_real_ibkr": True,
            "attempted": True,
            "connected": True,
            "paper_mode": True,
            "disconnected": True,
            "reason": None,
        },
        "readonly_checks": {
            "current_time_ok": True,
            "account_snapshot_ok": True,
            "open_orders_ok": True,
            "positions_ok": True,
        },
        "readiness_for_stage4f3": {
            "ready_to_build_manual_real_paper_submit_command": True,
            "blockers": [],
            "warnings": [],
        },
        "safety": {
            "live_orders_enabled": False,
            "market_data_enabled": False,
            "contract_qualification_enabled": False,
            "scheduler_changes_enabled": False,
            "lifecycle_wiring_enabled": False,
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    report.update(overrides)
    return report


@dataclass(frozen=True)
class DisconnectStatus:
    connected: bool = False
    reason: str | None = None


class ProprietaryThing:
    pass


class FakeExecutionClient:
    def __init__(self) -> None:
        self.submit_calls: list[dict] = []
        self.disconnect_calls = 0
        self.submit_result: object = SimpleNamespace(
            accepted=True,
            broker_order_id="9001",
            client_order_id="intent-stage4f3-001",
            reason=None,
            raw={
                "broker": "fake",
                "decimal": Decimal("1.25"),
                "object": ProprietaryThing(),
            },
        )
        self.submit_exception: BaseException | None = None
        self.disconnect_exception: BaseException | None = None
        self.disconnect_result: object = DisconnectStatus()

    def submit_order_plan(self, plan: dict) -> object:
        self.submit_calls.append(copy.deepcopy(plan))
        if self.submit_exception is not None:
            raise self.submit_exception
        return self.submit_result

    def disconnect(self) -> object:
        self.disconnect_calls += 1
        if self.disconnect_exception is not None:
            raise self.disconnect_exception
        return self.disconnect_result


class FactoryRecorder:
    def __init__(self, client: FakeExecutionClient | None = None) -> None:
        self.ib_calls = 0
        self.execution_client_calls = 0
        self.ib = object()
        self.client = client or FakeExecutionClient()
        self.configs: list[IbkrPaperConfig] = []

    def ib_factory(self) -> object:
        self.ib_calls += 1
        return self.ib

    def execution_client_factory(self, ib: object, config: IbkrPaperConfig) -> FakeExecutionClient:
        self.execution_client_calls += 1
        self.assert_ib = ib
        self.configs.append(config)
        return self.client


class ManualRealPaperSubmitCoreTests(unittest.TestCase):
    def build(
        self,
        *,
        ticket_report: dict | None = None,
        preflight_report: dict | None = None,
        config: dict[str, object] | IbkrPaperConfig | None = None,
        acknowledgements: list[str] | None = None,
        allow_real_ibkr: bool = True,
        allow_real_paper_submit: bool = True,
        recorder: FactoryRecorder | None = None,
    ) -> tuple[dict, FactoryRecorder]:
        recorder = recorder or FactoryRecorder()
        report = build_manual_real_paper_submit_report(
            ticket_report=ticket_report or valid_ticket(),
            connection_preflight_report=preflight_report or valid_preflight(),
            execution_client_factory=recorder.execution_client_factory,
            ib_factory=recorder.ib_factory,
            config=config or raw_config(),
            operator_acknowledgements=acknowledgements
            if acknowledgements is not None
            else list(REQUIRED_ACKNOWLEDGEMENTS),
            allow_real_ibkr=allow_real_ibkr,
            allow_real_paper_submit=allow_real_paper_submit,
            now_provider=fixed_now,
        )
        return report, recorder

    def assert_no_factory_calls(self, recorder: FactoryRecorder) -> None:
        self.assertEqual(recorder.ib_calls, 0)
        self.assertEqual(recorder.execution_client_calls, 0)
        self.assertEqual(recorder.client.submit_calls, [])
        self.assertEqual(recorder.client.disconnect_calls, 0)

    def test_missing_allow_real_ibkr_refuses_before_factories(self) -> None:
        report, recorder = self.build(allow_real_ibkr=False)

        self.assert_no_factory_calls(recorder)
        self.assertFalse(report["submission"]["attempted"])
        self.assertFalse(report["submission"]["submitted"])
        self.assertIn("allow_real_ibkr must be True", report["gates"]["reasons"])

    def test_missing_allow_real_paper_submit_refuses_before_factories(self) -> None:
        report, recorder = self.build(allow_real_paper_submit=False)

        self.assert_no_factory_calls(recorder)
        self.assertIn("allow_real_paper_submit must be True", report["gates"]["reasons"])

    def test_missing_config_port_refuses_before_factories(self) -> None:
        config = raw_config()
        config.pop("port")
        report, recorder = self.build(config=config)

        self.assert_no_factory_calls(recorder)
        self.assertFalse(report["gates"]["port_explicit"])
        self.assertIn("config must explicitly include port 4004", report["gates"]["reasons"])

    def test_live_config_refuses_before_factories(self) -> None:
        report, recorder = self.build(config=raw_config(trading_mode="LIVE"))

        self.assert_no_factory_calls(recorder)
        self.assertFalse(report["gates"]["config_valid"])
        self.assertIn("LIVE is rejected", json.dumps(report, sort_keys=True))

    def test_port_4002_refuses_before_factories(self) -> None:
        report, recorder = self.build(config=raw_config(port=4002))

        self.assert_no_factory_calls(recorder)
        self.assertFalse(report["gates"]["port_is_paper"])
        self.assertIn("config port must be exactly 4004 for IBKR paper", report["gates"]["reasons"])

    def test_ineligible_ticket_refuses_before_factories(self) -> None:
        ticket = valid_ticket(
            submit_gate={"eligible_for_future_manual_submit": False, "reasons": ["no"]}
        )
        report, recorder = self.build(ticket_report=ticket)

        self.assert_no_factory_calls(recorder)
        self.assertFalse(report["gates"]["ticket_eligible"])
        self.assertIn(
            "ticket submit gate must be eligible for future manual submit",
            report["gates"]["reasons"],
        )

    def test_failed_preflight_refuses_before_factories(self) -> None:
        preflight = valid_preflight(
            connection={
                "allow_real_ibkr": True,
                "attempted": True,
                "connected": False,
                "paper_mode": True,
                "disconnected": True,
                "reason": "down",
            }
        )
        report, recorder = self.build(preflight_report=preflight)

        self.assert_no_factory_calls(recorder)
        self.assertFalse(report["gates"]["preflight_ready"])
        self.assertIn("connection preflight must report connected True", report["gates"]["reasons"])

    def test_missing_acknowledgement_refuses_before_factories(self) -> None:
        provided = REQUIRED_ACKNOWLEDGEMENTS[:-1]
        report, recorder = self.build(acknowledgements=provided)

        self.assert_no_factory_calls(recorder)
        self.assertFalse(report["acknowledgements"]["exact_match"])
        self.assertEqual(report["acknowledgements"]["missing"], [REQUIRED_ACKNOWLEDGEMENTS[-1]])

    def test_giant_substring_acknowledgement_does_not_pass(self) -> None:
        report, recorder = self.build(acknowledgements=[" ".join(REQUIRED_ACKNOWLEDGEMENTS)])

        self.assert_no_factory_calls(recorder)
        self.assertEqual(report["acknowledgements"]["missing"], REQUIRED_ACKNOWLEDGEMENTS)

    def test_extra_acknowledgements_do_not_compensate_for_missing_required_text(self) -> None:
        provided = REQUIRED_ACKNOWLEDGEMENTS[:-1] + ["I reviewed something else."]
        report, recorder = self.build(acknowledgements=provided)

        self.assert_no_factory_calls(recorder)
        self.assertEqual(report["acknowledgements"]["missing"], [REQUIRED_ACKNOWLEDGEMENTS[-1]])

    def test_exact_acknowledgements_allow_single_submit_with_injected_factories(self) -> None:
        report, recorder = self.build()

        self.assertEqual(recorder.ib_calls, 1)
        self.assertEqual(recorder.execution_client_calls, 1)
        self.assertEqual(len(recorder.client.submit_calls), 1)
        self.assertEqual(recorder.client.disconnect_calls, 1)
        self.assertEqual(recorder.configs[0].port, 4004)
        self.assertTrue(report["gates"]["passed"])
        self.assertTrue(report["submission"]["submitted"])
        self.assertEqual(report["submission"]["broker_order_id"], "9001")

    def test_disconnect_is_attempted_after_successful_submit(self) -> None:
        report, recorder = self.build()

        self.assertTrue(report["cleanup"]["disconnect_attempted"])
        self.assertTrue(report["cleanup"]["disconnect_ok"])
        self.assertEqual(recorder.client.disconnect_calls, 1)

    def test_disconnect_is_attempted_after_rejected_submit_result(self) -> None:
        client = FakeExecutionClient()
        client.submit_result = BrokerSubmitResult(
            accepted=False,
            dry_run=True,
            broker_order_id=None,
            client_order_id="intent-stage4f3-001",
            reason="paper rejection",
            raw={"status": "Rejected"},
        )
        report, recorder = self.build(recorder=FactoryRecorder(client))

        self.assertTrue(report["submission"]["attempted"])
        self.assertFalse(report["submission"]["submitted"])
        self.assertEqual(report["submission"]["reason"], "paper rejection")
        self.assertEqual(recorder.client.disconnect_calls, 1)

    def test_disconnect_is_attempted_after_submit_exceptions(self) -> None:
        for exc in (TimeoutError("slow"), ConnectionError("lost"), RuntimeError("boom")):
            with self.subTest(exc=type(exc).__name__):
                client = FakeExecutionClient()
                client.submit_exception = exc
                report, recorder = self.build(recorder=FactoryRecorder(client))

                self.assertFalse(report["submission"]["submitted"])
                self.assertIn(type(exc).__name__, report["submission"]["reason"])
                self.assertEqual(recorder.client.disconnect_calls, 1)

    def test_disconnect_failure_does_not_mask_accepted_submit_result(self) -> None:
        client = FakeExecutionClient()
        client.disconnect_exception = RuntimeError("disconnect broke")
        report, _ = self.build(recorder=FactoryRecorder(client))

        self.assertTrue(report["submission"]["submitted"])
        self.assertEqual(report["submission"]["broker_order_id"], "9001")
        self.assertFalse(report["cleanup"]["disconnect_ok"])
        self.assertIn("disconnect failed: RuntimeError: disconnect broke", report["warnings"])

    def test_disconnect_failure_does_not_mask_rejected_submit_result(self) -> None:
        client = FakeExecutionClient()
        client.submit_result = BrokerSubmitResult(
            accepted=False,
            dry_run=True,
            broker_order_id=None,
            client_order_id="intent-stage4f3-001",
            reason="rejected by broker",
            raw={"status": "Rejected"},
        )
        client.disconnect_exception = RuntimeError("disconnect broke")
        report, _ = self.build(recorder=FactoryRecorder(client))

        self.assertFalse(report["submission"]["submitted"])
        self.assertEqual(report["submission"]["reason"], "rejected by broker")
        self.assertIn("disconnect broke", report["cleanup"]["disconnect_reason"])

    def test_disconnect_failure_does_not_mask_submit_exception(self) -> None:
        client = FakeExecutionClient()
        client.submit_exception = TimeoutError("submit timed out")
        client.disconnect_exception = RuntimeError("disconnect broke")
        report, _ = self.build(recorder=FactoryRecorder(client))

        self.assertFalse(report["submission"]["submitted"])
        self.assertIn("TimeoutError: submit timed out", report["submission"]["reason"])
        self.assertIn("RuntimeError: disconnect broke", report["cleanup"]["disconnect_reason"])

    def test_ib_factory_exception_is_reported_with_type_and_string(self) -> None:
        def fail_ib_factory() -> object:
            raise RuntimeError("ib factory down")

        recorder = FactoryRecorder()
        report = build_manual_real_paper_submit_report(
            ticket_report=valid_ticket(),
            connection_preflight_report=valid_preflight(),
            execution_client_factory=recorder.execution_client_factory,
            ib_factory=fail_ib_factory,
            config=raw_config(),
            operator_acknowledgements=list(REQUIRED_ACKNOWLEDGEMENTS),
            allow_real_ibkr=True,
            allow_real_paper_submit=True,
            now_provider=fixed_now,
        )

        self.assertIn("ib_factory failed: RuntimeError: ib factory down", report["errors"])
        self.assertEqual(recorder.execution_client_calls, 0)
        self.assertFalse(report["submission"]["attempted"])

    def test_execution_client_factory_exception_is_reported_with_type_and_string(self) -> None:
        calls = {"execution_client_factory": 0}

        def fail_execution_client_factory(ib: object, config: IbkrPaperConfig) -> object:
            calls["execution_client_factory"] += 1
            raise ConnectionError("client factory down")

        report = build_manual_real_paper_submit_report(
            ticket_report=valid_ticket(),
            connection_preflight_report=valid_preflight(),
            execution_client_factory=fail_execution_client_factory,
            ib_factory=lambda: object(),
            config=raw_config(),
            operator_acknowledgements=list(REQUIRED_ACKNOWLEDGEMENTS),
            allow_real_ibkr=True,
            allow_real_paper_submit=True,
            now_provider=fixed_now,
        )

        self.assertEqual(calls["execution_client_factory"], 1)
        self.assertIn(
            "execution_client_factory failed: ConnectionError: client factory down",
            report["errors"],
        )
        self.assertFalse(report["submission"]["attempted"])

    def test_submit_order_plan_exception_is_reported_with_type_and_string(self) -> None:
        client = FakeExecutionClient()
        client.submit_exception = RuntimeError("submit exploded")
        report, _ = self.build(recorder=FactoryRecorder(client))

        self.assertIn("submit_order_plan failed: RuntimeError: submit exploded", report["errors"])
        self.assertIn("RuntimeError: submit exploded", report["submission"]["reason"])

    def test_input_reports_are_not_mutated(self) -> None:
        ticket = valid_ticket()
        preflight = valid_preflight()
        before_ticket = copy.deepcopy(ticket)
        before_preflight = copy.deepcopy(preflight)

        self.build(ticket_report=ticket, preflight_report=preflight)

        self.assertEqual(ticket, before_ticket)
        self.assertEqual(preflight, before_preflight)

    def test_report_includes_required_top_level_fields_and_is_json_safe(self) -> None:
        report, _ = self.build()

        for key in (
            "dry_run",
            "manual_real_paper_submit",
            "generated_at",
            "gates",
            "ticket",
            "preflight",
            "acknowledgements",
            "submission",
            "cleanup",
            "safety",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("ProprietaryThing", serialized)
        self.assertNotIn("Decimal", serialized)

    def test_live_and_scheduler_override_flags_refuse_before_factories(self) -> None:
        ticket = valid_ticket(live_override=True)
        preflight = valid_preflight(scheduler_override=True)
        report, recorder = self.build(ticket_report=ticket, preflight_report=preflight)

        self.assert_no_factory_calls(recorder)
        self.assertIn("ticket live override flag must not be enabled", report["gates"]["reasons"])
        self.assertIn(
            "preflight scheduler/lifecycle override flag must not be enabled",
            report["gates"]["reasons"],
        )


class ManualRealPaperSubmitCliTests(unittest.TestCase):
    def test_cli_requires_dry_run_only_before_report_builder(self) -> None:
        calls = {"report_builder": 0}

        def report_builder(**kwargs):
            calls["report_builder"] += 1
            return {}

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_manual_real_paper_submit(
                [
                    "--ticket-json",
                    "{}",
                    "--preflight-json",
                    "{}",
                ],
                report_builder=report_builder,
            )

        self.assertEqual(code, 1)
        self.assertEqual(calls["report_builder"], 0)
        self.assertIn("--dry-run-only", stderr.getvalue())

    def test_cli_requires_allow_flags_before_factory_calls(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_manual_real_paper_submit(
                [
                    "--dry-run-only",
                    "--json",
                    "--ticket-json",
                    json.dumps(valid_ticket()),
                    "--preflight-json",
                    json.dumps(valid_preflight()),
                    "--ack",
                    REQUIRED_ACKNOWLEDGEMENTS[0],
                ]
            )

        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertFalse(report["submission"]["attempted"])
        self.assertIn("allow_real_ibkr must be True", report["gates"]["reasons"])
        self.assertIn("allow_real_paper_submit must be True", report["gates"]["reasons"])

    def test_cli_ack_uses_argparse_append(self) -> None:
        source = inspect.getsource(tool.run_manual_real_paper_submit)

        self.assertIn('parser.add_argument("--ack", action="append"', source)

    def test_cli_json_stdout_is_strict_json_only(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            tool.run_manual_real_paper_submit(
                [
                    "--dry-run-only",
                    "--json",
                    "--ticket-json",
                    json.dumps(valid_ticket()),
                    "--preflight-json",
                    json.dumps(valid_preflight()),
                ],
                report_builder=lambda **kwargs: {
                    "success": True,
                    "errors": [],
                    "submission": {"submitted": False},
                    "manual_real_paper_submit": True,
                },
            )

        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["manual_real_paper_submit"])

    def test_cli_exposes_no_live_market_data_qualification_or_scheduler_actions(self) -> None:
        source = inspect.getsource(tool)
        forbidden = [
            "--allow-live",
            "--live",
            "--market-data",
            "--qualify",
            "--scheduler",
            "--lifecycle",
        ]
        for value in forbidden:
            with self.subTest(value=value):
                self.assertNotIn(value, source)


class ManualRealPaperSubmitSafetyTests(unittest.TestCase):
    def test_stage4f3_files_do_not_call_forbidden_external_apis(self) -> None:
        forbidden = [
            "place" + "Order(",
            "cancel" + "Order(",
            "req" + "MktData",
            "qualify" + "Contracts",
            "yfinance",
            "requests",
            "urllib",
            "systemctl",
            "systemd",
            "socket.create_connection",
            "socket.socket",
            "asyncio.run",
            "asyncio.get_event_loop",
            "asyncio.new_event_loop",
            "uuid.uuid4",
            "random",
        ]
        for path in STAGE4F3_FILES:
            source = path.read_text(encoding="utf-8")
            for value in forbidden:
                with self.subTest(path=path.name, value=value):
                    self.assertNotIn(value, source)

    def test_no_daemon_scheduler_or_lifecycle_wiring_exists(self) -> None:
        for path in UNWIRED_RUNTIME_FILES:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("manual_real_paper_submit", source)


if __name__ == "__main__":
    unittest.main()
