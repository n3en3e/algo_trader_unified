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
from unittest import mock

from algo_trader_unified.core import paper_order_ticket_report
from algo_trader_unified.core.ibkr_paper_order_mapper import IBKR_PAPER_PORT
from algo_trader_unified.core.paper_order_ticket_report import (
    build_paper_order_ticket_report,
)
from algo_trader_unified.tools import paper_order_ticket_report as ticket_tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4E4_FILES = [
    ROOT / "core/paper_order_ticket_report.py",
    ROOT / "tools/paper_order_ticket_report.py",
]


def valid_intent(**overrides: object) -> dict[str, object]:
    intent: dict[str, object] = {
        "intent_id": "intent-stage4e4-001",
        "strategy_id": "S01_VOL_BASELINE",
        "symbol": "XSP",
        "asset_type": "OPTION",
        "side": "BUY",
        "quantity": 1,
        "order_type": "LIMIT",
        "limit_price": Decimal("1.25"),
        "time_in_force": "DAY",
        "metadata": {"expiry": "20260619", "strike": 525.0, "right": "C"},
    }
    intent.update(overrides)
    return intent


def valid_config(**overrides: object) -> dict[str, object]:
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


class PaperOrderTicketReportCoreTests(unittest.TestCase):
    def build(self, intent: dict | None = None, config: dict | None = None) -> dict:
        return build_paper_order_ticket_report(
            intent=valid_intent() if intent is None else intent,
            ibkr_config=valid_config() if config is None else config,
            now_provider=lambda: datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc),
        )

    def test_valid_intent_and_paper_config_are_eligible_for_future_manual_submit(self) -> None:
        report = self.build()

        self.assertTrue(report["success"])
        self.assertTrue(report["dry_run"])
        self.assertTrue(report["paper_order_ticket_report"])
        self.assertTrue(report["intent"]["valid"])
        self.assertTrue(report["broker_order_request"]["available"])
        self.assertTrue(report["ibkr_config"]["valid"])
        self.assertTrue(report["ibkr_order_plan"]["available"])
        self.assertTrue(report["ibkr_order_plan"]["ready_for_submission"])
        self.assertTrue(report["ibkr_order_plan"]["paper_only"])
        self.assertTrue(report["ibkr_order_plan"]["dry_run"])
        self.assertEqual(report["ibkr_order_plan"]["blockers"], [])
        self.assertTrue(report["submit_gate"]["eligible_for_future_manual_submit"])
        self.assertEqual(report["submit_gate"]["reasons"], [])
        self.assertEqual(
            report["submit_gate"]["required_operator_acknowledgements"],
            [
                "I understand this is PAPER only.",
                "I understand no live orders are allowed.",
                "I understand this ticket is not yet submitted.",
                "I understand scheduler/lifecycle wiring remains disabled.",
            ],
        )

    def test_client_order_id_is_preserved_through_request_and_plan(self) -> None:
        report = self.build(intent=valid_intent(intent_id="intent-exact-123"))

        self.assertEqual(report["broker_order_request"]["client_order_id"], "intent-exact-123")
        self.assertEqual(report["broker_order_request"]["intent_id"], "intent-exact-123")
        self.assertEqual(report["ibkr_order_plan"]["client_order_id"], "intent-exact-123")

    def test_input_intent_is_not_mutated(self) -> None:
        intent = valid_intent()
        before = copy.deepcopy(intent)

        self.build(intent=intent)

        self.assertEqual(intent, before)

    def test_invalid_intent_is_reported_without_crashing(self) -> None:
        report = self.build(intent={})

        self.assertTrue(report["success"])
        self.assertFalse(report["intent"]["valid"])
        self.assertFalse(report["broker_order_request"]["available"])
        self.assertFalse(report["submit_gate"]["eligible_for_future_manual_submit"])
        self.assertIn("build_broker_order_request failed: ValueError", report["errors"][0])
        self.assertIn("strategy_id is required", report["errors"][0])

    def test_invalid_live_unknown_and_live_port_config_are_reported(self) -> None:
        cases = [
            (valid_config(trading_mode="LIVE"), "LIVE is rejected"),
            (valid_config(trading_mode="UNKNOWN"), 'trading_mode must be exactly "PAPER"'),
            (valid_config(port=4002), "port 4002 is rejected"),
        ]
        for config, message in cases:
            with self.subTest(message=message):
                report = self.build(config=config)

                self.assertTrue(report["success"])
                self.assertFalse(report["ibkr_config"]["valid"])
                self.assertFalse(report["submit_gate"]["eligible_for_future_manual_submit"])
                self.assertIn("validate_ibkr_paper_config failed: ValueError", report["errors"][0])
                self.assertIn(message, report["errors"][0])

    def test_blocked_ibkr_order_plan_is_not_eligible(self) -> None:
        report = self.build(
            intent=valid_intent(
                asset_type="OPTION",
                metadata={"expiry": "20260619"},
            )
        )

        self.assertTrue(report["intent"]["valid"])
        self.assertTrue(report["ibkr_config"]["valid"])
        self.assertTrue(report["ibkr_order_plan"]["available"])
        self.assertFalse(report["ibkr_order_plan"]["ready_for_submission"])
        self.assertIn(
            "OPTION contract hint requires numeric metadata.strike",
            report["ibkr_order_plan"]["blockers"],
        )
        self.assertFalse(report["submit_gate"]["eligible_for_future_manual_submit"])
        self.assertIn("IBKR paper order plan has blockers", report["submit_gate"]["reasons"])

    def test_missing_option_metadata_is_not_eligible(self) -> None:
        report = self.build(intent=valid_intent(asset_type="OPTION", metadata={}))

        self.assertFalse(report["ibkr_order_plan"]["ready_for_submission"])
        self.assertIn("OPTION contract hint requires metadata.expiry", report["ibkr_order_plan"]["blockers"])
        self.assertIn("OPTION contract hint requires numeric metadata.strike", report["ibkr_order_plan"]["blockers"])
        self.assertIn("OPTION contract hint requires metadata.right", report["ibkr_order_plan"]["blockers"])
        self.assertFalse(report["submit_gate"]["eligible_for_future_manual_submit"])

    def test_exception_from_build_broker_order_request_is_caught_with_details(self) -> None:
        with mock.patch.object(
            paper_order_ticket_report,
            "build_broker_order_request",
            side_effect=KeyError("strategy_id"),
        ):
            report = self.build()

        self.assertFalse(report["intent"]["valid"])
        self.assertIn(
            "build_broker_order_request failed: KeyError: 'strategy_id'",
            report["errors"],
        )

    def test_exception_from_validate_ibkr_paper_config_is_caught_with_details(self) -> None:
        with mock.patch.object(
            paper_order_ticket_report,
            "validate_ibkr_paper_config",
            side_effect=ValueError("trading_mode must be PAPER"),
        ):
            report = self.build()

        self.assertFalse(report["ibkr_config"]["valid"])
        self.assertIn(
            "validate_ibkr_paper_config failed: ValueError: trading_mode must be PAPER",
            report["errors"],
        )

    def test_exception_from_build_ibkr_paper_order_plan_is_caught_with_details(self) -> None:
        with mock.patch.object(
            paper_order_ticket_report,
            "build_ibkr_paper_order_plan",
            side_effect=RuntimeError("plan failed"),
        ):
            report = self.build()

        self.assertFalse(report["ibkr_order_plan"]["available"])
        self.assertIn(
            "build_ibkr_paper_order_plan failed: RuntimeError: plan failed",
            report["errors"],
        )
        self.assertFalse(report["submit_gate"]["eligible_for_future_manual_submit"])

    def test_report_includes_required_top_level_fields_and_is_json_safe(self) -> None:
        report = self.build()

        for field in (
            "dry_run",
            "paper_order_ticket_report",
            "generated_at",
            "intent",
            "broker_order_request",
            "ibkr_config",
            "ibkr_order_plan",
            "submit_gate",
            "safety",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(field, report)
        json.dumps(report, sort_keys=True)

    def test_acknowledgements_are_absent_when_not_eligible(self) -> None:
        report = self.build(intent={})

        self.assertFalse(report["submit_gate"]["eligible_for_future_manual_submit"])
        self.assertEqual(report["submit_gate"]["required_operator_acknowledgements"], [])

    def test_recommendations_are_deterministic_and_non_binding(self) -> None:
        first = self.build()["recommendations"]
        second = self.build()["recommendations"]

        self.assertEqual(first, second)
        self.assertIn(
            "Review this ticket manually before any future paper submit phase.",
            first["ordered_next_steps"],
        )
        joined = "\n".join(first["ordered_next_steps"] + first["do_not_do_yet"]).lower()
        for forbidden in ("automated paper execution", "change scheduler cadence"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, joined)

    def test_safety_flags_are_all_disabled(self) -> None:
        report = self.build()

        self.assertEqual(
            report["safety"],
            {
                "broker_calls_enabled": False,
                "order_submission_enabled": False,
                "cancel_enabled": False,
                "market_data_enabled": False,
                "contract_qualification_enabled": False,
                "live_orders_enabled": False,
                "scheduler_changes_enabled": False,
            },
        )


class PaperOrderTicketReportCliTests(unittest.TestCase):
    def run_cli(
        self,
        argv: list[str],
        *,
        state_store_factory: object | None = None,
        report_builder: object | None = None,
    ) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        kwargs = {}
        if state_store_factory is not None:
            kwargs["state_store_factory"] = state_store_factory
        if report_builder is not None:
            kwargs["report_builder"] = report_builder
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = ticket_tool.run_paper_order_ticket_report(argv, **kwargs)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_requires_dry_run_only_before_state_loading(self) -> None:
        class ExplodingStateStore:
            def __init__(self, path: Path) -> None:
                raise AssertionError("StateStore must not load before dry-run gate")

        code, stdout, stderr = self.run_cli(
            ["--intent-id", "intent-stage4e4-001"],
            state_store_factory=ExplodingStateStore,
        )

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("--dry-run-only", stderr)

    def test_cli_rejects_both_or_neither_intent_sources(self) -> None:
        for argv in (
            ["--dry-run-only"],
            ["--dry-run-only", "--intent-id", "one", "--intent-json", "{}"],
        ):
            with self.subTest(argv=argv):
                code, stdout, stderr = self.run_cli(argv)

                self.assertEqual(code, 1)
                self.assertEqual(stdout, "")
                self.assertIn("exactly one", stderr)

    def test_cli_intent_json_fallback_outputs_strict_json(self) -> None:
        intent = valid_intent(limit_price=1.25)
        code, stdout, stderr = self.run_cli(
            [
                "--dry-run-only",
                "--json",
                "--intent-json",
                json.dumps(intent),
            ]
        )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertTrue(payload["paper_order_ticket_report"])
        self.assertTrue(payload["submit_gate"]["eligible_for_future_manual_submit"])

    def test_cli_intent_id_uses_read_only_state_store_api(self) -> None:
        calls: list[tuple[str, object]] = []

        class FakeStateStore:
            def __init__(self, path: Path) -> None:
                calls.append(("init", path))

            def get_order_intent(self, intent_id: str) -> dict:
                calls.append(("get_order_intent", intent_id))
                return valid_intent(intent_id=intent_id)

            def save(self) -> None:
                raise AssertionError("ticket CLI must not mutate StateStore")

        code, stdout, stderr = self.run_cli(
            ["--dry-run-only", "--json", "--intent-id", "intent-from-state"],
            state_store_factory=FakeStateStore,
        )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["intent"]["intent_id"], "intent-from-state")
        self.assertEqual(calls[1], ("get_order_intent", "intent-from-state"))

    def test_cli_json_and_dry_run_only_are_boolean_flags(self) -> None:
        source = inspect.getsource(ticket_tool.run_paper_order_ticket_report)

        self.assertIn('parser.add_argument("--dry-run-only", action="store_true")', source)
        self.assertIn('parser.add_argument("--json", action="store_true"', source)

    def test_cli_exposes_no_submit_cancel_market_data_or_qualification_actions(self) -> None:
        source = inspect.getsource(ticket_tool.run_paper_order_ticket_report)
        for token in ("--submit", "--cancel", "--market-data", "--qualify"):
            with self.subTest(token=token):
                self.assertNotIn(token, source)


class Stage4E4SafetyBoundaryTests(unittest.TestCase):
    def test_stage4e4_files_do_not_import_or_call_blocked_integrations(self) -> None:
        blocked_tokens = (
            "ib_" + "insync",
            "place" + "Order",
            "cancel" + "Order",
            "submit_" + "order_plan",
            "req" + "MktData",
            "qualify" + "Contracts",
            "y" + "finance",
            "requests",
            "url" + "lib",
            "system" + "ctl",
            "system" + "d",
            "socket." + "create_connection",
            "socket." + "socket",
            "asyncio." + "run",
            "asyncio." + "get_event_loop",
            "asyncio." + "new_event_loop",
            "uuid." + "uuid4",
            "rand" + "om",
        )
        for path in STAGE4E4_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_stage4e4_files_do_not_wire_scheduler_lifecycle_or_execution_clients(self) -> None:
        forbidden_tokens = (
            "IbkrPaperExecutionClient",
            "IbkrPaperReadOnlyClient",
            "PaperBrokerAdapter",
            "core.scheduler",
            "jobs.submission",
            "jobs.confirmation",
            "jobs.fill_confirmation",
        )
        for path in STAGE4E4_FILES:
            source = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
