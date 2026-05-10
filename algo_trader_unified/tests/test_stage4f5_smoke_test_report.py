from __future__ import annotations

import copy
import inspect
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from algo_trader_unified.core.stage4f5_smoke_test_report import (
    build_stage4f5_smoke_test_report,
)
from algo_trader_unified.tools import stage4f5_smoke_test_report as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4F5_FILES = [
    ROOT / "core/stage4f5_smoke_test_report.py",
    ROOT / "tools/stage4f5_smoke_test_report.py",
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


def valid_preflight(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "dry_run": True,
        "ibkr_paper_connection_preflight": True,
        "connection": {
            "allow_real_ibkr": True,
            "attempted": True,
            "connected": True,
            "paper_mode": True,
            "disconnected": True,
        },
        "readonly_checks": {
            "current_time_ok": True,
            "account_snapshot_ok": True,
            "open_orders_ok": True,
            "positions_ok": True,
        },
        "safety": {
            "real_ibkr_connection_enabled": True,
            "paper_order_submission_enabled": False,
            "cancel_enabled": False,
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
    _deep_update(report, overrides)
    return report


def valid_ticket(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "dry_run": True,
        "paper_order_ticket_report": True,
        "ibkr_order_plan": {
            "ready_for_submission": True,
            "paper_only": True,
            "dry_run": True,
            "blockers": [],
            "client_order_id": "intent-stage4f5-001",
        },
        "submit_gate": {"eligible_for_future_manual_submit": True},
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
    _deep_update(report, overrides)
    return report


def valid_submit(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "dry_run": True,
        "manual_real_paper_submit": True,
        "gates": {"passed": True, "reasons": []},
        "submission": {
            "attempted": True,
            "submitted": True,
            "broker_order_id": "9001",
            "client_order_id": "intent-stage4f5-001",
        },
        "safety": {
            "real_ibkr_enabled": True,
            "real_paper_submit_enabled": True,
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
    _deep_update(report, overrides)
    return report


def valid_status(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "dry_run": True,
        "manual_real_paper_order_control": True,
        "action": "status",
        "order": {"broker_order_id": "9001"},
        "status": {"status": "Filled"},
        "safety": order_control_safety(),
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def valid_cancel(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "dry_run": True,
        "manual_real_paper_order_control": True,
        "action": "cancel",
        "order": {"broker_order_id": "9001"},
        "cancel": {"cancelled": True, "reason": None},
        "safety": order_control_safety(),
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def order_control_safety(**overrides: object) -> dict[str, object]:
    safety: dict[str, object] = {
        "live_orders_enabled": False,
        "paper_order_submission_enabled": False,
        "market_data_enabled": False,
        "contract_qualification_enabled": False,
        "scheduler_changes_enabled": False,
        "lifecycle_wiring_enabled": False,
    }
    safety.update(overrides)
    return safety


def build(**overrides: object) -> dict:
    kwargs = {
        "connection_preflight_report": valid_preflight(),
        "ticket_report": valid_ticket(),
        "submit_report": valid_submit(),
        "order_control_reports": [valid_status()],
        "operator_notes": None,
        "now_provider": lambda: datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc),
    }
    kwargs.update(overrides)
    return build_stage4f5_smoke_test_report(**kwargs)


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


def assert_json_safe(test_case: unittest.TestCase, report: dict) -> None:
    serialized = json.dumps(report, sort_keys=True)
    test_case.assertNotIn("datetime", serialized)
    test_case.assertNotIn("Decimal", serialized)


class Stage4F5SmokeTestReportTests(unittest.TestCase):
    def assert_blocked(self, report: dict, expected: str) -> None:
        self.assertFalse(report["readiness_for_stage4f6"]["ready_for_stage4f_acceptance_report"])
        self.assertIn(expected, report["readiness_for_stage4f6"]["blockers"])

    def test_valid_artifacts_accept_smoke_test(self) -> None:
        report = build()

        self.assertTrue(report["smoke_test"]["accepted"])
        self.assertTrue(report["readiness_for_stage4f6"]["ready_for_stage4f_acceptance_report"])
        self.assertEqual(report["smoke_test"]["broker_order_id"], "9001")
        self.assertEqual(report["smoke_test"]["client_order_id"], "intent-stage4f5-001")
        self.assertTrue(report["smoke_test"]["one_order_only"])
        self.assertEqual(report["order_control_summary"]["last_known_status"], "Filled")

    def test_missing_connection_preflight_blocks_readiness(self) -> None:
        self.assert_blocked(
            build(connection_preflight_report=None),
            "connection preflight report is required",
        )

    def test_missing_ticket_report_blocks_readiness(self) -> None:
        self.assert_blocked(build(ticket_report=None), "ticket report is required")

    def test_missing_submit_report_blocks_readiness(self) -> None:
        self.assert_blocked(build(submit_report=None), "submit report is required")

    def test_invalid_preflight_blocks_readiness(self) -> None:
        report = build(connection_preflight_report=valid_preflight(connection={"connected": False}))

        self.assert_blocked(report, "connection.connected must be True")

    def test_invalid_ticket_blocks_readiness(self) -> None:
        report = build(ticket_report=valid_ticket(ibkr_order_plan={"ready_for_submission": False}))

        self.assert_blocked(report, "ticket plan ready_for_submission must be True")

    def test_failed_submit_blocks_readiness(self) -> None:
        report = build(submit_report=valid_submit(submission={"submitted": False}))

        self.assert_blocked(report, "submit submission.submitted must be True")

    def test_missing_broker_order_id_blocks_readiness(self) -> None:
        report = build(submit_report=valid_submit(submission={"broker_order_id": None}))

        self.assert_blocked(report, "submit broker_order_id is required")

    def test_client_order_id_mismatch_blocks_readiness(self) -> None:
        report = build(submit_report=valid_submit(submission={"client_order_id": "other"}))

        self.assert_blocked(report, "submit client_order_id must match ticket client_order_id")

    def test_broker_order_id_mismatch_in_status_blocks_readiness(self) -> None:
        report = build(order_control_reports=[valid_status(order={"broker_order_id": "9002"})])

        self.assert_blocked(report, "order_control[0] broker_order_id must match submitted broker_order_id")

    def test_broker_order_id_mismatch_in_cancel_blocks_readiness(self) -> None:
        report = build(order_control_reports=[valid_status(), valid_cancel(order={"broker_order_id": "9002"})])

        self.assert_blocked(report, "order_control[1] broker_order_id must match submitted broker_order_id")

    def test_no_status_report_blocks_readiness_and_warns(self) -> None:
        report = build(order_control_reports=[])

        self.assert_blocked(report, "at least one matching status report is required")
        self.assertIn(
            "no cancel report was supplied; verify the paper order is filled, closed, cancelled, or intentionally tracked",
            report["warnings"],
        )

    def test_operator_notes_only_waive_terminal_status_requirement(self) -> None:
        report = build(
            order_control_reports=[valid_status(status={"status": "Submitted"})],
            operator_notes={
                "order_intentionally_left_open": True,
                "manual_observation": "Submitted paper order left open for manual follow-up.",
                "follow_up_required": True,
                "cleanup_ticket": "OPS-4F5",
            },
        )

        self.assertTrue(report["smoke_test"]["accepted"])
        self.assertTrue(report["readiness_for_stage4f6"]["ready_for_stage4f_acceptance_report"])
        self.assertIn(
            "operator notes document that the paper order was intentionally left open for follow-up",
            report["warnings"],
        )

    def test_operator_notes_cannot_bypass_missing_or_invalid_artifacts(self) -> None:
        notes = {"order_intentionally_left_open": True, "manual_observation": "tracked"}

        cases = [
            ({"connection_preflight_report": None}, "connection preflight report is required"),
            ({"ticket_report": None}, "ticket report is required"),
            ({"submit_report": None}, "submit report is required"),
            (
                {"connection_preflight_report": valid_preflight(connection={"connected": False})},
                "connection.connected must be True",
            ),
            (
                {"ticket_report": valid_ticket(ibkr_order_plan={"client_order_id": "other"})},
                "submit client_order_id must match ticket client_order_id",
            ),
            (
                {"submit_report": valid_submit(submission={"submitted": False})},
                "submit submission.submitted must be True",
            ),
        ]
        for kwargs, expected in cases:
            with self.subTest(expected=expected):
                report = build(operator_notes=notes, **kwargs)
                self.assert_blocked(report, expected)

    def test_operator_notes_cannot_bypass_broker_or_client_mismatches(self) -> None:
        notes = {"order_intentionally_left_open": True, "manual_observation": "tracked"}

        broker_mismatch = build(
            order_control_reports=[valid_status(order={"broker_order_id": "9999"})],
            operator_notes=notes,
        )
        client_mismatch = build(
            submit_report=valid_submit(submission={"client_order_id": "other"}),
            operator_notes=notes,
        )

        self.assert_blocked(
            broker_mismatch,
            "order_control[0] broker_order_id must match submitted broker_order_id",
        )
        self.assert_blocked(client_mismatch, "submit client_order_id must match ticket client_order_id")

    def test_operator_notes_cannot_bypass_safety_flags(self) -> None:
        notes = {"order_intentionally_left_open": True, "manual_observation": "tracked"}
        report = build(
            ticket_report=valid_ticket(safety={"market_data_enabled": True}),
            operator_notes=notes,
        )

        self.assert_blocked(report, "unsafe flag enabled: market_data_enabled in supplied report 1")

    def test_cancel_report_matching_broker_order_id_is_summarized(self) -> None:
        report = build(order_control_reports=[valid_status(), valid_cancel()])

        self.assertTrue(report["smoke_test"]["cancel_seen"])
        self.assertTrue(report["order_control_summary"]["cancel_attempted"])
        self.assertTrue(report["order_control_summary"]["cancel_succeeded"])

    def test_cancel_failure_is_summarized_without_masking_smoke_result(self) -> None:
        report = build(
            order_control_reports=[
                valid_status(),
                valid_cancel(cancel={"cancelled": False, "reason": "already filled"}),
            ]
        )

        self.assertTrue(report["smoke_test"]["accepted"])
        self.assertFalse(report["order_control_summary"]["cancel_succeeded"])
        self.assertIn("cancel did not succeed: already filled", report["warnings"])

    def test_safety_flags_true_anywhere_block_readiness(self) -> None:
        cases = [
            ("live_orders_enabled", valid_submit(safety={"live_orders_enabled": True}), 2),
            ("market_data_enabled", valid_preflight(safety={"market_data_enabled": True}), 0),
            (
                "contract_qualification_enabled",
                valid_ticket(safety={"contract_qualification_enabled": True}),
                1,
            ),
            ("scheduler_changes_enabled", valid_status(safety=order_control_safety(scheduler_changes_enabled=True)), 3),
            ("lifecycle_wiring_enabled", valid_cancel(safety=order_control_safety(lifecycle_wiring_enabled=True)), 4),
        ]
        for key, artifact, index in cases:
            with self.subTest(key=key):
                kwargs: dict[str, object] = {}
                if index == 0:
                    kwargs["connection_preflight_report"] = artifact
                elif index == 1:
                    kwargs["ticket_report"] = artifact
                elif index == 2:
                    kwargs["submit_report"] = artifact
                elif index == 3:
                    kwargs["order_control_reports"] = [artifact]
                else:
                    kwargs["order_control_reports"] = [valid_status(), artifact]
                report = build(**kwargs)
                self.assertFalse(report["safety_checks"][{
                    "live_orders_enabled": "no_live_orders",
                    "market_data_enabled": "no_market_data",
                    "contract_qualification_enabled": "no_contract_qualification",
                    "scheduler_changes_enabled": "no_scheduler_changes",
                    "lifecycle_wiring_enabled": "no_lifecycle_wiring",
                }[key]])

    def test_multiple_submitted_broker_order_ids_blocks_readiness(self) -> None:
        submit = valid_submit(submissions=[
            {"submitted": True, "broker_order_id": "9001"},
            {"submitted": True, "broker_order_id": "9002"},
        ])

        report = build(submit_report=submit)

        self.assertFalse(report["safety_checks"]["no_extra_submissions_detected"])
        self.assert_blocked(report, "multiple submitted broker_order_ids detected")

    def test_input_reports_are_not_mutated(self) -> None:
        preflight = valid_preflight()
        ticket = valid_ticket()
        submit = valid_submit()
        controls = [valid_status()]
        original = copy.deepcopy((preflight, ticket, submit, controls))

        build(
            connection_preflight_report=preflight,
            ticket_report=ticket,
            submit_report=submit,
            order_control_reports=controls,
        )

        self.assertEqual((preflight, ticket, submit, controls), original)

    def test_report_includes_required_top_level_fields_and_is_json_safe(self) -> None:
        report = build(operator_notes={"manual_observation": datetime(2026, 5, 10, tzinfo=timezone.utc)})

        for key in (
            "dry_run",
            "stage4f5_smoke_test_report",
            "generated_at",
            "smoke_test",
            "artifact_checks",
            "sequence_checks",
            "safety_checks",
            "order_control_summary",
            "readiness_for_stage4f6",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        assert_json_safe(self, report)


class Stage4F5SmokeTestCliTests(unittest.TestCase):
    def test_cli_requires_dry_run_only_before_parsing_json(self) -> None:
        calls = {"report_builder": 0}

        def report_builder(**kwargs):
            calls["report_builder"] += 1
            return {}

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_stage4f5_smoke_test_report(
                [
                    "--connection-preflight-json",
                    "{bad",
                    "--ticket-json",
                    "{}",
                    "--submit-json",
                    "{}",
                ],
                report_builder=report_builder,
            )

        self.assertEqual(code, 1)
        self.assertEqual(calls["report_builder"], 0)
        self.assertIn("--dry-run-only", stderr.getvalue())
        self.assertNotIn("JSONDecodeError", stderr.getvalue())

    def test_cli_json_stdout_is_strict_json_only(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4f5_smoke_test_report(
                [
                    "--dry-run-only",
                    "--json",
                    "--connection-preflight-json",
                    json.dumps(valid_preflight()),
                    "--ticket-json",
                    json.dumps(valid_ticket()),
                    "--submit-json",
                    json.dumps(valid_submit()),
                    "--order-control-json",
                    json.dumps(valid_status()),
                ]
            )

        parsed = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(parsed["stage4f5_smoke_test_report"])

    def test_cli_reports_json_parse_errors_with_type_and_string(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4f5_smoke_test_report(
                [
                    "--dry-run-only",
                    "--json",
                    "--connection-preflight-json",
                    "{bad",
                    "--ticket-json",
                    "{}",
                    "--submit-json",
                    "{}",
                ]
            )

        parsed = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertIn("JSONDecodeError", parsed["errors"][0])

    def test_cli_order_control_json_supports_multiple_values(self) -> None:
        calls: dict[str, object] = {}

        def report_builder(**kwargs):
            calls.update(kwargs)
            return {
                "success": True,
                "readiness_for_stage4f6": {
                    "ready_for_stage4f_acceptance_report": True
                },
            }

        with redirect_stdout(io.StringIO()):
            code = tool.run_stage4f5_smoke_test_report(
                [
                    "--dry-run-only",
                    "--json",
                    "--connection-preflight-json",
                    "{}",
                    "--ticket-json",
                    "{}",
                    "--submit-json",
                    "{}",
                    "--order-control-json",
                    '{"action":"status"}',
                    "--order-control-json",
                    '{"action":"cancel"}',
                ],
                report_builder=report_builder,
            )

        self.assertEqual(code, 0)
        self.assertEqual(
            calls["order_control_reports"],
            [{"action": "status"}, {"action": "cancel"}],
        )

    def test_cli_uses_append_for_order_control_json(self) -> None:
        source = inspect.getsource(tool.run_stage4f5_smoke_test_report)

        self.assertIn('parser.add_argument("--order-control-json", action="append"', source)

    def test_cli_exposes_no_execution_or_market_data_actions(self) -> None:
        source = inspect.getsource(tool)
        forbidden = [
            "--allow-real-ibkr",
            "--allow-real-paper-submit",
            "--allow-live",
            "--live",
            "--market-data",
            "--qualify",
            "--status",
            "--cancel",
            "--scheduler",
            "--lifecycle",
        ]
        for value in forbidden:
            with self.subTest(value=value):
                self.assertNotIn(value, source)


class Stage4F5SmokeTestSafetyTests(unittest.TestCase):
    def test_stage4f5_files_do_not_call_forbidden_external_apis(self) -> None:
        forbidden = [
            "submit_" + "order_plan",
            "get_" + "order_status",
            "cancel_" + "order",
            "place" + "Order(",
            "cancel" + "Order(",
            "req" + "MktData",
            "qualify" + "Contracts",
            "yfinance",
            "requests",
            "urllib",
            "systemctl",
            "system" + "d",
            "socket.create_connection",
            "socket.socket",
            "asyncio.run",
            "asyncio.get_event_loop",
            "asyncio.new_event_loop",
            "uuid.uuid4",
            "random",
        ]
        for path in STAGE4F5_FILES:
            source = path.read_text(encoding="utf-8")
            for value in forbidden:
                with self.subTest(path=path.name, value=value):
                    self.assertNotIn(value, source)

    def test_no_daemon_scheduler_or_lifecycle_wiring_exists(self) -> None:
        for path in UNWIRED_RUNTIME_FILES:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("stage4f5_smoke_test_report", source)


if __name__ == "__main__":
    unittest.main()
