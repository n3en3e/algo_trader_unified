from __future__ import annotations

import copy
from datetime import datetime, timezone
from decimal import Decimal
import inspect
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4g1_lifecycle_intake_report import (
    build_stage4g1_lifecycle_intake_report,
)
from algo_trader_unified.tools import stage4g1_lifecycle_intake_report as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4G1_FILES = [
    ROOT / "core/stage4g1_lifecycle_intake_report.py",
    ROOT / "tools/stage4g1_lifecycle_intake_report.py",
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


def valid_stage4f(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4f_acceptance_report": True,
        "readiness_for_stage4g": {
            "ready_to_begin_manual_paper_lifecycle_validation": True,
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
    _deep_update(report, overrides)
    return report


def valid_smoke(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4f5_smoke_test_report": True,
        "smoke_test": {
            "accepted": True,
            "one_order_only": True,
            "broker_order_id": "9001",
            "client_order_id": "intent-stage4g1-001",
            "submitted": True,
            "status_seen": True,
            "cancel_seen": False,
            "terminal_or_safe_state_seen": True,
        },
        "order_control_summary": {
            "last_known_status": "Submitted",
            "cancel_attempted": False,
            "cancel_succeeded": None,
        },
        "safety_checks": {
            "no_live_orders": True,
            "no_market_data": True,
            "no_contract_qualification": True,
            "no_scheduler_changes": True,
            "no_lifecycle_wiring": True,
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
        "submission": {
            "attempted": True,
            "submitted": True,
            "broker_order_id": "9001",
            "client_order_id": "intent-stage4g1-001",
            "strategy_id": "S01_VOL_BASELINE",
            "symbol": "XSP",
            "side": "BUY",
            "quantity": 1,
            "order_type": "LIMIT",
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
    _deep_update(report, overrides)
    return report


def valid_status(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "dry_run": True,
        "manual_real_paper_order_control": True,
        "action": "status",
        "order": {"broker_order_id": "9001", "client_order_id": "intent-stage4g1-001"},
        "status": {
            "status": "Submitted",
            "filled": 0,
            "remaining": 1,
            "avg_fill_price": 0,
        },
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
        "order": {"broker_order_id": "9001", "client_order_id": "intent-stage4g1-001"},
        "cancel": {"cancelled": True},
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
        "stage4f_acceptance_report": valid_stage4f(),
        "smoke_test_report": valid_smoke(),
        "submit_report": valid_submit(),
        "order_control_reports": [valid_status()],
        "existing_state_snapshot": None,
        "now_provider": lambda: datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc),
    }
    kwargs.update(overrides)
    return build_stage4g1_lifecycle_intake_report(**kwargs)


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


class Stage4G1LifecycleIntakeReportTests(unittest.TestCase):
    def assert_blocked(self, report: dict, expected: str) -> None:
        self.assertFalse(
            report["readiness_for_stage4g2"][
                "ready_to_build_manual_lifecycle_state_preview"
            ]
        )
        self.assertIn(expected, report["readiness_for_stage4g2"]["blockers"])

    def assert_state(self, report: dict, expected: str) -> None:
        self.assertEqual(
            report["lifecycle_intake_candidate"]["suggested_internal_lifecycle_state"],
            expected,
        )

    def test_missing_4f_acceptance_report_blocks_readiness(self) -> None:
        self.assert_blocked(
            build(stage4f_acceptance_report=None),
            "stage4f_acceptance_report missing",
        )

    def test_4f_acceptance_not_ready_blocks_readiness(self) -> None:
        report = build(
            stage4f_acceptance_report=valid_stage4f(
                readiness_for_stage4g={
                    "ready_to_begin_manual_paper_lifecycle_validation": False
                }
            )
        )
        self.assert_blocked(report, "Stage 4F acceptance is not ready for Stage 4G")

    def test_missing_smoke_test_report_blocks_readiness(self) -> None:
        self.assert_blocked(build(smoke_test_report=None), "smoke_test_report missing")

    def test_unaccepted_smoke_test_blocks_readiness(self) -> None:
        report = build(smoke_test_report=valid_smoke(smoke_test={"accepted": False}))
        self.assert_blocked(report, "smoke_test.accepted must be True")

    def test_missing_submit_report_blocks_readiness(self) -> None:
        self.assert_blocked(build(submit_report=None), "submit_report missing")

    def test_failed_submit_blocks_readiness(self) -> None:
        report = build(submit_report=valid_submit(submission={"submitted": False}))
        self.assert_blocked(report, "submission.submitted must be True")

    def test_missing_broker_order_id_blocks_readiness(self) -> None:
        report = build(
            smoke_test_report=valid_smoke(smoke_test={"broker_order_id": None}),
            submit_report=valid_submit(submission={"broker_order_id": None}),
        )
        self.assert_blocked(report, "broker_order_id is required")

    def test_missing_client_order_id_blocks_readiness(self) -> None:
        report = build(
            smoke_test_report=valid_smoke(smoke_test={"client_order_id": None}),
            submit_report=valid_submit(submission={"client_order_id": None}),
        )
        self.assert_blocked(report, "client_order_id is required")

    def test_broker_order_id_mismatch_blocks_and_maps_to_needs_reconciliation(self) -> None:
        report = build(order_control_reports=[valid_status(order={"broker_order_id": "9002"})])

        self.assert_blocked(report, "broker_order_id mismatch across artifacts")
        self.assert_state(report, "needs_reconciliation")
        self.assertTrue(report["lifecycle_intake_candidate"]["reconciliation_required"])

    def test_client_order_id_mismatch_blocks_and_maps_to_needs_reconciliation(self) -> None:
        report = build(order_control_reports=[valid_status(order={"client_order_id": "other"})])

        self.assert_blocked(report, "client_order_id mismatch across artifacts")
        self.assert_state(report, "needs_reconciliation")

    def test_valid_accepted_smoke_submitted_order_and_submitted_status(self) -> None:
        report = build()

        self.assertTrue(
            report["readiness_for_stage4g2"][
                "ready_to_build_manual_lifecycle_state_preview"
            ]
        )
        self.assert_state(report, "broker_submitted")
        self.assertFalse(report["lifecycle_intake_candidate"]["reconciliation_required"])
        self.assertEqual(report["lifecycle_intake_candidate"]["broker_order_id"], "9001")
        self.assertEqual(report["lifecycle_intake_candidate"]["client_order_id"], "intent-stage4g1-001")
        self.assertEqual(report["lifecycle_intake_candidate"]["strategy_id"], "S01_VOL_BASELINE")

    def test_filled_status_with_remaining_zero_maps_to_broker_filled(self) -> None:
        report = build(order_control_reports=[valid_status(status={"status": "Filled", "remaining": 0})])

        self.assert_state(report, "broker_filled")
        self.assertFalse(report["lifecycle_intake_candidate"]["reconciliation_required"])

    def test_filled_status_with_remaining_string_zero_maps_to_broker_filled(self) -> None:
        report = build(order_control_reports=[valid_status(status={"status": "Filled", "remaining": "0"})])

        self.assert_state(report, "broker_filled")

    def test_filled_status_with_remaining_string_zero_point_zero_maps_to_broker_filled(self) -> None:
        report = build(order_control_reports=[valid_status(status={"status": "Filled", "remaining": "0.0"})])

        self.assert_state(report, "broker_filled")

    def test_filled_status_with_remaining_positive_maps_to_partial(self) -> None:
        report = build(order_control_reports=[valid_status(status={"status": "Filled", "remaining": 1})])

        self.assert_state(report, "broker_partially_filled")
        self.assertTrue(report["lifecycle_intake_candidate"]["reconciliation_required"])
        self.assert_blocked(report, "candidate state requires manual review: broker_partially_filled")

    def test_filled_status_with_remaining_string_positive_maps_to_partial(self) -> None:
        for value in ("1", "1.0"):
            with self.subTest(value=value):
                report = build(order_control_reports=[valid_status(status={"status": "Filled", "remaining": value})])
                self.assert_state(report, "broker_partially_filled")

    def test_partially_filled_status_maps_to_partial_and_requires_reconciliation(self) -> None:
        report = build(order_control_reports=[valid_status(status={"status": "PartiallyFilled", "remaining": "0.5"})])

        self.assert_state(report, "broker_partially_filled")
        self.assertTrue(report["lifecycle_intake_candidate"]["reconciliation_required"])

    def test_cancelled_status_maps_to_broker_cancelled(self) -> None:
        report = build(order_control_reports=[valid_status(status={"status": "Cancelled", "remaining": 1})])

        self.assert_state(report, "broker_cancelled")
        self.assertFalse(report["lifecycle_intake_candidate"]["reconciliation_required"])

    def test_rejected_or_inactive_status_requires_reconciliation(self) -> None:
        for status in ("Rejected", "Inactive"):
            with self.subTest(status=status):
                report = build(order_control_reports=[valid_status(status={"status": status, "remaining": 1})])
                self.assert_state(report, "broker_rejected_or_inactive")
                self.assertTrue(report["lifecycle_intake_candidate"]["reconciliation_required"])

    def test_unknown_status_requires_reconciliation(self) -> None:
        report = build(order_control_reports=[valid_status(status={"status": "Mystery", "remaining": 1})])

        self.assert_state(report, "unknown_broker_status")
        self.assertTrue(report["lifecycle_intake_candidate"]["reconciliation_required"])

    def test_unparseable_remaining_quantity_does_not_crash(self) -> None:
        report = build(order_control_reports=[valid_status(status={"status": "Filled", "remaining": object()})])

        self.assert_state(report, "unknown_broker_status")
        self.assertTrue(report["lifecycle_intake_candidate"]["reconciliation_required"])
        self.assertIn(
            "remaining_quantity could not be parsed for Filled status",
            report["lifecycle_intake_candidate"]["reconciliation_reasons"],
        )

    def test_cancel_attempted_but_final_status_unknown_maps_to_unverified(self) -> None:
        report = build(order_control_reports=[valid_cancel(cancel={"cancelled": False})])

        self.assert_state(report, "cancel_requested_unverified")
        self.assertTrue(report["lifecycle_intake_candidate"]["reconciliation_required"])
        self.assertIn(
            "candidate requires manual follow-up: cancel_requested_unverified",
            report["warnings"],
        )

    def test_status_report_required_unless_operator_notes_document_follow_up(self) -> None:
        report = build(order_control_reports=[])

        self.assert_blocked(report, "at least one status report is required")

        noted = build(
            smoke_test_report=valid_smoke(
                operator_notes={
                    "order_intentionally_left_open": True,
                    "manual_observation": "Submitted and left open for manual follow-up.",
                }
            ),
            order_control_reports=[],
        )
        self.assertNotIn("at least one status report is required", noted["readiness_for_stage4g2"]["blockers"])
        self.assert_state(noted, "submitted_unverified")

    def test_safety_flags_true_anywhere_block_readiness(self) -> None:
        cases = [
            ("live_orders_enabled", "no_live_orders", {"submit_report": valid_submit(safety={"live_orders_enabled": True})}),
            ("market_data_enabled", "no_market_data", {"stage4f_acceptance_report": valid_stage4f(safety={"market_data_enabled": True})}),
            ("contract_qualification_enabled", "no_contract_qualification", {"smoke_test_report": valid_smoke(safety_checks={"contract_qualification_enabled": True})}),
            ("scheduler_changes_enabled", "no_scheduler_changes", {"order_control_reports": [valid_status(safety=order_control_safety(scheduler_changes_enabled=True))]}),
            ("lifecycle_wiring_enabled", "no_lifecycle_wiring", {"order_control_reports": [valid_status(safety=order_control_safety(lifecycle_wiring_enabled=True))]}),
        ]
        for flag, check, kwargs in cases:
            with self.subTest(flag=flag):
                report = build(**kwargs)
                self.assertFalse(report["safety_checks"][check])
                self.assert_state(report, "unsafe_artifact")

    def test_state_snapshot_unresolved_reconciliation_blocks_readiness(self) -> None:
        report = build(existing_state_snapshot={"unresolved_needs_reconciliation_count": 1})

        self.assert_blocked(report, "unresolved NEEDS_RECONCILIATION records exist")

    def test_state_snapshot_active_halt_blocks_readiness(self) -> None:
        report = build(existing_state_snapshot={"active_halt": True})

        self.assert_blocked(report, "active halt is present")

    def test_none_and_malformed_nested_dicts_do_not_crash(self) -> None:
        report = build_stage4g1_lifecycle_intake_report(
            stage4f_acceptance_report={"readiness_for_stage4g": "bad"},
            smoke_test_report={"smoke_test": "bad"},
            submit_report={"submission": "bad"},
            order_control_reports=[{"status": "bad"}, "ignored"],  # type: ignore[list-item]
            existing_state_snapshot="bad",  # type: ignore[arg-type]
            now_provider=lambda: datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc),
        )

        self.assertFalse(report["readiness_for_stage4g2"]["ready_to_build_manual_lifecycle_state_preview"])
        assert_json_safe(self, report)

    def test_input_reports_are_not_mutated(self) -> None:
        stage4f = valid_stage4f()
        smoke = valid_smoke()
        submit = valid_submit()
        controls = [valid_status()]
        original = copy.deepcopy((stage4f, smoke, submit, controls))

        build_stage4g1_lifecycle_intake_report(
            stage4f_acceptance_report=stage4f,
            smoke_test_report=smoke,
            submit_report=submit,
            order_control_reports=controls,
            now_provider=lambda: datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual((stage4f, smoke, submit, controls), original)

    def test_report_includes_required_top_level_fields_and_is_json_safe(self) -> None:
        report = build(
            order_control_reports=[
                valid_status(status={"status": "Filled", "remaining": Decimal("0")})
            ]
        )

        for key in (
            "dry_run",
            "stage4g1_lifecycle_intake_report",
            "generated_at",
            "artifact_checks",
            "consistency_checks",
            "lifecycle_intake_candidate",
            "safety_checks",
            "state_snapshot_checks",
            "readiness_for_stage4g2",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        assert_json_safe(self, report)


class Stage4G1LifecycleIntakeCliTests(unittest.TestCase):
    def test_cli_requires_dry_run_only_before_parsing_json(self) -> None:
        calls = {"report_builder": 0}

        def report_builder(**kwargs):
            calls["report_builder"] += 1
            return {}

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_stage4g1_lifecycle_intake_report(
                [
                    "--stage4f-acceptance-json",
                    "{bad",
                    "--smoke-test-json",
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
            code = tool.run_stage4g1_lifecycle_intake_report(
                [
                    "--dry-run-only",
                    "--json",
                    "--stage4f-acceptance-json",
                    json.dumps(valid_stage4f()),
                    "--smoke-test-json",
                    json.dumps(valid_smoke()),
                    "--submit-json",
                    json.dumps(valid_submit()),
                    "--order-control-json",
                    json.dumps(valid_status()),
                ]
            )

        parsed = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(parsed["stage4g1_lifecycle_intake_report"])

    def test_cli_reports_json_parse_errors_with_type_and_string(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4g1_lifecycle_intake_report(
                [
                    "--dry-run-only",
                    "--json",
                    "--stage4f-acceptance-json",
                    "{bad",
                    "--smoke-test-json",
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
                "readiness_for_stage4g2": {
                    "ready_to_build_manual_lifecycle_state_preview": True
                },
            }

        with redirect_stdout(io.StringIO()):
            code = tool.run_stage4g1_lifecycle_intake_report(
                [
                    "--dry-run-only",
                    "--json",
                    "--stage4f-acceptance-json",
                    "{}",
                    "--smoke-test-json",
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
        source = inspect.getsource(tool.run_stage4g1_lifecycle_intake_report)

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


class Stage4G1LifecycleIntakeSafetyTests(unittest.TestCase):
    def test_stage4g1_files_do_not_call_forbidden_external_apis(self) -> None:
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
        for path in STAGE4G1_FILES:
            source = path.read_text(encoding="utf-8")
            for value in forbidden:
                with self.subTest(path=path.name, value=value):
                    self.assertNotIn(value, source)

    def test_no_daemon_scheduler_or_lifecycle_wiring_exists(self) -> None:
        for path in UNWIRED_RUNTIME_FILES:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("stage4g1_lifecycle_intake_report", source)


if __name__ == "__main__":
    unittest.main()
