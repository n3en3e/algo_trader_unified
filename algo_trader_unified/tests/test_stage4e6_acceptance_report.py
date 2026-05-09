from __future__ import annotations

import inspect
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from algo_trader_unified.core.stage4e_acceptance_report import (
    MODULE_CHECK_KEYS,
    PHASE_KEYS,
    SAFETY_CHECK_KEYS,
    build_stage4e_acceptance_report,
)
from algo_trader_unified.tools import stage4e_acceptance_report as report_tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4E6_FILES = [
    ROOT / "core/stage4e_acceptance_report.py",
    ROOT / "tools/stage4e_acceptance_report.py",
    ROOT / "tests/test_stage4e6_acceptance_report.py",
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


def passed_reports(**overrides: object) -> dict[str, dict]:
    reports = {key: {"success": True} for key in PHASE_KEYS}
    reports.update(overrides)
    return reports


def passed_module_checks(**overrides: bool) -> dict[str, bool]:
    checks = {key: True for key in MODULE_CHECK_KEYS}
    checks.update(overrides)
    return checks


def passed_safety_checks(**overrides: bool) -> dict[str, bool]:
    checks = {key: True for key in SAFETY_CHECK_KEYS}
    checks.update(overrides)
    return checks


def safe_state(**overrides: object) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "unresolved_needs_reconciliation_count": 0,
        "active_intents_count": 0,
        "open_positions_count": 0,
        "active_halt": False,
        "unknown_broker_exposure_count": 0,
    }
    snapshot.update(overrides)
    return snapshot


class Stage4EAcceptanceReportCoreTests(unittest.TestCase):
    def build(self, **kwargs: object) -> dict:
        defaults: dict[str, object] = {
            "reports": passed_reports(),
            "module_checks": passed_module_checks(),
            "safety_checks": passed_safety_checks(),
            "state_snapshot": safe_state(),
            "now_provider": lambda: datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
        }
        defaults.update(kwargs)
        return build_stage4e_acceptance_report(**defaults)

    def assert_not_ready_with(self, report: dict, expected: str) -> None:
        readiness = report["readiness_for_stage4f"]
        self.assertFalse(readiness["ready_to_begin_real_ibkr_paper_submit_planning"])
        self.assertIn(expected, readiness["blockers"])

    def test_none_inputs_do_not_raise_and_report_missing_inputs(self) -> None:
        report = build_stage4e_acceptance_report(
            reports=None,
            module_checks=None,
            safety_checks=None,
            state_snapshot=None,
            now_provider=lambda: datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
        )

        self.assertTrue(report["stage4e_acceptance_report"])
        self.assertFalse(
            report["readiness_for_stage4f"][
                "ready_to_begin_real_ibkr_paper_submit_planning"
            ]
        )
        self.assertIn(
            "state_snapshot unavailable",
            report["readiness_for_stage4f"]["blockers"],
        )
        json.dumps(report, sort_keys=True)

    def test_all_prior_phase_statuses_present_and_passed_are_ready(self) -> None:
        report = self.build()

        self.assertTrue(
            report["readiness_for_stage4f"][
                "ready_to_begin_real_ibkr_paper_submit_planning"
            ]
        )
        self.assertEqual(report["readiness_for_stage4f"]["blockers"], [])
        self.assertTrue(report["success"])

    def test_required_top_level_fields_are_present(self) -> None:
        report = self.build()

        for key in (
            "dry_run",
            "stage4e_acceptance_report",
            "generated_at",
            "phase_status",
            "module_checks",
            "safety_checks",
            "state_safety",
            "readiness_for_stage4f",
            "recommendations",
            "safety",
            "success",
            "errors",
            "warnings",
        ):
            with self.subTest(key=key):
                self.assertIn(key, report)

    def test_each_missing_phase_blocks_readiness(self) -> None:
        expected = {
            "stage4e1_readonly_client": "stage4e1_readonly_client missing or not passed",
            "stage4e2_readonly_preflight": "stage4e2_readonly_preflight missing or not passed",
            "stage4e3_fake_execution_client": (
                "stage4e3_fake_execution_client missing or not passed"
            ),
            "stage4e4_ticket_gate": "stage4e4_ticket_gate missing or not passed",
            "stage4e5_manual_submit_gate": (
                "stage4e5_manual_submit_gate missing or not passed"
            ),
        }
        for phase, blocker in expected.items():
            with self.subTest(phase=phase):
                reports = passed_reports()
                reports.pop(phase)
                self.assert_not_ready_with(self.build(reports=reports), blocker)

    def test_any_module_check_false_blocks_readiness(self) -> None:
        for key in MODULE_CHECK_KEYS:
            with self.subTest(key=key):
                self.assert_not_ready_with(
                    self.build(module_checks=passed_module_checks(**{key: False})),
                    f"module check failed: {key}",
                )

    def test_any_safety_check_false_blocks_readiness(self) -> None:
        for key in SAFETY_CHECK_KEYS:
            with self.subTest(key=key):
                self.assert_not_ready_with(
                    self.build(safety_checks=passed_safety_checks(**{key: False})),
                    f"safety check failed: {key}",
                )

    def test_unresolved_needs_reconciliation_blocks_readiness(self) -> None:
        report = self.build(
            state_snapshot=safe_state(unresolved_needs_reconciliation_count=2)
        )

        self.assert_not_ready_with(
            report,
            "unresolved NEEDS_RECONCILIATION records exist",
        )

    def test_needs_reconciliation_can_be_counted_from_positions(self) -> None:
        state = safe_state()
        state.pop("unresolved_needs_reconciliation_count")
        state["positions"] = {"pos_1": {"status": "NEEDS_RECONCILIATION"}}

        report = self.build(state_snapshot=state)

        self.assertEqual(report["state_safety"]["unresolved_needs_reconciliation_count"], 1)
        self.assert_not_ready_with(
            report,
            "unresolved NEEDS_RECONCILIATION records exist",
        )

    def test_active_halt_blocks_readiness(self) -> None:
        report = self.build(state_snapshot=safe_state(active_halt=True))

        self.assert_not_ready_with(report, "active halt is present")

    def test_explicit_unknown_broker_exposure_blocks_readiness(self) -> None:
        examples = (
            {"unknown_broker_exposure_count": 1},
            {"unknown_broker_exposures": [{"symbol": "XSP"}]},
            {"broker_exposure_unknown_count": 2},
            {"unreconciled_broker_positions_count": 3},
            {"broker_open_positions_count": 2, "internal_open_positions_count": 1},
        )
        for extra in examples:
            with self.subTest(extra=extra):
                state = safe_state()
                state.pop("unknown_broker_exposure_count")
                state.update(extra)
                self.assert_not_ready_with(
                    self.build(state_snapshot=state),
                    "unknown broker exposure exists",
                )

    def test_missing_unknown_broker_exposure_tracking_warns_without_inventing_failure(self) -> None:
        state = safe_state(open_positions_count=4)
        state.pop("unknown_broker_exposure_count")

        report = self.build(state_snapshot=state)

        self.assertTrue(
            report["readiness_for_stage4f"][
                "ready_to_begin_real_ibkr_paper_submit_planning"
            ]
        )
        self.assertIn(
            "unknown broker exposure not tracked in state_snapshot",
            report["readiness_for_stage4f"]["warnings"],
        )
        self.assertNotIn("unknown_broker_exposure_count", report["state_safety"])

    def test_open_positions_count_alone_is_not_unknown_broker_exposure(self) -> None:
        state = safe_state(open_positions_count=9)
        state.pop("unknown_broker_exposure_count")

        report = self.build(state_snapshot=state)

        blockers = report["readiness_for_stage4f"]["blockers"]
        self.assertNotIn("unknown broker exposure exists", blockers)

    def test_missing_unverifiable_inputs_are_blockers_not_crashes(self) -> None:
        report = self.build(
            reports={},
            module_checks={},
            safety_checks={},
            state_snapshot=None,
        )

        self.assertFalse(
            report["readiness_for_stage4f"][
                "ready_to_begin_real_ibkr_paper_submit_planning"
            ]
        )
        self.assertGreater(len(report["readiness_for_stage4f"]["blockers"]), 1)

    def test_recommendations_are_deterministic_and_non_binding(self) -> None:
        first = self.build()
        second = self.build()

        self.assertEqual(first["recommendations"], second["recommendations"])
        joined = "\n".join(
            first["recommendations"]["ordered_next_steps"]
            + first["recommendations"]["do_not_do_yet"]
        )
        self.assertIn("Do not enable automated paper execution yet.", joined)
        self.assertIn("Do not begin live trading.", joined)
        for item in first["recommendations"]["do_not_do_yet"]:
            self.assertTrue(item.startswith("Do not "))

    def test_safety_flags_keep_execution_disabled(self) -> None:
        report = self.build()

        self.assertEqual(
            report["safety"],
            {
                "real_ibkr_enabled": False,
                "paper_order_submission_enabled": False,
                "live_orders_enabled": False,
                "market_data_enabled": False,
                "contract_qualification_enabled": False,
                "scheduler_changes_enabled": False,
                "lifecycle_wiring_enabled": False,
            },
        )


class Stage4EAcceptanceReportCliTests(unittest.TestCase):
    def run_cli(
        self,
        argv: list[str],
        *,
        report_builder: object | None = None,
    ) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        kwargs = {}
        if report_builder is not None:
            kwargs["report_builder"] = report_builder
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = report_tool.run_stage4e_acceptance_report(argv, **kwargs)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_requires_dry_run_only_before_local_checks(self) -> None:
        def exploding_builder(**_kwargs: object) -> dict[str, object]:
            raise AssertionError("builder must not run before dry-run gate")

        code, stdout, stderr = self.run_cli(["--json"], report_builder=exploding_builder)

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("--dry-run-only", stderr)

    def test_cli_json_stdout_is_strict_json_only(self) -> None:
        code, stdout, stderr = self.run_cli(["--dry-run-only", "--json"])

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertTrue(payload["stage4e_acceptance_report"])

    def test_cli_boolean_flags_are_store_true(self) -> None:
        source = inspect.getsource(report_tool.run_stage4e_acceptance_report)

        self.assertIn('parser.add_argument("--dry-run-only", action="store_true")', source)
        self.assertIn('parser.add_argument("--json", action="store_true"', source)

    def test_cli_exposes_no_submit_cancel_market_data_or_qualification_actions(self) -> None:
        source = inspect.getsource(report_tool.run_stage4e_acceptance_report)
        for token in ("--submit", "--cancel", "--market-data", "--qualify"):
            with self.subTest(token=token):
                self.assertNotIn(token, source)


class Stage4EAcceptanceReportSafetyBoundaryTests(unittest.TestCase):
    def test_stage4e6_files_do_not_import_or_call_blocked_integrations(self) -> None:
        blocked_tokens = (
            "ib_" + "insync",
            "place" + "Order",
            "cancel" + "Order",
            "submit_" + "order_plan",
            "req" + "MktData",
            "qualify" + "Contracts",
            "y" + "finance",
            "requ" + "ests",
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
        for path in STAGE4E6_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_stage4e6_files_do_not_wire_clients_scheduler_or_lifecycle(self) -> None:
        forbidden_tokens = (
            "IbkrPaper" + "ExecutionClient",
            "IbkrPaper" + "ReadOnlyClient",
            "PaperBroker" + "Adapter",
            "core." + "scheduler",
            "jobs." + "submission",
            "jobs." + "confirmation",
            "jobs." + "fill_confirmation",
        )
        for path in STAGE4E6_FILES:
            source = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_submit_path_is_not_added_to_runtime_wiring(self) -> None:
        token = "submit_" + "order_plan"
        for path in UNWIRED_RUNTIME_FILES:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
