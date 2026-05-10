from __future__ import annotations

import inspect
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from algo_trader_unified.core.stage4f_acceptance_report import (
    MODULE_CHECK_KEYS,
    PHASE_KEYS,
    SAFETY_CHECK_KEYS,
    build_stage4f_acceptance_report,
)
from algo_trader_unified.tools import stage4f_acceptance_report as report_tool


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
STAGE4F6_FILES = [
    ROOT / "core/stage4f_acceptance_report.py",
    ROOT / "tools/stage4f_acceptance_report.py",
    ROOT / "tests/test_stage4f6_acceptance_report.py",
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


def accepted_smoke_report(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4f5_smoke_test_report": True,
        "smoke_test": {
            "accepted": True,
            "one_order_only": True,
            "broker_order_id": "9001",
            "client_order_id": "intent-stage4f6-001",
            "submitted": True,
            "status_seen": True,
            "cancel_seen": False,
            "terminal_or_safe_state_seen": True,
        },
        "safety_checks": {
            "no_live_orders": True,
            "no_market_data": True,
            "no_contract_qualification": True,
            "no_scheduler_changes": True,
            "no_lifecycle_wiring": True,
            "no_extra_submissions_detected": True,
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


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


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class Stage4F6AcceptanceReportCoreTests(unittest.TestCase):
    def build(self, **kwargs: object) -> dict:
        defaults: dict[str, object] = {
            "reports": passed_reports(),
            "module_checks": passed_module_checks(),
            "safety_checks": passed_safety_checks(),
            "smoke_test_report": accepted_smoke_report(),
            "state_snapshot": safe_state(),
            "now_provider": lambda: datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc),
        }
        defaults.update(kwargs)
        return build_stage4f_acceptance_report(**defaults)

    def assert_not_ready_with(self, report: dict, expected: str) -> None:
        readiness = report["readiness_for_stage4g"]
        self.assertFalse(readiness["ready_to_begin_manual_paper_lifecycle_validation"])
        self.assertIn(expected, readiness["blockers"])

    def test_none_inputs_do_not_raise_and_report_missing_inputs(self) -> None:
        report = build_stage4f_acceptance_report(
            reports=None,
            module_checks=None,
            safety_checks=None,
            smoke_test_report=None,
            state_snapshot=None,
            now_provider=lambda: datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc),
        )

        self.assertTrue(report["stage4f_acceptance_report"])
        self.assertFalse(
            report["readiness_for_stage4g"][
                "ready_to_begin_manual_paper_lifecycle_validation"
            ]
        )
        self.assertIn(
            "smoke_test_report missing",
            report["readiness_for_stage4g"]["blockers"],
        )
        json.dumps(report, sort_keys=True)

    def test_malformed_smoke_test_report_missing_smoke_dict_blocks_without_crash(self) -> None:
        report = self.build(smoke_test_report={"stage4f5_smoke_test_report": True})

        self.assert_not_ready_with(
            report,
            "smoke_test_report.smoke_test missing or malformed",
        )
        self.assertIn(
            "smoke_test_report.smoke_test missing or malformed",
            report["errors"],
        )

    def test_malformed_smoke_test_report_missing_nested_keys_blocks_without_crash(self) -> None:
        report = self.build(
            smoke_test_report={
                "stage4f5_smoke_test_report": True,
                "smoke_test": {"accepted": True, "submitted": True},
                "safety_checks": accepted_smoke_report()["safety_checks"],
            }
        )

        self.assert_not_ready_with(report, "smoke_test.one_order_only must be True")
        self.assert_not_ready_with(report, "smoke_test.status_seen must be True")

    def test_all_phase_statuses_present_and_accepted_smoke_test_are_ready(self) -> None:
        report = self.build()

        self.assertTrue(
            report["readiness_for_stage4g"][
                "ready_to_begin_manual_paper_lifecycle_validation"
            ]
        )
        self.assertEqual(report["readiness_for_stage4g"]["blockers"], [])
        self.assertTrue(report["safety"]["real_ibkr_paper_manual_execution_proven"])

    def test_required_top_level_fields_are_present(self) -> None:
        report = self.build()

        for key in (
            "dry_run",
            "stage4f_acceptance_report",
            "generated_at",
            "phase_status",
            "module_checks",
            "safety_checks",
            "smoke_test",
            "state_safety",
            "readiness_for_stage4g",
            "recommendations",
            "safety",
            "success",
            "errors",
            "warnings",
        ):
            with self.subTest(key=key):
                self.assertIn(key, report)

    def test_each_missing_phase_blocks_readiness(self) -> None:
        for phase in PHASE_KEYS:
            with self.subTest(phase=phase):
                reports = passed_reports()
                reports.pop(phase)
                self.assert_not_ready_with(
                    self.build(reports=reports),
                    f"{phase} missing or not passed",
                )

    def test_missing_smoke_test_report_blocks_readiness(self) -> None:
        self.assert_not_ready_with(
            self.build(smoke_test_report=None),
            "smoke_test_report missing",
        )

    def test_smoke_test_boolean_gates_block_readiness(self) -> None:
        cases = {
            "accepted": "smoke_test.accepted must be True",
            "one_order_only": "smoke_test.one_order_only must be True",
            "submitted": "smoke_test.submitted must be True",
            "status_seen": "smoke_test.status_seen must be True",
            "terminal_or_safe_state_seen": (
                "smoke_test.terminal_or_safe_state_seen must be True"
            ),
        }
        for key, expected in cases.items():
            with self.subTest(key=key):
                self.assert_not_ready_with(
                    self.build(smoke_test_report=accepted_smoke_report(smoke_test={key: False})),
                    expected,
                )

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
        self.assert_not_ready_with(
            self.build(state_snapshot=safe_state(unresolved_needs_reconciliation_count=1)),
            "unresolved NEEDS_RECONCILIATION records exist",
        )

    def test_active_halt_blocks_readiness(self) -> None:
        self.assert_not_ready_with(
            self.build(state_snapshot=safe_state(active_halt=True)),
            "active halt is present",
        )

    def test_explicit_unknown_broker_exposure_blocks_readiness(self) -> None:
        examples = (
            {"unknown_broker_exposure_count": 1},
            {"unknown_broker_exposures": [{"symbol": "XSP"}]},
            {"broker_exposure_unknown_count": 2},
            {"unreconciled_broker_positions_count": 3},
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
            report["readiness_for_stage4g"][
                "ready_to_begin_manual_paper_lifecycle_validation"
            ]
        )
        self.assertIn(
            "unknown broker exposure not tracked in state_snapshot",
            report["readiness_for_stage4g"]["warnings"],
        )
        self.assertNotIn("unknown_broker_exposure_count", report["state_safety"])

    def test_open_positions_count_alone_is_not_unknown_broker_exposure(self) -> None:
        state = safe_state(open_positions_count=9)
        state.pop("unknown_broker_exposure_count")

        report = self.build(state_snapshot=state)

        self.assertNotIn(
            "unknown broker exposure exists",
            report["readiness_for_stage4g"]["blockers"],
        )

    def test_smoke_report_unsafe_flags_block_readiness(self) -> None:
        unsafe_cases = (
            ("no_live_orders", False, "smoke_test_report safety check failed: no_live_orders"),
            ("no_market_data", False, "smoke_test_report safety check failed: no_market_data"),
            (
                "no_contract_qualification",
                False,
                "smoke_test_report safety check failed: no_contract_qualification",
            ),
            (
                "no_scheduler_changes",
                False,
                "smoke_test_report safety check failed: no_scheduler_changes",
            ),
            (
                "no_lifecycle_wiring",
                False,
                "smoke_test_report safety check failed: no_lifecycle_wiring",
            ),
            (
                "live_orders_enabled",
                True,
                "smoke_test_report unsafe flag enabled: live_orders_enabled",
            ),
            (
                "market_data_enabled",
                True,
                "smoke_test_report unsafe flag enabled: market_data_enabled",
            ),
            (
                "contract_qualification_enabled",
                True,
                "smoke_test_report unsafe flag enabled: contract_qualification_enabled",
            ),
            (
                "scheduler_changes_enabled",
                True,
                "smoke_test_report unsafe flag enabled: scheduler_changes_enabled",
            ),
            (
                "lifecycle_wiring_enabled",
                True,
                "smoke_test_report unsafe flag enabled: lifecycle_wiring_enabled",
            ),
        )
        for key, value, expected in unsafe_cases:
            with self.subTest(key=key):
                self.assert_not_ready_with(
                    self.build(smoke_test_report=accepted_smoke_report(safety_checks={key: value})),
                    expected,
                )

    def test_recommendations_are_deterministic_and_non_binding(self) -> None:
        first = self.build()
        second = self.build()

        self.assertEqual(first["recommendations"], second["recommendations"])
        joined = "\n".join(
            first["recommendations"]["ordered_next_steps"]
            + first["recommendations"]["do_not_do_yet"]
        )
        self.assertIn(
            "Begin Stage 4G manual paper lifecycle validation behind explicit operator gates.",
            joined,
        )
        self.assertIn("Do not enable automated paper trading yet.", joined)
        self.assertIn("Do not begin live trading.", joined)

    def test_report_is_json_safe(self) -> None:
        report = self.build(
            smoke_test_report=accepted_smoke_report(smoke_test={"broker_order_id": object()})
        )

        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("object at 0x", serialized)


class Stage4F6AcceptanceReportCliTests(unittest.TestCase):
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
            code = report_tool.run_stage4f_acceptance_report(argv, **kwargs)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_requires_dry_run_only_before_json_loading(self) -> None:
        def exploding_builder(**_kwargs: object) -> dict[str, object]:
            raise AssertionError("builder must not run before dry-run gate")

        code, stdout, stderr = self.run_cli(
            ["--json", "--smoke-test-json", "{"],
            report_builder=exploding_builder,
        )

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("--dry-run-only", stderr)

    def test_cli_json_stdout_is_strict_json_only(self) -> None:
        code, stdout, stderr = self.run_cli(
            [
                "--dry-run-only",
                "--json",
                "--smoke-test-json",
                json.dumps(accepted_smoke_report()),
            ]
        )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertTrue(payload["stage4f_acceptance_report"])

    def test_cli_json_parse_error_reports_exception_type(self) -> None:
        code, stdout, stderr = self.run_cli(
            ["--dry-run-only", "--json", "--smoke-test-json", "{"]
        )

        self.assertEqual(code, 1)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertFalse(payload["success"])
        self.assertIn("JSONDecodeError", payload["errors"][0])

    def test_cli_boolean_flags_are_store_true(self) -> None:
        source = inspect.getsource(report_tool.run_stage4f_acceptance_report)

        self.assertIn('parser.add_argument("--dry-run-only", action="store_true")', source)
        self.assertIn('parser.add_argument("--json", action="store_true"', source)

    def test_cli_exposes_no_submit_cancel_status_market_data_or_qualification_actions(self) -> None:
        source = inspect.getsource(report_tool.run_stage4f_acceptance_report)
        for token in (
            "--submit",
            "--cancel",
            "--status",
            "--market-data",
            "--qualify",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, source)


class Stage4F6AcceptanceReportSafetyBoundaryTests(unittest.TestCase):
    def test_stage4f6_files_do_not_import_or_call_blocked_integrations(self) -> None:
        blocked_tokens = (
            "ib_" + "insync",
            "place" + "Order",
            "cancel" + "Order",
            "submit_" + "order_plan",
            "get_" + "order_status",
            "cancel_" + "order",
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
        for path in STAGE4F6_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_stage4f6_files_do_not_wire_clients_scheduler_or_lifecycle(self) -> None:
        forbidden_tokens = (
            "IbkrPaper" + "ExecutionClient",
            "IbkrPaper" + "ReadOnlyClient",
            "PaperBroker" + "Adapter",
            "core." + "scheduler",
            "jobs." + "submission",
            "jobs." + "confirmation",
            "jobs." + "fill_confirmation",
        )
        for path in STAGE4F6_FILES:
            source = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_stage4f6_added_no_scheduler_lifecycle_wiring(self) -> None:
        for token in (
            "stage4f_acceptance_report",
            "build_stage4f_acceptance_report",
        ):
            for path in UNWIRED_RUNTIME_FILES:
                source = path.read_text(encoding="utf-8")
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
