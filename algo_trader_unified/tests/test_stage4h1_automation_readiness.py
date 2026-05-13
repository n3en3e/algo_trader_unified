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

from algo_trader_unified.core.stage4h1_automation_readiness import (
    build_stage4h1_automation_readiness_report,
)
from algo_trader_unified.tools import stage4h1_automation_readiness as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4H1_FILES = [
    ROOT / "core/stage4h1_automation_readiness.py",
    ROOT / "tools/stage4h1_automation_readiness.py",
]
UNWIRED_RUNTIME_FILES = [
    ROOT / "tools/daemon.py",
    ROOT / "core/scheduler.py",
    ROOT / "core/scheduler_cadence.py",
    ROOT / "core/job_chain.py",
    ROOT / "jobs/submission.py",
    ROOT / "jobs/confirmation.py",
    ROOT / "jobs/fill_confirmation.py",
    ROOT / "jobs/position_transitions.py",
]


def valid_stage4g_report(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4g6_lifecycle_write_acceptance_report": True,
        "generated_at": "2026-05-12T12:00:00+00:00",
        "artifact_checks": {
            "executor_report_present": True,
            "executor_report_ready": True,
            "executor_completed": True,
            "state_store_write_succeeded": True,
            "ledger_write_succeeded": True,
            "rollback_not_required": True,
        },
        "safety_checks": {
            "no_live_orders": True,
            "no_market_data": True,
            "no_contract_qualification": True,
            "no_scheduler_changes": True,
            "no_lifecycle_wiring": True,
            "no_automated_paper_trading": True,
        },
        "readiness_for_stage4h": {
            "ready_to_begin_controlled_automated_paper_trading_launch": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def clean_risk_snapshot(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "kill_switch_available": True,
        "soft_halt_available": True,
        "hard_halt_available": True,
        "daily_loss_limit_available": True,
        "max_position_limit_available": True,
        "risk_controls_clean": True,
        "risk_bypass_enabled": False,
    }
    _deep_update(report, overrides)
    return report


def build(
    *,
    stage4g_report: dict | None = None,
    module_checks: dict | None = None,
    safety_checks: dict | None = None,
    state_snapshot: dict | None = None,
    strategy_registry_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
) -> dict:
    return build_stage4h1_automation_readiness_report(
        stage4g_acceptance_report=valid_stage4g_report()
        if stage4g_report is None
        else stage4g_report,
        module_checks=module_checks,
        safety_checks=safety_checks,
        state_snapshot=state_snapshot,
        strategy_registry_snapshot=strategy_registry_snapshot,
        risk_snapshot=risk_snapshot,
        now_provider=lambda: datetime(2026, 5, 12, 14, 0, tzinfo=timezone.utc),
    )


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


def assert_not_ready(test_case: unittest.TestCase, report: dict) -> None:
    test_case.assertFalse(
        report["readiness_for_stage4h2"]["ready_to_build_automation_wiring_preview"]
    )
    test_case.assertFalse(report["success"])


def assert_json_safe(test_case: unittest.TestCase, report: dict) -> None:
    serialized = json.dumps(report, sort_keys=True)
    test_case.assertNotIn("Decimal", serialized)
    test_case.assertNotIn("datetime", serialized)


class Stage4H1AutomationReadinessTests(unittest.TestCase):
    def test_missing_stage4g_acceptance_report_blocks_readiness(self) -> None:
        report = build_stage4h1_automation_readiness_report(
            stage4g_acceptance_report=None
        )
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["stage4g_acceptance_present"])

    def test_stage4g_acceptance_not_ready_blocks_readiness(self) -> None:
        report = build(
            stage4g_report=valid_stage4g_report(
                readiness_for_stage4h={
                    "ready_to_begin_controlled_automated_paper_trading_launch": False
                }
            )
        )
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["stage4g_acceptance_ready"])

    def test_valid_4g_acceptance_without_optional_snapshots_is_ready_with_warnings(self) -> None:
        report = build()
        self.assertTrue(
            report["readiness_for_stage4h2"]["ready_to_build_automation_wiring_preview"]
        )
        self.assertIn("state snapshot missing", " ".join(report["warnings"]))
        self.assertIn("risk snapshot missing", " ".join(report["warnings"]))
        self.assertIn("strategy registry snapshot missing", " ".join(report["warnings"]))

    def test_unsafe_flags_block_readiness(self) -> None:
        cases = [
            ("live_trading_enabled", "no_live_orders"),
            ("automated_paper_trading_enabled", "no_automated_paper_trading_enabled"),
            ("scheduler_wiring_enabled", "no_scheduler_changes"),
            ("lifecycle_wiring_enabled", "no_lifecycle_wiring"),
            ("broker_submission_enabled", "no_broker_submission_enabled"),
            ("market_data_enabled", "no_market_data"),
            ("contract_qualification_enabled", "no_contract_qualification"),
            ("all_strategies_enabled", "no_all_strategy_enablement"),
        ]
        for flag, check in cases:
            with self.subTest(flag=flag):
                report = build(safety_checks={flag: True})
                assert_not_ready(self, report)
                self.assertFalse(report["safety_checks"][check])

    def test_active_halt_in_state_snapshot_blocks_readiness(self) -> None:
        report = build(state_snapshot={"active_halt": True})
        assert_not_ready(self, report)
        self.assertTrue(report["state_checks"]["active_halt"])

    def test_unresolved_needs_reconciliation_in_state_snapshot_blocks_readiness(self) -> None:
        report = build(state_snapshot={"unresolved_needs_reconciliation_count": 1})
        assert_not_ready(self, report)
        self.assertEqual(1, report["state_checks"]["unresolved_needs_reconciliation_count"])

    def test_unresolved_needs_reconciliation_status_is_counted(self) -> None:
        report = build(state_snapshot={"orders": [{"status": "NEEDS_RECONCILIATION"}]})
        assert_not_ready(self, report)
        self.assertEqual(1, report["state_checks"]["unresolved_needs_reconciliation_count"])

    def test_risk_bypass_enabled_blocks_readiness(self) -> None:
        report = build(risk_snapshot=clean_risk_snapshot(risk_bypass_enabled=True))
        assert_not_ready(self, report)
        self.assertFalse(report["risk_checks"]["risk_controls_clean"])

    def test_provided_risk_snapshot_missing_required_controls_blocks_readiness(self) -> None:
        for key in (
            "kill_switch_available",
            "hard_halt_available",
            "daily_loss_limit_available",
        ):
            with self.subTest(key=key):
                risk = clean_risk_snapshot()
                del risk[key]
                report = build(risk_snapshot=risk)
                assert_not_ready(self, report)
                self.assertFalse(report["risk_checks"][key])

    def test_provided_risk_snapshot_required_control_false_blocks_readiness(self) -> None:
        for key in (
            "kill_switch_available",
            "hard_halt_available",
            "daily_loss_limit_available",
        ):
            with self.subTest(key=key):
                report = build(risk_snapshot=clean_risk_snapshot(**{key: False}))
                assert_not_ready(self, report)
                self.assertFalse(report["risk_checks"][key])

    def test_clean_provided_risk_snapshot_passes_core_risk_gate(self) -> None:
        report = build(risk_snapshot=clean_risk_snapshot())
        self.assertTrue(report["risk_checks"]["kill_switch_available"])
        self.assertTrue(report["risk_checks"]["hard_halt_available"])
        self.assertTrue(report["risk_checks"]["daily_loss_limit_available"])
        self.assertTrue(report["readiness_for_stage4h2"]["ready_to_build_automation_wiring_preview"])

    def test_missing_optional_snapshots_warn_without_crashing(self) -> None:
        report = build()
        self.assertTrue(report["success"])
        self.assertFalse(report["state_checks"]["state_snapshot_present"])
        self.assertFalse(report["risk_checks"]["risk_snapshot_present"])
        self.assertFalse(
            report["strategy_candidate_checks"]["strategy_registry_snapshot_present"]
        )

    def test_strategy_ids_shape_records_sorted_candidates(self) -> None:
        report = build(
            strategy_registry_snapshot={
                "paper_eligible_strategy_ids": ["S03", "S01", "S02"]
            }
        )
        self.assertEqual(
            ["S01", "S02", "S03"],
            report["strategy_candidate_checks"]["candidate_strategy_ids"],
        )
        self.assertTrue(
            report["strategy_candidate_checks"]["single_strategy_launch_required"]
        )

    def test_strategies_dict_shape_uses_safe_access_and_filters_candidates(self) -> None:
        report = build(
            strategy_registry_snapshot={
                "strategies": [
                    {"strategy_id": "S02", "paper_eligible": True, "enabled": False},
                    {"strategy_id": "S01", "paper_eligible": True},
                    {"strategy_id": "LIVE", "paper_eligible": True, "live_only": True},
                    {"strategy_id": "AUTO", "paper_eligible": True, "automated_enabled": True},
                    {"paper_eligible": True},
                    None,
                    "bad",
                ]
            }
        )
        self.assertEqual(
            ["S01", "S02"],
            report["strategy_candidate_checks"]["candidate_strategy_ids"],
        )
        self.assertEqual(2, report["strategy_candidate_checks"]["paper_eligible_strategy_count"])

    def test_candidates_dict_shape_records_candidate(self) -> None:
        report = build(
            strategy_registry_snapshot={
                "candidates": [{"strategy_id": "S01_VOL_BASELINE", "paper_eligible": True}]
            }
        )
        self.assertEqual(
            ["S01_VOL_BASELINE"],
            report["strategy_candidate_checks"]["candidate_strategy_ids"],
        )

    def test_malformed_strategy_entries_do_not_crash_or_count(self) -> None:
        report = build(strategy_registry_snapshot={"candidates": [None, "bad", [], {}]})
        assert_not_ready(self, report)
        self.assertEqual([], report["strategy_candidate_checks"]["candidate_strategy_ids"])

    def test_strategy_registry_list_of_strings_vs_dicts_never_type_errors(self) -> None:
        report = build(
            strategy_registry_snapshot={
                "paper_eligible_strategy_ids": ["S02"],
                "strategies": ["bad", {"strategy_id": "S01", "paper_eligible": True}],
            }
        )
        self.assertEqual(
            ["S01", "S02"],
            report["strategy_candidate_checks"]["candidate_strategy_ids"],
        )

    def test_recommendations_do_not_include_live_or_all_strategy_enablement(self) -> None:
        report = build()
        ordered = " ".join(report["recommendations"]["ordered_next_steps"]).lower()
        self.assertNotIn("live trading", ordered)
        self.assertNotIn("all strategies", ordered)
        self.assertIn("Do not enable live trading.", report["recommendations"]["do_not_do_yet"])
        self.assertIn(
            "Do not enable all strategies at once.",
            report["recommendations"]["do_not_do_yet"],
        )

    def test_report_includes_required_top_level_fields_and_is_json_safe(self) -> None:
        report = build(risk_snapshot={"kill_switch_available": Decimal("1")})
        for key in (
            "dry_run",
            "stage4h1_automation_readiness_report",
            "generated_at",
            "artifact_checks",
            "module_checks",
            "state_checks",
            "strategy_candidate_checks",
            "risk_checks",
            "safety_checks",
            "readiness_for_stage4h2",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        assert_json_safe(self, report)

    def test_input_reports_and_snapshots_are_not_mutated(self) -> None:
        stage4g = valid_stage4g_report()
        modules = {"automation_wiring_present": True}
        safety = {"live_trading_enabled": False}
        state = {"open_positions_count": 1}
        registry = {"paper_eligible_strategy_ids": ["S01"]}
        risk = clean_risk_snapshot()
        before = copy.deepcopy((stage4g, modules, safety, state, registry, risk))
        build(
            stage4g_report=stage4g,
            module_checks=modules,
            safety_checks=safety,
            state_snapshot=state,
            strategy_registry_snapshot=registry,
            risk_snapshot=risk,
        )
        self.assertEqual(before, (stage4g, modules, safety, state, registry, risk))

    def test_cli_requires_dry_run_only_before_parsing(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_stage4h1_automation_readiness([])
        self.assertEqual(1, code)
        self.assertIn("--dry-run-only", stderr.getvalue())

    def test_cli_json_stdout_is_strict_json(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4h1_automation_readiness(
                [
                    "--dry-run-only",
                    "--json",
                    "--stage4g-acceptance-json",
                    json.dumps(valid_stage4g_report()),
                    "--strategy-registry-json",
                    json.dumps({"paper_eligible_strategy_ids": ["S01"]}),
                    "--risk-snapshot-json",
                    json.dumps(clean_risk_snapshot()),
                ]
            )
        self.assertEqual(0, code)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4h1_automation_readiness_report"])

    def test_cli_json_parse_errors_include_exception_type(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4h1_automation_readiness(
                [
                    "--dry-run-only",
                    "--json",
                    "--stage4g-acceptance-json",
                    "{bad",
                ]
            )
        self.assertEqual(1, code)
        parsed = json.loads(stdout.getvalue())
        self.assertIn("JSONDecodeError", parsed["errors"][0])

    def test_cli_exposes_no_forbidden_actions(self) -> None:
        source = inspect.getsource(tool.run_stage4h1_automation_readiness)
        for forbidden in (
            "--submit",
            "--cancel",
            "--status",
            "--market-data",
            "--qualification",
            "--allow-state-write",
            "--state-write-action",
            "--allow-ledger-write",
            "--ledger-write-action",
            "--scheduler",
            "--lifecycle-action",
        ):
            self.assertNotIn(forbidden, source)

    def test_stage4h1_files_do_not_call_forbidden_external_apis_or_mutators(self) -> None:
        forbidden = (
            "submit_order_plan(",
            "get_order_status(",
            "cancel_order(",
            "placeOrder",
            "cancelOrder",
            "reqMktData",
            "qualifyContracts",
            "StateStore(",
            ".save(",
            ".write(",
            ".update(",
            ".append_event(",
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
            "time.time",
            "datetime.now",
            ".jsonl",
            "open(",
            "ib_insync",
        )
        for path in STAGE4H1_FILES:
            source = path.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, source, f"{value} unexpectedly found in {path}")

    def test_no_daemon_scheduler_or_lifecycle_wiring_exists(self) -> None:
        for path in UNWIRED_RUNTIME_FILES:
            if path.exists():
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("stage4h1_automation_readiness", source)


if __name__ == "__main__":
    unittest.main()
