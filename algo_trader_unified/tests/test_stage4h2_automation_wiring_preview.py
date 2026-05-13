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

from algo_trader_unified.core.stage4h2_automation_wiring_preview import (
    build_stage4h2_automation_wiring_preview_report,
)
from algo_trader_unified.tools import stage4h2_automation_wiring_preview as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4H2_FILES = [
    ROOT / "core/stage4h2_automation_wiring_preview.py",
    ROOT / "tools/stage4h2_automation_wiring_preview.py",
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


def valid_stage4h1_report(candidates: list[str] | None = None, **overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4h1_automation_readiness_report": True,
        "generated_at": "2026-05-12T14:00:00+00:00",
        "artifact_checks": {
            "stage4g_acceptance_present": True,
            "stage4g_acceptance_ready": True,
        },
        "strategy_candidate_checks": {
            "candidate_strategy_ids": ["S01"] if candidates is None else candidates,
        },
        "readiness_for_stage4h2": {
            "ready_to_build_automation_wiring_preview": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def clean_risk_snapshot(**overrides: object) -> dict:
    report: dict[str, object] = {
        "kill_switch_available": True,
        "hard_halt_available": True,
        "daily_loss_limit_available": True,
        "risk_bypass_enabled": False,
    }
    _deep_update(report, overrides)
    return report


def build(
    *,
    stage4h1_report: dict | None = None,
    strategy_registry_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    explicit_preview_strategy_id: str | None = None,
) -> dict:
    return build_stage4h2_automation_wiring_preview_report(
        stage4h1_readiness_report=valid_stage4h1_report()
        if stage4h1_report is None
        else stage4h1_report,
        strategy_registry_snapshot=strategy_registry_snapshot,
        scheduler_snapshot=scheduler_snapshot,
        lifecycle_snapshot=lifecycle_snapshot,
        risk_snapshot=risk_snapshot,
        state_snapshot=state_snapshot,
        explicit_preview_strategy_id=explicit_preview_strategy_id,
        now_provider=lambda: datetime(2026, 5, 13, 14, 0, tzinfo=timezone.utc),
    )


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


def assert_not_ready(test_case: unittest.TestCase, report: dict) -> None:
    test_case.assertFalse(
        report["readiness_for_stage4h3"]["ready_to_build_automation_wiring_dry_run"]
    )
    test_case.assertFalse(report["success"])


def assert_json_safe(test_case: unittest.TestCase, report: dict) -> None:
    serialized = json.dumps(report, sort_keys=True)
    test_case.assertNotIn("Decimal", serialized)
    test_case.assertNotIn("datetime", serialized)


class Stage4H2AutomationWiringPreviewTests(unittest.TestCase):
    def test_missing_stage4h1_readiness_report_blocks_readiness(self) -> None:
        report = build_stage4h2_automation_wiring_preview_report(
            stage4h1_readiness_report=None
        )
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["stage4h1_readiness_present"])

    def test_stage4h1_readiness_not_ready_blocks_readiness(self) -> None:
        report = build(
            stage4h1_report=valid_stage4h1_report(
                readiness_for_stage4h2={
                    "ready_to_build_automation_wiring_preview": False
                }
            )
        )
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["stage4h1_readiness_ready"])

    def test_valid_stage4h1_plus_one_candidate_creates_preview_and_readiness(self) -> None:
        report = build(risk_snapshot=clean_risk_snapshot())
        self.assertTrue(
            report["readiness_for_stage4h3"]["ready_to_build_automation_wiring_dry_run"]
        )
        self.assertEqual("S01", report["strategy_selection"]["selected_preview_strategy_id"])
        self.assertTrue(report["wiring_preview"]["available"])

    def test_zero_candidates_blocks_readiness(self) -> None:
        report = build(stage4h1_report=valid_stage4h1_report(candidates=[]))
        assert_not_ready(self, report)
        self.assertEqual([], report["strategy_selection"]["candidate_strategy_ids"])

    def test_multiple_candidates_without_explicit_selection_blocks_readiness(self) -> None:
        report = build(stage4h1_report=valid_stage4h1_report(candidates=["S02", "S01"]))
        assert_not_ready(self, report)
        self.assertIn("explicitly selected", " ".join(report["readiness_for_stage4h3"]["blockers"]))

    def test_multiple_candidates_with_eligible_explicit_selection_passes(self) -> None:
        report = build(
            stage4h1_report=valid_stage4h1_report(candidates=["S02", "S01"]),
            explicit_preview_strategy_id="S02",
            risk_snapshot=clean_risk_snapshot(),
        )
        self.assertTrue(report["success"])
        self.assertEqual("S02", report["strategy_selection"]["selected_preview_strategy_id"])

    def test_explicit_preview_strategy_id_not_in_candidate_list_blocks_readiness(self) -> None:
        report = build(
            stage4h1_report=valid_stage4h1_report(candidates=["S01"]),
            explicit_preview_strategy_id="S02",
        )
        assert_not_ready(self, report)
        self.assertIn("not eligible", " ".join(report["readiness_for_stage4h3"]["blockers"]))

    def test_snapshot_preview_strategy_id_is_used_without_top_level_explicit(self) -> None:
        report = build(
            stage4h1_report=valid_stage4h1_report(candidates=["S02", "S01"]),
            strategy_registry_snapshot={"preview_strategy_id": "S01"},
            risk_snapshot=clean_risk_snapshot(),
        )
        self.assertTrue(report["success"])
        self.assertEqual("S01", report["strategy_selection"]["selected_preview_strategy_id"])

    def test_snapshot_preview_strategy_id_not_eligible_blocks_readiness(self) -> None:
        report = build(
            stage4h1_report=valid_stage4h1_report(candidates=["S01"]),
            strategy_registry_snapshot={"preview_strategy_id": "S02"},
        )
        assert_not_ready(self, report)
        self.assertIn("not eligible", " ".join(report["readiness_for_stage4h3"]["blockers"]))

    def test_cli_passes_explicit_preview_strategy_id_without_mutating_registry(self) -> None:
        calls: list[dict] = []
        registry = {"paper_eligible_strategy_ids": ["S01", "S02"]}

        def fake_builder(**kwargs: object) -> dict:
            calls.append(kwargs)
            return {
                "readiness_for_stage4h3": {
                    "ready_to_build_automation_wiring_dry_run": True,
                    "blockers": [],
                    "warnings": [],
                },
                "success": True,
            }

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4h2_automation_wiring_preview(
                [
                    "--dry-run-only",
                    "--json",
                    "--stage4h1-readiness-json",
                    json.dumps(valid_stage4h1_report(candidates=["S01", "S02"])),
                    "--strategy-registry-json",
                    json.dumps(registry),
                    "--explicit-preview-strategy-id",
                    "S02",
                ],
                report_builder=fake_builder,
            )
        self.assertEqual(0, code)
        self.assertEqual("S02", calls[0]["explicit_preview_strategy_id"])
        self.assertNotIn("preview_strategy_id", calls[0]["strategy_registry_snapshot"])
        self.assertNotIn("preview_strategy_id", registry)

    def test_candidate_strategy_ids_sorted_deterministically(self) -> None:
        report = build(
            stage4h1_report=valid_stage4h1_report(candidates=["S03", "S01", "S02"]),
            explicit_preview_strategy_id="S02",
        )
        self.assertEqual(["S01", "S02", "S03"], report["strategy_selection"]["candidate_strategy_ids"])

    def test_strategy_parsing_supports_expected_registry_shapes(self) -> None:
        shapes = [
            {"strategies": [{"strategy_id": "S01", "paper_eligible": True}]},
            {"candidates": [{"strategy_id": "S01", "paper_eligible": True}]},
            {"paper_eligible_strategy_ids": ["S01"]},
        ]
        for snapshot in shapes:
            with self.subTest(snapshot=snapshot):
                report = build(stage4h1_report=valid_stage4h1_report(candidates=[]), strategy_registry_snapshot=snapshot)
                self.assertEqual(["S01"], report["strategy_selection"]["candidate_strategy_ids"])

    def test_malformed_strategy_entries_do_not_crash(self) -> None:
        report = build(
            stage4h1_report=valid_stage4h1_report(candidates=[]),
            strategy_registry_snapshot={"strategies": [None, "bad", [], {}]},
        )
        assert_not_ready(self, report)
        self.assertEqual([], report["strategy_selection"]["candidate_strategy_ids"])

    def test_live_only_and_already_enabled_strategies_are_excluded(self) -> None:
        report = build(
            stage4h1_report=valid_stage4h1_report(candidates=[]),
            strategy_registry_snapshot={
                "strategies": [
                    {"strategy_id": "LIVE", "paper_eligible": True, "live_only": True},
                    {"strategy_id": "AUTO", "paper_eligible": True, "automated_enabled": True},
                    {"strategy_id": "ENABLED", "paper_eligible": True, "enabled": True},
                    {"strategy_id": "S01", "paper_eligible": True, "enabled": False},
                ]
            },
            risk_snapshot=clean_risk_snapshot(),
        )
        self.assertTrue(report["success"])
        self.assertEqual(["S01"], report["strategy_selection"]["candidate_strategy_ids"])
        self.assertEqual("S01", report["strategy_selection"]["selected_preview_strategy_id"])

    def test_selected_preview_strategy_is_not_enabled(self) -> None:
        report = build(risk_snapshot=clean_risk_snapshot())
        self.assertEqual("S01", report["strategy_selection"]["selected_preview_strategy_id"])
        self.assertFalse(report["safety_checks"]["no_automated_paper_trading_enabled"] is False)

    def test_scheduler_and_lifecycle_previews_are_structured_not_flat_strings(self) -> None:
        report = build()
        scheduler = report["wiring_preview"]["proposed_scheduler_wiring_preview"]
        lifecycle = report["wiring_preview"]["proposed_lifecycle_wiring_preview"]
        self.assertIsInstance(scheduler, dict)
        self.assertIsInstance(scheduler["jobs"][0], dict)
        self.assertIsInstance(lifecycle, dict)
        self.assertIsInstance(lifecycle["flows"][0], dict)

    def test_proposed_scheduler_job_remains_disabled_and_non_executing(self) -> None:
        report = build()
        job = report["wiring_preview"]["proposed_scheduler_wiring_preview"]["jobs"][0]
        self.assertTrue(job["disabled"])
        self.assertFalse(job["would_register"])
        self.assertFalse(job["would_execute"])
        self.assertTrue(job["paper_only"])

    def test_proposed_lifecycle_preview_would_not_execute(self) -> None:
        report = build()
        flows = report["wiring_preview"]["proposed_lifecycle_wiring_preview"]["flows"]
        self.assertTrue(flows)
        self.assertTrue(all(flow["would_execute"] is False for flow in flows))

    def test_scheduler_snapshot_already_enabled_blocks_readiness(self) -> None:
        report = build(scheduler_snapshot={"scheduler_automation_enabled": True})
        assert_not_ready(self, report)
        self.assertTrue(report["scheduler_checks"]["scheduler_already_enabled"])

    def test_lifecycle_snapshot_already_enabled_blocks_readiness(self) -> None:
        report = build(lifecycle_snapshot={"lifecycle_automation_enabled": True})
        assert_not_ready(self, report)
        self.assertTrue(report["lifecycle_checks"]["lifecycle_already_enabled"])

    def test_active_selected_strategy_job_blocks_unless_disabled_dry_run_only(self) -> None:
        blocked = build(scheduler_snapshot={"jobs": [{"strategy_id": "S01", "disabled": False}]})
        assert_not_ready(self, blocked)
        allowed = build(
            scheduler_snapshot={
                "jobs": [{"strategy_id": "S01", "disabled": True, "dry_run_only": True}]
            },
            risk_snapshot=clean_risk_snapshot(),
        )
        self.assertTrue(allowed["success"])

    def test_risk_bypass_enabled_blocks_readiness(self) -> None:
        report = build(risk_snapshot=clean_risk_snapshot(risk_bypass_enabled=True))
        assert_not_ready(self, report)
        self.assertTrue(report["risk_checks"]["risk_bypass_enabled"])

    def test_provided_risk_snapshot_missing_required_control_blocks_readiness(self) -> None:
        for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
            with self.subTest(key=key):
                risk = clean_risk_snapshot()
                del risk[key]
                report = build(risk_snapshot=risk)
                assert_not_ready(self, report)
                self.assertFalse(report["risk_checks"][key])

    def test_missing_risk_snapshot_warns_without_crashing(self) -> None:
        report = build()
        self.assertTrue(report["success"])
        self.assertFalse(report["risk_checks"]["risk_snapshot_present"])
        self.assertIn("risk snapshot missing", " ".join(report["warnings"]))

    def test_active_halt_and_reconciliation_block_readiness(self) -> None:
        for snapshot in (
            {"active_halt": True},
            {"unresolved_needs_reconciliation_count": 1},
            {"needs_reconciliation_count": 1},
        ):
            with self.subTest(snapshot=snapshot):
                report = build(state_snapshot=snapshot)
                assert_not_ready(self, report)

    def test_active_intents_warn_and_open_positions_do_not_block_by_themselves(self) -> None:
        report = build(
            state_snapshot={"active_intents_count": 1, "open_positions_count": 3},
            risk_snapshot=clean_risk_snapshot(),
        )
        self.assertTrue(report["success"])
        self.assertEqual(3, report["state_checks"]["open_positions_count"])
        self.assertIn("active intents are present", " ".join(report["warnings"]))

    def test_missing_state_snapshot_warns_without_crashing(self) -> None:
        report = build()
        self.assertTrue(report["success"])
        self.assertFalse(report["state_checks"]["state_snapshot_present"])
        self.assertIn("state snapshot missing", " ".join(report["warnings"]))

    def test_unsafe_flags_block_readiness(self) -> None:
        cases = [
            ("live_trading_enabled", "no_live_orders"),
            ("automated_paper_trading_enabled", "no_automated_paper_trading_enabled"),
            ("broker_submission_enabled", "no_broker_submission_enabled"),
            ("all_strategies_enabled", "no_all_strategy_enablement"),
            ("market_data_enabled", "no_market_data"),
            ("contract_qualification_enabled", "no_contract_qualification"),
            ("daemon_wiring_enabled", "no_daemon_wiring_enabled"),
        ]
        for flag, check in cases:
            with self.subTest(flag=flag):
                report = build(stage4h1_report=valid_stage4h1_report(**{flag: True}))
                assert_not_ready(self, report)
                self.assertFalse(report["safety_checks"][check])

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
            "stage4h2_automation_wiring_preview_report",
            "generated_at",
            "artifact_checks",
            "strategy_selection",
            "wiring_preview",
            "safety_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "state_checks",
            "readiness_for_stage4h3",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        assert_json_safe(self, report)

    def test_input_reports_and_snapshots_are_not_mutated(self) -> None:
        stage4h1 = valid_stage4h1_report(candidates=["S01", "S02"])
        registry = {"paper_eligible_strategy_ids": ["S01", "S02"]}
        scheduler = {"jobs": []}
        lifecycle = {"flows": []}
        risk = clean_risk_snapshot()
        state = {"open_positions_count": 1}
        before = copy.deepcopy((stage4h1, registry, scheduler, lifecycle, risk, state))
        build_stage4h2_automation_wiring_preview_report(
            stage4h1_readiness_report=stage4h1,
            strategy_registry_snapshot=registry,
            scheduler_snapshot=scheduler,
            lifecycle_snapshot=lifecycle,
            risk_snapshot=risk,
            state_snapshot=state,
            explicit_preview_strategy_id="S01",
        )
        self.assertEqual(before, (stage4h1, registry, scheduler, lifecycle, risk, state))

    def test_cli_requires_dry_run_only_before_parsing(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_stage4h2_automation_wiring_preview([])
        self.assertEqual(1, code)
        self.assertIn("--dry-run-only", stderr.getvalue())

    def test_cli_json_stdout_is_strict_json(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4h2_automation_wiring_preview(
                [
                    "--dry-run-only",
                    "--json",
                    "--stage4h1-readiness-json",
                    json.dumps(valid_stage4h1_report()),
                    "--risk-snapshot-json",
                    json.dumps(clean_risk_snapshot()),
                ]
            )
        self.assertEqual(0, code)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4h2_automation_wiring_preview_report"])

    def test_cli_json_parse_errors_include_exception_type(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4h2_automation_wiring_preview(
                ["--dry-run-only", "--json", "--stage4h1-readiness-json", "{bad"]
            )
        self.assertEqual(1, code)
        parsed = json.loads(stdout.getvalue())
        self.assertIn("JSONDecodeError", parsed["errors"][0])

    def test_cli_exposes_no_forbidden_actions(self) -> None:
        source = inspect.getsource(tool.run_stage4h2_automation_wiring_preview)
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
            "--scheduler-enable",
            "--lifecycle-enable",
        ):
            self.assertNotIn(forbidden, source)

    def test_stage4h2_files_do_not_call_forbidden_external_apis_or_mutators(self) -> None:
        forbidden_calls = (
            "submit_order_plan(",
            "get_order_status(",
            "cancel_order(",
            "placeOrder(",
            "cancelOrder(",
            "reqMktData(",
            "qualifyContracts(",
            "StateStore(",
            "scheduler.add_job(",
            "add_job(",
            "start(",
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
            "open(",
            "ib_insync",
        )
        for path in STAGE4H2_FILES:
            source = path.read_text(encoding="utf-8")
            for value in forbidden_calls:
                self.assertNotIn(value, source, f"{value} unexpectedly found in {path}")

    def test_no_daemon_scheduler_or_lifecycle_wiring_exists(self) -> None:
        for path in UNWIRED_RUNTIME_FILES:
            if path.exists():
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("stage4h2_automation_wiring_preview", source)


if __name__ == "__main__":
    unittest.main()
