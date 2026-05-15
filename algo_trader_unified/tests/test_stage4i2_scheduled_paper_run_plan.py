from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
import json
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4i2_scheduled_paper_run_plan import (
    MARKET_WINDOW_MANUAL_WARNING,
    build_stage4i2_scheduled_paper_run_plan_report,
)
from algo_trader_unified.tools import stage4i2_scheduled_paper_run_plan as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4I2_FILES = [
    ROOT / "core/stage4i2_scheduled_paper_run_plan.py",
    ROOT / "tools/stage4i2_scheduled_paper_run_plan.py",
]


def valid_stage4i1_report(**overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4i1_scheduled_paper_run_readiness_report": True,
        "generated_at": "2026-05-14T14:00:00+00:00",
        "artifact_checks": {
            "stage4i1_report_present": True,
            "stage4i1_report_ready": True,
            "selected_strategy_present": True,
        },
        "selected_strategy": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
        },
        "safety_checks": {
            "no_live_trading": True,
            "no_all_strategy_enablement": True,
            "no_broker_submission_enabled": True,
            "no_market_data": True,
            "no_contract_qualification": True,
            "no_order_submission": True,
        },
        "readiness_for_stage4i2": {
            "ready_to_build_first_scheduled_run_plan": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def matching_activation_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "activation_record": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "enabled_strategy_count": 1,
            "live_trading_enabled": False,
            "all_strategies_enabled": False,
            "broker_submission_enabled": False,
        },
        "active_strategy_ids": ["S01"],
    }
    _deep_update(snapshot, overrides)
    return snapshot


def clean_state_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "active_halt": False,
        "unresolved_needs_reconciliation_count": 0,
        "active_intents_count": 0,
        "open_positions_count": 2,
    }
    _deep_update(snapshot, overrides)
    return snapshot


def clean_risk_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "kill_switch_available": True,
        "hard_halt_available": True,
        "daily_loss_limit_available": True,
        "max_position_limit_available": True,
        "risk_bypass_enabled": False,
    }
    _deep_update(snapshot, overrides)
    return snapshot


def clean_paper_broker_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "mode": "PAPER",
        "paper_trading": True,
        "ibkr_port": 4004,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }
    _deep_update(snapshot, overrides)
    return snapshot


def clean_market_window_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "is_trading_day": True,
        "market_open": True,
        "allowed_to_schedule_paper_run": True,
        "reason": "operator supplied clean paper planning window",
    }
    _deep_update(snapshot, overrides)
    return snapshot


def valid_run_window(**overrides: object) -> dict:
    config: dict[str, object] = {
        "schedule_label": "first-paper-plan",
        "timezone": "America/New_York",
        "first_run_date": "2026-05-18",
        "run_time": "09:45",
        "cadence": "once",
        "max_runtime_seconds": 300,
        "dry_run_only": True,
        "operator_notes": "plan only",
    }
    _deep_update(config, overrides)
    return config


def build(
    *,
    i1: dict | None = None,
    activation_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    market_window_snapshot: dict | None = None,
    run_window_config: dict | None = None,
) -> dict:
    return build_stage4i2_scheduled_paper_run_plan_report(
        stage4i1_readiness_report=valid_stage4i1_report() if i1 is None else i1,
        activation_snapshot=activation_snapshot,
        state_snapshot=state_snapshot,
        risk_snapshot=risk_snapshot,
        scheduler_snapshot=scheduler_snapshot,
        lifecycle_snapshot=lifecycle_snapshot,
        paper_broker_snapshot=paper_broker_snapshot,
        market_window_snapshot=market_window_snapshot,
        run_window_config=run_window_config,
        now_provider=lambda: datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc),
    )


def build_with_all_snapshots(**kwargs: object) -> dict:
    defaults: dict[str, object] = {
        "activation_snapshot": matching_activation_snapshot(),
        "state_snapshot": clean_state_snapshot(),
        "risk_snapshot": clean_risk_snapshot(),
        "scheduler_snapshot": {},
        "lifecycle_snapshot": {},
        "paper_broker_snapshot": clean_paper_broker_snapshot(),
        "market_window_snapshot": clean_market_window_snapshot(),
        "run_window_config": valid_run_window(),
    }
    defaults.update(kwargs)
    return build(**defaults)


def assert_ready(test_case: unittest.TestCase, report: dict) -> None:
    test_case.assertTrue(
        report["readiness_for_stage4i3"]["ready_to_build_scheduled_run_dry_run"]
    )


def assert_not_ready(test_case: unittest.TestCase, report: dict) -> None:
    test_case.assertFalse(
        report["readiness_for_stage4i3"]["ready_to_build_scheduled_run_dry_run"]
    )
    test_case.assertTrue(report["readiness_for_stage4i3"]["blockers"])


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class Stage4I2ScheduledPaperRunPlanTests(unittest.TestCase):
    def test_missing_stage4i1_report_blocks_readiness(self) -> None:
        report = build_stage4i2_scheduled_paper_run_plan_report(
            stage4i1_readiness_report=None
        )
        assert_not_ready(self, report)

    def test_stage4i1_report_not_ready_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            i1=valid_stage4i1_report(
                readiness_for_stage4i2={"ready_to_build_first_scheduled_run_plan": False}
            )
        )
        assert_not_ready(self, report)

    def test_missing_selected_strategy_blocks_readiness(self) -> None:
        i1 = valid_stage4i1_report()
        i1["selected_strategy"].pop("selected_strategy_id")  # type: ignore[index,union-attr]
        report = build_with_all_snapshots(i1=i1)
        assert_not_ready(self, report)

    def test_selected_strategy_extraction_uses_safe_selected_strategy_traversal(self) -> None:
        report = build_with_all_snapshots(
            i1=valid_stage4i1_report(
                activation_payload_checks={"selected_strategy_id": "S99"},
                selected_strategy={
                    "selected_strategy_id": "S01",
                    "paper_only": True,
                    "one_strategy_only": True,
                },
            )
        )
        self.assertEqual("S01", report["selected_strategy"]["selected_strategy_id"])

    def test_more_than_one_activated_strategy_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            i1=valid_stage4i1_report(selected_strategy={"one_strategy_only": False})
        )
        assert_not_ready(self, report)

    def test_activation_snapshot_matching_selected_strategy_passes(self) -> None:
        report = build_with_all_snapshots()
        assert_ready(self, report)
        self.assertTrue(report["activation_checks"]["activation_snapshot_matches"])

    def test_activation_snapshot_mismatched_selected_strategy_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            activation_snapshot=matching_activation_snapshot(
                activation_record={"selected_strategy_id": "S02"}
            )
        )
        assert_not_ready(self, report)

    def test_activation_snapshot_active_strategy_ids_with_multiple_ids_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            activation_snapshot=matching_activation_snapshot(active_strategy_ids=["S01", "S02"])
        )
        assert_not_ready(self, report)

    def test_activation_snapshot_malformed_arrays_do_not_crash(self) -> None:
        report = build(
            activation_snapshot={
                "activations": [
                    "bad",
                    {
                        "selected_strategy_id": "S01",
                        "paper_only": True,
                        "enabled_strategy_count": 1,
                    },
                ],
                "active_strategy_ids": ["S01"],
            },
            state_snapshot=clean_state_snapshot(),
            risk_snapshot=clean_risk_snapshot(),
            scheduler_snapshot={},
            lifecycle_snapshot={},
            paper_broker_snapshot=clean_paper_broker_snapshot(),
            market_window_snapshot=clean_market_window_snapshot(),
            run_window_config=valid_run_window(),
        )
        assert_ready(self, report)
        self.assertIn("malformed activation snapshot entry ignored", report["warnings"])

    def test_active_halt_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(state_snapshot=clean_state_snapshot(active_halt=True))
        assert_not_ready(self, report)

    def test_unresolved_needs_reconciliation_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            state_snapshot=clean_state_snapshot(unresolved_needs_reconciliation_count=1)
        )
        assert_not_ready(self, report)

    def test_needs_reconciliation_count_blocks_readiness(self) -> None:
        state_snapshot = clean_state_snapshot(needs_reconciliation_count=1)
        state_snapshot.pop("unresolved_needs_reconciliation_count")
        report = build_with_all_snapshots(
            state_snapshot=state_snapshot
        )
        assert_not_ready(self, report)

    def test_active_intents_block_unless_safe(self) -> None:
        unsafe = build_with_all_snapshots(
            state_snapshot=clean_state_snapshot(active_intents_count=1)
        )
        assert_not_ready(self, unsafe)
        safe = build_with_all_snapshots(
            state_snapshot=clean_state_snapshot(
                active_intents_count=1, active_intents_safe_for_enablement=True
            )
        )
        assert_ready(self, safe)
        self.assertTrue(safe["warnings"])

    def test_open_positions_count_alone_does_not_block(self) -> None:
        report = build_with_all_snapshots(
            state_snapshot=clean_state_snapshot(open_positions_count=4)
        )
        assert_ready(self, report)

    def test_scheduler_broad_automation_enabled_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            scheduler_snapshot={"scheduler_automation_enabled": True}
        )
        assert_not_ready(self, report)

    def test_selected_strategy_job_already_active_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            scheduler_snapshot={"jobs": [{"strategy_id": "S01"}]}
        )
        assert_not_ready(self, report)

    def test_disabled_selected_strategy_job_does_not_block(self) -> None:
        report = build_with_all_snapshots(
            scheduler_snapshot={"jobs": [{"strategy_id": "S01", "disabled": True, "dry_run_only": True}]}
        )
        assert_ready(self, report)

    def test_lifecycle_automation_enabled_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            lifecycle_snapshot={"lifecycle_automation_enabled": True}
        )
        assert_not_ready(self, report)

    def test_lifecycle_transition_execution_enabled_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            lifecycle_snapshot={"lifecycle_transition_execution_enabled": True}
        )
        assert_not_ready(self, report)

    def test_risk_snapshot_blockers(self) -> None:
        for key in (
            "kill_switch_available",
            "hard_halt_available",
            "daily_loss_limit_available",
        ):
            with self.subTest(key=key):
                report = build_with_all_snapshots(
                    risk_snapshot=clean_risk_snapshot(**{key: False})
                )
                assert_not_ready(self, report)
        bypass = build_with_all_snapshots(risk_snapshot=clean_risk_snapshot(risk_bypass_enabled=True))
        assert_not_ready(self, bypass)

    def test_paper_broker_snapshot_blockers(self) -> None:
        cases = [
            {"mode": "LIVE"},
            {"ibkr_port": 4002},
            {"paper_trading": False},
            {"live_trading_enabled": True},
            {"broker_submission_enabled": True},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                report = build_with_all_snapshots(
                    paper_broker_snapshot=clean_paper_broker_snapshot(**overrides)
                )
                assert_not_ready(self, report)

    def test_market_window_disallow_blocks_and_closed_market_only_warns(self) -> None:
        blocked = build_with_all_snapshots(
            market_window_snapshot=clean_market_window_snapshot(
                allowed_to_schedule_paper_run=False
            )
        )
        assert_not_ready(self, blocked)
        closed = build_with_all_snapshots(
            market_window_snapshot=clean_market_window_snapshot(market_open=False)
        )
        assert_ready(self, closed)
        self.assertTrue(any("market is currently closed" in item for item in closed["warnings"]))

    def test_missing_market_window_snapshot_emits_manual_warning(self) -> None:
        report = build_with_all_snapshots(market_window_snapshot=None)
        assert_ready(self, report)
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, report["warnings"])

    def test_run_window_dry_run_false_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(run_window_config=valid_run_window(dry_run_only=False))
        assert_not_ready(self, report)

    def test_unsupported_cadence_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(run_window_config=valid_run_window(cadence="hourly"))
        assert_not_ready(self, report)

    def test_cadence_validation_strips_and_is_case_insensitive(self) -> None:
        report = build_with_all_snapshots(run_window_config=valid_run_window(cadence=" DAILY "))
        assert_ready(self, report)
        self.assertEqual("daily", report["run_plan"]["proposed_schedule"]["cadence"])

    def test_missing_run_window_config_warns_and_uses_safe_defaults(self) -> None:
        report = build_with_all_snapshots(run_window_config=None)
        assert_ready(self, report)
        schedule = report["run_plan"]["proposed_schedule"]
        self.assertEqual("once", schedule["cadence"])
        self.assertTrue(schedule["dry_run_only"])
        self.assertTrue(any("run window config missing" in item for item in report["warnings"]))

    def test_proposed_schedule_is_structured_and_disabled(self) -> None:
        report = build_with_all_snapshots()
        schedule = report["run_plan"]["proposed_schedule"]
        self.assertIsInstance(schedule, dict)
        self.assertEqual("S01", schedule["selected_strategy_id"])
        self.assertFalse(schedule["would_register"])
        self.assertFalse(schedule["would_execute"])
        self.assertFalse(schedule["scheduler_job_enabled"])

    def test_proposed_execution_flow_is_ordered_structured_and_disabled(self) -> None:
        report = build_with_all_snapshots()
        flow = report["run_plan"]["proposed_execution_flow"]
        expected = [
            "pre_run_snapshot_check",
            "risk_gate_check",
            "activation_artifact_check",
            "scheduler_gate_check",
            "lifecycle_gate_check",
            "market_window_check",
            "strategy_scan_preview",
            "signal_to_intent_preview",
            "intent_to_ticket_preview",
            "paper_order_submission_gate_preview",
            "state_ledger_tracking_preview",
            "alert_report_preview",
            "post_run_reconciliation_preview",
        ]
        self.assertEqual(expected, [step["stage"] for step in flow])
        self.assertEqual(list(range(1, 14)), [step["sequence_number"] for step in flow])
        for step in flow:
            self.assertIn("target_component", step)
            self.assertIsInstance(step["payload"], dict)
            self.assertFalse(step["would_execute"])
            self.assertFalse(step["would_submit"])
            self.assertTrue(step["paper_only"])
            self.assertFalse(step["live_trading_enabled"])

    def test_broker_submission_gate_step_has_would_submit_false(self) -> None:
        report = build_with_all_snapshots()
        step = [
            item
            for item in report["run_plan"]["proposed_execution_flow"]
            if item["stage"] == "paper_order_submission_gate_preview"
        ][0]
        self.assertFalse(step["would_submit"])

    def test_proposed_run_id_is_deterministic_and_stable(self) -> None:
        first = build_with_all_snapshots(run_window_config=valid_run_window(cadence="daily"))
        second = build_with_all_snapshots(run_window_config=valid_run_window(cadence="daily"))
        run_id = first["run_plan"]["proposed_run_id"]
        self.assertEqual(run_id, second["run_plan"]["proposed_run_id"])
        self.assertEqual("s01_first_paper_plan_daily_0945_2026_05_14", run_id)
        self.assertNotRegex(run_id, r"\d{2}:\d{2}|\.\d+")

    def test_recommendations_do_not_recommend_disallowed_actions(self) -> None:
        report = build_with_all_snapshots()
        text = json.dumps(report["recommendations"]).lower()
        for forbidden in (
            "place orders now",
            "enable all strategies",
            "enable live trading",
            "enable scheduler jobs now",
            "bypass risk controls",
        ):
            self.assertIn(f"do not {forbidden}", text)
        self.assertNotIn("place orders now.", json.dumps(report["recommendations"]["ordered_next_steps"]).lower())

    def test_report_includes_required_top_level_fields_and_is_json_safe(self) -> None:
        report = build_with_all_snapshots()
        for key in (
            "dry_run",
            "stage4i2_scheduled_paper_run_plan_report",
            "generated_at",
            "artifact_checks",
            "selected_strategy",
            "run_plan",
            "activation_checks",
            "state_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "paper_broker_checks",
            "market_window_checks",
            "safety_checks",
            "readiness_for_stage4i3",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report, sort_keys=True)

    def test_inputs_are_not_mutated(self) -> None:
        inputs = {
            "i1": valid_stage4i1_report(),
            "activation_snapshot": matching_activation_snapshot(),
            "state_snapshot": clean_state_snapshot(),
            "risk_snapshot": clean_risk_snapshot(),
            "scheduler_snapshot": {},
            "lifecycle_snapshot": {},
            "paper_broker_snapshot": clean_paper_broker_snapshot(),
            "market_window_snapshot": clean_market_window_snapshot(),
            "run_window_config": valid_run_window(cadence=" DAILY "),
        }
        before = copy.deepcopy(inputs)
        build(**inputs)
        self.assertEqual(before, inputs)

    def test_safety_flags_from_stage4i1_and_snapshots_block_readiness(self) -> None:
        i1 = valid_stage4i1_report(safety_checks={"no_market_data": False})
        report = build_with_all_snapshots(i1=i1)
        assert_not_ready(self, report)
        scan = build_with_all_snapshots(scheduler_snapshot={"strategy_scan_execution_enabled": True})
        assert_not_ready(self, scan)
        registration = build_with_all_snapshots(
            scheduler_snapshot={"scheduler_registration_enabled": True}
        )
        assert_not_ready(self, registration)

    def test_cli_requires_dry_run_only_before_parsing(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = tool.run_stage4i2_scheduled_paper_run_plan(
                ["--stage4i1-readiness-json", "{not json"]
            )
        self.assertEqual(1, code)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("--dry-run-only", stderr.getvalue())

    def test_cli_json_stdout_is_strict_json_only(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = tool.run_stage4i2_scheduled_paper_run_plan(
                [
                    "--dry-run-only",
                    "--json",
                    "--stage4i1-readiness-json",
                    json.dumps(valid_stage4i1_report()),
                    "--activation-snapshot-json",
                    json.dumps(matching_activation_snapshot()),
                    "--state-snapshot-json",
                    json.dumps(clean_state_snapshot()),
                    "--risk-snapshot-json",
                    json.dumps(clean_risk_snapshot()),
                    "--scheduler-snapshot-json",
                    "{}",
                    "--lifecycle-snapshot-json",
                    "{}",
                    "--paper-broker-snapshot-json",
                    json.dumps(clean_paper_broker_snapshot()),
                    "--market-window-snapshot-json",
                    json.dumps(clean_market_window_snapshot()),
                    "--run-window-config-json",
                    json.dumps(valid_run_window()),
                ]
            )
        self.assertEqual(0, code)
        self.assertEqual("", stderr.getvalue())
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4i2_scheduled_paper_run_plan_report"])

    def test_cli_json_parse_error_reports_exception_type(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = tool.run_stage4i2_scheduled_paper_run_plan(
                ["--dry-run-only", "--json", "--stage4i1-readiness-json", "{bad"]
            )
        self.assertEqual(1, code)
        parsed = json.loads(stdout.getvalue())
        self.assertIn("JSONDecodeError", parsed["errors"][0])

    def test_cli_exposes_no_forbidden_action_options(self) -> None:
        parser_source = (ROOT / "tools/stage4i2_scheduled_paper_run_plan.py").read_text()
        for forbidden in (
            "--submit",
            "--cancel",
            "--status",
            "--market-data",
            "--qualify",
            "--scheduler-enable",
            "--lifecycle-enable",
        ):
            self.assertNotIn(forbidden, parser_source)

    def test_no_forbidden_calls_or_production_wiring_in_stage4i2_files(self) -> None:
        source = "\n".join(path.read_text() for path in STAGE4I2_FILES)
        forbidden_patterns = [
            r"\bsubmit_order_plan\s*\(",
            r"\bget_order_status\s*\(",
            r"\bcancel_order\s*\(",
            r"\bplaceOrder\s*\(",
            r"\bcancelOrder\s*\(",
            r"\breqMktData\s*\(",
            r"\bqualifyContracts\s*\(",
            r"\badd_job\s*\(",
            r"\brun_scan\s*\(",
            r"\bscan_now\s*\(",
            r"\bsocket\.create_connection\s*\(",
            r"\bsocket\.socket\s*\(",
            r"\basyncio\.run\s*\(",
            r"\basyncio\.get_event_loop\s*\(",
            r"\basyncio\.new_event_loop\s*\(",
            r"\buuid\.uuid4\s*\(",
            r"\brandom\.",
            r"\btime\.time\s*\(",
            r"\bdatetime\.now\s*\(",
            r"\bStateStore\s*\(",
            r"\bib_insync\b",
            r"\byfinance\b",
            r"\brequests\b",
            r"\burllib\b",
            r"\bsystemctl\b",
            r"\bsystemd\b",
        ]
        for pattern in forbidden_patterns:
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, source))


if __name__ == "__main__":
    unittest.main()
