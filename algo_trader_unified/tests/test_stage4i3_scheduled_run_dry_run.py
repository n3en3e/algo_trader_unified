from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
import json
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4i3_scheduled_run_dry_run import (
    MARKET_WINDOW_MANUAL_WARNING,
    PRE_RUN_CHECKS_REQUIRED,
    build_stage4i3_scheduled_run_dry_run_report,
)
from algo_trader_unified.tools import stage4i3_scheduled_run_dry_run as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4I3_FILES = [
    ROOT / "core/stage4i3_scheduled_run_dry_run.py",
    ROOT / "tools/stage4i3_scheduled_run_dry_run.py",
]


FLOW_STAGES = [
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


def valid_stage4i2_report(**overrides: object) -> dict:
    flow = [
        {
            "sequence_number": index,
            "stage": stage,
            "target_component": f"{stage}_component",
            "payload": {"selected_strategy_id": "S01"},
            "would_execute": False,
            "would_submit": False,
            "paper_only": True,
            "live_trading_enabled": False,
        }
        for index, stage in enumerate(FLOW_STAGES, start=1)
    ]
    report: dict[str, object] = {
        "dry_run": True,
        "stage4i2_scheduled_paper_run_plan_report": True,
        "generated_at": "2026-05-14T14:00:00+00:00",
        "artifact_checks": {
            "stage4i2_report_present": True,
            "stage4i2_report_ready": True,
            "selected_strategy_present": True,
        },
        "selected_strategy": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
        },
        "run_plan": {
            "available": True,
            "proposed_schedule": {
                "selected_strategy_id": "S01",
                "cadence": "once",
                "dry_run_only": True,
                "would_register": False,
                "would_execute": False,
                "scheduler_job_enabled": False,
            },
            "proposed_execution_flow": flow,
        },
        "safety_checks": {
            "no_live_trading": True,
            "no_all_strategy_enablement": True,
            "no_broker_submission_enabled": True,
            "no_market_data": True,
            "no_contract_qualification": True,
            "no_order_submission": True,
            "no_strategy_scan_execution": True,
            "no_scheduler_registration": True,
            "no_lifecycle_execution": True,
        },
        "readiness_for_stage4i3": {
            "ready_to_build_scheduled_run_dry_run": True,
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
        "reason": "operator supplied clean paper run window",
    }
    _deep_update(snapshot, overrides)
    return snapshot


def build(
    *,
    i2: dict | None = None,
    activation_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    market_window_snapshot: dict | None = None,
) -> dict:
    return build_stage4i3_scheduled_run_dry_run_report(
        stage4i2_run_plan_report=valid_stage4i2_report() if i2 is None else i2,
        activation_snapshot=activation_snapshot,
        state_snapshot=state_snapshot,
        risk_snapshot=risk_snapshot,
        scheduler_snapshot=scheduler_snapshot,
        lifecycle_snapshot=lifecycle_snapshot,
        paper_broker_snapshot=paper_broker_snapshot,
        market_window_snapshot=market_window_snapshot,
        now_provider=lambda: datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc),
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
    }
    defaults.update(kwargs)
    return build(**defaults)


def assert_ready(test_case: unittest.TestCase, report: dict) -> None:
    test_case.assertTrue(
        report["readiness_for_stage4i4"]["ready_to_build_scheduler_activation_gate"]
    )


def assert_not_ready(test_case: unittest.TestCase, report: dict) -> None:
    test_case.assertFalse(
        report["readiness_for_stage4i4"]["ready_to_build_scheduler_activation_gate"]
    )
    test_case.assertTrue(report["readiness_for_stage4i4"]["blockers"])


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class Stage4I3ScheduledRunDryRunTests(unittest.TestCase):
    def test_missing_stage4i2_report_blocks_readiness(self) -> None:
        report = build_stage4i3_scheduled_run_dry_run_report(stage4i2_run_plan_report=None)
        assert_not_ready(self, report)

    def test_stage4i2_report_not_ready_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            i2=valid_stage4i2_report(
                readiness_for_stage4i3={"ready_to_build_scheduled_run_dry_run": False}
            )
        )
        assert_not_ready(self, report)

    def test_missing_selected_strategy_blocks_readiness_and_uses_safe_traversal(self) -> None:
        i2 = valid_stage4i2_report(selected_strategy="bad")
        report = build_with_all_snapshots(i2=i2)
        assert_not_ready(self, report)
        self.assertIsNone(report["selected_strategy"]["selected_strategy_id"])

    def test_missing_run_plan_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(i2=valid_stage4i2_report(run_plan=None))
        assert_not_ready(self, report)

    def test_proposed_schedule_disabled_fields_block_when_true(self) -> None:
        for key in ("would_register", "would_execute", "scheduler_job_enabled"):
            with self.subTest(key=key):
                report = build_with_all_snapshots(
                    i2=valid_stage4i2_report(run_plan={"proposed_schedule": {key: True}})
                )
                assert_not_ready(self, report)

    def test_malformed_execution_flow_blocks_without_crashing(self) -> None:
        report = build_with_all_snapshots(
            i2=valid_stage4i2_report(run_plan={"proposed_execution_flow": "bad"})
        )
        assert_not_ready(self, report)

    def test_non_dict_flow_step_blocks_without_crashing(self) -> None:
        i2 = valid_stage4i2_report()
        i2["run_plan"]["proposed_execution_flow"].append("bad")  # type: ignore[index,union-attr]
        report = build_with_all_snapshots(i2=i2)
        assert_not_ready(self, report)

    def test_flow_step_missing_target_or_payload_blocks_readiness(self) -> None:
        for step in (
            {"stage": "custom", "payload": {}, "would_execute": False},
            {"stage": "custom", "target_component": "component", "would_execute": False},
        ):
            with self.subTest(step=step):
                i2 = valid_stage4i2_report()
                i2["run_plan"]["proposed_execution_flow"] = [step]  # type: ignore[index]
                report = build_with_all_snapshots(i2=i2)
                assert_not_ready(self, report)

    def test_flow_step_would_execute_or_broker_would_submit_blocks(self) -> None:
        executable = valid_stage4i2_report()
        executable["run_plan"]["proposed_execution_flow"][0]["would_execute"] = True  # type: ignore[index]
        assert_not_ready(self, build_with_all_snapshots(i2=executable))
        broker = valid_stage4i2_report()
        broker["run_plan"]["proposed_execution_flow"][9]["would_submit"] = True  # type: ignore[index]
        assert_not_ready(self, build_with_all_snapshots(i2=broker))

    def test_trace_handles_target_function_and_target_component_and_preserves_order(self) -> None:
        flow = [
            {
                "stage": "pre_run_snapshot_check",
                "target_function": "pre_check_fn",
                "payload": {"a": 1},
                "would_execute": False,
            },
            {
                "stage": "strategy_scan_preview",
                "target_component": "strategy_preview_component",
                "payload": {"b": 2},
                "would_execute": False,
            },
        ]
        i2 = valid_stage4i2_report(run_plan={"proposed_execution_flow": flow})
        report = build_with_all_snapshots(i2=i2)
        assert_ready(self, report)
        trace = report["dry_run_trace"]
        self.assertEqual(["pre_check_fn", "strategy_preview_component"], [
            item.get("target_function") or item.get("target_component") for item in trace
        ])
        self.assertEqual([1, 2], [item["sequence_number"] for item in trace])

    def test_trace_items_are_simulated_and_never_enable_forbidden_actions(self) -> None:
        report = build_with_all_snapshots()
        for item in report["dry_run_trace"]:
            self.assertEqual("simulated", item["status"])
            self.assertFalse(item["would_execute"])
            self.assertFalse(item.get("would_submit", False))
            self.assertFalse(item["would_write_state"])
            self.assertFalse(item["would_write_ledger"])
            self.assertFalse(item["would_register_scheduler"])
            self.assertTrue(item["paper_only"])
            self.assertFalse(item["live_trading_enabled"])

    def test_expected_stage_placeholders_are_deterministic(self) -> None:
        report = build_with_all_snapshots()
        by_stage = {item["source_stage"]: item["dry_run_result"] for item in report["dry_run_trace"]}
        self.assertEqual(PRE_RUN_CHECKS_REQUIRED, by_stage["pre_run_snapshot_check"]["checks_required"])
        self.assertEqual("not_executed_in_4I3", by_stage["strategy_scan_preview"]["scan_output_placeholder"])
        self.assertEqual("not_created_in_4I3", by_stage["signal_to_intent_preview"]["intent_placeholder"])
        self.assertEqual("not_created_in_4I3", by_stage["intent_to_ticket_preview"]["ticket_placeholder"])
        self.assertEqual("blocked_in_4I3", by_stage["paper_order_submission_gate_preview"]["submission_placeholder"])
        self.assertFalse(by_stage["state_ledger_tracking_preview"]["would_write_state"])
        self.assertFalse(by_stage["state_ledger_tracking_preview"]["would_write_ledger"])
        self.assertEqual("not_executed_in_4I3", by_stage["post_run_reconciliation_preview"]["reconciliation_placeholder"])

    def test_activation_snapshot_rules(self) -> None:
        assert_ready(self, build_with_all_snapshots())
        assert_not_ready(
            self,
            build_with_all_snapshots(
                activation_snapshot=matching_activation_snapshot(
                    activation_record={"selected_strategy_id": "S02"}
                )
            ),
        )
        assert_not_ready(
            self,
            build_with_all_snapshots(
                activation_snapshot=matching_activation_snapshot(active_strategy_ids=["S01", "S02"])
            ),
        )
        report = build_with_all_snapshots(
            activation_snapshot={
                "active_strategy_ids": ["S01"],
                "activations": ["bad", {"selected_strategy_id": "S01", "paper_only": True}],
            }
        )
        assert_ready(self, report)
        self.assertIn("malformed activation snapshot entry ignored", report["warnings"])

    def test_state_snapshot_blockers_and_active_intent_warning(self) -> None:
        assert_not_ready(self, build_with_all_snapshots(state_snapshot=clean_state_snapshot(active_halt=True)))
        assert_not_ready(
            self,
            build_with_all_snapshots(
                state_snapshot=clean_state_snapshot(unresolved_needs_reconciliation_count=1)
            ),
        )
        state = clean_state_snapshot(needs_reconciliation_count=1)
        state.pop("unresolved_needs_reconciliation_count")
        assert_not_ready(self, build_with_all_snapshots(state_snapshot=state))
        assert_not_ready(
            self,
            build_with_all_snapshots(state_snapshot=clean_state_snapshot(active_intents_count=1)),
        )
        safe = build_with_all_snapshots(
            state_snapshot=clean_state_snapshot(
                active_intents_count=1, active_intents_safe_for_enablement=True
            )
        )
        assert_ready(self, safe)
        self.assertTrue(safe["warnings"])
        assert_ready(self, build_with_all_snapshots(state_snapshot=clean_state_snapshot(open_positions_count=5)))

    def test_scheduler_lifecycle_risk_and_broker_blockers(self) -> None:
        assert_not_ready(
            self, build_with_all_snapshots(scheduler_snapshot={"scheduler_automation_enabled": True})
        )
        assert_not_ready(
            self, build_with_all_snapshots(scheduler_snapshot={"all_strategy_scheduler_enabled": True})
        )
        assert_not_ready(self, build_with_all_snapshots(scheduler_snapshot={"jobs": [{"strategy_id": "S01"}]}))
        assert_ready(
            self,
            build_with_all_snapshots(
                scheduler_snapshot={"jobs": [{"strategy_id": "S01", "disabled": True, "dry_run_only": True}]}
            ),
        )
        assert_not_ready(
            self, build_with_all_snapshots(lifecycle_snapshot={"lifecycle_automation_enabled": True})
        )
        assert_ready(
            self, build_with_all_snapshots(lifecycle_snapshot={"disabled": True, "dry_run_only": True})
        )
        assert_not_ready(
            self,
            build_with_all_snapshots(
                lifecycle_snapshot={"lifecycle_transition_execution_enabled": True}
            ),
        )
        for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
            assert_not_ready(self, build_with_all_snapshots(risk_snapshot=clean_risk_snapshot(**{key: False})))
        assert_not_ready(self, build_with_all_snapshots(risk_snapshot=clean_risk_snapshot(risk_bypass_enabled=True)))
        for overrides in (
            {"mode": "LIVE"},
            {"ibkr_port": 4002},
            {"paper_trading": False},
            {"live_trading_enabled": True},
            {"broker_submission_enabled": True},
        ):
            assert_not_ready(
                self,
                build_with_all_snapshots(
                    paper_broker_snapshot=clean_paper_broker_snapshot(**overrides)
                ),
            )

    def test_market_window_blockers_warnings_and_missing_optional_snapshots(self) -> None:
        assert_not_ready(
            self,
            build_with_all_snapshots(
                market_window_snapshot=clean_market_window_snapshot(
                    allowed_to_schedule_paper_run=False
                )
            ),
        )
        closed = build_with_all_snapshots(
            market_window_snapshot=clean_market_window_snapshot(market_open=False)
        )
        assert_ready(self, closed)
        self.assertTrue(any("market is currently closed" in item for item in closed["warnings"]))
        missing = build()
        assert_ready(self, missing)
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, missing["warnings"])

    def test_stage4i2_safety_false_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            i2=valid_stage4i2_report(safety_checks={"no_market_data": False})
        )
        assert_not_ready(self, report)

    def test_report_fields_json_safe_and_inputs_not_mutated(self) -> None:
        inputs = {
            "i2": valid_stage4i2_report(),
            "activation_snapshot": matching_activation_snapshot(),
            "state_snapshot": clean_state_snapshot(),
            "risk_snapshot": clean_risk_snapshot(),
            "scheduler_snapshot": {},
            "lifecycle_snapshot": {},
            "paper_broker_snapshot": clean_paper_broker_snapshot(),
            "market_window_snapshot": clean_market_window_snapshot(),
        }
        before = copy.deepcopy(inputs)
        report = build(**inputs)
        for key in (
            "dry_run",
            "stage4i3_scheduled_run_dry_run_report",
            "generated_at",
            "artifact_checks",
            "selected_strategy",
            "schedule_checks",
            "flow_checks",
            "dry_run_trace",
            "dry_run_trace_checks",
            "activation_checks",
            "state_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "paper_broker_checks",
            "market_window_checks",
            "safety_checks",
            "readiness_for_stage4i4",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report, sort_keys=True)
        self.assertEqual(before, inputs)

    def test_recommendations_do_not_include_disallowed_next_actions(self) -> None:
        text = json.dumps(build_with_all_snapshots()["recommendations"]).lower()
        for forbidden in (
            "do not enable live trading",
            "do not enable all strategies",
            "do not place orders now",
            "do not register scheduler jobs now",
            "do not bypass risk controls",
        ):
            self.assertIn(forbidden, text)
        next_steps = json.dumps(build_with_all_snapshots()["recommendations"]["ordered_next_steps"]).lower()
        self.assertNotIn("place orders now", next_steps)
        self.assertNotIn("enable live trading", next_steps)
        self.assertNotIn("enable all strategies", next_steps)
        self.assertNotIn("register scheduler jobs now", next_steps)

    def test_cli_requires_dry_run_only_before_parsing(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = tool.run_stage4i3_scheduled_run_dry_run(["--stage4i2-plan-json", "{bad"])
        self.assertEqual(1, code)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("--dry-run-only", stderr.getvalue())

    def test_cli_json_stdout_is_strict_json_only_and_parse_errors_report_type(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = tool.run_stage4i3_scheduled_run_dry_run(
                [
                    "--dry-run-only",
                    "--json",
                    "--stage4i2-plan-json",
                    json.dumps(valid_stage4i2_report()),
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
                ]
            )
        self.assertEqual(0, code)
        self.assertEqual("", stderr.getvalue())
        self.assertTrue(json.loads(stdout.getvalue())["stage4i3_scheduled_run_dry_run_report"])

        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = tool.run_stage4i3_scheduled_run_dry_run(
                ["--dry-run-only", "--json", "--stage4i2-plan-json", "{bad"]
            )
        self.assertEqual(1, code)
        self.assertIn("JSONDecodeError", json.loads(stdout.getvalue())["errors"][0])

    def test_cli_exposes_no_forbidden_action_options(self) -> None:
        parser_source = (ROOT / "tools/stage4i3_scheduled_run_dry_run.py").read_text()
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

    def test_no_forbidden_calls_or_production_wiring_in_stage4i3_files(self) -> None:
        source = "\n".join(path.read_text() for path in STAGE4I3_FILES)
        forbidden_patterns = [
            r"\bsubmit_order_plan\s*\(",
            r"\bget_order_status\s*\(",
            r"\bcancel_order\s*\(",
            r"\bplaceOrder\s*\(",
            r"\bcancelOrder\s*\(",
            r"\breqMktData\s*\(",
            r"\bqualifyContracts\s*\(",
            r"\bscheduler\.add_job\s*\(",
            r"\badd_job\s*\(",
            r"\bstart\s*\(",
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
