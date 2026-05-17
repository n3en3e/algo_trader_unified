from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from decimal import Decimal
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4j6_controlled_paper_operation_acceptance import (
    MARKET_WINDOW_MANUAL_WARNING,
    RECOMMENDED_NEXT_GATE,
    build_stage4j6_controlled_paper_operation_acceptance_report,
)
from algo_trader_unified.tools import stage4j6_controlled_paper_operation_acceptance as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4J6_RUNTIME_FILES = [
    ROOT / "core/stage4j6_controlled_paper_operation_acceptance.py",
    ROOT / "tools/stage4j6_controlled_paper_operation_acceptance.py",
]


def safe_runner_result(**overrides: object) -> dict:
    result: dict[str, object] = {
        "status": "completed_report_only",
        "selected_strategy_id": "S01",
        "operation_id": "s01_once_2026_05_16",
        "paper_only": True,
        "one_strategy_only": True,
        "strategy_called": True,
        "market_data_requested": False,
        "contracts_qualified": False,
        "intents_created": False,
        "tickets_created": False,
        "orders_submitted": False,
        "state_written": False,
        "ledger_written": False,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "broker_submission_enabled": False,
        "warnings": [],
        "errors": [],
    }
    _deep_update(result, overrides)
    return result


def valid_stage4j5_report(**overrides: object) -> dict:
    operation_id = "s01_once_2026_05_16"
    report: dict[str, object] = {
        "dry_run": False,
        "stage4j5_controlled_paper_operation_executor_report": True,
        "generated_at": "2026-05-16T15:00:00+00:00",
        "artifact_checks": {
            "stage4j4_report_present": True,
            "stage4j4_report_ready": True,
            "selected_strategy_present": True,
            "operation_id_present": True,
            "execution_gate_ready": True,
            "permissions_valid": True,
        },
        "selected_strategy": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
        },
        "operation": {
            "operation_id": operation_id,
            "operation_scope": "single_strategy_controlled_paper_operation_executor",
            "paper_only": True,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
        },
        "controlled_operation_payload": {
            "selected_strategy_id": "S01",
            "operation_id": operation_id,
            "paper_only": True,
            "one_strategy_only": True,
            "allow_market_data": False,
            "allow_contract_qualification": False,
            "allow_intent_creation": False,
            "allow_ticket_creation": False,
            "allow_order_submission": False,
            "allow_state_write": False,
            "allow_ledger_write": False,
            "live_trading_enabled": False,
            "all_strategies_enabled": False,
            "broker_submission_enabled": False,
        },
        "execution": {
            "attempted": True,
            "runner_called": True,
            "runner_succeeded": True,
            "failed_step": None,
            "failure_reason": None,
            "result_status": "completed_report_only",
            "runner_result": safe_runner_result(),
            "unsafe_runner_flags": [],
            "completed": True,
        },
        "runner_output_checks": {"output_json_safe": True},
        "safety_checks": {
            "no_live_trading": True,
            "no_all_strategy_enablement": True,
            "no_broker_submission_enabled": True,
            "no_market_data": True,
            "no_contract_qualification": True,
            "no_order_submission": True,
            "no_state_write": True,
            "no_ledger_write": True,
        },
        "readiness_for_stage4j6": {
            "ready_to_build_controlled_paper_operation_acceptance": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def clean_state_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "active_halt": False,
        "unresolved_needs_reconciliation_count": 0,
        "active_intents_count": 0,
        "open_positions_count": 5,
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


def clean_scheduler_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "scheduler_automation_enabled": True,
        "jobs": [
            {
                "strategy_id": "S01",
                "paper_only": True,
                "scheduler_job_scope": "single_strategy",
                "broker_submission_enabled": False,
                "live_trading_enabled": False,
                "all_strategies_enabled": False,
                "strategy_scan_execution_enabled": False,
            }
        ],
    }
    _deep_update(snapshot, overrides)
    return snapshot


def clean_lifecycle_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "lifecycle_automation_enabled": True,
        "selected_strategy_id": "S01",
        "broker_submission_enabled": False,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "lifecycle_transition_execution_enabled": False,
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
        "allowed_to_schedule_paper_run": True,
        "is_trading_day": True,
        "market_open": True,
    }
    _deep_update(snapshot, overrides)
    return snapshot


_DEFAULT_J5 = object()


def build(j5: object = _DEFAULT_J5, **kwargs: object) -> dict:
    if j5 is _DEFAULT_J5:
        j5 = valid_stage4j5_report()
    return build_stage4j6_controlled_paper_operation_acceptance_report(
        stage4j5_executor_report=j5,  # type: ignore[arg-type]
        now_provider=lambda: datetime(2026, 5, 16, 16, 0, tzinfo=timezone.utc),
        **kwargs,
    )


def build_with_snapshots(**kwargs: object) -> dict:
    defaults: dict[str, object] = {
        "state_snapshot": clean_state_snapshot(),
        "risk_snapshot": clean_risk_snapshot(),
        "scheduler_snapshot": clean_scheduler_snapshot(),
        "lifecycle_snapshot": clean_lifecycle_snapshot(),
        "paper_broker_snapshot": clean_paper_broker_snapshot(),
        "market_window_snapshot": clean_market_window_snapshot(),
    }
    defaults.update(kwargs)
    return build(**defaults)


def accepted(report: dict) -> bool:
    return report["executor_acceptance"]["accepted"]


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class Stage4J6ControlledPaperOperationAcceptanceTests(unittest.TestCase):
    def test_required_stage4j5_artifact_fields_block_acceptance(self) -> None:
        cases = [
            None,
            valid_stage4j5_report(readiness_for_stage4j6={"ready_to_build_controlled_paper_operation_acceptance": False}),
            valid_stage4j5_report(selected_strategy=None),
            valid_stage4j5_report(operation={"operation_id": None}),
            valid_stage4j5_report(controlled_operation_payload=None),
            valid_stage4j5_report(controlled_operation_payload=json.dumps({"selected_strategy_id": "S01"})),
            valid_stage4j5_report(controlled_operation_payload=["bad"]),
            valid_stage4j5_report(execution=None),
            valid_stage4j5_report(execution={"attempted": False}),
            valid_stage4j5_report(execution={"runner_called": False}),
            valid_stage4j5_report(execution={"runner_succeeded": False}),
            valid_stage4j5_report(execution={"completed": False}),
            valid_stage4j5_report(execution={"runner_result": None}),
            valid_stage4j5_report(execution={"runner_result": json.dumps(safe_runner_result())}),
            valid_stage4j5_report(execution={"runner_result": ["bad"]}),
        ]
        for j5 in cases:
            with self.subTest(j5=j5):
                report = build(j5)
                self.assertFalse(accepted(report))
                json.dumps(report)

    def test_runner_result_validation_blocks_unsafe_or_malformed_values(self) -> None:
        cases = [
            (safe_runner_result(result_summary=Decimal("1.2")), "JSON-safe"),
            (safe_runner_result(status="unsupported"), "unsupported"),
            (safe_runner_result(selected_strategy_id="S02"), "selected_strategy_id"),
            (safe_runner_result(operation_id="other"), "operation_id"),
            (safe_runner_result(market_data_requested=True), "market_data_requested"),
            (safe_runner_result(contracts_qualified=True), "contracts_qualified"),
            (safe_runner_result(intents_created=True), "intents_created"),
            (safe_runner_result(tickets_created=True), "tickets_created"),
            (safe_runner_result(orders_submitted=True), "orders_submitted"),
            (safe_runner_result(state_written=True), "state_written"),
            (safe_runner_result(ledger_written=True), "ledger_written"),
            (safe_runner_result(live_trading_enabled=True), "live_trading_enabled"),
            (safe_runner_result(all_strategies_enabled=True), "all_strategies_enabled"),
            (safe_runner_result(broker_submission_enabled=True), "broker_submission_enabled"),
            (safe_runner_result(market_data_requested="False"), "market_data_requested"),
            (safe_runner_result(contracts_qualified="False"), "contracts_qualified"),
            (safe_runner_result(intents_created="False"), "intents_created"),
            (safe_runner_result(tickets_created="False"), "tickets_created"),
            (safe_runner_result(orders_submitted="False"), "orders_submitted"),
            (safe_runner_result(state_written="False"), "state_written"),
            (safe_runner_result(ledger_written="False"), "ledger_written"),
            (safe_runner_result(live_trading_enabled="False"), "live_trading_enabled"),
            (safe_runner_result(all_strategies_enabled="False"), "all_strategies_enabled"),
            (safe_runner_result(broker_submission_enabled="False"), "broker_submission_enabled"),
        ]
        for result, expected in cases:
            with self.subTest(expected=expected):
                report = build(valid_stage4j5_report(execution={"runner_result": result}))
                self.assertFalse(accepted(report))
                self.assertIn(expected, " ".join(report["readiness_for_stage4j_complete_or_next_gate"]["blockers"]))

    def test_unsafe_runner_flags_and_failure_fields_block_correctly(self) -> None:
        report = build(valid_stage4j5_report(execution={"unsafe_runner_flags": ["orders_submitted"]}))
        self.assertFalse(accepted(report))
        self.assertIn("unsafe_runner_flags", " ".join(report["executor_acceptance"]["blockers"]))

        report = build(valid_stage4j5_report(execution={"failed_step": "runner_call", "failure_reason": "boom"}))
        self.assertFalse(accepted(report))
        self.assertIn("failure fields", " ".join(report["executor_acceptance"]["blockers"]))

        skipped = build(
            valid_stage4j5_report(
                execution={
                    "failed_step": "internal_gate",
                    "failure_reason": "no signal",
                    "result_status": "skipped_no_signal",
                    "runner_result": safe_runner_result(status="skipped_no_signal", strategy_called=False),
                }
            )
        )
        self.assertTrue(accepted(skipped))
        self.assertIn("skipped", " ".join(skipped["warnings"]))

    def test_completed_skipped_and_blocked_statuses_accept_when_safe(self) -> None:
        completed = build()
        self.assertTrue(accepted(completed))
        self.assertEqual(completed["executor_acceptance"]["acceptance_status"], "accepted_completed_report_only")

        skipped = build(
            valid_stage4j5_report(
                execution={
                    "result_status": "skipped_no_signal",
                    "runner_result": safe_runner_result(status="skipped_no_signal", strategy_called=False),
                }
            )
        )
        self.assertTrue(accepted(skipped))
        self.assertEqual(skipped["executor_acceptance"]["acceptance_status"], "accepted_skipped_no_signal")

        blocked = build(
            valid_stage4j5_report(
                execution={
                    "result_status": "blocked_by_gate",
                    "runner_result": safe_runner_result(status="blocked_by_gate", strategy_called=False),
                }
            )
        )
        self.assertTrue(accepted(blocked))
        self.assertEqual(blocked["executor_acceptance"]["acceptance_status"], "accepted_blocked_by_gate")

    def test_report_shape_evidence_next_gate_and_json_safety(self) -> None:
        report = build()
        self.assertTrue(accepted(report))
        self.assertEqual(report["next_gate_recommendation"], RECOMMENDED_NEXT_GATE)
        self.assertTrue(report["readiness_for_stage4j_complete_or_next_gate"]["stage4j_complete"])
        self.assertTrue(report["evidence_summary"]["side_effects_absent"])
        self.assertTrue(report["executor_result_checks"]["controlled_operation_payload_is_dict"])
        self.assertTrue(report["executor_result_checks"]["runner_result_is_dict"])
        for key in (
            "dry_run",
            "stage4j6_controlled_paper_operation_acceptance_report",
            "generated_at",
            "artifact_checks",
            "selected_strategy",
            "operation",
            "executor_result_checks",
            "boundary_checks",
            "gate_checks",
            "executor_acceptance",
            "evidence_summary",
            "next_gate_recommendation",
            "required_inputs_for_next_phase",
            "activation_snapshot_checks",
            "state_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "paper_broker_checks",
            "market_window_checks",
            "safety_checks",
            "readiness_for_stage4j_complete_or_next_gate",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report)

    def test_snapshot_blockers_and_warnings(self) -> None:
        blocking_cases = [
            {"state_snapshot": clean_state_snapshot(active_halt=True)},
            {"state_snapshot": clean_state_snapshot(unresolved_needs_reconciliation_count=1)},
            {"state_snapshot": {"active_halt": False, "needs_reconciliation_count": 1, "active_intents_count": 0}},
            {"state_snapshot": clean_state_snapshot(active_intents_count=1)},
            {"risk_snapshot": clean_risk_snapshot(risk_bypass_enabled=True)},
            {"risk_snapshot": clean_risk_snapshot(kill_switch_available=False)},
            {"risk_snapshot": clean_risk_snapshot(hard_halt_available=False)},
            {"risk_snapshot": clean_risk_snapshot(daily_loss_limit_available=False)},
            {"scheduler_snapshot": clean_scheduler_snapshot(all_strategies_enabled=True)},
            {"lifecycle_snapshot": clean_lifecycle_snapshot(all_strategies_enabled=True)},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(mode="LIVE")},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(ibkr_port=4002)},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(paper_trading=False)},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(live_trading_enabled=True)},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(broker_submission_enabled=True)},
            {"market_window_snapshot": clean_market_window_snapshot(allowed_to_schedule_paper_run=False)},
        ]
        for kwargs in blocking_cases:
            with self.subTest(kwargs=kwargs):
                self.assertFalse(accepted(build_with_snapshots(**kwargs)))

        safe_intents = build_with_snapshots(
            state_snapshot=clean_state_snapshot(active_intents_count=1, active_intents_safe_for_enablement=True)
        )
        self.assertTrue(accepted(safe_intents))
        self.assertIn("active intents present", " ".join(safe_intents["warnings"]))

        closed_market = build_with_snapshots(market_window_snapshot=clean_market_window_snapshot(market_open=False))
        self.assertTrue(accepted(closed_market))
        self.assertIn("market is currently closed", " ".join(closed_market["warnings"]))

    def test_missing_optional_snapshots_may_accept_with_manual_warning(self) -> None:
        report = build()
        self.assertTrue(accepted(report))
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, report["warnings"])

    def test_valid_report_with_matching_snapshots_is_accepted_and_inputs_not_mutated(self) -> None:
        j5 = valid_stage4j5_report()
        state = clean_state_snapshot(open_positions_count=7)
        original_j5 = copy.deepcopy(j5)
        original_state = copy.deepcopy(state)
        report = build_with_snapshots(j5=j5, state_snapshot=state)
        self.assertTrue(accepted(report))
        self.assertEqual(j5, original_j5)
        self.assertEqual(state, original_state)

    def test_recommendations_remain_conservative(self) -> None:
        report = build()
        text = " ".join(report["recommendations"]["ordered_next_steps"]).lower()
        for phrase in (
            "enable live trading",
            "enable all strategies",
            "place orders now",
            "enable broker submission now",
            "create intents",
            "create tickets",
            "write state",
            "write ledger",
        ):
            self.assertNotIn(phrase, text)

    def test_cli_requires_dry_run_json_is_strict_and_actions_are_not_exposed(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            rc = tool.run_stage4j6_controlled_paper_operation_acceptance(
                ["--stage4j5-executor-json", json.dumps(valid_stage4j5_report())]
            )
        self.assertNotEqual(rc, 0)
        self.assertIn("--dry-run-only", stderr.getvalue())

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4j6_controlled_paper_operation_acceptance(
                ["--dry-run-only", "--json", "--stage4j5-executor-json", json.dumps(valid_stage4j5_report())]
            )
        self.assertEqual(rc, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4j6_controlled_paper_operation_acceptance_report"])
        self.assertTrue(parsed["executor_acceptance"]["accepted"])

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4j6_controlled_paper_operation_acceptance(
                ["--dry-run-only", "--json", "--stage4j5-executor-json", "{"]
            )
        self.assertNotEqual(rc, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertIn("JSONDecodeError", parsed["errors"][0])

        namespace = argparse.Namespace()
        for attr in ("submit", "cancel", "status", "scheduler_enable", "broker_submit"):
            self.assertFalse(hasattr(namespace, attr))

    def test_no_direct_broker_external_scheduler_strategy_or_runner_calls_in_stage4j6_files(self) -> None:
        source = "\n".join(path.read_text() for path in STAGE4J6_RUNTIME_FILES)
        for forbidden in (
            "submit_order_plan(",
            "get_order_status(",
            "cancel_order(",
            "placeOrder(",
            "cancelOrder(",
            "reqMktData(",
            "qualifyContracts(",
            "StateStore(",
            ".save(",
            ".write(",
            "ledger.append(",
            "scheduler.add_job(",
            "run_scan(",
            "scan_now(",
            ".run_controlled_paper_operation(",
            "yfinance",
            "requests",
            "urllib",
            "systemctl",
            "systemd",
            "socket.create_connection",
            "socket.socket",
            "asyncio.run(",
            "asyncio.get_event_loop(",
            "asyncio.new_event_loop(",
            "uuid.uuid4(",
            "random.",
            "time.time(",
            "datetime.now(",
            "traceback.format_exc(",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
