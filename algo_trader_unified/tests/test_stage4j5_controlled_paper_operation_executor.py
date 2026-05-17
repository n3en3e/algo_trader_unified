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

from algo_trader_unified.core.stage4j5_controlled_paper_operation_executor import (
    MARKET_WINDOW_MANUAL_WARNING,
    build_stage4j5_controlled_paper_operation_executor_report,
)
from algo_trader_unified.tools import stage4j5_controlled_paper_operation_executor as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4J5_FILES = [
    ROOT / "core/stage4j5_controlled_paper_operation_executor.py",
    ROOT / "tools/stage4j5_controlled_paper_operation_executor.py",
]


def valid_stage4j4_report(**overrides: object) -> dict:
    operation_id = "s01_once_2026_05_16"
    report: dict[str, object] = {
        "dry_run": True,
        "stage4j4_controlled_paper_operation_execution_gate_report": True,
        "generated_at": "2026-05-16T14:00:00+00:00",
        "artifact_checks": {
            "stage4j3_report_present": True,
            "stage4j3_report_ready": True,
            "selected_strategy_present": True,
            "operation_id_present": True,
            "dry_run_trace_present": True,
            "dry_run_trace_clean": True,
        },
        "selected_strategy": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
        },
        "operation": {
            "operation_id": operation_id,
            "operation_scope": "single_strategy_controlled_scheduled_paper_operation",
            "paper_only": True,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
        },
        "execution_gate": {
            "available": True,
            "selected_strategy_id": "S01",
            "operation_id": operation_id,
            "ready_for_4J5": True,
        },
        "proposed_execution_permissions_for_4J5": {
            "selected_strategy_id": "S01",
            "operation_id": operation_id,
            "permission_scope": "single_strategy_controlled_paper_operation_executor",
            "paper_only": True,
            "one_strategy_only": True,
            "may_build_executor_next_phase": True,
            "may_call_strategy_next_phase": True,
            "may_fetch_market_data_next_phase": False,
            "may_qualify_contracts_next_phase": False,
            "may_create_intent_next_phase": False,
            "may_create_ticket_next_phase": False,
            "may_submit_order_next_phase": False,
            "may_write_state_next_phase": False,
            "may_write_ledger_next_phase": False,
            "live_trading_enabled": False,
            "all_strategies_enabled": False,
            "broker_submission_enabled": False,
        },
        "safety_checks": {
            "no_live_trading": True,
            "no_all_strategy_enablement": True,
            "no_broker_submission_enabled": True,
            "no_market_data": True,
            "no_contract_qualification": True,
            "no_order_submission": True,
            "no_strategy_scan_execution": True,
            "no_lifecycle_transition_execution": True,
            "no_state_write": True,
            "no_ledger_write": True,
        },
        "readiness_for_stage4j5": {
            "ready_to_build_controlled_paper_operation_executor": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


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
        "result_summary": "report only",
        "warnings": [],
        "errors": [],
    }
    _deep_update(result, overrides)
    return result


_DEFAULT_RESULT = object()


class FakeRunner:
    def __init__(self, result: object | None = _DEFAULT_RESULT, *, raises: Exception | None = None) -> None:
        self.result = safe_runner_result() if result is _DEFAULT_RESULT else result
        self.raises = raises
        self.calls: list[dict] = []

    def run_controlled_paper_operation(self, payload: dict) -> object:
        self.calls.append(copy.deepcopy(payload))
        if self.raises is not None:
            raise self.raises
        return self.result


class MissingMethodRunner:
    pass


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


_DEFAULT_J4 = object()
_DEFAULT_RUNNER = object()


def build(j4: object = _DEFAULT_J4, runner: object | None = _DEFAULT_RUNNER, **kwargs: object) -> dict:
    if j4 is _DEFAULT_J4:
        j4 = valid_stage4j4_report()
    if j4 is None:
        stage4j4_report = None
    else:
        stage4j4_report = j4
    if runner is _DEFAULT_RUNNER:
        runner = FakeRunner()
    return build_stage4j5_controlled_paper_operation_executor_report(
        stage4j4_execution_gate_report=stage4j4_report,  # type: ignore[arg-type]
        selected_strategy_operation_runner=runner,
        allow_controlled_paper_operation_execution=True,
        now_provider=lambda: datetime(2026, 5, 16, 15, 0, tzinfo=timezone.utc),
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


def ready(report: dict) -> bool:
    return report["readiness_for_stage4j6"]["ready_to_build_controlled_paper_operation_acceptance"]


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class Stage4J5ControlledPaperOperationExecutorTests(unittest.TestCase):
    def test_stage4j4_gating_and_allow_flag_block_runner_call(self) -> None:
        cases = [
            None,
            valid_stage4j4_report(readiness_for_stage4j5={"ready_to_build_controlled_paper_operation_executor": False}),
            valid_stage4j4_report(selected_strategy=None),
            valid_stage4j4_report(operation={"operation_id": None}),
        ]
        for j4 in cases:
            runner = FakeRunner()
            report = build_stage4j5_controlled_paper_operation_executor_report(
                stage4j4_execution_gate_report=j4,  # type: ignore[arg-type]
                selected_strategy_operation_runner=runner,
                allow_controlled_paper_operation_execution=True,
            )
            self.assertFalse(ready(report))
            self.assertEqual(runner.calls, [])

        runner = FakeRunner()
        blocked = build_stage4j5_controlled_paper_operation_executor_report(
            stage4j4_execution_gate_report=valid_stage4j4_report(),
            selected_strategy_operation_runner=runner,
            allow_controlled_paper_operation_execution=False,
        )
        self.assertFalse(ready(blocked))
        self.assertEqual(runner.calls, [])

    def test_missing_runner_blocks_safely(self) -> None:
        self.assertFalse(ready(build(runner=None)))
        self.assertFalse(ready(build(runner=MissingMethodRunner())))

    def test_valid_gate_calls_runner_once_with_exact_payload(self) -> None:
        runner = FakeRunner()
        report = build(runner=runner)
        self.assertTrue(ready(report))
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(runner.calls[0], report["controlled_operation_payload"])
        self.assertNotIn("stage4j4_controlled_paper_operation_execution_gate_report", runner.calls[0])
        self.assertNotIn("state_snapshot", runner.calls[0])
        payload = report["controlled_operation_payload"]
        self.assertIsInstance(payload, dict)
        json.dumps(payload)
        for key in (
            "allow_market_data",
            "allow_contract_qualification",
            "allow_intent_creation",
            "allow_ticket_creation",
            "allow_order_submission",
            "allow_state_write",
            "allow_ledger_write",
            "live_trading_enabled",
            "all_strategies_enabled",
            "broker_submission_enabled",
        ):
            self.assertFalse(payload[key])

    def test_runner_exception_and_none_output_are_safe(self) -> None:
        failure = build(runner=FakeRunner(raises=RuntimeError("boom")))
        self.assertFalse(ready(failure))
        self.assertEqual(failure["execution"]["failure_reason"], "RuntimeError: boom")
        json.dumps(failure)

        none_output = build(runner=FakeRunner(result=None))
        self.assertFalse(ready(none_output))
        self.assertEqual(none_output["execution"]["runner_result"], {})
        self.assertFalse(none_output["runner_output_checks"]["runner_result_present"])
        json.dumps(none_output)

    def test_runner_output_validation_blocks_unsafe_values(self) -> None:
        cases = [
            ("not dict", "runner output must be a dict"),
            ({"status": "bad"}, "runner status is unsupported"),
            (safe_runner_result(selected_strategy_id="S02"), "runner selected_strategy_id does not match"),
            (safe_runner_result(operation_id="other"), "runner operation_id does not match"),
            (safe_runner_result(market_data_requested=True), "market_data_requested"),
            (safe_runner_result(contracts_qualified=True), "contracts_qualified"),
            (safe_runner_result(intents_created=True), "intents_created"),
            (safe_runner_result(tickets_created=True), "tickets_created"),
            (safe_runner_result(orders_submitted=True), "orders_submitted"),
            (safe_runner_result(state_written=True), "critical failure"),
            (safe_runner_result(ledger_written=True), "critical failure"),
            (safe_runner_result(live_trading_enabled=True), "live_trading_enabled"),
            (safe_runner_result(all_strategies_enabled=True), "all_strategies_enabled"),
            (safe_runner_result(broker_submission_enabled=True), "broker_submission_enabled"),
            (safe_runner_result(market_data_requested="False"), "market_data_requested"),
            (safe_runner_result(result_summary=Decimal("1.2")), "JSON-safe"),
        ]
        for result, expected in cases:
            with self.subTest(result=result):
                report = build(runner=FakeRunner(result=result))
                self.assertFalse(ready(report))
                self.assertIn(expected, " ".join(report["readiness_for_stage4j6"]["blockers"]))
                json.dumps(report)

    def test_safe_completed_skipped_and_blocked_statuses_may_pass(self) -> None:
        self.assertTrue(ready(build(runner=FakeRunner(result=safe_runner_result()))))
        skipped = build(runner=FakeRunner(result=safe_runner_result(status="skipped_no_signal", strategy_called=False)))
        self.assertTrue(ready(skipped))
        self.assertIn("skipped", " ".join(skipped["warnings"]))
        blocked = build(runner=FakeRunner(result=safe_runner_result(status="blocked_by_gate", strategy_called=False)))
        self.assertTrue(ready(blocked))
        self.assertIn("blocked internally", " ".join(blocked["warnings"]))

    def test_permission_blockers_prevent_runner_call(self) -> None:
        permission_cases = [
            {"may_fetch_market_data_next_phase": True},
            {"may_qualify_contracts_next_phase": True},
            {"may_submit_order_next_phase": True},
            {"may_write_state_next_phase": True},
        ]
        for mutation in permission_cases:
            report_input = valid_stage4j4_report(proposed_execution_permissions_for_4J5=mutation)
            runner = FakeRunner()
            report = build(j4=report_input, runner=runner)
            self.assertFalse(ready(report))
            self.assertEqual(runner.calls, [])

    def test_state_risk_scheduler_lifecycle_broker_and_market_blockers(self) -> None:
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
                runner = FakeRunner()
                report = build_with_snapshots(runner=runner, **kwargs)
                self.assertFalse(ready(report))
                self.assertEqual(runner.calls, [])

        safe_intents = build_with_snapshots(
            state_snapshot=clean_state_snapshot(active_intents_count=1, active_intents_safe_for_enablement=True)
        )
        self.assertTrue(ready(safe_intents))
        self.assertIn("active intents present", " ".join(safe_intents["warnings"]))

        closed_market = build_with_snapshots(market_window_snapshot=clean_market_window_snapshot(market_open=False))
        self.assertTrue(ready(closed_market))
        self.assertIn("market is currently closed", " ".join(closed_market["warnings"]))

    def test_missing_optional_snapshots_may_be_ready_with_manual_warning_and_report_shape(self) -> None:
        report = build()
        self.assertTrue(ready(report))
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, report["warnings"])
        for key in (
            "dry_run",
            "stage4j5_controlled_paper_operation_executor_report",
            "generated_at",
            "artifact_checks",
            "selected_strategy",
            "operation",
            "controlled_operation_payload",
            "gate_checks",
            "execution",
            "runner_output_checks",
            "activation_snapshot_checks",
            "state_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "paper_broker_checks",
            "market_window_checks",
            "safety_checks",
            "readiness_for_stage4j6",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report)

    def test_inputs_not_mutated_and_recommendations_stay_conservative(self) -> None:
        report_input = valid_stage4j4_report()
        snapshot = clean_state_snapshot(open_positions_count=7)
        original_report = copy.deepcopy(report_input)
        original_snapshot = copy.deepcopy(snapshot)
        report = build(j4=report_input, state_snapshot=snapshot)
        self.assertEqual(report_input, original_report)
        self.assertEqual(snapshot, original_snapshot)

        text = " ".join(report["recommendations"]["ordered_next_steps"])
        for phrase in (
            "enable live trading",
            "enable all strategies",
            "place orders now",
            "enable market data now",
            "enable contract qualification now",
            "create intents",
            "create tickets",
            "write state",
            "write ledger",
            "enable broker submission broadly",
        ):
            self.assertNotIn(phrase, text.lower())

    def test_cli_requires_safety_flags_json_is_strict_and_actions_are_not_exposed(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            rc = tool.run_stage4j5_controlled_paper_operation_executor(
                ["--stage4j4-gate-json", json.dumps(valid_stage4j4_report())]
            )
        self.assertNotEqual(rc, 0)
        self.assertIn("--dry-run-only", stderr.getvalue())

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4j5_controlled_paper_operation_executor(
                [
                    "--dry-run-only",
                    "--json",
                    "--stage4j4-gate-json",
                    json.dumps(valid_stage4j4_report()),
                ]
            )
        self.assertNotEqual(rc, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertFalse(parsed["success"])
        self.assertIn("--allow-controlled-paper-operation-execution", parsed["errors"][0])

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4j5_controlled_paper_operation_executor(
                [
                    "--dry-run-only",
                    "--json",
                    "--allow-controlled-paper-operation-execution",
                    "--stage4j4-gate-json",
                    json.dumps(valid_stage4j4_report()),
                ]
            )
        self.assertNotEqual(rc, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4j5_controlled_paper_operation_executor_report"])
        self.assertFalse(parsed["execution"]["runner_called"])

        namespace = argparse.Namespace()
        for attr in ("submit", "cancel", "status", "market_data", "qualification", "broker_submit"):
            self.assertFalse(hasattr(namespace, attr))

    def test_no_direct_broker_external_scheduler_or_strategy_calls_in_stage4j5_files(self) -> None:
        source = "\n".join(path.read_text() for path in STAGE4J5_FILES)
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
