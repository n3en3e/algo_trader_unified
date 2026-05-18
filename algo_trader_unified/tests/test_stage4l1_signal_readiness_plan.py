from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4l1_signal_readiness_plan import (
    MARKET_WINDOW_MANUAL_WARNING,
    build_stage4l1_signal_readiness_plan_report,
)
from algo_trader_unified.tools import stage4l1_signal_readiness_plan as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4L1_RUNTIME_FILES = [
    ROOT / "core/stage4l1_signal_readiness_plan.py",
    ROOT / "tools/stage4l1_signal_readiness_plan.py",
]


def valid_stage4k6_report(*, include_md: bool = True, include_cq: bool = True, **overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4k6_market_data_contract_acceptance_report": True,
        "generated_at": "2026-05-17T18:00:00+00:00",
        "selected_strategy": {"selected_strategy_id": "S01", "paper_only": True, "one_strategy_only": True},
        "operation": {
            "operation_id": "s01_once_2026_05_16",
            "operation_scope": "single_strategy_market_data_contract_acceptance",
            "paper_only": True,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
        },
        "provider_result_acceptance": {"accepted": True},
        "operation_audit": {"operation_audit_passed": True},
        "accepted_market_data_outputs": {
            "available": include_md,
            "selected_strategy_id": "S01",
            "operation_id": "s01_once_2026_05_16",
            "accepted_result_count": 1 if include_md else 0,
            "accepted_results": [_accepted_result("market_data")] if include_md else [],
            "read_only_for_future_stages": True,
        },
        "accepted_contract_qualification_outputs": {
            "available": include_cq,
            "selected_strategy_id": "S01",
            "operation_id": "s01_once_2026_05_16",
            "accepted_result_count": 1 if include_cq else 0,
            "accepted_results": [_accepted_result("contract_qualification")] if include_cq else [],
            "read_only_for_future_stages": True,
        },
        "boundary_checks": {
            "no_provider_called_in_4k6": True,
            "no_direct_ib_call": True,
            "no_direct_reqMktData": True,
            "no_direct_qualifyContracts": True,
            "no_direct_reqContractDetails": True,
            "no_strategy_scan": True,
            "no_intents_created": True,
            "no_tickets_created": True,
            "no_orders_submitted": True,
            "no_broker_submission": True,
            "no_state_written": True,
            "no_ledger_written": True,
            "no_live_trading": True,
            "no_all_strategy_enablement": True,
            "no_scheduler_registration": True,
            "no_lifecycle_execution": True,
        },
        "safety_checks": {
            "no_live_trading": True,
            "no_all_strategy_enablement": True,
            "no_broker_submission_enabled": True,
            "no_direct_market_data": True,
            "no_direct_contract_qualification": True,
            "no_strategy_scan": True,
            "no_signal_execution": True,
            "no_order_submission": True,
            "no_intent_creation": True,
            "no_ticket_creation": True,
            "no_state_write": True,
            "no_ledger_write": True,
        },
        "readiness_for_next_phase": {
            "stage4k_complete": True,
            "ready_to_proceed_after_stage4k": True,
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def _accepted_result(provider_type: str, **overrides: object) -> dict:
    result: dict[str, object] = {
        "provider_type": provider_type,
        "symbol": "SPY",
        "payload_id": f"{provider_type}-001",
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
        "order_submission_enabled": False,
        "state_write_enabled": False,
        "ledger_write_enabled": False,
        "direct_ib_call_made": False,
        "reqMktData_called": False,
        "qualifyContracts_called": False,
        "reqContractDetails_called": False,
    }
    _deep_update(result, overrides)
    return result


def clean_state_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"active_halt": False, "unresolved_needs_reconciliation_count": 0, "active_intents_count": 0, "open_positions_count": 3}
    _deep_update(snapshot, overrides)
    return snapshot


def clean_risk_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"kill_switch_available": True, "hard_halt_available": True, "daily_loss_limit_available": True, "max_position_limit_available": True, "risk_bypass_enabled": False}
    _deep_update(snapshot, overrides)
    return snapshot


def clean_scheduler_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"jobs": [{"strategy_id": "S01", "broker_submission_enabled": False, "live_trading_enabled": False, "all_strategies_enabled": False, "strategy_scan_execution_enabled": False}]}
    _deep_update(snapshot, overrides)
    return snapshot


def clean_lifecycle_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"lifecycle_automation_enabled": True, "selected_strategy_id": "S01", "broker_submission_enabled": False, "live_trading_enabled": False, "all_strategies_enabled": False, "lifecycle_transition_execution_enabled": False}
    _deep_update(snapshot, overrides)
    return snapshot


def clean_paper_broker_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"mode": "PAPER", "paper_trading": True, "ibkr_port": 4004, "live_trading_enabled": False, "broker_submission_enabled": False}
    _deep_update(snapshot, overrides)
    return snapshot


def clean_market_window_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"allowed_to_schedule_paper_run": True, "is_trading_day": True, "market_open": True}
    _deep_update(snapshot, overrides)
    return snapshot


_DEFAULT_REPORT = object()


def build(report: object = _DEFAULT_REPORT, **kwargs: object) -> dict:
    return build_stage4l1_signal_readiness_plan_report(
        stage4k6_acceptance_report=valid_stage4k6_report() if report is _DEFAULT_REPORT else report,  # type: ignore[arg-type]
        now_provider=lambda: datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc),
        **kwargs,
    )


def build_with_snapshots(report: object = _DEFAULT_REPORT, **kwargs: object) -> dict:
    defaults: dict[str, object] = {
        "strategy_registry_snapshot": ["S01"],
        "signal_schema_snapshot": {"selected_strategy_id": "S01", "expected_input_sections": ["selected_strategy_id", "operation_id"], "requires_market_data": False, "requires_contract_qualification": False},
        "state_snapshot": clean_state_snapshot(),
        "risk_snapshot": clean_risk_snapshot(),
        "scheduler_snapshot": clean_scheduler_snapshot(),
        "lifecycle_snapshot": clean_lifecycle_snapshot(),
        "paper_broker_snapshot": clean_paper_broker_snapshot(),
        "market_window_snapshot": clean_market_window_snapshot(),
    }
    defaults.update(kwargs)
    return build(valid_stage4k6_report() if report is _DEFAULT_REPORT else report, **defaults)


def ready(report: dict) -> bool:
    return report["readiness_for_stage4l2"]["ready_to_build_signal_readiness_gate"]


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class Stage4L1SignalReadinessPlanTests(unittest.TestCase):
    def test_valid_reports_with_market_data_qualification_or_both_are_ready_for_4l2(self) -> None:
        cases = [
            valid_stage4k6_report(include_md=True, include_cq=False),
            valid_stage4k6_report(include_md=False, include_cq=True),
            valid_stage4k6_report(include_md=True, include_cq=True),
        ]
        for source in cases:
            with self.subTest(md=source["accepted_market_data_outputs"]["available"], cq=source["accepted_contract_qualification_outputs"]["available"]):  # type: ignore[index]
                report = build_with_snapshots(source)
                self.assertTrue(ready(report))
                self.assertTrue(report["success"])
                self.assertEqual(report["readiness_for_stage4l2"]["next_recommended_phase"], "Stage 4L-2 signal readiness gate")
                self.assertNotIn("full paper trading is active", json.dumps(report).lower())
                json.dumps(report)

    def test_missing_or_not_complete_4k6_blocks_readiness(self) -> None:
        cases = [
            None,
            valid_stage4k6_report(stage4k6_market_data_contract_acceptance_report=False),
            valid_stage4k6_report(success=False),
            valid_stage4k6_report(readiness_for_next_phase={"stage4k_complete": False, "ready_to_proceed_after_stage4k": True}),
            valid_stage4k6_report(readiness_for_next_phase={"stage4k_complete": True, "ready_to_proceed_after_stage4k": False}),
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertFalse(ready(build_with_snapshots(source)))

    def test_selected_strategy_operation_and_accepted_output_artifacts_are_required(self) -> None:
        cases = [
            valid_stage4k6_report(selected_strategy={"selected_strategy_id": "", "paper_only": True, "one_strategy_only": True}),
            valid_stage4k6_report(operation={"operation_id": ""}),
            valid_stage4k6_report(accepted_market_data_outputs=None),
            valid_stage4k6_report(accepted_contract_qualification_outputs=None),
            valid_stage4k6_report(include_md=False, include_cq=False),
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertFalse(ready(build_with_snapshots(source)))

    def test_accepted_output_validation_blocks_malformed_or_unsafe_results(self) -> None:
        unsafe_cases = [
            {"read_only_for_future_stages": False},
            {"accepted_results": "not-list"},
            {"accepted_results": [None]},
            {"accepted_results": [_accepted_result("market_data", value={1, 2})]},
            {"accepted_results": [_accepted_result("market_data", live_trading_enabled=True)]},
            {"accepted_results": [_accepted_result("market_data", broker_submission_enabled=True)]},
            {"accepted_results": [_accepted_result("market_data", order_submitted=True)]},
            {"accepted_results": [_accepted_result("market_data", state_written=True)]},
            {"accepted_results": [_accepted_result("market_data", ledger_written=True)]},
            {"accepted_results": [_accepted_result("market_data", direct_ib_call_made=True)]},
            {"accepted_results": [_accepted_result("market_data", reqMktData_called=True)]},
            {"accepted_results": [_accepted_result("market_data", qualifyContracts_called=True)]},
            {"accepted_results": [_accepted_result("market_data", reqContractDetails_called=True)]},
            {"accepted_results": [_accepted_result("market_data", live_trading_enabled="False")]},
            {"accepted_results": [_accepted_result("market_data", failure_reason="bad\nline")]},
            {"accepted_results": [_accepted_result("market_data", skipped_reason="<object at 0xABCDEF>")]},
        ]
        for override in unsafe_cases:
            with self.subTest(override=override):
                source = valid_stage4k6_report(include_md=True, include_cq=False)
                _deep_update(source["accepted_market_data_outputs"], override)  # type: ignore[arg-type,index]
                self.assertFalse(ready(build_with_snapshots(source)))

    def test_accepted_results_none_defaults_empty_and_blocks_only_when_no_other_category_exists(self) -> None:
        source = valid_stage4k6_report(include_md=True, include_cq=True)
        source["accepted_market_data_outputs"]["accepted_results"] = None  # type: ignore[index]
        self.assertTrue(ready(build_with_snapshots(source)))
        source = valid_stage4k6_report(include_md=True, include_cq=False)
        source["accepted_market_data_outputs"]["accepted_results"] = None  # type: ignore[index]
        self.assertFalse(ready(build_with_snapshots(source)))

    def test_provider_acceptance_operation_audit_and_4k_boundary_checks_block(self) -> None:
        cases = [
            valid_stage4k6_report(provider_result_acceptance={"accepted": False}),
            valid_stage4k6_report(operation_audit={"operation_audit_passed": False}),
            valid_stage4k6_report(boundary_checks={"no_direct_ib_call": False}),
            valid_stage4k6_report(boundary_checks={"no_strategy_scan": False}),
            valid_stage4k6_report(boundary_checks={"no_broker_submission": False}),
            valid_stage4k6_report(safety_checks={"no_live_trading": False}),
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertFalse(ready(build_with_snapshots(source)))

    def test_strategy_registry_shapes_allow_or_block_selected_strategy(self) -> None:
        self.assertTrue(ready(build_with_snapshots(strategy_registry_snapshot=["S01", "S02"])))
        self.assertFalse(ready(build_with_snapshots(strategy_registry_snapshot=["S02"])))
        self.assertTrue(ready(build_with_snapshots(strategy_registry_snapshot=[{"strategy_id": "S01", "paper_eligible": True}])))
        self.assertFalse(ready(build_with_snapshots(strategy_registry_snapshot=[{"strategy_id": "S01", "paper_eligible": False}])))
        self.assertFalse(ready(build_with_snapshots(strategy_registry_snapshot={"all_strategies_enabled": True, "candidate_strategy_ids": ["S01"]})))
        malformed = build_with_snapshots(strategy_registry_snapshot={"strategies": "bad"})
        self.assertTrue(ready(malformed))
        self.assertTrue(malformed["warnings"])

    def test_signal_schema_rules_are_type_safe_and_block_unsafe_permissions(self) -> None:
        cases = [
            {"requires_market_data": True},
            {"requires_contract_qualification": True},
            {"expected_input_sections": "selected_strategy_id"},
            {"expected_input_sections": ["selected_strategy_id", 7]},
            {"expected_input_sections": ["missing_section"]},
            {"allow_strategy_scan": True},
            {"allow_intent_creation": True},
            {"allow_order_submission": True},
            {"live_trading_enabled": True},
            {"broker_submission_enabled": True},
        ]
        sources = [
            valid_stage4k6_report(include_md=False, include_cq=True),
            valid_stage4k6_report(include_md=True, include_cq=False),
            valid_stage4k6_report(),
            valid_stage4k6_report(),
            valid_stage4k6_report(),
            valid_stage4k6_report(),
            valid_stage4k6_report(),
            valid_stage4k6_report(),
            valid_stage4k6_report(),
            valid_stage4k6_report(),
        ]
        for schema, source in zip(cases, sources):
            with self.subTest(schema=schema):
                self.assertFalse(ready(build_with_snapshots(source, signal_schema_snapshot=schema)))
        self.assertTrue(ready(build_with_snapshots(signal_schema_snapshot={"expected_input_sections": ["selected_strategy_id", "operation_id", "accepted_market_data_results", "accepted_contract_qualification_results"]})))
        self.assertFalse(ready(build_with_snapshots(signal_schema_snapshot=[])))  # type: ignore[arg-type]

    def test_proposed_inputs_and_flow_are_deterministic_non_executing_and_json_safe(self) -> None:
        report = build_with_snapshots()
        inputs = report["proposed_signal_readiness_inputs"]
        self.assertIsInstance(inputs, dict)
        self.assertTrue(inputs["read_only"])
        self.assertTrue(inputs["accepted_market_data_results"])
        self.assertTrue(inputs["accepted_contract_qualification_results"])
        for key in (
            "allow_strategy_scan",
            "allow_signal_execution",
            "allow_intent_creation",
            "allow_ticket_creation",
            "allow_order_submission",
            "allow_broker_submission",
            "allow_state_write",
            "allow_ledger_write",
            "live_trading_enabled",
            "all_strategies_enabled",
        ):
            self.assertIs(inputs[key], False)
        flow = report["proposed_4l2_validation_flow"]
        self.assertEqual([item["sequence_number"] for item in flow], list(range(1, 8)))
        self.assertEqual(len(flow), 7)
        for item in flow:
            self.assertFalse(item["would_execute_strategy_now"])
            self.assertFalse(item["would_calculate_signal_now"])
            self.assertFalse(item["would_create_intent_now"])
            self.assertFalse(item["would_create_ticket_now"])
            self.assertFalse(item["would_submit_order_now"])
            self.assertFalse(item["would_write_state_now"])
            self.assertFalse(item["would_write_ledger_now"])
            self.assertFalse(item["live_trading_enabled"])
            self.assertFalse(item["broker_submission_enabled"])
        json.dumps(report)

    def test_stale_4j_keys_block_readiness(self) -> None:
        stale_keys = [
            "proposed_execution_permissions_for_4J5",
            "may_call_strategy_next_phase",
            "may_build_executor_next_phase",
            "may_fetch_market_data_next_phase",
        ]
        for key in stale_keys:
            with self.subTest(key=key):
                source = valid_stage4k6_report()
                source["accepted_market_data_outputs"][key] = False  # type: ignore[index]
                self.assertFalse(ready(build_with_snapshots(source)))

    def test_state_risk_scheduler_lifecycle_broker_and_market_window_blockers(self) -> None:
        cases = [
            {"state_snapshot": clean_state_snapshot(active_halt=True)},
            {"state_snapshot": clean_state_snapshot(unresolved_needs_reconciliation_count=None, needs_reconciliation_count=1)},
            {"state_snapshot": clean_state_snapshot(active_intents_count=1)},
            {"scheduler_snapshot": {"all_strategy_scheduler_enabled": True}},
            {"lifecycle_snapshot": {"all_strategy_lifecycle_enabled": True}},
            {"risk_snapshot": clean_risk_snapshot(risk_bypass_enabled=True)},
            {"risk_snapshot": {"kill_switch_available": False, "hard_halt_available": True, "daily_loss_limit_available": True}},
            {"risk_snapshot": {"kill_switch_available": True, "hard_halt_available": False, "daily_loss_limit_available": True}},
            {"risk_snapshot": {"kill_switch_available": True, "hard_halt_available": True, "daily_loss_limit_available": False}},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(mode="LIVE")},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(ibkr_port=4002)},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(paper_trading=False)},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(live_trading_enabled=True)},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(broker_submission_enabled=True)},
            {"market_window_snapshot": clean_market_window_snapshot(allowed_to_schedule_paper_run=False)},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                self.assertFalse(ready(build_with_snapshots(**kwargs)))
        safe_intents = build_with_snapshots(state_snapshot=clean_state_snapshot(active_intents_count=1, active_intents_safe_for_enablement=True))
        self.assertTrue(ready(safe_intents))
        self.assertTrue(safe_intents["warnings"])
        closed = build_with_snapshots(market_window_snapshot=clean_market_window_snapshot(market_open=False, is_trading_day=False))
        self.assertTrue(ready(closed))
        self.assertTrue(closed["warnings"])

    def test_missing_market_window_warns_but_does_not_block_clean_4k6(self) -> None:
        report = build(strategy_registry_snapshot=["S01"], signal_schema_snapshot={"expected_input_sections": ["selected_strategy_id"]})
        self.assertTrue(ready(report))
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, report["warnings"])

    def test_report_top_level_fields_recommendations_and_no_mutation(self) -> None:
        source = valid_stage4k6_report()
        original = copy.deepcopy(source)
        report = build_with_snapshots(source)
        self.assertEqual(source, original)
        for key in (
            "dry_run",
            "stage4l1_signal_readiness_plan_report",
            "artifact_checks",
            "selected_strategy",
            "operation",
            "stage4k_completion_checks",
            "accepted_output_checks",
            "strategy_registry_checks",
            "signal_schema_checks",
            "proposed_signal_readiness_inputs",
            "proposed_4l2_validation_flow",
            "boundary_checks",
            "required_inputs_for_4l2",
            "activation_snapshot_checks",
            "state_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "paper_broker_checks",
            "market_window_checks",
            "safety_checks",
            "readiness_for_stage4l2",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        recommendation_text = json.dumps(report["recommendations"]).lower()
        self.assertIn("build stage 4l-2 signal readiness gate", recommendation_text)
        self.assertIn("do not enable live trading", recommendation_text)
        self.assertIn("do not place orders now", recommendation_text)
        self.assertNotIn("paper trading is fully active", recommendation_text)

    def test_cli_requires_dry_run_before_parsing_and_json_stdout_is_strict_json(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = tool.run_stage4l1_signal_readiness_plan(["--json", "--stage4k6-acceptance-json", "{bad"])
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("--dry-run-only", stderr.getvalue())

        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = tool.run_stage4l1_signal_readiness_plan(
                [
                    "--dry-run-only",
                    "--json",
                    "--stage4k6-acceptance-json",
                    json.dumps(valid_stage4k6_report(include_md=True, include_cq=False)),
                    "--strategy-registry-snapshot-json",
                    json.dumps(["S01"]),
                    "--signal-schema-snapshot-json",
                    json.dumps({"expected_input_sections": ["selected_strategy_id"]}),
                ]
            )
        self.assertEqual(code, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4l1_signal_readiness_plan_report"])

    def test_stage4l1_files_do_not_expose_or_call_forbidden_runtime_paths(self) -> None:
        forbidden_call_patterns = [
            "submit_order_plan",
            "get_order_status",
            "cancel_order",
            "placeOrder",
            "cancelOrder",
            "reqMktData",
            "qualifyContracts",
            "reqContractDetails",
            "request_controlled_market_data",
            "qualify_controlled_contracts",
            "add_job",
            "start",
            "run_scan",
            "scan_now",
            "run_controlled_paper_operation",
            "yfinance",
            "requests",
            "urllib",
            "systemctl",
            "systemd",
            "socket",
            "asyncio.run",
            "asyncio.get_event_loop",
            "asyncio.new_event_loop",
            "uuid.uuid4",
            "random",
            "time.time",
            "datetime.now",
            "traceback.format_exc",
        ]
        for path in STAGE4L1_RUNTIME_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("ib_insync", text)
            self.assertNotIn("StateStore(", text)
            self.assertNotIn(".write(", text)
            for token in forbidden_call_patterns:
                self.assertNotIn(f"{token}(", text, token)
            for forbidden_cli_fragment in (
                "--submit",
                "--cancel",
                "--status",
                "--provider-execute",
                "--direct-market-data",
                "--direct-contract-qualification",
                "--strategy-scan",
                "--signal-execute",
                "--scheduler-enable",
                "--broker-submit",
            ):
                self.assertNotIn(forbidden_cli_fragment, text)


if __name__ == "__main__":
    unittest.main()
