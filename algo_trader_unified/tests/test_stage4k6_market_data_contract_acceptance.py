from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4k6_market_data_contract_acceptance import (
    MARKET_WINDOW_MANUAL_WARNING,
    build_stage4k6_market_data_contract_acceptance_report,
)
from algo_trader_unified.tools import stage4k6_market_data_contract_acceptance as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4K6_RUNTIME_FILES = [
    ROOT / "core/stage4k6_market_data_contract_acceptance.py",
    ROOT / "tools/stage4k6_market_data_contract_acceptance.py",
]


def valid_stage4k5_report(*, include_md: bool = True, include_cq: bool = True, **overrides: object) -> dict:
    trace: list[dict[str, object]] = []
    applied: list[dict[str, object]] = []
    if include_md:
        trace.append(_trace(1, "market_data", "request_controlled_market_data", "md-001-SPY", {"bid": 1.0, "ask": 1.1}))
        applied.append(_operation("market_data", "request_controlled_market_data", "md-001-SPY"))
    if include_cq:
        trace.append(_trace(len(trace) + 1, "contract_qualification", "qualify_controlled_contracts", "cq-001-SPY", {"qualified": True}))
        applied.append(_operation("contract_qualification", "qualify_controlled_contracts", "cq-001-SPY"))
    report: dict[str, object] = {
        "dry_run": False,
        "stage4k5_market_data_contract_executor_report": True,
        "generated_at": "2026-05-17T16:00:00+00:00",
        "artifact_checks": {
            "stage4k5_report_present": True,
            "stage4k5_report_ready": True,
            "selected_strategy_present": True,
            "operation_id_present": True,
            "executor_results_present": True,
            "provider_call_trace_present": True,
            "operation_lists_present": True,
        },
        "selected_strategy": {"selected_strategy_id": "S01", "paper_only": True, "one_strategy_only": True},
        "operation": {
            "operation_id": "s01_once_2026_05_16",
            "operation_scope": "single_strategy_controlled_market_data_contract_execution",
            "paper_only": True,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
        },
        "market_data_execution_results": {
            "attempted": include_md,
            "provider_called": include_md,
            "direct_ib_call_made": False,
            "reqMktData_called": False,
            "success": include_md,
        },
        "contract_qualification_execution_results": {
            "attempted": include_cq,
            "provider_called": include_cq,
            "direct_ib_call_made": False,
            "qualifyContracts_called": False,
            "reqContractDetails_called": False,
            "success": include_cq,
        },
        "provider_call_trace": trace,
        "applied_operations": applied,
        "failed_operations": [],
        "skipped_operations": [],
        "boundary_checks": {
            "no_direct_ib_call": True,
            "no_reqMktData": True,
            "no_qualifyContracts": True,
            "no_reqContractDetails": True,
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
            "controlled_market_data_provider_call_only_if_allowed": True,
            "controlled_contract_qualification_provider_call_only_if_allowed": True,
        },
        "safety_checks": {
            "no_live_trading": True,
            "no_all_strategy_enablement": True,
            "no_broker_submission_enabled": True,
            "no_direct_market_data": True,
            "no_direct_contract_qualification": True,
            "no_order_submission": True,
            "no_intent_creation": True,
            "no_ticket_creation": True,
            "no_state_write": True,
            "no_ledger_write": True,
        },
        "readiness_for_stage4k6": {"ready_to_build_market_data_contract_acceptance": True, "blockers": [], "warnings": []},
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def _trace(sequence: int, provider_type: str, method: str, payload_id: str, result: dict[str, object]) -> dict[str, object]:
    return {
        "sequence_number": sequence,
        "provider_type": provider_type,
        "provider_method": method,
        "payload_id": payload_id,
        "selected_strategy_id": "S01",
        "operation_id": "s01_once_2026_05_16",
        "input_payload": {"selected_strategy_id": "S01", "operation_id": "s01_once_2026_05_16", "payload_id": payload_id},
        "provider_called": True,
        "direct_ib_call_made": False,
        "success": True,
        "result": {
            **result,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
            "order_submission_enabled": False,
            "state_write_enabled": False,
            "ledger_write_enabled": False,
            "direct_ib_call_made": False,
            "reqMktData_called": False,
            "qualifyContracts_called": False,
            "reqContractDetails_called": False,
        },
        "failure_reason": None,
        "skipped_reason": None,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
        "order_submission_enabled": False,
        "state_write_enabled": False,
        "ledger_write_enabled": False,
    }


def _operation(provider_type: str, method: str, payload_id: str, status: str = "applied") -> dict[str, object]:
    return {
        "provider_type": provider_type,
        "provider_method": method,
        "payload_id": payload_id,
        "selected_strategy_id": "S01",
        "operation_id": "s01_once_2026_05_16",
        "target": "SPY",
        "status": status,
        "failure_reason": None,
        "skipped_reason": None,
    }


def clean_state_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"active_halt": False, "unresolved_needs_reconciliation_count": 0, "active_intents_count": 0, "open_positions_count": 7}
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
    if report is _DEFAULT_REPORT:
        report = valid_stage4k5_report()
    return build_stage4k6_market_data_contract_acceptance_report(
        stage4k5_executor_report=report,  # type: ignore[arg-type]
        now_provider=lambda: datetime(2026, 5, 17, 17, 0, tzinfo=timezone.utc),
        **kwargs,
    )


def build_with_snapshots(report: object = _DEFAULT_REPORT, **kwargs: object) -> dict:
    defaults: dict[str, object] = {
        "state_snapshot": clean_state_snapshot(),
        "risk_snapshot": clean_risk_snapshot(),
        "scheduler_snapshot": clean_scheduler_snapshot(),
        "lifecycle_snapshot": clean_lifecycle_snapshot(),
        "paper_broker_snapshot": clean_paper_broker_snapshot(),
        "market_window_snapshot": clean_market_window_snapshot(),
    }
    defaults.update(kwargs)
    return build(valid_stage4k5_report() if report is _DEFAULT_REPORT else report, **defaults)


def ready(report: dict) -> bool:
    return report["readiness_for_next_phase"]["ready_to_proceed_after_stage4k"]


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class Stage4K6MarketDataContractAcceptanceTests(unittest.TestCase):
    def test_valid_reports_complete_stage4k_for_market_data_qualification_or_both(self) -> None:
        for source in (
            valid_stage4k5_report(include_md=True, include_cq=False),
            valid_stage4k5_report(include_md=False, include_cq=True),
            valid_stage4k5_report(include_md=True, include_cq=True),
        ):
            with self.subTest(source=source["market_data_execution_results"]):
                report = build_with_snapshots(source)
                self.assertTrue(ready(report))
                self.assertTrue(report["readiness_for_next_phase"]["stage4k_complete"])
                self.assertEqual(report["readiness_for_next_phase"]["next_recommended_phase"], "Stage 4L signal/readiness integration")
                self.assertNotIn("full paper trading is active", json.dumps(report).lower())
                json.dumps(report)

    def test_missing_or_not_ready_4k5_artifacts_block_readiness(self) -> None:
        cases = [
            None,
            valid_stage4k5_report(stage4k5_market_data_contract_executor_report=False),
            valid_stage4k5_report(success=False),
            valid_stage4k5_report(readiness_for_stage4k6={"ready_to_build_market_data_contract_acceptance": False}),
            valid_stage4k5_report(selected_strategy={"selected_strategy_id": ""}),
            valid_stage4k5_report(operation={"operation_id": ""}),
            valid_stage4k5_report(market_data_execution_results=None),
            valid_stage4k5_report(contract_qualification_execution_results=None),
            valid_stage4k5_report(provider_call_trace=None),
            valid_stage4k5_report(provider_call_trace="bad"),
            valid_stage4k5_report(applied_operations=None),
            valid_stage4k5_report(failed_operations=None),
            valid_stage4k5_report(skipped_operations=None),
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertFalse(ready(build(case)))

    def test_provider_trace_validation_blocks_malformed_non_native_and_unsafe_values(self) -> None:
        bad_trace = _trace(1, "market_data", "request_controlled_market_data", "md-001-SPY", {"ok": True})
        mutations = [
            lambda item: item.pop("payload_id"),
            lambda item: item.update(selected_strategy_id="S02"),
            lambda item: item.update(operation_id="other"),
            lambda item: item.update(input_payload="bad"),
            lambda item: item.update(result=["bad"]),
            lambda item: item.update(provider_called="False"),
            lambda item: item.update(success="True"),
            lambda item: item.update(direct_ib_call_made=True),
            lambda item: item["result"].update(reqMktData_called="False"),  # type: ignore[index,union-attr]
            lambda item: item.update(failure_reason="bad\nreason"),
            lambda item: item.update(skipped_reason="<object at 0xabc123>"),
        ]
        for mutate in mutations:
            trace = copy.deepcopy(bad_trace)
            mutate(trace)
            report = valid_stage4k5_report(include_md=True, include_cq=False, provider_call_trace=[trace])
            with self.subTest(trace=trace):
                self.assertFalse(ready(build(report)))
        report = valid_stage4k5_report(include_md=True, include_cq=False, provider_call_trace=["bad"])
        self.assertFalse(ready(build(report)))

    def test_execution_summary_native_bool_and_direct_call_gates(self) -> None:
        cases = [
            {"market_data_execution_results": {"attempted": True, "provider_called": False, "direct_ib_call_made": False, "reqMktData_called": False, "success": True}},
            {"market_data_execution_results": {"attempted": True, "provider_called": True, "direct_ib_call_made": False, "reqMktData_called": False, "success": False}},
            {"market_data_execution_results": {"attempted": True, "provider_called": True, "direct_ib_call_made": True, "reqMktData_called": False, "success": True}},
            {"market_data_execution_results": {"attempted": True, "provider_called": True, "direct_ib_call_made": False, "reqMktData_called": True, "success": True}},
            {"contract_qualification_execution_results": {"attempted": True, "provider_called": False, "direct_ib_call_made": False, "qualifyContracts_called": False, "reqContractDetails_called": False, "success": True}},
            {"contract_qualification_execution_results": {"attempted": True, "provider_called": True, "direct_ib_call_made": False, "qualifyContracts_called": False, "reqContractDetails_called": False, "success": False}},
            {"contract_qualification_execution_results": {"attempted": True, "provider_called": True, "direct_ib_call_made": True, "qualifyContracts_called": False, "reqContractDetails_called": False, "success": True}},
            {"contract_qualification_execution_results": {"attempted": True, "provider_called": True, "direct_ib_call_made": False, "qualifyContracts_called": True, "reqContractDetails_called": False, "success": True}},
            {"contract_qualification_execution_results": {"attempted": True, "provider_called": True, "direct_ib_call_made": False, "qualifyContracts_called": False, "reqContractDetails_called": True, "success": True}},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                self.assertFalse(ready(build(valid_stage4k5_report(**overrides))))
        self.assertFalse(ready(build(valid_stage4k5_report(include_md=False, include_cq=False, provider_call_trace=[], applied_operations=[]))))

    def test_operation_audit_matches_by_stable_ids_not_list_index(self) -> None:
        report = valid_stage4k5_report(include_md=True, include_cq=True)
        report["applied_operations"] = list(reversed(report["applied_operations"]))  # type: ignore[index,arg-type]
        self.assertTrue(ready(build_with_snapshots(report)))
        report = valid_stage4k5_report(include_md=True, include_cq=True)
        report["applied_operations"] = [_operation("market_data", "request_controlled_market_data", "wrong")]
        self.assertFalse(ready(build(report)))
        report = valid_stage4k5_report(include_md=True, include_cq=True)
        report["applied_operations"] = report["applied_operations"][:1]  # type: ignore[index]
        self.assertFalse(ready(build(report)))

    def test_failed_and_unexpected_skipped_operations_block_without_crashing(self) -> None:
        cases = [
            valid_stage4k5_report(failed_operations=[_operation("market_data", "request_controlled_market_data", "md-001-SPY", "failed")]),
            valid_stage4k5_report(skipped_operations=[_operation("market_data", "request_controlled_market_data", "md-001-SPY", "skipped")]),
            valid_stage4k5_report(applied_operations=["bad"]),
            valid_stage4k5_report(failed_operations=["bad"]),
            valid_stage4k5_report(skipped_operations=["bad"]),
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertFalse(ready(build(case)))

    def test_safe_expected_skipped_trace_does_not_create_false_operation_mismatch(self) -> None:
        skipped_trace = _trace(2, "contract_qualification", "qualify_controlled_contracts", "cq-001-SPY", {})
        skipped_trace.update(provider_called=False, success=False, skipped_reason="provider category not required")
        skipped_op = _operation("contract_qualification", "qualify_controlled_contracts", "cq-001-SPY", "skipped")
        skipped_op["skipped_reason"] = "provider category not required"
        report = valid_stage4k5_report(include_md=True, include_cq=False)
        report["provider_call_trace"].append(skipped_trace)  # type: ignore[union-attr]
        report["skipped_operations"] = [skipped_op]
        self.assertTrue(ready(build_with_snapshots(report)))

    def test_unsafe_provider_results_and_stale_keys_block_readiness(self) -> None:
        unsafe_flags = [
            "live_trading_enabled",
            "broker_submission_enabled",
            "order_submitted",
            "state_written",
            "ledger_written",
            "direct_ib_call_made",
            "reqMktData_called",
            "qualifyContracts_called",
            "reqContractDetails_called",
        ]
        for flag in unsafe_flags:
            report = valid_stage4k5_report(include_md=True, include_cq=False)
            report["provider_call_trace"][0]["result"][flag] = True  # type: ignore[index]
            with self.subTest(flag=flag):
                self.assertFalse(ready(build(report)))
        for key in (
            "proposed_execution_permissions_for_" + "4J5",
            "may_call_strategy_next_phase",
            "may_build_executor_next_phase",
            "may_fetch_market_data_next_phase",
        ):
            self.assertFalse(ready(build(valid_stage4k5_report(**{key: True}))))

    def test_accepted_outputs_include_only_safe_successful_results_and_are_read_only(self) -> None:
        failed_trace = _trace(2, "market_data", "request_controlled_market_data", "md-002-SPY", {"bad": True})
        failed_trace.update(success=False, failure_reason="provider failed")
        report = valid_stage4k5_report(include_md=True, include_cq=False)
        report["provider_call_trace"].append(failed_trace)  # type: ignore[union-attr]
        built = build(report)
        self.assertFalse(ready(built))
        outputs = built["accepted_market_data_outputs"]
        self.assertEqual(outputs["accepted_result_count"], 1)
        self.assertTrue(outputs["read_only_for_future_stages"])

    def test_snapshot_blockers_and_market_window_warning(self) -> None:
        blockers = [
            {"state_snapshot": clean_state_snapshot(active_halt=True)},
            {"state_snapshot": clean_state_snapshot(unresolved_needs_reconciliation_count=1)},
            {"state_snapshot": clean_state_snapshot(active_intents_count=1)},
            {"scheduler_snapshot": clean_scheduler_snapshot(all_strategy_scheduler_enabled=True)},
            {"lifecycle_snapshot": clean_lifecycle_snapshot(all_strategy_lifecycle_enabled=True)},
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
        for kwargs in blockers:
            with self.subTest(kwargs=kwargs):
                self.assertFalse(ready(build_with_snapshots(**kwargs)))
        report = build_with_snapshots(market_window_snapshot=clean_market_window_snapshot(market_open=False))
        self.assertTrue(ready(report))
        self.assertIn("market is currently closed", " ".join(report["warnings"]))
        report = build(valid_stage4k5_report())
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, report["warnings"])

    def test_report_is_json_safe_and_inputs_are_not_mutated(self) -> None:
        source = valid_stage4k5_report()
        before = copy.deepcopy(source)
        report = build(source)
        self.assertEqual(source, before)
        json.dumps(report)
        for key in (
            "dry_run",
            "stage4k6_market_data_contract_acceptance_report",
            "stage4k5_executor_checks",
            "provider_result_acceptance",
            "operation_audit",
            "accepted_market_data_outputs",
            "accepted_contract_qualification_outputs",
            "boundary_checks",
            "required_inputs_for_next_phase",
            "readiness_for_next_phase",
        ):
            self.assertIn(key, report)

    def test_recommendations_do_not_include_disallowed_actions(self) -> None:
        report = build(valid_stage4k5_report())
        text = json.dumps(report["recommendations"]["ordered_next_steps"]).lower()
        for forbidden in (
            "recommend live trading",
            "enable live trading.",
            "enable all strategies.",
            "place orders now.",
            "create intents or tickets now.",
            "write state or ledger now.",
            "direct reqmktdata",
            "direct qualifycontracts",
            "direct reqcontractdetails",
            "strategy scans now.",
        ):
            self.assertNotIn(forbidden, text)

    def test_cli_requires_dry_run_before_parsing_and_json_stdout_is_strict_json(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            rc = tool.run_stage4k6_market_data_contract_acceptance(["--stage4k5-executor-json", "{"])
        self.assertNotEqual(rc, 0)
        self.assertIn("--dry-run-only", err.getvalue())
        out = io.StringIO()
        with redirect_stdout(out):
            rc = tool.run_stage4k6_market_data_contract_acceptance(["--dry-run-only", "--json", "--stage4k5-executor-json", json.dumps(valid_stage4k5_report())])
        parsed = json.loads(out.getvalue())
        self.assertEqual(rc, 0)
        self.assertTrue(parsed["stage4k6_market_data_contract_acceptance_report"])
        out = io.StringIO()
        with redirect_stdout(out):
            rc = tool.run_stage4k6_market_data_contract_acceptance(["--dry-run-only", "--json", "--stage4k5-executor-json", "{"])
        self.assertNotEqual(rc, 0)
        self.assertIn("JSONDecodeError", json.loads(out.getvalue())["errors"][0])

    def test_no_direct_broker_provider_external_scheduler_strategy_or_runner_calls_in_stage4k6_files(self) -> None:
        forbidden_tokens = [
            "submit_order_plan(",
            "get_order_status(",
            "cancel_order(",
            "placeOrder(",
            "cancelOrder(",
            "reqMktData(",
            "qualifyContracts(",
            "reqContractDetails(",
            "request_controlled_market_data(",
            "qualify_controlled_contracts(",
            "StateStore(",
            "ledger.append",
            "ledger.write",
            "jsonl",
            "scheduler.add_job",
            "add_job(",
            ".start(",
            "run_scan(",
            "scan_now(",
            "run_controlled_paper_operation(",
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
            "random.",
            "time.time",
            "datetime.now",
            "traceback.format_exc",
        ]
        for path in STAGE4K6_RUNTIME_FILES:
            source = path.read_text()
            for token in forbidden_tokens:
                with self.subTest(path=path, token=token):
                    self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
