from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4l2_signal_readiness_gate import (
    MARKET_WINDOW_MANUAL_WARNING,
    REQUIRED_ACKNOWLEDGEMENTS,
    build_stage4l2_signal_readiness_gate_report,
)
from algo_trader_unified.tools import stage4l2_signal_readiness_gate as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4L2_RUNTIME_FILES = [
    ROOT / "core/stage4l2_signal_readiness_gate.py",
    ROOT / "tools/stage4l2_signal_readiness_gate.py",
]


def valid_stage4l1_report(*, include_md: bool = True, include_cq: bool = True, **overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4l1_signal_readiness_plan_report": True,
        "generated_at": "2026-05-18T12:00:00+00:00",
        "selected_strategy": {"selected_strategy_id": "S01", "paper_only": True, "one_strategy_only": True},
        "operation": {
            "operation_id": "s01_once_2026_05_16",
            "operation_scope": "single_strategy_signal_readiness_plan",
            "paper_only": True,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
        },
        "proposed_signal_readiness_inputs": _signal_inputs(include_md=include_md, include_cq=include_cq),
        "proposed_4l2_validation_flow": _stage4l1_flow(),
        "boundary_checks": {
            "no_provider_called_in_4l1": True,
            "no_market_data_fetched": True,
            "no_contracts_qualified": True,
            "no_direct_ib_call": True,
            "no_direct_reqMktData": True,
            "no_direct_qualifyContracts": True,
            "no_direct_reqContractDetails": True,
            "no_strategy_scan": True,
            "no_signal_execution": True,
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
        "readiness_for_stage4l2": {
            "ready_to_build_signal_readiness_gate": True,
            "next_recommended_phase": "Stage 4L-2 signal readiness gate",
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def _signal_inputs(*, include_md: bool = True, include_cq: bool = True, **overrides: object) -> dict:
    inputs: dict[str, object] = {
        "selected_strategy_id": "S01",
        "operation_id": "s01_once_2026_05_16",
        "source_stage": "4L-1",
        "source_capability_stage": "4K-6",
        "input_scope": "single_strategy_signal_readiness_inputs",
        "paper_only": True,
        "one_strategy_only": True,
        "read_only": True,
        "market_data_inputs_available": include_md,
        "contract_qualification_inputs_available": include_cq,
        "accepted_market_data_results": [_accepted_result("market_data")] if include_md else [],
        "accepted_contract_qualification_results": [_accepted_result("contract_qualification")] if include_cq else [],
        "allow_strategy_scan": False,
        "allow_signal_execution": False,
        "allow_intent_creation": False,
        "allow_ticket_creation": False,
        "allow_order_submission": False,
        "allow_broker_submission": False,
        "allow_state_write": False,
        "allow_ledger_write": False,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "generated_at": "2026-05-18T12:00:00+00:00",
    }
    _deep_update(inputs, overrides)
    return inputs


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


def _stage4l1_flow() -> list[dict]:
    names = [
        "validate_stage4k_acceptance",
        "validate_selected_strategy_scope",
        "validate_accepted_market_data_inputs",
        "validate_accepted_contract_qualification_inputs",
        "validate_signal_schema_requirements",
        "validate_no_execution_permissions",
        "prepare_stage4l2_signal_readiness_gate_inputs",
    ]
    return [
        {
            "sequence_number": index + 1,
            "step_name": name,
            "target_component": "stage4l2_signal_readiness_gate",
            "input_sections": [],
            "would_execute_strategy_now": False,
            "would_calculate_signal_now": False,
            "would_create_intent_now": False,
            "would_create_ticket_now": False,
            "would_submit_order_now": False,
            "would_write_state_now": False,
            "would_write_ledger_now": False,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
            "paper_only": True,
        }
        for index, name in enumerate(names)
    ]


def clean_state_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"active_halt": False, "unresolved_needs_reconciliation_count": 0, "active_intents_count": 0, "open_positions_count": 3}
    _deep_update(snapshot, overrides)
    return snapshot


def clean_risk_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"kill_switch_available": True, "hard_halt_available": True, "daily_loss_limit_available": True, "max_position_limit_available": True, "risk_bypass_enabled": False}
    _deep_update(snapshot, overrides)
    return snapshot


def clean_scheduler_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"jobs": [{"strategy_id": "S01", "broker_submission_enabled": False, "live_trading_enabled": False, "all_strategies_enabled": False, "strategy_scan_execution_enabled": False, "signal_execution_enabled": False}]}
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


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


_DEFAULT_REPORT = object()


def build(report: object = _DEFAULT_REPORT, **kwargs: object) -> dict:
    acknowledgements = kwargs.pop("operator_acknowledgements", list(REQUIRED_ACKNOWLEDGEMENTS))
    return build_stage4l2_signal_readiness_gate_report(
        stage4l1_signal_readiness_plan_report=valid_stage4l1_report() if report is _DEFAULT_REPORT else report,  # type: ignore[arg-type]
        operator_acknowledgements=acknowledgements,  # type: ignore[arg-type]
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
    return build(valid_stage4l1_report() if report is _DEFAULT_REPORT else report, **defaults)


def ready(report: dict) -> bool:
    return report["readiness_for_stage4l3"]["ready_to_execute_controlled_signal_readiness_validator"]


class Stage4L2SignalReadinessGateTests(unittest.TestCase):
    def test_valid_clean_4l1_report_with_acknowledgements_is_ready_for_4l3_only(self) -> None:
        report = build_with_snapshots()
        self.assertTrue(ready(report))
        self.assertTrue(report["success"])
        self.assertEqual(report["readiness_for_stage4l3"]["next_recommended_phase"], "Stage 4L-3 controlled signal readiness validator")
        self.assertTrue(report["proposed_4l3_signal_readiness_payload"]["allow_controlled_signal_readiness_validator_call"])
        self.assertNotIn("full paper trading is active", json.dumps(report).lower())
        json.dumps(report)

    def test_missing_or_not_ready_4l1_blocks_readiness(self) -> None:
        cases = [
            None,
            valid_stage4l1_report(stage4l1_signal_readiness_plan_report=False),
            valid_stage4l1_report(success=False),
            valid_stage4l1_report(readiness_for_stage4l2={"ready_to_build_signal_readiness_gate": False}),
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertFalse(ready(build_with_snapshots(source)))

    def test_selected_strategy_and_operation_are_required_from_4l1(self) -> None:
        cases = [
            valid_stage4l1_report(selected_strategy={"selected_strategy_id": "", "paper_only": True, "one_strategy_only": True}),
            valid_stage4l1_report(selected_strategy={"selected_strategy_id": "S01", "paper_only": False, "one_strategy_only": True}),
            valid_stage4l1_report(selected_strategy={"selected_strategy_id": "S01", "paper_only": True, "one_strategy_only": False}),
            valid_stage4l1_report(operation={"operation_id": ""}),
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertFalse(ready(build_with_snapshots(source)))

    def test_operator_acknowledgements_are_exact_and_type_safe(self) -> None:
        self.assertFalse(ready(build_with_snapshots(operator_acknowledgements=None)))
        self.assertFalse(ready(build_with_snapshots(operator_acknowledgements="ACK_4L2_SIGNAL_READINESS_GATE_ONLY")))  # type: ignore[arg-type]
        missing_one = list(REQUIRED_ACKNOWLEDGEMENTS[:-1])
        self.assertFalse(ready(build_with_snapshots(operator_acknowledgements=missing_one)))
        substring = list(REQUIRED_ACKNOWLEDGEMENTS[:-1]) + ["ACK_SINGLE_STRATEGY_ONLY_EXTRA"]
        self.assertFalse(ready(build_with_snapshots(operator_acknowledgements=substring)))
        extra = list(REQUIRED_ACKNOWLEDGEMENTS) + ["ACK_EXTRA"]
        report = build_with_snapshots(operator_acknowledgements=extra)
        self.assertTrue(ready(report))
        self.assertTrue(report["operator_acknowledgement_checks"]["extra_acknowledgements"])

    def test_signal_inputs_block_malformed_or_unsafe_values(self) -> None:
        input_cases = [
            None,
            "bad",
            _signal_inputs(read_only=False),
            _signal_inputs(market_data_inputs_available=False, contract_qualification_inputs_available=False, accepted_market_data_results=[], accepted_contract_qualification_results=[]),
            _signal_inputs(accepted_market_data_results="bad"),
            _signal_inputs(accepted_market_data_results=[None]),
            _signal_inputs(accepted_market_data_results=[_accepted_result("market_data", value={1, 2})]),
            _signal_inputs(accepted_market_data_results=[_accepted_result("market_data", live_trading_enabled=True)]),
            _signal_inputs(accepted_market_data_results=[_accepted_result("market_data", broker_submission_enabled=True)]),
            _signal_inputs(accepted_market_data_results=[_accepted_result("market_data", order_submitted=True)]),
            _signal_inputs(accepted_market_data_results=[_accepted_result("market_data", state_written=True)]),
            _signal_inputs(accepted_market_data_results=[_accepted_result("market_data", ledger_written=True)]),
            _signal_inputs(accepted_market_data_results=[_accepted_result("market_data", direct_ib_call_made=True)]),
            _signal_inputs(accepted_market_data_results=[_accepted_result("market_data", reqMktData_called=True)]),
            _signal_inputs(accepted_market_data_results=[_accepted_result("market_data", qualifyContracts_called=True)]),
            _signal_inputs(accepted_market_data_results=[_accepted_result("market_data", reqContractDetails_called=True)]),
            _signal_inputs(accepted_market_data_results=[_accepted_result("market_data", live_trading_enabled="False")]),
            _signal_inputs(accepted_market_data_results=[_accepted_result("market_data", failure_reason="bad\nline")]),
            _signal_inputs(accepted_market_data_results=[_accepted_result("market_data", skipped_reason="<object at 0xABCDEF>")]),
        ]
        for inputs in input_cases:
            with self.subTest(inputs=inputs):
                source = valid_stage4l1_report(proposed_signal_readiness_inputs=inputs)
                self.assertFalse(ready(build_with_snapshots(source)))

    def test_signal_input_disabled_flags_must_be_native_false(self) -> None:
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
            with self.subTest(key=key):
                self.assertFalse(ready(build_with_snapshots(valid_stage4l1_report(proposed_signal_readiness_inputs=_signal_inputs(**{key: True})))))
                self.assertFalse(ready(build_with_snapshots(valid_stage4l1_report(proposed_signal_readiness_inputs=_signal_inputs(**{key: "False"})))))

    def test_inherited_4l2_validation_flow_exact_contract_is_required(self) -> None:
        self.assertFalse(ready(build_with_snapshots(valid_stage4l1_report(proposed_4l2_validation_flow=None))))
        shorter = _stage4l1_flow()[:-1]
        self.assertFalse(ready(build_with_snapshots(valid_stage4l1_report(proposed_4l2_validation_flow=shorter))))
        bad_sequence = _stage4l1_flow()
        bad_sequence[0]["sequence_number"] = 0
        self.assertFalse(ready(build_with_snapshots(valid_stage4l1_report(proposed_4l2_validation_flow=bad_sequence))))
        bad_name = _stage4l1_flow()
        bad_name[2]["step_name"] = "unexpected"
        self.assertFalse(ready(build_with_snapshots(valid_stage4l1_report(proposed_4l2_validation_flow=bad_name))))
        bad_permission = _stage4l1_flow()
        bad_permission[0]["would_execute_strategy_now"] = True
        self.assertFalse(ready(build_with_snapshots(valid_stage4l1_report(proposed_4l2_validation_flow=bad_permission))))

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
            ({"requires_market_data": True}, valid_stage4l1_report(include_md=False, include_cq=True)),
            ({"requires_contract_qualification": True}, valid_stage4l1_report(include_md=True, include_cq=False)),
            ({"expected_input_sections": "selected_strategy_id"}, valid_stage4l1_report()),
            ({"expected_input_sections": ["selected_strategy_id", 7]}, valid_stage4l1_report()),
            ({"expected_input_sections": ["missing_section"]}, valid_stage4l1_report()),
            ({"allow_strategy_scan": True}, valid_stage4l1_report()),
            ({"allow_signal_execution": True}, valid_stage4l1_report()),
            ({"allow_intent_creation": True}, valid_stage4l1_report()),
            ({"allow_order_submission": True}, valid_stage4l1_report()),
            ({"live_trading_enabled": True}, valid_stage4l1_report()),
        ]
        for schema, source in cases:
            with self.subTest(schema=schema):
                self.assertFalse(ready(build_with_snapshots(source, signal_schema_snapshot=schema)))
        self.assertTrue(ready(build_with_snapshots(signal_schema_snapshot={"expected_input_sections": ["selected_strategy_id", "operation_id", "accepted_market_data_results", "accepted_contract_qualification_results"]})))
        self.assertFalse(ready(build_with_snapshots(signal_schema_snapshot=[])))  # type: ignore[arg-type]

    def test_proposed_payload_and_flow_are_deterministic_non_executing_and_json_safe(self) -> None:
        report = build_with_snapshots()
        payload = report["proposed_4l3_signal_readiness_payload"]
        self.assertIsInstance(payload, dict)
        self.assertTrue(payload["allow_controlled_signal_readiness_validator_call"])
        for key in (
            "allow_strategy_scan",
            "allow_signal_execution",
            "allow_intent_creation",
            "allow_ticket_creation",
            "allow_order_submission",
            "allow_broker_submission",
            "allow_state_write",
            "allow_ledger_write",
            "allow_direct_reqMktData",
            "allow_direct_qualifyContracts",
            "allow_direct_reqContractDetails",
            "allow_provider_execution",
            "live_trading_enabled",
            "all_strategies_enabled",
        ):
            self.assertIs(payload[key], False)
        flow = report["proposed_4l3_validation_flow"]
        self.assertEqual([item["sequence_number"] for item in flow], list(range(1, 8)))
        self.assertEqual(len(flow), 7)
        for item in flow:
            self.assertFalse(item["would_call_signal_readiness_validator_now"])
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
            "proposed_execution_permissions_for_" + "4J5",
            "may_call_" + "strategy_next_phase",
            "may_build_" + "executor_next_phase",
            "may_fetch_" + "market_data_next_phase",
        ]
        for key in stale_keys:
            with self.subTest(key=key):
                source = valid_stage4l1_report()
                source["proposed_signal_readiness_inputs"][key] = False  # type: ignore[index]
                self.assertFalse(ready(build_with_snapshots(source)))

    def test_state_risk_scheduler_lifecycle_broker_and_market_window_blockers(self) -> None:
        cases = [
            {"state_snapshot": clean_state_snapshot(active_halt=True)},
            {"state_snapshot": clean_state_snapshot(unresolved_needs_reconciliation_count=None, needs_reconciliation_count=1)},
            {"state_snapshot": clean_state_snapshot(active_intents_count=1)},
            {"scheduler_snapshot": {"all_strategy_scheduler_enabled": True}},
            {"scheduler_snapshot": {"jobs": [{"strategy_id": "S01", "broker_submission_enabled": False, "live_trading_enabled": False, "all_strategies_enabled": False, "strategy_scan_execution_enabled": True, "signal_execution_enabled": False}]}},
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

    def test_missing_market_window_warns_but_does_not_block_clean_4l1(self) -> None:
        report = build(strategy_registry_snapshot=["S01"], signal_schema_snapshot={"expected_input_sections": ["selected_strategy_id"]})
        self.assertTrue(ready(report))
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, report["warnings"])

    def test_report_top_level_fields_recommendations_and_no_mutation(self) -> None:
        source = valid_stage4l1_report()
        original = copy.deepcopy(source)
        report = build_with_snapshots(source)
        self.assertEqual(source, original)
        for key in (
            "dry_run",
            "stage4l2_signal_readiness_gate_report",
            "artifact_checks",
            "selected_strategy",
            "operation",
            "stage4l1_plan_checks",
            "operator_acknowledgement_checks",
            "signal_input_checks",
            "strategy_registry_checks",
            "signal_schema_checks",
            "signal_readiness_gate",
            "proposed_4l3_signal_readiness_payload",
            "proposed_4l3_validation_flow",
            "boundary_checks",
            "required_inputs_for_4l3",
            "activation_snapshot_checks",
            "state_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "paper_broker_checks",
            "market_window_checks",
            "safety_checks",
            "readiness_for_stage4l3",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        recommendation_text = json.dumps(report["recommendations"]).lower()
        self.assertIn("build stage 4l-3 controlled signal readiness validator", recommendation_text)
        self.assertIn("do not enable live trading", recommendation_text)
        self.assertIn("do not place orders now", recommendation_text)
        self.assertNotIn("paper trading is fully active", recommendation_text)

    def test_cli_requires_dry_run_before_parsing_and_json_stdout_is_strict_json(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = tool.run_stage4l2_signal_readiness_gate(["--json", "--stage4l1-plan-json", "{bad"])
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("--dry-run-only", stderr.getvalue())

        stdout = io.StringIO()
        args = [
            "--dry-run-only",
            "--json",
            "--stage4l1-plan-json",
            json.dumps(valid_stage4l1_report(include_md=True, include_cq=False)),
            "--strategy-registry-snapshot-json",
            json.dumps(["S01"]),
            "--signal-schema-snapshot-json",
            json.dumps({"expected_input_sections": ["selected_strategy_id"]}),
        ]
        for ack in REQUIRED_ACKNOWLEDGEMENTS:
            args.extend(["--ack", ack])
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = tool.run_stage4l2_signal_readiness_gate(args)
        self.assertEqual(code, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4l2_signal_readiness_gate_report"])

    def test_stage4l2_files_do_not_expose_or_call_forbidden_runtime_paths(self) -> None:
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
            "StateStore(",
            "add_job",
            "start",
            "run_scan",
            "scan_now",
            "run_controlled_paper_operation",
            "calculate_signal",
            "generate_signal",
            "signal_validator",
            "validate_signal_readiness",
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
            "traceback.format_exc",
        ]
        for path in STAGE4L2_RUNTIME_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("ib_insync", text)
            self.assertNotIn(".write(", text)
            for token in forbidden_call_patterns:
                if token.endswith("("):
                    self.assertNotIn(token, text, token)
                else:
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
                "--validator-execute",
                "--scheduler-enable",
                "--broker-submit",
            ):
                self.assertNotIn(forbidden_cli_fragment, text)
