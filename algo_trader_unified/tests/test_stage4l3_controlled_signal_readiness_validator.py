from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4l3_controlled_signal_readiness_validator import (
    MARKET_WINDOW_MANUAL_WARNING,
    build_stage4l3_controlled_signal_readiness_validator_report,
)
from algo_trader_unified.tools import stage4l3_controlled_signal_readiness_validator as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4L3_RUNTIME_FILES = [
    ROOT / "core/stage4l3_controlled_signal_readiness_validator.py",
    ROOT / "tools/stage4l3_controlled_signal_readiness_validator.py",
]


def valid_stage4l2_report(**overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4l2_signal_readiness_gate_report": True,
        "generated_at": "2026-05-18T12:00:00+00:00",
        "selected_strategy": {"selected_strategy_id": "S01", "paper_only": True, "one_strategy_only": True},
        "operation": {
            "operation_id": "s01_once_2026_05_16",
            "operation_scope": "single_strategy_signal_readiness_gate",
            "paper_only": True,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
        },
        "proposed_4l3_signal_readiness_payload": _payload(),
        "proposed_4l3_validation_flow": _flow(),
        "boundary_checks": {
            "no_signal_readiness_validator_called": True,
            "no_provider_called": True,
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
        "readiness_for_stage4l3": {
            "ready_to_execute_controlled_signal_readiness_validator": True,
            "next_recommended_phase": "Stage 4L-3 controlled signal readiness validator",
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def _payload(**overrides: object) -> dict:
    payload: dict[str, object] = {
        "selected_strategy_id": "S01",
        "operation_id": "s01_once_2026_05_16",
        "source_stage": "4L-2",
        "source_plan_stage": "4L-1",
        "input_scope": "single_strategy_controlled_signal_readiness_validation",
        "paper_only": True,
        "one_strategy_only": True,
        "read_only": True,
        "allow_controlled_signal_readiness_validator_call": True,
        "allow_strategy_scan": False,
        "allow_signal_execution": False,
        "allow_intent_creation": False,
        "allow_ticket_creation": False,
        "allow_order_submission": False,
        "allow_broker_submission": False,
        "allow_state_write": False,
        "allow_ledger_write": False,
        "allow_direct_reqMktData": False,
        "allow_direct_qualifyContracts": False,
        "allow_direct_reqContractDetails": False,
        "allow_provider_execution": False,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "signal_readiness_inputs": _inputs(),
        "generated_at": "2026-05-18T12:00:00+00:00",
    }
    _deep_update(payload, overrides)
    return payload


def _inputs(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "selected_strategy_id": "S01",
        "operation_id": "s01_once_2026_05_16",
        "read_only": True,
        "paper_only": True,
        "one_strategy_only": True,
        "market_data_inputs_available": True,
        "contract_qualification_inputs_available": False,
        "accepted_market_data_results": [{"symbol": "SPY", "bid": 1.0, "ask": 1.1}],
        "accepted_contract_qualification_results": [],
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
    }
    _deep_update(inputs, overrides)
    return inputs


def _flow() -> list[dict]:
    names = [
        "validate_stage4l1_plan",
        "validate_operator_acknowledgements",
        "validate_signal_readiness_inputs",
        "validate_strategy_registry_scope",
        "validate_signal_schema_scope",
        "validate_no_execution_permissions",
        "prepare_stage4l3_controlled_validator_payload",
    ]
    return [
        {
            "sequence_number": index + 1,
            "step_name": name,
            "target_component": "stage4l3_controlled_signal_readiness_validator",
            "would_call_signal_readiness_validator_now": False,
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


def validator_result(**overrides: object) -> dict:
    result: dict[str, object] = {
        "signal_readiness_validated": True,
        "selected_strategy_ready_for_signal_evaluation": True,
        "validator_called": True,
        "strategy_scan_executed": False,
        "signal_execution_performed": False,
        "intent_created": False,
        "ticket_created": False,
        "order_submitted": False,
        "broker_submission_enabled": False,
        "state_written": False,
        "ledger_written": False,
        "direct_ib_call_made": False,
        "provider_call_made": False,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "readiness_reasons": ["inputs present"],
        "blockers": [],
        "warnings": [],
        "signal_input_summary": {"accepted_categories": ["market_data"]},
        "validator_version": "test",
    }
    _deep_update(result, overrides)
    return result


_DEFAULT_VALIDATOR_RESULT = object()


class RecordingValidator:
    def __init__(self, result: object | None = _DEFAULT_VALIDATOR_RESULT) -> None:
        self.result = validator_result() if result is _DEFAULT_VALIDATOR_RESULT else result
        self.calls: list[dict] = []

    def validate_signal_readiness(self, payload: dict) -> object:
        self.calls.append(copy.deepcopy(payload))
        return self.result


class RaisingValidator:
    def validate_signal_readiness(self, payload: dict) -> dict:
        raise ValueError("bad\n<object at 0xABCDEF>")


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
    snapshot: dict[str, object] = {"selected_strategy_id": "S01", "broker_submission_enabled": False, "live_trading_enabled": False, "all_strategies_enabled": False, "lifecycle_transition_execution_enabled": False}
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
_DEFAULT_VALIDATOR = object()


def build(report: object = _DEFAULT_REPORT, validator: object | None = _DEFAULT_VALIDATOR, **kwargs: object) -> dict:
    return build_stage4l3_controlled_signal_readiness_validator_report(
        stage4l2_signal_readiness_gate_report=valid_stage4l2_report() if report is _DEFAULT_REPORT else report,  # type: ignore[arg-type]
        controlled_signal_readiness_validator=RecordingValidator() if validator is _DEFAULT_VALIDATOR else validator,
        now_provider=lambda: datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc),
        **kwargs,
    )


def build_with_snapshots(report: object = _DEFAULT_REPORT, validator: object | None = _DEFAULT_VALIDATOR, **kwargs: object) -> dict:
    defaults: dict[str, object] = {
        "strategy_registry_snapshot": ["S01"],
        "signal_schema_snapshot": {"selected_strategy_id": "S01", "expected_input_sections": ["selected_strategy_id", "operation_id"]},
        "state_snapshot": clean_state_snapshot(),
        "risk_snapshot": clean_risk_snapshot(),
        "scheduler_snapshot": clean_scheduler_snapshot(),
        "lifecycle_snapshot": clean_lifecycle_snapshot(),
        "paper_broker_snapshot": clean_paper_broker_snapshot(),
        "market_window_snapshot": clean_market_window_snapshot(),
    }
    defaults.update(kwargs)
    return build(valid_stage4l2_report() if report is _DEFAULT_REPORT else report, validator=validator, **defaults)


def ready(report: dict) -> bool:
    return report["readiness_for_stage4l4"]["ready_to_build_signal_readiness_acceptance"]


class Stage4L3ControlledSignalReadinessValidatorTests(unittest.TestCase):
    def test_valid_clean_4l2_report_with_safe_validator_is_ready_for_4l4_only(self) -> None:
        report = build_with_snapshots()
        self.assertTrue(ready(report))
        self.assertTrue(report["success"])
        self.assertEqual(report["readiness_for_stage4l4"]["next_recommended_phase"], "Stage 4L-4 signal readiness acceptance")
        self.assertTrue(report["validator_execution_result"]["validator_called"])
        self.assertTrue(report["signal_readiness_result_acceptance"]["accepted"])
        self.assertNotIn("full paper trading is active", json.dumps(report).lower())
        json.dumps(report)

    def test_missing_or_not_ready_4l2_blocks_readiness(self) -> None:
        cases = [
            None,
            valid_stage4l2_report(stage4l2_signal_readiness_gate_report=False),
            valid_stage4l2_report(success=False),
            valid_stage4l2_report(readiness_for_stage4l3={"ready_to_execute_controlled_signal_readiness_validator": False}),
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertFalse(ready(build_with_snapshots(source)))

    def test_selected_strategy_operation_payload_and_permission_are_required(self) -> None:
        cases = [
            valid_stage4l2_report(selected_strategy={"selected_strategy_id": "", "paper_only": True, "one_strategy_only": True}),
            valid_stage4l2_report(operation={"operation_id": ""}),
            valid_stage4l2_report(proposed_4l3_signal_readiness_payload=None),
            valid_stage4l2_report(proposed_4l3_signal_readiness_payload="bad"),
            valid_stage4l2_report(proposed_4l3_signal_readiness_payload=_payload(allow_controlled_signal_readiness_validator_call=False)),
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertFalse(ready(build_with_snapshots(source)))

    def test_missing_or_invalid_validator_blocks_when_permission_is_true(self) -> None:
        self.assertFalse(ready(build_with_snapshots(validator=None)))
        self.assertFalse(ready(build_with_snapshots(validator=object())))
        report = build_with_snapshots(validator=lambda payload: validator_result())
        self.assertTrue(ready(report))

    def test_validator_receives_exact_sanitized_payload_and_inputs_are_not_mutated(self) -> None:
        source = valid_stage4l2_report()
        original = copy.deepcopy(source)
        validator = RecordingValidator()
        report = build_with_snapshots(source, validator=validator)
        self.assertTrue(ready(report))
        self.assertEqual(validator.calls, [source["proposed_4l3_signal_readiness_payload"]])
        self.assertNotIn("stage4l2_signal_readiness_gate_report", validator.calls[0])
        self.assertEqual(source, original)

    def test_payload_and_signal_input_disabled_flags_must_be_native_false(self) -> None:
        payload_flags = [
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
        ]
        for key in payload_flags:
            for value in (True, "False"):
                with self.subTest(key=key, value=value):
                    self.assertFalse(ready(build_with_snapshots(valid_stage4l2_report(proposed_4l3_signal_readiness_payload=_payload(**{key: value})))))
        input_flags = [
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
        ]
        for key in input_flags:
            with self.subTest(input_key=key):
                self.assertFalse(ready(build_with_snapshots(valid_stage4l2_report(proposed_4l3_signal_readiness_payload=_payload(signal_readiness_inputs=_inputs(**{key: True}))))))
                self.assertFalse(ready(build_with_snapshots(valid_stage4l2_report(proposed_4l3_signal_readiness_payload=_payload(signal_readiness_inputs=_inputs(**{key: "False"}))))))

    def test_signal_readiness_inputs_require_native_dict_and_accepted_category(self) -> None:
        cases = [
            _payload(signal_readiness_inputs=None),
            _payload(signal_readiness_inputs="bad"),
            _payload(signal_readiness_inputs=_inputs(market_data_inputs_available=False, contract_qualification_inputs_available=False, accepted_market_data_results=[], accepted_contract_qualification_results=[])),
            _payload(signal_readiness_inputs=_inputs(read_only=False)),
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertFalse(ready(build_with_snapshots(valid_stage4l2_report(proposed_4l3_signal_readiness_payload=payload))))

    def test_validation_flow_exact_contract_and_malformed_arrays_block_without_index_error(self) -> None:
        cases: list[object] = [
            None,
            "bad",
            _flow()[:-1],
            _flow() + [_flow()[0]],
            [None],
        ]
        bad_sequence = _flow()
        bad_sequence[0]["sequence_number"] = 0
        cases.append(bad_sequence)
        bad_name = _flow()
        bad_name[2]["step_name"] = "unexpected"
        cases.append(bad_name)
        missing_name = _flow()
        missing_name[4].pop("step_name")
        cases.append(missing_name)
        extra_permission = _flow()
        extra_permission[0]["would_execute_strategy_now"] = True
        cases.append(extra_permission)
        for flow in cases:
            with self.subTest(flow=flow):
                self.assertFalse(ready(build_with_snapshots(valid_stage4l2_report(proposed_4l3_validation_flow=flow))))

    def test_validator_result_malformed_or_unsafe_blocks_cleanly(self) -> None:
        cases: list[object] = [
            None,
            "bad",
            validator_result(value={1, 2}),
            {key: value for key, value in validator_result().items() if key != "validator_called"},
            validator_result(strategy_scan_executed="False"),
            validator_result(strategy_scan_executed=True),
            validator_result(signal_execution_performed=True),
            validator_result(intent_created=True),
            validator_result(ticket_created=True),
            validator_result(order_submitted=True),
            validator_result(broker_submission_enabled=True),
            validator_result(state_written=True),
            validator_result(ledger_written=True),
            validator_result(direct_ib_call_made=True),
            validator_result(provider_call_made=True),
            validator_result(live_trading_enabled=True),
            validator_result(all_strategies_enabled=True),
            validator_result(readiness_reasons="bad"),
            validator_result(signal_input_summary="bad"),
            validator_result(order_tickets=[{"id": "x"}]),
            validator_result(**{"may_call_" + "strategy_next_phase": False}),
        ]
        for result in cases:
            with self.subTest(result=result):
                self.assertFalse(ready(build_with_snapshots(validator=RecordingValidator(result))))

    def test_safe_not_ready_results_are_not_code_failures_but_block_4l4_readiness(self) -> None:
        not_validated = build_with_snapshots(validator=RecordingValidator(validator_result(signal_readiness_validated=False, blockers=["not validated"])))
        self.assertFalse(ready(not_validated))
        self.assertFalse(not_validated["validator_execution_result"]["success"])

        not_ready = build_with_snapshots(validator=RecordingValidator(validator_result(selected_strategy_ready_for_signal_evaluation=False, blockers=["market setup not ready"])))
        self.assertFalse(ready(not_ready))
        self.assertTrue(not_ready["validator_execution_result"]["success"])
        self.assertTrue(not_ready["signal_readiness_result_acceptance"]["validator_execution_safe"])
        self.assertFalse(not_ready["signal_readiness_result_acceptance"]["strategy_signal_ready"])

    def test_validator_exception_is_flattened_and_sanitized(self) -> None:
        report = build_with_snapshots(validator=RaisingValidator())
        self.assertFalse(ready(report))
        reason = report["validator_execution_result"]["failure_reason"]
        self.assertTrue(reason.startswith("ValueError: bad "))
        self.assertNotIn("\n", reason)
        self.assertNotRegex(reason, r"0x[0-9A-Fa-f]+")

    def test_call_trace_and_acceptance_fields_are_json_safe_and_non_executing(self) -> None:
        report = build_with_snapshots()
        trace = report["validator_call_trace"]
        for key in (
            "validator_method",
            "selected_strategy_id",
            "operation_id",
            "input_payload",
            "validator_called",
            "direct_strategy_scan_called",
            "signal_execution_performed",
            "direct_ib_call_made",
            "provider_call_made",
            "success",
            "result",
            "failure_reason",
            "skipped_reason",
        ):
            self.assertIn(key, trace)
        self.assertFalse(trace["direct_strategy_scan_called"])
        self.assertFalse(trace["signal_execution_performed"])
        self.assertFalse(trace["direct_ib_call_made"])
        self.assertFalse(trace["provider_call_made"])
        json.dumps(trace)

    def test_stale_4j_keys_block_readiness(self) -> None:
        stale_keys = [
            "proposed_execution_permissions_for_" + "4J5",
            "may_call_" + "strategy_next_phase",
            "may_build_" + "executor_next_phase",
            "may_fetch_" + "market_data_next_phase",
        ]
        for key in stale_keys:
            with self.subTest(key=key):
                payload = _payload()
                payload[key] = False
                self.assertFalse(ready(build_with_snapshots(valid_stage4l2_report(proposed_4l3_signal_readiness_payload=payload))))

    def test_state_risk_scheduler_lifecycle_broker_and_market_window_blockers(self) -> None:
        cases = [
            {"state_snapshot": clean_state_snapshot(active_halt=True)},
            {"state_snapshot": clean_state_snapshot(unresolved_needs_reconciliation_count=None, needs_reconciliation_count=1)},
            {"state_snapshot": clean_state_snapshot(active_intents_count=1)},
            {"scheduler_snapshot": {"all_strategy_scheduler_enabled": True}},
            {"scheduler_snapshot": {"jobs": [{"strategy_id": "S01", "broker_submission_enabled": False, "live_trading_enabled": False, "all_strategies_enabled": False, "strategy_scan_execution_enabled": True, "signal_execution_enabled": False}]}},
            {"lifecycle_snapshot": {"all_strategy_lifecycle_enabled": True}},
            {"lifecycle_snapshot": clean_lifecycle_snapshot(lifecycle_transition_execution_enabled=True)},
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

    def test_missing_market_window_warns_but_does_not_block_clean_report(self) -> None:
        report = build(
            strategy_registry_snapshot=["S01"],
            signal_schema_snapshot={"expected_input_sections": ["selected_strategy_id"]},
            state_snapshot=clean_state_snapshot(),
            risk_snapshot=clean_risk_snapshot(),
            scheduler_snapshot=clean_scheduler_snapshot(),
            lifecycle_snapshot=clean_lifecycle_snapshot(),
            paper_broker_snapshot=clean_paper_broker_snapshot(),
        )
        self.assertTrue(ready(report))
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, report["warnings"])

    def test_report_top_level_fields_recommendations_and_no_mutation(self) -> None:
        source = valid_stage4l2_report()
        original = copy.deepcopy(source)
        report = build_with_snapshots(source)
        self.assertEqual(source, original)
        for key in (
            "dry_run",
            "stage4l3_controlled_signal_readiness_validator_report",
            "artifact_checks",
            "selected_strategy",
            "operation",
            "stage4l2_gate_checks",
            "validator_payload_checks",
            "validator_availability_checks",
            "validator_execution_result",
            "validator_call_trace",
            "signal_readiness_result_acceptance",
            "boundary_checks",
            "required_inputs_for_4l4",
            "activation_snapshot_checks",
            "state_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "paper_broker_checks",
            "market_window_checks",
            "safety_checks",
            "readiness_for_stage4l4",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        recommendation_text = json.dumps(report["recommendations"]).lower()
        self.assertIn("build stage 4l-4 signal readiness acceptance", recommendation_text)
        for forbidden in ("enable live trading", "place orders now", "enable broker submission now", "create intents", "write state"):
            self.assertIn(f"do not {forbidden}", recommendation_text)
        self.assertNotIn("4j", recommendation_text)

    def test_cli_requires_dry_run_before_parsing_and_json_stdout_is_strict_json(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = tool.run_stage4l3_controlled_signal_readiness_validator(["--json", "--stage4l2-gate-json", "{bad"])
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("--dry-run-only", stderr.getvalue())

        stdout = io.StringIO()
        args = [
            "--dry-run-only",
            "--json",
            "--stage4l2-gate-json",
            json.dumps(valid_stage4l2_report()),
        ]
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = tool.run_stage4l3_controlled_signal_readiness_validator(args)
        self.assertEqual(code, 1)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4l3_controlled_signal_readiness_validator_report"])
        self.assertFalse(parsed["validator_execution_result"]["validator_called"])

    def test_stage4l3_files_do_not_expose_or_call_forbidden_runtime_paths(self) -> None:
        forbidden_call_patterns = [
            "submit_order_plan",
            "get_order_status",
            "cancel_order",
            "placeOrder",
            "cancelOrder",
            "request_controlled_market_data",
            "qualify_controlled_contracts",
            "StateStore(",
            "ledger.append",
            "ledger.write",
            "scheduler.add_job",
            "add_job",
            "run_scan",
            "scan_now",
            "run_controlled_paper_operation",
            "calculate_signal",
            "generate_signal",
            "execute_signal",
            "create_intent",
            "build_ticket",
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
        for path in STAGE4L3_RUNTIME_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("ib_insync", text)
            self.assertNotIn(".write(", text)
            for token in forbidden_call_patterns:
                if token.endswith("(") or "." in token:
                    self.assertNotIn(token, text, token)
                else:
                    self.assertNotIn(f"{token}(", text, token)
            for direct_method in ("reqMktData", "qualifyContracts", "reqContractDetails"):
                self.assertNotIn(f"{direct_method}(", text)
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
