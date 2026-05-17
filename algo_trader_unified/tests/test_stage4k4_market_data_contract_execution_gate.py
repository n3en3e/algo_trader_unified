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

from algo_trader_unified.core.stage4k4_market_data_contract_execution_gate import (
    MARKET_WINDOW_MANUAL_WARNING,
    REQUIRED_OPERATOR_ACKNOWLEDGEMENTS,
    build_stage4k4_market_data_contract_execution_gate_report,
)
from algo_trader_unified.tools import stage4k4_market_data_contract_execution_gate as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4K4_RUNTIME_FILES = [
    ROOT / "core/stage4k4_market_data_contract_execution_gate.py",
    ROOT / "tools/stage4k4_market_data_contract_execution_gate.py",
]


def valid_stage4k3_report(**overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4k3_market_data_contract_dry_run_report": True,
        "generated_at": "2026-05-17T14:00:00+00:00",
        "artifact_checks": {
            "stage4k2_report_present": True,
            "stage4k2_report_ready": True,
            "selected_strategy_present": True,
            "operation_id_present": True,
            "proposed_provider_payloads_present": True,
        },
        "selected_strategy": {"selected_strategy_id": "S01", "paper_only": True, "one_strategy_only": True},
        "operation": {
            "operation_id": "s01_once_2026_05_16",
            "operation_scope": "single_strategy_market_data_contract_dry_run",
            "paper_only": True,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
        },
        "dry_run_trace": [_trace_step(index) for index in range(1, 4)],
        "dry_run_provider_payloads": {
            "market_data_provider_payloads": [_provider_payload("SPY", request_id="md-001-SPY")],
            "contract_qualification_provider_payloads": [_provider_payload("SPY", qualification_id="cq-001-SPY")],
        },
        "market_data_dry_run_results": {
            "attempted": False,
            "provider_called": False,
            "direct_ib_call_made": False,
            "reqMktData_called": False,
        },
        "contract_qualification_dry_run_results": {
            "attempted": False,
            "provider_called": False,
            "direct_ib_call_made": False,
            "qualifyContracts_called": False,
            "reqContractDetails_called": False,
        },
        "boundary_checks": {
            "no_market_data_fetched": True,
            "no_contracts_qualified": True,
            "no_provider_called": True,
            "no_direct_ib_call": True,
            "no_reqMktData": True,
            "no_qualifyContracts": True,
            "no_reqContractDetails": True,
            "no_strategy_scan": True,
            "no_intents_created": True,
            "no_tickets_created": True,
            "no_orders_submitted": True,
            "no_state_written": True,
            "no_ledger_written": True,
            "no_broker_submission": True,
            "no_live_trading": True,
            "no_all_strategy_enablement": True,
            "no_scheduler_registration": True,
            "no_lifecycle_execution": True,
        },
        "safety_checks": {
            "no_live_trading": True,
            "no_all_strategy_enablement": True,
            "no_broker_submission_enabled": True,
            "no_market_data": True,
            "no_contract_qualification": True,
            "no_order_submission": True,
            "no_intent_creation": True,
            "no_ticket_creation": True,
            "no_state_write": True,
            "no_ledger_write": True,
        },
        "readiness_for_stage4k4": {"ready_to_build_market_data_contract_execution_gate": True, "blockers": [], "warnings": []},
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def _trace_step(index: int, **overrides: object) -> dict:
    item = {
        "sequence_number": index,
        "step_name": f"step_{index}",
        "status": "simulated",
        "input_payload": {"selected_strategy_id": "S01", "operation_id": "s01_once_2026_05_16"},
        "simulated_result": {"status": "simulated"},
        "would_execute_now": False,
        "would_fetch_market_data_now": False,
        "would_qualify_contracts_now": False,
        "would_call_strategy_now": False,
        "would_create_intents_now": False,
        "would_create_tickets_now": False,
        "would_submit_orders_now": False,
        "would_write_state_now": False,
        "would_write_ledger_now": False,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }
    item.update(overrides)
    return item


def _provider_payload(symbol: str, **extra: object) -> dict:
    payload = {
        "selected_strategy_id": "S01",
        "operation_id": "s01_once_2026_05_16",
        "symbol": symbol,
        "allow_live_trading": False,
        "allow_broker_submission": False,
        "allow_order_submission": False,
        "allow_state_write": False,
        "allow_ledger_write": False,
    }
    payload.update(extra)
    return payload


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
        report = valid_stage4k3_report()
    acknowledgements = kwargs.pop("operator_acknowledgements", list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS))
    return build_stage4k4_market_data_contract_execution_gate_report(
        stage4k3_dry_run_report=report,  # type: ignore[arg-type]
        operator_acknowledgements=acknowledgements,  # type: ignore[arg-type]
        now_provider=lambda: datetime(2026, 5, 17, 15, 0, tzinfo=timezone.utc),
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
    report = defaults.pop("report", _DEFAULT_REPORT)
    return build(report, **defaults)


def ready(report: dict) -> bool:
    return report["readiness_for_stage4k5"]["ready_to_execute_controlled_market_data_contract_providers"]


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


def _with_payloads(value: object) -> dict:
    report = valid_stage4k3_report()
    report["dry_run_provider_payloads"] = value
    return report


class Stage4K4MarketDataContractExecutionGateTests(unittest.TestCase):
    def test_stage4k3_artifact_strategy_operation_and_ack_gates(self) -> None:
        cases = [
            None,
            valid_stage4k3_report(stage4k3_market_data_contract_dry_run_report=False),
            valid_stage4k3_report(success=False),
            valid_stage4k3_report(readiness_for_stage4k4={"ready_to_build_market_data_contract_execution_gate": False}),
            valid_stage4k3_report(selected_strategy={"selected_strategy_id": ""}),
            valid_stage4k3_report(operation={"operation_id": ""}),
        ]
        for report in cases:
            with self.subTest(report=report):
                self.assertFalse(ready(build(report)))
        self.assertFalse(ready(build(operator_acknowledgements=None)))
        self.assertFalse(ready(build(operator_acknowledgements="bad")))  # type: ignore[arg-type]
        self.assertFalse(ready(build(operator_acknowledgements=REQUIRED_OPERATOR_ACKNOWLEDGEMENTS[:-1])))
        extra = build(operator_acknowledgements=[*REQUIRED_OPERATOR_ACKNOWLEDGEMENTS, "ACK_EXTRA"])
        self.assertTrue(ready(extra))
        self.assertIn("extra operator acknowledgements ignored", extra["warnings"])
        self.assertFalse(ready(build(operator_acknowledgements=["prefix_ACK_NO_ORDER_SUBMISSION", *REQUIRED_OPERATOR_ACKNOWLEDGEMENTS[:-1]])))

    def test_trace_payload_and_result_validation_blocks_without_crashing(self) -> None:
        cases = [
            valid_stage4k3_report(dry_run_trace=None),
            valid_stage4k3_report(dry_run_trace="bad"),
            valid_stage4k3_report(dry_run_trace=[_trace_step(1), "bad"]),
            valid_stage4k3_report(dry_run_trace=[_trace_step(1, status="done")]),
            valid_stage4k3_report(dry_run_trace=[_trace_step(1, would_execute_now=True)]),
            valid_stage4k3_report(dry_run_trace=[_trace_step(1, would_fetch_market_data_now=True)]),
            valid_stage4k3_report(dry_run_trace=[_trace_step(1, would_qualify_contracts_now=True)]),
            valid_stage4k3_report(dry_run_trace=[_trace_step(1, would_submit_orders_now=True)]),
            valid_stage4k3_report(dry_run_trace=[_trace_step(1, would_write_state_now="False")]),
            valid_stage4k3_report(market_data_dry_run_results={"attempted": False, "provider_called": True, "direct_ib_call_made": False, "reqMktData_called": False}),
            valid_stage4k3_report(market_data_dry_run_results={"attempted": False, "provider_called": False, "direct_ib_call_made": True, "reqMktData_called": False}),
            valid_stage4k3_report(market_data_dry_run_results={"attempted": False, "provider_called": False, "direct_ib_call_made": False, "reqMktData_called": True}),
            valid_stage4k3_report(contract_qualification_dry_run_results={"attempted": False, "provider_called": True, "direct_ib_call_made": False, "qualifyContracts_called": False, "reqContractDetails_called": False}),
            valid_stage4k3_report(contract_qualification_dry_run_results={"attempted": False, "provider_called": False, "direct_ib_call_made": True, "qualifyContracts_called": False, "reqContractDetails_called": False}),
            valid_stage4k3_report(contract_qualification_dry_run_results={"attempted": False, "provider_called": False, "direct_ib_call_made": False, "qualifyContracts_called": True, "reqContractDetails_called": False}),
            valid_stage4k3_report(contract_qualification_dry_run_results={"attempted": False, "provider_called": False, "direct_ib_call_made": False, "qualifyContracts_called": False, "reqContractDetails_called": True}),
        ]
        for report in cases:
            with self.subTest(report=report):
                self.assertFalse(ready(build(report)))
                json.dumps(build(report))

    def test_provider_payload_lists_allow_none_single_sided_and_combined_plans(self) -> None:
        md_only = _with_payloads({"market_data_provider_payloads": [_provider_payload("SPY")], "contract_qualification_provider_payloads": []})
        cq_only = _with_payloads({"market_data_provider_payloads": [], "contract_qualification_provider_payloads": [_provider_payload("SPY")]})
        combined = _with_payloads({"market_data_provider_payloads": [_provider_payload("SPY")], "contract_qualification_provider_payloads": [_provider_payload("SPY")]})
        md_none = _with_payloads({"market_data_provider_payloads": None, "contract_qualification_provider_payloads": [_provider_payload("SPY")]})
        cq_none = _with_payloads({"market_data_provider_payloads": [_provider_payload("SPY")], "contract_qualification_provider_payloads": None})
        both_none = _with_payloads({"market_data_provider_payloads": None, "contract_qualification_provider_payloads": None})
        for report in (md_only, cq_only, combined, md_none, cq_none):
            with self.subTest(report=report):
                self.assertTrue(ready(build_with_snapshots(report=report)))
        self.assertFalse(ready(build_with_snapshots(report=both_none)))

    def test_provider_payload_strict_permissions_and_json_safety(self) -> None:
        cases = [
            _with_payloads(None),
            _with_payloads({"market_data_provider_payloads": ["bad"], "contract_qualification_provider_payloads": [_provider_payload("SPY")]}),
            _with_payloads({"market_data_provider_payloads": [json.dumps(_provider_payload("SPY"))], "contract_qualification_provider_payloads": [_provider_payload("SPY")]}),
            _with_payloads({"market_data_provider_payloads": [_provider_payload("SPY", bad=Decimal("1.2"))], "contract_qualification_provider_payloads": [_provider_payload("SPY")]}),
        ]
        for key in ("allow_live_trading", "allow_broker_submission", "allow_order_submission", "allow_state_write", "allow_ledger_write"):
            cases.append(_with_payloads({"market_data_provider_payloads": [_provider_payload("SPY", **{key: True})], "contract_qualification_provider_payloads": [_provider_payload("SPY")]}))
            cases.append(_with_payloads({"market_data_provider_payloads": [_provider_payload("SPY", **{key: "False"})], "contract_qualification_provider_payloads": [_provider_payload("SPY")]}))
        for report in cases:
            with self.subTest(report=report):
                self.assertFalse(ready(build(report)))

    def test_permissions_payload_shape_and_stale_schema_absence(self) -> None:
        md_only = build_with_snapshots(report=_with_payloads({"market_data_provider_payloads": [_provider_payload("SPY", request_id="md-001")], "contract_qualification_provider_payloads": None}))
        self.assertTrue(ready(md_only))
        self.assertTrue(md_only["market_data_execution_permissions"]["may_call_controlled_market_data_provider_in_4k5"])
        self.assertFalse(md_only["contract_qualification_execution_permissions"]["may_call_controlled_contract_qualification_provider_in_4k5"])
        self.assertFalse(md_only["market_data_execution_permissions"]["may_call_reqMktData_directly"])

        cq_only = build_with_snapshots(report=_with_payloads({"market_data_provider_payloads": None, "contract_qualification_provider_payloads": [_provider_payload("SPY", qualification_id="cq-001")]}))
        self.assertTrue(ready(cq_only))
        self.assertFalse(cq_only["market_data_execution_permissions"]["may_call_controlled_market_data_provider_in_4k5"])
        self.assertTrue(cq_only["contract_qualification_execution_permissions"]["may_call_controlled_contract_qualification_provider_in_4k5"])
        self.assertFalse(cq_only["contract_qualification_execution_permissions"]["may_call_qualifyContracts_directly"])
        self.assertFalse(cq_only["contract_qualification_execution_permissions"]["may_call_reqContractDetails_directly"])

        payload = build_with_snapshots()["proposed_4k5_execution_payload"]
        self.assertIsInstance(payload, dict)
        json.dumps(payload)
        self.assertTrue(payload["allow_controlled_market_data_provider_call"])
        self.assertTrue(payload["allow_controlled_contract_qualification_provider_call"])
        for key in (
            "allow_direct_reqMktData",
            "allow_direct_qualifyContracts",
            "allow_direct_reqContractDetails",
            "allow_strategy_scan",
            "allow_intent_creation",
            "allow_ticket_creation",
            "allow_order_submission",
            "allow_broker_submission",
            "allow_state_write",
            "allow_ledger_write",
            "live_trading_enabled",
            "all_strategies_enabled",
        ):
            self.assertFalse(payload[key])
        for stale in ("proposed_execution_permissions_for_4J5", "may_call_strategy_next_phase", "may_build_executor_next_phase", "may_fetch_market_data_next_phase"):
            self.assertNotIn(stale, payload)

    def test_snapshot_blockers_and_warnings(self) -> None:
        blocking_cases = [
            {"state_snapshot": clean_state_snapshot(active_halt=True)},
            {"state_snapshot": clean_state_snapshot(unresolved_needs_reconciliation_count=1)},
            {"state_snapshot": {"needs_reconciliation_count": 1, "active_intents_count": 0}},
            {"state_snapshot": clean_state_snapshot(active_intents_count=1)},
            {"scheduler_snapshot": clean_scheduler_snapshot(all_strategies_enabled=True)},
            {"scheduler_snapshot": clean_scheduler_snapshot(strategy_scan_execution_enabled=True)},
            {"lifecycle_snapshot": clean_lifecycle_snapshot(all_strategies_enabled=True)},
            {"lifecycle_snapshot": clean_lifecycle_snapshot(lifecycle_transition_execution_enabled=True)},
            {"risk_snapshot": clean_risk_snapshot(risk_bypass_enabled=True)},
            {"risk_snapshot": clean_risk_snapshot(kill_switch_available=False)},
            {"risk_snapshot": clean_risk_snapshot(hard_halt_available=False)},
            {"risk_snapshot": clean_risk_snapshot(daily_loss_limit_available=False)},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(mode="LIVE")},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(ibkr_port=4002)},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(paper_trading=False)},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(live_trading_enabled=True)},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(broker_submission_enabled=True)},
            {"market_window_snapshot": clean_market_window_snapshot(allowed_to_schedule_paper_run=False)},
        ]
        for kwargs in blocking_cases:
            with self.subTest(kwargs=kwargs):
                self.assertFalse(ready(build_with_snapshots(**kwargs)))
        safe_intents = build_with_snapshots(state_snapshot=clean_state_snapshot(active_intents_count=1, active_intents_safe_for_enablement=True))
        self.assertTrue(ready(safe_intents))
        self.assertIn("active intents present", " ".join(safe_intents["warnings"]))
        closed = build_with_snapshots(market_window_snapshot=clean_market_window_snapshot(market_open=False))
        self.assertTrue(ready(closed))
        self.assertIn("market is currently closed", " ".join(closed["warnings"]))
        missing_window = build_with_snapshots(market_window_snapshot=None)
        self.assertTrue(ready(missing_window))
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, missing_window["warnings"])

    def test_report_shape_json_safety_recommendations_and_no_mutation(self) -> None:
        source = valid_stage4k3_report()
        original = copy.deepcopy(source)
        report = build(source)
        self.assertTrue(ready(report))
        self.assertEqual(source, original)
        for key in (
            "dry_run",
            "stage4k4_market_data_contract_execution_gate_report",
            "stage4k3_dry_run_checks",
            "operator_acknowledgement_checks",
            "market_data_execution_permissions",
            "contract_qualification_execution_permissions",
            "execution_gate",
            "proposed_4k5_execution_payload",
            "boundary_checks",
            "required_inputs_for_4k5",
            "readiness_for_stage4k5",
        ):
            self.assertIn(key, report)
        self.assertEqual(report["generated_at"], "2026-05-17T15:00:00+00:00")
        json.dumps(report)
        text = " ".join(report["recommendations"]["ordered_next_steps"]).lower()
        for phrase in ("enable live trading", "enable all strategies", "place orders now", "create intents", "write state", "direct reqmktdata"):
            self.assertNotIn(phrase, text)

    def test_cli_requires_dry_run_json_is_strict_and_actions_are_not_exposed(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            rc = tool.run_stage4k4_market_data_contract_execution_gate(["--stage4k3-dry-run-json", "{"])
        self.assertNotEqual(rc, 0)
        self.assertIn("--dry-run-only", stderr.getvalue())

        args = ["--dry-run-only", "--json", "--stage4k3-dry-run-json", json.dumps(valid_stage4k3_report())]
        for ack in REQUIRED_OPERATOR_ACKNOWLEDGEMENTS:
            args.extend(["--ack", ack])
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4k4_market_data_contract_execution_gate(args)
        self.assertEqual(rc, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4k4_market_data_contract_execution_gate_report"])
        self.assertTrue(parsed["readiness_for_stage4k5"]["ready_to_execute_controlled_market_data_contract_providers"])

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4k4_market_data_contract_execution_gate(["--dry-run-only", "--json", "--stage4k3-dry-run-json", "{"])
        self.assertNotEqual(rc, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertIn("JSONDecodeError", parsed["errors"][0])

        namespace = argparse.Namespace()
        for attr in ("submit", "cancel", "status", "market_data_execute", "contract_qualification_execute", "scheduler_enable", "broker_submit"):
            self.assertFalse(hasattr(namespace, attr))

    def test_no_direct_broker_external_scheduler_strategy_or_runner_calls_in_stage4k4_files(self) -> None:
        source = "\n".join(path.read_text() for path in STAGE4K4_RUNTIME_FILES)
        for forbidden in (
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
            ".save(",
            ".write(",
            ".update(",
            "ledger.append(",
            "ledger.write(",
            "scheduler.add_job(",
            "add_job(",
            ".start(",
            "run_scan(",
            "scan_now(",
            "run_controlled_paper_operation(",
            "yfinance.",
            "requests.",
            "urllib.",
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
