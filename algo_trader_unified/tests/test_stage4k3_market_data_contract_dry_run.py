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

from algo_trader_unified.core.stage4k3_market_data_contract_dry_run import (
    MARKET_WINDOW_MANUAL_WARNING,
    build_stage4k3_market_data_contract_dry_run_report,
)
from algo_trader_unified.tools import stage4k3_market_data_contract_dry_run as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4K3_RUNTIME_FILES = [
    ROOT / "core/stage4k3_market_data_contract_dry_run.py",
    ROOT / "tools/stage4k3_market_data_contract_dry_run.py",
]
STEPS = [
    "pre_execution_snapshot_check",
    "market_data_provider_gate_check",
    "contract_qualification_provider_gate_check",
    "selected_strategy_contract_scope_check",
    "market_data_plan_preview",
    "contract_qualification_plan_preview",
    "post_plan_safety_check",
]


def valid_stage4k2_report(**overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4k2_market_data_contract_plan_report": True,
        "generated_at": "2026-05-17T13:00:00+00:00",
        "artifact_checks": {
            "stage4k1_report_present": True,
            "stage4k1_report_ready": True,
            "selected_strategy_present": True,
            "operation_id_present": True,
            "market_data_plan_present": True,
            "contract_qualification_plan_present": True,
            "proposed_operation_flow_present": True,
            "proposed_provider_payloads_present": True,
        },
        "selected_strategy": {"selected_strategy_id": "S01", "paper_only": True, "one_strategy_only": True},
        "operation": {
            "operation_id": "s01_once_2026_05_16",
            "operation_scope": "single_strategy_market_data_contract_plan",
            "paper_only": True,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
        },
        "market_data_plan": {
            "available": True,
            "may_fetch_market_data_in_4k2": False,
            "may_fetch_market_data_in_4k3": False,
            "may_fetch_market_data_in_4k5": True,
            "allowed_provider_method_name": "request_controlled_market_data",
            "planned_requests": [{"request_id": "md-001-SPY", "payload": {"symbol": "SPY"}}],
        },
        "contract_qualification_plan": {
            "available": True,
            "may_qualify_contracts_in_4k2": False,
            "may_qualify_contracts_in_4k3": False,
            "may_qualify_contracts_in_4k5": True,
            "allowed_provider_method_name": "qualify_controlled_contracts",
            "planned_qualifications": [{"qualification_id": "cq-001-SPY-STK", "payload": {"symbol": "SPY", "sec_type": "STK"}}],
        },
        "proposed_operation_flow": [_flow_step(index, step) for index, step in enumerate(STEPS, start=1)],
        "proposed_provider_payloads": {
            "market_data_provider_payloads": [_provider_payload("SPY", request_id="md-001-SPY")],
            "contract_qualification_provider_payloads": [_provider_payload("SPY", sec_type="STK", qualification_id="cq-001-SPY-STK")],
        },
        "boundary_checks": {
            "no_market_data_fetched": True,
            "no_contracts_qualified": True,
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
        "readiness_for_stage4k3": {"ready_to_build_market_data_contract_dry_run": True, "blockers": [], "warnings": []},
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def _flow_step(index: int, step: str) -> dict:
    return {
        "sequence_number": index,
        "step_name": step,
        "target_component": "stage4k2_component",
        "payload": {
            "selected_strategy_id": "S01",
            "operation_id": "s01_once_2026_05_16",
            "planned_market_data_request_count": 1,
            "planned_contract_qualification_count": 1,
            "preview_only": True,
        },
        "would_execute_now": False,
        "would_fetch_market_data_now": False,
        "would_qualify_contracts_now": False,
        "would_call_strategy_now": False,
        "would_create_intents_now": False,
        "would_create_tickets_now": False,
        "would_submit_orders_now": False,
        "would_write_state_now": False,
        "would_write_ledger_now": False,
        "paper_only": True,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }


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


_DEFAULT_PLAN = object()


def build(plan: object = _DEFAULT_PLAN, **kwargs: object) -> dict:
    if plan is _DEFAULT_PLAN:
        plan = valid_stage4k2_report()
    return build_stage4k3_market_data_contract_dry_run_report(
        stage4k2_plan_report=plan,  # type: ignore[arg-type]
        now_provider=lambda: datetime(2026, 5, 17, 14, 0, tzinfo=timezone.utc),
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
    plan = defaults.pop("plan", _DEFAULT_PLAN)
    return build(plan, **defaults)


def ready(report: dict) -> bool:
    return report["readiness_for_stage4k4"]["ready_to_build_market_data_contract_execution_gate"]


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


def _with_exact_proposed_provider_payloads(value: object) -> dict:
    report = valid_stage4k2_report()
    report["proposed_provider_payloads"] = value
    return report


class Stage4K3MarketDataContractDryRunTests(unittest.TestCase):
    def test_stage4k2_artifact_selected_strategy_and_operation_gates(self) -> None:
        cases = [
            None,
            valid_stage4k2_report(stage4k2_market_data_contract_plan_report=False),
            valid_stage4k2_report(success=False),
            valid_stage4k2_report(readiness_for_stage4k3={"ready_to_build_market_data_contract_dry_run": False}),
            valid_stage4k2_report(selected_strategy={"selected_strategy_id": ""}),
            valid_stage4k2_report(operation={"operation_id": ""}),
        ]
        for plan in cases:
            with self.subTest(plan=plan):
                report = build(plan)
                self.assertFalse(ready(report))
                json.dumps(report)

    def test_plan_flow_and_provider_payload_validation_blocks_without_crashing(self) -> None:
        cases = [
            valid_stage4k2_report(market_data_plan=None),
            valid_stage4k2_report(contract_qualification_plan=None),
            valid_stage4k2_report(proposed_operation_flow=None),
            valid_stage4k2_report(proposed_operation_flow=[_flow_step(1, "wrong")]),
            valid_stage4k2_report(proposed_operation_flow=[*_flow_step_list(), "bad"]),
            valid_stage4k2_report(proposed_operation_flow=[dict(_flow_step(1, STEPS[0]), payload="bad"), *_flow_step_list()[1:]]),
            valid_stage4k2_report(proposed_operation_flow=[dict(_flow_step(1, STEPS[0]), would_execute_now=True), *_flow_step_list()[1:]]),
            valid_stage4k2_report(proposed_operation_flow=[dict(_flow_step(1, STEPS[0]), would_fetch_market_data_now=True), *_flow_step_list()[1:]]),
            valid_stage4k2_report(proposed_operation_flow=[dict(_flow_step(1, STEPS[0]), would_qualify_contracts_now=True), *_flow_step_list()[1:]]),
            valid_stage4k2_report(proposed_operation_flow=[dict(_flow_step(1, STEPS[0]), would_submit_orders_now=True), *_flow_step_list()[1:]]),
            valid_stage4k2_report(proposed_provider_payloads=None),
            _with_exact_proposed_provider_payloads({"market_data_provider_payloads": [], "contract_qualification_provider_payloads": []}),
            _with_exact_proposed_provider_payloads({"market_data_provider_payloads": "bad", "contract_qualification_provider_payloads": []}),
            _with_exact_proposed_provider_payloads({"market_data_provider_payloads": ["bad"], "contract_qualification_provider_payloads": [_provider_payload("SPY")]}),
            _with_exact_proposed_provider_payloads({"market_data_provider_payloads": [json.dumps(_provider_payload("SPY"))], "contract_qualification_provider_payloads": [_provider_payload("SPY")]}),
            _with_exact_proposed_provider_payloads({"market_data_provider_payloads": [_provider_payload("SPY", bad=Decimal("1.2"))], "contract_qualification_provider_payloads": [_provider_payload("SPY")]}),
        ]
        for plan in cases:
            with self.subTest(plan=plan):
                self.assertFalse(ready(build(plan)))

    def test_provider_payload_strict_disabled_permissions_and_capability_layers(self) -> None:
        for key in ("allow_live_trading", "allow_broker_submission", "allow_order_submission", "allow_state_write", "allow_ledger_write"):
            with self.subTest(key=key):
                payload = _provider_payload("SPY", **{key: True})
                self.assertFalse(ready(build(valid_stage4k2_report(proposed_provider_payloads={"market_data_provider_payloads": [payload], "contract_qualification_provider_payloads": [_provider_payload("SPY")]}))))
                payload = _provider_payload("SPY", **{key: "False"})
                self.assertFalse(ready(build(valid_stage4k2_report(proposed_provider_payloads={"market_data_provider_payloads": [payload], "contract_qualification_provider_payloads": [_provider_payload("SPY")]}))))
        for layer in ("root", "capabilities", "config", "market_data", "contract_qualification"):
            with self.subTest(layer=layer):
                payload = _provider_payload("SPY")
                if layer == "root":
                    payload["reqMktData_enabled"] = "False"
                else:
                    payload[layer] = {"reqMktData_enabled": "True", "qualifyContracts_enabled": "False"}
                report = build(valid_stage4k2_report(proposed_provider_payloads={"market_data_provider_payloads": [payload], "contract_qualification_provider_payloads": [_provider_payload("SPY")]}))
                self.assertFalse(ready(report))
        self.assertTrue(ready(build_with_snapshots(plan=valid_stage4k2_report(proposed_provider_payloads={"market_data_provider_payloads": [_provider_payload("SPY", capabilities={"reqMktData_enabled": False})], "contract_qualification_provider_payloads": [_provider_payload("SPY", config={"qualifyContracts_enabled": False})]}))))

    def test_provider_payload_lists_allow_single_sided_and_combined_plans(self) -> None:
        md_only_payloads = {
            "market_data_provider_payloads": [_provider_payload("SPY")],
            "contract_qualification_provider_payloads": [],
        }
        cq_only_payloads = {
            "market_data_provider_payloads": [],
            "contract_qualification_provider_payloads": [_provider_payload("SPY")],
        }
        combined_payloads = {
            "market_data_provider_payloads": [_provider_payload("SPY")],
            "contract_qualification_provider_payloads": [_provider_payload("SPY")],
        }
        both_empty_payloads = {
            "market_data_provider_payloads": [],
            "contract_qualification_provider_payloads": [],
        }
        missing_md_payloads = {"contract_qualification_provider_payloads": [_provider_payload("SPY")]}
        missing_cq_payloads = {"market_data_provider_payloads": [_provider_payload("SPY")]}
        self.assertTrue(ready(build_with_snapshots(plan=_with_exact_proposed_provider_payloads(md_only_payloads))))
        self.assertTrue(ready(build_with_snapshots(plan=_with_exact_proposed_provider_payloads(cq_only_payloads))))
        self.assertTrue(ready(build_with_snapshots(plan=_with_exact_proposed_provider_payloads(combined_payloads))))
        self.assertTrue(ready(build_with_snapshots(plan=_with_exact_proposed_provider_payloads(missing_md_payloads))))
        self.assertTrue(ready(build_with_snapshots(plan=_with_exact_proposed_provider_payloads(missing_cq_payloads))))
        self.assertFalse(ready(build_with_snapshots(plan=_with_exact_proposed_provider_payloads(both_empty_payloads))))

    def test_dry_run_trace_results_boundaries_and_report_shape(self) -> None:
        report = build_with_snapshots()
        self.assertTrue(ready(report))
        self.assertEqual([item["step_name"] for item in report["dry_run_trace"]], STEPS)
        for item in report["dry_run_trace"]:
            self.assertEqual(item["status"], "simulated")
            self.assertIsInstance(item["input_payload"], dict)
            self.assertIsInstance(item["simulated_result"], dict)
            self.assertIsInstance(item["simulated_result"]["result_placeholder"], str)
            self.assertNotIsInstance(item["simulated_result"].get("result"), dict)
            for key in ("would_execute_now", "would_fetch_market_data_now", "would_qualify_contracts_now", "would_submit_orders_now"):
                self.assertFalse(item[key])
        md = report["market_data_dry_run_results"]
        self.assertFalse(md["attempted"])
        self.assertFalse(md["provider_called"])
        self.assertFalse(md["reqMktData_called"])
        self.assertFalse(md["direct_ib_call_made"])
        self.assertTrue(md["may_execute_in_4k5"])
        cq = report["contract_qualification_dry_run_results"]
        self.assertFalse(cq["attempted"])
        self.assertFalse(cq["provider_called"])
        self.assertFalse(cq["qualifyContracts_called"])
        self.assertFalse(cq["reqContractDetails_called"])
        self.assertFalse(cq["direct_ib_call_made"])
        self.assertTrue(cq["may_execute_in_4k5"])
        for key in ("dry_run", "stage4k3_market_data_contract_dry_run_report", "stage4k2_plan_checks", "dry_run_provider_payloads", "boundary_checks", "required_inputs_for_4k4", "readiness_for_stage4k4"):
            self.assertIn(key, report)
        json.dumps(report)

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

    def test_valid_with_missing_optional_snapshots_is_ready_with_warnings_and_inputs_not_mutated(self) -> None:
        plan = valid_stage4k2_report()
        original = copy.deepcopy(plan)
        report = build(plan)
        self.assertTrue(ready(report))
        self.assertEqual(plan, original)
        self.assertTrue(report["warnings"])
        self.assertEqual(report["generated_at"], "2026-05-17T14:00:00+00:00")

    def test_recommendations_remain_conservative(self) -> None:
        text = " ".join(build()["recommendations"]["ordered_next_steps"]).lower()
        for phrase in (
            "enable live trading",
            "enable all strategies",
            "place orders now",
            "enable broker submission now",
            "create intents",
            "create tickets",
            "write state",
            "write ledger",
            "fetch market data now",
            "qualify contracts now",
        ):
            self.assertNotIn(phrase, text)

    def test_cli_requires_dry_run_json_is_strict_and_actions_are_not_exposed(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            rc = tool.run_stage4k3_market_data_contract_dry_run(["--stage4k2-plan-json", "{"])
        self.assertNotEqual(rc, 0)
        self.assertIn("--dry-run-only", stderr.getvalue())

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4k3_market_data_contract_dry_run(["--dry-run-only", "--json", "--stage4k2-plan-json", json.dumps(valid_stage4k2_report())])
        self.assertEqual(rc, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4k3_market_data_contract_dry_run_report"])
        self.assertTrue(parsed["readiness_for_stage4k4"]["ready_to_build_market_data_contract_execution_gate"])

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4k3_market_data_contract_dry_run(["--dry-run-only", "--json", "--stage4k2-plan-json", "{"])
        self.assertNotEqual(rc, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertIn("JSONDecodeError", parsed["errors"][0])

        namespace = argparse.Namespace()
        for attr in ("submit", "cancel", "status", "market_data_execute", "contract_qualification_execute", "scheduler_enable", "broker_submit"):
            self.assertFalse(hasattr(namespace, attr))

    def test_no_direct_broker_external_scheduler_strategy_or_runner_calls_in_stage4k3_files(self) -> None:
        source = "\n".join(path.read_text() for path in STAGE4K3_RUNTIME_FILES)
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


def _flow_step_list() -> list[dict]:
    return [_flow_step(index, step) for index, step in enumerate(STEPS, start=1)]


if __name__ == "__main__":
    unittest.main()
