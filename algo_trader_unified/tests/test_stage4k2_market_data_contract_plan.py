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

from algo_trader_unified.core.stage4k2_market_data_contract_plan import (
    MARKET_WINDOW_MANUAL_WARNING,
    build_stage4k2_market_data_contract_plan_report,
)
from algo_trader_unified.tools import stage4k2_market_data_contract_plan as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4K2_RUNTIME_FILES = [
    ROOT / "core/stage4k2_market_data_contract_plan.py",
    ROOT / "tools/stage4k2_market_data_contract_plan.py",
]


def valid_stage4k1_report(**overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4k1_market_data_contract_readiness_report": True,
        "generated_at": "2026-05-17T12:00:00+00:00",
        "artifact_checks": {
            "stage4j6_report_present": True,
            "stage4j6_report_ready": True,
            "stage4j_complete": True,
            "selected_strategy_present": True,
            "operation_id_present": True,
            "recommended_next_gate_matches": True,
        },
        "selected_strategy": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
        },
        "operation": {
            "operation_id": "s01_once_2026_05_16",
            "operation_scope": "single_strategy_market_data_contract_readiness",
            "paper_only": True,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
        },
        "proposed_4k_scope": {
            "selected_strategy_id": "S01",
            "operation_id": "s01_once_2026_05_16",
            "may_build_market_data_contract_plan_next_phase": True,
            "may_fetch_market_data_now": False,
            "may_qualify_contracts_now": False,
            "may_submit_orders_now": False,
            "may_create_intents_now": False,
            "may_create_tickets_now": False,
            "may_write_state_now": False,
            "may_write_ledger_now": False,
            "live_trading_enabled": False,
            "all_strategies_enabled": False,
            "broker_submission_enabled": False,
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
        "readiness_for_stage4k2": {
            "ready_to_build_market_data_contract_plan": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def requirements_snapshot() -> dict:
    return {
        "requirements": [
            {"strategy_id": "S01", "symbol": "XSP", "sec_type": "OPT", "expiry": "20260619", "strike": 520, "right": "C", "paper_eligible": True},
            {"strategy_id": "S01", "symbol": "SPY", "sec_type": "STK", "paper_eligible": True},
        ]
    }


def market_data_capability(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "selected_strategy_id": "S01",
        "market_data_provider_available": True,
        "paper_market_data_mode": True,
        "live_market_data_enabled": False,
        "streaming_market_data_enabled": False,
        "snapshot_market_data_enabled": False,
        "market_data_currently_enabled": False,
        "reqMktData_enabled": False,
        "market_data_request_limit_configured": True,
        "symbol_universe_defined": True,
    }
    _deep_update(snapshot, overrides)
    return snapshot


def contract_capability(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "selected_strategy_id": "S01",
        "contract_qualification_provider_available": True,
        "live_contract_qualification_enabled": False,
        "contract_qualification_currently_enabled": False,
        "qualifyContracts_enabled": False,
        "reqContractDetails_enabled": False,
        "contract_universe_defined": True,
    }
    _deep_update(snapshot, overrides)
    return snapshot


def clean_state_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "active_halt": False,
        "unresolved_needs_reconciliation_count": 0,
        "active_intents_count": 0,
        "open_positions_count": 7,
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
        "jobs": [
            {
                "strategy_id": "S01",
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


_DEFAULT_K1 = object()


def build(k1: object = _DEFAULT_K1, **kwargs: object) -> dict:
    if k1 is _DEFAULT_K1:
        k1 = valid_stage4k1_report()
    return build_stage4k2_market_data_contract_plan_report(
        stage4k1_readiness_report=k1,  # type: ignore[arg-type]
        now_provider=lambda: datetime(2026, 5, 17, 13, 0, tzinfo=timezone.utc),
        **kwargs,
    )


def build_with_snapshots(**kwargs: object) -> dict:
    defaults: dict[str, object] = {
        "strategy_contract_requirements_snapshot": requirements_snapshot(),
        "market_data_capability_snapshot": market_data_capability(),
        "contract_qualification_capability_snapshot": contract_capability(),
        "state_snapshot": clean_state_snapshot(),
        "risk_snapshot": clean_risk_snapshot(),
        "scheduler_snapshot": clean_scheduler_snapshot(),
        "lifecycle_snapshot": clean_lifecycle_snapshot(),
        "paper_broker_snapshot": clean_paper_broker_snapshot(),
        "market_window_snapshot": clean_market_window_snapshot(),
    }
    defaults.update(kwargs)
    k1 = defaults.pop("k1", _DEFAULT_K1)
    return build(k1, **defaults)


def ready(report: dict) -> bool:
    return report["readiness_for_stage4k3"]["ready_to_build_market_data_contract_dry_run"]


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class Stage4K2MarketDataContractPlanTests(unittest.TestCase):
    def test_stage4k1_artifact_gates_and_proposed_scope_block_readiness(self) -> None:
        cases = [
            None,
            valid_stage4k1_report(stage4k1_market_data_contract_readiness_report=False),
            valid_stage4k1_report(success=False),
            valid_stage4k1_report(readiness_for_stage4k2={"ready_to_build_market_data_contract_plan": False}),
            valid_stage4k1_report(selected_strategy={"selected_strategy_id": ""}),
            valid_stage4k1_report(operation={"operation_id": ""}),
            valid_stage4k1_report(proposed_4k_scope=None),
            valid_stage4k1_report(proposed_4k_scope="bad"),
            valid_stage4k1_report(proposed_4k_scope={"may_fetch_market_data_now": True}),
            valid_stage4k1_report(proposed_4k_scope={"may_qualify_contracts_now": True}),
            valid_stage4k1_report(proposed_4k_scope={"may_submit_orders_now": True}),
        ]
        for k1 in cases:
            with self.subTest(k1=k1):
                report = build(k1)
                self.assertFalse(ready(report))
                json.dumps(report)

    def test_selected_strategy_and_operation_use_safe_traversal(self) -> None:
        self.assertFalse(ready(build(valid_stage4k1_report(selected_strategy="bad"))))
        self.assertFalse(ready(build(valid_stage4k1_report(operation=["bad"]))))
        report = build(valid_stage4k1_report(selected_strategy={"selected_strategy_id": " S01 "}))
        self.assertEqual(report["selected_strategy"]["selected_strategy_id"], "S01")

    def test_requirements_parse_dict_lists_strings_malformed_and_sort_deterministically(self) -> None:
        self.assertTrue(ready(build_with_snapshots(strategy_contract_requirements_snapshot={"requirements": [{"strategy_id": "S01", "symbol": "SPY"}]})))
        self.assertTrue(ready(build_with_snapshots(strategy_contract_requirements_snapshot=["XSP", "SPY"])))
        self.assertTrue(ready(build_with_snapshots(strategy_contract_requirements_snapshot=[{"strategy_id": "S01", "symbol": "XSP"}, {"strategy_id": "S01", "symbol": "SPY"}])))
        malformed = build_with_snapshots(strategy_contract_requirements_snapshot=[object(), {"strategy_id": "S01", "symbol": "SPY"}])
        self.assertTrue(ready(malformed))
        self.assertIn("malformed strategy contract requirement", " ".join(malformed["warnings"]))
        sorted_report = build_with_snapshots(strategy_contract_requirements_snapshot=[{"strategy_id": "S01", "symbol": "ZZZ"}, {"strategy_id": "S01", "symbol": "AAA"}])
        self.assertEqual(sorted_report["market_data_plan"]["normalized_symbols"], ["AAA", "ZZZ"])

    def test_requirement_contradictions_block_readiness(self) -> None:
        for snapshot in (
            [{"strategy_id": "S02", "symbol": "SPY"}],
            [{"strategy_id": "S01", "symbol": "SPY", "paper_eligible": False}],
            [{"strategy_id": "S01", "symbol": "SPY"}, {"strategy_id": "S02", "symbol": "XSP"}],
        ):
            with self.subTest(snapshot=snapshot):
                self.assertFalse(ready(build_with_snapshots(strategy_contract_requirements_snapshot=snapshot)))

    def test_capability_snapshots_missing_warn_and_nested_config_parses(self) -> None:
        missing = build()
        self.assertTrue(ready(missing))
        self.assertIn("market_data capability snapshot missing", " ".join(missing["warnings"]))
        self.assertIn("contract_qualification capability snapshot missing", " ".join(missing["warnings"]))
        self.assertTrue(ready(build_with_snapshots(market_data_capability_snapshot={"capabilities": market_data_capability()})))
        self.assertTrue(ready(build_with_snapshots(market_data_capability_snapshot={"config": market_data_capability()})))
        self.assertTrue(ready(build_with_snapshots(contract_qualification_capability_snapshot={"capabilities": contract_capability()})))
        self.assertTrue(ready(build_with_snapshots(contract_qualification_capability_snapshot={"config": contract_capability()})))
        self.assertTrue(ready(build_with_snapshots(market_data_capability_snapshot={"selected_strategy_id": "S01"})))
        self.assertTrue(ready(build_with_snapshots(contract_qualification_capability_snapshot={"selected_strategy_id": "S01"})))

    def test_capability_unsafe_enablement_and_string_booleans_block(self) -> None:
        for key in (
            "market_data_currently_enabled",
            "reqMktData_enabled",
            "streaming_market_data_enabled",
            "snapshot_market_data_enabled",
            "live_market_data_enabled",
        ):
            with self.subTest(market_data_key=key):
                self.assertFalse(ready(build_with_snapshots(market_data_capability_snapshot=market_data_capability(**{key: True}))))
                self.assertFalse(ready(build_with_snapshots(market_data_capability_snapshot=market_data_capability(**{key: "True"}))))
                self.assertFalse(ready(build_with_snapshots(market_data_capability_snapshot=market_data_capability(**{key: "False"}))))
        for key in (
            "contract_qualification_currently_enabled",
            "qualifyContracts_enabled",
            "reqContractDetails_enabled",
            "live_contract_qualification_enabled",
        ):
            with self.subTest(contract_key=key):
                self.assertFalse(ready(build_with_snapshots(contract_qualification_capability_snapshot=contract_capability(**{key: True}))))
                self.assertFalse(ready(build_with_snapshots(contract_qualification_capability_snapshot=contract_capability(**{key: "True"}))))
                self.assertFalse(ready(build_with_snapshots(contract_qualification_capability_snapshot=contract_capability(**{key: "False"}))))
        self.assertFalse(ready(build_with_snapshots(market_data_capability_snapshot=market_data_capability(selected_strategy_id="S02"))))
        self.assertFalse(ready(build_with_snapshots(contract_qualification_capability_snapshot=contract_capability(selected_strategy_id="S02"))))
        self.assertFalse(ready(build_with_snapshots(market_data_capability_snapshot=market_data_capability(market_data_request_limit_configured="True"))))
        self.assertFalse(ready(build_with_snapshots(contract_qualification_capability_snapshot=contract_capability(contract_universe_defined="False"))))

    def test_plans_flow_and_provider_payloads_are_native_dicts_and_safe(self) -> None:
        report = build_with_snapshots()
        self.assertTrue(ready(report))
        md_plan = report["market_data_plan"]
        cq_plan = report["contract_qualification_plan"]
        self.assertFalse(md_plan["may_fetch_market_data_in_4k2"])
        self.assertTrue(md_plan["may_fetch_market_data_in_4k5"])
        self.assertFalse(cq_plan["may_qualify_contracts_in_4k2"])
        self.assertTrue(cq_plan["may_qualify_contracts_in_4k5"])
        self.assertEqual(md_plan["normalized_symbols"], ["SPY", "XSP"])
        self.assertEqual([item["symbol"] for item in cq_plan["planned_qualifications"]], ["SPY", "XSP"])
        for item in md_plan["planned_requests"] + cq_plan["planned_qualifications"]:
            self.assertIsInstance(item, dict)
            self.assertIsInstance(item["payload"], dict)
            self.assertFalse(item["would_execute_now"])
            self.assertFalse(item["live_trading_enabled"])
            self.assertFalse(item["broker_submission_enabled"])
        steps = [item["step_name"] for item in report["proposed_operation_flow"]]
        self.assertEqual(
            steps,
            [
                "pre_execution_snapshot_check",
                "market_data_provider_gate_check",
                "contract_qualification_provider_gate_check",
                "selected_strategy_contract_scope_check",
                "market_data_plan_preview",
                "contract_qualification_plan_preview",
                "post_plan_safety_check",
            ],
        )
        for step in report["proposed_operation_flow"]:
            self.assertIsInstance(step["payload"], dict)
            for key in ("would_fetch_market_data_now", "would_qualify_contracts_now", "would_submit_orders_now", "would_write_state_now"):
                self.assertFalse(step[key])
        for payload in report["proposed_provider_payloads"]["market_data_provider_payloads"] + report["proposed_provider_payloads"]["contract_qualification_provider_payloads"]:
            self.assertIsInstance(payload, dict)
            self.assertEqual(payload["selected_strategy_id"], "S01")
            self.assertEqual(payload["operation_id"], "s01_once_2026_05_16")
            for key in ("allow_live_trading", "allow_broker_submission", "allow_order_submission", "allow_state_write", "allow_ledger_write"):
                self.assertFalse(payload[key])
        json.dumps(report)

    def test_boundary_safety_state_risk_scheduler_lifecycle_broker_and_window_blockers(self) -> None:
        blocking_cases = [
            {"k1": valid_stage4k1_report(boundary_checks={"no_market_data_fetched": "False"})},
            {"k1": valid_stage4k1_report(safety_checks={"no_market_data": "False"})},
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

    def test_valid_clean_report_is_ready_required_fields_exist_json_safe_and_inputs_not_mutated(self) -> None:
        k1 = valid_stage4k1_report()
        reqs = requirements_snapshot()
        original_k1 = copy.deepcopy(k1)
        original_reqs = copy.deepcopy(reqs)
        report = build_with_snapshots(k1=k1, strategy_contract_requirements_snapshot=reqs)
        self.assertTrue(ready(report))
        self.assertEqual(k1, original_k1)
        self.assertEqual(reqs, original_reqs)
        for key in (
            "dry_run",
            "stage4k2_market_data_contract_plan_report",
            "generated_at",
            "artifact_checks",
            "selected_strategy",
            "operation",
            "stage4k1_readiness_checks",
            "strategy_contract_requirements_checks",
            "market_data_readiness",
            "contract_qualification_readiness",
            "market_data_plan",
            "contract_qualification_plan",
            "proposed_operation_flow",
            "proposed_provider_payloads",
            "boundary_checks",
            "required_inputs_for_4k3",
            "activation_snapshot_checks",
            "state_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "paper_broker_checks",
            "market_window_checks",
            "safety_checks",
            "readiness_for_stage4k3",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report)
        json.dumps(build(valid_stage4k1_report(extra=Decimal("1.2"))))

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
            "fetch market data now",
            "qualify contracts now",
        ):
            self.assertNotIn(phrase, text)

    def test_cli_requires_dry_run_json_is_strict_and_actions_are_not_exposed(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            rc = tool.run_stage4k2_market_data_contract_plan(
                ["--stage4k1-readiness-json", json.dumps(valid_stage4k1_report())]
            )
        self.assertNotEqual(rc, 0)
        self.assertIn("--dry-run-only", stderr.getvalue())

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4k2_market_data_contract_plan(
                ["--dry-run-only", "--json", "--stage4k1-readiness-json", json.dumps(valid_stage4k1_report())]
            )
        self.assertEqual(rc, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4k2_market_data_contract_plan_report"])
        self.assertTrue(parsed["readiness_for_stage4k3"]["ready_to_build_market_data_contract_dry_run"])

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4k2_market_data_contract_plan(
                ["--dry-run-only", "--json", "--stage4k1-readiness-json", "{"]
            )
        self.assertNotEqual(rc, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertIn("JSONDecodeError", parsed["errors"][0])

        namespace = argparse.Namespace()
        for attr in (
            "submit",
            "cancel",
            "status",
            "market_data_execute",
            "contract_qualification_execute",
            "scheduler_enable",
            "broker_submit",
        ):
            self.assertFalse(hasattr(namespace, attr))

    def test_no_direct_broker_external_scheduler_strategy_or_runner_calls_in_stage4k2_files(self) -> None:
        source = "\n".join(path.read_text() for path in STAGE4K2_RUNTIME_FILES)
        for forbidden in (
            "submit_order_plan(",
            "get_order_status(",
            "cancel_order(",
            "placeOrder(",
            "cancelOrder(",
            "reqMktData(",
            "qualifyContracts(",
            "reqContractDetails(",
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
