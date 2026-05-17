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

from algo_trader_unified.core.stage4k1_market_data_contract_readiness import (
    MARKET_WINDOW_MANUAL_WARNING,
    build_stage4k1_market_data_contract_readiness_report,
)
from algo_trader_unified.tools import stage4k1_market_data_contract_readiness as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4K1_RUNTIME_FILES = [
    ROOT / "core/stage4k1_market_data_contract_readiness.py",
    ROOT / "tools/stage4k1_market_data_contract_readiness.py",
]


def valid_stage4j6_report(**overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4j6_controlled_paper_operation_acceptance_report": True,
        "generated_at": "2026-05-16T16:00:00+00:00",
        "artifact_checks": {
            "stage4j5_report_present": True,
            "stage4j5_report_ready": True,
            "selected_strategy_present": True,
            "operation_id_present": True,
        },
        "selected_strategy": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
        },
        "operation": {
            "operation_id": "s01_once_2026_05_16",
            "operation_scope": "single_strategy_controlled_paper_operation_acceptance",
            "paper_only": True,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
        },
        "executor_acceptance": {
            "accepted": True,
            "acceptance_status": "accepted_completed_report_only",
            "selected_strategy_id": "S01",
            "operation_id": "s01_once_2026_05_16",
            "reason": "accepted_completed_report_only",
            "blockers": [],
            "warnings": [],
        },
        "boundary_checks": {
            "no_market_data_requested": True,
            "no_contracts_qualified": True,
            "no_intents_created": True,
            "no_tickets_created": True,
            "no_orders_submitted": True,
            "no_state_written": True,
            "no_ledger_written": True,
            "no_live_trading": True,
            "no_all_strategy_enablement": True,
            "no_broker_submission": True,
            "no_direct_strategy_scan": True,
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
        "readiness_for_stage4j_complete_or_next_gate": {
            "stage4j_complete": True,
            "ready_for_next_explicit_gate": True,
            "recommended_next_gate": "stage4k_market_data_and_contract_qualification_gate",
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def strategy_registry(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "strategies": [{"strategy_id": "S01", "paper_eligible": True}],
    }
    _deep_update(snapshot, overrides)
    return snapshot


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
        "jobs": [
            {
                "strategy_id": "S01",
                "paper_only": True,
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


_DEFAULT_J6 = object()


def build(j6: object = _DEFAULT_J6, **kwargs: object) -> dict:
    if j6 is _DEFAULT_J6:
        j6 = valid_stage4j6_report()
    return build_stage4k1_market_data_contract_readiness_report(
        stage4j6_acceptance_report=j6,  # type: ignore[arg-type]
        now_provider=lambda: datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc),
        **kwargs,
    )


def build_with_snapshots(**kwargs: object) -> dict:
    defaults: dict[str, object] = {
        "strategy_registry_snapshot": strategy_registry(),
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
    return build(**defaults)


def ready(report: dict) -> bool:
    return report["readiness_for_stage4k2"]["ready_to_build_market_data_contract_plan"]


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class Stage4K1MarketDataContractReadinessTests(unittest.TestCase):
    def test_stage4j6_artifact_gates_block_readiness(self) -> None:
        cases = [
            None,
            valid_stage4j6_report(stage4j6_controlled_paper_operation_acceptance_report=False),
            valid_stage4j6_report(success=False),
            valid_stage4j6_report(readiness_for_stage4j_complete_or_next_gate={"stage4j_complete": False}),
            valid_stage4j6_report(readiness_for_stage4j_complete_or_next_gate={"ready_for_next_explicit_gate": False}),
            valid_stage4j6_report(readiness_for_stage4j_complete_or_next_gate={"recommended_next_gate": "other"}),
            valid_stage4j6_report(executor_acceptance={"accepted": False}),
            valid_stage4j6_report(selected_strategy={"selected_strategy_id": ""}),
            valid_stage4j6_report(operation={"operation_id": ""}),
        ]
        for j6 in cases:
            with self.subTest(j6=j6):
                report = build(j6)
                self.assertFalse(ready(report))
                json.dumps(report)

    def test_stage4j6_boundary_and_safety_flags_block_when_not_strict_true(self) -> None:
        boundary_cases = [
            "no_market_data_requested",
            "no_contracts_qualified",
            "no_orders_submitted",
            "no_state_written",
            "no_ledger_written",
            "no_intents_created",
            "no_tickets_created",
            "no_live_trading",
            "no_all_strategy_enablement",
            "no_broker_submission",
        ]
        for key in boundary_cases:
            with self.subTest(boundary=key):
                report = build(valid_stage4j6_report(boundary_checks={key: False}))
                self.assertFalse(ready(report))
                self.assertIn(key, " ".join(report["readiness_for_stage4k2"]["blockers"]))
        for key in (
            "no_live_trading",
            "no_all_strategy_enablement",
            "no_broker_submission_enabled",
            "no_market_data",
            "no_contract_qualification",
            "no_order_submission",
            "no_intent_creation",
            "no_ticket_creation",
            "no_state_write",
            "no_ledger_write",
        ):
            with self.subTest(safety=key):
                report = build(valid_stage4j6_report(safety_checks={key: "False"}))
                self.assertFalse(ready(report))
                self.assertIn(key, " ".join(report["readiness_for_stage4k2"]["blockers"]))

    def test_selected_strategy_and_operation_use_safe_traversal(self) -> None:
        self.assertFalse(ready(build(valid_stage4j6_report(selected_strategy="bad"))))
        self.assertFalse(ready(build(valid_stage4j6_report(operation=["bad"]))))
        report = build(valid_stage4j6_report(selected_strategy={"selected_strategy_id": " S01 "}))
        self.assertEqual(report["selected_strategy"]["selected_strategy_id"], "S01")

    def test_strategy_registry_parsing_and_blockers(self) -> None:
        self.assertTrue(ready(build_with_snapshots(strategy_registry_snapshot=strategy_registry())))
        self.assertTrue(ready(build_with_snapshots(strategy_registry_snapshot=["S01", "S02"])))
        self.assertTrue(ready(build_with_snapshots(strategy_registry_snapshot=[{"strategy_id": "S01", "paper_eligible": True}])))
        blocking = [
            {"strategies": [{"strategy_id": "S01", "paper_eligible": False}]},
            {"strategies": [{"strategy_id": "S02", "paper_eligible": True}]},
            {"strategies": [{"strategy_id": "S01", "active": True}, {"strategy_id": "S02", "active": True}]},
            {"active_strategy_ids": ["S01", "S02"], "strategies": [{"strategy_id": "S01", "paper_eligible": True}]},
        ]
        for snapshot in blocking:
            with self.subTest(snapshot=snapshot):
                self.assertFalse(ready(build_with_snapshots(strategy_registry_snapshot=snapshot)))
        malformed = build(strategy_registry_snapshot={"strategies": [object()]})
        self.assertTrue(ready(malformed))
        self.assertIn("malformed strategy registry", " ".join(malformed["warnings"]))

    def test_missing_capability_snapshots_warn_but_may_be_ready(self) -> None:
        report = build()
        self.assertTrue(ready(report))
        self.assertIn("market_data capability snapshot missing", " ".join(report["warnings"]))
        self.assertIn("contract_qualification capability snapshot missing", " ".join(report["warnings"]))

    def test_market_data_capability_blockers_nested_and_strict_booleans(self) -> None:
        for key in (
            "market_data_currently_enabled",
            "reqMktData_enabled",
            "streaming_market_data_enabled",
            "snapshot_market_data_enabled",
            "live_market_data_enabled",
        ):
            with self.subTest(key=key):
                self.assertFalse(ready(build_with_snapshots(market_data_capability_snapshot=market_data_capability(**{key: True}))))
                self.assertFalse(ready(build_with_snapshots(market_data_capability_snapshot=market_data_capability(**{key: "True"}))))
                self.assertFalse(ready(build_with_snapshots(market_data_capability_snapshot=market_data_capability(**{key: "False"}))))
        self.assertFalse(ready(build_with_snapshots(market_data_capability_snapshot=market_data_capability(selected_strategy_id="S02"))))
        self.assertFalse(ready(build_with_snapshots(market_data_capability_snapshot=market_data_capability(paper_market_data_mode=False))))
        self.assertTrue(ready(build_with_snapshots(market_data_capability_snapshot={"capabilities": market_data_capability()})))
        self.assertTrue(ready(build_with_snapshots(market_data_capability_snapshot={"config": market_data_capability()})))
        self.assertTrue(ready(build_with_snapshots(market_data_capability_snapshot={"selected_strategy_id": "S01"})))

    def test_contract_capability_blockers_nested_and_strict_booleans(self) -> None:
        for key in (
            "contract_qualification_currently_enabled",
            "qualifyContracts_enabled",
            "reqContractDetails_enabled",
            "live_contract_qualification_enabled",
        ):
            with self.subTest(key=key):
                self.assertFalse(ready(build_with_snapshots(contract_qualification_capability_snapshot=contract_capability(**{key: True}))))
                self.assertFalse(ready(build_with_snapshots(contract_qualification_capability_snapshot=contract_capability(**{key: "True"}))))
                self.assertFalse(ready(build_with_snapshots(contract_qualification_capability_snapshot=contract_capability(**{key: "False"}))))
        self.assertFalse(ready(build_with_snapshots(contract_qualification_capability_snapshot=contract_capability(selected_strategy_id="S02"))))
        self.assertTrue(ready(build_with_snapshots(contract_qualification_capability_snapshot={"capabilities": contract_capability()})))
        self.assertTrue(ready(build_with_snapshots(contract_qualification_capability_snapshot={"config": contract_capability()})))
        self.assertTrue(ready(build_with_snapshots(contract_qualification_capability_snapshot={"selected_strategy_id": "S01"})))

    def test_proposed_scope_never_grants_current_execution_permissions(self) -> None:
        report = build_with_snapshots()
        self.assertTrue(ready(report))
        scope = report["proposed_4k_scope"]
        self.assertTrue(scope["may_build_market_data_contract_plan_next_phase"])
        for key in (
            "may_fetch_market_data_now",
            "may_qualify_contracts_now",
            "may_submit_orders_now",
            "may_create_intents_now",
            "may_create_tickets_now",
            "may_write_state_now",
            "may_write_ledger_now",
            "live_trading_enabled",
            "all_strategies_enabled",
            "broker_submission_enabled",
        ):
            self.assertFalse(scope[key])

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

    def test_valid_clean_report_with_matching_snapshots_is_ready_and_inputs_not_mutated(self) -> None:
        j6 = valid_stage4j6_report()
        md = market_data_capability()
        original_j6 = copy.deepcopy(j6)
        original_md = copy.deepcopy(md)
        report = build_with_snapshots(j6=j6, market_data_capability_snapshot=md)
        self.assertTrue(ready(report))
        self.assertEqual(j6, original_j6)
        self.assertEqual(md, original_md)
        for key in (
            "dry_run",
            "stage4k1_market_data_contract_readiness_report",
            "generated_at",
            "artifact_checks",
            "selected_strategy",
            "operation",
            "stage4j_completion_checks",
            "strategy_registry_checks",
            "market_data_readiness",
            "contract_qualification_readiness",
            "proposed_4k_scope",
            "boundary_checks",
            "required_inputs_for_4k2",
            "activation_snapshot_checks",
            "state_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "paper_broker_checks",
            "market_window_checks",
            "safety_checks",
            "readiness_for_stage4k2",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report)
        weird = build(valid_stage4j6_report(extra=Decimal("1.2")))
        json.dumps(weird)

    def test_recommendations_remain_conservative(self) -> None:
        report = build()
        text = " ".join(report["recommendations"]["ordered_next_steps"]).lower()
        forbidden = (
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
        )
        for phrase in forbidden:
            self.assertNotIn(phrase, text)

    def test_cli_requires_dry_run_json_is_strict_and_actions_are_not_exposed(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            rc = tool.run_stage4k1_market_data_contract_readiness(
                ["--stage4j6-acceptance-json", json.dumps(valid_stage4j6_report())]
            )
        self.assertNotEqual(rc, 0)
        self.assertIn("--dry-run-only", stderr.getvalue())

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4k1_market_data_contract_readiness(
                ["--dry-run-only", "--json", "--stage4j6-acceptance-json", json.dumps(valid_stage4j6_report())]
            )
        self.assertEqual(rc, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4k1_market_data_contract_readiness_report"])
        self.assertTrue(parsed["readiness_for_stage4k2"]["ready_to_build_market_data_contract_plan"])

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4k1_market_data_contract_readiness(
                ["--dry-run-only", "--json", "--stage4j6-acceptance-json", "{"]
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

    def test_no_direct_broker_external_scheduler_strategy_or_runner_calls_in_stage4k1_files(self) -> None:
        source = "\n".join(path.read_text() for path in STAGE4K1_RUNTIME_FILES)
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
