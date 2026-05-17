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

from algo_trader_unified.core import stage4j4_controlled_paper_operation_execution_gate as core
from algo_trader_unified.core.stage4j4_controlled_paper_operation_execution_gate import (
    MARKET_WINDOW_MANUAL_WARNING,
    REQUIRED_OPERATOR_ACKNOWLEDGEMENTS,
    build_stage4j4_controlled_paper_operation_execution_gate_report,
)
from algo_trader_unified.tools import stage4j4_controlled_paper_operation_execution_gate as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4J4_FILES = [
    ROOT / "core/stage4j4_controlled_paper_operation_execution_gate.py",
    ROOT / "tools/stage4j4_controlled_paper_operation_execution_gate.py",
]


def valid_stage4j3_report(**overrides: object) -> dict:
    operation_id = "s01_once_2026_05_16"
    report: dict[str, object] = {
        "dry_run": True,
        "stage4j3_controlled_paper_operation_dry_run_report": True,
        "generated_at": "2026-05-16T13:00:00+00:00",
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
        "dry_run_trace": _dry_run_trace(operation_id),
        "dry_run_trace_checks": {
            "trace_available": True,
            "trace_order_matches_plan": True,
            "all_trace_items_simulated": True,
            "no_strategy_call": True,
            "no_market_data": True,
            "no_contract_qualification": True,
            "no_intent_created": True,
            "no_ticket_created": True,
            "no_broker_submission": True,
            "no_state_write": True,
            "no_ledger_write": True,
            "payloads_json_safe": True,
            "input_payloads_are_dicts": True,
            "simulated_results_are_dicts": True,
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
        "readiness_for_stage4j4": {
            "ready_to_build_controlled_paper_operation_execution_gate": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def _dry_run_trace(operation_id: str) -> list[dict[str, object]]:
    stages = [
        "pre_operation_snapshot_check",
        "risk_gate_check",
        "selected_strategy_operation_preview",
    ]
    return [
        {
            "sequence_number": index,
            "source_stage": "4J-2",
            "dry_run_stage": "4J-3",
            "target_component": f"{stage}_component",
            "input_payload": {
                "operation_id": operation_id,
                "selected_strategy_id": "S01",
                "stage": stage,
                "preview_only": True,
            },
            "simulated_result": {"simulated_pass": True, "stage": stage},
            "would_execute": False,
            "would_call_strategy": False,
            "would_fetch_market_data": False,
            "would_qualify_contracts": False,
            "would_create_intent": False,
            "would_create_ticket": False,
            "would_submit_order": False,
            "would_write_state": False,
            "would_write_ledger": False,
            "paper_only": True,
            "live_trading_enabled": False,
            "status": "simulated",
        }
        for index, stage in enumerate(stages, start=1)
    ]


def scheduler_activation_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "scheduler_activation_record": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
            "live_trading_enabled": False,
            "all_strategies_enabled": False,
            "broker_submission_enabled": False,
            "strategy_scan_execution_enabled": False,
        }
    }
    _deep_update(snapshot, overrides)
    return snapshot


def lifecycle_activation_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "lifecycle_activation_record": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
            "live_trading_enabled": False,
            "all_strategies_enabled": False,
            "broker_submission_enabled": False,
            "lifecycle_transition_execution_enabled": False,
        }
    }
    _deep_update(snapshot, overrides)
    return snapshot


def activation_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "activation_record": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "live_trading_enabled": False,
            "all_strategies_enabled": False,
            "broker_submission_enabled": False,
        },
        "active_strategy_ids": ["S01"],
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


_DEFAULT_J3 = object()


def build(j3: object = _DEFAULT_J3, **kwargs: object) -> dict:
    acknowledgements = kwargs.pop("operator_acknowledgements", list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS))
    return build_stage4j4_controlled_paper_operation_execution_gate_report(
        stage4j3_dry_run_report=valid_stage4j3_report() if j3 is _DEFAULT_J3 else j3,  # type: ignore[arg-type]
        operator_acknowledgements=acknowledgements,  # type: ignore[arg-type]
        now_provider=lambda: datetime(2026, 5, 16, 14, 0, tzinfo=timezone.utc),
        **kwargs,
    )


def build_with_snapshots(**kwargs: object) -> dict:
    defaults: dict[str, object] = {
        "scheduler_activation_snapshot": scheduler_activation_snapshot(),
        "lifecycle_activation_snapshot": lifecycle_activation_snapshot(),
        "activation_snapshot": activation_snapshot(),
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
    return report["readiness_for_stage4j5"]["ready_to_build_controlled_paper_operation_executor"]


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class CustomPayload:
    pass


class Stage4J4ControlledPaperOperationExecutionGateTests(unittest.TestCase):
    def test_stage4j3_gating_and_required_fields(self) -> None:
        self.assertFalse(ready(build(j3=None)))
        self.assertFalse(
            ready(
                build(
                    j3=valid_stage4j3_report(
                        readiness_for_stage4j4={
                            "ready_to_build_controlled_paper_operation_execution_gate": False
                        }
                    )
                )
            )
        )
        self.assertFalse(ready(build(j3=valid_stage4j3_report(selected_strategy=None))))
        self.assertFalse(ready(build(j3=valid_stage4j3_report(operation={"operation_id": None}))))
        self.assertFalse(ready(build(j3=valid_stage4j3_report(dry_run_trace=[]))))

    def test_malformed_trace_blocks_without_crashing(self) -> None:
        cases = []
        malformed = valid_stage4j3_report(dry_run_trace="bad")
        cases.append(malformed)
        for mutation in (
            lambda item: "bad",
            lambda item: {key: value for key, value in item.items() if key != "input_payload"},
            lambda item: {key: value for key, value in item.items() if key != "simulated_result"},
            lambda item: {**item, "input_payload": "{}"},
            lambda item: {**item, "simulated_result": "{}"},
            lambda item: {**item, "input_payload": {"unsafe": Decimal("1.1")}},
            lambda item: {**item, "simulated_result": {"unsafe": CustomPayload()}},
        ):
            report = valid_stage4j3_report()
            trace = report["dry_run_trace"]  # type: ignore[index]
            trace[0] = mutation(trace[0])  # type: ignore[index]
            cases.append(report)
        for report in cases:
            with self.subTest(report=report):
                rendered = build(j3=report)
                self.assertFalse(ready(rendered))
                json.dumps(rendered)

    def test_trace_disabled_flags_are_strict_false(self) -> None:
        for key, value in (
            ("status", "executed"),
            ("would_execute", True),
            ("would_execute", "False"),
            ("would_call_strategy", True),
            ("would_fetch_market_data", True),
            ("would_qualify_contracts", True),
            ("would_create_intent", True),
            ("would_create_ticket", True),
            ("would_submit_order", True),
            ("would_write_state", True),
            ("would_write_ledger", True),
        ):
            report = valid_stage4j3_report()
            trace = report["dry_run_trace"]  # type: ignore[index]
            trace[0][key] = value  # type: ignore[index]
            with self.subTest(key=key, value=value):
                self.assertFalse(ready(build(j3=report)))

    def test_operator_acknowledgements_are_exact_list_items(self) -> None:
        report = build(operator_acknowledgements=None)
        self.assertFalse(ready(report))
        self.assertEqual(report["acknowledgement_checks"]["provided"], [])

        giant = " ".join(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS)
        self.assertFalse(ready(build(operator_acknowledgements=[giant])))

        extras_do_not_compensate = REQUIRED_OPERATOR_ACKNOWLEDGEMENTS[:-1] + ["extra"]
        self.assertFalse(ready(build(operator_acknowledgements=extras_do_not_compensate)))

        padded = [f" {value} " for value in REQUIRED_OPERATOR_ACKNOWLEDGEMENTS]
        self.assertTrue(ready(build(operator_acknowledgements=padded)))

    def test_permissions_are_tightly_scoped(self) -> None:
        report = build()
        permissions = report["proposed_execution_permissions_for_4J5"]
        self.assertTrue(permissions["may_build_executor_next_phase"])
        self.assertTrue(permissions["may_call_strategy_next_phase"])
        for key in (
            "may_fetch_market_data_next_phase",
            "may_qualify_contracts_next_phase",
            "may_create_intent_next_phase",
            "may_create_ticket_next_phase",
            "may_submit_order_next_phase",
            "may_write_state_next_phase",
            "may_write_ledger_next_phase",
            "live_trading_enabled",
            "all_strategies_enabled",
            "broker_submission_enabled",
        ):
            self.assertFalse(permissions[key])

        blocked = build(operator_acknowledgements=[])
        self.assertFalse(blocked["proposed_execution_permissions_for_4J5"]["may_build_executor_next_phase"])
        self.assertFalse(blocked["proposed_execution_permissions_for_4J5"]["may_call_strategy_next_phase"])

    def test_activation_snapshot_blockers(self) -> None:
        self.assertTrue(ready(build_with_snapshots(scheduler_activation_snapshot=scheduler_activation_snapshot())))
        self.assertFalse(ready(build_with_snapshots(scheduler_activation_snapshot=scheduler_activation_snapshot(scheduler_activation_record={"selected_strategy_id": "S02"}))))
        self.assertFalse(ready(build_with_snapshots(scheduler_activation_snapshot=scheduler_activation_snapshot(scheduler_activation_record={"strategy_scan_execution_enabled": True}))))
        self.assertTrue(ready(build_with_snapshots(lifecycle_activation_snapshot=lifecycle_activation_snapshot())))
        self.assertFalse(ready(build_with_snapshots(lifecycle_activation_snapshot=lifecycle_activation_snapshot(lifecycle_activation_record={"selected_strategy_id": "S02"}))))
        self.assertFalse(ready(build_with_snapshots(lifecycle_activation_snapshot=lifecycle_activation_snapshot(lifecycle_activation_record={"lifecycle_transition_execution_enabled": True}))))
        self.assertTrue(ready(build_with_snapshots(activation_snapshot=activation_snapshot())))
        self.assertFalse(ready(build_with_snapshots(activation_snapshot=activation_snapshot(active_strategy_ids=["S01", "S02"]))))
        self.assertFalse(ready(build_with_snapshots(activation_snapshot=activation_snapshot(active_strategy_ids=["S02"]))))

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
            {"scheduler_snapshot": clean_scheduler_snapshot(jobs=[{"strategy_id": "S01", "broker_submission_enabled": True}])},
            {"lifecycle_snapshot": clean_lifecycle_snapshot(all_strategies_enabled=True)},
            {"lifecycle_snapshot": clean_lifecycle_snapshot(lifecycle_transition_execution_enabled=True)},
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

        safe_intents = build_with_snapshots(
            state_snapshot=clean_state_snapshot(active_intents_count=1, active_intents_safe_for_enablement=True)
        )
        self.assertTrue(ready(safe_intents))
        self.assertIn("active intents present", " ".join(safe_intents["warnings"]))

        closed_market = build_with_snapshots(market_window_snapshot=clean_market_window_snapshot(market_open=False))
        self.assertTrue(ready(closed_market))
        self.assertIn("market is currently closed", " ".join(closed_market["warnings"]))

    def test_valid_missing_optional_snapshots_ready_with_warnings_and_report_shape(self) -> None:
        report = build()
        self.assertTrue(ready(report))
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, report["warnings"])
        for key in (
            "dry_run",
            "stage4j4_controlled_paper_operation_execution_gate_report",
            "generated_at",
            "artifact_checks",
            "selected_strategy",
            "operation",
            "acknowledgement_checks",
            "execution_gate",
            "proposed_execution_permissions_for_4J5",
            "proposed_pre_execution_checks",
            "proposed_execution_trace_requirements",
            "proposed_post_execution_checks",
            "disabled_components",
            "required_inputs_for_4J5",
            "dry_run_trace_checks",
            "activation_snapshot_checks",
            "state_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "paper_broker_checks",
            "market_window_checks",
            "safety_checks",
            "readiness_for_stage4j5",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        self.assertTrue(report["execution_gate"]["ready_for_4J5"])
        json.dumps(report)

    def test_recommendations_remain_conservative_and_inputs_not_mutated(self) -> None:
        report_input = valid_stage4j3_report()
        snapshot = clean_state_snapshot(open_positions_count=7)
        original_report = copy.deepcopy(report_input)
        original_snapshot = copy.deepcopy(snapshot)
        report = build(j3=report_input, state_snapshot=snapshot)
        self.assertEqual(report_input, original_report)
        self.assertEqual(snapshot, original_snapshot)

        text = " ".join(report["recommendations"]["ordered_next_steps"])
        disallowed = (
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
        )
        for phrase in disallowed:
            self.assertNotIn(phrase, text.lower())

    def test_cli_requires_dry_run_json_is_strict_ack_append_and_actions_are_not_exposed(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            rc = tool.run_stage4j4_controlled_paper_operation_execution_gate(
                ["--stage4j3-dry-run-json", json.dumps(valid_stage4j3_report())]
            )
        self.assertNotEqual(rc, 0)
        self.assertIn("--dry-run-only", stderr.getvalue())

        stdout = io.StringIO()
        args = [
            "--dry-run-only",
            "--json",
            "--stage4j3-dry-run-json",
            json.dumps(valid_stage4j3_report()),
        ]
        for ack in REQUIRED_OPERATOR_ACKNOWLEDGEMENTS:
            args.extend(["--ack", ack])
        with redirect_stdout(stdout):
            rc = tool.run_stage4j4_controlled_paper_operation_execution_gate(args)
        self.assertEqual(rc, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4j4_controlled_paper_operation_execution_gate_report"])

        parser_source = Path(tool.__file__).read_text()
        self.assertIn('parser.add_argument("--ack", action="append"', parser_source)
        namespace = argparse.Namespace()
        self.assertFalse(hasattr(namespace, "submit"))

        source = "\n".join(path.read_text() for path in STAGE4J4_FILES)
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
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
