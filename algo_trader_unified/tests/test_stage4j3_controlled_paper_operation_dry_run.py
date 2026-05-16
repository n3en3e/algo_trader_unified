from __future__ import annotations

import copy
from datetime import datetime, timezone
from enum import Enum
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core import stage4j3_controlled_paper_operation_dry_run as core
from algo_trader_unified.core.stage4j3_controlled_paper_operation_dry_run import (
    MARKET_WINDOW_MANUAL_WARNING,
    build_stage4j3_controlled_paper_operation_dry_run_report,
)
from algo_trader_unified.tools import stage4j3_controlled_paper_operation_dry_run as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4J3_FILES = [
    ROOT / "core/stage4j3_controlled_paper_operation_dry_run.py",
    ROOT / "tools/stage4j3_controlled_paper_operation_dry_run.py",
]
FLOW_STAGES = list(core.EXPECTED_SIMULATED_RESULTS)


def valid_stage4j2_report(**overrides: object) -> dict:
    operation_id = "s01_daily_2026_05_16"
    report: dict[str, object] = {
        "dry_run": True,
        "stage4j2_controlled_paper_operation_plan_report": True,
        "generated_at": "2026-05-16T12:30:00+00:00",
        "selected_strategy": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
        },
        "payload_checks": {
            "broker_submission_disabled": True,
            "strategy_scan_execution_disabled": True,
            "lifecycle_transition_execution_disabled": True,
            "market_data_disabled": True,
            "contract_qualification_disabled": True,
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
            "no_direct_scheduler_registration": True,
            "no_direct_lifecycle_execution": True,
            "no_state_write": True,
            "no_ledger_write": True,
        },
        "operation_plan": {
            "available": True,
            "operation_id": operation_id,
            "controlled_operation_scope": "single_strategy_controlled_scheduled_paper_operation",
            "proposed_operation_window": {
                "operation_id": operation_id,
                "dry_run_only": True,
                "would_register_scheduler": False,
                "would_execute_operation": False,
                "would_submit_orders": False,
                "paper_only": True,
                "live_trading_enabled": False,
            },
            "proposed_operation_flow": _operation_flow(operation_id),
            "proposed_post_operation_checks": [
                {"check": "dry_run_plan_summary_review", "would_execute": False},
                {"check": "post_operation_reconciliation_preview", "would_execute": False},
            ],
        },
        "readiness_for_stage4j3": {
            "ready_to_build_controlled_paper_operation_dry_run": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def _operation_flow(operation_id: str) -> list[dict[str, object]]:
    flow: list[dict[str, object]] = []
    for index, stage in enumerate(FLOW_STAGES, start=1):
        flow.append(
            {
                "sequence_number": index,
                "stage": stage,
                "target_component": f"{stage}_component",
                "payload": {
                    "operation_id": operation_id,
                    "selected_strategy_id": "S01",
                    "stage": stage,
                    "preview_only": True,
                },
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
            }
        )
    flow[1].pop("target_component")
    flow[1]["target_function"] = "risk_gate_check_function"
    return flow


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


_DEFAULT_J2 = object()


def build(j2: object = _DEFAULT_J2, **kwargs: object) -> dict:
    return build_stage4j3_controlled_paper_operation_dry_run_report(
        stage4j2_operation_plan_report=valid_stage4j2_report() if j2 is _DEFAULT_J2 else j2,  # type: ignore[arg-type]
        now_provider=lambda: datetime(2026, 5, 16, 13, 0, tzinfo=timezone.utc),
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
    return report["readiness_for_stage4j4"]["ready_to_build_controlled_paper_operation_execution_gate"]


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class PayloadEnum(Enum):
    VALUE = "value"


class CustomPayload:
    pass


class Stage4J3ControlledPaperOperationDryRunTests(unittest.TestCase):
    def test_stage4j2_gating_and_required_fields(self) -> None:
        self.assertFalse(ready(build(j2=None)))
        self.assertFalse(
            ready(
                build(
                    j2=valid_stage4j2_report(
                        readiness_for_stage4j3={
                            "ready_to_build_controlled_paper_operation_dry_run": False
                        }
                    )
                )
            )
        )
        self.assertFalse(ready(build(j2=valid_stage4j2_report(selected_strategy=None))))
        self.assertFalse(
            ready(build(j2=valid_stage4j2_report(operation_plan={"operation_id": None})))
        )
        self.assertFalse(
            ready(build(j2=valid_stage4j2_report(operation_plan={"proposed_operation_flow": []})))
        )

    def test_flow_validation_blocks_malformed_steps_without_crashing(self) -> None:
        cases = []
        report = valid_stage4j2_report()
        report["operation_plan"]["proposed_operation_flow"] = "bad"  # type: ignore[index]
        cases.append(report)

        for mutation in (
            lambda step: "bad",
            lambda step: {key: value for key, value in step.items() if key != "target_component"},
            lambda step: {**step, "payload": "{}"},
            lambda step: {key: value for key, value in step.items() if key != "sequence_number"},
        ):
            mutated = valid_stage4j2_report()
            flow = mutated["operation_plan"]["proposed_operation_flow"]  # type: ignore[index]
            flow[0] = mutation(flow[0])  # type: ignore[index]
            cases.append(mutated)

        for report in cases:
            with self.subTest(report=report):
                self.assertFalse(ready(build(j2=report)))

    def test_payload_validator_rejects_non_json_safe_values(self) -> None:
        unsafe_values = [
            datetime(2026, 5, 16, tzinfo=timezone.utc),
            ("tuple",),
            PayloadEnum.VALUE,
            CustomPayload(),
            lambda: None,
            Path("x"),
        ]
        for value in unsafe_values:
            mutated = valid_stage4j2_report()
            flow = mutated["operation_plan"]["proposed_operation_flow"]  # type: ignore[index]
            flow[0]["payload"]["unsafe"] = value  # type: ignore[index]
            with self.subTest(value=type(value).__name__):
                report = build(j2=mutated)
                self.assertFalse(ready(report))
                json.dumps(report)

    def test_trace_preserves_order_targets_and_native_dicts(self) -> None:
        report = build_with_snapshots()
        self.assertTrue(ready(report))
        trace = report["dry_run_trace"]
        self.assertEqual([item["sequence_number"] for item in trace], list(range(1, len(FLOW_STAGES) + 1)))
        self.assertIn("target_component", trace[0])
        self.assertIn("target_function", trace[1])
        for item in trace:
            self.assertEqual(item["source_stage"], "4J-2")
            self.assertEqual(item["dry_run_stage"], "4J-3")
            self.assertEqual(item["status"], "simulated")
            self.assertIsInstance(item["input_payload"], dict)
            self.assertIsInstance(item["simulated_result"], dict)
            self.assertNotIsInstance(item["input_payload"], str)
            self.assertNotIsInstance(item["simulated_result"], str)
            self.assertFalse(item["would_execute"])
            self.assertFalse(item["would_write_state"])
            self.assertFalse(item["would_write_ledger"])
            self.assertFalse(item["live_trading_enabled"])

    def test_unsafe_flow_boolean_flags_block_strictly(self) -> None:
        for key, value in (
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
            mutated = valid_stage4j2_report()
            flow = mutated["operation_plan"]["proposed_operation_flow"]  # type: ignore[index]
            flow[0][key] = value  # type: ignore[index]
            with self.subTest(key=key, value=value):
                self.assertFalse(ready(build(j2=mutated)))

    def test_exact_flat_simulated_result_shapes(self) -> None:
        report = build_with_snapshots()
        by_stage = {
            item["input_payload"]["stage"]: item["simulated_result"]
            for item in report["dry_run_trace"]
        }
        for stage, expected in core.EXPECTED_SIMULATED_RESULTS.items():
            result = by_stage[stage]
            self.assertEqual(set(result), set(expected))
            for key, value in result.items():
                self.assertTrue(core._primitive_json_safe(value))
                if key.endswith("_placeholder"):
                    self.assertIsInstance(value, str)
                    self.assertNotIsInstance(value, dict)
        self.assertEqual(
            by_stage["selected_strategy_operation_preview"]["strategy_call_placeholder"],
            "not_executed_in_4J3",
        )
        self.assertEqual(
            by_stage["market_data_gate_preview"]["market_data_placeholder"],
            "not_fetched_in_4J3",
        )
        self.assertEqual(
            by_stage["contract_qualification_gate_preview"]["contract_qualification_placeholder"],
            "not_qualified_in_4J3",
        )
        self.assertEqual(
            by_stage["strategy_scan_gate_preview"]["strategy_scan_placeholder"],
            "not_executed_in_4J3",
        )
        self.assertEqual(
            by_stage["signal_to_intent_gate_preview"]["intent_placeholder"],
            "not_created_in_4J3",
        )
        self.assertEqual(
            by_stage["order_ticket_gate_preview"]["ticket_placeholder"],
            "not_created_in_4J3",
        )
        self.assertEqual(
            by_stage["broker_submission_gate_preview"]["broker_submission_placeholder"],
            "blocked_in_4J3",
        )
        self.assertFalse(by_stage["state_ledger_tracking_gate_preview"]["would_write_state"])
        self.assertFalse(by_stage["state_ledger_tracking_gate_preview"]["would_write_ledger"])
        self.assertEqual(
            by_stage["alert_report_gate_preview"]["alert_placeholder"],
            "not_emitted_in_4J3",
        )
        self.assertEqual(
            by_stage["post_operation_reconciliation_gate_preview"]["reconciliation_placeholder"],
            "not_executed_in_4J3",
        )

    def test_activation_snapshot_blockers(self) -> None:
        self.assertTrue(ready(build_with_snapshots(scheduler_activation_snapshot=scheduler_activation_snapshot())))
        self.assertFalse(
            ready(
                build_with_snapshots(
                    scheduler_activation_snapshot=scheduler_activation_snapshot(
                        scheduler_activation_record={"selected_strategy_id": "S02"}
                    )
                )
            )
        )
        self.assertFalse(
            ready(
                build_with_snapshots(
                    scheduler_activation_snapshot=scheduler_activation_snapshot(
                        scheduler_activation_record={"strategy_scan_execution_enabled": True}
                    )
                )
            )
        )
        self.assertTrue(ready(build_with_snapshots(lifecycle_activation_snapshot=lifecycle_activation_snapshot())))
        self.assertFalse(
            ready(
                build_with_snapshots(
                    lifecycle_activation_snapshot=lifecycle_activation_snapshot(
                        lifecycle_activation_record={"selected_strategy_id": "S02"}
                    )
                )
            )
        )
        self.assertFalse(
            ready(
                build_with_snapshots(
                    lifecycle_activation_snapshot=lifecycle_activation_snapshot(
                        lifecycle_activation_record={"lifecycle_transition_execution_enabled": True}
                    )
                )
            )
        )
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
            "stage4j3_controlled_paper_operation_dry_run_report",
            "generated_at",
            "artifact_checks",
            "selected_strategy",
            "controlled_operation_scope",
            "operation_id",
            "operation",
            "dry_run_trace",
            "dry_run_trace_checks",
            "simulated_pre_operation_gates",
            "simulated_operation_results",
            "simulated_post_operation_checks",
            "disabled_components_confirmed",
            "required_inputs_for_4J4",
            "activation_snapshot_checks",
            "state_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "paper_broker_checks",
            "market_window_checks",
            "safety_checks",
            "readiness_for_stage4j4",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report)

    def test_recommendations_remain_conservative_and_inputs_not_mutated(self) -> None:
        report_input = valid_stage4j2_report()
        snapshot = clean_state_snapshot(open_positions_count=7)
        original_report = copy.deepcopy(report_input)
        original_snapshot = copy.deepcopy(snapshot)
        report = build(j2=report_input, state_snapshot=snapshot)
        self.assertEqual(report_input, original_report)
        self.assertEqual(snapshot, original_snapshot)

        text = " ".join(report["recommendations"]["ordered_next_steps"])
        for phrase in (
            "enable live trading",
            "enable all strategies",
            "place orders now",
            "run strategy scans now",
            "enable market data now",
            "enable contract qualification now",
            "enable broker submission broadly",
        ):
            self.assertNotIn(phrase, text.lower())

    def test_cli_requires_dry_run_json_is_strict_and_actions_are_not_exposed(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            rc = tool.run_stage4j3_controlled_paper_operation_dry_run(
                ["--stage4j2-plan-json", json.dumps(valid_stage4j2_report())]
            )
        self.assertNotEqual(rc, 0)
        self.assertIn("--dry-run-only", stderr.getvalue())

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4j3_controlled_paper_operation_dry_run(
                [
                    "--dry-run-only",
                    "--json",
                    "--stage4j2-plan-json",
                    json.dumps(valid_stage4j2_report()),
                ]
            )
        self.assertEqual(rc, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4j3_controlled_paper_operation_dry_run_report"])

        source = "\n".join(path.read_text() for path in STAGE4J3_FILES)
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
