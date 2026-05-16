from __future__ import annotations

import copy
from datetime import datetime, timezone
from enum import Enum
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core import stage4j2_controlled_paper_operation_plan as core
from algo_trader_unified.core.stage4j2_controlled_paper_operation_plan import (
    MARKET_WINDOW_MANUAL_WARNING,
    build_stage4j2_controlled_paper_operation_plan_report,
)
from algo_trader_unified.tools import stage4j2_controlled_paper_operation_plan as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4J2_FILES = [
    ROOT / "core/stage4j2_controlled_paper_operation_plan.py",
    ROOT / "tools/stage4j2_controlled_paper_operation_plan.py",
]


def valid_stage4j1_report(**overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4j1_controlled_paper_operation_readiness_report": True,
        "generated_at": "2026-05-16T12:00:00+00:00",
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
        "readiness_for_stage4j2": {
            "ready_to_build_controlled_paper_operation_plan": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


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


_DEFAULT_J1 = object()


def build(j1: object = _DEFAULT_J1, **kwargs: object) -> dict:
    return build_stage4j2_controlled_paper_operation_plan_report(
        stage4j1_readiness_report=valid_stage4j1_report() if j1 is _DEFAULT_J1 else j1,  # type: ignore[arg-type]
        now_provider=lambda: datetime(2026, 5, 16, 12, 30, tzinfo=timezone.utc),
        **kwargs,
    )


def build_with_snapshots(**kwargs: object) -> dict:
    defaults: dict[str, object] = {
        "operation_window_config": {"cadence": " daily ", "dry_run_only": True},
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
    return report["readiness_for_stage4j3"]["ready_to_build_controlled_paper_operation_dry_run"]


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


class Stage4J2ControlledPaperOperationPlanTests(unittest.TestCase):
    def test_stage4j1_report_gating_and_safe_selected_strategy_traversal(self) -> None:
        self.assertFalse(ready(build(j1=None)))
        self.assertFalse(ready(build(j1="bad")))
        self.assertFalse(
            ready(
                build(
                    j1=valid_stage4j1_report(
                        readiness_for_stage4j2={"ready_to_build_controlled_paper_operation_plan": False}
                    )
                )
            )
        )
        self.assertFalse(ready(build(j1=valid_stage4j1_report(selected_strategy=None))))
        malformed = build(j1={"selected_strategy": []})
        self.assertFalse(ready(malformed))
        self.assertIsNone(malformed["selected_strategy"]["selected_strategy_id"])

    def test_operation_window_config_validation_is_none_safe_and_deterministic(self) -> None:
        missing = build()
        self.assertTrue(ready(missing))
        self.assertIn("operation window config missing", " ".join(missing["warnings"]))
        self.assertEqual(missing["operation_window_checks"]["cadence"], "once")
        self.assertEqual(missing["operation_plan"]["operation_id"], "s01_once_2026_05_16")

        none_cadence = build(operation_window_config={"cadence": None})
        self.assertTrue(ready(none_cadence))
        self.assertEqual(none_cadence["operation_window_checks"]["cadence"], "once")

        missing_cadence = build(operation_window_config={})
        self.assertTrue(ready(missing_cadence))
        self.assertEqual(missing_cadence["operation_window_checks"]["cadence"], "once")

        supported = build(operation_window_config={"cadence": " MARKET_Open "})
        self.assertTrue(ready(supported))
        self.assertEqual(supported["operation_window_checks"]["cadence"], "market_open")
        self.assertEqual(supported["operation_plan"]["operation_id"], "s01_market_open_2026_05_16")

        self.assertFalse(ready(build(operation_window_config={"cadence": "weekly"})))
        self.assertFalse(ready(build(operation_window_config={"dry_run_only": False})))

    def test_valid_reports_are_ready_with_missing_optional_or_matching_snapshots(self) -> None:
        missing = build()
        self.assertTrue(ready(missing))
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, missing["warnings"])

        matching = build_with_snapshots()
        self.assertTrue(ready(matching))
        self.assertEqual(matching["operation_window_checks"]["cadence"], "daily")
        self.assertEqual(matching["operation_plan"]["operation_id"], "s01_daily_2026_05_16")

    def test_report_shape_flow_order_disabled_flags_and_json_safety(self) -> None:
        report = build_with_snapshots()
        self.assertTrue(ready(report))
        for key in (
            "dry_run",
            "stage4j2_controlled_paper_operation_plan_report",
            "generated_at",
            "artifact_checks",
            "selected_strategy",
            "controlled_operation_scope",
            "operation_plan",
            "operation_window_checks",
            "activation_snapshot_checks",
            "state_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "paper_broker_checks",
            "market_window_checks",
            "safety_checks",
            "readiness_for_stage4j3",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report)

        window = report["operation_plan"]["proposed_operation_window"]
        self.assertFalse(window["would_register_scheduler"])
        self.assertFalse(window["would_execute_operation"])
        self.assertFalse(window["would_submit_orders"])

        flow = report["operation_plan"]["proposed_operation_flow"]
        self.assertEqual([step["stage"] for step in flow], [stage for stage, _ in core.FLOW_STAGES])
        for index, step in enumerate(flow, start=1):
            self.assertEqual(step["sequence_number"], index)
            self.assertIsInstance(step["payload"], dict)
            self.assertTrue(core._primitive_json_safe(step["payload"]))
            for key in (
                "would_execute",
                "would_call_strategy",
                "would_fetch_market_data",
                "would_qualify_contracts",
                "would_create_intent",
                "would_create_ticket",
                "would_submit_order",
                "would_write_state",
                "would_write_ledger",
            ):
                self.assertIs(step[key], False)
            self.assertTrue(step["paper_only"])
            self.assertFalse(step["live_trading_enabled"])

    def test_operation_plan_blockers_catch_unsafe_mutations(self) -> None:
        plan = build_with_snapshots()["operation_plan"]
        for key in ("would_register_scheduler", "would_execute_operation", "would_submit_orders"):
            mutated = copy.deepcopy(plan)
            mutated["proposed_operation_window"][key] = True
            self.assertTrue(core._operation_plan_blockers(mutated))

        for key in (
            "would_execute",
            "would_call_strategy",
            "would_fetch_market_data",
            "would_qualify_contracts",
            "would_create_intent",
            "would_create_ticket",
            "would_submit_order",
            "would_write_state",
            "would_write_ledger",
        ):
            mutated = copy.deepcopy(plan)
            mutated["proposed_operation_flow"][0][key] = True
            self.assertTrue(core._operation_plan_blockers(mutated), key)

        mutated = copy.deepcopy(plan)
        mutated["proposed_operation_flow"][0]["would_execute"] = "False"
        self.assertTrue(core._operation_plan_blockers(mutated))
        mutated = copy.deepcopy(plan)
        mutated["proposed_operation_flow"][0] = "bad"
        self.assertTrue(core._operation_plan_blockers(mutated))

    def test_payload_validator_rejects_non_primitive_payload_data(self) -> None:
        unsafe_values = [
            datetime(2026, 5, 16, tzinfo=timezone.utc),
            ("tuple",),
            PayloadEnum.VALUE,
            CustomPayload(),
            lambda: None,
            Path("x"),
        ]
        for value in unsafe_values:
            with self.subTest(value=type(value).__name__):
                self.assertFalse(core._primitive_json_safe({"value": value}))

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

    def test_stage4j1_safety_and_recommendations_remain_conservative(self) -> None:
        unsafe = valid_stage4j1_report(safety_checks={"no_live_trading": False})
        self.assertFalse(ready(build(j1=unsafe)))
        text = " ".join(build_with_snapshots()["recommendations"]["ordered_next_steps"])
        forbidden_recommendations = [
            "enable live trading",
            "enable all strategies",
            "place orders now",
            "run strategy scans now",
            "enable market data now",
            "enable contract qualification now",
            "enable broker submission broadly",
        ]
        for phrase in forbidden_recommendations:
            self.assertNotIn(phrase, text.lower())

    def test_inputs_are_not_mutated(self) -> None:
        j1 = valid_stage4j1_report()
        snapshot = clean_state_snapshot(open_positions_count=7)
        original_j1 = copy.deepcopy(j1)
        original_snapshot = copy.deepcopy(snapshot)
        build(j1=j1, state_snapshot=snapshot)
        self.assertEqual(j1, original_j1)
        self.assertEqual(snapshot, original_snapshot)

    def test_cli_requires_dry_run_json_is_strict_and_actions_are_not_exposed(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            rc = tool.run_stage4j2_controlled_paper_operation_plan(
                ["--stage4j1-readiness-json", json.dumps(valid_stage4j1_report())]
            )
        self.assertNotEqual(rc, 0)
        self.assertIn("--dry-run-only", stderr.getvalue())

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4j2_controlled_paper_operation_plan(
                [
                    "--dry-run-only",
                    "--json",
                    "--stage4j1-readiness-json",
                    json.dumps(valid_stage4j1_report()),
                ]
            )
        self.assertEqual(rc, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4j2_controlled_paper_operation_plan_report"])

        help_text = tool.run_stage4j2_controlled_paper_operation_plan.__code__.co_names
        self.assertNotIn("submit", help_text)
        source = "\n".join(path.read_text() for path in STAGE4J2_FILES)
        for forbidden in (
            "submit_order_plan(",
            "get_order_status(",
            "cancel_order(",
            "placeOrder(",
            "cancelOrder(",
            "reqMktData(",
            "qualifyContracts(",
            ".save(",
            ".write(",
            "scheduler.add_job(",
            "asyncio.run(",
            "uuid.uuid4(",
            "random.",
            "time.time(",
            "datetime.now(",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
