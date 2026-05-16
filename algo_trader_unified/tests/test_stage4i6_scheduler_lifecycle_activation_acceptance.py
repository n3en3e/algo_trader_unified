from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
import json
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4i6_scheduler_lifecycle_activation_acceptance import (
    MARKET_WINDOW_MANUAL_WARNING,
    build_stage4i6_scheduler_lifecycle_activation_acceptance_report,
)
from algo_trader_unified.tools import stage4i6_scheduler_lifecycle_activation_acceptance as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4I6_FILES = [
    ROOT / "core/stage4i6_scheduler_lifecycle_activation_acceptance.py",
    ROOT / "tools/stage4i6_scheduler_lifecycle_activation_acceptance.py",
]


def valid_stage4i5_report(**overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4i5_scheduler_lifecycle_activation_executor_report": True,
        "generated_at": "2026-05-15T12:00:00+00:00",
        "selected_strategy": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
        },
        "scheduler_activation_payload": safe_payload(),
        "lifecycle_activation_payload": safe_payload(),
        "execution": {
            "attempted": True,
            "scheduler_activation_succeeded": True,
            "lifecycle_activation_succeeded": True,
            "audit_write_attempted": False,
            "audit_write_succeeded": False,
            "completed": True,
        },
        "applied_operations": [
            {"target": "scheduler_activation", "status": "succeeded"},
            {"target": "lifecycle_activation", "status": "succeeded"},
        ],
        "skipped_operations": [],
        "rollback": {"rollback_required": False},
        "readiness_for_stage4i6": {
            "ready_to_build_scheduler_lifecycle_activation_acceptance": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def safe_payload(**overrides: object) -> dict:
    payload: dict[str, object] = {
        "selected_strategy_id": "S01",
        "paper_only": True,
        "one_strategy_only": True,
        "activation_scope": "single_strategy_scheduled_paper_run",
        "source_stage": "4I-5",
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "broker_submission_enabled": False,
        "strategy_scan_execution_enabled": False,
        "lifecycle_transition_execution_enabled": False,
        "market_data_enabled": False,
        "contract_qualification_enabled": False,
    }
    _deep_update(payload, overrides)
    return payload


def scheduler_activation_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "scheduler_activation_record": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
            "scheduler_job_scope": "single_strategy",
            "live_trading_enabled": False,
            "all_strategies_enabled": False,
            "broker_submission_enabled": False,
            "strategy_scan_execution_enabled": False,
            "lifecycle_transition_execution_enabled": False,
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
            "lifecycle_scope": "single_strategy",
            "live_trading_enabled": False,
            "all_strategies_enabled": False,
            "broker_submission_enabled": False,
            "strategy_scan_execution_enabled": False,
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
        "open_positions_count": 2,
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


_DEFAULT_I5 = object()


def build(i5: object = _DEFAULT_I5, **kwargs: object) -> dict:
    return build_stage4i6_scheduler_lifecycle_activation_acceptance_report(
        stage4i5_activation_executor_report=valid_stage4i5_report() if i5 is _DEFAULT_I5 else i5,  # type: ignore[arg-type]
        now_provider=lambda: datetime(2026, 5, 15, 12, 30, tzinfo=timezone.utc),
        **kwargs,
    )


def build_with_snapshots(**kwargs: object) -> dict:
    defaults: dict[str, object] = {
        "scheduler_activation_snapshot": scheduler_activation_snapshot(),
        "lifecycle_activation_snapshot": lifecycle_activation_snapshot(),
        "audit_snapshot": {"events": [{"selected_strategy_id": "S01", "source_stage": "4I-5"}]},
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
    return report["readiness_for_next_phase"]["ready_to_proceed_after_stage4i"]


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class Stage4I6SchedulerLifecycleActivationAcceptanceTests(unittest.TestCase):
    def test_missing_and_not_ready_stage4i5_report_block_readiness(self) -> None:
        self.assertFalse(ready(build(i5=None)))
        self.assertFalse(
            ready(
                build(
                    i5=valid_stage4i5_report(
                        readiness_for_stage4i6={
                            "ready_to_build_scheduler_lifecycle_activation_acceptance": False
                        }
                    )
                )
            )
        )

    def test_executor_outcome_gates_block_readiness(self) -> None:
        cases = [
            {"execution": {"attempted": False}},
            {"execution": {"scheduler_activation_succeeded": False}},
            {"execution": {"lifecycle_activation_succeeded": False}},
            {"execution": {"completed": False}},
            {"rollback": {"rollback_required": True}},
            {
                "execution": {
                    "audit_write_attempted": True,
                    "audit_write_succeeded": False,
                }
            },
            {"selected_strategy": {"selected_strategy_id": None}},
        ]
        for override in cases:
            with self.subTest(override=override):
                self.assertFalse(ready(build(i5=valid_stage4i5_report(**override))))

    def test_payload_presence_and_safety_gates(self) -> None:
        cases = [
            {"scheduler_activation_payload": None},
            {"lifecycle_activation_payload": None},
            {"scheduler_activation_payload": safe_payload(selected_strategy_id="S02")},
            {"lifecycle_activation_payload": safe_payload(selected_strategy_id="S02")},
            {"scheduler_activation_payload": safe_payload(live_trading_enabled=True)},
            {"scheduler_activation_payload": safe_payload(live_trading_enabled="False")},
            {"scheduler_activation_payload": safe_payload(all_strategies_enabled=True)},
            {"scheduler_activation_payload": safe_payload(broker_submission_enabled=True)},
            {"scheduler_activation_payload": safe_payload(strategy_scan_execution_enabled=True)},
            {"scheduler_activation_payload": safe_payload(lifecycle_transition_execution_enabled=True)},
            {"lifecycle_activation_payload": safe_payload(live_trading_enabled=True)},
            {"lifecycle_activation_payload": safe_payload(all_strategies_enabled=True)},
            {"lifecycle_activation_payload": safe_payload(broker_submission_enabled=True)},
            {"lifecycle_activation_payload": safe_payload(strategy_scan_execution_enabled=True)},
            {"lifecycle_activation_payload": safe_payload(lifecycle_transition_execution_enabled=True)},
        ]
        for override in cases:
            with self.subTest(override=override):
                self.assertFalse(ready(build(i5=valid_stage4i5_report(**override))))

    def test_applied_and_skipped_operation_gates_are_safe_on_malformed_items(self) -> None:
        cases = [
            {"applied_operations": [{"target": "lifecycle_activation"}]},
            {"applied_operations": [{"target": "scheduler_activation"}]},
            {"applied_operations": [{"status": "succeeded"}, {"target": "scheduler_activation"}]},
            {"skipped_operations": [{"status": "skipped"}]},
        ]
        for override in cases:
            with self.subTest(override=override):
                self.assertFalse(ready(build(i5=valid_stage4i5_report(**override))))

    def test_scheduler_lifecycle_activation_snapshots(self) -> None:
        self.assertTrue(
            ready(
                build(
                    scheduler_activation_snapshot=scheduler_activation_snapshot(),
                    lifecycle_activation_snapshot=lifecycle_activation_snapshot(),
                )
            )
        )
        cases = [
            {"scheduler_activation_snapshot": scheduler_activation_snapshot(scheduler_activation_record={"selected_strategy_id": "S02"})},
            {"scheduler_activation_snapshot": scheduler_activation_snapshot(scheduler_activation_record={"strategy_scan_execution_enabled": True})},
            {"lifecycle_activation_snapshot": lifecycle_activation_snapshot(lifecycle_activation_record={"selected_strategy_id": "S02"})},
            {"lifecycle_activation_snapshot": lifecycle_activation_snapshot(lifecycle_activation_record={"lifecycle_transition_execution_enabled": True})},
            {"scheduler_activation_snapshot": {"scheduler_activations": [None, {"selected_strategy_id": "S02"}]}},
            {"lifecycle_activation_snapshot": {"lifecycle_activations": [None, {"selected_strategy_id": "S02"}]}},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                self.assertFalse(ready(build(**kwargs)))

    def test_audit_snapshot_matching_and_malformed_schemas(self) -> None:
        for audit_snapshot in (
            {"events": [{"selected_strategy_id": "S01"}]},
            {"events": [{"payload": {"selected_strategy_id": "S01"}}]},
            {"events": [{"data": {"selected_strategy_id": "S01"}}]},
            {"events": [None, {"payload": "bad"}, {"data": {"selected_strategy_id": "S01"}}]},
        ):
            with self.subTest(audit_snapshot=audit_snapshot):
                self.assertTrue(ready(build(audit_snapshot=audit_snapshot)))
        self.assertFalse(ready(build(audit_snapshot={"events": [{"selected_strategy_id": "S02"}]})))

    def test_missing_activation_snapshots_warn_but_clean_executor_may_be_ready(self) -> None:
        report = build()
        self.assertTrue(ready(report))
        self.assertTrue(report["warnings"])
        self.assertTrue(report["snapshot_checks"]["scheduler_activation_snapshot_matches"])
        self.assertTrue(report["snapshot_checks"]["lifecycle_activation_snapshot_matches"])

    def test_activation_snapshot_rules(self) -> None:
        self.assertTrue(ready(build(activation_snapshot=activation_snapshot())))
        self.assertFalse(
            ready(build(activation_snapshot=activation_snapshot(active_strategy_ids=["S01", "S02"])))
        )
        self.assertFalse(
            ready(
                build(
                    activation_snapshot=activation_snapshot(
                        activation_record={"selected_strategy_id": "S02"}
                    )
                )
            )
        )

    def test_state_risk_scheduler_lifecycle_broker_market_blockers(self) -> None:
        cases = [
            {"state_snapshot": clean_state_snapshot(active_halt=True)},
            {
                "state_snapshot": {
                    "active_halt": False,
                    "needs_reconciliation_count": 1,
                    "active_intents_count": 0,
                    "open_positions_count": 2,
                }
            },
            {"state_snapshot": clean_state_snapshot(active_intents_count=1)},
            {"scheduler_snapshot": {"all_strategy_scheduler_enabled": True}},
            {
                "scheduler_snapshot": clean_scheduler_snapshot(
                    jobs=[{"strategy_id": "S01", "broker_submission_enabled": True}]
                )
            },
            {"lifecycle_snapshot": clean_lifecycle_snapshot(selected_strategy_id="S02")},
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
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                self.assertFalse(ready(build(**kwargs)))

        safe_intents = build(state_snapshot=clean_state_snapshot(active_intents_count=1, active_intents_safe_for_enablement=True))
        self.assertTrue(ready(safe_intents))
        self.assertTrue(any("active intents present" in item for item in safe_intents["warnings"]))

    def test_matching_scheduler_and_lifecycle_snapshots_are_allowed(self) -> None:
        report = build_with_snapshots()
        self.assertTrue(ready(report))
        self.assertTrue(report["scheduler_checks"]["scheduler_already_enabled"])
        self.assertTrue(report["scheduler_checks"]["selected_strategy_job_matches"])
        self.assertTrue(report["lifecycle_checks"]["lifecycle_already_enabled"])
        self.assertTrue(report["lifecycle_checks"]["lifecycle_matches_selected_strategy"])

    def test_market_window_warnings(self) -> None:
        missing = build()
        self.assertTrue(ready(missing))
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, missing["warnings"])
        closed = build(market_window_snapshot=clean_market_window_snapshot(market_open=False))
        self.assertTrue(ready(closed))
        self.assertTrue(closed["warnings"])
        holiday = build(market_window_snapshot=clean_market_window_snapshot(is_trading_day=False))
        self.assertTrue(ready(holiday))
        self.assertTrue(holiday["warnings"])

    def test_valid_clean_report_with_missing_and_matching_snapshots(self) -> None:
        self.assertTrue(ready(build()))
        self.assertTrue(ready(build_with_snapshots()))

    def test_recommendations_avoid_disallowed_actions(self) -> None:
        text = json.dumps(build()["recommendations"]).lower()
        for forbidden in (
            "enable live trading",
            "enable all strategies",
            "place orders now",
            "bypass risk controls",
            "enable broker submission broadly",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertIn(f"do not {forbidden}", text)
        self.assertNotIn("proceed to live trading", text)

    def test_report_fields_json_safe_and_inputs_not_mutated(self) -> None:
        inputs = {
            "i5": valid_stage4i5_report(),
            "scheduler_activation_snapshot": scheduler_activation_snapshot(),
            "lifecycle_activation_snapshot": lifecycle_activation_snapshot(),
            "audit_snapshot": {"events": [{"payload": {"selected_strategy_id": "S01"}}]},
            "activation_snapshot": activation_snapshot(),
            "state_snapshot": clean_state_snapshot(),
            "risk_snapshot": clean_risk_snapshot(),
            "scheduler_snapshot": clean_scheduler_snapshot(),
            "lifecycle_snapshot": clean_lifecycle_snapshot(),
            "paper_broker_snapshot": clean_paper_broker_snapshot(),
            "market_window_snapshot": clean_market_window_snapshot(),
        }
        before = copy.deepcopy(inputs)
        report = build(**inputs)
        for key in (
            "dry_run",
            "stage4i6_scheduler_lifecycle_activation_acceptance_report",
            "generated_at",
            "artifact_checks",
            "selected_strategy",
            "payload_checks",
            "applied_operation_checks",
            "snapshot_checks",
            "state_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "paper_broker_checks",
            "market_window_checks",
            "safety_checks",
            "readiness_for_next_phase",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report, sort_keys=True)
        self.assertEqual(before, inputs)

    def test_cli_requires_dry_run_only_before_parsing_and_json_is_strict(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = tool.run_stage4i6_scheduler_lifecycle_activation_acceptance(
                ["--stage4i5-executor-json", "{bad"]
            )
        self.assertEqual(1, code)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("--dry-run-only", stderr.getvalue())

        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = tool.run_stage4i6_scheduler_lifecycle_activation_acceptance(
                [
                    "--dry-run-only",
                    "--json",
                    "--stage4i5-executor-json",
                    json.dumps(valid_stage4i5_report()),
                ]
            )
        self.assertEqual(0, code)
        self.assertEqual("", stderr.getvalue())
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4i6_scheduler_lifecycle_activation_acceptance_report"])

        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = tool.run_stage4i6_scheduler_lifecycle_activation_acceptance(
                ["--dry-run-only", "--json", "--stage4i5-executor-json", "{bad"]
            )
        self.assertEqual(1, code)
        self.assertIn("JSONDecodeError", json.loads(stdout.getvalue())["errors"][0])

    def test_cli_exposes_no_forbidden_action_options(self) -> None:
        parser_source = (ROOT / "tools/stage4i6_scheduler_lifecycle_activation_acceptance.py").read_text()
        for forbidden in (
            "--submit",
            "--cancel",
            "--status",
            "--market-data",
            "--qualify",
            "--scheduler-enable",
            "--lifecycle-enable",
        ):
            self.assertNotIn(forbidden, parser_source)

    def test_no_forbidden_calls_or_production_wiring_in_stage4i6_files(self) -> None:
        source = "\n".join(path.read_text() for path in STAGE4I6_FILES)
        forbidden_patterns = [
            r"\bsubmit_order_plan\s*\(",
            r"\bget_order_status\s*\(",
            r"\bcancel_order\s*\(",
            r"\bplaceOrder\s*\(",
            r"\bcancelOrder\s*\(",
            r"\breqMktData\s*\(",
            r"\bqualifyContracts\s*\(",
            r"\bStateStore\s*\([^)]*\)\.(?:save|write|update)",
            r"\bledger\.(?:append|write)\s*\(",
            r"\bscheduler\.add_job\s*\(",
            r"\badd_job\s*\(",
            r"\bstart\s*\(",
            r"\brun_scan\s*\(",
            r"\bscan_now\s*\(",
            r"\byfinance\b",
            r"\brequests\b",
            r"\burllib\b",
            r"\bsystemctl\b",
            r"\bsystemd\b",
            r"\bsocket\.create_connection\s*\(",
            r"\bsocket\.socket\s*\(",
            r"\basyncio\.run\s*\(",
            r"\basyncio\.get_event_loop\s*\(",
            r"\basyncio\.new_event_loop\s*\(",
            r"\buuid\.uuid4\s*\(",
            r"\brandom\.",
            r"\btime\.time\s*\(",
            r"\bdatetime\.now\s*\(",
            r"\bStateStore\s*\(",
            r"\bib_insync\b",
        ]
        for pattern in forbidden_patterns:
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, source))


if __name__ == "__main__":
    unittest.main()
