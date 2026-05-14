from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4h6_one_strategy_activation_acceptance import (
    build_stage4h6_one_strategy_activation_acceptance_report,
)
from algo_trader_unified.tools import stage4h6_one_strategy_activation_acceptance as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4H6_FILES = [
    ROOT / "core/stage4h6_one_strategy_activation_acceptance.py",
    ROOT / "tools/stage4h6_one_strategy_activation_acceptance.py",
]


def valid_activation_payload(**overrides: object) -> dict:
    payload: dict[str, object] = {
        "selected_strategy_id": "S01",
        "paper_only": True,
        "activation_scope": "single_strategy_paper_only",
        "enabled_strategy_count": 1,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "broker_submission_enabled": False,
        "automated_paper_trading_enabled_for_selected_strategy": True,
        "scheduler_enabled_for_selected_strategy": True,
        "lifecycle_enabled_for_selected_strategy": True,
        "source_stage": "4H-5",
        "required_runtime_guards": [{"name": "paper-only", "required": True}],
        "required_monitoring": [{"name": "selected-strategy-health", "required": True}],
        "required_kill_switches": [{"name": "operator-halt", "required": True}],
    }
    _deep_update(payload, overrides)
    return payload


def valid_stage4h5_report(**overrides: object) -> dict:
    payload = valid_activation_payload()
    report: dict[str, object] = {
        "dry_run": False,
        "stage4h5_one_strategy_activation_executor_report": True,
        "generated_at": "2026-05-14T14:00:00+00:00",
        "selected_strategy": {"selected_strategy_id": "S01", "paper_only": True},
        "activation_payload": payload,
        "execution": {
            "attempted": True,
            "activation_write_attempted": True,
            "activation_write_succeeded": True,
            "audit_write_attempted": True,
            "audit_write_succeeded": True,
            "completed": True,
            "failed_step": None,
            "failure_reason": None,
        },
        "applied_operations": [
            {"operation": "activation_write", "payload": copy.deepcopy(payload)}
        ],
        "skipped_operations": [],
        "rollback": {"rollback_required": False},
        "readiness_for_stage4h6": {
            "ready_to_build_one_strategy_activation_acceptance_report": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


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


def matching_activation_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "activation_record": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "enabled_strategy_count": 1,
            "live_trading_enabled": False,
            "all_strategies_enabled": False,
            "broker_submission_enabled": False,
        },
        "active_strategy_ids": ["S01"],
    }
    _deep_update(snapshot, overrides)
    return snapshot


def matching_audit_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "selected_strategy_id": "S01",
        "source_stage": "4H-5",
        "events": [{"selected_strategy_id": "S01", "source_stage": "4H-5"}],
    }
    _deep_update(snapshot, overrides)
    return snapshot


def build(
    *,
    h5: dict | None = None,
    activation_snapshot: dict | None = None,
    audit_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
) -> dict:
    return build_stage4h6_one_strategy_activation_acceptance_report(
        stage4h5_activation_executor_report=valid_stage4h5_report() if h5 is None else h5,
        activation_snapshot=activation_snapshot,
        audit_snapshot=audit_snapshot,
        state_snapshot=state_snapshot,
        scheduler_snapshot=scheduler_snapshot,
        lifecycle_snapshot=lifecycle_snapshot,
        risk_snapshot=risk_snapshot,
        paper_broker_snapshot=paper_broker_snapshot,
        now_provider=lambda: datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc),
    )


def build_with_all_snapshots(**kwargs: object) -> dict:
    defaults: dict[str, object] = {
        "activation_snapshot": matching_activation_snapshot(),
        "audit_snapshot": matching_audit_snapshot(),
        "state_snapshot": clean_state_snapshot(),
        "scheduler_snapshot": {},
        "lifecycle_snapshot": {},
        "risk_snapshot": clean_risk_snapshot(),
        "paper_broker_snapshot": clean_paper_broker_snapshot(),
    }
    defaults.update(kwargs)
    return build(**defaults)


def assert_ready(test_case: unittest.TestCase, report: dict) -> None:
    test_case.assertTrue(
        report["readiness_for_next_phase"][
            "ready_to_build_first_scheduled_paper_automation_run"
        ]
    )


def assert_not_ready(test_case: unittest.TestCase, report: dict) -> None:
    test_case.assertFalse(
        report["readiness_for_next_phase"][
            "ready_to_build_first_scheduled_paper_automation_run"
        ]
    )
    test_case.assertTrue(report["readiness_for_next_phase"]["blockers"])


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class Stage4H6OneStrategyActivationAcceptanceTests(unittest.TestCase):
    def test_missing_stage4h5_report_blocks_readiness(self) -> None:
        report = build_stage4h6_one_strategy_activation_acceptance_report(
            stage4h5_activation_executor_report=None
        )
        assert_not_ready(self, report)

    def test_stage4h5_report_not_ready_blocks_readiness(self) -> None:
        report = build(
            h5=valid_stage4h5_report(
                readiness_for_stage4h6={
                    "ready_to_build_one_strategy_activation_acceptance_report": False
                }
            )
        )
        assert_not_ready(self, report)

    def test_execution_completed_false_blocks_readiness(self) -> None:
        report = build(h5=valid_stage4h5_report(execution={"completed": False}))
        assert_not_ready(self, report)

    def test_activation_write_succeeded_false_blocks_readiness(self) -> None:
        report = build(h5=valid_stage4h5_report(execution={"activation_write_succeeded": False}))
        assert_not_ready(self, report)

    def test_audit_write_succeeded_false_when_attempted_blocks_readiness(self) -> None:
        report = build(
            h5=valid_stage4h5_report(
                execution={"audit_write_attempted": True, "audit_write_succeeded": False}
            )
        )
        assert_not_ready(self, report)

    def test_rollback_required_blocks_readiness(self) -> None:
        report = build(h5=valid_stage4h5_report(rollback={"rollback_required": True}))
        assert_not_ready(self, report)

    def test_skipped_operations_non_empty_blocks_readiness(self) -> None:
        report = build(h5=valid_stage4h5_report(skipped_operations=["audit_writer"]))
        assert_not_ready(self, report)

    def test_missing_activation_payload_blocks_readiness(self) -> None:
        h5 = valid_stage4h5_report()
        h5.pop("activation_payload")
        report = build(h5=h5)
        assert_not_ready(self, report)

    def test_selected_strategy_mismatch_blocks_readiness(self) -> None:
        report = build(h5=valid_stage4h5_report(selected_strategy={"selected_strategy_id": "S02"}))
        assert_not_ready(self, report)

    def test_more_than_one_activated_strategy_blocks_readiness(self) -> None:
        report = build(
            h5=valid_stage4h5_report(
                activation_payload={
                    "enabled_strategy_count": 2,
                    "active_strategy_ids": ["S01", "S02"],
                }
            )
        )
        assert_not_ready(self, report)

    def test_activation_payload_safety_flags_block_readiness(self) -> None:
        cases = [
            {"paper_only": False},
            {"live_trading_enabled": True},
            {"all_strategies_enabled": True},
            {"broker_submission_enabled": True},
        ]
        for case in cases:
            with self.subTest(case=case):
                report = build(h5=valid_stage4h5_report(activation_payload=case))
                assert_not_ready(self, report)

    def test_applied_operations_payload_safety_flags_block_readiness(self) -> None:
        for key in ("live_trading_enabled", "broker_submission_enabled"):
            payload = valid_activation_payload(**{key: True})
            report = build(
                h5=valid_stage4h5_report(
                    applied_operations=[{"operation": "activation_write", "payload": payload}]
                )
            )
            with self.subTest(key=key):
                assert_not_ready(self, report)

    def test_applied_operations_with_malformed_payloads_do_not_crash(self) -> None:
        report = build(
            h5=valid_stage4h5_report(
                applied_operations=[
                    "activation_writer.activate_one_strategy",
                    {"operation": "activation_write", "payload": "not a dict"},
                    {"operation": "audit_write"},
                ]
            )
        )
        assert_ready(self, report)
        self.assertTrue(report["warnings"])

    def test_activation_snapshot_matching_payload_passes_snapshot_check(self) -> None:
        report = build_with_all_snapshots()
        assert_ready(self, report)
        self.assertTrue(report["snapshot_checks"]["activation_snapshot_matches"])

    def test_activation_snapshot_mismatch_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            activation_snapshot=matching_activation_snapshot(
                activation_record={"selected_strategy_id": "S02"}
            )
        )
        assert_not_ready(self, report)

    def test_activation_snapshot_active_strategy_ids_with_multiple_ids_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            activation_snapshot=matching_activation_snapshot(active_strategy_ids=["S01", "S02"])
        )
        assert_not_ready(self, report)

    def test_activation_snapshot_arrays_with_non_dict_entries_do_not_crash(self) -> None:
        report = build(
            activation_snapshot={
                "activations": [
                    "bad",
                    {
                        "selected_strategy_id": "S01",
                        "paper_only": True,
                        "enabled_strategy_count": 1,
                        "live_trading_enabled": False,
                        "broker_submission_enabled": False,
                    },
                ],
                "active_strategy_ids": ["S01"],
            }
        )
        assert_ready(self, report)

    def test_audit_snapshot_matching_selected_strategy_passes(self) -> None:
        report = build_with_all_snapshots()
        assert_ready(self, report)
        self.assertTrue(report["snapshot_checks"]["audit_snapshot_matches"])

    def test_audit_snapshot_mismatch_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            audit_snapshot=matching_audit_snapshot(selected_strategy_id="S02")
        )
        assert_not_ready(self, report)

    def test_audit_snapshot_events_arrays_with_non_dict_entries_do_not_crash(self) -> None:
        report = build(
            audit_snapshot={"events": ["bad", {"selected_strategy_id": "S01"}]}
        )
        assert_ready(self, report)

    def test_missing_optional_snapshots_warn_but_do_not_crash(self) -> None:
        report = build()
        assert_ready(self, report)
        self.assertGreaterEqual(len(report["warnings"]), 6)

    def test_state_snapshot_blockers(self) -> None:
        cases = [
            clean_state_snapshot(active_halt=True),
            clean_state_snapshot(unresolved_needs_reconciliation_count=1),
            {
                "active_halt": False,
                "needs_reconciliation_count": 1,
                "active_intents_count": 0,
                "open_positions_count": 2,
            },
            clean_state_snapshot(active_intents_count=1),
        ]
        for snapshot in cases:
            with self.subTest(snapshot=snapshot):
                report = build_with_all_snapshots(state_snapshot=snapshot)
                assert_not_ready(self, report)

    def test_active_intents_safe_for_enablement_warns_but_allows_readiness(self) -> None:
        report = build_with_all_snapshots(
            state_snapshot=clean_state_snapshot(
                active_intents_count=1,
                active_intents_safe_for_enablement=True,
            )
        )
        assert_ready(self, report)
        self.assertTrue(report["warnings"])

    def test_scheduler_broad_or_selected_automation_blocks_readiness(self) -> None:
        cases = [
            {"scheduler_automation_enabled": True},
            {"all_strategy_scheduler_enabled": True},
            {"jobs": [{"strategy_id": "S01", "disabled": False}]},
        ]
        for snapshot in cases:
            with self.subTest(snapshot=snapshot):
                report = build_with_all_snapshots(scheduler_snapshot=snapshot)
                assert_not_ready(self, report)

    def test_disabled_dry_run_scheduler_job_does_not_block(self) -> None:
        report = build_with_all_snapshots(
            scheduler_snapshot={
                "jobs": [{"strategy_id": "S01", "disabled": True, "dry_run_only": True}]
            }
        )
        assert_ready(self, report)

    def test_lifecycle_automation_blocks_readiness(self) -> None:
        cases = [
            {"lifecycle_automation_enabled": True},
            {"lifecycle_transition_execution_enabled": True},
        ]
        for snapshot in cases:
            with self.subTest(snapshot=snapshot):
                report = build_with_all_snapshots(lifecycle_snapshot=snapshot)
                assert_not_ready(self, report)

    def test_risk_snapshot_blockers(self) -> None:
        cases = [
            clean_risk_snapshot(risk_bypass_enabled=True),
            clean_risk_snapshot(kill_switch_available=False),
            clean_risk_snapshot(hard_halt_available=False),
            clean_risk_snapshot(daily_loss_limit_available=False),
        ]
        for snapshot in cases:
            with self.subTest(snapshot=snapshot):
                report = build_with_all_snapshots(risk_snapshot=snapshot)
                assert_not_ready(self, report)

    def test_paper_broker_snapshot_blockers(self) -> None:
        cases = [
            clean_paper_broker_snapshot(mode="LIVE"),
            clean_paper_broker_snapshot(ibkr_port=4002),
            clean_paper_broker_snapshot(paper_trading=False),
            clean_paper_broker_snapshot(live_trading_enabled=True),
            clean_paper_broker_snapshot(broker_submission_enabled=True),
        ]
        for snapshot in cases:
            with self.subTest(snapshot=snapshot):
                report = build_with_all_snapshots(paper_broker_snapshot=snapshot)
                assert_not_ready(self, report)

    def test_valid_clean_report_with_matching_snapshots_is_ready(self) -> None:
        report = build_with_all_snapshots()
        assert_ready(self, report)
        self.assertEqual([], report["errors"])

    def test_recommendations_do_not_include_disallowed_actions(self) -> None:
        report = build_with_all_snapshots()
        text = json.dumps(report["recommendations"]).lower()
        self.assertNotIn("enable live trading", text.replace("do not enable live trading", ""))
        self.assertNotIn("enable all strategies", text.replace("do not enable all strategies", ""))
        self.assertNotIn("place orders now", text.replace("do not place orders now", ""))

    def test_report_includes_required_top_level_fields_and_is_json_safe(self) -> None:
        report = build_with_all_snapshots()
        for key in (
            "dry_run",
            "stage4h6_one_strategy_activation_acceptance_report",
            "generated_at",
            "artifact_checks",
            "activation_payload_checks",
            "snapshot_checks",
            "state_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "risk_checks",
            "paper_broker_checks",
            "safety_checks",
            "readiness_for_next_phase",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report, sort_keys=True)

    def test_input_reports_and_snapshots_are_not_mutated(self) -> None:
        h5 = valid_stage4h5_report()
        activation = matching_activation_snapshot()
        audit = matching_audit_snapshot()
        state = clean_state_snapshot()
        before = copy.deepcopy((h5, activation, audit, state))
        build(h5=h5, activation_snapshot=activation, audit_snapshot=audit, state_snapshot=state)
        self.assertEqual(before, (h5, activation, audit, state))

    def test_cli_requires_dry_run_only_before_parsing(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_stage4h6_one_strategy_activation_acceptance(
                ["--json", "--stage4h5-executor-json", "{"]
            )
        self.assertEqual(1, code)
        self.assertIn("--dry-run-only", stderr.getvalue())

    def test_cli_json_stdout_is_strict_json(self) -> None:
        stdout = io.StringIO()
        args = [
            "--dry-run-only",
            "--json",
            "--stage4h5-executor-json",
            json.dumps(valid_stage4h5_report()),
        ]
        with redirect_stdout(stdout):
            code = tool.run_stage4h6_one_strategy_activation_acceptance(args)
        self.assertEqual(0, code)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4h6_one_strategy_activation_acceptance_report"])

    def test_cli_exposes_no_execution_or_broad_enablement_actions(self) -> None:
        parser_source = Path(tool.__file__).read_text()
        disallowed_options = [
            "--submit",
            "--cancel",
            "--status",
            "--market-data",
            "--qualification",
            "--qualify",
            "--scheduler-enable",
            "--lifecycle-enable",
        ]
        for option in disallowed_options:
            self.assertNotIn(option, parser_source)

    def test_stage4h6_source_has_no_forbidden_runtime_call_tokens(self) -> None:
        forbidden = [
            "submit_order_plan(",
            "get_order_status(",
            "cancel_order(",
            ".placeOrder(",
            ".cancelOrder(",
            ".reqMktData(",
            ".qualifyContracts(",
            ".add_job(",
            "add_job(",
            ".run_scan(",
            "run_scan(",
            ".scan_now(",
            "scan_now(",
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
            "random.",
            "time.time",
            "datetime.now",
            "StateStore(",
            ".save(",
            ".write_json(",
            ".append_event(",
            ".append_jsonl(",
            "ib_insync",
        ]
        for path in STAGE4H6_FILES:
            source = path.read_text()
            for token in forbidden:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
