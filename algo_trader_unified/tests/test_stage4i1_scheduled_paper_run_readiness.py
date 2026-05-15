from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4i1_scheduled_paper_run_readiness import (
    MARKET_WINDOW_MANUAL_WARNING,
    build_stage4i1_scheduled_paper_run_readiness_report,
)
from algo_trader_unified.tools import stage4i1_scheduled_paper_run_readiness as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4I1_FILES = [
    ROOT / "core/stage4i1_scheduled_paper_run_readiness.py",
    ROOT / "tools/stage4i1_scheduled_paper_run_readiness.py",
]


def valid_stage4h6_report(**overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4h6_one_strategy_activation_acceptance_report": True,
        "generated_at": "2026-05-14T14:00:00+00:00",
        "artifact_checks": {
            "stage4h5_report_present": True,
            "stage4h5_report_ready": True,
        },
        "activation_payload_checks": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
            "enabled_strategy_count": 1,
            "live_trading_disabled": True,
            "all_strategies_disabled": True,
            "broker_submission_disabled": True,
        },
        "selected_strategy": {"selected_strategy_id": "S01", "paper_only": True},
        "readiness_for_next_phase": {
            "ready_to_build_first_scheduled_paper_automation_run": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


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
        "is_trading_day": True,
        "market_open": True,
        "market_close": "16:00",
        "current_time": "2026-05-14T12:00:00-04:00",
        "timezone": "America/New_York",
        "next_run_window": "2026-05-14T13:00:00-04:00",
        "allowed_to_schedule_paper_run": True,
        "reason": "operator supplied clean paper planning window",
    }
    _deep_update(snapshot, overrides)
    return snapshot


def build(
    *,
    h6: dict | None = None,
    activation_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    market_window_snapshot: dict | None = None,
) -> dict:
    return build_stage4i1_scheduled_paper_run_readiness_report(
        stage4h6_activation_acceptance_report=valid_stage4h6_report() if h6 is None else h6,
        activation_snapshot=activation_snapshot,
        state_snapshot=state_snapshot,
        risk_snapshot=risk_snapshot,
        scheduler_snapshot=scheduler_snapshot,
        lifecycle_snapshot=lifecycle_snapshot,
        paper_broker_snapshot=paper_broker_snapshot,
        market_window_snapshot=market_window_snapshot,
        now_provider=lambda: datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc),
    )


def build_with_all_snapshots(**kwargs: object) -> dict:
    defaults: dict[str, object] = {
        "activation_snapshot": matching_activation_snapshot(),
        "state_snapshot": clean_state_snapshot(),
        "risk_snapshot": clean_risk_snapshot(),
        "scheduler_snapshot": {},
        "lifecycle_snapshot": {},
        "paper_broker_snapshot": clean_paper_broker_snapshot(),
        "market_window_snapshot": clean_market_window_snapshot(),
    }
    defaults.update(kwargs)
    return build(**defaults)


def assert_ready(test_case: unittest.TestCase, report: dict) -> None:
    test_case.assertTrue(
        report["readiness_for_stage4i2"]["ready_to_build_first_scheduled_run_plan"]
    )


def assert_not_ready(test_case: unittest.TestCase, report: dict) -> None:
    test_case.assertFalse(
        report["readiness_for_stage4i2"]["ready_to_build_first_scheduled_run_plan"]
    )
    test_case.assertTrue(report["readiness_for_stage4i2"]["blockers"])


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class Stage4I1ScheduledPaperRunReadinessTests(unittest.TestCase):
    def test_missing_stage4h6_report_blocks_readiness(self) -> None:
        report = build_stage4i1_scheduled_paper_run_readiness_report(
            stage4h6_activation_acceptance_report=None
        )
        assert_not_ready(self, report)

    def test_stage4h6_report_not_ready_blocks_readiness(self) -> None:
        report = build(
            h6=valid_stage4h6_report(
                readiness_for_next_phase={
                    "ready_to_build_first_scheduled_paper_automation_run": False
                }
            )
        )
        assert_not_ready(self, report)

    def test_missing_selected_strategy_blocks_readiness(self) -> None:
        h6 = valid_stage4h6_report()
        h6["activation_payload_checks"].pop("selected_strategy_id")  # type: ignore[index,union-attr]
        h6["selected_strategy"].pop("selected_strategy_id")  # type: ignore[index,union-attr]
        report = build(h6=h6)
        assert_not_ready(self, report)

    def test_malformed_stage4h6_without_activation_payload_checks_does_not_raise(self) -> None:
        report = build(h6={"stage4h6_one_strategy_activation_acceptance_report": True})
        assert_not_ready(self, report)

    def test_selected_strategy_extraction_prefers_activation_payload_checks(self) -> None:
        report = build(
            h6=valid_stage4h6_report(
                activation_payload_checks={"selected_strategy_id": "S99"},
                selected_strategy={"selected_strategy_id": "S01", "paper_only": True},
            )
        )
        self.assertEqual("S99", report["selected_strategy"]["selected_strategy_id"])

    def test_more_than_one_activated_strategy_blocks_readiness(self) -> None:
        report = build(
            h6=valid_stage4h6_report(
                activation_payload_checks={"one_strategy_only": False, "enabled_strategy_count": 2}
            )
        )
        assert_not_ready(self, report)

    def test_activation_snapshot_matching_selected_strategy_passes(self) -> None:
        report = build_with_all_snapshots()
        assert_ready(self, report)
        self.assertTrue(report["activation_checks"]["activation_snapshot_matches"])

    def test_activation_snapshot_mismatched_selected_strategy_blocks_readiness(self) -> None:
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

    def test_activation_snapshot_malformed_arrays_do_not_crash(self) -> None:
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

    def test_missing_activation_snapshot_warns_but_does_not_crash(self) -> None:
        report = build_with_all_snapshots(activation_snapshot=None)
        assert_ready(self, report)
        self.assertTrue(report["activation_checks"]["activation_snapshot_present"] is False)
        self.assertTrue(report["warnings"])

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

    def test_scheduler_broad_automation_enabled_blocks_readiness(self) -> None:
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

    def test_lifecycle_automation_enabled_blocks_readiness(self) -> None:
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
            clean_risk_snapshot(max_position_limit_available=False),
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

    def test_market_window_explicit_false_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            market_window_snapshot=clean_market_window_snapshot(
                allowed_to_schedule_paper_run=False
            )
        )
        assert_not_ready(self, report)

    def test_market_closed_warns_but_does_not_block_by_itself(self) -> None:
        report = build_with_all_snapshots(
            market_window_snapshot=clean_market_window_snapshot(market_open=False)
        )
        assert_ready(self, report)
        self.assertTrue(any("market is currently closed" in item for item in report["warnings"]))

    def test_missing_market_window_snapshot_emits_manual_warning(self) -> None:
        report = build_with_all_snapshots(market_window_snapshot=None)
        assert_ready(self, report)
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, report["warnings"])
        text = json.dumps(report["recommendations"]).lower()
        self.assertIn("exchange hours", text)
        self.assertIn("holiday", text)

    def test_missing_optional_snapshots_warn_but_do_not_crash(self) -> None:
        report = build()
        assert_ready(self, report)
        self.assertGreaterEqual(len(report["warnings"]), 7)

    def test_valid_clean_report_with_missing_optional_snapshots_is_ready_with_warnings(self) -> None:
        report = build()
        assert_ready(self, report)
        self.assertTrue(report["warnings"])

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
            "stage4i1_scheduled_paper_run_readiness_report",
            "generated_at",
            "artifact_checks",
            "selected_strategy",
            "activation_checks",
            "state_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "paper_broker_checks",
            "market_window_checks",
            "safety_checks",
            "readiness_for_stage4i2",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report, sort_keys=True)

    def test_input_reports_and_snapshots_are_not_mutated(self) -> None:
        h6 = valid_stage4h6_report()
        activation = matching_activation_snapshot()
        state = clean_state_snapshot()
        risk = clean_risk_snapshot()
        broker = clean_paper_broker_snapshot()
        before = copy.deepcopy((h6, activation, state, risk, broker))
        build(
            h6=h6,
            activation_snapshot=activation,
            state_snapshot=state,
            risk_snapshot=risk,
            paper_broker_snapshot=broker,
        )
        self.assertEqual(before, (h6, activation, state, risk, broker))

    def test_cli_requires_dry_run_only_before_parsing(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_stage4i1_scheduled_paper_run_readiness(
                ["--json", "--stage4h6-acceptance-json", "{"]
            )
        self.assertEqual(1, code)
        self.assertIn("--dry-run-only", stderr.getvalue())

    def test_cli_json_stdout_is_strict_json(self) -> None:
        stdout = io.StringIO()
        args = [
            "--dry-run-only",
            "--json",
            "--stage4h6-acceptance-json",
            json.dumps(valid_stage4h6_report()),
        ]
        with redirect_stdout(stdout):
            code = tool.run_stage4i1_scheduled_paper_run_readiness(args)
        self.assertEqual(0, code)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4i1_scheduled_paper_run_readiness_report"])

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

    def test_stage4i1_source_has_no_forbidden_runtime_call_tokens(self) -> None:
        forbidden = [
            "submit_order_plan(",
            "get_order_status(",
            "cancel_order(",
            ".placeOrder(",
            ".cancelOrder(",
            ".reqMktData(",
            ".qualifyContracts(",
            "StateStore(",
            ".save(",
            ".write_json(",
            ".update_state(",
            ".append_event(",
            ".append_jsonl(",
            "ledger.append",
            "ledger.write",
            "scheduler.add_job(",
            ".add_job(",
            "add_job(",
            ".start(",
            "run_scan(",
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
            "ib_insync",
        ]
        for path in STAGE4I1_FILES:
            source = path.read_text()
            for token in forbidden:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
