from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.strategy_quality_decision_report import (
    BLOCKED_BY_LOCAL_DATA,
    BLOCKED_BY_READINESS,
    BLOCKED_BY_SAFETY_OR_HALT,
    INSUFFICIENT_EVIDENCE,
    NEEDS_MORE_OBSERVABILITY,
    READY_FOR_STRATEGY_TUNING,
    build_strategy_quality_decision_report,
)
from algo_trader_unified.tools import strategy_quality_decision_report as tool


NY = ZoneInfo("America/New_York")
IDS = [S01_VOL_BASELINE, S02_VOL_ENHANCED]


def realism_report(**overrides):
    report = {
        "dry_run": True,
        "strategy_realism_report": True,
        "generated_at": "2026-05-05T17:00:00-04:00",
        "session_date": "2026-05-05",
        "strategy_ids": IDS,
        "per_strategy": {
            S01_VOL_BASELINE: {
                "signals_generated": 0,
                "signals_skipped": 2,
                "skip_reasons": {"SKIP_VIX_GATE": 2},
                "top_skip_reason": "SKIP_VIX_GATE",
                "readiness_available": True,
                "readiness_passed": True,
                "dirty_state": False,
                "calendar_expired": False,
                "iv_baseline_available": True,
            },
            S02_VOL_ENHANCED: {
                "signals_generated": 0,
                "signals_skipped": 1,
                "skip_reasons": {"SKIP_VIX_GATE": 1},
                "top_skip_reason": "SKIP_VIX_GATE",
                "readiness_available": True,
                "readiness_passed": True,
                "dirty_state": False,
                "calendar_expired": False,
                "iv_baseline_available": True,
            },
        },
        "aggregate": {"skip_reasons": {"SKIP_VIX_GATE": 3}, "top_skip_reason": "SKIP_VIX_GATE"},
        "readiness": {
            "account_snapshot_fresh": True,
            "nlv_valid": True,
            "halt_active": False,
            "missing_readiness_strategy_ids": [],
        },
        "success": True,
        "errors": [],
    }
    report.update(overrides)
    return report


def readiness_report(**overrides):
    report = {
        "dry_run": True,
        "readiness_data_quality_report": True,
        "generated_at": "2026-05-05T17:05:00-04:00",
        "strategy_ids": IDS,
        "inputs": {"strategy_realism_success": True, "session_date": "2026-05-05"},
        "per_strategy": {
            S01_VOL_BASELINE: {
                "top_skip_reason": "SKIP_VIX_GATE",
                "likely_blocker_category": "strategy_filter_problem",
                "readiness_available": True,
                "readiness_passed": True,
                "dirty_state": False,
                "calendar_expired": False,
                "iv_baseline_available": True,
                "account_snapshot_fresh": True,
                "nlv_valid": True,
                "halt_active": False,
            },
            S02_VOL_ENHANCED: {
                "top_skip_reason": "SKIP_VIX_GATE",
                "likely_blocker_category": "strategy_filter_problem",
                "readiness_available": True,
                "readiness_passed": True,
                "dirty_state": False,
                "calendar_expired": False,
                "iv_baseline_available": True,
                "account_snapshot_fresh": True,
                "nlv_valid": True,
                "halt_active": False,
            },
        },
        "aggregate": {
            "dominant_blocker_category": "strategy_filter_problem",
            "dominant_skip_reason": "SKIP_VIX_GATE",
        },
        "data_quality": {
            "missing_readiness_strategy_ids": [],
            "stale_or_missing_inputs": [],
        },
        "success": True,
        "errors": [],
    }
    report.update(overrides)
    return report


def local_report(**overrides):
    report = {
        "dry_run": True,
        "local_input_audit_report": True,
        "generated_at": "2026-05-05T17:10:00-04:00",
        "strategy_ids": IDS,
        "per_strategy": {
            S01_VOL_BASELINE: {
                "readiness_available": True,
                "readiness_passed": True,
                "iv_baseline_available": True,
                "calendar_expired": False,
                "input_blockers": [],
            },
            S02_VOL_ENHANCED: {
                "readiness_available": True,
                "readiness_passed": True,
                "iv_baseline_available": True,
                "calendar_expired": False,
                "input_blockers": [],
            },
        },
        "aggregate": {
            "blocking_input_count": 0,
            "stale_input_count": 0,
            "missing_input_count": 0,
            "dominant_input_issue": None,
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    report.update(overrides)
    return report


_DEFAULT = object()


def build(realism=_DEFAULT, readiness=_DEFAULT, local=_DEFAULT):
    return build_strategy_quality_decision_report(
        strategy_realism_report=realism_report() if realism is _DEFAULT else realism,
        readiness_data_quality_report=readiness_report() if readiness is _DEFAULT else readiness,
        local_input_audit_report=local_report() if local is _DEFAULT else local,
        strategy_ids=IDS,
        now_provider=lambda: datetime(2026, 5, 5, 17, 30, tzinfo=NY),
    )


class StrategyQualityDecisionReportTests(unittest.TestCase):
    def test_report_includes_required_fields_and_is_json_safe(self) -> None:
        report = build()

        self.assertTrue(report["dry_run"])
        self.assertTrue(report["strategy_quality_decision_report"])
        self.assertEqual(report["generated_at"], "2026-05-05T17:30:00-04:00")
        self.assertEqual(report["strategy_ids"], IDS)
        self.assertTrue(report["inputs"]["strategy_realism_success"])
        self.assertTrue(report["inputs"]["readiness_data_quality_success"])
        self.assertTrue(report["inputs"]["local_input_audit_success"])
        self.assertEqual(report["inputs"]["session_date"], "2026-05-05")
        self.assertIn("per_strategy", report)
        self.assertIn("aggregate", report)
        self.assertIn("evidence", report)
        self.assertIn("recommendations", report)
        self.assertEqual(
            report["safety"],
            {
                "broker_calls_enabled": False,
                "market_data_enabled": False,
                "external_fetch_enabled": False,
                "paper_live_orders_enabled": False,
                "strategy_changes_enabled": False,
                "lifecycle_changes_enabled": False,
            },
        )
        json.dumps(report, sort_keys=True)

    def test_core_builder_accepts_plain_dicts_without_dependencies(self) -> None:
        report = build()

        self.assertEqual(
            report["aggregate"]["overall_decision"], READY_FOR_STRATEGY_TUNING
        )
        self.assertEqual(report["errors"], [])

    def test_missing_or_failed_upstream_reports_are_insufficient_evidence(self) -> None:
        for kwargs in (
            {"realism": None},
            {"readiness": {}},
            {"local": {"local_input_audit_report": True, "success": False}},
        ):
            report = build(**kwargs)
            self.assertEqual(report["aggregate"]["overall_decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(report["success"])
            self.assertTrue(report["errors"])
            for item in report["per_strategy"].values():
                self.assertEqual(item["decision"], INSUFFICIENT_EVIDENCE)

    def test_failed_upstream_success_false_is_graceful(self) -> None:
        report = build(realism=realism_report(success=False))

        self.assertEqual(report["aggregate"]["overall_decision"], INSUFFICIENT_EVIDENCE)
        self.assertFalse(report["success"])
        self.assertTrue(any("unsuccessful" in error for error in report["errors"]))

    def test_safety_halt_evidence_produces_safety_decision(self) -> None:
        readiness = readiness_report(
            per_strategy={
                S01_VOL_BASELINE: {
                    "top_skip_reason": "SKIP_NEEDS_RECONCILIATION",
                    "likely_blocker_category": "halt_or_safety_problem",
                    "halt_active": True,
                },
                S02_VOL_ENHANCED: readiness_report()["per_strategy"][S02_VOL_ENHANCED],
            }
        )
        report = build(readiness=readiness)

        self.assertEqual(
            report["per_strategy"][S01_VOL_BASELINE]["decision"],
            BLOCKED_BY_SAFETY_OR_HALT,
        )
        self.assertEqual(
            report["aggregate"]["overall_decision"], BLOCKED_BY_SAFETY_OR_HALT
        )

    def test_safety_can_dominate_aggregate_when_inputs_are_failed(self) -> None:
        report = build(
            realism=realism_report(success=False, readiness={"halt_active": True}),
            readiness=readiness_report(success=False),
        )

        self.assertEqual(
            report["aggregate"]["overall_decision"], BLOCKED_BY_SAFETY_OR_HALT
        )
        for item in report["per_strategy"].values():
            self.assertEqual(item["decision"], INSUFFICIENT_EVIDENCE)

    def test_readiness_evidence_produces_readiness_decision(self) -> None:
        realism = realism_report(
            per_strategy={
                S01_VOL_BASELINE: {
                    "signals_skipped": 1,
                    "skip_reasons": {"SKIP_READINESS_FAILED": 1},
                    "readiness_available": True,
                    "readiness_passed": False,
                    "dirty_state": False,
                    "calendar_expired": False,
                },
                S02_VOL_ENHANCED: realism_report()["per_strategy"][S02_VOL_ENHANCED],
            }
        )
        report = build(realism=realism)

        self.assertEqual(
            report["per_strategy"][S01_VOL_BASELINE]["decision"], BLOCKED_BY_READINESS
        )
        self.assertEqual(report["aggregate"]["overall_decision"], BLOCKED_BY_READINESS)

    def test_local_data_issue_produces_local_data_decision(self) -> None:
        local = local_report(
            per_strategy={
                S01_VOL_BASELINE: {
                    "readiness_available": True,
                    "readiness_passed": True,
                    "iv_baseline_available": False,
                    "calendar_expired": False,
                    "input_blockers": ["iv_rank"],
                },
                S02_VOL_ENHANCED: local_report()["per_strategy"][S02_VOL_ENHANCED],
            },
            aggregate={
                "blocking_input_count": 0,
                "stale_input_count": 0,
                "missing_input_count": 1,
                "dominant_input_issue": "iv_rank",
            },
        )
        report = build(local=local)

        self.assertEqual(
            report["per_strategy"][S01_VOL_BASELINE]["decision"],
            BLOCKED_BY_LOCAL_DATA,
        )
        self.assertEqual(report["aggregate"]["overall_decision"], BLOCKED_BY_LOCAL_DATA)

    def test_unknown_or_missing_skip_reasons_need_more_observability(self) -> None:
        realism = realism_report(
            per_strategy={
                S01_VOL_BASELINE: {
                    "signals_generated": 0,
                    "signals_skipped": 1,
                    "skip_reasons": {"UNKNOWN_SKIP_REASON": 1},
                    "readiness_available": True,
                    "readiness_passed": True,
                    "dirty_state": False,
                    "calendar_expired": False,
                },
                S02_VOL_ENHANCED: {
                    "signals_generated": 0,
                    "signals_skipped": 0,
                    "skip_reasons": {},
                    "readiness_available": True,
                    "readiness_passed": True,
                    "dirty_state": False,
                    "calendar_expired": False,
                },
            },
            aggregate={"skip_reasons": {"UNKNOWN_SKIP_REASON": 1}},
        )
        readiness = readiness_report(
            per_strategy={
                S01_VOL_BASELINE: {
                    "top_skip_reason": "UNKNOWN_SKIP_REASON",
                    "likely_blocker_category": "unknown",
                    "readiness_available": True,
                    "readiness_passed": True,
                    "dirty_state": False,
                    "calendar_expired": False,
                    "halt_active": False,
                },
                S02_VOL_ENHANCED: {
                    "top_skip_reason": None,
                    "likely_blocker_category": "unknown",
                    "readiness_available": True,
                    "readiness_passed": True,
                    "dirty_state": False,
                    "calendar_expired": False,
                    "halt_active": False,
                },
            }
        )
        report = build(realism=realism, readiness=readiness)

        self.assertEqual(
            report["per_strategy"][S01_VOL_BASELINE]["decision"],
            NEEDS_MORE_OBSERVABILITY,
        )
        self.assertEqual(
            report["per_strategy"][S02_VOL_ENHANCED]["decision"],
            NEEDS_MORE_OBSERVABILITY,
        )
        self.assertEqual(
            report["aggregate"]["overall_decision"], NEEDS_MORE_OBSERVABILITY
        )

    def test_clean_inputs_and_strategy_filter_problem_are_ready_for_tuning(self) -> None:
        report = build()

        self.assertEqual(
            report["per_strategy"][S01_VOL_BASELINE]["decision"],
            READY_FOR_STRATEGY_TUNING,
        )
        self.assertEqual(
            report["aggregate"]["overall_decision"], READY_FOR_STRATEGY_TUNING
        )

    def test_insufficient_data_produces_insufficient_evidence(self) -> None:
        readiness = readiness_report(
            per_strategy={
                S01_VOL_BASELINE: {
                    "top_skip_reason": "SKIP_MISC_UNCLASSIFIED",
                    "likely_blocker_category": "unclassified",
                    "readiness_available": True,
                    "readiness_passed": True,
                    "dirty_state": False,
                    "calendar_expired": False,
                    "halt_active": False,
                },
                S02_VOL_ENHANCED: readiness_report()["per_strategy"][S02_VOL_ENHANCED],
            }
        )
        realism = realism_report(
            per_strategy={
                S01_VOL_BASELINE: {
                    "signals_generated": 0,
                    "signals_skipped": 1,
                    "skip_reasons": {"SKIP_MISC_UNCLASSIFIED": 1},
                    "readiness_available": True,
                    "readiness_passed": True,
                    "dirty_state": False,
                    "calendar_expired": False,
                },
                S02_VOL_ENHANCED: realism_report()["per_strategy"][S02_VOL_ENHANCED],
            }
        )
        report = build(realism=realism, readiness=readiness)

        self.assertEqual(
            report["per_strategy"][S01_VOL_BASELINE]["decision"],
            INSUFFICIENT_EVIDENCE,
        )

    def test_aggregate_priority_rules_are_conservative(self) -> None:
        readiness = readiness_report(
            per_strategy={
                S01_VOL_BASELINE: {
                    "top_skip_reason": "SKIP_READINESS_FAILED",
                    "likely_blocker_category": "readiness_problem",
                    "readiness_available": True,
                    "readiness_passed": False,
                    "dirty_state": False,
                    "calendar_expired": False,
                    "halt_active": False,
                },
                S02_VOL_ENHANCED: readiness_report()["per_strategy"][S02_VOL_ENHANCED],
            }
        )
        local = local_report(
            per_strategy={
                S01_VOL_BASELINE: local_report()["per_strategy"][S01_VOL_BASELINE],
                S02_VOL_ENHANCED: {
                    "readiness_available": True,
                    "readiness_passed": True,
                    "iv_baseline_available": False,
                    "calendar_expired": False,
                    "input_blockers": ["iv_rank"],
                },
            }
        )
        report = build(readiness=readiness, local=local)

        self.assertEqual(report["aggregate"]["overall_decision"], BLOCKED_BY_READINESS)

    def test_dominant_blocker_and_skip_reason_ties_are_alphabetical(self) -> None:
        realism = realism_report(
            per_strategy={
                S01_VOL_BASELINE: {
                    "signals_skipped": 2,
                    "skip_reasons": {"SKIP_Z_READINESS_FAILED": 1, "SKIP_A_READINESS_FAILED": 1},
                    "readiness_available": True,
                    "readiness_passed": False,
                    "dirty_state": False,
                    "calendar_expired": False,
                },
                S02_VOL_ENHANCED: {
                    "signals_skipped": 1,
                    "skip_reasons": {"SKIP_IV_BASELINE_MISSING": 1},
                    "readiness_available": True,
                    "readiness_passed": True,
                    "dirty_state": False,
                    "calendar_expired": False,
                },
            },
            aggregate={
                "skip_reasons": {
                    "SKIP_Z_READINESS_FAILED": 1,
                    "SKIP_A_READINESS_FAILED": 1,
                    "SKIP_IV_BASELINE_MISSING": 1,
                }
            },
        )
        readiness = readiness_report(
            per_strategy={
                S01_VOL_BASELINE: {
                    "top_skip_reason": "SKIP_A_READINESS_FAILED",
                    "likely_blocker_category": "readiness_problem",
                    "readiness_available": True,
                    "readiness_passed": False,
                    "dirty_state": False,
                    "calendar_expired": False,
                    "halt_active": False,
                },
                S02_VOL_ENHANCED: {
                    "top_skip_reason": "SKIP_IV_BASELINE_MISSING",
                    "likely_blocker_category": "data_problem",
                    "readiness_available": True,
                    "readiness_passed": True,
                    "dirty_state": False,
                    "calendar_expired": False,
                    "halt_active": False,
                },
            }
        )
        local = local_report(
            per_strategy={
                S01_VOL_BASELINE: local_report()["per_strategy"][S01_VOL_BASELINE],
                S02_VOL_ENHANCED: {
                    "readiness_available": True,
                    "readiness_passed": True,
                    "iv_baseline_available": False,
                    "calendar_expired": False,
                    "input_blockers": ["iv_rank"],
                },
            }
        )
        report = build(realism=realism, readiness=readiness, local=local)

        self.assertEqual(report["aggregate"]["dominant_blocker_category"], "data_problem")
        self.assertEqual(report["aggregate"]["dominant_skip_reason"], "SKIP_A_READINESS_FAILED")

    def test_recommendations_are_deterministic_and_non_binding(self) -> None:
        first = build()["recommendations"]
        second = build()["recommendations"]

        self.assertEqual(first, second)
        text = json.dumps(first, sort_keys=True).lower()
        self.assertNotIn("set threshold", text)
        self.assertNotIn("position size", text)
        self.assertNotIn("place order", text)
        self.assertNotIn("broker", " ".join(first["ordered_next_steps"]).lower())
        self.assertIn("Do not change position sizing from this report.", first["do_not_do_yet"])

    def test_modules_do_not_import_broker_market_data_systemd_or_scheduler(self) -> None:
        core_source = Path("algo_trader_unified/core/strategy_quality_decision_report.py").read_text(
            encoding="utf-8"
        )
        tool_source = Path("algo_trader_unified/tools/strategy_quality_decision_report.py").read_text(
            encoding="utf-8"
        )
        combined = core_source + "\n" + tool_source
        for forbidden in (
            "ib_insync",
            "reqMktData",
            "placeOrder",
            "yfinance",
            "requests",
            "urllib",
            "systemd",
            "UnifiedScheduler",
            "scheduler_cadence",
            "market_open_scan",
        ):
            self.assertNotIn(forbidden, combined)


class StrategyQualityDecisionCliTests(unittest.TestCase):
    def test_missing_dry_run_only_exits_before_any_loader_or_factory(self) -> None:
        calls = []

        def factory(name):
            def _inner(*args, **kwargs):
                calls.append(name)
                return mock.Mock()

            return _inner

        err = io.StringIO()
        with redirect_stderr(err):
            code = tool.run_strategy_quality_decision_report(
                [],
                state_store_factory=factory("state"),
                ledger_reader_factory=factory("ledger"),
                readiness_provider_factory=factory("readiness"),
                strategy_realism_report_builder=factory("realism"),
                readiness_data_quality_report_builder=factory("readiness_report"),
                local_input_audit_report_builder=factory("local"),
                report_builder=factory("report"),
            )
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("--dry-run-only", err.getvalue())

    def test_json_outputs_strict_json_stdout_only(self) -> None:
        report = {"success": True, "dry_run": True, "strategy_quality_decision_report": True}
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = tool.run_strategy_quality_decision_report(
                ["--dry-run-only", "--json"],
                state_store_factory=lambda path: mock.Mock(),
                ledger_reader_factory=lambda root: mock.Mock(),
                readiness_provider_factory=lambda **kwargs: lambda: None,
                strategy_realism_report_builder=lambda **kwargs: realism_report(),
                readiness_data_quality_report_builder=lambda **kwargs: readiness_report(),
                local_input_audit_report_builder=lambda **kwargs: local_report(),
                report_builder=lambda **kwargs: report,
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue()), report)
        self.assertEqual(err.getvalue(), "")

    def test_human_output_is_default_and_report_path_does_not_start_jobs(self) -> None:
        calls = []
        report = build()

        def state_factory(path):
            calls.append("state")
            return mock.Mock()

        def ledger_factory(root):
            calls.append("ledger")
            return mock.Mock()

        def readiness_factory(**kwargs):
            calls.append("readiness")
            return lambda: None

        def realism_builder(**kwargs):
            calls.append("realism")
            return realism_report()

        def readiness_builder(**kwargs):
            calls.append("readiness_report")
            return readiness_report()

        def local_builder(**kwargs):
            calls.append("local")
            return local_report()

        def report_builder(**kwargs):
            calls.append("report")
            return report

        out = io.StringIO()
        with redirect_stdout(out):
            code = tool.run_strategy_quality_decision_report(
                ["--dry-run-only"],
                state_store_factory=state_factory,
                ledger_reader_factory=ledger_factory,
                readiness_provider_factory=readiness_factory,
                strategy_realism_report_builder=realism_builder,
                readiness_data_quality_report_builder=readiness_builder,
                local_input_audit_report_builder=local_builder,
                report_builder=report_builder,
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            calls, ["state", "ledger", "readiness", "realism", "readiness_report", "local", "report"]
        )
        self.assertIn("Strategy quality decision report", out.getvalue())
        self.assertIn("overall_decision", out.getvalue())


if __name__ == "__main__":
    unittest.main()
