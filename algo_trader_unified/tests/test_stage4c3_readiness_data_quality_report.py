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
from algo_trader_unified.core.readiness_data_quality_report import (
    build_readiness_data_quality_report,
)
from algo_trader_unified.tools import readiness_data_quality_report as tool


NY = ZoneInfo("America/New_York")


def realism_report(**overrides):
    report = {
        "dry_run": True,
        "strategy_realism_report": True,
        "generated_at": "2026-05-05T17:00:00-04:00",
        "session_date": "2026-05-05",
        "strategy_ids": [S01_VOL_BASELINE, S02_VOL_ENHANCED],
        "per_strategy": {
            S01_VOL_BASELINE: {
                "skip_reasons": {},
                "top_skip_reason": None,
                "readiness_available": True,
                "readiness_passed": True,
                "dirty_state": False,
                "calendar_expired": False,
                "iv_baseline_available": True,
            },
            S02_VOL_ENHANCED: {
                "skip_reasons": {},
                "top_skip_reason": None,
                "readiness_available": True,
                "readiness_passed": True,
                "dirty_state": False,
                "calendar_expired": False,
                "iv_baseline_available": True,
            },
        },
        "aggregate": {"skip_reasons": {}, "top_skip_reason": None},
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


def build(report):
    return build_readiness_data_quality_report(
        strategy_realism_report=report,
        now_provider=lambda: datetime(2026, 5, 5, 17, 30, tzinfo=NY),
    )


class ReadinessDataQualityReportTests(unittest.TestCase):
    def test_report_includes_required_fields_and_is_json_safe(self) -> None:
        report = build(realism_report())

        self.assertTrue(report["dry_run"])
        self.assertTrue(report["readiness_data_quality_report"])
        self.assertEqual(report["generated_at"], "2026-05-05T17:30:00-04:00")
        self.assertEqual(report["strategy_ids"], [S01_VOL_BASELINE, S02_VOL_ENHANCED])
        self.assertTrue(report["inputs"]["used_strategy_realism_report"])
        self.assertTrue(report["inputs"]["strategy_realism_success"])
        self.assertEqual(report["inputs"]["session_date"], "2026-05-05")
        self.assertIn("aggregate", report)
        self.assertIn("data_quality", report)
        self.assertEqual(
            report["safety"],
            {
                "broker_calls_enabled": False,
                "market_data_enabled": False,
                "paper_live_orders_enabled": False,
                "strategy_changes_enabled": False,
                "lifecycle_changes_enabled": False,
            },
        )
        json.dumps(report, sort_keys=True)

    def test_readiness_problem_evidence_produces_readiness_diagnosis(self) -> None:
        report = realism_report(
            per_strategy={
                S01_VOL_BASELINE: {
                    "skip_reasons": {"SKIP_READINESS_NOT_EVALUATED": 1},
                    "readiness_available": False,
                },
                S02_VOL_ENHANCED: {"skip_reasons": {}, "readiness_available": True},
            },
            aggregate={"skip_reasons": {"SKIP_READINESS_NOT_EVALUATED": 1}},
        )
        built = build(report)
        item = built["per_strategy"][S01_VOL_BASELINE]

        self.assertEqual(item["likely_blocker_category"], "readiness_problem")
        self.assertIn("readiness cadence", item["diagnosis"])
        self.assertIn("readiness", built["data_quality"]["stale_or_missing_inputs"])

    def test_data_problem_evidence_produces_local_data_diagnosis(self) -> None:
        report = realism_report(
            per_strategy={
                S01_VOL_BASELINE: {
                    "skip_reasons": {"SKIP_IV_BASELINE_MISSING": 2},
                    "readiness_available": True,
                    "iv_baseline_available": False,
                    "calendar_expired": False,
                },
                S02_VOL_ENHANCED: {
                    "skip_reasons": {},
                    "readiness_available": True,
                    "iv_baseline_available": True,
                    "calendar_expired": False,
                },
            },
            aggregate={"skip_reasons": {"SKIP_IV_BASELINE_MISSING": 2}},
        )
        built = build(report)

        self.assertEqual(
            built["per_strategy"][S01_VOL_BASELINE]["likely_blocker_category"],
            "data_problem",
        )
        self.assertIn("local inputs", built["per_strategy"][S01_VOL_BASELINE]["diagnosis"])
        self.assertTrue(
            any("IV rank source freshness" in step for step in built["recommendations"]["ordered_next_steps"])
        )

    def test_strategy_filter_diagnosis_does_not_recommend_threshold_changes(self) -> None:
        report = realism_report(
            per_strategy={
                S01_VOL_BASELINE: {
                    "skip_reasons": {"SKIP_VIX_GATE": 1},
                    "readiness_available": True,
                },
                S02_VOL_ENHANCED: {"skip_reasons": {}, "readiness_available": True},
            },
            aggregate={"skip_reasons": {"SKIP_VIX_GATE": 1}},
        )
        built = build(report)
        text = json.dumps(built, sort_keys=True)

        self.assertEqual(
            built["per_strategy"][S01_VOL_BASELINE]["likely_blocker_category"],
            "strategy_filter_problem",
        )
        self.assertIn("before considering threshold tuning", text)
        self.assertNotIn("set threshold", text.lower())
        self.assertNotIn("new sizing", text.lower())

    def test_halt_and_unknown_diagnoses_are_specific(self) -> None:
        report = realism_report(
            per_strategy={
                S01_VOL_BASELINE: {"skip_reasons": {"SKIP_NEEDS_RECONCILIATION": 1}},
                S02_VOL_ENHANCED: {"skip_reasons": {"UNKNOWN_SKIP_REASON": 1}},
            },
            aggregate={
                "skip_reasons": {
                    "SKIP_NEEDS_RECONCILIATION": 1,
                    "UNKNOWN_SKIP_REASON": 1,
                }
            },
        )
        built = build(report)

        self.assertEqual(
            built["per_strategy"][S01_VOL_BASELINE]["likely_blocker_category"],
            "halt_or_safety_problem",
        )
        self.assertIn("halt and reconciliation", built["per_strategy"][S01_VOL_BASELINE]["diagnosis"])
        self.assertEqual(
            built["per_strategy"][S02_VOL_ENHANCED]["likely_blocker_category"],
            "unknown",
        )
        self.assertIn("reason-code logging", built["per_strategy"][S02_VOL_ENHANCED]["diagnosis"])

    def test_dominant_category_and_skip_reason_ties_are_alphabetical(self) -> None:
        report = realism_report(
            per_strategy={
                S01_VOL_BASELINE: {
                    "skip_reasons": {
                        "SKIP_Z_READINESS_NOT_EVALUATED": 1,
                        "SKIP_A_READINESS_NOT_EVALUATED": 1,
                    },
                    "readiness_available": False,
                },
                S02_VOL_ENHANCED: {
                    "skip_reasons": {"SKIP_IV_BASELINE_MISSING": 1},
                    "iv_baseline_available": False,
                },
            },
            aggregate={
                "skip_reasons": {
                    "SKIP_Z_READINESS_NOT_EVALUATED": 1,
                    "SKIP_A_READINESS_NOT_EVALUATED": 1,
                    "SKIP_IV_BASELINE_MISSING": 1,
                }
            },
        )
        built = build(report)

        self.assertEqual(
            built["per_strategy"][S01_VOL_BASELINE]["top_skip_reason"],
            "SKIP_A_READINESS_NOT_EVALUATED",
        )
        self.assertEqual(built["aggregate"]["dominant_blocker_category"], "data_problem")
        self.assertEqual(built["aggregate"]["dominant_skip_reason"], "SKIP_A_READINESS_NOT_EVALUATED")

    def test_ordered_next_steps_are_deterministic(self) -> None:
        first = build(
            realism_report(
                per_strategy={
                    S01_VOL_BASELINE: {"skip_reasons": {"SKIP_READINESS_FAILED": 1}},
                    S02_VOL_ENHANCED: {"skip_reasons": {"SKIP_IV_BASELINE_MISSING": 1}},
                },
                aggregate={
                    "skip_reasons": {
                        "SKIP_IV_BASELINE_MISSING": 1,
                        "SKIP_READINESS_FAILED": 1,
                    }
                },
            )
        )["recommendations"]["ordered_next_steps"]
        second = build(
            realism_report(
                per_strategy={
                    S02_VOL_ENHANCED: {"skip_reasons": {"SKIP_IV_BASELINE_MISSING": 1}},
                    S01_VOL_BASELINE: {"skip_reasons": {"SKIP_READINESS_FAILED": 1}},
                },
                aggregate={
                    "skip_reasons": {
                        "SKIP_READINESS_FAILED": 1,
                        "SKIP_IV_BASELINE_MISSING": 1,
                    }
                },
            )
        )["recommendations"]["ordered_next_steps"]
        self.assertEqual(first, second)

    def test_missing_readiness_and_unsuccessful_input_are_graceful(self) -> None:
        report = realism_report(
            success=False,
            errors=["ledger paths unavailable from ledger_reader"],
            per_strategy={
                S01_VOL_BASELINE: {"readiness_available": False, "skip_reasons": {}},
                S02_VOL_ENHANCED: {"readiness_available": False, "skip_reasons": {}},
            },
            readiness={
                "account_snapshot_fresh": None,
                "nlv_valid": None,
                "halt_active": False,
                "missing_readiness_strategy_ids": [S02_VOL_ENHANCED, S01_VOL_BASELINE],
            },
        )
        built = build(report)

        self.assertFalse(built["success"])
        self.assertEqual(
            built["data_quality"]["missing_readiness_strategy_ids"],
            [S01_VOL_BASELINE, S02_VOL_ENHANCED],
        )
        self.assertTrue(any("unsuccessful" in message for message in built["errors"]))
        self.assertIn("account_snapshot", built["data_quality"]["stale_or_missing_inputs"])

    def test_modules_do_not_import_broker_market_data_systemd_or_scheduler(self) -> None:
        core_source = Path("algo_trader_unified/core/readiness_data_quality_report.py").read_text(
            encoding="utf-8"
        )
        tool_source = Path("algo_trader_unified/tools/readiness_data_quality_report.py").read_text(
            encoding="utf-8"
        )
        combined = core_source + "\n" + tool_source
        for forbidden in (
            "ib_insync",
            "reqMktData",
            "placeOrder",
            "IBKR",
            "systemd",
            "UnifiedScheduler",
            "scheduler_cadence",
            "market_open_scan",
        ):
            self.assertNotIn(forbidden, combined)


class ReadinessDataQualityCliTests(unittest.TestCase):
    def test_missing_dry_run_only_exits_before_any_loader_or_factory(self) -> None:
        calls = []

        def factory(name):
            def _inner(*args, **kwargs):
                calls.append(name)
                return mock.Mock()

            return _inner

        err = io.StringIO()
        with redirect_stderr(err):
            code = tool.run_readiness_data_quality_report(
                [],
                state_store_factory=factory("state"),
                ledger_reader_factory=factory("ledger"),
                readiness_provider_factory=factory("readiness"),
                strategy_realism_report_builder=factory("realism"),
                report_builder=factory("report"),
            )
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("--dry-run-only", err.getvalue())

    def test_json_outputs_strict_json_stdout_only(self) -> None:
        report = {"success": True, "dry_run": True, "readiness_data_quality_report": True}
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = tool.run_readiness_data_quality_report(
                ["--dry-run-only", "--json"],
                state_store_factory=lambda path: mock.Mock(),
                ledger_reader_factory=lambda root: mock.Mock(),
                readiness_provider_factory=lambda **kwargs: lambda: None,
                strategy_realism_report_builder=lambda **kwargs: {"success": True},
                report_builder=lambda **kwargs: report,
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue()), report)
        self.assertEqual(err.getvalue(), "")

    def test_human_output_is_default_and_report_path_does_not_start_jobs(self) -> None:
        calls = []
        report = {
            "success": True,
            "dry_run": True,
            "readiness_data_quality_report": True,
            "generated_at": "2026-05-05T17:00:00-04:00",
            "aggregate": {
                "dominant_blocker_category": "readiness_problem",
                "dominant_skip_reason": "SKIP_READINESS_FAILED",
            },
            "per_strategy": {},
            "recommendations": {"ordered_next_steps": ["Fix readiness provider cadence before tuning strategy filters."]},
        }

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
            return {"success": True}

        def report_builder(**kwargs):
            calls.append("report")
            return report

        out = io.StringIO()
        with redirect_stdout(out):
            code = tool.run_readiness_data_quality_report(
                ["--dry-run-only"],
                state_store_factory=state_factory,
                ledger_reader_factory=ledger_factory,
                readiness_provider_factory=readiness_factory,
                strategy_realism_report_builder=realism_builder,
                report_builder=report_builder,
            )
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["state", "ledger", "readiness", "realism", "report"])
        self.assertIn("Readiness data-quality report", out.getvalue())
        self.assertIn("dominant_blocker_category", out.getvalue())


if __name__ == "__main__":
    unittest.main()
