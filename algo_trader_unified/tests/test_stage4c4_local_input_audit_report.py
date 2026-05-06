from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
from zoneinfo import ZoneInfo

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.local_input_audit_report import build_local_input_audit_report
from algo_trader_unified.tools import local_input_audit_report as tool


NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class FakeSnapshot:
    account_snapshot_fresh: bool = True
    nlv_valid: bool = True
    halt_active_by_strategy: dict[str, bool] | None = None
    dirty_state_by_strategy: dict[str, bool] | None = None
    calendar_expired_by_strategy: dict[str, bool] | None = None
    iv_baseline_available_by_strategy: dict[str, bool | None] | None = None


class FakeStateStore:
    def __init__(self) -> None:
        self.readiness = {
            S01_VOL_BASELINE: {
                "ready_for_entries": True,
                "dirty_state": False,
                "halt_active": False,
                "calendar_expired": False,
                "iv_baseline_available": True,
            },
            S02_VOL_ENHANCED: {
                "ready_for_entries": True,
                "dirty_state": False,
                "halt_active": False,
                "calendar_expired": False,
                "iv_baseline_available": True,
            },
        }
        self.saved = False

    def get_readiness(self, strategy_id):
        value = self.readiness.get(strategy_id)
        return dict(value) if isinstance(value, dict) else None

    def save(self):
        self.saved = True
        raise AssertionError("local input audit report must not save StateStore")


def snapshot(**overrides):
    ids = [S01_VOL_BASELINE, S02_VOL_ENHANCED]
    base = {
        "halt_active_by_strategy": {strategy_id: False for strategy_id in ids},
        "dirty_state_by_strategy": {strategy_id: False for strategy_id in ids},
        "calendar_expired_by_strategy": {strategy_id: False for strategy_id in ids},
        "iv_baseline_available_by_strategy": {strategy_id: True for strategy_id in ids},
    }
    base.update(overrides)
    return FakeSnapshot(**base)


def build(root: Path, **overrides):
    kwargs = {
        "strategy_ids": [S01_VOL_BASELINE, S02_VOL_ENHANCED],
        "state_store": FakeStateStore(),
        "readiness_provider": lambda: snapshot(),
        "snapshots_dir": root / "data" / "snapshots",
        "halt_state_path": root / "data" / "state" / "halt_state.json",
        "now_provider": lambda: datetime(2026, 5, 5, 17, 0, tzinfo=NY),
    }
    kwargs.update(overrides)
    return build_local_input_audit_report(**kwargs)


class LocalInputAuditReportTests(unittest.TestCase):
    def test_report_includes_required_fields_and_is_json_safe(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots_dir = root / "data" / "snapshots"
            snapshots_dir.mkdir(parents=True)
            (snapshots_dir / "account.json").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-05-05T16:59:00-04:00",
                        "nlv": 100000,
                    }
                ),
                encoding="utf-8",
            )
            report = build(
                root,
                iv_store={
                    "available_by_strategy": {
                        S01_VOL_BASELINE: True,
                        S02_VOL_ENHANCED: True,
                    },
                    "latest_timestamp_by_strategy": {
                        S01_VOL_BASELINE: "2026-05-05T16:55:00-04:00",
                        S02_VOL_ENHANCED: "2026-05-05T16:55:00-04:00",
                    },
                    "stale_by_strategy": {
                        S01_VOL_BASELINE: False,
                        S02_VOL_ENHANCED: False,
                    },
                },
                vix_snapshot_path=_write_json(
                    snapshots_dir / "vix.json",
                    {"timestamp": "2026-05-05T16:55:00-04:00", "vix": 18.2},
                ),
                market_calendar_path=_write_json(
                    snapshots_dir / "market_calendar.json",
                    {
                        "session_available": True,
                        "calendar_expired_by_strategy": {
                            S01_VOL_BASELINE: False,
                            S02_VOL_ENHANCED: False,
                        },
                    },
                ),
            )

            self.assertTrue(report["dry_run"])
            self.assertTrue(report["local_input_audit_report"])
            self.assertEqual(report["generated_at"], "2026-05-05T17:00:00-04:00")
            self.assertEqual(report["strategy_ids"], [S01_VOL_BASELINE, S02_VOL_ENHANCED])
            self.assertEqual(
                set(report["inputs_checked"]),
                {
                    "iv_rank",
                    "vix",
                    "market_calendar",
                    "account_snapshot",
                    "readiness_snapshot",
                },
            )
            self.assertEqual(report["iv_rank"]["status"], "ok")
            self.assertEqual(report["vix"]["status"], "ok")
            self.assertEqual(report["market_calendar"]["status"], "ok")
            self.assertEqual(report["account_snapshot"]["status"], "ok")
            self.assertEqual(report["readiness_snapshot"]["status"], "ok")
            self.assertEqual(
                report["safety"],
                {
                    "broker_calls_enabled": False,
                    "market_data_enabled": False,
                    "external_fetch_enabled": False,
                    "paper_live_orders_enabled": False,
                    "strategy_changes_enabled": False,
                },
            )
            json.dumps(report, sort_keys=True)

    def test_missing_sources_are_reported_without_crashing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build(
                root,
                iv_store=root / "data" / "snapshots" / "iv_rank.json",
                vix_snapshot_path=root / "data" / "snapshots" / "vix.json",
                market_calendar_path=root / "data" / "snapshots" / "market_calendar.json",
            )

            self.assertEqual(report["iv_rank"]["status"], "missing")
            self.assertEqual(report["vix"]["status"], "missing")
            self.assertEqual(report["market_calendar"]["status"], "missing")
            self.assertEqual(report["account_snapshot"]["status"], "ok")
            self.assertGreaterEqual(report["aggregate"]["missing_input_count"], 3)

    def test_corrupted_json_and_malformed_timestamps_are_invalid_not_crashes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots_dir = root / "data" / "snapshots"
            snapshots_dir.mkdir(parents=True)
            (snapshots_dir / "iv_rank.json").write_text("{bad", encoding="utf-8")
            (snapshots_dir / "vix.json").write_text("{bad", encoding="utf-8")
            (snapshots_dir / "market_calendar.json").write_text("{bad", encoding="utf-8")
            (snapshots_dir / "account.json").write_text(
                json.dumps({"timestamp": "not-a-date", "nlv": 100}),
                encoding="utf-8",
            )

            report = build(
                root,
                iv_store=snapshots_dir / "iv_rank.json",
                vix_snapshot_path=snapshots_dir / "vix.json",
                market_calendar_path=snapshots_dir / "market_calendar.json",
            )

            self.assertEqual(report["iv_rank"]["status"], "invalid")
            self.assertEqual(report["vix"]["status"], "invalid")
            self.assertEqual(report["market_calendar"]["status"], "invalid")
            self.assertEqual(report["account_snapshot"]["status"], "invalid")
            self.assertTrue(any("malformed" in warning for warning in report["warnings"]))

    def test_stale_iv_account_invalid_nlv_missing_readiness_and_halt_are_reflected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots_dir = root / "data" / "snapshots"
            state_dir = root / "data" / "state"
            snapshots_dir.mkdir(parents=True)
            state_dir.mkdir(parents=True)
            (snapshots_dir / "account.json").write_text(
                json.dumps({"timestamp": "2026-05-05T16:30:00-04:00", "nlv": -1}),
                encoding="utf-8",
            )
            (state_dir / "halt_state.json").write_text(
                json.dumps({"scope": "strategy", "id": S02_VOL_ENHANCED, "tier": "hard"}),
                encoding="utf-8",
            )
            store = FakeStateStore()
            store.readiness.pop(S02_VOL_ENHANCED)

            report = build(
                root,
                state_store=store,
                readiness_provider=lambda: None,
                iv_store={
                    "available_by_strategy": {
                        S01_VOL_BASELINE: True,
                        S02_VOL_ENHANCED: False,
                    },
                    "stale_by_strategy": {
                        S01_VOL_BASELINE: True,
                        S02_VOL_ENHANCED: False,
                    },
                },
            )

            self.assertEqual(report["iv_rank"]["status"], "stale")
            self.assertEqual(report["account_snapshot"]["status"], "invalid")
            self.assertEqual(
                report["readiness_snapshot"]["missing_readiness_strategy_ids"],
                [S02_VOL_ENHANCED],
            )
            self.assertTrue(
                report["readiness_snapshot"]["halt_active_by_strategy"][S02_VOL_ENHANCED]
            )
            self.assertIn(
                "halt_or_safety_block",
                report["per_strategy"][S02_VOL_ENHANCED]["input_blockers"],
            )
            self.assertFalse(store.saved)

    def test_account_snapshot_freshness_unknown_without_structured_freshness_input(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots_dir = root / "data" / "snapshots"
            snapshots_dir.mkdir(parents=True)
            (snapshots_dir / "account.json").write_text(
                json.dumps({"timestamp": "2026-05-05T16:00:00-04:00", "nlv": 100000}),
                encoding="utf-8",
            )

            report = build(
                root,
                readiness_provider=lambda: snapshot(account_snapshot_fresh=None),
            )

            self.assertTrue(report["account_snapshot"]["available"])
            self.assertIsNone(report["account_snapshot"]["account_snapshot_fresh"])
            self.assertEqual(report["account_snapshot"]["nlv_valid"], True)
            self.assertEqual(report["account_snapshot"]["status"], "unavailable")
            self.assertNotEqual(report["account_snapshot"]["status"], "stale")

    def test_dominant_input_issue_tie_breaks_alphabetically_and_steps_are_deterministic(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = build(
                root,
                readiness_provider=lambda: None,
                iv_store=root / "missing_iv.json",
                vix_snapshot_path=root / "missing_vix.json",
            )
            second = build(
                root,
                readiness_provider=lambda: None,
                vix_snapshot_path=root / "missing_vix.json",
                iv_store=root / "missing_iv.json",
            )

            self.assertEqual(first["aggregate"]["dominant_input_issue"], "account_snapshot")
            self.assertEqual(
                first["recommendations"]["ordered_next_steps"],
                second["recommendations"]["ordered_next_steps"],
            )

    def test_modules_do_not_import_broker_market_data_systemd_or_scheduler(self) -> None:
        core_source = Path("algo_trader_unified/core/local_input_audit_report.py").read_text(
            encoding="utf-8"
        )
        tool_source = Path("algo_trader_unified/tools/local_input_audit_report.py").read_text(
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
            "IBKR",
            "systemd",
            "UnifiedScheduler",
            "scheduler_cadence",
            "market_open_scan",
        ):
            self.assertNotIn(forbidden, combined)


class LocalInputAuditCliTests(unittest.TestCase):
    def test_missing_dry_run_only_exits_before_any_loader_or_factory(self) -> None:
        calls = []

        def factory(name):
            def _inner(*args, **kwargs):
                calls.append(name)
                return mock.Mock()

            return _inner

        err = io.StringIO()
        with redirect_stderr(err):
            code = tool.run_local_input_audit_report(
                [],
                state_store_factory=factory("state"),
                readiness_provider_factory=factory("readiness"),
                report_builder=factory("report"),
            )
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("--dry-run-only", err.getvalue())

    def test_json_outputs_strict_json_stdout_only_and_flags_are_boolean(self) -> None:
        report = {"success": True, "dry_run": True, "local_input_audit_report": True}
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = tool.run_local_input_audit_report(
                ["--dry-run-only", "--json"],
                state_store_factory=lambda path: mock.Mock(),
                readiness_provider_factory=lambda **kwargs: lambda: None,
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
            "local_input_audit_report": True,
            "generated_at": "2026-05-05T17:00:00-04:00",
            "aggregate": {
                "dominant_input_issue": "iv_rank",
                "missing_input_count": 1,
                "stale_input_count": 0,
            },
            "per_strategy": {
                S01_VOL_BASELINE: {
                    "likely_input_issue": "iv_rank",
                    "input_blockers": ["iv_rank"],
                }
            },
            "recommendations": {"ordered_next_steps": ["IV rank unavailable locally; inspect IV store capture before tuning IV thresholds."]},
        }

        def state_factory(path):
            calls.append("state")
            return mock.Mock()

        def readiness_factory(**kwargs):
            calls.append("readiness")
            return lambda: None

        def report_builder(**kwargs):
            calls.append("report")
            return report

        out = io.StringIO()
        with redirect_stdout(out):
            code = tool.run_local_input_audit_report(
                ["--dry-run-only"],
                state_store_factory=state_factory,
                readiness_provider_factory=readiness_factory,
                report_builder=report_builder,
            )
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["state", "readiness", "report"])
        self.assertIn("Local input audit report", out.getvalue())
        self.assertIn("dominant_input_issue", out.getvalue())


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
