from __future__ import annotations

import contextlib
import io
import json
import py_compile
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.scheduler import JOB_SPECS, JOB_S01_VOL_SCAN
from algo_trader_unified.core.readiness_report import build_dry_run_readiness_report
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.tools.dry_run_readiness import main as readiness_main


NOW = datetime(2026, 5, 4, 14, 0, tzinfo=timezone.utc)


class Phase3YCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def state_store(self) -> StateStore:
        return StateStore(self.root / "data/state/portfolio_state.json")

    def report(self, state_store=None, **kwargs) -> dict:
        return build_dry_run_readiness_report(
            root_dir=kwargs.pop("root_dir", self.root),
            state_store=state_store,
            now=kwargs.pop("now", NOW),
            **kwargs,
        )


class CoreReadinessReportTests(Phase3YCase):
    def test_fresh_valid_state_is_json_safe_and_read_only(self) -> None:
        store = self.state_store()
        before_state = deepcopy(store.state)
        before_text = store.path.read_text(encoding="utf-8")

        report = self.report(store)

        self.assertIs(report["dry_run"], True)
        self.assertIn(report["status"], {"ready", "warning"})
        self.assertTrue(report["ready"])
        self.assertEqual(report["next_action"], "run_dry_run_chain")
        json.dumps(report)
        self.assertEqual(store.state, before_state)
        self.assertEqual(store.path.read_text(encoding="utf-8"), before_text)
        self.assertFalse((self.root / "data/ledger/order_ledger.jsonl").exists())
        self.assertFalse((self.root / "data/ledger/execution_ledger.jsonl").exists())

    def test_missing_required_scheduler_job_blocks(self) -> None:
        store = self.state_store()
        job_specs = dict(JOB_SPECS)
        job_specs.pop(JOB_S01_VOL_SCAN)

        report = self.report(store, job_specs=job_specs)

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["next_action"], "fix_scheduler_registration")
        self.assertTrue(
            any(issue["code"] == "missing_required_job" for issue in report["blocking_issues"])
        )

    def test_optional_s02_missing_warns_only(self) -> None:
        from algo_trader_unified.config.scheduler import JOB_S02_MANAGEMENT_SCAN, JOB_S02_VOL_SCAN

        store = self.state_store()
        job_specs = dict(JOB_SPECS)
        job_specs.pop(JOB_S02_VOL_SCAN)
        job_specs.pop(JOB_S02_MANAGEMENT_SCAN)

        report = self.report(store, job_specs=job_specs)

        self.assertNotEqual(report["status"], "blocked")
        self.assertTrue(
            any(warning["code"] == "missing_optional_s02_job" for warning in report["warnings"])
        )

    def test_forbidden_scheduler_job_blocks(self) -> None:
        store = self.state_store()
        job_specs = dict(JOB_SPECS)
        job_specs["live_order_submission"] = object()

        report = self.report(store, job_specs=job_specs)

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(
            any(issue["code"] == "forbidden_scheduler_job" for issue in report["blocking_issues"])
        )

    def test_missing_required_function_blocks_with_clean_monkeypatch(self) -> None:
        store = self.state_store()
        import algo_trader_unified.jobs.submission as submission

        with mock.patch.object(submission, "run_intent_submission_job", None):
            report = self.report(store)

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(
            any(issue["code"] == "missing_required_function" for issue in report["blocking_issues"])
        )

    def test_missing_state_file_is_not_created(self) -> None:
        state_path = self.root / "data/state/portfolio_state.json"

        report = self.report(None)

        self.assertFalse(state_path.exists())
        self.assertNotEqual(report["status"], "blocked")

    def test_corrupt_state_file_blocks_without_rewrite(self) -> None:
        state_path = self.root / "data/state/portfolio_state.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text("{bad json", encoding="utf-8")
        before = state_path.read_text(encoding="utf-8")
        try:
            store_or_error = StateStore(state_path)
        except Exception as exc:
            store_or_error = exc

        report = self.report(store_or_error)

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(state_path.read_text(encoding="utf-8"), before)
        self.assertTrue(
            any(issue["code"] == "corrupt_state_store" for issue in report["blocking_issues"])
        )

    def test_ledger_missing_readable_and_corrupt_cases(self) -> None:
        store = self.state_store()
        missing_report = self.report(store)
        self.assertFalse(
            any(issue["code"] == "corrupt_ledger" for issue in missing_report["blocking_issues"])
        )

        ledger_dir = self.root / "data/ledger"
        ledger_dir.mkdir(parents=True)
        (ledger_dir / "order_ledger.jsonl").write_text('{"event_type":"ORDER"}\n', encoding="utf-8")
        (ledger_dir / "execution_ledger.jsonl").write_text("{bad json\n", encoding="utf-8")

        corrupt_report = self.report(store)

        self.assertEqual(corrupt_report["status"], "blocked")
        self.assertTrue(
            any(issue["code"] == "corrupt_ledger" for issue in corrupt_report["blocking_issues"])
        )

    def test_state_summary_counts_and_pending_warnings(self) -> None:
        store = self.state_store()
        store.state["order_intents"] = {
            "oi1": {"intent_id": "oi1", "status": "created", "updated_at": NOW.isoformat()},
            "oi2": {"intent_id": "oi2", "status": "filled", "updated_at": NOW.isoformat()},
            "oi3": {"intent_id": "oi3", "status": "position_opened"},
        }
        store.state["positions"] = {
            "p1": {"position_id": "p1", "strategy_id": "s01", "symbol": "XSP", "status": "open"},
            "p2": {"position_id": "p2", "strategy_id": "s02", "symbol": "XSP", "status": "closed"},
        }
        store.state["close_intents"] = {
            "ci1": {
                "close_intent_id": "ci1",
                "position_id": "p1",
                "status": "submitted",
                "updated_at": NOW.isoformat(),
            }
        }

        report = self.report(store)
        summary = report["summary"]

        self.assertEqual(summary["total_order_intents_count"], 3)
        self.assertEqual(summary["active_order_intents_count"], 2)
        self.assertEqual(summary["total_close_intents_count"], 1)
        self.assertEqual(summary["active_close_intents_count"], 1)
        self.assertEqual(summary["open_positions_count"], 1)
        self.assertEqual(summary["closed_positions_count"], 1)
        self.assertEqual(report["next_action"], "run_dry_run_chain")
        self.assertTrue(any("pending_order_intent" in warning["code"] for warning in report["warnings"]))

    def test_stale_pending_lifecycle_warns_not_blocks(self) -> None:
        store = self.state_store()
        old = (NOW - timedelta(minutes=90)).isoformat()
        store.state["order_intents"] = {
            "oi1": {"intent_id": "oi1", "status": "created", "updated_at": old}
        }

        report = self.report(store)

        self.assertNotEqual(report["status"], "blocked")
        self.assertTrue(any(warning["code"] == "stale_order_intent" for warning in report["warnings"]))


class InconsistentStateTests(Phase3YCase):
    def test_duplicate_active_close_intents_block(self) -> None:
        store = self.state_store()
        store.state["positions"] = {
            "p1": {"position_id": "p1", "strategy_id": "s01", "symbol": "XSP", "status": "open"}
        }
        store.state["close_intents"] = {
            "ci1": {"close_intent_id": "ci1", "position_id": "p1", "status": "created"},
            "ci2": {"close_intent_id": "ci2", "position_id": "p1", "status": "submitted"},
        }

        report = self.report(store)

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any(issue["code"] == "duplicate_active_close_intents" for issue in report["blocking_issues"]))

    def test_position_active_close_id_missing_blocks(self) -> None:
        store = self.state_store()
        store.state["positions"] = {
            "p1": {
                "position_id": "p1",
                "strategy_id": "s01",
                "symbol": "XSP",
                "status": "open",
                "active_close_intent_id": "missing",
            }
        }

        report = self.report(store)

        self.assertTrue(any(issue["code"] == "position_missing_active_close_intent" for issue in report["blocking_issues"]))

    def test_position_closed_close_intent_is_not_active_by_itself(self) -> None:
        store = self.state_store()
        store.state["positions"] = {
            "p1": {"position_id": "p1", "strategy_id": "s01", "symbol": "XSP", "status": "closed"}
        }
        store.state["close_intents"] = {
            "ci1": {"close_intent_id": "ci1", "position_id": "p1", "status": "position_closed"}
        }

        report = self.report(store)

        self.assertFalse(
            any(issue["code"] == "position_closed_intent_still_active" for issue in report["blocking_issues"])
        )

    def test_open_position_linked_to_position_closed_close_blocks(self) -> None:
        store = self.state_store()
        store.state["positions"] = {
            "p1": {
                "position_id": "p1",
                "strategy_id": "s01",
                "symbol": "XSP",
                "status": "open",
                "active_close_intent_id": "ci1",
            }
        }
        store.state["close_intents"] = {
            "ci1": {"close_intent_id": "ci1", "position_id": "p1", "status": "position_closed"}
        }

        report = self.report(store)

        self.assertTrue(any(issue["code"] == "open_position_closed_close_intent" for issue in report["blocking_issues"]))

    def test_duplicate_open_positions_for_strategy_symbol_block(self) -> None:
        store = self.state_store()
        store.state["positions"] = {
            "p1": {"position_id": "p1", "strategy_id": "s01", "symbol": "XSP", "status": "open"},
            "p2": {"position_id": "p2", "strategy_id": "s01", "symbol": "XSP", "status": "open"},
        }

        report = self.report(store)

        self.assertTrue(any(issue["code"] == "duplicate_open_positions" for issue in report["blocking_issues"]))


class DryRunReadinessCliTests(Phase3YCase):
    def run_cli(self, *args: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = readiness_main(list(args))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_json_output_is_strict_json_and_warning_exits_zero(self) -> None:
        code, stdout, stderr = self.run_cli("--root-dir", str(self.root), "--json", "--now", "2026-05-04T14:00:00Z")

        payload = json.loads(stdout)
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIs(payload["dry_run"], True)
        self.assertFalse((self.root / "data/state/portfolio_state.json").exists())

    def test_human_output_contains_status_and_counts(self) -> None:
        self.state_store()

        code, stdout, stderr = self.run_cli("--root-dir", str(self.root), "--dry-run")

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertRegex(stdout, r"READY|WARNING|BLOCKED")
        self.assertIn("blocking_issues:", stdout)
        self.assertIn("next_action:", stdout)

    def test_blocked_report_exits_nonzero(self) -> None:
        state_path = self.root / "data/state/portfolio_state.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text("{bad json", encoding="utf-8")

        code, stdout, stderr = self.run_cli("--root-dir", str(self.root))

        self.assertEqual(code, 1)
        self.assertEqual(stderr, "")
        self.assertIn("BLOCKED", stdout)

    def test_invalid_now_exits_nonzero_without_mutation(self) -> None:
        code, stdout, stderr = self.run_cli("--root-dir", str(self.root), "--now", "not-a-date")

        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("--now must be an ISO timestamp", stderr)
        self.assertFalse((self.root / "data/state/portfolio_state.json").exists())


class ReadinessSafetySourceTests(unittest.TestCase):
    def test_readiness_sources_do_not_import_live_dependencies_or_start_scheduler(self) -> None:
        paths = [
            Path("algo_trader_unified/core/readiness_report.py"),
            Path("algo_trader_unified/tools/dry_run_readiness.py"),
        ]
        forbidden = [
            "ib_insync",
            "yfinance",
            "requests",
            "scheduler.start(",
            "run_dry_run_job_chain(",
            "LedgerAppender.append",
            ".save(",
            "placeOrder",
            "cancelOrder",
            "except:",
        ]
        for path in paths:
            source = path.read_text(encoding="utf-8")
            for snippet in forbidden:
                self.assertNotIn(snippet, source, msg=f"{snippet} found in {path}")

    def test_py_compile_clean(self) -> None:
        py_compile.compile("algo_trader_unified/core/readiness_report.py", doraise=True)
        py_compile.compile("algo_trader_unified/tools/dry_run_readiness.py", doraise=True)


if __name__ == "__main__":
    unittest.main()
