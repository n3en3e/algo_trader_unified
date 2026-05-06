from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from algo_trader_unified.core.paper_broker_preflight_report import (
    build_paper_broker_preflight_report,
)
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.tools import paper_broker_preflight_report as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4D2_FILES = [
    ROOT / "core/paper_broker_preflight_report.py",
    ROOT / "tools/paper_broker_preflight_report.py",
]


def clean_state() -> dict:
    return {
        "positions": {},
        "order_intents": {},
        "close_intents": {},
        "active_halt": False,
    }


def fixed_now() -> datetime:
    return datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)


class PaperBrokerPreflightReportTests(unittest.TestCase):
    def test_report_includes_required_fields_and_is_json_safe(self) -> None:
        report = build_paper_broker_preflight_report(
            state_snapshot=clean_state(),
            now_provider=fixed_now,
        )

        self.assertTrue(report["dry_run"])
        self.assertTrue(report["paper_broker_preflight_report"])
        self.assertEqual(report["generated_at"], "2026-05-06T12:00:00+00:00")
        for key in (
            "inputs",
            "broker_contract",
            "scheduler_lifecycle_boundary",
            "state_safety",
            "readiness_for_next_phase",
            "recommendations",
            "safety",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        self.assertTrue(report["success"])
        json.dumps(report, sort_keys=True)

    def test_local_contract_inspection_marks_stage4d1_contract_ready(self) -> None:
        report = build_paper_broker_preflight_report(
            state_snapshot=clean_state(),
            now_provider=fixed_now,
        )

        self.assertEqual(
            report["broker_contract"],
            {
                "broker_adapter_protocol_present": True,
                "required_methods_present": True,
                "result_shapes_present": True,
                "raw_field_json_safe_contract_present": True,
                "live_mode_rejected": True,
                "paper_mode_allowed": True,
                "dry_run_mode_allowed": True,
                "fake_adapter_contract_present": True,
            },
        )
        self.assertTrue(
            report["readiness_for_next_phase"]["ready_to_design_ibkr_paper_adapter"]
        )

    def test_clean_inputs_are_ready_for_next_design_phase(self) -> None:
        acceptance = {
            "acceptance_report": True,
            "success": True,
            "startup_gate": {"halt_active": False},
            "state": {
                "unresolved_needs_reconciliation_count": 0,
                "active_intents_count": 0,
                "open_positions_count": 0,
            },
        }
        strategy = {
            "strategy_quality_decision_report": True,
            "success": True,
        }
        report = build_paper_broker_preflight_report(
            acceptance_report=acceptance,
            strategy_quality_decision_report=strategy,
            now_provider=fixed_now,
        )

        self.assertTrue(report["inputs"]["acceptance_report_success"])
        self.assertTrue(report["inputs"]["strategy_quality_decision_success"])
        self.assertTrue(
            report["readiness_for_next_phase"]["ready_to_design_ibkr_paper_adapter"]
        )
        self.assertEqual(report["readiness_for_next_phase"]["blockers"], [])

    def test_live_mode_allowed_input_blocks_next_phase(self) -> None:
        report = build_paper_broker_preflight_report(
            broker_contract_report={
                "broker_contract": {
                    "broker_adapter_protocol_present": True,
                    "required_methods_present": True,
                    "result_shapes_present": True,
                    "raw_field_json_safe_contract_present": True,
                    "live_mode_rejected": False,
                    "paper_mode_allowed": True,
                    "dry_run_mode_allowed": True,
                    "fake_adapter_contract_present": True,
                }
            },
            state_snapshot=clean_state(),
            now_provider=fixed_now,
        )

        self.assertFalse(
            report["readiness_for_next_phase"]["ready_to_design_ibkr_paper_adapter"]
        )
        self.assertIn(
            "broker contract check failed: live_mode_rejected",
            report["readiness_for_next_phase"]["blockers"],
        )
        self.assertTrue(report["success"])

    def test_missing_broker_contract_blocks_next_phase(self) -> None:
        report = build_paper_broker_preflight_report(
            broker_contract_report={"broker_contract": {}},
            state_snapshot=clean_state(),
            now_provider=fixed_now,
        )

        self.assertFalse(
            report["readiness_for_next_phase"]["ready_to_design_ibkr_paper_adapter"]
        )
        self.assertIn(
            "broker contract check failed: broker_adapter_protocol_present",
            report["readiness_for_next_phase"]["blockers"],
        )

    def test_adapter_wiring_input_blocks_next_phase(self) -> None:
        report = build_paper_broker_preflight_report(
            acceptance_report={
                "acceptance_report": True,
                "success": True,
                "scheduler_lifecycle_boundary": {
                    "adapter_wired_into_daemon": False,
                    "adapter_wired_into_scheduler": True,
                    "adapter_wired_into_lifecycle_jobs": False,
                    "lifecycle_cadence_intent_level_only": True,
                    "fill_simulation_scheduled": False,
                    "position_transition_scheduled": False,
                },
            },
            state_snapshot=clean_state(),
            now_provider=fixed_now,
        )

        self.assertFalse(
            report["readiness_for_next_phase"]["ready_to_design_ibkr_paper_adapter"]
        )
        self.assertIn(
            "scheduler lifecycle boundary violated: adapter_wired_into_scheduler",
            report["readiness_for_next_phase"]["blockers"],
        )

    def test_reconciliation_and_active_halt_block_next_phase(self) -> None:
        report = build_paper_broker_preflight_report(
            state_snapshot={
                "positions": {
                    "pos_1": {"status": "NEEDS_RECONCILIATION"},
                    "pos_2": {"status": "open"},
                },
                "order_intents": {"intent_1": {"status": "created"}},
                "close_intents": {},
                "active_halt": True,
            },
            now_provider=fixed_now,
        )

        self.assertEqual(
            report["state_safety"]["unresolved_needs_reconciliation_count"],
            1,
        )
        self.assertEqual(report["state_safety"]["active_intents_count"], 1)
        self.assertEqual(report["state_safety"]["open_positions_count"], 1)
        self.assertTrue(report["state_safety"]["active_halt"])
        self.assertIn(
            "unresolved NEEDS_RECONCILIATION records exist",
            report["readiness_for_next_phase"]["blockers"],
        )
        self.assertIn(
            "active halt is present",
            report["readiness_for_next_phase"]["blockers"],
        )

    def test_missing_state_snapshot_is_conservative_but_report_succeeds(self) -> None:
        report = build_paper_broker_preflight_report(now_provider=fixed_now)

        self.assertTrue(report["success"])
        self.assertFalse(
            report["readiness_for_next_phase"]["ready_to_design_ibkr_paper_adapter"]
        )
        self.assertIn(
            "state safety snapshot unavailable for NEEDS_RECONCILIATION",
            report["readiness_for_next_phase"]["blockers"],
        )

    def test_recommendations_are_non_binding_and_do_not_enable_execution(self) -> None:
        report = build_paper_broker_preflight_report(
            state_snapshot=clean_state(),
            now_provider=fixed_now,
        )
        ordered = "\n".join(report["recommendations"]["ordered_next_steps"]).lower()

        self.assertIn("behind brokeradapter protocol", ordered)
        self.assertIn("keep adapter unmounted", ordered)
        self.assertNotIn("enable paper execution", ordered)
        self.assertNotIn("place paper orders", ordered)
        self.assertNotIn("enable live", ordered)
        self.assertIn(
            "Do not enable paper order submission yet.",
            report["recommendations"]["do_not_do_yet"],
        )


class FakeStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.state = clean_state()
        self.saved = False

    def save(self) -> None:
        self.saved = True
        raise AssertionError("preflight report must not mutate StateStore")


class PaperBrokerPreflightCliTests(unittest.TestCase):
    def test_cli_requires_dry_run_before_state_factory_or_load(self) -> None:
        calls = []

        def state_store_factory(path: Path):
            calls.append(path)
            raise AssertionError("factory must not be called")

        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = tool.run_paper_broker_preflight_report(
                ["--json"],
                state_store_factory=state_store_factory,
            )

        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("--dry-run-only", err.getvalue())

    def test_cli_json_writes_strict_json_to_stdout_only(self) -> None:
        out = io.StringIO()
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_store = StateStore(root / "data/state/portfolio_state.json")
            state_store.state["halt_state"] = {"tier": "info"}
            state_store.save()
            with redirect_stdout(out), redirect_stderr(err):
                code = tool.run_paper_broker_preflight_report(
                    ["--dry-run-only", "--json", "--root", str(root)]
                )

        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["paper_broker_preflight_report"])

    def test_cli_human_readable_output_is_default(self) -> None:
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = tool.run_paper_broker_preflight_report(
                ["--dry-run-only"],
                state_store_factory=FakeStateStore,
            )

        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        self.assertIn("Paper broker preflight report", out.getvalue())
        with self.assertRaises(json.JSONDecodeError):
            json.loads(out.getvalue())

    def test_cli_flags_are_boolean_store_true_actions(self) -> None:
        actions = []
        original_add_argument = argparse.ArgumentParser.add_argument

        def spy_add_argument(self, *args, **kwargs):
            if "--dry-run-only" in args or "--json" in args:
                actions.append((args, kwargs.get("action")))
            return original_add_argument(self, *args, **kwargs)

        argparse.ArgumentParser.add_argument = spy_add_argument
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                tool.run_paper_broker_preflight_report(
                    ["--dry-run-only"],
                    state_store_factory=FakeStateStore,
                )
        finally:
            argparse.ArgumentParser.add_argument = original_add_argument

        self.assertIn((("--dry-run-only",), "store_true"), actions)
        self.assertIn((("--json",), "store_true"), actions)

    def test_cli_does_not_mutate_state_store_or_write_ledgers_or_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_store = StateStore(root / "data/state/portfolio_state.json")
            before = json.dumps(state_store.state, sort_keys=True)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = tool.run_paper_broker_preflight_report(
                    ["--dry-run-only", "--root", str(root)]
                )
            after_store = StateStore(root / "data/state/portfolio_state.json")
            ledger_dir = root / "data/ledger"
            snapshots_dir = root / "data/snapshots"

        self.assertEqual(code, 0)
        self.assertEqual(json.dumps(after_store.state, sort_keys=True), before)
        self.assertFalse(ledger_dir.exists())
        self.assertFalse(snapshots_dir.exists())


class Stage4D2SafetyBoundaryTests(unittest.TestCase):
    def test_stage4d2_files_do_not_import_or_call_blocked_integrations(self) -> None:
        blocked_tokens = (
            "ib_" + "insync",
            "req" + "MktData",
            "place" + "Order",
            "y" + "finance",
            "requests",
            "url" + "lib",
            "system" + "ctl",
            "system" + "d",
        )
        for path in STAGE4D2_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_stage4d2_files_do_not_import_broker_client_symbols(self) -> None:
        for path in STAGE4D2_FILES:
            source = path.read_text(encoding="utf-8")
            forbidden_lines = (
                "from ib",
                "import ib",
                " IB(",
                "= IB",
            )
            for token in forbidden_lines:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_contract_inspection_does_not_use_source_regex_or_ast_parsing(self) -> None:
        source = (ROOT / "core/paper_broker_preflight_report.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("import ast", source)
        self.assertNotIn("import re", source)
        self.assertNotIn(".read_text(", source)
        self.assertNotIn("ast.parse", source)
        self.assertNotIn("re.compile", source)
        self.assertNotIn("re.search", source)
        self.assertNotIn("re.match", source)

    def test_report_core_does_not_start_scheduler_or_run_jobs(self) -> None:
        source = (ROOT / "core/paper_broker_preflight_report.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn(".start(", source)
        self.assertNotIn("run_bounded", source)
        self.assertNotIn("run_intent_submission_job", source)
        self.assertNotIn("run_position_transitions_job", source)


if __name__ == "__main__":
    unittest.main()
