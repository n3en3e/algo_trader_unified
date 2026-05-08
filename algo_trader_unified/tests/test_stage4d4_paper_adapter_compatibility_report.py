from __future__ import annotations

import argparse
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from algo_trader_unified.core import paper_adapter_compatibility_report as core_report
from algo_trader_unified.core.paper_adapter_compatibility_report import (
    build_paper_adapter_compatibility_report,
)
from algo_trader_unified.core.paper_broker_adapter import build_broker_order_request
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.tools import paper_adapter_compatibility_report as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4D4_FILES = [
    ROOT / "core/paper_adapter_compatibility_report.py",
    ROOT / "tools/paper_adapter_compatibility_report.py",
]


def fixed_now() -> datetime:
    return datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


def valid_intent(intent_id: str = "intent_1", strategy_id: str = "S01_VOL_BASELINE") -> dict:
    return {
        "intent_id": intent_id,
        "strategy_id": strategy_id,
        "symbol": "XSP",
        "asset_type": "index_option",
        "side": "BUY",
        "quantity": 1,
        "order_type": "LIMIT",
        "limit_price": Decimal("1.25"),
        "time_in_force": "DAY",
        "metadata": {"note": "dry-run"},
    }


def state_store_intent(intent_id: str = "intent_1") -> dict:
    intent = valid_intent(intent_id)
    intent["limit_price"] = 1.25
    return intent


class PaperAdapterCompatibilityReportCoreTests(unittest.TestCase):
    def build(self, intents: list[dict], strategy_ids: list[str] | None = None) -> dict:
        return build_paper_adapter_compatibility_report(
            order_intents=intents,
            strategy_ids=strategy_ids or ["S01_VOL_BASELINE"],
            now_provider=fixed_now,
        )

    def test_report_is_pure_json_safe_and_has_required_fields(self) -> None:
        intents = [valid_intent()]
        before = copy.deepcopy(intents)

        report = self.build(intents)

        self.assertEqual(intents, before)
        self.assertTrue(report["dry_run"])
        self.assertTrue(report["paper_adapter_compatibility_report"])
        self.assertEqual(report["generated_at"], "2026-05-08T12:00:00+00:00")
        for key in (
            "strategy_ids",
            "inputs",
            "per_strategy",
            "aggregate",
            "validation",
            "recommendations",
            "safety",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        self.assertTrue(report["success"])
        json.dumps(report, sort_keys=True)

    def test_valid_intents_are_counted_compatible_and_deterministic(self) -> None:
        intents = [valid_intent("intent_1"), valid_intent("intent_2")]

        report = self.build(intents)

        self.assertEqual(report["aggregate"]["total_intents_seen"], 2)
        self.assertEqual(report["aggregate"]["total_intents_valid"], 2)
        self.assertEqual(report["aggregate"]["total_intents_invalid"], 0)
        self.assertEqual(report["aggregate"]["compatibility_rate"], 1.0)
        self.assertTrue(report["aggregate"]["all_intents_compatible"])
        self.assertTrue(report["validation"]["deterministic_client_order_ids"])
        self.assertEqual(report["validation"]["duplicate_client_order_ids"], [])
        self.assertEqual(
            report["per_strategy"]["S01_VOL_BASELINE"][
                "sample_valid_client_order_ids"
            ],
            ["intent_1", "intent_2"],
        )
        first = build_broker_order_request(intents[0])
        second = build_broker_order_request(intents[0])
        self.assertEqual(first.client_order_id, second.client_order_id)

    def test_invalid_intents_are_grouped_by_stable_reasons(self) -> None:
        intents = [
            {**valid_intent("missing_strategy"), "strategy_id": ""},
            {**valid_intent("missing_symbol"), "symbol": "", "underlying": ""},
            {**valid_intent("bad_side"), "side": "HOLD"},
            {**valid_intent("bad_quantity"), "quantity": 0},
            {**valid_intent("market_with_limit"), "order_type": "MARKET"},
            {**valid_intent("limit_without_price"), "limit_price": None},
        ]

        report = self.build(intents)

        self.assertEqual(report["aggregate"]["total_intents_seen"], 6)
        self.assertEqual(report["aggregate"]["total_intents_valid"], 0)
        self.assertEqual(report["aggregate"]["total_intents_invalid"], 6)
        self.assertFalse(report["aggregate"]["all_intents_compatible"])
        reasons = report["per_strategy"]["S01_VOL_BASELINE"]["invalid_reasons"]
        self.assertEqual(reasons["ValueError: symbol or underlying is required"], 1)
        self.assertEqual(reasons["ValueError: side must be one of ['BUY', 'SELL']"], 1)
        self.assertEqual(reasons["ValueError: quantity must be positive numeric"], 1)
        self.assertEqual(
            reasons["ValueError: limit_price is only applicable for LIMIT orders"],
            1,
        )
        self.assertEqual(reasons["ValueError: limit_price must be positive numeric"], 1)
        missing_reasons = report["per_strategy"]["__missing_strategy_id__"][
            "invalid_reasons"
        ]
        self.assertEqual(missing_reasons["ValueError: strategy_id is required"], 1)
        self.assertEqual(report["validation"]["unsupported_sides"], ["HOLD"])
        self.assertEqual(
            report["validation"]["missing_required_fields"]["strategy_id"],
            1,
        )
        self.assertEqual(
            report["validation"]["missing_required_fields"]["symbol_or_underlying"],
            1,
        )
        self.assertEqual(
            report["validation"]["missing_required_fields"]["quantity"],
            1,
        )
        self.assertEqual(
            report["validation"]["missing_required_fields"]["limit_price"],
            1,
        )

    def test_unsupported_order_type_is_reported(self) -> None:
        report = self.build([{**valid_intent("bad_type"), "order_type": "STOP"}])

        self.assertEqual(report["validation"]["unsupported_order_types"], ["STOP"])
        self.assertEqual(
            report["aggregate"]["dominant_invalid_reason"],
            "ValueError: order_type must be one of ['LIMIT', 'MARKET']",
        )
        self.assertIn(
            "Fix unsupported order_type before paper adapter wiring.",
            report["recommendations"]["ordered_next_steps"],
        )

    def test_malformed_intent_exception_is_caught_and_processing_continues(self) -> None:
        report = build_paper_adapter_compatibility_report(
            order_intents=[None, valid_intent("intent_2")],  # type: ignore[list-item]
            strategy_ids=["S01_VOL_BASELINE"],
            now_provider=fixed_now,
        )

        self.assertEqual(report["aggregate"]["total_intents_seen"], 2)
        self.assertEqual(report["aggregate"]["total_intents_valid"], 1)
        self.assertEqual(report["aggregate"]["total_intents_invalid"], 1)
        self.assertIn(
            "ValueError: intent must be a dict",
            report["per_strategy"]["__missing_strategy_id__"]["invalid_reasons"],
        )

    def test_duplicate_client_order_ids_are_compatibility_blockers(self) -> None:
        report = self.build([valid_intent("dup"), valid_intent("dup")])

        self.assertEqual(report["validation"]["duplicate_client_order_ids"], ["dup"])
        self.assertFalse(report["validation"]["deterministic_client_order_ids"])
        self.assertFalse(report["aggregate"]["all_intents_compatible"])
        self.assertEqual(report["aggregate"]["total_intents_valid"], 0)
        self.assertEqual(report["aggregate"]["total_intents_invalid"], 2)
        self.assertEqual(
            report["per_strategy"]["S01_VOL_BASELINE"]["intents_valid"],
            0,
        )
        self.assertEqual(
            report["per_strategy"]["S01_VOL_BASELINE"]["intents_invalid"],
            2,
        )
        self.assertEqual(
            report["per_strategy"]["S01_VOL_BASELINE"]["invalid_reasons"][
                "duplicate client_order_id"
            ],
            2,
        )
        self.assertEqual(
            report["aggregate"]["dominant_invalid_reason"],
            "duplicate client_order_id",
        )
        self.assertIn(
            "Investigate duplicate client_order_id generation before paper adapter wiring.",
            report["recommendations"]["ordered_next_steps"],
        )

    def test_three_duplicate_client_order_ids_count_three_affected_occurrences(self) -> None:
        report = self.build(
            [valid_intent("dup"), valid_intent("dup"), valid_intent("dup")]
        )

        self.assertEqual(report["validation"]["duplicate_client_order_ids"], ["dup"])
        self.assertEqual(report["aggregate"]["total_intents_valid"], 0)
        self.assertEqual(report["aggregate"]["total_intents_invalid"], 3)
        self.assertEqual(
            report["per_strategy"]["S01_VOL_BASELINE"]["invalid_reasons"][
                "duplicate client_order_id"
            ],
            3,
        )

    def test_second_determinism_translation_exception_is_caught(self) -> None:
        original = core_report.build_broker_order_request
        calls = {"count": 0}

        def flaky_builder(intent):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("second pass failed")
            return original(intent)

        core_report.build_broker_order_request = flaky_builder
        try:
            report = self.build([valid_intent("intent_1")])
        finally:
            core_report.build_broker_order_request = original

        self.assertEqual(report["aggregate"]["total_intents_invalid"], 1)
        self.assertFalse(report["validation"]["deterministic_client_order_ids"])
        self.assertIn(
            "RuntimeError: second pass failed",
            report["per_strategy"]["S01_VOL_BASELINE"]["invalid_reasons"],
        )

    def test_second_determinism_translation_failure_updates_validation_breakdown(self) -> None:
        original = core_report.build_broker_order_request
        calls = {"count": 0}

        def flaky_builder(intent):
            calls["count"] += 1
            if calls["count"] == 2:
                raise ValueError("quantity must be positive numeric")
            return original(intent)

        core_report.build_broker_order_request = flaky_builder
        try:
            report = self.build([valid_intent("intent_1")])
        finally:
            core_report.build_broker_order_request = original

        self.assertEqual(report["aggregate"]["total_intents_valid"], 0)
        self.assertEqual(report["aggregate"]["total_intents_invalid"], 1)
        self.assertEqual(
            report["validation"]["missing_required_fields"]["quantity"],
            1,
        )

    def test_empty_intent_list_is_graceful_with_insufficient_evidence_step(self) -> None:
        report = self.build([])

        self.assertTrue(report["success"])
        self.assertEqual(report["aggregate"]["total_intents_seen"], 0)
        self.assertFalse(report["aggregate"]["all_intents_compatible"])
        self.assertIn(
            "Collect more dry-run intents before paper adapter compatibility can be assessed.",
            report["recommendations"]["ordered_next_steps"],
        )

    def test_dominant_invalid_reason_tie_breaks_alphabetically(self) -> None:
        report = self.build(
            [
                {**valid_intent("bad_side"), "side": "HOLD"},
                {**valid_intent("bad_type"), "order_type": "STOP"},
            ]
        )

        self.assertEqual(
            report["aggregate"]["dominant_invalid_reason"],
            "ValueError: order_type must be one of ['LIMIT', 'MARKET']",
        )
        self.assertEqual(
            report["recommendations"]["ordered_next_steps"],
            ["Fix unsupported order_type before paper adapter wiring."],
        )

    def test_report_does_not_require_filesystem_or_state_store_inputs(self) -> None:
        report = self.build([valid_intent()])

        self.assertTrue(report["success"])
        self.assertEqual(report["inputs"]["order_intents_count"], 1)


class FakeStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.saved = False
        self.order_intents = [state_store_intent()]

    def list_order_intents(self):
        return copy.deepcopy(self.order_intents)

    def save(self) -> None:
        self.saved = True
        raise AssertionError("compatibility report must not mutate StateStore")


class PaperAdapterCompatibilityCliTests(unittest.TestCase):
    def test_cli_requires_dry_run_before_state_factory_or_load(self) -> None:
        calls = []

        def state_store_factory(path: Path):
            calls.append(path)
            raise AssertionError("factory must not be called")

        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = tool.run_paper_adapter_compatibility_report(
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
            state_store.create_order_intent(state_store_intent())
            with redirect_stdout(out), redirect_stderr(err):
                code = tool.run_paper_adapter_compatibility_report(
                    ["--dry-run-only", "--json", "--root", str(root)]
                )

        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["paper_adapter_compatibility_report"])

    def test_cli_human_readable_output_is_default(self) -> None:
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = tool.run_paper_adapter_compatibility_report(
                ["--dry-run-only"],
                state_store_factory=FakeStateStore,
            )

        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        self.assertIn("Paper adapter compatibility report", out.getvalue())
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
                tool.run_paper_adapter_compatibility_report(
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
            state_store.create_order_intent(state_store_intent())
            before = json.dumps(state_store.state, sort_keys=True)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = tool.run_paper_adapter_compatibility_report(
                    ["--dry-run-only", "--root", str(root)]
                )
            after_store = StateStore(root / "data/state/portfolio_state.json")
            ledger_dir = root / "data/ledger"
            snapshots_dir = root / "data/snapshots"

        self.assertEqual(code, 0)
        self.assertEqual(json.dumps(after_store.state, sort_keys=True), before)
        self.assertFalse(ledger_dir.exists())
        self.assertFalse(snapshots_dir.exists())


class Stage4D4SafetyBoundaryTests(unittest.TestCase):
    def test_stage4d4_files_do_not_import_or_call_blocked_integrations(self) -> None:
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
        for path in STAGE4D4_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_stage4d4_files_do_not_instantiate_or_submit_through_adapter(self) -> None:
        blocked_tokens = (
            "Paper" + "BrokerAdapter(",
            "submit" + "_order_intent(",
        )
        for path in STAGE4D4_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
