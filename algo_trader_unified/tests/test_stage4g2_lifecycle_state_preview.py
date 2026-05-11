from __future__ import annotations

import copy
from datetime import datetime, timezone
from decimal import Decimal
import inspect
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4g2_lifecycle_state_preview import (
    build_stage4g2_lifecycle_state_preview,
)
from algo_trader_unified.tools import stage4g2_lifecycle_state_preview as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4G2_FILES = [
    ROOT / "core/stage4g2_lifecycle_state_preview.py",
    ROOT / "tools/stage4g2_lifecycle_state_preview.py",
]
UNWIRED_RUNTIME_FILES = [
    ROOT / "tools/daemon.py",
    ROOT / "core/scheduler.py",
    ROOT / "core/scheduler_cadence.py",
    ROOT / "core/job_chain.py",
    ROOT / "jobs/submission.py",
    ROOT / "jobs/confirmation.py",
    ROOT / "jobs/fill_confirmation.py",
    ROOT / "jobs/position_transitions.py",
]


def valid_intake(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4g1_lifecycle_intake_report": True,
        "generated_at": "2026-05-10T12:00:00+00:00",
        "lifecycle_intake_candidate": {
            "available": True,
            "broker_order_id": "9001",
            "client_order_id": "intent-stage4g2-001",
            "strategy_id": "S01_VOL_BASELINE",
            "symbol": "XSP",
            "action": "BUY",
            "quantity": 1,
            "order_type": "LIMIT",
            "filled_quantity": 0,
            "remaining_quantity": 1,
            "avg_fill_price": 0,
            "suggested_internal_lifecycle_state": "broker_submitted",
            "reconciliation_required": False,
            "reconciliation_reasons": [],
        },
        "safety_checks": {
            "no_live_orders": True,
            "no_market_data": True,
            "no_contract_qualification": True,
            "no_scheduler_changes": True,
            "no_lifecycle_wiring": True,
            "no_state_mutation": True,
            "no_ledger_writes": True,
        },
        "readiness_for_stage4g2": {
            "ready_to_build_manual_lifecycle_state_preview": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def build(**overrides: object) -> dict:
    kwargs = {
        "lifecycle_intake_report": valid_intake(),
        "existing_state_snapshot": None,
        "now_provider": lambda: datetime(2026, 5, 10, 12, 30, tzinfo=timezone.utc),
    }
    kwargs.update(overrides)
    return build_stage4g2_lifecycle_state_preview(**kwargs)


def candidate(**overrides: object) -> dict[str, object]:
    report = valid_intake()
    base = copy.deepcopy(report["lifecycle_intake_candidate"])
    assert isinstance(base, dict)
    _deep_update(base, overrides)
    return base


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


def assert_json_safe(test_case: unittest.TestCase, report: dict) -> None:
    serialized = json.dumps(report, sort_keys=True)
    test_case.assertNotIn("datetime", serialized)
    test_case.assertNotIn("Decimal", serialized)


class Stage4G2LifecycleStatePreviewTests(unittest.TestCase):
    def assert_blocked(self, report: dict, expected: str) -> None:
        self.assertFalse(
            report["readiness_for_stage4g3"][
                "ready_to_build_manual_state_write_proposal"
            ]
        )
        self.assertIn(expected, report["readiness_for_stage4g3"]["blockers"])

    def assert_preview_state(self, report: dict, expected: str) -> None:
        self.assertEqual(report["preview"]["proposed_lifecycle_state"], expected)

    def test_missing_lifecycle_intake_report_blocks_readiness(self) -> None:
        self.assert_blocked(
            build(lifecycle_intake_report=None),
            "lifecycle_intake_report missing",
        )

    def test_lifecycle_intake_not_ready_blocks_readiness(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(
                readiness_for_stage4g2={
                    "ready_to_build_manual_lifecycle_state_preview": False
                }
            )
        )
        self.assert_blocked(report, "Stage 4G-1 intake is not ready for Stage 4G-2")

    def test_missing_lifecycle_candidate_blocks_readiness(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(lifecycle_intake_candidate={"available": False})
        )
        self.assert_blocked(report, "lifecycle_intake_candidate is not available")

    def test_missing_broker_order_id_blocks_readiness(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(
                lifecycle_intake_candidate={"broker_order_id": None}
            )
        )
        self.assert_blocked(report, "broker_order_id is required")

    def test_missing_client_order_id_blocks_readiness(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(
                lifecycle_intake_candidate={"client_order_id": None}
            )
        )
        self.assert_blocked(report, "client_order_id is required")

    def test_broker_submitted_maps_to_paper_order_submitted(self) -> None:
        report = build()

        self.assertTrue(
            report["readiness_for_stage4g3"][
                "ready_to_build_manual_state_write_proposal"
            ]
        )
        self.assert_preview_state(report, "paper_order_submitted")
        self.assertEqual(report["preview"]["proposed_order_record"]["status"], "submitted")
        self.assertIsNone(report["preview"]["proposed_position_record"])

    def test_broker_filled_maps_to_paper_order_filled_and_creates_position(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(
                lifecycle_intake_candidate=candidate(
                    suggested_internal_lifecycle_state="broker_filled",
                    filled_quantity="1",
                    remaining_quantity="0",
                    avg_fill_price=Decimal("1.25"),
                )
            )
        )

        self.assert_preview_state(report, "paper_order_filled")
        self.assertEqual(report["preview"]["proposed_order_record"]["status"], "filled")
        position = report["preview"]["proposed_position_record"]
        self.assertEqual(position["quantity"], 1.0)
        self.assertEqual(position["avg_fill_price"], 1.25)

    def test_broker_cancelled_maps_to_paper_order_cancelled(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(
                lifecycle_intake_candidate=candidate(
                    suggested_internal_lifecycle_state="broker_cancelled"
                )
            )
        )

        self.assert_preview_state(report, "paper_order_cancelled")
        self.assertEqual(report["preview"]["proposed_order_record"]["status"], "cancelled")
        self.assertIsNone(report["preview"]["proposed_position_record"])

    def test_broker_partially_filled_maps_and_blocks_readiness(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(
                lifecycle_intake_candidate=candidate(
                    suggested_internal_lifecycle_state="broker_partially_filled",
                    filled_quantity="0.5",
                    reconciliation_required=True,
                )
            )
        )

        self.assert_preview_state(report, "paper_order_partially_filled_review")
        self.assertEqual(
            report["preview"]["proposed_order_record"]["status"],
            "partially_filled",
        )
        self.assertEqual(report["preview"]["proposed_position_record"]["quantity"], 0.5)
        self.assert_blocked(
            report,
            "proposed lifecycle state requires manual review: paper_order_partially_filled_review",
        )

    def test_broker_rejected_or_inactive_maps_and_blocks_readiness(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(
                lifecycle_intake_candidate=candidate(
                    suggested_internal_lifecycle_state="broker_rejected_or_inactive",
                    reconciliation_required=True,
                )
            )
        )

        self.assert_preview_state(report, "paper_order_rejected_or_inactive_review")
        self.assert_blocked(
            report,
            "proposed lifecycle state requires manual review: paper_order_rejected_or_inactive_review",
        )

    def test_submitted_unverified_maps_with_warning(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(
                lifecycle_intake_candidate=candidate(
                    suggested_internal_lifecycle_state="submitted_unverified",
                    reconciliation_required=True,
                )
            )
        )

        self.assert_preview_state(report, "paper_order_submitted_unverified")
        self.assertTrue(
            report["readiness_for_stage4g3"][
                "ready_to_build_manual_state_write_proposal"
            ]
        )
        self.assertIn(
            "manual follow-up required before relying on paper_order_submitted_unverified",
            report["warnings"],
        )

    def test_cancel_requested_unverified_maps_with_warning(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(
                lifecycle_intake_candidate=candidate(
                    suggested_internal_lifecycle_state="cancel_requested_unverified",
                    reconciliation_required=True,
                )
            )
        )

        self.assert_preview_state(report, "paper_cancel_requested_unverified")
        self.assertTrue(
            report["readiness_for_stage4g3"][
                "ready_to_build_manual_state_write_proposal"
            ]
        )
        self.assertIn(
            "manual follow-up required before relying on paper_cancel_requested_unverified",
            report["warnings"],
        )

    def test_unknown_broker_status_maps_and_blocks_readiness(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(
                lifecycle_intake_candidate=candidate(
                    suggested_internal_lifecycle_state="unknown_broker_status",
                    reconciliation_required=True,
                )
            )
        )

        self.assert_preview_state(report, "paper_order_unknown_status_review")
        self.assert_blocked(
            report,
            "proposed lifecycle state requires manual review: paper_order_unknown_status_review",
        )

    def test_needs_reconciliation_blocks_readiness(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(
                lifecycle_intake_candidate=candidate(
                    suggested_internal_lifecycle_state="needs_reconciliation",
                    reconciliation_required=True,
                )
            )
        )

        self.assert_preview_state(report, "needs_reconciliation")
        self.assert_blocked(
            report,
            "proposed lifecycle state requires manual review: needs_reconciliation",
        )

    def test_unsafe_artifact_blocks_readiness(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(
                lifecycle_intake_candidate=candidate(
                    suggested_internal_lifecycle_state="unsafe_artifact",
                    reconciliation_required=True,
                )
            )
        )

        self.assert_preview_state(report, "unsafe_artifact")
        self.assert_blocked(
            report,
            "proposed lifecycle state requires manual review: unsafe_artifact",
        )

    def test_filled_candidate_with_unparseable_quantity_blocks_readiness(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(
                lifecycle_intake_candidate=candidate(
                    suggested_internal_lifecycle_state="broker_filled",
                    filled_quantity=object(),
                    quantity=object(),
                )
            )
        )

        self.assert_blocked(report, "filled candidate lacks usable position quantity")
        self.assertIsNone(report["preview"]["proposed_position_record"])

    def test_numeric_parsing_handles_common_quantity_types(self) -> None:
        for value, expected in ((1, 1.0), (1.5, 1.5), (Decimal("2.5"), 2.5), ("3.5", 3.5)):
            with self.subTest(value=value):
                report = build(
                    lifecycle_intake_report=valid_intake(
                        lifecycle_intake_candidate=candidate(
                            suggested_internal_lifecycle_state="broker_filled",
                            filled_quantity=value,
                            quantity=value,
                            remaining_quantity="0",
                        )
                    )
                )
                self.assertEqual(
                    report["preview"]["proposed_position_record"]["quantity"],
                    expected,
                )

    def test_duplicate_client_order_id_blocks_readiness(self) -> None:
        report = build(existing_state_snapshot={"existing_client_order_ids": ["intent-stage4g2-001"]})

        self.assert_blocked(report, "duplicate client_order_id exists in state snapshot")
        self.assertTrue(report["conflict_checks"]["duplicate_client_order_id"])

    def test_duplicate_broker_order_id_blocks_readiness(self) -> None:
        report = build(existing_state_snapshot={"existing_broker_order_ids": ["9001"]})

        self.assert_blocked(report, "duplicate broker_order_id exists in state snapshot")
        self.assertTrue(report["conflict_checks"]["duplicate_broker_order_id"])

    def test_duplicate_position_key_blocks_readiness_for_existing_position_key(self) -> None:
        report = build(
            existing_state_snapshot={
                "existing_position_keys": ["S01_VOL_BASELINE:XSP:intent-stage4g2-001"]
            }
        )

        self.assert_blocked(report, "duplicate position_key exists in state snapshot")
        self.assertTrue(report["conflict_checks"]["duplicate_position_key"])

    def test_open_positions_count_alone_does_not_create_duplicate_conflict(self) -> None:
        report = build(existing_state_snapshot={"open_positions_count": 4})

        self.assertTrue(
            report["readiness_for_stage4g3"][
                "ready_to_build_manual_state_write_proposal"
            ]
        )
        self.assertFalse(report["conflict_checks"]["duplicate_position_key"])

    def test_unresolved_reconciliation_in_state_snapshot_blocks_readiness(self) -> None:
        report = build(existing_state_snapshot={"unresolved_needs_reconciliation_count": 1})

        self.assert_blocked(report, "unresolved NEEDS_RECONCILIATION records exist")

    def test_active_halt_in_state_snapshot_blocks_readiness(self) -> None:
        report = build(existing_state_snapshot={"active_halt": True})

        self.assert_blocked(report, "active halt is present")

    def test_safety_flags_true_anywhere_block_readiness(self) -> None:
        cases = [
            ("live_orders_enabled", "no_live_orders"),
            ("market_data_enabled", "no_market_data"),
            ("contract_qualification_enabled", "no_contract_qualification"),
            ("scheduler_changes_enabled", "no_scheduler_changes"),
            ("lifecycle_wiring_enabled", "no_lifecycle_wiring"),
            ("state_mutation_enabled", "no_state_mutation"),
            ("ledger_writes_enabled", "no_ledger_writes"),
        ]
        for flag, check in cases:
            with self.subTest(flag=flag):
                report = build(lifecycle_intake_report=valid_intake(safety={flag: True}))
                self.assertFalse(report["safety_checks"][check])
                self.assert_blocked(
                    report,
                    f"unsafe flag enabled: {flag} in supplied report 0",
                )

    def test_failed_upstream_safety_checks_block_readiness(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(
                safety_checks={"no_market_data": False}
            )
        )

        self.assertFalse(report["safety_checks"]["no_market_data"])
        self.assert_blocked(
            report,
            "unsafe safety check failed: no_market_data in supplied report 0",
        )

    def test_write_plan_always_disables_writes_and_wiring(self) -> None:
        report = build()

        self.assertEqual(
            report["write_plan"],
            {
                "state_store_write_enabled": False,
                "ledger_write_enabled": False,
                "lifecycle_transition_enabled": False,
                "daemon_wiring_enabled": False,
                "scheduler_wiring_enabled": False,
            },
        )

    def test_proposed_ledger_events_are_preview_only_and_deterministic(self) -> None:
        now = lambda: datetime(2026, 5, 10, 15, 45, tzinfo=timezone.utc)
        first = build(now_provider=now)
        second = build(now_provider=now)

        self.assertEqual(
            first["preview"]["proposed_ledger_events"],
            second["preview"]["proposed_ledger_events"],
        )
        for event in first["preview"]["proposed_ledger_events"]:
            self.assertTrue(event["preview_only"])
            self.assertEqual(event["timestamp"], "2026-05-10T15:45:00+00:00")

    def test_input_reports_are_not_mutated(self) -> None:
        intake = valid_intake()
        snapshot = {"existing_client_order_ids": []}
        original = copy.deepcopy((intake, snapshot))

        build_stage4g2_lifecycle_state_preview(
            lifecycle_intake_report=intake,
            existing_state_snapshot=snapshot,
            now_provider=lambda: datetime(2026, 5, 10, 12, 30, tzinfo=timezone.utc),
        )

        self.assertEqual((intake, snapshot), original)

    def test_none_and_malformed_nested_dicts_do_not_crash(self) -> None:
        report = build_stage4g2_lifecycle_state_preview(
            lifecycle_intake_report={
                "stage4g1_lifecycle_intake_report": True,
                "readiness_for_stage4g2": "bad",
                "lifecycle_intake_candidate": "bad",
            },
            existing_state_snapshot="bad",  # type: ignore[arg-type]
            now_provider=lambda: datetime(2026, 5, 10, 12, 30, tzinfo=timezone.utc),
        )

        self.assertFalse(
            report["readiness_for_stage4g3"][
                "ready_to_build_manual_state_write_proposal"
            ]
        )
        assert_json_safe(self, report)

    def test_report_includes_required_top_level_fields_and_is_json_safe(self) -> None:
        report = build(
            lifecycle_intake_report=valid_intake(
                lifecycle_intake_candidate=candidate(quantity=Decimal("1.5"))
            )
        )

        for key in (
            "dry_run",
            "stage4g2_lifecycle_state_preview",
            "generated_at",
            "artifact_checks",
            "input_summary",
            "preview",
            "write_plan",
            "conflict_checks",
            "safety_checks",
            "state_snapshot_checks",
            "readiness_for_stage4g3",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        assert_json_safe(self, report)


class Stage4G2LifecycleStatePreviewCliTests(unittest.TestCase):
    def test_cli_requires_dry_run_only_before_parsing_json(self) -> None:
        calls = {"report_builder": 0}

        def report_builder(**kwargs):
            calls["report_builder"] += 1
            return {}

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_stage4g2_lifecycle_state_preview(
                ["--lifecycle-intake-json", "{bad"],
                report_builder=report_builder,
            )

        self.assertEqual(code, 1)
        self.assertEqual(calls["report_builder"], 0)
        self.assertIn("--dry-run-only", stderr.getvalue())
        self.assertNotIn("JSONDecodeError", stderr.getvalue())

    def test_cli_json_stdout_is_strict_json_only(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4g2_lifecycle_state_preview(
                [
                    "--dry-run-only",
                    "--json",
                    "--lifecycle-intake-json",
                    json.dumps(valid_intake()),
                ]
            )

        parsed = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(parsed["stage4g2_lifecycle_state_preview"])

    def test_cli_reports_json_parse_errors_with_type_and_string(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4g2_lifecycle_state_preview(
                ["--dry-run-only", "--json", "--lifecycle-intake-json", "{bad"]
            )

        parsed = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertIn("JSONDecodeError", parsed["errors"][0])

    def test_cli_passes_optional_state_snapshot_json(self) -> None:
        calls: dict[str, object] = {}

        def report_builder(**kwargs):
            calls.update(kwargs)
            return {
                "success": True,
                "readiness_for_stage4g3": {
                    "ready_to_build_manual_state_write_proposal": True
                },
            }

        with redirect_stdout(io.StringIO()):
            code = tool.run_stage4g2_lifecycle_state_preview(
                [
                    "--dry-run-only",
                    "--json",
                    "--lifecycle-intake-json",
                    "{}",
                    "--state-snapshot-json",
                    '{"active_halt": false}',
                ],
                report_builder=report_builder,
            )

        self.assertEqual(code, 0)
        self.assertEqual(calls["existing_state_snapshot"], {"active_halt": False})

    def test_cli_exposes_no_execution_or_mutation_actions(self) -> None:
        source = inspect.getsource(tool)
        forbidden = [
            "--allow-real-ibkr",
            "--allow-real-paper-submit",
            "--allow-live",
            "--live",
            "--market-data",
            "--qualify",
            "--status",
            "--cancel",
            "--submit",
            "--state-write",
            "--ledger-write",
            "--scheduler",
        ]
        for value in forbidden:
            with self.subTest(value=value):
                self.assertNotIn(value, source)


class Stage4G2LifecycleStatePreviewSafetyTests(unittest.TestCase):
    def test_stage4g2_files_do_not_call_forbidden_external_apis_or_mutators(self) -> None:
        forbidden = [
            "submit_" + "order_plan",
            "get_" + "order_status",
            "cancel_" + "order",
            "place" + "Order(",
            "cancel" + "Order(",
            "req" + "MktData",
            "qualify" + "Contracts",
            "State" + "Store(",
            ".save(",
            ".write(",
            "ledger.write",
            "ledger.append",
            "yfinance",
            "requests",
            "urllib",
            "systemctl",
            "system" + "d",
            "socket.create_connection",
            "socket.socket",
            "asyncio.run",
            "asyncio.get_event_loop",
            "asyncio.new_event_loop",
            "uuid.uuid4",
            "random",
            "time.time",
            "datetime.now",
        ]
        for path in STAGE4G2_FILES:
            source = path.read_text(encoding="utf-8")
            for value in forbidden:
                with self.subTest(path=path.name, value=value):
                    self.assertNotIn(value, source)

    def test_no_daemon_scheduler_or_lifecycle_wiring_exists(self) -> None:
        for path in UNWIRED_RUNTIME_FILES:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("stage4g2_lifecycle_state_preview", source)


if __name__ == "__main__":
    unittest.main()
