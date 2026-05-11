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

from algo_trader_unified.core.stage4g3_state_write_proposal import (
    build_stage4g3_state_write_proposal,
)
from algo_trader_unified.tools import stage4g3_state_write_proposal as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4G3_FILES = [
    ROOT / "core/stage4g3_state_write_proposal.py",
    ROOT / "tools/stage4g3_state_write_proposal.py",
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


def valid_preview(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4g2_lifecycle_state_preview": True,
        "generated_at": "2026-05-10T12:30:00+00:00",
        "preview": {
            "available": True,
            "proposed_lifecycle_state": "paper_order_submitted",
            "proposed_order_record": {
                "record_type": "paper_order",
                "preview_only": True,
                "broker_order_id": "9001",
                "client_order_id": "intent-stage4g3-001",
                "strategy_id": "S01_VOL_BASELINE",
                "symbol": "XSP",
                "action": "BUY",
                "quantity": 1,
                "filled_quantity": 0,
                "remaining_quantity": 1,
                "avg_fill_price": 0,
                "order_type": "LIMIT",
                "status": "submitted",
                "lifecycle_state": "paper_order_submitted",
            },
            "proposed_position_record": None,
            "proposed_reconciliation_flags": {
                "reconciliation_required": False,
                "reconciliation_reasons": [],
            },
            "proposed_operator_actions": [],
            "proposed_ledger_events": [
                {
                    "event_type": "paper_order_lifecycle_state_preview",
                    "preview_only": True,
                    "timestamp": "2026-05-10T12:30:00+00:00",
                    "client_order_id": "intent-stage4g3-001",
                    "broker_order_id": "9001",
                    "proposed_lifecycle_state": "paper_order_submitted",
                    "proposed_order_status": "submitted",
                }
            ],
        },
        "write_plan": {
            "state_store_write_enabled": False,
            "ledger_write_enabled": False,
            "lifecycle_transition_enabled": False,
            "daemon_wiring_enabled": False,
            "scheduler_wiring_enabled": False,
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
        "readiness_for_stage4g3": {
            "ready_to_build_manual_state_write_proposal": True,
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
        "lifecycle_state_preview_report": valid_preview(),
        "existing_state_snapshot": None,
        "operator_notes": None,
        "now_provider": lambda: datetime(2026, 5, 10, 13, 0, tzinfo=timezone.utc),
    }
    kwargs.update(overrides)
    return build_stage4g3_state_write_proposal(**kwargs)


def preview_payload(**overrides: object) -> dict[str, object]:
    report = valid_preview()
    base = copy.deepcopy(report["preview"])
    assert isinstance(base, dict)
    _deep_update(base, overrides)
    return base


def order_record(**overrides: object) -> dict[str, object]:
    preview = preview_payload()
    base = copy.deepcopy(preview["proposed_order_record"])
    assert isinstance(base, dict)
    _deep_update(base, overrides)
    return base


def position_record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "record_type": "paper_position",
        "preview_only": True,
        "position_key": "S01_VOL_BASELINE:XSP:intent-stage4g3-001",
        "broker_order_id": "9001",
        "client_order_id": "intent-stage4g3-001",
        "strategy_id": "S01_VOL_BASELINE",
        "symbol": "XSP",
        "quantity": 1,
        "avg_fill_price": Decimal("1.25"),
        "source_lifecycle_state": "paper_order_filled",
    }
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


class Stage4G3StateWriteProposalTests(unittest.TestCase):
    def assert_blocked(self, report: dict, expected: str) -> None:
        self.assertFalse(
            report["readiness_for_stage4g4"][
                "ready_to_build_manual_state_write_dry_run"
            ]
        )
        self.assertIn(expected, report["readiness_for_stage4g4"]["blockers"])

    def test_missing_lifecycle_state_preview_report_blocks_readiness(self) -> None:
        self.assert_blocked(
            build(lifecycle_state_preview_report=None),
            "lifecycle_state_preview_report missing",
        )

    def test_lifecycle_state_preview_not_ready_blocks_readiness(self) -> None:
        report = build(
            lifecycle_state_preview_report=valid_preview(
                readiness_for_stage4g3={
                    "ready_to_build_manual_state_write_proposal": False
                }
            )
        )
        self.assert_blocked(report, "Stage 4G-2 preview is not ready for Stage 4G-3")

    def test_missing_preview_available_blocks_readiness(self) -> None:
        report = build(lifecycle_state_preview_report=valid_preview(preview={"available": False}))
        self.assert_blocked(report, "Stage 4G-2 preview is not available")

    def test_missing_proposed_order_record_blocks_readiness(self) -> None:
        report = build(
            lifecycle_state_preview_report=valid_preview(
                preview={"proposed_order_record": None}
            )
        )
        self.assert_blocked(report, "proposed_order_record is required")

    def test_missing_broker_order_id_blocks_readiness(self) -> None:
        report = build(
            lifecycle_state_preview_report=valid_preview(
                preview={"proposed_order_record": order_record(broker_order_id=None)}
            )
        )
        self.assert_blocked(report, "broker_order_id is required")

    def test_missing_client_order_id_blocks_readiness(self) -> None:
        report = build(
            lifecycle_state_preview_report=valid_preview(
                preview={"proposed_order_record": order_record(client_order_id=None)}
            )
        )
        self.assert_blocked(report, "client_order_id is required")

    def test_paper_order_submitted_creates_order_operation_and_transition(self) -> None:
        report = build()

        self.assertTrue(
            report["readiness_for_stage4g4"][
                "ready_to_build_manual_state_write_dry_run"
            ]
        )
        operations = report["proposal"]["proposed_state_store_operations"]
        self.assertEqual([item["operation"] for item in operations], ["upsert_order"])
        payload = operations[0]["payload"]
        self.assertEqual(payload["client_order_id"], "intent-stage4g3-001")
        self.assertEqual(payload["broker_order_id"], "9001")
        self.assertTrue(payload["paper_only"])
        self.assertEqual(payload["source_stage"], "4G-3")
        self.assertTrue(payload["derived_from_stage4g2"])
        self.assertEqual(
            report["proposal"]["proposed_lifecycle_transition"]["transition_to"],
            "paper_order_submitted",
        )

    def test_paper_order_filled_creates_order_position_ledger_and_transition(self) -> None:
        report = build(
            lifecycle_state_preview_report=valid_preview(
                preview=preview_payload(
                    proposed_lifecycle_state="paper_order_filled",
                    proposed_order_record=order_record(
                        status="filled",
                        lifecycle_state="paper_order_filled",
                        filled_quantity=1,
                        remaining_quantity=0,
                        avg_fill_price=Decimal("1.25"),
                    ),
                    proposed_position_record=position_record(),
                    proposed_ledger_events=[
                        {
                            "event_type": "paper_order_lifecycle_state_preview",
                            "timestamp": "2026-05-10T12:30:00+00:00",
                            "client_order_id": "intent-stage4g3-001",
                            "broker_order_id": "9001",
                        },
                        {
                            "event_type": "paper_position_state_preview",
                            "timestamp": "2026-05-10T12:30:00+00:00",
                            "client_order_id": "intent-stage4g3-001",
                            "broker_order_id": "9001",
                        },
                    ],
                )
            )
        )

        operations = report["proposal"]["proposed_state_store_operations"]
        self.assertEqual(
            [item["operation"] for item in operations],
            ["upsert_order", "upsert_position"],
        )
        self.assertEqual(operations[1]["payload"]["quantity"], 1)
        self.assertEqual(operations[1]["payload"]["avg_fill_price"], 1.25)
        self.assertEqual(
            [item["event_type"] for item in report["proposal"]["proposed_ledger_events"]],
            ["paper_order_lifecycle_state_preview", "paper_position_state_preview"],
        )
        self.assertEqual(
            report["proposal"]["proposed_lifecycle_transition"]["transition_to"],
            "paper_order_filled",
        )

    def test_proposed_state_store_operations_are_structured(self) -> None:
        report = build()
        for operation in report["proposal"]["proposed_state_store_operations"]:
            self.assertIsInstance(operation["operation"], str)
            self.assertTrue(operation["operation"])
            self.assertIsInstance(operation["payload"], dict)

    def test_paper_order_cancelled_creates_order_operation_without_position(self) -> None:
        report = build(
            lifecycle_state_preview_report=valid_preview(
                preview=preview_payload(
                    proposed_lifecycle_state="paper_order_cancelled",
                    proposed_order_record=order_record(
                        status="cancelled",
                        lifecycle_state="paper_order_cancelled",
                    ),
                    proposed_position_record=None,
                )
            )
        )

        operations = report["proposal"]["proposed_state_store_operations"]
        self.assertEqual([item["operation"] for item in operations], ["upsert_order"])
        self.assertEqual(
            report["proposal"]["proposed_lifecycle_transition"]["transition_to"],
            "paper_order_cancelled",
        )

    def test_unverified_supported_states_create_warning_and_transition(self) -> None:
        for state in (
            "paper_order_submitted_unverified",
            "paper_cancel_requested_unverified",
        ):
            with self.subTest(state=state):
                report = build(
                    lifecycle_state_preview_report=valid_preview(
                        preview=preview_payload(
                            proposed_lifecycle_state=state,
                            proposed_order_record=order_record(lifecycle_state=state),
                        )
                    )
                )
                self.assertTrue(
                    report["readiness_for_stage4g4"][
                        "ready_to_build_manual_state_write_dry_run"
                    ]
                )
                self.assertIn(
                    f"manual follow-up required before writing {state}",
                    report["warnings"],
                )
                self.assertEqual(
                    report["proposal"]["proposed_lifecycle_transition"]["transition_to"],
                    state,
                )

    def test_unsupported_lifecycle_states_block_readiness(self) -> None:
        for state in (
            "paper_order_partially_filled_review",
            "paper_order_rejected_or_inactive_review",
            "paper_order_unknown_status_review",
            "needs_reconciliation",
            "unsafe_artifact",
        ):
            with self.subTest(state=state):
                report = build(
                    lifecycle_state_preview_report=valid_preview(
                        preview=preview_payload(
                            proposed_lifecycle_state=state,
                            proposed_order_record=order_record(lifecycle_state=state),
                        ),
                        readiness_for_stage4g3={
                            "ready_to_build_manual_state_write_proposal": True
                        },
                    )
                )
                self.assert_blocked(
                    report,
                    f"unsupported lifecycle state for 4G-3 proposal: {state}",
                )

    def test_missing_proposed_ledger_events_generates_minimal_event_but_blocks(self) -> None:
        report = build(
            lifecycle_state_preview_report=valid_preview(
                preview={"proposed_ledger_events": []}
            ),
            now_provider=lambda: datetime(2026, 5, 10, 14, 15, tzinfo=timezone.utc),
        )

        self.assert_blocked(
            report,
            "proposed_ledger_events missing from Stage 4G-2 preview",
        )
        events = report["proposal"]["proposed_ledger_events"]
        self.assertEqual(events[0]["event_type"], "PAPER_LIFECYCLE_STATE_PROPOSED")
        self.assertEqual(events[0]["timestamp"], "2026-05-10T14:15:00+00:00")
        self.assertEqual(events[0]["generated_at"], "2026-05-10T14:15:00+00:00")

    def test_proposed_ledger_events_ordering_is_deterministic(self) -> None:
        first = build()
        second = build()
        self.assertEqual(
            first["proposal"]["proposed_ledger_events"],
            second["proposal"]["proposed_ledger_events"],
        )

    def test_any_write_plan_flag_true_in_4g2_blocks_readiness(self) -> None:
        for key in (
            "state_store_write_enabled",
            "ledger_write_enabled",
            "lifecycle_transition_enabled",
            "daemon_wiring_enabled",
            "scheduler_wiring_enabled",
        ):
            with self.subTest(key=key):
                report = build(
                    lifecycle_state_preview_report=valid_preview(
                        write_plan={key: True}
                    )
                )
                self.assert_blocked(
                    report,
                    f"Stage 4G-2 write_plan flag must be False: {key}",
                )

    def test_4g3_write_plan_always_disables_writes_and_wiring(self) -> None:
        self.assertEqual(
            build()["write_plan"],
            {
                "state_store_write_enabled": False,
                "ledger_write_enabled": False,
                "lifecycle_transition_enabled": False,
                "daemon_wiring_enabled": False,
                "scheduler_wiring_enabled": False,
            },
        )

    def test_duplicate_client_order_id_blocks_readiness(self) -> None:
        report = build(existing_state_snapshot={"existing_client_order_ids": ["intent-stage4g3-001"]})
        self.assert_blocked(report, "duplicate client_order_id exists in state snapshot")
        self.assertTrue(report["conflict_checks"]["duplicate_client_order_id"])

    def test_duplicate_broker_order_id_blocks_readiness(self) -> None:
        report = build(existing_state_snapshot={"existing_broker_order_ids": ["9001"]})
        self.assert_blocked(report, "duplicate broker_order_id exists in state snapshot")
        self.assertTrue(report["conflict_checks"]["duplicate_broker_order_id"])

    def test_open_positions_count_alone_does_not_create_duplicate_conflict(self) -> None:
        report = build(existing_state_snapshot={"open_positions_count": 4})
        self.assertTrue(
            report["readiness_for_stage4g4"][
                "ready_to_build_manual_state_write_dry_run"
            ]
        )
        self.assertFalse(report["conflict_checks"]["duplicate_position_key"])

    def test_unresolved_reconciliation_in_state_snapshot_blocks_readiness(self) -> None:
        self.assert_blocked(
            build(existing_state_snapshot={"unresolved_needs_reconciliation_count": 1}),
            "unresolved NEEDS_RECONCILIATION records exist",
        )

    def test_active_halt_in_state_snapshot_blocks_readiness(self) -> None:
        self.assert_blocked(
            build(existing_state_snapshot={"active_halt": True}),
            "active halt is present",
        )

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
                report = build(lifecycle_state_preview_report=valid_preview(safety={flag: True}))
                self.assertFalse(report["safety_checks"][check])
                self.assert_blocked(
                    report,
                    f"unsafe flag enabled: {flag} in supplied report 0",
                )

    def test_operator_notes_cannot_bypass_hard_failures(self) -> None:
        report = build(
            lifecycle_state_preview_report=valid_preview(
                preview={"proposed_order_record": order_record(client_order_id=None)}
            ),
            operator_notes={
                "manual_observation": "reviewed",
                "cleanup_ticket": None,
                "operator_initials": "AB",
                "follow_up_required": False,
            },
        )
        self.assert_blocked(report, "client_order_id is required")
        self.assertEqual(report["operator_notes"]["operator_initials"], "AB")

    def test_input_reports_are_not_mutated(self) -> None:
        preview = valid_preview()
        snapshot = {"existing_client_order_ids": []}
        notes = {"manual_observation": "reviewed"}
        original = copy.deepcopy((preview, snapshot, notes))

        build_stage4g3_state_write_proposal(
            lifecycle_state_preview_report=preview,
            existing_state_snapshot=snapshot,
            operator_notes=notes,
            now_provider=lambda: datetime(2026, 5, 10, 13, 0, tzinfo=timezone.utc),
        )

        self.assertEqual((preview, snapshot, notes), original)

    def test_report_includes_required_top_level_fields_and_is_json_safe(self) -> None:
        report = build(
            lifecycle_state_preview_report=valid_preview(
                preview={"proposed_order_record": order_record(quantity=Decimal("1.5"))}
            ),
            operator_notes={"manual_observation": object()},
        )

        for key in (
            "dry_run",
            "stage4g3_state_write_proposal",
            "generated_at",
            "artifact_checks",
            "input_summary",
            "proposal",
            "write_plan",
            "conflict_checks",
            "safety_checks",
            "state_snapshot_checks",
            "readiness_for_stage4g4",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        assert_json_safe(self, report)


class Stage4G3StateWriteProposalCliTests(unittest.TestCase):
    def test_cli_requires_dry_run_only_before_parsing_json(self) -> None:
        calls = {"report_builder": 0}

        def report_builder(**kwargs):
            calls["report_builder"] += 1
            return {}

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_stage4g3_state_write_proposal(
                ["--lifecycle-state-preview-json", "{bad"],
                report_builder=report_builder,
            )

        self.assertEqual(code, 1)
        self.assertEqual(calls["report_builder"], 0)
        self.assertIn("--dry-run-only", stderr.getvalue())
        self.assertNotIn("JSONDecodeError", stderr.getvalue())

    def test_cli_json_stdout_is_strict_json_only(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4g3_state_write_proposal(
                [
                    "--dry-run-only",
                    "--json",
                    "--lifecycle-state-preview-json",
                    json.dumps(valid_preview()),
                ]
            )

        parsed = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(parsed["stage4g3_state_write_proposal"])

    def test_cli_reports_json_parse_errors_with_type_and_string(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4g3_state_write_proposal(
                ["--dry-run-only", "--json", "--lifecycle-state-preview-json", "{bad"]
            )

        parsed = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertIn("JSONDecodeError", parsed["errors"][0])

    def test_cli_passes_optional_state_snapshot_and_operator_notes_json(self) -> None:
        calls: dict[str, object] = {}

        def report_builder(**kwargs):
            calls.update(kwargs)
            return {
                "success": True,
                "readiness_for_stage4g4": {
                    "ready_to_build_manual_state_write_dry_run": True
                },
            }

        with redirect_stdout(io.StringIO()):
            code = tool.run_stage4g3_state_write_proposal(
                [
                    "--dry-run-only",
                    "--json",
                    "--lifecycle-state-preview-json",
                    "{}",
                    "--state-snapshot-json",
                    '{"active_halt": false}',
                    "--operator-notes-json",
                    '{"operator_initials": "AB"}',
                ],
                report_builder=report_builder,
            )

        self.assertEqual(code, 0)
        self.assertEqual(calls["existing_state_snapshot"], {"active_halt": False})
        self.assertEqual(calls["operator_notes"], {"operator_initials": "AB"})

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


class Stage4G3StateWriteProposalSafetyTests(unittest.TestCase):
    def test_stage4g3_files_do_not_call_forbidden_external_apis_or_mutators(self) -> None:
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
        for path in STAGE4G3_FILES:
            source = path.read_text(encoding="utf-8")
            for value in forbidden:
                with self.subTest(path=path.name, value=value):
                    self.assertNotIn(value, source)

    def test_no_daemon_scheduler_or_lifecycle_wiring_exists(self) -> None:
        for path in UNWIRED_RUNTIME_FILES:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("stage4g3_state_write_proposal", source)


if __name__ == "__main__":
    unittest.main()
