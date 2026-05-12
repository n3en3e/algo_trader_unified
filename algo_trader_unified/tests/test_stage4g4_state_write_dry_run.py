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

from algo_trader_unified.core.stage4g4_state_write_dry_run import (
    REQUIRED_ACKNOWLEDGEMENTS,
    build_stage4g4_state_write_dry_run,
)
from algo_trader_unified.tools import stage4g4_state_write_dry_run as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4G4_FILES = [
    ROOT / "core/stage4g4_state_write_dry_run.py",
    ROOT / "tools/stage4g4_state_write_dry_run.py",
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


def valid_proposal(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4g3_state_write_proposal": True,
        "generated_at": "2026-05-10T13:00:00+00:00",
        "proposal": {
            "available": True,
            "proposed_state_store_operations": [
                {
                    "operation": "upsert_order",
                    "payload": {
                        "client_order_id": "intent-stage4g4-001",
                        "broker_order_id": "9001",
                        "strategy_id": "S01_VOL_BASELINE",
                        "symbol": "XSP",
                        "action": "BUY",
                        "quantity": 1,
                        "order_type": "LIMIT",
                        "status": "submitted",
                        "paper_only": True,
                    },
                },
                {
                    "operation": "upsert_position",
                    "payload": {
                        "position_key": "S01_VOL_BASELINE:XSP:intent-stage4g4-001",
                        "client_order_id": "intent-stage4g4-001",
                        "broker_order_id": "9001",
                        "symbol": "XSP",
                        "quantity": 1,
                        "avg_fill_price": Decimal("1.25"),
                        "paper_only": True,
                    },
                },
            ],
            "proposed_ledger_events": [
                {
                    "event_type": "paper_order_lifecycle_state_preview",
                    "timestamp": "2026-05-10T12:30:00+00:00",
                    "client_order_id": "intent-stage4g4-001",
                    "broker_order_id": "9001",
                },
                {
                    "event_type": "paper_position_state_preview",
                    "timestamp": "2026-05-10T12:31:00+00:00",
                    "client_order_id": "intent-stage4g4-001",
                    "broker_order_id": "9001",
                },
            ],
            "proposed_lifecycle_transition": {
                "transition_to": "paper_order_filled",
                "proposal_only": True,
                "enabled": False,
            },
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
        "readiness_for_stage4g4": {
            "ready_to_build_manual_state_write_dry_run": True,
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
        "state_write_proposal_report": valid_proposal(),
        "existing_state_snapshot": None,
        "operator_acknowledgements": list(REQUIRED_ACKNOWLEDGEMENTS),
        "now_provider": lambda: datetime(2026, 5, 10, 14, 0, tzinfo=timezone.utc),
    }
    kwargs.update(overrides)
    return build_stage4g4_state_write_dry_run(**kwargs)


def proposal_payload(**overrides: object) -> dict[str, object]:
    report = valid_proposal()
    base = copy.deepcopy(report["proposal"])
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


class Stage4G4StateWriteDryRunTests(unittest.TestCase):
    def assert_blocked(self, report: dict, expected: str) -> None:
        self.assertFalse(
            report["readiness_for_stage4g5"][
                "ready_to_build_manual_state_write_executor"
            ]
        )
        self.assertIn(expected, report["readiness_for_stage4g5"]["blockers"])

    def test_missing_state_write_proposal_report_blocks_readiness(self) -> None:
        self.assert_blocked(
            build(state_write_proposal_report=None),
            "state_write_proposal_report missing",
        )

    def test_state_write_proposal_not_ready_blocks_readiness(self) -> None:
        report = build(
            state_write_proposal_report=valid_proposal(
                readiness_for_stage4g4={
                    "ready_to_build_manual_state_write_dry_run": False
                }
            )
        )
        self.assert_blocked(report, "Stage 4G-3 proposal is not ready for Stage 4G-4")

    def test_missing_proposal_available_blocks_readiness(self) -> None:
        report = build(state_write_proposal_report=valid_proposal(proposal={"available": False}))
        self.assert_blocked(report, "Stage 4G-3 proposal is not available")

    def test_missing_proposed_state_store_operations_blocks_readiness(self) -> None:
        report = build(
            state_write_proposal_report=valid_proposal(
                proposal={"proposed_state_store_operations": []}
            )
        )
        self.assert_blocked(report, "proposed_state_store_operations are required")

    def test_missing_proposed_ledger_events_blocks_readiness(self) -> None:
        report = build(
            state_write_proposal_report=valid_proposal(
                proposal={"proposed_ledger_events": []}
            )
        )
        self.assert_blocked(report, "proposed_ledger_events are required")

    def test_missing_proposed_lifecycle_transition_blocks_readiness(self) -> None:
        report = build(
            state_write_proposal_report=valid_proposal(
                proposal={"proposed_lifecycle_transition": None}
            )
        )
        self.assert_blocked(report, "proposed_lifecycle_transition is required")

    def test_malformed_proposed_state_store_operations_blocks_readiness(self) -> None:
        report = build(
            state_write_proposal_report=valid_proposal(
                proposal={"proposed_state_store_operations": ["bad"]}
            )
        )
        self.assert_blocked(
            report,
            "proposed_state_store_operations must be structured",
        )

    def test_unrecognized_operation_name_blocks_readiness(self) -> None:
        report = build(
            state_write_proposal_report=valid_proposal(
                proposal={
                    "proposed_state_store_operations": [
                        {"operation": "delete_order", "payload": {}}
                    ]
                }
            )
        )
        self.assert_blocked(report, "unrecognized proposed StateStore operation")

    def test_proposed_state_store_operations_require_operation_and_payload(self) -> None:
        report = build(
            state_write_proposal_report=valid_proposal(
                proposal={
                    "proposed_state_store_operations": [
                        {"payload": {}},
                        {"operation": "upsert_order"},
                    ]
                }
            )
        )
        blockers = report["readiness_for_stage4g5"]["blockers"]
        self.assertIn("proposed StateStore operation missing operation at index 0", blockers)
        self.assertIn("proposed StateStore operation missing payload at index 1", blockers)

    def test_operation_payload_must_be_json_safe_dict(self) -> None:
        report = build(
            state_write_proposal_report=valid_proposal(
                proposal={
                    "proposed_state_store_operations": [
                        {"operation": "upsert_order", "payload": "bad"}
                    ]
                }
            )
        )
        self.assert_blocked(
            report,
            "proposed StateStore operation payload must be a JSON-safe dict at index 0",
        )

    def test_dry_run_operations_are_structured_and_never_execute(self) -> None:
        report = build()
        operations = report["dry_run_packet"]["dry_run_operations"]
        self.assertTrue(
            report["readiness_for_stage4g5"][
                "ready_to_build_manual_state_write_executor"
            ]
        )
        for operation in operations:
            self.assertIn("sequence_number", operation)
            self.assertIn("operation_type", operation)
            self.assertIn("operation", operation)
            self.assertIn("payload", operation)
            self.assertIn("target", operation)
            self.assertIs(operation["would_execute"], False)

    def test_dry_run_operation_order_is_state_order_position_ledger_lifecycle(self) -> None:
        report = build()
        operations = report["dry_run_packet"]["dry_run_operations"]

        self.assertEqual(
            [(item["target"], item["operation"]) for item in operations],
            [
                ("StateStore", "upsert_order"),
                ("StateStore", "upsert_position"),
                ("Ledger", "append_ledger_event"),
                ("Ledger", "append_ledger_event"),
                ("Lifecycle", "record_lifecycle_transition"),
            ],
        )
        self.assertEqual(
            [item["sequence_number"] for item in operations],
            [1, 2, 3, 4, 5],
        )

    def test_bad_state_store_operation_order_blocks_readiness(self) -> None:
        proposal = proposal_payload()
        operations = proposal["proposed_state_store_operations"]
        assert isinstance(operations, list)
        proposal["proposed_state_store_operations"] = [operations[1], operations[0]]
        report = build(state_write_proposal_report=valid_proposal(proposal=proposal))
        self.assert_blocked(
            report,
            "proposed StateStore operations are not deterministically ordered",
        )

    def test_rollback_simulation_is_static_descriptive_and_not_executable(self) -> None:
        rollback = build()["dry_run_packet"]["rollback_simulation"]

        self.assertTrue(rollback["available"])
        self.assertEqual(
            rollback["rollback_steps"],
            [
                "Rollback requires manual StateStore/ledger file reversion using standard system backups."
            ],
        )
        self.assertEqual(
            rollback["rollback_limitations"],
            ["no automated rollback is supported in this phase"],
        )
        serialized = json.dumps(rollback, sort_keys=True)
        for forbidden in ("python", "sql", "bash", "rm ", "mv ", "cp ", "UPDATE "):
            self.assertNotIn(forbidden, serialized)

    def test_missing_acknowledgements_block_readiness_but_report_renders(self) -> None:
        report = build(operator_acknowledgements=[])

        self.assertTrue(report["success"])
        self.assertTrue(report["dry_run_packet"]["available"])
        self.assert_blocked(report, "required operator acknowledgements missing")
        self.assertEqual(
            report["acknowledgement_checks"]["missing"],
            list(REQUIRED_ACKNOWLEDGEMENTS),
        )

    def test_giant_substring_acknowledgement_does_not_pass(self) -> None:
        report = build(
            operator_acknowledgements=[" ".join(REQUIRED_ACKNOWLEDGEMENTS)]
        )
        self.assert_blocked(report, "required operator acknowledgements missing")

    def test_extra_acknowledgements_do_not_compensate_for_missing_required_ack(self) -> None:
        report = build(
            operator_acknowledgements=list(REQUIRED_ACKNOWLEDGEMENTS[:-1])
            + ["I reviewed something else."]
        )
        self.assert_blocked(report, "required operator acknowledgements missing")
        self.assertEqual(
            report["acknowledgement_checks"]["missing"],
            [REQUIRED_ACKNOWLEDGEMENTS[-1]],
        )

    def test_all_exact_acknowledgements_pass_with_optional_strip(self) -> None:
        report = build(
            operator_acknowledgements=[
                f" {value} " for value in REQUIRED_ACKNOWLEDGEMENTS
            ]
        )
        self.assertTrue(report["acknowledgement_checks"]["exact_match"])
        self.assertTrue(
            report["readiness_for_stage4g5"][
                "ready_to_build_manual_state_write_executor"
            ]
        )

    def test_any_input_write_plan_flag_true_blocks_readiness(self) -> None:
        for key in (
            "state_store_write_enabled",
            "ledger_write_enabled",
            "lifecycle_transition_enabled",
            "daemon_wiring_enabled",
            "scheduler_wiring_enabled",
        ):
            with self.subTest(key=key):
                report = build(
                    state_write_proposal_report=valid_proposal(
                        write_plan={key: True}
                    )
                )
                self.assert_blocked(
                    report,
                    f"Stage 4G-3 write_plan flag must be False: {key}",
                )

    def test_output_write_plan_always_disables_all_write_flags(self) -> None:
        report = build(state_write_proposal_report=valid_proposal(write_plan={}))
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

    def test_generated_timestamp_comes_only_from_now_provider(self) -> None:
        report = build(
            now_provider=lambda: datetime(2026, 5, 10, 15, 45, tzinfo=timezone.utc)
        )
        self.assertEqual(report["generated_at"], "2026-05-10T15:45:00+00:00")

    def test_duplicate_client_order_id_blocks_readiness(self) -> None:
        report = build(
            existing_state_snapshot={
                "existing_client_order_ids": ["intent-stage4g4-001"]
            }
        )
        self.assert_blocked(
            report,
            "duplicate client_order_id exists in state snapshot",
        )

    def test_duplicate_broker_order_id_blocks_readiness(self) -> None:
        report = build(existing_state_snapshot={"existing_broker_order_ids": ["9001"]})
        self.assert_blocked(
            report,
            "duplicate broker_order_id exists in state snapshot",
        )

    def test_duplicate_position_key_blocks_readiness(self) -> None:
        report = build(
            existing_state_snapshot={
                "existing_position_keys": ["S01_VOL_BASELINE:XSP:intent-stage4g4-001"]
            }
        )
        self.assert_blocked(
            report,
            "duplicate position_key exists in state snapshot",
        )

    def test_open_positions_count_alone_does_not_create_duplicate_conflict(self) -> None:
        report = build(existing_state_snapshot={"open_positions_count": 7})
        self.assertFalse(report["conflict_checks"]["duplicate_position_key"])
        self.assertTrue(
            report["readiness_for_stage4g5"][
                "ready_to_build_manual_state_write_executor"
            ]
        )

    def test_unresolved_needs_reconciliation_blocks_readiness(self) -> None:
        report = build(
            existing_state_snapshot={
                "positions": {"p1": {"status": "NEEDS_RECONCILIATION"}}
            }
        )
        self.assert_blocked(report, "unresolved NEEDS_RECONCILIATION records exist")

    def test_active_halt_blocks_readiness(self) -> None:
        report = build(existing_state_snapshot={"active_halt": True})
        self.assert_blocked(report, "active halt is present")

    def test_safety_flags_anywhere_block_readiness(self) -> None:
        flag_cases = {
            "live_orders_enabled": "no_live_orders",
            "market_data_enabled": "no_market_data",
            "contract_qualification_enabled": "no_contract_qualification",
            "scheduler_wiring_enabled": "no_scheduler_changes",
            "lifecycle_wiring_enabled": "no_lifecycle_wiring",
        }
        for flag, check in flag_cases.items():
            with self.subTest(flag=flag):
                report = build(
                    state_write_proposal_report=valid_proposal(
                        nested={"deeper": {flag: True}}
                    )
                )
                self.assertFalse(report["safety_checks"][check])
                self.assertFalse(
                    report["readiness_for_stage4g5"][
                        "ready_to_build_manual_state_write_executor"
                    ]
                )

    def test_input_reports_are_not_mutated(self) -> None:
        proposal = valid_proposal()
        snapshot = {"existing_client_order_ids": []}
        original_proposal = copy.deepcopy(proposal)
        original_snapshot = copy.deepcopy(snapshot)

        build(
            state_write_proposal_report=proposal,
            existing_state_snapshot=snapshot,
        )

        self.assertEqual(proposal, original_proposal)
        self.assertEqual(snapshot, original_snapshot)

    def test_report_includes_required_top_level_fields_and_is_json_safe(self) -> None:
        report = build()
        for key in (
            "dry_run",
            "stage4g4_state_write_dry_run",
            "generated_at",
            "artifact_checks",
            "operation_schema_checks",
            "acknowledgement_checks",
            "dry_run_packet",
            "write_plan",
            "conflict_checks",
            "safety_checks",
            "state_snapshot_checks",
            "readiness_for_stage4g5",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        assert_json_safe(self, report)


class Stage4G4StateWriteDryRunCliTests(unittest.TestCase):
    def test_cli_requires_dry_run_only_before_loading_json(self) -> None:
        calls = {"report_builder": 0}

        def report_builder(**kwargs):
            calls["report_builder"] += 1
            return {}

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_stage4g4_state_write_dry_run(
                ["--state-write-proposal-json", "{bad"],
                report_builder=report_builder,
            )

        self.assertEqual(code, 1)
        self.assertEqual(calls["report_builder"], 0)
        self.assertIn("--dry-run-only", stderr.getvalue())
        self.assertNotIn("JSONDecodeError", stderr.getvalue())

    def test_cli_requires_dry_run_only_before_argparse_required_json(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_stage4g4_state_write_dry_run([])

        self.assertEqual(code, 1)
        self.assertIn("--dry-run-only", stderr.getvalue())
        self.assertNotIn("--state-write-proposal-json", stderr.getvalue())

    def test_cli_ack_uses_argparse_append(self) -> None:
        source = inspect.getsource(tool.run_stage4g4_state_write_dry_run)
        self.assertIn('parser.add_argument("--ack", action="append"', source)

    def test_cli_json_stdout_is_strict_json_only(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4g4_state_write_dry_run(
                [
                    "--dry-run-only",
                    "--json",
                    "--state-write-proposal-json",
                    json.dumps(valid_proposal(), default=str),
                    *[
                        part
                        for ack in REQUIRED_ACKNOWLEDGEMENTS
                        for part in ("--ack", ack)
                    ],
                ]
            )

        parsed = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(parsed["stage4g4_state_write_dry_run"])

    def test_cli_reports_json_parse_errors_with_type_and_string(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4g4_state_write_dry_run(
                ["--dry-run-only", "--json", "--state-write-proposal-json", "{bad"]
            )

        parsed = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertIn("JSONDecodeError", parsed["errors"][0])

    def test_cli_passes_optional_state_snapshot_and_acknowledgements(self) -> None:
        calls: dict[str, object] = {}

        def report_builder(**kwargs):
            calls.update(kwargs)
            return {
                "success": True,
                "readiness_for_stage4g5": {
                    "ready_to_build_manual_state_write_executor": True
                },
            }

        with redirect_stdout(io.StringIO()):
            code = tool.run_stage4g4_state_write_dry_run(
                [
                    "--dry-run-only",
                    "--json",
                    "--state-write-proposal-json",
                    "{}",
                    "--state-snapshot-json",
                    '{"active_halt": false}',
                    "--ack",
                    REQUIRED_ACKNOWLEDGEMENTS[0],
                ],
                report_builder=report_builder,
            )

        self.assertEqual(code, 0)
        self.assertEqual(calls["existing_state_snapshot"], {"active_halt": False})
        self.assertEqual(
            calls["operator_acknowledgements"],
            [REQUIRED_ACKNOWLEDGEMENTS[0]],
        )

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
            "--execute",
            "--ledger-write",
            "--scheduler",
        ]
        for value in forbidden:
            with self.subTest(value=value):
                self.assertNotIn(value, source)


class Stage4G4StateWriteDryRunSafetyTests(unittest.TestCase):
    def test_stage4g4_files_do_not_call_forbidden_external_apis_or_mutators(self) -> None:
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
        for path in STAGE4G4_FILES:
            source = path.read_text(encoding="utf-8")
            for value in forbidden:
                with self.subTest(path=path.name, value=value):
                    self.assertNotIn(value, source)

    def test_no_daemon_scheduler_or_lifecycle_wiring_exists(self) -> None:
        for path in UNWIRED_RUNTIME_FILES:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("stage4g4_state_write_dry_run", source)


if __name__ == "__main__":
    unittest.main()
