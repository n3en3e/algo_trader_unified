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

from algo_trader_unified.core.stage4g6_lifecycle_write_acceptance import (
    build_stage4g6_lifecycle_write_acceptance_report,
)
from algo_trader_unified.tools import stage4g6_lifecycle_write_acceptance as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4G6_FILES = [
    ROOT / "core/stage4g6_lifecycle_write_acceptance.py",
    ROOT / "tools/stage4g6_lifecycle_write_acceptance.py",
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


def valid_executor_report(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "dry_run": False,
        "stage4g5_state_write_executor": True,
        "generated_at": "2026-05-10T15:00:00+00:00",
        "gates": {"passed": True, "reasons": []},
        "execution": {
            "attempted": True,
            "state_store_write_attempted": True,
            "ledger_write_attempted": True,
            "state_store_write_succeeded": True,
            "ledger_write_succeeded": True,
            "lifecycle_transition_executed": False,
            "completed": True,
            "failed_step": None,
            "failure_reason": None,
        },
        "applied_operations": [
            {
                "sequence_number": 1,
                "target": "StateStore",
                "operation": "upsert_order",
                "client_order_id": "intent-stage4g6-001",
                "broker_order_id": "9001",
                "result": {"status": "ok"},
            },
            {
                "sequence_number": 2,
                "target": "StateStore",
                "operation": "upsert_position",
                "client_order_id": "intent-stage4g6-001",
                "broker_order_id": "9001",
                "result": {"status": "ok"},
            },
            {
                "sequence_number": 3,
                "target": "Ledger",
                "operation": "append_event",
                "client_order_id": "intent-stage4g6-001",
                "broker_order_id": "9001",
                "result": {"status": "ok"},
            },
        ],
        "skipped_operations": [],
        "rollback": {
            "rollback_required": False,
            "rollback_attempted": False,
        },
        "write_plan": {
            "state_store_write_enabled": True,
            "ledger_write_enabled": True,
            "lifecycle_transition_enabled": False,
            "daemon_wiring_enabled": False,
            "scheduler_wiring_enabled": False,
        },
        "safety": {
            "paper_state_write_enabled": True,
            "paper_ledger_write_enabled": True,
            "live_orders_enabled": False,
            "market_data_enabled": False,
            "contract_qualification_enabled": False,
            "scheduler_changes_enabled": False,
            "lifecycle_wiring_enabled": False,
            "automated_paper_trading_enabled": False,
        },
        "readiness_for_stage4g6": {
            "ready_to_build_manual_lifecycle_write_acceptance_report": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def matching_state_snapshot() -> dict[str, object]:
    return {
        "order_records": [
            {
                "client_order_id": "intent-stage4g6-001",
                "broker_order_id": "9001",
                "symbol": "XSP",
            }
        ],
        "position_records": [
            {
                "client_order_id": "intent-stage4g6-001",
                "broker_order_id": "9001",
                "symbol": "XSP",
            }
        ],
        "unresolved_needs_reconciliation_count": 0,
        "active_halt": False,
    }


def matching_ledger_snapshot() -> dict[str, object]:
    return {
        "events": [
            {
                "event_type": "paper_order_lifecycle_state_write",
                "client_order_id": "intent-stage4g6-001",
                "broker_order_id": "9001",
            }
        ],
        "event_types": ["paper_order_lifecycle_state_write"],
    }


def build(
    *,
    executor_report: dict | None = None,
    state_snapshot: dict | None = None,
    ledger_snapshot: dict | None = None,
) -> dict:
    return build_stage4g6_lifecycle_write_acceptance_report(
        state_write_executor_report=valid_executor_report()
        if executor_report is None
        else executor_report,
        existing_state_snapshot=state_snapshot,
        ledger_snapshot=ledger_snapshot,
        now_provider=lambda: datetime(2026, 5, 10, 16, 0, tzinfo=timezone.utc),
    )


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


def assert_not_ready(test_case: unittest.TestCase, report: dict) -> None:
    test_case.assertFalse(
        report["readiness_for_stage4h"][
            "ready_to_begin_controlled_automated_paper_trading_launch"
        ]
    )
    test_case.assertFalse(report["success"])


def assert_json_safe(test_case: unittest.TestCase, report: dict) -> None:
    serialized = json.dumps(report, sort_keys=True)
    test_case.assertNotIn("datetime", serialized)
    test_case.assertNotIn("Decimal", serialized)


class Stage4G6LifecycleWriteAcceptanceTests(unittest.TestCase):
    def test_missing_executor_report_blocks_readiness(self) -> None:
        report = build_stage4g6_lifecycle_write_acceptance_report(
            state_write_executor_report=None
        )
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["executor_report_present"])

    def test_malformed_executor_report_blocks_readiness(self) -> None:
        report = build_stage4g6_lifecycle_write_acceptance_report(
            state_write_executor_report="bad"  # type: ignore[arg-type]
        )
        assert_not_ready(self, report)
        self.assertIn("state_write_executor_report must be a dict", report["errors"])

    def test_executor_report_not_ready_blocks_readiness(self) -> None:
        report = build(
            executor_report=valid_executor_report(
                readiness_for_stage4g6={
                    "ready_to_build_manual_lifecycle_write_acceptance_report": False
                }
            )
        )
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["executor_report_ready"])

    def test_execution_completed_false_blocks_readiness(self) -> None:
        report = build(executor_report=valid_executor_report(execution={"completed": False}))
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["executor_completed"])

    def test_state_store_write_succeeded_false_blocks_readiness(self) -> None:
        report = build(
            executor_report=valid_executor_report(
                execution={"state_store_write_succeeded": False}
            )
        )
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["state_store_write_succeeded"])

    def test_ledger_write_succeeded_false_blocks_readiness(self) -> None:
        report = build(
            executor_report=valid_executor_report(execution={"ledger_write_succeeded": False})
        )
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["ledger_write_succeeded"])

    def test_rollback_required_blocks_readiness(self) -> None:
        report = build(
            executor_report=valid_executor_report(
                rollback={"rollback_required": True, "rollback_attempted": False}
            )
        )
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["rollback_not_required"])

    def test_skipped_operations_non_empty_blocks_readiness(self) -> None:
        report = build(
            executor_report=valid_executor_report(
                skipped_operations=[{"sequence_number": 4, "skip_reason": "failed"}]
            )
        )
        assert_not_ready(self, report)
        self.assertFalse(report["operation_checks"]["skipped_operations_empty"])

    def test_lifecycle_transition_executed_blocks_readiness(self) -> None:
        report = build(
            executor_report=valid_executor_report(
                execution={"lifecycle_transition_executed": True}
            )
        )
        assert_not_ready(self, report)
        self.assertFalse(report["operation_checks"]["lifecycle_transition_not_executed"])

    def test_applied_operation_with_lifecycle_target_blocks_readiness(self) -> None:
        executor = valid_executor_report()
        operations = executor["applied_operations"]  # type: ignore[index]
        operations.append(  # type: ignore[union-attr]
            {
                "sequence_number": 4,
                "target": "Lifecycle",
                "operation": "record_lifecycle_transition",
                "client_order_id": "intent-stage4g6-001",
                "broker_order_id": "9001",
                "result": {"status": "ok"},
            }
        )
        report = build(executor_report=executor)
        assert_not_ready(self, report)
        self.assertFalse(report["operation_checks"]["lifecycle_transition_not_executed"])

    def test_missing_applied_operations_blocks_readiness(self) -> None:
        report = build(executor_report=valid_executor_report(applied_operations=[]))
        assert_not_ready(self, report)
        self.assertFalse(report["operation_checks"]["applied_operations_present"])

    def test_missing_state_store_operation_blocks_readiness(self) -> None:
        executor = valid_executor_report()
        executor["applied_operations"] = [
            executor["applied_operations"][2]  # type: ignore[index]
        ]
        report = build(executor_report=executor)
        assert_not_ready(self, report)
        self.assertFalse(report["operation_checks"]["state_store_operation_present"])

    def test_missing_ledger_operation_blocks_readiness(self) -> None:
        executor = valid_executor_report()
        executor["applied_operations"] = executor["applied_operations"][:2]  # type: ignore[index]
        report = build(executor_report=executor)
        assert_not_ready(self, report)
        self.assertFalse(report["operation_checks"]["ledger_operation_present"])

    def test_malformed_operation_schema_blocks_readiness(self) -> None:
        executor = valid_executor_report()
        operation = copy.deepcopy(executor["applied_operations"][0])  # type: ignore[index]
        del operation["result"]
        executor["applied_operations"] = [operation]
        report = build(executor_report=executor)
        assert_not_ready(self, report)
        self.assertFalse(report["operation_checks"]["applied_operation_schema_valid"])

    def test_non_dict_applied_operations_block_without_crashing(self) -> None:
        executor = valid_executor_report(applied_operations=[None, "bad", 7, []])
        report = build(executor_report=executor)
        assert_not_ready(self, report)
        self.assertFalse(report["operation_checks"]["applied_operation_schema_valid"])
        self.assertIn("applied_operations[0] must be a dict", report["readiness_for_stage4h"]["blockers"])

    def test_applied_operation_missing_expected_keys_blocks_without_crashing(self) -> None:
        executor = valid_executor_report(
            applied_operations=[{"target": "StateStore", "operation": "upsert_order"}]
        )
        report = build(executor_report=executor)
        assert_not_ready(self, report)
        self.assertFalse(report["operation_checks"]["applied_operation_schema_valid"])

    def test_invalid_operation_order_blocks_readiness(self) -> None:
        executor = valid_executor_report()
        operations = executor["applied_operations"]  # type: ignore[index]
        executor["applied_operations"] = [operations[2], operations[0], operations[1]]  # type: ignore[index]
        report = build(executor_report=executor)
        assert_not_ready(self, report)
        self.assertFalse(report["operation_checks"]["deterministic_operation_order"])

    def test_executor_errors_block_readiness(self) -> None:
        report = build(executor_report=valid_executor_report(errors=["writer failed"]))
        assert_not_ready(self, report)
        self.assertIn("writer failed", report["errors"])

    def test_safety_flags_block_readiness(self) -> None:
        cases = [
            ("live_orders_enabled", "no_live_orders"),
            ("market_data_enabled", "no_market_data"),
            ("contract_qualification_enabled", "no_contract_qualification"),
            ("scheduler_changes_enabled", "no_scheduler_changes"),
            ("lifecycle_wiring_enabled", "no_lifecycle_wiring"),
            ("automated_paper_trading_enabled", "no_automated_paper_trading"),
        ]
        for flag, check in cases:
            with self.subTest(flag=flag):
                report = build(executor_report=valid_executor_report(safety={flag: True}))
                assert_not_ready(self, report)
                self.assertFalse(report["safety_checks"][check])

    def test_scheduler_and_lifecycle_write_plan_flags_block_readiness(self) -> None:
        for flag, check in (
            ("daemon_wiring_enabled", "no_scheduler_changes"),
            ("scheduler_wiring_enabled", "no_scheduler_changes"),
            ("lifecycle_transition_enabled", "no_lifecycle_wiring"),
        ):
            with self.subTest(flag=flag):
                report = build(executor_report=valid_executor_report(write_plan={flag: True}))
                assert_not_ready(self, report)
                self.assertFalse(report["safety_checks"][check])

    def test_id_mismatch_across_applied_operations_blocks_readiness(self) -> None:
        executor = valid_executor_report()
        executor["applied_operations"][1]["client_order_id"] = "other"  # type: ignore[index]
        report = build(executor_report=executor)
        assert_not_ready(self, report)
        self.assertFalse(report["operation_checks"]["applied_operation_ids_consistent"])

    def test_every_applied_operation_shares_same_client_and_broker_ids(self) -> None:
        report = build()
        self.assertTrue(report["operation_checks"]["applied_operation_ids_consistent"])
        self.assertEqual("intent-stage4g6-001", report["consistency_checks"]["client_order_id"])
        self.assertEqual("9001", report["consistency_checks"]["broker_order_id"])

    def test_explicit_state_snapshot_with_matching_ids_passes(self) -> None:
        report = build(state_snapshot=matching_state_snapshot())
        self.assertTrue(report["post_write_snapshot_checks"]["state_order_seen"])
        self.assertTrue(report["post_write_snapshot_checks"]["state_position_seen"])

    def test_explicit_state_snapshot_with_mismatched_ids_blocks_readiness(self) -> None:
        snapshot = matching_state_snapshot()
        snapshot["order_records"][0]["broker_order_id"] = "other"  # type: ignore[index]
        report = build(state_snapshot=snapshot)
        assert_not_ready(self, report)
        self.assertFalse(report["consistency_checks"]["state_snapshot_matches_when_provided"])

    def test_open_positions_count_alone_does_not_create_unknown_exposure_blocker(self) -> None:
        report = build(state_snapshot={"open_positions_count": 99})
        self.assertTrue(report["consistency_checks"]["state_snapshot_matches_when_provided"])
        self.assertNotIn(
            "unknown exposure",
            " ".join(report["readiness_for_stage4h"]["blockers"]).lower(),
        )

    def test_explicit_ledger_snapshot_with_matching_ids_passes(self) -> None:
        report = build(ledger_snapshot=matching_ledger_snapshot())
        self.assertTrue(report["post_write_snapshot_checks"]["ledger_event_seen"])

    def test_explicit_ledger_snapshot_with_mismatched_ids_blocks_readiness(self) -> None:
        snapshot = matching_ledger_snapshot()
        snapshot["events"][0]["client_order_id"] = "other"  # type: ignore[index]
        report = build(ledger_snapshot=snapshot)
        assert_not_ready(self, report)
        self.assertFalse(report["consistency_checks"]["ledger_snapshot_matches_when_provided"])

    def test_ledger_event_type_mismatch_blocks_readiness(self) -> None:
        report = build(ledger_snapshot={"event_types": ["daily_digest"]})
        assert_not_ready(self, report)

    def test_missing_snapshots_warn_but_do_not_crash(self) -> None:
        report = build()
        self.assertTrue(
            report["readiness_for_stage4h"][
                "ready_to_begin_controlled_automated_paper_trading_launch"
            ]
        )
        self.assertIn("state snapshot missing", " ".join(report["warnings"]))
        self.assertIn("ledger snapshot missing", " ".join(report["warnings"]))

    def test_unresolved_needs_reconciliation_blocks_readiness(self) -> None:
        snapshot = matching_state_snapshot()
        snapshot["unresolved_needs_reconciliation_count"] = 1
        report = build(state_snapshot=snapshot)
        assert_not_ready(self, report)

    def test_active_halt_blocks_readiness(self) -> None:
        snapshot = matching_state_snapshot()
        snapshot["active_halt"] = True
        report = build(state_snapshot=snapshot)
        assert_not_ready(self, report)

    def test_valid_clean_executor_with_matching_snapshots_is_ready(self) -> None:
        report = build(
            state_snapshot=matching_state_snapshot(),
            ledger_snapshot=matching_ledger_snapshot(),
        )
        self.assertTrue(
            report["readiness_for_stage4h"][
                "ready_to_begin_controlled_automated_paper_trading_launch"
            ]
        )
        self.assertTrue(report["success"])

    def test_recommendations_do_not_include_live_or_all_strategy_enablement(self) -> None:
        report = build()
        text = json.dumps(report["recommendations"]).lower()
        self.assertIn("do not enable live trading", text)
        self.assertIn("do not enable all strategies at once", text)
        self.assertNotIn("enable live trading", report["recommendations"]["ordered_next_steps"])

    def test_input_reports_are_not_mutated(self) -> None:
        executor = valid_executor_report()
        state = matching_state_snapshot()
        ledger = matching_ledger_snapshot()
        before = (copy.deepcopy(executor), copy.deepcopy(state), copy.deepcopy(ledger))
        build(executor_report=executor, state_snapshot=state, ledger_snapshot=ledger)
        self.assertEqual(before, (executor, state, ledger))

    def test_report_includes_required_top_level_fields_and_is_json_safe(self) -> None:
        report = build()
        for key in (
            "dry_run",
            "stage4g6_lifecycle_write_acceptance_report",
            "generated_at",
            "artifact_checks",
            "operation_checks",
            "consistency_checks",
            "post_write_snapshot_checks",
            "safety_checks",
            "readiness_for_stage4h",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        assert_json_safe(self, report)

    def test_writer_decimal_payloads_are_json_safe(self) -> None:
        executor = valid_executor_report()
        executor["applied_operations"][0]["result"] = {"price": Decimal("1.25")}  # type: ignore[index]
        report = build(executor_report=executor)
        assert_json_safe(self, report)

    def test_cli_requires_dry_run_only_before_parsing(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_stage4g6_lifecycle_write_acceptance([])
        self.assertEqual(1, code)
        self.assertIn("--dry-run-only", stderr.getvalue())

    def test_cli_json_stdout_is_strict_json(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4g6_lifecycle_write_acceptance(
                [
                    "--dry-run-only",
                    "--json",
                    "--state-write-executor-json",
                    json.dumps(valid_executor_report()),
                    "--state-snapshot-json",
                    json.dumps(matching_state_snapshot()),
                    "--ledger-snapshot-json",
                    json.dumps(matching_ledger_snapshot()),
                ]
            )
        self.assertEqual(0, code)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4g6_lifecycle_write_acceptance_report"])

    def test_cli_json_parse_errors_include_exception_type(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4g6_lifecycle_write_acceptance(
                [
                    "--dry-run-only",
                    "--json",
                    "--state-write-executor-json",
                    "{bad",
                ]
            )
        self.assertEqual(1, code)
        parsed = json.loads(stdout.getvalue())
        self.assertIn("JSONDecodeError", parsed["errors"][0])

    def test_cli_exposes_no_forbidden_actions(self) -> None:
        source = inspect.getsource(tool.run_stage4g6_lifecycle_write_acceptance)
        for forbidden in (
            "--submit",
            "--cancel",
            "--status",
            "--market-data",
            "--qualification",
            "--allow-state-write",
            "--state-write-action",
            "--allow-ledger-write",
            "--ledger-write-action",
            "--scheduler",
            "--lifecycle-action",
        ):
            self.assertNotIn(forbidden, source)

    def test_stage4g6_files_do_not_call_forbidden_external_apis_or_mutators(self) -> None:
        forbidden = (
            "submit_order_plan(",
            "get_order_status(",
            "cancel_order(",
            "placeOrder",
            "cancelOrder",
            "reqMktData",
            "qualifyContracts",
            "StateStore(",
            ".save(",
            ".write(",
            ".update(",
            ".append_event(",
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
            "random",
            "time.time",
            "datetime.now",
            ".jsonl",
            "open(",
        )
        for path in STAGE4G6_FILES:
            source = path.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, source, f"{value} unexpectedly found in {path}")

    def test_no_daemon_scheduler_or_lifecycle_wiring_exists(self) -> None:
        for path in UNWIRED_RUNTIME_FILES:
            if path.exists():
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("stage4g6_lifecycle_write_acceptance", source)


if __name__ == "__main__":
    unittest.main()
