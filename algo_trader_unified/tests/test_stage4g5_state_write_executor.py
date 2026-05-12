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

from algo_trader_unified.core.stage4g5_state_write_executor import (
    REQUIRED_ACKNOWLEDGEMENTS,
    build_stage4g5_state_write_executor_report,
)
from algo_trader_unified.tools import stage4g5_state_write_executor as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4G5_FILES = [
    ROOT / "core/stage4g5_state_write_executor.py",
    ROOT / "tools/stage4g5_state_write_executor.py",
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


class FakeStateStoreWriter:
    def __init__(self, *, fail_on: str | None = None, result: dict | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail_on = fail_on
        self.result = result

    def upsert_order(self, payload: dict) -> dict:
        self.calls.append(("upsert_order", copy.deepcopy(payload)))
        if self.fail_on == "upsert_order":
            raise RuntimeError("state order write failed")
        return copy.deepcopy(self.result) if self.result is not None else {"status": "ok", "record": payload}

    def upsert_position(self, payload: dict) -> dict:
        self.calls.append(("upsert_position", copy.deepcopy(payload)))
        if self.fail_on == "upsert_position":
            raise RuntimeError("state position write failed")
        return copy.deepcopy(self.result) if self.result is not None else {"status": "ok", "record": payload}


class FakeLedgerWriter:
    def __init__(self, *, fail: bool = False, result: dict | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail = fail
        self.result = result

    def append_event(self, event: dict) -> dict:
        self.calls.append(("append_event", copy.deepcopy(event)))
        if self.fail:
            raise RuntimeError("ledger write failed")
        return copy.deepcopy(self.result) if self.result is not None else {"status": "ok", "record": event}


def valid_dry_run_report(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4g4_state_write_dry_run": True,
        "generated_at": "2026-05-10T14:00:00+00:00",
        "operation_schema_checks": {
            "operations_structured": True,
            "recognized_operations": True,
            "deterministic_operation_order": True,
        },
        "dry_run_packet": {
            "available": True,
            "dry_run_operations": [
                {
                    "sequence_number": 1,
                    "operation_type": "state_store_operation",
                    "target": "StateStore",
                    "operation": "upsert_order",
                    "would_execute": False,
                    "payload": {
                        "client_order_id": "intent-stage4g5-001",
                        "broker_order_id": "9001",
                        "strategy_id": "S01_VOL_BASELINE",
                        "symbol": "XSP",
                        "paper_only": True,
                    },
                },
                {
                    "sequence_number": 2,
                    "operation_type": "state_store_operation",
                    "target": "StateStore",
                    "operation": "upsert_position",
                    "would_execute": False,
                    "payload": {
                        "position_key": "S01_VOL_BASELINE:XSP:intent-stage4g5-001",
                        "client_order_id": "intent-stage4g5-001",
                        "broker_order_id": "9001",
                        "quantity": 1,
                        "avg_fill_price": Decimal("1.25"),
                        "paper_only": True,
                    },
                },
                {
                    "sequence_number": 3,
                    "operation_type": "ledger_event",
                    "target": "Ledger",
                    "operation": "append_event",
                    "would_execute": False,
                    "payload": {
                        "event_type": "paper_order_lifecycle_state_write",
                        "timestamp": "2026-05-10T12:30:00+00:00",
                        "client_order_id": "intent-stage4g5-001",
                        "broker_order_id": "9001",
                    },
                },
                {
                    "sequence_number": 4,
                    "operation_type": "lifecycle_transition",
                    "target": "Lifecycle",
                    "operation": "record_lifecycle_transition",
                    "would_execute": False,
                    "payload": {
                        "transition_to": "paper_order_filled",
                        "proposal_only": True,
                        "enabled": False,
                    },
                },
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
        "readiness_for_stage4g5": {
            "ready_to_build_manual_state_write_executor": True,
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
        "state_write_dry_run_report": valid_dry_run_report(),
        "state_store_writer": FakeStateStoreWriter(),
        "ledger_writer": FakeLedgerWriter(),
        "operator_acknowledgements": list(REQUIRED_ACKNOWLEDGEMENTS),
        "allow_state_write": True,
        "allow_ledger_write": True,
        "now_provider": lambda: datetime(2026, 5, 10, 15, 0, tzinfo=timezone.utc),
    }
    kwargs.update(overrides)
    return build_stage4g5_state_write_executor_report(**kwargs)


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


class Stage4G5StateWriteExecutorTests(unittest.TestCase):
    def assert_refused_without_calls(self, report: dict, state: FakeStateStoreWriter, ledger: FakeLedgerWriter) -> None:
        self.assertFalse(report["gates"]["passed"])
        self.assertFalse(report["execution"]["attempted"])
        self.assertEqual([], state.calls)
        self.assertEqual([], ledger.calls)

    def test_missing_dry_run_report_refuses_before_writer_calls(self) -> None:
        state = FakeStateStoreWriter()
        ledger = FakeLedgerWriter()
        report = build(state_write_dry_run_report=None, state_store_writer=state, ledger_writer=ledger)
        self.assert_refused_without_calls(report, state, ledger)
        self.assertIn("state_write_dry_run_report missing", report["gates"]["reasons"])

    def test_dry_run_report_not_ready_refuses_before_writer_calls(self) -> None:
        state = FakeStateStoreWriter()
        ledger = FakeLedgerWriter()
        report = build(
            state_write_dry_run_report=valid_dry_run_report(
                readiness_for_stage4g5={"ready_to_build_manual_state_write_executor": False}
            ),
            state_store_writer=state,
            ledger_writer=ledger,
        )
        self.assert_refused_without_calls(report, state, ledger)

    def test_missing_allow_flags_refuse_before_writer_calls(self) -> None:
        for key in ("allow_state_write", "allow_ledger_write"):
            state = FakeStateStoreWriter()
            ledger = FakeLedgerWriter()
            report = build(**{key: False, "state_store_writer": state, "ledger_writer": ledger})
            self.assert_refused_without_calls(report, state, ledger)

    def test_missing_acknowledgement_refuses_before_writer_calls(self) -> None:
        state = FakeStateStoreWriter()
        ledger = FakeLedgerWriter()
        report = build(
            operator_acknowledgements=REQUIRED_ACKNOWLEDGEMENTS[:-1],
            state_store_writer=state,
            ledger_writer=ledger,
        )
        self.assert_refused_without_calls(report, state, ledger)
        self.assertEqual([REQUIRED_ACKNOWLEDGEMENTS[-1]], report["acknowledgement_checks"]["missing"])

    def test_giant_substring_acknowledgement_does_not_pass(self) -> None:
        state = FakeStateStoreWriter()
        ledger = FakeLedgerWriter()
        report = build(
            operator_acknowledgements=[" ".join(REQUIRED_ACKNOWLEDGEMENTS)],
            state_store_writer=state,
            ledger_writer=ledger,
        )
        self.assert_refused_without_calls(report, state, ledger)

    def test_extra_acknowledgements_do_not_compensate_for_missing_required_ack(self) -> None:
        state = FakeStateStoreWriter()
        ledger = FakeLedgerWriter()
        report = build(
            operator_acknowledgements=REQUIRED_ACKNOWLEDGEMENTS[:-1] + ["I reviewed everything."],
            state_store_writer=state,
            ledger_writer=ledger,
        )
        self.assert_refused_without_calls(report, state, ledger)

    def test_all_exact_acknowledgements_pass_gates(self) -> None:
        report = build()
        self.assertTrue(report["gates"]["passed"])
        self.assertTrue(report["acknowledgement_checks"]["exact_match"])

    def test_unrecognized_state_store_operation_refuses_before_writer_calls(self) -> None:
        dry_run = valid_dry_run_report()
        operations = dry_run["dry_run_packet"]["dry_run_operations"]  # type: ignore[index]
        operations[0]["operation"] = "delete_order"  # type: ignore[index]
        state = FakeStateStoreWriter()
        ledger = FakeLedgerWriter()
        report = build(state_write_dry_run_report=dry_run, state_store_writer=state, ledger_writer=ledger)
        self.assert_refused_without_calls(report, state, ledger)
        self.assertIn("unrecognized StateStore operation", report["gates"]["reasons"])

    def test_malformed_operation_payload_refuses_before_writer_calls(self) -> None:
        dry_run = valid_dry_run_report()
        operations = dry_run["dry_run_packet"]["dry_run_operations"]  # type: ignore[index]
        operations[0]["payload"] = "bad"  # type: ignore[index]
        state = FakeStateStoreWriter()
        ledger = FakeLedgerWriter()
        report = build(state_write_dry_run_report=dry_run, state_store_writer=state, ledger_writer=ledger)
        self.assert_refused_without_calls(report, state, ledger)
        self.assertIn("StateStore operation payload must be a JSON-safe dict", report["gates"]["reasons"])

    def test_valid_packet_writes_state_then_ledger_in_deterministic_order(self) -> None:
        state = FakeStateStoreWriter()
        ledger = FakeLedgerWriter()
        report = build(state_store_writer=state, ledger_writer=ledger)
        self.assertEqual(
            [("upsert_order", "intent-stage4g5-001"), ("upsert_position", "intent-stage4g5-001")],
            [(name, payload["client_order_id"]) for name, payload in state.calls],
        )
        self.assertEqual([("append_event", "intent-stage4g5-001")], [(name, payload["client_order_id"]) for name, payload in ledger.calls])
        self.assertEqual([1, 2, 3], [item["sequence_number"] for item in report["applied_operations"]])

    def test_lifecycle_transition_is_not_executed(self) -> None:
        report = build()
        self.assertFalse(report["execution"]["lifecycle_transition_executed"])
        self.assertNotIn(4, [item["sequence_number"] for item in report["applied_operations"]])

    def test_writer_return_values_are_json_safe(self) -> None:
        report = build(state_store_writer=FakeStateStoreWriter(result={"status": "ok", "value": Decimal("1.5")}))
        assert_json_safe(self, report)

    def test_state_writer_exception_stops_before_ledger_write(self) -> None:
        state = FakeStateStoreWriter(fail_on="upsert_order")
        ledger = FakeLedgerWriter()
        report = build(state_store_writer=state, ledger_writer=ledger)
        self.assertEqual([], ledger.calls)
        self.assertEqual("StateStore.upsert_order#1", report["execution"]["failed_step"])
        self.assertIn("RuntimeError: state order write failed", report["execution"]["failure_reason"])

    def test_state_writer_exception_keeps_prior_success_and_skips_remaining(self) -> None:
        state = FakeStateStoreWriter(fail_on="upsert_position")
        ledger = FakeLedgerWriter()
        report = build(state_store_writer=state, ledger_writer=ledger)
        self.assertEqual([1], [item["sequence_number"] for item in report["applied_operations"]])
        self.assertEqual([2, 3], [item["sequence_number"] for item in report["skipped_operations"]])
        self.assertTrue(report["rollback"]["rollback_required"])

    def test_ledger_exception_requires_manual_rollback_after_state_success(self) -> None:
        ledger = FakeLedgerWriter(fail=True)
        report = build(ledger_writer=ledger)
        self.assertEqual([1, 2], [item["sequence_number"] for item in report["applied_operations"]])
        self.assertEqual([3], [item["sequence_number"] for item in report["skipped_operations"]])
        self.assertTrue(report["rollback"]["rollback_required"])
        self.assertFalse(report["rollback"]["rollback_attempted"])
        self.assertIn("RuntimeError: ledger write failed", report["execution"]["failure_reason"])

    def test_duplicate_success_with_matching_ids_is_accepted(self) -> None:
        result = {
            "status": "already_exists",
            "record": {"client_order_id": "intent-stage4g5-001", "broker_order_id": "9001"},
        }
        report = build(state_store_writer=FakeStateStoreWriter(result=result))
        self.assertTrue(report["execution"]["state_store_write_succeeded"])

    def test_already_exists_with_mismatched_ids_fails_closed(self) -> None:
        result = {
            "status": "already_exists",
            "record": {"client_order_id": "other", "broker_order_id": "9001"},
        }
        ledger = FakeLedgerWriter()
        report = build(state_store_writer=FakeStateStoreWriter(result=result), ledger_writer=ledger)
        self.assertFalse(report["execution"]["completed"])
        self.assertEqual([], ledger.calls)
        self.assertIn("ValueError:", report["execution"]["failure_reason"])

    def test_duplicate_conflict_with_mismatched_ids_fails_closed(self) -> None:
        result = {
            "status": "duplicate",
            "record": {"client_order_id": "intent-stage4g5-001", "broker_order_id": "other"},
        }
        report = build(state_store_writer=FakeStateStoreWriter(result=result))
        self.assertFalse(report["readiness_for_stage4g6"]["ready_to_build_manual_lifecycle_write_acceptance_report"])

    def test_readiness_true_only_when_state_and_ledger_writes_succeed(self) -> None:
        report = build()
        self.assertTrue(report["readiness_for_stage4g6"]["ready_to_build_manual_lifecycle_write_acceptance_report"])
        self.assertFalse(report["rollback"]["rollback_required"])

    def test_readiness_false_when_rollback_required(self) -> None:
        report = build(ledger_writer=FakeLedgerWriter(fail=True))
        self.assertFalse(report["readiness_for_stage4g6"]["ready_to_build_manual_lifecycle_write_acceptance_report"])

    def test_write_plan_and_safety_outputs_do_not_enable_automation_or_live(self) -> None:
        report = build()
        self.assertFalse(report["write_plan"]["lifecycle_transition_enabled"])
        self.assertFalse(report["write_plan"]["daemon_wiring_enabled"])
        self.assertFalse(report["write_plan"]["scheduler_wiring_enabled"])
        self.assertFalse(report["safety"]["live_orders_enabled"])
        self.assertFalse(report["safety"]["automated_paper_trading_enabled"])

    def test_input_report_is_not_mutated(self) -> None:
        dry_run = valid_dry_run_report()
        before = copy.deepcopy(dry_run)
        build(state_write_dry_run_report=dry_run)
        self.assertEqual(before, dry_run)

    def test_report_includes_required_top_level_fields_and_is_json_safe(self) -> None:
        report = build()
        for key in (
            "dry_run",
            "stage4g5_state_write_executor",
            "generated_at",
            "gates",
            "acknowledgement_checks",
            "execution",
            "applied_operations",
            "skipped_operations",
            "rollback",
            "write_plan",
            "safety",
            "readiness_for_stage4g6",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        assert_json_safe(self, report)

    def test_cli_requires_dry_run_only_before_parsing(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_stage4g5_state_write_executor([])
        self.assertEqual(1, code)
        self.assertIn("--dry-run-only", stderr.getvalue())

    def test_cli_ack_uses_append_action(self) -> None:
        source = inspect.getsource(tool.run_stage4g5_state_write_executor)
        self.assertIn('parser.add_argument("--ack", action="append"', source)

    def test_cli_json_stdout_is_strict_json(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4g5_state_write_executor(
                [
                    "--dry-run-only",
                    "--json",
                    "--state-write-dry-run-json",
                    json.dumps(valid_dry_run_report(), default=str),
                ]
            )
        self.assertEqual(1, code)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4g5_state_write_executor"])

    def test_cli_exposes_no_forbidden_actions(self) -> None:
        source = inspect.getsource(tool.run_stage4g5_state_write_executor)
        for forbidden in (
            "--submit",
            "--cancel",
            "--status",
            "--market-data",
            "--qualification",
            "--scheduler",
            "--lifecycle",
        ):
            self.assertNotIn(forbidden, source)

    def test_stage4g5_files_do_not_call_forbidden_external_apis_or_mutators(self) -> None:
        forbidden = (
            "submit_order_plan",
            "get_order_status",
            "cancel_order",
            "placeOrder",
            "cancelOrder",
            "reqMktData",
            "qualifyContracts",
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
            "StateStore(",
            ".jsonl",
            "open(",
        )
        for path in STAGE4G5_FILES:
            source = path.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, source, f"{value} unexpectedly found in {path}")

    def test_no_daemon_scheduler_or_lifecycle_wiring_exists(self) -> None:
        for path in UNWIRED_RUNTIME_FILES:
            if path.exists():
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("stage4g5_state_write_executor", source)


if __name__ == "__main__":
    unittest.main()
