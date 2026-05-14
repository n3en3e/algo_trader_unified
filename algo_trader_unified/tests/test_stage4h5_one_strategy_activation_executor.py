from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4h5_one_strategy_activation_executor import (
    REQUIRED_OPERATOR_ACKNOWLEDGEMENTS,
    build_stage4h5_one_strategy_activation_executor_report,
)
from algo_trader_unified.tools import stage4h5_one_strategy_activation_executor as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4H5_FILES = [
    ROOT / "core/stage4h5_one_strategy_activation_executor.py",
    ROOT / "tools/stage4h5_one_strategy_activation_executor.py",
]


class FakeActivationWriter:
    def __init__(self, response: dict | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self.response = {"status": "created", "record": {"selected_strategy_id": "S01"}}
        if response is not None:
            self.response = response
        self.error = error

    def activate_one_strategy(self, payload: dict) -> dict:
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return self.response


class FakeAuditWriter:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self.error = error

    def append_activation_audit(self, event: dict) -> dict:
        self.calls.append(event)
        if self.error is not None:
            raise self.error
        return {"status": "appended"}


def valid_stage4h4_report(**overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4h4_one_strategy_enablement_gate_report": True,
        "generated_at": "2026-05-14T14:00:00+00:00",
        "selected_strategy": {"selected_strategy_id": "S01", "paper_only": True},
        "activation_candidate": {
            "available": True,
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
            "max_enabled_strategy_count": 1,
        },
        "proposed_activation_flags": {
            "enable_automated_paper_trading_for_selected_strategy": True,
            "enable_scheduler_for_selected_strategy": True,
            "enable_lifecycle_for_selected_strategy": True,
            "enable_broker_submission_for_selected_strategy": False,
            "enable_live_trading": False,
            "enable_all_strategies": False,
        },
        "proposed_runtime_guards": [{"name": "paper-only mode guard", "required": True}],
        "proposed_monitoring_requirements": [
            {"name": "selected strategy health visible", "required": True}
        ],
        "proposed_kill_switch_requirements": [
            {"name": "operator kill switch available", "required": True}
        ],
        "readiness_for_stage4h5": {
            "ready_to_build_one_strategy_activation_executor": True,
            "blockers": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def clean_state_snapshot(**overrides: object) -> dict:
    report: dict[str, object] = {
        "active_halt": False,
        "unresolved_needs_reconciliation_count": 0,
        "active_intents_count": 0,
    }
    _deep_update(report, overrides)
    return report


def clean_risk_snapshot(**overrides: object) -> dict:
    report: dict[str, object] = {
        "kill_switch_available": True,
        "hard_halt_available": True,
        "daily_loss_limit_available": True,
        "risk_bypass_enabled": False,
    }
    _deep_update(report, overrides)
    return report


def paper_broker_snapshot(**overrides: object) -> dict:
    report: dict[str, object] = {
        "mode": "PAPER",
        "paper_trading": True,
        "ibkr_port": 4004,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }
    _deep_update(report, overrides)
    return report


def build(
    *,
    h4: dict | None = None,
    writer: FakeActivationWriter | None = None,
    audit_writer: FakeAuditWriter | None = None,
    acks: list[str] | None = None,
    allow: bool = True,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker: dict | None = None,
) -> tuple[dict, FakeActivationWriter]:
    actual_writer = writer or FakeActivationWriter()
    report = build_stage4h5_one_strategy_activation_executor_report(
        stage4h4_enablement_gate_report=valid_stage4h4_report() if h4 is None else h4,
        activation_writer=actual_writer,
        audit_writer=audit_writer,
        operator_acknowledgements=list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS) if acks is None else acks,
        allow_activation_write=allow,
        state_snapshot=clean_state_snapshot() if state_snapshot is None else state_snapshot,
        risk_snapshot=clean_risk_snapshot() if risk_snapshot is None else risk_snapshot,
        scheduler_snapshot={} if scheduler_snapshot is None else scheduler_snapshot,
        lifecycle_snapshot={} if lifecycle_snapshot is None else lifecycle_snapshot,
        paper_broker_snapshot=paper_broker_snapshot() if paper_broker is None else paper_broker,
        now_provider=lambda: datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc),
    )
    return report, actual_writer


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


def assert_not_attempted(test_case: unittest.TestCase, report: dict, writer: FakeActivationWriter) -> None:
    test_case.assertFalse(report["execution"]["attempted"])
    test_case.assertEqual([], writer.calls)


class Stage4H5OneStrategyActivationExecutorTests(unittest.TestCase):
    def test_missing_stage4h4_gate_report_refuses_before_writer_calls(self) -> None:
        writer = FakeActivationWriter()
        report = build_stage4h5_one_strategy_activation_executor_report(
            stage4h4_enablement_gate_report=None,
            activation_writer=writer,
        )
        assert_not_attempted(self, report, writer)

    def test_stage4h4_gate_not_ready_refuses_before_writer_calls(self) -> None:
        report, writer = build(
            h4=valid_stage4h4_report(
                readiness_for_stage4h5={
                    "ready_to_build_one_strategy_activation_executor": False
                }
            )
        )
        assert_not_attempted(self, report, writer)

    def test_missing_allow_activation_write_refuses_before_writer_calls(self) -> None:
        report, writer = build(allow=False)
        assert_not_attempted(self, report, writer)
        self.assertFalse(report["gates"]["allow_activation_write"])

    def test_acknowledgement_validation_is_exact_and_safe(self) -> None:
        direct_writer = FakeActivationWriter()
        direct = build_stage4h5_one_strategy_activation_executor_report(
            stage4h4_enablement_gate_report=valid_stage4h4_report(),
            activation_writer=direct_writer,
            operator_acknowledgements=None,
            allow_activation_write=True,
            state_snapshot=clean_state_snapshot(),
            risk_snapshot=clean_risk_snapshot(),
            scheduler_snapshot={},
            lifecycle_snapshot={},
            paper_broker_snapshot=paper_broker_snapshot(),
        )
        assert_not_attempted(self, direct, direct_writer)
        self.assertEqual([], direct["acknowledgement_checks"]["provided"])

        giant, giant_writer = build(acks=[" ".join(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS)])
        assert_not_attempted(self, giant, giant_writer)
        self.assertFalse(giant["acknowledgement_checks"]["exact_match"])

        extra, extra_writer = build(acks=REQUIRED_OPERATOR_ACKNOWLEDGEMENTS[:-1] + ["extra"])
        assert_not_attempted(self, extra, extra_writer)
        self.assertIn(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS[-1], extra["acknowledgement_checks"]["missing"])

        exact, exact_writer = build(
            acks=["  " + item + "  " for item in REQUIRED_OPERATOR_ACKNOWLEDGEMENTS]
        )
        self.assertTrue(exact["acknowledgement_checks"]["exact_match"])
        self.assertEqual(1, len(exact_writer.calls))

    def test_paper_broker_snapshot_blockers_refuse_before_writer_calls(self) -> None:
        missing_writer = FakeActivationWriter()
        missing = build_stage4h5_one_strategy_activation_executor_report(
            stage4h4_enablement_gate_report=valid_stage4h4_report(),
            activation_writer=missing_writer,
            operator_acknowledgements=list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS),
            allow_activation_write=True,
            state_snapshot=clean_state_snapshot(),
            risk_snapshot=clean_risk_snapshot(),
            scheduler_snapshot={},
            lifecycle_snapshot={},
            paper_broker_snapshot=None,
        )
        assert_not_attempted(self, missing, missing_writer)

        cases = [
            paper_broker_snapshot(mode="LIVE"),
            paper_broker_snapshot(ibkr_port=4002),
            paper_broker_snapshot(paper_trading=False),
            paper_broker_snapshot(live_trading_enabled=True),
            paper_broker_snapshot(broker_submission_enabled=True),
        ]
        for case in cases:
            with self.subTest(case=case):
                report, writer = build(paper_broker=case)
                assert_not_attempted(self, report, writer)

    def test_risk_state_scheduler_and_lifecycle_blockers_refuse_before_writer_calls(self) -> None:
        cases = [
            {"risk_snapshot": clean_risk_snapshot(risk_bypass_enabled=True)},
            {"risk_snapshot": clean_risk_snapshot(kill_switch_available=False)},
            {"state_snapshot": clean_state_snapshot(active_halt=True)},
            {"state_snapshot": clean_state_snapshot(unresolved_needs_reconciliation_count=1)},
            {"state_snapshot": clean_state_snapshot(active_intents_count=1)},
            {"scheduler_snapshot": {"scheduler_automation_enabled": True}},
            {"lifecycle_snapshot": {"lifecycle_automation_enabled": True}},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                report, writer = build(**kwargs)
                assert_not_attempted(self, report, writer)

    def test_valid_gate_calls_activation_writer_once_with_exact_payload(self) -> None:
        audit = FakeAuditWriter()
        report, writer = build(audit_writer=audit)
        self.assertTrue(report["execution"]["attempted"])
        self.assertEqual(1, len(writer.calls))
        self.assertEqual(report["activation_payload"], writer.calls[0])
        self.assertNotIn("stage4h4_one_strategy_enablement_gate_report", writer.calls[0])
        payload = report["activation_payload"]
        self.assertEqual("S01", payload["selected_strategy_id"])
        self.assertTrue(payload["paper_only"])
        self.assertEqual(1, payload["enabled_strategy_count"])
        self.assertEqual("4H-5", payload["source_stage"])
        self.assertFalse(payload["live_trading_enabled"])
        self.assertFalse(payload["all_strategies_enabled"])
        self.assertFalse(payload["broker_submission_enabled"])
        self.assertTrue(report["readiness_for_stage4h6"]["ready_to_build_one_strategy_activation_acceptance_report"])

    def test_payload_preserves_stage4h4_guard_monitoring_and_kill_switch_arrays(self) -> None:
        h4 = valid_stage4h4_report(
            proposed_runtime_guards=[{"b": 1}, {"a": 2}],
            proposed_monitoring_requirements=[{"m": "one"}, {"m": "two"}],
            proposed_kill_switch_requirements=[{"k": "one"}],
        )
        report, _writer = build(h4=h4)
        payload = report["activation_payload"]
        self.assertEqual(h4["proposed_runtime_guards"], payload["required_runtime_guards"])
        self.assertEqual(h4["proposed_monitoring_requirements"], payload["required_monitoring"])
        self.assertEqual(h4["proposed_kill_switch_requirements"], payload["required_kill_switches"])

    def test_missing_proposed_arrays_are_handled_safely(self) -> None:
        h4 = valid_stage4h4_report()
        h4.pop("proposed_runtime_guards")
        h4.pop("proposed_monitoring_requirements")
        h4.pop("proposed_kill_switch_requirements")
        report, writer = build(h4=h4)
        self.assertEqual([], report["activation_payload"]["required_runtime_guards"])
        self.assertEqual([], report["activation_payload"]["required_monitoring"])
        self.assertEqual([], report["activation_payload"]["required_kill_switches"])
        self.assertEqual(1, len(writer.calls))
        self.assertTrue(report["warnings"])

    def test_activation_writer_exception_is_captured_and_audit_not_called(self) -> None:
        writer = FakeActivationWriter(error=RuntimeError("boom"))
        audit = FakeAuditWriter()
        report, _writer = build(writer=writer, audit_writer=audit)
        self.assertTrue(report["execution"]["activation_write_attempted"])
        self.assertFalse(report["execution"]["activation_write_succeeded"])
        self.assertIn("RuntimeError: boom", report["execution"]["failure_reason"])
        self.assertEqual([], audit.calls)

    def test_already_exists_matching_record_is_idempotent_success(self) -> None:
        writer = FakeActivationWriter(
            response={
                "status": "already_exists",
                "record": {
                    "selected_strategy_id": "S01",
                    "paper_only": True,
                    "live_trading_enabled": False,
                    "all_strategies_enabled": False,
                    "enabled_strategy_count": 1,
                },
            }
        )
        audit = FakeAuditWriter()
        report, _writer = build(writer=writer, audit_writer=audit)
        self.assertTrue(report["execution"]["activation_write_succeeded"])
        self.assertTrue(report["readiness_for_stage4h6"]["ready_to_build_one_strategy_activation_acceptance_report"])

    def test_already_exists_mismatch_or_unsafe_flags_fail_closed(self) -> None:
        cases = [
            {"selected_strategy_id": "S02", "paper_only": True, "live_trading_enabled": False, "all_strategies_enabled": False, "enabled_strategy_count": 1},
            {"selected_strategy_id": "S01", "paper_only": True, "live_trading_enabled": True, "all_strategies_enabled": False, "enabled_strategy_count": 1},
            {"selected_strategy_id": "S01", "paper_only": True, "live_trading_enabled": False, "all_strategies_enabled": True, "enabled_strategy_count": 1},
        ]
        for record in cases:
            with self.subTest(record=record):
                audit = FakeAuditWriter()
                report, _writer = build(
                    writer=FakeActivationWriter(response={"status": "already_exists", "record": record}),
                    audit_writer=audit,
                )
                self.assertFalse(report["execution"]["activation_write_succeeded"])
                self.assertEqual([], audit.calls)

    def test_conflict_fails_closed_and_skips_audit(self) -> None:
        audit = FakeAuditWriter()
        report, _writer = build(
            writer=FakeActivationWriter(response={"status": "conflict"}),
            audit_writer=audit,
        )
        self.assertFalse(report["execution"]["activation_write_succeeded"])
        self.assertEqual([], audit.calls)

    def test_audit_exception_sets_manual_rollback_and_preserves_applied_operation(self) -> None:
        report, _writer = build(audit_writer=FakeAuditWriter(error=ValueError("audit down")))
        self.assertTrue(report["execution"]["activation_write_succeeded"])
        self.assertFalse(report["execution"]["audit_write_succeeded"])
        self.assertTrue(report["rollback"]["rollback_required"])
        self.assertFalse(report["rollback"]["rollback_attempted"])
        self.assertEqual(
            "no automated rollback is supported in this phase",
            report["rollback"]["rollback_limitations"],
        )
        self.assertEqual(["activation_writer.activate_one_strategy"], report["applied_operations"])
        self.assertEqual(["automated_rollback"], report["skipped_operations"])
        self.assertIn("ValueError: audit down", report["execution"]["failure_reason"])

    def test_report_includes_required_top_level_fields_and_is_json_safe(self) -> None:
        report, _writer = build(audit_writer=FakeAuditWriter())
        for key in (
            "dry_run",
            "stage4h5_one_strategy_activation_executor_report",
            "generated_at",
            "gates",
            "acknowledgement_checks",
            "selected_strategy",
            "activation_payload",
            "execution",
            "applied_operations",
            "skipped_operations",
            "rollback",
            "safety",
            "readiness_for_stage4h6",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report, sort_keys=True)

    def test_input_reports_and_snapshots_are_not_mutated(self) -> None:
        h4 = valid_stage4h4_report()
        state = clean_state_snapshot()
        risk = clean_risk_snapshot()
        paper = paper_broker_snapshot()
        before = copy.deepcopy((h4, state, risk, paper))
        build(h4=h4, state_snapshot=state, risk_snapshot=risk, paper_broker=paper)
        self.assertEqual(before, (h4, state, risk, paper))

    def test_cli_requires_dry_run_only_before_parsing(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_stage4h5_one_strategy_activation_executor(
                ["--json", "--stage4h4-gate-json", "{"]
            )
        self.assertEqual(1, code)
        self.assertIn("--dry-run-only", stderr.getvalue())

    def test_cli_ack_uses_action_append_and_json_stdout_is_strict_json(self) -> None:
        source = Path(tool.__file__).read_text()
        self.assertIn('parser.add_argument("--ack", action="append"', source)
        stdout = io.StringIO()
        args = [
            "--dry-run-only",
            "--json",
            "--stage4h4-gate-json",
            json.dumps(valid_stage4h4_report()),
            "--state-snapshot-json",
            json.dumps(clean_state_snapshot()),
            "--risk-snapshot-json",
            json.dumps(clean_risk_snapshot()),
            "--scheduler-snapshot-json",
            "{}",
            "--lifecycle-snapshot-json",
            "{}",
            "--paper-broker-snapshot-json",
            json.dumps(paper_broker_snapshot()),
        ]
        for ack in REQUIRED_OPERATOR_ACKNOWLEDGEMENTS:
            args.extend(["--ack", ack])
        with redirect_stdout(stdout):
            code = tool.run_stage4h5_one_strategy_activation_executor(args)
        self.assertEqual(1, code)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4h5_one_strategy_activation_executor_report"])

    def test_cli_exposes_no_execution_or_broad_enablement_actions(self) -> None:
        parser_source = Path(tool.__file__).read_text()
        disallowed_options = [
            "--submit",
            "--cancel",
            "--status",
            "--market-data",
            "--qualification",
            "--qualify",
            "--scheduler-enable",
            "--lifecycle-enable",
        ]
        for option in disallowed_options:
            self.assertNotIn(option, parser_source)

    def test_stage4h5_source_has_no_forbidden_runtime_call_tokens(self) -> None:
        forbidden = [
            "submit_order_plan(",
            "get_order_status(",
            "cancel_order(",
            ".placeOrder(",
            ".cancelOrder(",
            ".reqMktData(",
            ".qualifyContracts(",
            ".add_job(",
            "add_job(",
            ".run_scan(",
            "run_scan(",
            ".scan_now(",
            "scan_now(",
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
            "random.",
            "time.time",
            "datetime.now",
            "StateStore(",
            ".save(",
            ".write_json(",
            ".append_event(",
            ".append_jsonl(",
            "ib_insync",
        ]
        for path in STAGE4H5_FILES:
            source = path.read_text()
            for token in forbidden:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
