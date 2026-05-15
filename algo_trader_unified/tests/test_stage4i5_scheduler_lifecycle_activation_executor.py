from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
import json
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4i5_scheduler_lifecycle_activation_executor import (
    MARKET_WINDOW_MANUAL_WARNING,
    REQUIRED_OPERATOR_ACKNOWLEDGEMENTS,
    build_stage4i5_scheduler_lifecycle_activation_executor_report,
)
from algo_trader_unified.tools import stage4i5_scheduler_lifecycle_activation_executor as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4I5_FILES = [
    ROOT / "core/stage4i5_scheduler_lifecycle_activation_executor.py",
    ROOT / "tools/stage4i5_scheduler_lifecycle_activation_executor.py",
]


class FakeSchedulerWriter:
    def __init__(self, result: dict | None = None, exc: Exception | None = None) -> None:
        self.result = {"status": "created", "record": {}} if result is None else result
        self.exc = exc
        self.calls: list[dict] = []

    def activate_scheduler(self, payload: dict) -> dict:
        self.calls.append(copy.deepcopy(payload))
        if self.exc is not None:
            raise self.exc
        return copy.deepcopy(self.result)


class FakeLifecycleWriter:
    def __init__(self, result: dict | None = None, exc: Exception | None = None) -> None:
        self.result = {"status": "created", "record": {}} if result is None else result
        self.exc = exc
        self.calls: list[dict] = []

    def activate_lifecycle(self, payload: dict) -> dict:
        self.calls.append(copy.deepcopy(payload))
        if self.exc is not None:
            raise self.exc
        return copy.deepcopy(self.result)


class FakeAuditWriter:
    def __init__(self, result: dict | None = None, exc: Exception | None = None) -> None:
        self.result = {"status": "appended"} if result is None else result
        self.exc = exc
        self.calls: list[dict] = []

    def append_scheduler_lifecycle_activation_audit(self, event: dict) -> dict:
        self.calls.append(copy.deepcopy(event))
        if self.exc is not None:
            raise self.exc
        return copy.deepcopy(self.result)


def valid_stage4i4_report(**overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4i4_scheduler_lifecycle_activation_gate_report": True,
        "generated_at": "2026-05-15T12:00:00+00:00",
        "selected_strategy": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
        },
        "scheduler_lifecycle_activation_candidate": {
            "available": True,
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
            "scheduler_activation_allowed_next_phase": True,
            "lifecycle_activation_allowed_next_phase": True,
            "broker_submission_allowed_next_phase": False,
            "live_trading_enabled": False,
            "all_strategies_enabled": False,
            "enabled_strategy_count": 1,
        },
        "proposed_scheduler_activation": {
            "available": True,
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
            "would_register_in_4I4": False,
            "proposed_enabled_in_4I5": True,
            "scheduler_job_enabled_now": False,
        },
        "proposed_lifecycle_activation": {
            "available": True,
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
            "would_execute_in_4I4": False,
            "proposed_enabled_in_4I5": True,
            "lifecycle_execution_enabled_now": False,
        },
        "readiness_for_stage4i5": {
            "ready_to_build_scheduler_lifecycle_activation_executor": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def matching_activation_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "activation_record": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "enabled_strategy_count": 1,
            "live_trading_enabled": False,
            "all_strategies_enabled": False,
            "broker_submission_enabled": False,
        },
        "active_strategy_ids": ["S01"],
    }
    _deep_update(snapshot, overrides)
    return snapshot


def clean_state_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "active_halt": False,
        "unresolved_needs_reconciliation_count": 0,
        "active_intents_count": 0,
        "open_positions_count": 2,
    }
    _deep_update(snapshot, overrides)
    return snapshot


def clean_risk_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "kill_switch_available": True,
        "hard_halt_available": True,
        "daily_loss_limit_available": True,
        "max_position_limit_available": True,
        "risk_bypass_enabled": False,
    }
    _deep_update(snapshot, overrides)
    return snapshot


def clean_paper_broker_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "mode": "PAPER",
        "paper_trading": True,
        "ibkr_port": 4004,
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }
    _deep_update(snapshot, overrides)
    return snapshot


def clean_market_window_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {
        "is_trading_day": True,
        "market_open": True,
        "allowed_to_schedule_paper_run": True,
    }
    _deep_update(snapshot, overrides)
    return snapshot


def build(
    *,
    i4: dict | None = None,
    scheduler_writer: FakeSchedulerWriter | None = None,
    lifecycle_writer: FakeLifecycleWriter | None = None,
    audit_writer: FakeAuditWriter | None = None,
    activation_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    market_window_snapshot: dict | None = None,
    operator_acknowledgements: list[str] | None = None,
    allow: bool = True,
) -> dict:
    return build_stage4i5_scheduler_lifecycle_activation_executor_report(
        stage4i4_activation_gate_report=valid_stage4i4_report() if i4 is None else i4,
        scheduler_activation_writer=scheduler_writer or FakeSchedulerWriter(),
        lifecycle_activation_writer=lifecycle_writer or FakeLifecycleWriter(),
        audit_writer=audit_writer,
        activation_snapshot=activation_snapshot,
        state_snapshot=state_snapshot,
        risk_snapshot=risk_snapshot,
        scheduler_snapshot=scheduler_snapshot,
        lifecycle_snapshot=lifecycle_snapshot,
        paper_broker_snapshot=paper_broker_snapshot,
        market_window_snapshot=market_window_snapshot,
        operator_acknowledgements=operator_acknowledgements,
        allow_scheduler_lifecycle_activation=allow,
        now_provider=lambda: datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc),
    )


def build_clean(**kwargs: object) -> dict:
    defaults: dict[str, object] = {
        "activation_snapshot": matching_activation_snapshot(),
        "state_snapshot": clean_state_snapshot(),
        "risk_snapshot": clean_risk_snapshot(),
        "scheduler_snapshot": {},
        "lifecycle_snapshot": {},
        "paper_broker_snapshot": clean_paper_broker_snapshot(),
        "market_window_snapshot": clean_market_window_snapshot(),
        "operator_acknowledgements": list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS),
    }
    defaults.update(kwargs)
    return build(**defaults)


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class Stage4I5SchedulerLifecycleActivationExecutorTests(unittest.TestCase):
    def assert_not_attempted(self, report: dict, scheduler: FakeSchedulerWriter, lifecycle: FakeLifecycleWriter) -> None:
        self.assertFalse(report["execution"]["attempted"])
        self.assertEqual([], scheduler.calls)
        self.assertEqual([], lifecycle.calls)
        self.assertFalse(
            report["readiness_for_stage4i6"][
                "ready_to_build_scheduler_lifecycle_activation_acceptance"
            ]
        )

    def test_missing_stage4i4_gate_report_refuses_before_writer_calls(self) -> None:
        scheduler = FakeSchedulerWriter()
        lifecycle = FakeLifecycleWriter()
        report = build_stage4i5_scheduler_lifecycle_activation_executor_report(
            stage4i4_activation_gate_report=None,
            scheduler_activation_writer=scheduler,
            lifecycle_activation_writer=lifecycle,
            operator_acknowledgements=list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS),
            allow_scheduler_lifecycle_activation=True,
        )
        self.assert_not_attempted(report, scheduler, lifecycle)

    def test_stage4i4_gate_not_ready_and_missing_allow_refuse_before_writer_calls(self) -> None:
        for kwargs in (
            {"i4": valid_stage4i4_report(readiness_for_stage4i5={"ready_to_build_scheduler_lifecycle_activation_executor": False})},
            {"allow": False},
        ):
            with self.subTest(kwargs=kwargs):
                scheduler = FakeSchedulerWriter()
                lifecycle = FakeLifecycleWriter()
                report = build_clean(scheduler_writer=scheduler, lifecycle_writer=lifecycle, **kwargs)
                self.assert_not_attempted(report, scheduler, lifecycle)

    def test_acknowledgement_rules(self) -> None:
        for acks in (
            None,
            [],
            [" ".join(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS)],
            ["extra"] + REQUIRED_OPERATOR_ACKNOWLEDGEMENTS[:-1],
        ):
            with self.subTest(acks=acks):
                scheduler = FakeSchedulerWriter()
                lifecycle = FakeLifecycleWriter()
                report = build_clean(
                    scheduler_writer=scheduler,
                    lifecycle_writer=lifecycle,
                    operator_acknowledgements=acks,  # type: ignore[arg-type]
                )
                self.assert_not_attempted(report, scheduler, lifecycle)
        none_report = build_clean(operator_acknowledgements=None)
        self.assertEqual([], none_report["acknowledgement_checks"]["provided"])
        exact = build_clean(
            operator_acknowledgements=[f"  {ack}  " for ack in REQUIRED_OPERATOR_ACKNOWLEDGEMENTS]
            + [object()]  # type: ignore[list-item]
        )
        self.assertTrue(exact["execution"]["attempted"])

    def test_gate_field_blockers_refuse_before_writer_calls(self) -> None:
        cases = [
            valid_stage4i4_report(selected_strategy="bad"),
            valid_stage4i4_report(
                scheduler_lifecycle_activation_candidate={"broker_submission_allowed_next_phase": True}
            ),
            valid_stage4i4_report(
                scheduler_lifecycle_activation_candidate={"live_trading_enabled": True}
            ),
            valid_stage4i4_report(
                scheduler_lifecycle_activation_candidate={"all_strategies_enabled": True}
            ),
            valid_stage4i4_report(proposed_scheduler_activation={"available": False}),
            valid_stage4i4_report(proposed_lifecycle_activation={"available": False}),
        ]
        for i4 in cases:
            with self.subTest(i4=i4):
                scheduler = FakeSchedulerWriter()
                lifecycle = FakeLifecycleWriter()
                report = build_clean(i4=i4, scheduler_writer=scheduler, lifecycle_writer=lifecycle)
                self.assert_not_attempted(report, scheduler, lifecycle)

    def test_valid_gate_calls_writers_once_in_order_with_exact_payloads(self) -> None:
        scheduler = FakeSchedulerWriter()
        lifecycle = FakeLifecycleWriter()
        audit = FakeAuditWriter()
        report = build_clean(scheduler_writer=scheduler, lifecycle_writer=lifecycle, audit_writer=audit)
        self.assertEqual(1, len(scheduler.calls))
        self.assertEqual(1, len(lifecycle.calls))
        self.assertEqual(1, len(audit.calls))
        self.assertEqual(report["scheduler_activation_payload"], scheduler.calls[0])
        self.assertEqual(report["lifecycle_activation_payload"], lifecycle.calls[0])
        self.assertTrue(report["execution"]["scheduler_activation_succeeded"])
        self.assertTrue(report["execution"]["lifecycle_activation_succeeded"])
        self.assertTrue(report["execution"]["audit_write_succeeded"])
        self.assertTrue(report["readiness_for_stage4i6"]["ready_to_build_scheduler_lifecycle_activation_acceptance"])
        self.assertEqual(
            ["scheduler_activation", "lifecycle_activation", "audit"],
            [item["target"] for item in report["applied_operations"]],
        )

    def test_payloads_are_one_strategy_paper_only_and_hardcode_disabled_execution_flags(self) -> None:
        report = build_clean()
        for key in ("scheduler_activation_payload", "lifecycle_activation_payload"):
            payload = report[key]
            self.assertEqual("S01", payload["selected_strategy_id"])
            self.assertTrue(payload["paper_only"])
            self.assertTrue(payload["one_strategy_only"])
            self.assertFalse(payload["live_trading_enabled"])
            self.assertFalse(payload["all_strategies_enabled"])
            self.assertFalse(payload["broker_submission_enabled"])
            self.assertFalse(payload["strategy_scan_execution_enabled"])
            self.assertFalse(payload["lifecycle_transition_execution_enabled"])
            self.assertFalse(payload["market_data_enabled"])
            self.assertFalse(payload["contract_qualification_enabled"])
        unsafe_i4 = valid_stage4i4_report(
            proposed_scheduler_activation={
                "job_payload": {
                    "strategy_scan_execution_enabled": True,
                    "lifecycle_transition_execution_enabled": True,
                }
            },
            proposed_lifecycle_activation={
                "lifecycle_payload": {
                    "strategy_scan_execution_enabled": True,
                    "lifecycle_transition_execution_enabled": True,
                }
            },
        )
        unsafe_report = build_clean(i4=unsafe_i4)
        self.assertFalse(unsafe_report["scheduler_activation_payload"]["strategy_scan_execution_enabled"])
        self.assertFalse(unsafe_report["scheduler_activation_payload"]["lifecycle_transition_execution_enabled"])
        self.assertFalse(unsafe_report["lifecycle_activation_payload"]["strategy_scan_execution_enabled"])
        self.assertFalse(unsafe_report["lifecycle_activation_payload"]["lifecycle_transition_execution_enabled"])

    def test_scheduler_writer_exception_conflict_and_already_exists_handling(self) -> None:
        lifecycle = FakeLifecycleWriter()
        report = build_clean(
            scheduler_writer=FakeSchedulerWriter(exc=RuntimeError("boom")),
            lifecycle_writer=lifecycle,
        )
        self.assertEqual("scheduler_activation", report["execution"]["failed_step"])
        self.assertIn("RuntimeError: boom", report["execution"]["failure_reason"])
        self.assertEqual([], lifecycle.calls)

        match = {
            "status": "already_exists",
            "record": {
                "selected_strategy_id": "S01",
                "paper_only": True,
                "one_strategy_only": True,
                "live_trading_enabled": False,
                "all_strategies_enabled": False,
                "broker_submission_enabled": False,
                "scheduler_job_scope": "single_strategy",
                "strategy_scan_execution_enabled": False,
                "lifecycle_transition_execution_enabled": False,
            },
        }
        self.assertTrue(
            build_clean(scheduler_writer=FakeSchedulerWriter(match))["execution"][
                "scheduler_activation_succeeded"
            ]
        )
        mismatch = copy.deepcopy(match)
        mismatch["record"]["live_trading_enabled"] = True
        failed = build_clean(scheduler_writer=FakeSchedulerWriter(mismatch))
        self.assertEqual("scheduler_activation", failed["execution"]["failed_step"])
        conflict = build_clean(scheduler_writer=FakeSchedulerWriter({"status": "conflict"}))
        self.assertEqual("scheduler_activation", conflict["execution"]["failed_step"])

    def test_lifecycle_writer_partial_failure_and_already_exists_handling(self) -> None:
        report = build_clean(lifecycle_writer=FakeLifecycleWriter(exc=ValueError("bad lifecycle")))
        self.assertTrue(report["execution"]["scheduler_activation_succeeded"])
        self.assertFalse(report["execution"]["lifecycle_activation_succeeded"])
        self.assertTrue(report["rollback"]["rollback_required"])
        self.assertFalse(report["rollback"]["rollback_attempted"])
        self.assertIn("no automated rollback is supported", report["rollback"]["rollback_limitations"])
        self.assertEqual("scheduler_activation", report["applied_operations"][0]["target"])
        self.assertEqual("audit", report["skipped_operations"][0]["target"])

        match = {
            "status": "already_exists",
            "record": {
                "selected_strategy_id": "S01",
                "paper_only": True,
                "one_strategy_only": True,
                "live_trading_enabled": False,
                "all_strategies_enabled": False,
                "broker_submission_enabled": False,
                "lifecycle_scope": "single_strategy",
                "strategy_scan_execution_enabled": False,
                "lifecycle_transition_execution_enabled": False,
            },
        }
        self.assertTrue(
            build_clean(lifecycle_writer=FakeLifecycleWriter(match))["execution"][
                "lifecycle_activation_succeeded"
            ]
        )
        mismatch = copy.deepcopy(match)
        mismatch["record"]["broker_submission_enabled"] = True
        failed = build_clean(lifecycle_writer=FakeLifecycleWriter(mismatch))
        self.assertEqual("lifecycle_activation", failed["execution"]["failed_step"])

    def test_audit_called_only_after_success_and_audit_failure_requires_review(self) -> None:
        audit = FakeAuditWriter(exc=RuntimeError("audit down"))
        report = build_clean(audit_writer=audit)
        self.assertEqual(1, len(audit.calls))
        self.assertTrue(report["execution"]["scheduler_activation_succeeded"])
        self.assertTrue(report["execution"]["lifecycle_activation_succeeded"])
        self.assertFalse(report["execution"]["audit_write_succeeded"])
        self.assertTrue(report["rollback"]["rollback_required"])
        self.assertEqual("audit", report["execution"]["failed_step"])
        self.assertFalse(report["readiness_for_stage4i6"]["ready_to_build_scheduler_lifecycle_activation_acceptance"])

        scheduler_fail_audit = FakeAuditWriter()
        build_clean(
            scheduler_writer=FakeSchedulerWriter(exc=RuntimeError("scheduler down")),
            audit_writer=scheduler_fail_audit,
        )
        self.assertEqual([], scheduler_fail_audit.calls)

    def test_state_risk_scheduler_lifecycle_broker_market_blockers(self) -> None:
        cases = [
            {"state_snapshot": clean_state_snapshot(active_halt=True)},
            {"state_snapshot": clean_state_snapshot(unresolved_needs_reconciliation_count=1)},
            {"state_snapshot": clean_state_snapshot(active_intents_count=1)},
            {"scheduler_snapshot": {"scheduler_automation_enabled": True}},
            {"scheduler_snapshot": {"all_strategy_scheduler_enabled": True}},
            {"scheduler_snapshot": {"jobs": [{"strategy_id": "S01"}]}},
            {"lifecycle_snapshot": {"lifecycle_automation_enabled": True}},
            {"lifecycle_snapshot": {"lifecycle_transition_execution_enabled": True}},
            {"risk_snapshot": clean_risk_snapshot(risk_bypass_enabled=True)},
            {"risk_snapshot": clean_risk_snapshot(kill_switch_available=False)},
            {"risk_snapshot": clean_risk_snapshot(hard_halt_available=False)},
            {"risk_snapshot": clean_risk_snapshot(daily_loss_limit_available=False)},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(mode="LIVE")},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(ibkr_port=4002)},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(paper_trading=False)},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(live_trading_enabled=True)},
            {"paper_broker_snapshot": clean_paper_broker_snapshot(broker_submission_enabled=True)},
            {"market_window_snapshot": clean_market_window_snapshot(allowed_to_schedule_paper_run=False)},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                scheduler = FakeSchedulerWriter()
                lifecycle = FakeLifecycleWriter()
                report = build_clean(scheduler_writer=scheduler, lifecycle_writer=lifecycle, **kwargs)
                self.assert_not_attempted(report, scheduler, lifecycle)

        safe_intents = build_clean(
            state_snapshot=clean_state_snapshot(
                active_intents_count=1, active_intents_safe_for_enablement=True
            )
        )
        self.assertTrue(safe_intents["execution"]["attempted"])
        self.assertTrue(any("active intents present" in item for item in safe_intents["warnings"]))

    def test_market_window_missing_warns_and_closed_or_holiday_warns_only(self) -> None:
        missing = build(
            operator_acknowledgements=list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS),
            allow=True,
        )
        self.assertTrue(missing["execution"]["attempted"])
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, missing["warnings"])
        for key in ("market_open", "is_trading_day"):
            report = build_clean(market_window_snapshot=clean_market_window_snapshot(**{key: False}))
            self.assertTrue(report["execution"]["attempted"])
            self.assertTrue(report["warnings"])

    def test_report_fields_json_safe_and_inputs_not_mutated(self) -> None:
        inputs = {
            "i4": valid_stage4i4_report(),
            "activation_snapshot": matching_activation_snapshot(),
            "state_snapshot": clean_state_snapshot(),
            "risk_snapshot": clean_risk_snapshot(),
            "scheduler_snapshot": {},
            "lifecycle_snapshot": {},
            "paper_broker_snapshot": clean_paper_broker_snapshot(),
            "market_window_snapshot": clean_market_window_snapshot(),
            "operator_acknowledgements": list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS),
        }
        before = copy.deepcopy(inputs)
        report = build(**inputs)
        for key in (
            "dry_run",
            "stage4i5_scheduler_lifecycle_activation_executor_report",
            "generated_at",
            "gates",
            "acknowledgement_checks",
            "selected_strategy",
            "scheduler_activation_payload",
            "lifecycle_activation_payload",
            "execution",
            "applied_operations",
            "skipped_operations",
            "rollback",
            "safety",
            "readiness_for_stage4i6",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report, sort_keys=True)
        self.assertEqual(before, inputs)

    def test_cli_requires_dry_run_only_before_parsing_and_json_is_strict(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = tool.run_stage4i5_scheduler_lifecycle_activation_executor(
                ["--stage4i4-gate-json", "{bad"]
            )
        self.assertEqual(1, code)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("--dry-run-only", stderr.getvalue())

        stdout = io.StringIO()
        stderr = io.StringIO()
        args = [
            "--dry-run-only",
            "--json",
            "--stage4i4-gate-json",
            json.dumps(valid_stage4i4_report()),
        ]
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = tool.run_stage4i5_scheduler_lifecycle_activation_executor(args)
        self.assertEqual(1, code)
        self.assertEqual("", stderr.getvalue())
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4i5_scheduler_lifecycle_activation_executor_report"])

        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = tool.run_stage4i5_scheduler_lifecycle_activation_executor(
                ["--dry-run-only", "--json", "--stage4i4-gate-json", "{bad"]
            )
        self.assertEqual(1, code)
        self.assertIn("JSONDecodeError", json.loads(stdout.getvalue())["errors"][0])

    def test_cli_ack_uses_append_and_exposes_no_forbidden_action_options(self) -> None:
        parser_source = (ROOT / "tools/stage4i5_scheduler_lifecycle_activation_executor.py").read_text()
        self.assertIn('parser.add_argument("--ack", action="append"', parser_source)
        for forbidden in (
            "--submit",
            "--cancel",
            "--status",
            "--market-data",
            "--qualify",
            "--scheduler-enable",
            "--lifecycle-enable",
        ):
            self.assertNotIn(forbidden, parser_source)

    def test_no_forbidden_calls_or_production_wiring_in_stage4i5_files(self) -> None:
        source = "\n".join(path.read_text() for path in STAGE4I5_FILES)
        forbidden_patterns = [
            r"\bsubmit_order_plan\s*\(",
            r"\bget_order_status\s*\(",
            r"\bcancel_order\s*\(",
            r"\bplaceOrder\s*\(",
            r"\bcancelOrder\s*\(",
            r"\breqMktData\s*\(",
            r"\bqualifyContracts\s*\(",
            r"\bStateStore\s*\([^)]*\)\.(?:save|write|update)",
            r"\bledger\.(?:append|write)\s*\(",
            r"\bscheduler\.add_job\s*\(",
            r"\badd_job\s*\(",
            r"\bstart\s*\(",
            r"\brun_scan\s*\(",
            r"\bscan_now\s*\(",
            r"\byfinance\b",
            r"\brequests\b",
            r"\burllib\b",
            r"\bsystemctl\b",
            r"\bsystemd\b",
            r"\bsocket\.create_connection\s*\(",
            r"\bsocket\.socket\s*\(",
            r"\basyncio\.run\s*\(",
            r"\basyncio\.get_event_loop\s*\(",
            r"\basyncio\.new_event_loop\s*\(",
            r"\buuid\.uuid4\s*\(",
            r"\brandom\.",
            r"\btime\.time\s*\(",
            r"\bdatetime\.now\s*\(",
            r"\bStateStore\s*\(",
            r"\bib_insync\b",
        ]
        for pattern in forbidden_patterns:
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, source))


if __name__ == "__main__":
    unittest.main()
