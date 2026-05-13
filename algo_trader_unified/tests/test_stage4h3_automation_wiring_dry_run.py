from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4h3_automation_wiring_dry_run import (
    _operation_checks,
    build_stage4h3_automation_wiring_dry_run_report,
)
from algo_trader_unified.tools import stage4h3_automation_wiring_dry_run as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4H3_FILES = [
    ROOT / "core/stage4h3_automation_wiring_dry_run.py",
    ROOT / "tools/stage4h3_automation_wiring_dry_run.py",
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


def valid_stage4h2_report(**overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4h2_automation_wiring_preview_report": True,
        "generated_at": "2026-05-13T14:00:00+00:00",
        "strategy_selection": {
            "candidate_strategy_ids": ["S01"],
            "selected_preview_strategy_id": "S01",
            "single_strategy_selected": True,
        },
        "wiring_preview": {
            "available": True,
            "proposed_scheduler_wiring_preview": {
                "jobs": [
                    {
                        "job_id": "stage4h3_dry_run_S01",
                        "strategy_id": "S01",
                        "trigger_description": "operator-controlled Stage 4H-3 dry-run cadence preview",
                        "disabled": True,
                        "would_register": False,
                        "would_execute": False,
                        "paper_only": True,
                    }
                ]
            },
            "proposed_lifecycle_wiring_preview": {
                "flows": [
                    {"name": "signal_to_intent", "would_execute": False, "paper_only": True},
                    {"name": "intent_to_ticket", "would_execute": False, "paper_only": True},
                    {"name": "ticket_to_paper_submit", "would_execute": False, "paper_only": True},
                ]
            },
            "proposed_risk_gates": [
                {"name": "kill_switch_check", "would_execute": False},
                {"name": "hard_halt_check", "would_execute": False},
            ],
        },
        "safety_checks": {
            "no_live_orders": True,
            "no_market_data": True,
            "no_contract_qualification": True,
            "no_broker_submission_enabled": True,
            "no_scheduler_wiring_enabled": True,
            "no_lifecycle_wiring_enabled": True,
            "no_automated_paper_trading_enabled": True,
            "no_daemon_wiring_enabled": True,
            "no_all_strategy_enablement": True,
        },
        "readiness_for_stage4h3": {
            "ready_to_build_automation_wiring_dry_run": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
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
        "account_type": "PAPER",
        "live_trading_enabled": False,
        "broker_submission_enabled": False,
    }
    _deep_update(report, overrides)
    return report


def build(
    *,
    stage4h2_report: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker: dict | None = None,
) -> dict:
    return build_stage4h3_automation_wiring_dry_run_report(
        stage4h2_wiring_preview_report=valid_stage4h2_report()
        if stage4h2_report is None
        else stage4h2_report,
        state_snapshot=state_snapshot,
        risk_snapshot=risk_snapshot,
        scheduler_snapshot=scheduler_snapshot,
        lifecycle_snapshot=lifecycle_snapshot,
        paper_broker_snapshot=paper_broker,
        now_provider=lambda: datetime(2026, 5, 13, 15, 0, tzinfo=timezone.utc),
    )


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


def assert_not_ready(test_case: unittest.TestCase, report: dict) -> None:
    test_case.assertFalse(
        report["readiness_for_stage4h4"][
            "ready_to_build_one_strategy_automation_enablement_gate"
        ]
    )
    test_case.assertFalse(report["success"])


def flat_operations(report: dict) -> list[dict]:
    packet = report["dry_run_packet"]
    keys = [
        "risk_gate_dry_run_operations",
        "state_ledger_tracking_dry_run_operations",
        "scheduler_dry_run_operations",
        "lifecycle_dry_run_operations",
        "signal_to_intent_dry_run_operations",
        "intent_to_ticket_dry_run_operations",
        "ticket_to_paper_submit_dry_run_operations",
        "paper_broker_guard_dry_run_operations",
    ]
    result: list[dict] = []
    for key in keys:
        result.extend(packet[key])
    return sorted(result, key=lambda item: item["sequence_number"])


class Stage4H3AutomationWiringDryRunTests(unittest.TestCase):
    def test_missing_stage4h2_preview_report_blocks_readiness(self) -> None:
        report = build_stage4h3_automation_wiring_dry_run_report(
            stage4h2_wiring_preview_report=None
        )
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["stage4h2_preview_present"])

    def test_stage4h2_preview_not_ready_blocks_readiness(self) -> None:
        report = build(
            stage4h2_report=valid_stage4h2_report(
                readiness_for_stage4h3={
                    "ready_to_build_automation_wiring_dry_run": False
                }
            )
        )
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["stage4h2_preview_ready"])

    def test_missing_selected_preview_strategy_id_blocks_readiness(self) -> None:
        report = build(
            stage4h2_report=valid_stage4h2_report(
                strategy_selection={"selected_preview_strategy_id": ""}
            )
        )
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["selected_strategy_present"])

    def test_selected_strategy_is_read_from_stage4h2_strategy_selection_only(self) -> None:
        report = build(
            stage4h2_report=valid_stage4h2_report(
                strategy_selection={"selected_preview_strategy_id": "S01"},
                strategy_registry_snapshot={"preview_strategy_id": "S99"},
            ),
            risk_snapshot=clean_risk_snapshot(),
        )
        self.assertTrue(report["success"])
        self.assertEqual("S01", report["selected_strategy"]["selected_preview_strategy_id"])

    def test_missing_wiring_preview_available_blocks_readiness(self) -> None:
        report = build(stage4h2_report=valid_stage4h2_report(wiring_preview={"available": False}))
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["wiring_preview_available"])

    def test_flat_string_scheduler_preview_blocks_readiness(self) -> None:
        report = build(
            stage4h2_report=valid_stage4h2_report(
                wiring_preview={
                    "available": True,
                    "proposed_scheduler_wiring_preview": "flat",
                    "proposed_lifecycle_wiring_preview": {"flows": []},
                }
            )
        )
        assert_not_ready(self, report)

    def test_flat_string_lifecycle_preview_blocks_readiness(self) -> None:
        report = build(
            stage4h2_report=valid_stage4h2_report(
                wiring_preview={
                    "available": True,
                    "proposed_scheduler_wiring_preview": {"jobs": []},
                    "proposed_lifecycle_wiring_preview": "flat",
                }
            )
        )
        assert_not_ready(self, report)

    def test_malformed_scheduler_job_entries_do_not_crash_and_block(self) -> None:
        report = build(
            stage4h2_report=valid_stage4h2_report(
                wiring_preview={
                    "available": True,
                    "proposed_scheduler_wiring_preview": {"jobs": [None, "bad"]},
                    "proposed_lifecycle_wiring_preview": {"flows": [{"name": "x", "would_execute": False}]},
                }
            )
        )
        assert_not_ready(self, report)

    def test_malformed_lifecycle_flow_entries_do_not_crash_and_block(self) -> None:
        report = build(
            stage4h2_report=valid_stage4h2_report(
                wiring_preview={
                    "available": True,
                    "proposed_scheduler_wiring_preview": {
                        "jobs": [{"job_id": "j", "strategy_id": "S01", "disabled": True, "would_register": False, "would_execute": False}]
                    },
                    "proposed_lifecycle_wiring_preview": {"flows": [None, "bad"]},
                }
            )
        )
        assert_not_ready(self, report)

    def test_valid_structured_stage4h2_preview_builds_dry_run_packet(self) -> None:
        report = build(risk_snapshot=clean_risk_snapshot(), paper_broker=paper_broker_snapshot())
        self.assertTrue(report["success"])
        self.assertTrue(report["dry_run_packet"]["available"])
        self.assertTrue(report["dry_run_packet"]["scheduler_dry_run_operations"])
        self.assertTrue(report["dry_run_packet"]["final_enablement_gate_preview"])

    def test_every_operation_has_target_and_payload_schema(self) -> None:
        report = build(risk_snapshot=clean_risk_snapshot())
        for operation in flat_operations(report):
            self.assertTrue(operation.get("target_function") or operation.get("target_component"))
            self.assertIsInstance(operation.get("payload"), dict)

    def test_operation_checks_detect_missing_target_or_payload(self) -> None:
        checks = _operation_checks(
            {
                "scheduler_dry_run_operations": [
                    {"sequence_number": 1, "would_execute": False, "would_register": False, "payload": {}}
                ],
                "ticket_to_paper_submit_dry_run_operations": [
                    {"sequence_number": 2, "would_execute": False, "would_submit": False, "target_function": "build_ibkr_paper_order_plan"}
                ],
            }
        )
        self.assertFalse(checks["target_schema_valid"])
        self.assertFalse(checks["payload_schema_valid"])

    def test_all_dry_run_operations_are_disabled(self) -> None:
        report = build(risk_snapshot=clean_risk_snapshot())
        self.assertTrue(report["operation_checks"]["all_operations_would_execute_false"])
        for operation in flat_operations(report):
            self.assertFalse(operation["would_execute"])
            self.assertTrue(operation["paper_only"])
            self.assertFalse(operation["live_trading_enabled"])

    def test_scheduler_operations_have_disabled_register_and_execute_false(self) -> None:
        report = build(risk_snapshot=clean_risk_snapshot())
        for operation in report["dry_run_packet"]["scheduler_dry_run_operations"]:
            self.assertTrue(operation["payload"]["disabled"])
            self.assertFalse(operation["would_register"])
            self.assertFalse(operation["would_execute"])

    def test_broker_order_operations_have_would_submit_false(self) -> None:
        report = build(risk_snapshot=clean_risk_snapshot())
        for operation in report["dry_run_packet"]["ticket_to_paper_submit_dry_run_operations"]:
            self.assertFalse(operation["would_submit"])

    def test_dry_run_operation_ordering_is_deterministic(self) -> None:
        first = [op["sequence_number"] for op in flat_operations(build())]
        second = [op["sequence_number"] for op in flat_operations(build())]
        self.assertEqual(first, second)
        self.assertEqual(list(range(1, len(first) + 1)), first)

    def test_scheduler_snapshot_already_enabled_blocks_readiness(self) -> None:
        report = build(scheduler_snapshot={"scheduler_automation_enabled": True})
        assert_not_ready(self, report)
        self.assertTrue(report["scheduler_checks"]["scheduler_already_enabled"])

    def test_lifecycle_snapshot_already_enabled_blocks_readiness(self) -> None:
        report = build(lifecycle_snapshot={"lifecycle_automation_enabled": True})
        assert_not_ready(self, report)
        self.assertTrue(report["lifecycle_checks"]["lifecycle_already_enabled"])

    def test_selected_strategy_job_already_enabled_blocks_unless_disabled_dry_run_only(self) -> None:
        blocked = build(scheduler_snapshot={"jobs": [{"strategy_id": "S01", "disabled": False}]})
        assert_not_ready(self, blocked)
        allowed = build(
            scheduler_snapshot={"jobs": [{"strategy_id": "S01", "disabled": True, "dry_run_only": True}]},
            risk_snapshot=clean_risk_snapshot(),
        )
        self.assertTrue(allowed["success"])

    def test_risk_snapshot_blockers(self) -> None:
        cases = [
            ("risk_bypass_enabled", True),
            ("kill_switch_available", False),
            ("hard_halt_available", False),
            ("daily_loss_limit_available", False),
        ]
        for key, value in cases:
            with self.subTest(key=key):
                report = build(risk_snapshot=clean_risk_snapshot(**{key: value}))
                assert_not_ready(self, report)

    def test_missing_optional_snapshots_warn_without_crashing(self) -> None:
        report = build()
        self.assertIn("risk snapshot missing", " ".join(report["warnings"]))
        self.assertIn("state snapshot missing", " ".join(report["warnings"]))
        self.assertIn("paper broker snapshot missing", " ".join(report["warnings"]))

    def test_state_snapshot_blockers_and_warnings(self) -> None:
        assert_not_ready(self, build(state_snapshot={"active_halt": True}))
        assert_not_ready(self, build(state_snapshot={"unresolved_needs_reconciliation_count": 1}))
        warn = build(state_snapshot={"active_intents_count": 1}, risk_snapshot=clean_risk_snapshot())
        self.assertTrue(warn["success"])
        self.assertIn("active intents", " ".join(warn["warnings"]))
        open_only = build(state_snapshot={"open_positions_count": 3}, risk_snapshot=clean_risk_snapshot())
        self.assertTrue(open_only["success"])

    def test_paper_broker_snapshot_blockers(self) -> None:
        cases = [
            {"mode": "LIVE"},
            {"paper_trading": False},
            {"ibkr_port": 4002},
            {"live_trading_enabled": True},
            {"broker_submission_enabled": True},
        ]
        for case in cases:
            with self.subTest(case=case):
                report = build(paper_broker=paper_broker_snapshot(**case))
                assert_not_ready(self, report)

    def test_unsafe_flags_block_readiness(self) -> None:
        cases = [
            ("live_trading_enabled", "no_live_orders"),
            ("automated_paper_trading_enabled", "no_automated_paper_trading_enabled"),
            ("broker_submission_enabled", "no_broker_submission_enabled"),
            ("all_strategies_enabled", "no_all_strategy_enablement"),
            ("market_data_enabled", "no_market_data"),
            ("contract_qualification_enabled", "no_contract_qualification"),
        ]
        for flag, check in cases:
            with self.subTest(flag=flag):
                report = build(stage4h2_report=valid_stage4h2_report(**{flag: True}))
                assert_not_ready(self, report)
                self.assertFalse(report["safety_checks"][check])

    def test_recommendations_do_not_include_disallowed_enablement(self) -> None:
        report = build()
        text = json.dumps(report["recommendations"]).lower()
        self.assertNotIn("all-strategy automation", text)
        self.assertNotIn("place orders now.", " ".join(report["recommendations"]["ordered_next_steps"]).lower())

    def test_report_includes_required_top_level_fields_and_is_json_safe(self) -> None:
        report = build(risk_snapshot=clean_risk_snapshot(), paper_broker=paper_broker_snapshot())
        for key in (
            "dry_run",
            "stage4h3_automation_wiring_dry_run_report",
            "generated_at",
            "artifact_checks",
            "selected_strategy",
            "dry_run_packet",
            "operation_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "risk_checks",
            "state_checks",
            "paper_broker_checks",
            "safety_checks",
            "readiness_for_stage4h4",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report, sort_keys=True)

    def test_input_reports_are_not_mutated(self) -> None:
        h2 = valid_stage4h2_report()
        risk = clean_risk_snapshot()
        before = copy.deepcopy((h2, risk))
        build(stage4h2_report=h2, risk_snapshot=risk)
        self.assertEqual(before, (h2, risk))

    def test_cli_requires_dry_run_only_before_parsing(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_stage4h3_automation_wiring_dry_run(
                ["--json", "--stage4h2-preview-json", "{"]
            )
        self.assertEqual(1, code)
        self.assertIn("--dry-run-only", stderr.getvalue())

    def test_cli_json_stdout_is_strict_json(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = tool.run_stage4h3_automation_wiring_dry_run(
                [
                    "--dry-run-only",
                    "--json",
                    "--stage4h2-preview-json",
                    json.dumps(valid_stage4h2_report()),
                    "--risk-snapshot-json",
                    json.dumps(clean_risk_snapshot()),
                ]
            )
        self.assertEqual(0, code)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4h3_automation_wiring_dry_run_report"])

    def test_cli_exposes_no_execution_actions(self) -> None:
        parser_source = Path(tool.__file__).read_text()
        disallowed_options = [
            "--submit",
            "--cancel",
            "--status",
            "--market-data",
            "--qualify",
            "--state-write",
            "--ledger-write",
            "--scheduler-enable",
        ]
        for option in disallowed_options:
            self.assertNotIn(option, parser_source)

    def test_stage4h3_source_has_no_forbidden_runtime_call_tokens(self) -> None:
        forbidden = [
            ".submit_order_plan(",
            "submit_order_plan(",
            ".get_order_status(",
            "get_order_status(",
            ".cancel_order(",
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
        ]
        for path in STAGE4H3_FILES:
            source = path.read_text()
            for token in forbidden:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_no_daemon_scheduler_or_lifecycle_runtime_wiring_changed(self) -> None:
        for path in UNWIRED_RUNTIME_FILES:
            if not path.exists():
                continue
            source = path.read_text()
            self.assertNotIn("stage4h3_automation_wiring_dry_run", source)


if __name__ == "__main__":
    unittest.main()
