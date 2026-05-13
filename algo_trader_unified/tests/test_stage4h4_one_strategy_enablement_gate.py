from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4h4_one_strategy_enablement_gate import (
    REQUIRED_OPERATOR_ACKNOWLEDGEMENTS,
    build_stage4h4_one_strategy_enablement_gate_report,
)
from algo_trader_unified.tools import stage4h4_one_strategy_enablement_gate as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4H4_FILES = [
    ROOT / "core/stage4h4_one_strategy_enablement_gate.py",
    ROOT / "tools/stage4h4_one_strategy_enablement_gate.py",
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


def valid_stage4h3_report(**overrides: object) -> dict:
    report: dict[str, object] = {
        "dry_run": True,
        "stage4h3_automation_wiring_dry_run_report": True,
        "generated_at": "2026-05-13T15:00:00+00:00",
        "selected_strategy": {
            "selected_preview_strategy_id": "S01",
            "paper_only": True,
            "enabled": False,
        },
        "dry_run_packet": {
            "available": True,
            "risk_gate_dry_run_operations": [
                operation("risk_gate_dry_run_operations", "RiskManager")
            ],
            "state_ledger_tracking_dry_run_operations": [
                operation("state_ledger_tracking_dry_run_operations", "StateStore")
            ],
            "scheduler_dry_run_operations": [
                operation(
                    "scheduler_dry_run_operations",
                    "Scheduler",
                    would_register=False,
                )
            ],
            "lifecycle_dry_run_operations": [
                operation("lifecycle_dry_run_operations", "LifecycleRouter")
            ],
            "signal_to_intent_dry_run_operations": [
                operation(
                    "signal_to_intent_dry_run_operations",
                    None,
                    target_function="build_stage4g1_lifecycle_intake_report",
                )
            ],
            "intent_to_ticket_dry_run_operations": [
                operation(
                    "intent_to_ticket_dry_run_operations",
                    None,
                    target_function="build_broker_order_request",
                )
            ],
            "ticket_to_paper_submit_dry_run_operations": [
                operation(
                    "ticket_to_paper_submit_dry_run_operations",
                    None,
                    target_function="build_ibkr_paper_order_plan",
                    would_submit=False,
                )
            ],
            "paper_broker_guard_dry_run_operations": [
                operation("paper_broker_guard_dry_run_operations", "PaperBrokerGuard")
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
        "readiness_for_stage4h4": {
            "ready_to_build_one_strategy_automation_enablement_gate": True,
            "blockers": [],
            "warnings": [],
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def operation(
    group: str,
    target_component: str | None,
    *,
    target_function: str | None = None,
    would_register: bool | None = None,
    would_submit: bool | None = None,
) -> dict:
    result: dict[str, object] = {
        "group": group,
        "strategy_id": "S01",
        "operation": f"{group}_preview",
        "payload": {"strategy_id": "S01", "paper_only": True},
        "would_execute": False,
        "paper_only": True,
        "live_trading_enabled": False,
    }
    if target_component is not None:
        result["target_component"] = target_component
    if target_function is not None:
        result["target_function"] = target_function
    if would_register is not None:
        result["would_register"] = would_register
    if would_submit is not None:
        result["would_submit"] = would_submit
    return result


def clean_risk_snapshot(**overrides: object) -> dict:
    report: dict[str, object] = {
        "kill_switch_available": True,
        "hard_halt_available": True,
        "daily_loss_limit_available": True,
        "risk_bypass_enabled": False,
    }
    _deep_update(report, overrides)
    return report


def clean_state_snapshot(**overrides: object) -> dict:
    report: dict[str, object] = {
        "active_halt": False,
        "unresolved_needs_reconciliation_count": 0,
        "active_intents_count": 0,
        "open_positions_count": 2,
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
    stage4h3_report: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker: dict | None = None,
    acks: list[str] | None = None,
) -> dict:
    return build_stage4h4_one_strategy_enablement_gate_report(
        stage4h3_wiring_dry_run_report=valid_stage4h3_report()
        if stage4h3_report is None
        else stage4h3_report,
        state_snapshot=clean_state_snapshot() if state_snapshot is None else state_snapshot,
        risk_snapshot=clean_risk_snapshot() if risk_snapshot is None else risk_snapshot,
        scheduler_snapshot={} if scheduler_snapshot is None else scheduler_snapshot,
        lifecycle_snapshot={} if lifecycle_snapshot is None else lifecycle_snapshot,
        paper_broker_snapshot=paper_broker_snapshot() if paper_broker is None else paper_broker,
        operator_acknowledgements=list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS) if acks is None else acks,
        now_provider=lambda: datetime(2026, 5, 13, 16, 0, tzinfo=timezone.utc),
    )


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


def assert_not_ready(test_case: unittest.TestCase, report: dict) -> None:
    test_case.assertFalse(
        report["readiness_for_stage4h5"][
            "ready_to_build_one_strategy_activation_executor"
        ]
    )


class Stage4H4OneStrategyEnablementGateTests(unittest.TestCase):
    def test_missing_stage4h3_dry_run_report_blocks_readiness(self) -> None:
        report = build_stage4h4_one_strategy_enablement_gate_report(
            stage4h3_wiring_dry_run_report=None
        )
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["stage4h3_dry_run_present"])
        self.assertTrue(report["success"])

    def test_stage4h3_dry_run_not_ready_blocks_readiness(self) -> None:
        report = build(
            stage4h3_report=valid_stage4h3_report(
                readiness_for_stage4h4={
                    "ready_to_build_one_strategy_automation_enablement_gate": False
                }
            )
        )
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["stage4h3_dry_run_ready"])

    def test_missing_selected_strategy_blocks_readiness(self) -> None:
        report = build(
            stage4h3_report=valid_stage4h3_report(
                selected_strategy={"selected_preview_strategy_id": ""}
            )
        )
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["selected_strategy_present"])

    def test_selected_strategy_is_read_from_stage4h3_selected_strategy_block(self) -> None:
        report = build(
            stage4h3_report=valid_stage4h3_report(
                selected_strategy={"selected_preview_strategy_id": "S77"},
            )
        )
        self.assertTrue(
            report["readiness_for_stage4h5"][
                "ready_to_build_one_strategy_activation_executor"
            ]
        )
        self.assertEqual("S77", report["selected_strategy"]["selected_strategy_id"])

    def test_operator_acknowledgements_none_is_safely_coerced_to_empty_list(self) -> None:
        report = build_stage4h4_one_strategy_enablement_gate_report(
            stage4h3_wiring_dry_run_report=valid_stage4h3_report(),
            paper_broker_snapshot=paper_broker_snapshot(),
            operator_acknowledgements=None,
        )
        assert_not_ready(self, report)
        self.assertEqual([], report["acknowledgement_checks"]["provided"])
        self.assertEqual(
            REQUIRED_OPERATOR_ACKNOWLEDGEMENTS,
            report["acknowledgement_checks"]["missing"],
        )

    def test_missing_acknowledgements_block_readiness_but_report_renders(self) -> None:
        report = build(acks=[])
        assert_not_ready(self, report)
        self.assertTrue(report["success"])
        self.assertTrue(report["acknowledgement_checks"]["missing"])

    def test_giant_substring_acknowledgement_does_not_pass(self) -> None:
        report = build(acks=[" ".join(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS)])
        assert_not_ready(self, report)
        self.assertFalse(report["acknowledgement_checks"]["exact_match"])

    def test_extra_acknowledgements_do_not_compensate_for_missing_required(self) -> None:
        report = build(acks=REQUIRED_OPERATOR_ACKNOWLEDGEMENTS[:-1] + ["extra"])
        assert_not_ready(self, report)
        self.assertIn(
            REQUIRED_OPERATOR_ACKNOWLEDGEMENTS[-1],
            report["acknowledgement_checks"]["missing"],
        )

    def test_exact_acknowledgements_pass_acknowledgement_gate(self) -> None:
        report = build(acks=["  " + item + "  " for item in REQUIRED_OPERATOR_ACKNOWLEDGEMENTS])
        self.assertTrue(report["acknowledgement_checks"]["exact_match"])
        self.assertEqual([], report["acknowledgement_checks"]["missing"])

    def test_missing_paper_broker_snapshot_blocks_readiness_for_stage4h5(self) -> None:
        report = build_stage4h4_one_strategy_enablement_gate_report(
            stage4h3_wiring_dry_run_report=valid_stage4h3_report(),
            state_snapshot=clean_state_snapshot(),
            risk_snapshot=clean_risk_snapshot(),
            scheduler_snapshot={},
            lifecycle_snapshot={},
            paper_broker_snapshot=None,
            operator_acknowledgements=list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS),
        )
        assert_not_ready(self, report)
        self.assertFalse(report["paper_broker_checks"]["paper_broker_snapshot_present"])

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
                self.assertFalse(report["paper_broker_checks"]["paper_config_valid"])

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

    def test_state_snapshot_blockers_and_safe_active_intents_warning(self) -> None:
        assert_not_ready(self, build(state_snapshot=clean_state_snapshot(active_halt=True)))
        assert_not_ready(
            self,
            build(state_snapshot=clean_state_snapshot(unresolved_needs_reconciliation_count=1)),
        )
        assert_not_ready(self, build(state_snapshot=clean_state_snapshot(active_intents_count=1)))
        assert_not_ready(
            self,
            build(
                state_snapshot=clean_state_snapshot(
                    active_intents_count=1,
                    active_intents_safe_for_enablement=False,
                )
            ),
        )
        warned = build(
            state_snapshot=clean_state_snapshot(
                active_intents_count=1,
                active_intents_safe_for_enablement=True,
            )
        )
        self.assertTrue(
            warned["readiness_for_stage4h5"][
                "ready_to_build_one_strategy_activation_executor"
            ]
        )
        self.assertIn("active intents", " ".join(warned["warnings"]))

    def test_open_positions_count_alone_does_not_block(self) -> None:
        report = build(state_snapshot=clean_state_snapshot(open_positions_count=9))
        self.assertTrue(
            report["readiness_for_stage4h5"][
                "ready_to_build_one_strategy_activation_executor"
            ]
        )

    def test_scheduler_and_lifecycle_already_enabled_block_readiness(self) -> None:
        assert_not_ready(self, build(scheduler_snapshot={"scheduler_automation_enabled": True}))
        assert_not_ready(self, build(lifecycle_snapshot={"lifecycle_automation_enabled": True}))

    def test_selected_strategy_job_already_enabled_blocks_unless_disabled_dry_run_only(self) -> None:
        blocked = build(scheduler_snapshot={"jobs": [{"strategy_id": "S01", "disabled": False}]})
        assert_not_ready(self, blocked)
        allowed = build(
            scheduler_snapshot={
                "jobs": [{"strategy_id": "S01", "disabled": True, "dry_run_only": True}]
            }
        )
        self.assertTrue(
            allowed["readiness_for_stage4h5"][
                "ready_to_build_one_strategy_activation_executor"
            ]
        )

    def test_unsafe_flags_block_readiness(self) -> None:
        cases = [
            ("all_strategies_enabled", "no_all_strategy_enablement"),
            ("live_trading_enabled", "no_live_orders"),
            ("automated_paper_trading_enabled", "no_automated_paper_trading_enabled"),
            ("market_data_enabled", "no_market_data"),
            ("contract_qualification_enabled", "no_contract_qualification"),
        ]
        for flag, check in cases:
            with self.subTest(flag=flag):
                report = build(stage4h3_report=valid_stage4h3_report(**{flag: True}))
                assert_not_ready(self, report)
                self.assertFalse(report["safety_checks"][check])

    def test_dry_run_operation_schema_blockers(self) -> None:
        missing_target = valid_stage4h3_report()
        missing_target["dry_run_packet"]["risk_gate_dry_run_operations"][0].pop(
            "target_component"
        )
        assert_not_ready(self, build(stage4h3_report=missing_target))

        bad_payload = valid_stage4h3_report()
        bad_payload["dry_run_packet"]["risk_gate_dry_run_operations"][0]["payload"] = "bad"
        assert_not_ready(self, build(stage4h3_report=bad_payload))

        would_execute = valid_stage4h3_report()
        would_execute["dry_run_packet"]["risk_gate_dry_run_operations"][0][
            "would_execute"
        ] = True
        assert_not_ready(self, build(stage4h3_report=would_execute))

        would_register = valid_stage4h3_report()
        would_register["dry_run_packet"]["scheduler_dry_run_operations"][0][
            "would_register"
        ] = True
        assert_not_ready(self, build(stage4h3_report=would_register))

        would_submit = valid_stage4h3_report()
        would_submit["dry_run_packet"]["ticket_to_paper_submit_dry_run_operations"][0][
            "would_submit"
        ] = True
        assert_not_ready(self, build(stage4h3_report=would_submit))

    def test_activation_candidate_and_proposed_flags_are_tightly_limited(self) -> None:
        report = build()
        candidate = report["activation_candidate"]
        flags = report["proposed_activation_flags"]
        self.assertTrue(candidate["one_strategy_only"])
        self.assertTrue(candidate["paper_only"])
        self.assertEqual(1, candidate["max_enabled_strategy_count"])
        self.assertFalse(candidate["live_trading_enabled"])
        self.assertFalse(candidate["all_strategies_enabled"])
        self.assertFalse(candidate["broker_submission_allowed_next_phase"])
        self.assertTrue(flags["enable_automated_paper_trading_for_selected_strategy"])
        self.assertTrue(flags["enable_scheduler_for_selected_strategy"])
        self.assertTrue(flags["enable_lifecycle_for_selected_strategy"])
        self.assertFalse(flags["enable_broker_submission_for_selected_strategy"])
        self.assertFalse(flags["enable_live_trading"])
        self.assertFalse(flags["enable_all_strategies"])

    def test_proposed_scheduler_and_lifecycle_activation_are_future_stage_only(self) -> None:
        report = build()
        job = report["proposed_scheduler_activation"]["jobs"][0]
        flow = report["proposed_lifecycle_activation"]["flows"][0]
        self.assertEqual("S01", job["strategy_id"])
        self.assertTrue(job["paper_only"])
        self.assertTrue(job["future_enablement_only"])
        self.assertFalse(job["would_register_in_4h4"])
        self.assertTrue(job["proposed_enabled_in_4h5"])
        self.assertEqual("S01", flow["strategy_id"])
        self.assertTrue(flow["paper_only"])
        self.assertTrue(flow["future_enablement_only"])
        self.assertFalse(flow["would_execute_in_4h4"])
        self.assertTrue(flow["proposed_enabled_in_4h5"])

    def test_report_includes_required_top_level_fields_and_is_json_safe(self) -> None:
        report = build()
        for key in (
            "dry_run",
            "stage4h4_one_strategy_enablement_gate_report",
            "generated_at",
            "artifact_checks",
            "selected_strategy",
            "acknowledgement_checks",
            "activation_candidate",
            "proposed_activation_flags",
            "proposed_scheduler_activation",
            "proposed_lifecycle_activation",
            "proposed_runtime_guards",
            "proposed_monitoring_requirements",
            "proposed_kill_switch_requirements",
            "safety_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "risk_checks",
            "state_checks",
            "paper_broker_checks",
            "readiness_for_stage4h5",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report, sort_keys=True)

    def test_input_reports_and_snapshots_are_not_mutated(self) -> None:
        h3 = valid_stage4h3_report()
        state = clean_state_snapshot()
        risk = clean_risk_snapshot()
        paper = paper_broker_snapshot()
        before = copy.deepcopy((h3, state, risk, paper))
        build(
            stage4h3_report=h3,
            state_snapshot=state,
            risk_snapshot=risk,
            paper_broker=paper,
        )
        self.assertEqual(before, (h3, state, risk, paper))

    def test_cli_requires_dry_run_only_before_parsing(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = tool.run_stage4h4_one_strategy_enablement_gate(
                ["--json", "--stage4h3-dry-run-json", "{"]
            )
        self.assertEqual(1, code)
        self.assertIn("--dry-run-only", stderr.getvalue())

    def test_cli_ack_uses_action_append(self) -> None:
        source = Path(tool.__file__).read_text()
        self.assertIn('parser.add_argument("--ack", action="append"', source)

    def test_cli_json_stdout_is_strict_json(self) -> None:
        stdout = io.StringIO()
        args = [
            "--dry-run-only",
            "--json",
            "--stage4h3-dry-run-json",
            json.dumps(valid_stage4h3_report()),
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
            code = tool.run_stage4h4_one_strategy_enablement_gate(args)
        self.assertEqual(0, code)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4h4_one_strategy_enablement_gate_report"])

    def test_cli_exposes_no_execution_actions(self) -> None:
        parser_source = Path(tool.__file__).read_text()
        disallowed_options = [
            "--submit",
            "--cancel",
            "--status",
            "--market-data",
            "--qualification",
            "--qualify",
            "--state-write",
            "--ledger-write",
            "--scheduler-enable",
            "--lifecycle-enable",
        ]
        for option in disallowed_options:
            self.assertNotIn(option, parser_source)

    def test_stage4h4_source_has_no_forbidden_runtime_call_tokens(self) -> None:
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
            "ib_insync",
        ]
        for path in STAGE4H4_FILES:
            source = path.read_text()
            for token in forbidden:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_no_daemon_scheduler_or_lifecycle_runtime_wiring_exists(self) -> None:
        for path in UNWIRED_RUNTIME_FILES:
            if not path.exists():
                continue
            source = path.read_text()
            self.assertNotIn("stage4h4_one_strategy_enablement_gate", source)


if __name__ == "__main__":
    unittest.main()
