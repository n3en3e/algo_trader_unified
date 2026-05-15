from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
import json
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4i4_scheduler_lifecycle_activation_gate import (
    MARKET_WINDOW_MANUAL_WARNING,
    REQUIRED_OPERATOR_ACKNOWLEDGEMENTS,
    build_stage4i4_scheduler_lifecycle_activation_gate_report,
)
from algo_trader_unified.tools import stage4i4_scheduler_lifecycle_activation_gate as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4I4_FILES = [
    ROOT / "core/stage4i4_scheduler_lifecycle_activation_gate.py",
    ROOT / "tools/stage4i4_scheduler_lifecycle_activation_gate.py",
]


def valid_stage4i3_report(**overrides: object) -> dict:
    trace = [
        {
            "sequence_number": 1,
            "source_stage": "pre_run_snapshot_check",
            "target_component": "pre_run_snapshot_check_component",
            "dry_run_result": {"simulated_pass": True},
            "would_execute": False,
            "would_submit": False,
            "would_write_state": False,
            "would_write_ledger": False,
            "would_register_scheduler": False,
            "paper_only": True,
            "live_trading_enabled": False,
            "status": "simulated",
        },
        {
            "sequence_number": 2,
            "source_stage": "strategy_scan_preview",
            "target_component": "strategy_preview_component",
            "dry_run_result": {
                "would_call_strategy_scan": False,
                "would_create_intent": False,
                "would_create_ticket": False,
                "would_submit_order": False,
            },
            "would_execute": False,
            "would_write_state": False,
            "would_write_ledger": False,
            "would_register_scheduler": False,
            "paper_only": True,
            "live_trading_enabled": False,
            "status": "simulated",
        },
    ]
    report: dict[str, object] = {
        "dry_run": True,
        "stage4i3_scheduled_run_dry_run_report": True,
        "generated_at": "2026-05-15T12:00:00+00:00",
        "artifact_checks": {
            "stage4i3_report_present": True,
            "stage4i3_report_ready": True,
            "selected_strategy_present": True,
            "dry_run_trace_present": True,
            "dry_run_trace_clean": True,
        },
        "selected_strategy": {
            "selected_strategy_id": "S01",
            "paper_only": True,
            "one_strategy_only": True,
        },
        "dry_run_trace": trace,
        "dry_run_trace_checks": {
            "trace_available": True,
            "trace_order_matches_plan": True,
            "all_trace_items_simulated": True,
            "no_strategy_scan_called": True,
            "no_intent_created": True,
            "no_ticket_created": True,
            "no_broker_submission": True,
            "no_state_write": True,
            "no_ledger_write": True,
            "no_scheduler_registration": True,
            "no_lifecycle_execution": True,
        },
        "safety_checks": {
            "no_live_trading": True,
            "no_all_strategy_enablement": True,
            "no_broker_submission_enabled": True,
            "no_market_data": True,
            "no_contract_qualification": True,
            "no_order_submission": True,
            "no_strategy_scan_execution": True,
            "no_scheduler_registration": True,
            "no_lifecycle_execution": True,
            "no_state_write": True,
            "no_ledger_write": True,
        },
        "readiness_for_stage4i4": {
            "ready_to_build_scheduler_activation_gate": True,
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
        "reason": "operator supplied clean paper run window",
    }
    _deep_update(snapshot, overrides)
    return snapshot


def build(
    *,
    i3: dict | None = None,
    activation_snapshot: dict | None = None,
    state_snapshot: dict | None = None,
    risk_snapshot: dict | None = None,
    scheduler_snapshot: dict | None = None,
    lifecycle_snapshot: dict | None = None,
    paper_broker_snapshot: dict | None = None,
    market_window_snapshot: dict | None = None,
    operator_acknowledgements: list[str] | None = None,
) -> dict:
    return build_stage4i4_scheduler_lifecycle_activation_gate_report(
        stage4i3_dry_run_report=valid_stage4i3_report() if i3 is None else i3,
        activation_snapshot=activation_snapshot,
        state_snapshot=state_snapshot,
        risk_snapshot=risk_snapshot,
        scheduler_snapshot=scheduler_snapshot,
        lifecycle_snapshot=lifecycle_snapshot,
        paper_broker_snapshot=paper_broker_snapshot,
        market_window_snapshot=market_window_snapshot,
        operator_acknowledgements=operator_acknowledgements,
        now_provider=lambda: datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc),
    )


def build_with_all_snapshots(**kwargs: object) -> dict:
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


def assert_ready(test_case: unittest.TestCase, report: dict) -> None:
    test_case.assertTrue(
        report["readiness_for_stage4i5"][
            "ready_to_build_scheduler_lifecycle_activation_executor"
        ]
    )


def assert_not_ready(test_case: unittest.TestCase, report: dict) -> None:
    test_case.assertFalse(
        report["readiness_for_stage4i5"][
            "ready_to_build_scheduler_lifecycle_activation_executor"
        ]
    )
    test_case.assertTrue(report["readiness_for_stage4i5"]["blockers"])


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


class Stage4I4SchedulerLifecycleActivationGateTests(unittest.TestCase):
    def test_missing_stage4i3_report_blocks_readiness(self) -> None:
        report = build_stage4i4_scheduler_lifecycle_activation_gate_report(
            stage4i3_dry_run_report=None
        )
        assert_not_ready(self, report)

    def test_stage4i3_report_not_ready_blocks_readiness(self) -> None:
        report = build_with_all_snapshots(
            i3=valid_stage4i3_report(
                readiness_for_stage4i4={"ready_to_build_scheduler_activation_gate": False}
            )
        )
        assert_not_ready(self, report)

    def test_missing_selected_strategy_blocks_and_safe_traversal(self) -> None:
        report = build_with_all_snapshots(i3=valid_stage4i3_report(selected_strategy="bad"))
        assert_not_ready(self, report)
        self.assertIsNone(report["selected_strategy"]["selected_strategy_id"])

    def test_missing_dry_run_trace_blocks_without_keyerror(self) -> None:
        i3 = valid_stage4i3_report()
        i3.pop("dry_run_trace")
        report = build_with_all_snapshots(i3=i3)
        assert_not_ready(self, report)
        self.assertFalse(report["artifact_checks"]["dry_run_trace_present"])

    def test_source_uses_safe_dry_run_trace_default(self) -> None:
        source = (ROOT / "core/stage4i4_scheduler_lifecycle_activation_gate.py").read_text()
        self.assertIn('data.get("dry_run_trace", [])', source)
        self.assertIsNone(re.search(r"\[[\"']dry_run_trace[\"']\]", source))

    def test_dry_run_trace_unsafe_items_and_strict_false_fields_block(self) -> None:
        cases = [
            ("malformed", "bad"),
            ("would_execute_true", {"would_execute": True}),
            ("would_execute_string", {"would_execute": "False"}),
            ("would_submit_true", {"would_submit": True}),
            ("would_submit_string", {"would_submit": "False"}),
            ("would_write_state_true", {"would_write_state": True}),
            ("would_write_state_string", {"would_write_state": "False"}),
            ("would_write_ledger_true", {"would_write_ledger": True}),
            ("would_write_ledger_string", {"would_write_ledger": "False"}),
            ("would_register_scheduler_true", {"would_register_scheduler": True}),
            ("would_register_scheduler_string", {"would_register_scheduler": "False"}),
        ]
        for label, replacement in cases:
            with self.subTest(label=label):
                i3 = valid_stage4i3_report()
                if isinstance(replacement, dict):
                    i3["dry_run_trace"][0].update(replacement)  # type: ignore[index,union-attr]
                else:
                    i3["dry_run_trace"][0] = replacement  # type: ignore[index]
                assert_not_ready(self, build_with_all_snapshots(i3=i3))

    def test_safety_checks_values_as_strings_do_not_pass(self) -> None:
        for key in (
            "no_live_trading",
            "no_all_strategy_enablement",
            "no_broker_submission_enabled",
            "no_market_data",
            "no_contract_qualification",
            "no_order_submission",
            "no_strategy_scan_execution",
            "no_scheduler_registration",
            "no_lifecycle_execution",
            "no_state_write",
            "no_ledger_write",
        ):
            with self.subTest(key=key):
                report = build_with_all_snapshots(
                    i3=valid_stage4i3_report(safety_checks={key: "True"})
                )
                assert_not_ready(self, report)
        assert_not_ready(
            self,
            build_with_all_snapshots(
                paper_broker_snapshot=clean_paper_broker_snapshot(
                    broker_submission_enabled="False"
                )
            ),
        )

    def test_acknowledgement_rules(self) -> None:
        missing = build_with_all_snapshots(operator_acknowledgements=[])
        assert_not_ready(self, missing)
        none_ack = build_with_all_snapshots(operator_acknowledgements=None)
        assert_not_ready(self, none_ack)
        self.assertEqual([], none_ack["acknowledgement_checks"]["provided"])
        giant = build_with_all_snapshots(
            operator_acknowledgements=[" ".join(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS)]
        )
        assert_not_ready(self, giant)
        extra_only = build_with_all_snapshots(operator_acknowledgements=["extra"])
        assert_not_ready(self, extra_only)
        exact = build_with_all_snapshots(
            operator_acknowledgements=[
                f"  {ack}  " for ack in REQUIRED_OPERATOR_ACKNOWLEDGEMENTS
            ]
            + [object()]  # type: ignore[list-item]
        )
        assert_ready(self, exact)

    def test_candidate_and_proposals_are_one_strategy_paper_only_and_next_phase_only(self) -> None:
        report = build_with_all_snapshots()
        assert_ready(self, report)
        candidate = report["scheduler_lifecycle_activation_candidate"]
        self.assertTrue(candidate["available"])
        self.assertEqual("S01", candidate["selected_strategy_id"])
        self.assertTrue(candidate["paper_only"])
        self.assertTrue(candidate["one_strategy_only"])
        self.assertEqual(1, candidate["enabled_strategy_count"])
        self.assertFalse(candidate["broker_submission_allowed_next_phase"])
        self.assertFalse(candidate["live_trading_enabled"])
        self.assertFalse(candidate["all_strategies_enabled"])
        self.assertFalse(candidate["strategy_scan_execution_enabled_now"])
        self.assertFalse(candidate["order_submission_enabled_now"])
        scheduler = report["proposed_scheduler_activation"]
        lifecycle = report["proposed_lifecycle_activation"]
        self.assertTrue(scheduler["proposed_enabled_in_4I5"])
        self.assertFalse(scheduler["would_register_in_4I4"])
        self.assertFalse(scheduler["scheduler_job_enabled_now"])
        self.assertEqual("single_strategy", scheduler["job_scope"])
        self.assertTrue(lifecycle["proposed_enabled_in_4I5"])
        self.assertFalse(lifecycle["would_execute_in_4I4"])
        self.assertFalse(lifecycle["lifecycle_execution_enabled_now"])
        self.assertEqual("single_strategy", lifecycle["lifecycle_scope"])

    def test_proposals_not_enabled_when_gates_fail(self) -> None:
        report = build_with_all_snapshots(operator_acknowledgements=[])
        assert_not_ready(self, report)
        self.assertFalse(report["proposed_scheduler_activation"]["proposed_enabled_in_4I5"])
        self.assertFalse(report["proposed_lifecycle_activation"]["proposed_enabled_in_4I5"])

    def test_activation_snapshot_rules(self) -> None:
        assert_ready(self, build_with_all_snapshots())
        assert_not_ready(
            self,
            build_with_all_snapshots(
                activation_snapshot=matching_activation_snapshot(
                    activation_record={"selected_strategy_id": "S02"}
                )
            ),
        )
        assert_not_ready(
            self,
            build_with_all_snapshots(
                activation_snapshot=matching_activation_snapshot(active_strategy_ids=["S01", "S02"])
            ),
        )
        report = build_with_all_snapshots(
            activation_snapshot={
                "active_strategy_ids": ["S01"],
                "activations": ["bad", {"selected_strategy_id": "S01", "paper_only": True}],
            }
        )
        assert_ready(self, report)
        self.assertIn("malformed activation snapshot entry ignored", report["warnings"])

    def test_state_snapshot_blockers_and_active_intent_warning(self) -> None:
        assert_not_ready(self, build_with_all_snapshots(state_snapshot=clean_state_snapshot(active_halt=True)))
        assert_not_ready(
            self,
            build_with_all_snapshots(
                state_snapshot=clean_state_snapshot(unresolved_needs_reconciliation_count=1)
            ),
        )
        state = clean_state_snapshot(needs_reconciliation_count=1)
        state.pop("unresolved_needs_reconciliation_count")
        assert_not_ready(self, build_with_all_snapshots(state_snapshot=state))
        assert_not_ready(
            self,
            build_with_all_snapshots(state_snapshot=clean_state_snapshot(active_intents_count=1)),
        )
        safe = build_with_all_snapshots(
            state_snapshot=clean_state_snapshot(
                active_intents_count=1, active_intents_safe_for_enablement=True
            )
        )
        assert_ready(self, safe)
        self.assertTrue(safe["warnings"])
        assert_ready(self, build_with_all_snapshots(state_snapshot=clean_state_snapshot(open_positions_count=5)))

    def test_scheduler_lifecycle_risk_and_broker_blockers(self) -> None:
        assert_not_ready(
            self, build_with_all_snapshots(scheduler_snapshot={"scheduler_automation_enabled": True})
        )
        assert_not_ready(
            self, build_with_all_snapshots(scheduler_snapshot={"all_strategy_scheduler_enabled": True})
        )
        assert_not_ready(self, build_with_all_snapshots(scheduler_snapshot={"jobs": [{"strategy_id": "S01"}]}))
        assert_ready(
            self,
            build_with_all_snapshots(
                scheduler_snapshot={"jobs": [{"strategy_id": "S01", "disabled": True, "dry_run_only": True}]}
            ),
        )
        assert_not_ready(
            self, build_with_all_snapshots(lifecycle_snapshot={"lifecycle_automation_enabled": True})
        )
        assert_ready(
            self, build_with_all_snapshots(lifecycle_snapshot={"disabled": True, "dry_run_only": True})
        )
        assert_not_ready(
            self,
            build_with_all_snapshots(
                lifecycle_snapshot={"lifecycle_transition_execution_enabled": True}
            ),
        )
        for key in ("kill_switch_available", "hard_halt_available", "daily_loss_limit_available"):
            assert_not_ready(self, build_with_all_snapshots(risk_snapshot=clean_risk_snapshot(**{key: False})))
        assert_not_ready(self, build_with_all_snapshots(risk_snapshot=clean_risk_snapshot(risk_bypass_enabled=True)))
        for overrides in (
            {"mode": "LIVE"},
            {"ibkr_port": 4002},
            {"paper_trading": False},
            {"live_trading_enabled": True},
            {"broker_submission_enabled": True},
        ):
            assert_not_ready(
                self,
                build_with_all_snapshots(
                    paper_broker_snapshot=clean_paper_broker_snapshot(**overrides)
                ),
            )

    def test_market_window_blockers_warnings_and_missing_optional_snapshots(self) -> None:
        assert_not_ready(
            self,
            build_with_all_snapshots(
                market_window_snapshot=clean_market_window_snapshot(
                    allowed_to_schedule_paper_run=False
                )
            ),
        )
        closed = build_with_all_snapshots(
            market_window_snapshot=clean_market_window_snapshot(market_open=False)
        )
        assert_ready(self, closed)
        self.assertTrue(any("market is currently closed" in item for item in closed["warnings"]))
        missing = build(operator_acknowledgements=list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS))
        assert_ready(self, missing)
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, missing["warnings"])

    def test_valid_clean_reports_with_and_without_optional_snapshots(self) -> None:
        assert_ready(
            self,
            build(operator_acknowledgements=list(REQUIRED_OPERATOR_ACKNOWLEDGEMENTS)),
        )
        assert_ready(self, build_with_all_snapshots())

    def test_report_fields_json_safe_and_inputs_not_mutated(self) -> None:
        inputs = {
            "i3": valid_stage4i3_report(),
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
            "stage4i4_scheduler_lifecycle_activation_gate_report",
            "generated_at",
            "artifact_checks",
            "selected_strategy",
            "acknowledgement_checks",
            "scheduler_lifecycle_activation_candidate",
            "proposed_scheduler_activation",
            "proposed_lifecycle_activation",
            "proposed_runtime_guards",
            "proposed_pre_activation_checks",
            "proposed_post_activation_checks",
            "activation_checks",
            "state_checks",
            "risk_checks",
            "scheduler_checks",
            "lifecycle_checks",
            "paper_broker_checks",
            "market_window_checks",
            "safety_checks",
            "readiness_for_stage4i5",
            "recommendations",
            "success",
            "errors",
            "warnings",
        ):
            self.assertIn(key, report)
        json.dumps(report, sort_keys=True)
        self.assertEqual(before, inputs)

    def test_recommendations_do_not_include_disallowed_next_actions(self) -> None:
        recommendations = build_with_all_snapshots()["recommendations"]
        next_steps = json.dumps(recommendations["ordered_next_steps"]).lower()
        self.assertNotIn("place orders now", next_steps)
        self.assertNotIn("enable live trading", next_steps)
        self.assertNotIn("enable all strategies", next_steps)
        self.assertNotIn("register scheduler jobs in 4i-4", next_steps)
        text = json.dumps(recommendations).lower()
        for expected in (
            "do not enable live trading",
            "do not enable all strategies",
            "do not place orders now",
            "do not register scheduler jobs in 4i-4",
        ):
            self.assertIn(expected, text)

    def test_cli_requires_dry_run_only_before_parsing(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = tool.run_stage4i4_scheduler_lifecycle_activation_gate(
                ["--stage4i3-dry-run-json", "{bad"]
            )
        self.assertEqual(1, code)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("--dry-run-only", stderr.getvalue())

    def test_cli_json_stdout_is_strict_json_only_and_parse_errors_report_type(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        args = [
            "--dry-run-only",
            "--json",
            "--stage4i3-dry-run-json",
            json.dumps(valid_stage4i3_report()),
            "--activation-snapshot-json",
            json.dumps(matching_activation_snapshot()),
            "--state-snapshot-json",
            json.dumps(clean_state_snapshot()),
            "--risk-snapshot-json",
            json.dumps(clean_risk_snapshot()),
            "--scheduler-snapshot-json",
            "{}",
            "--lifecycle-snapshot-json",
            "{}",
            "--paper-broker-snapshot-json",
            json.dumps(clean_paper_broker_snapshot()),
            "--market-window-snapshot-json",
            json.dumps(clean_market_window_snapshot()),
        ]
        for ack in REQUIRED_OPERATOR_ACKNOWLEDGEMENTS:
            args.extend(["--ack", ack])
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = tool.run_stage4i4_scheduler_lifecycle_activation_gate(args)
        self.assertEqual(0, code)
        self.assertEqual("", stderr.getvalue())
        self.assertTrue(
            json.loads(stdout.getvalue())["stage4i4_scheduler_lifecycle_activation_gate_report"]
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = tool.run_stage4i4_scheduler_lifecycle_activation_gate(
                ["--dry-run-only", "--json", "--stage4i3-dry-run-json", "{bad"]
            )
        self.assertEqual(1, code)
        self.assertIn("JSONDecodeError", json.loads(stdout.getvalue())["errors"][0])

    def test_cli_ack_uses_append_and_exposes_no_forbidden_action_options(self) -> None:
        parser_source = (ROOT / "tools/stage4i4_scheduler_lifecycle_activation_gate.py").read_text()
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

    def test_no_forbidden_calls_or_production_wiring_in_stage4i4_files(self) -> None:
        source = "\n".join(path.read_text() for path in STAGE4I4_FILES)
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
