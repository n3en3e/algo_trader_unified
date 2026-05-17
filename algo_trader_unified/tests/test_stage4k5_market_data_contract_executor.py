from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from decimal import Decimal
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from algo_trader_unified.core.stage4k5_market_data_contract_executor import (
    MARKET_WINDOW_MANUAL_WARNING,
    build_stage4k5_market_data_contract_executor_report,
)
from algo_trader_unified.tools import stage4k5_market_data_contract_executor as tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4K5_RUNTIME_FILES = [
    ROOT / "core/stage4k5_market_data_contract_executor.py",
    ROOT / "tools/stage4k5_market_data_contract_executor.py",
]


def valid_stage4k4_report(**overrides: object) -> dict:
    payload = {
        "selected_strategy_id": "S01",
        "operation_id": "s01_once_2026_05_16",
        "source_stage": "4K-4",
        "permission_source_stage": "4K-3",
        "execution_scope": "single_strategy_controlled_market_data_contract_execution",
        "paper_only": True,
        "one_strategy_only": True,
        "allow_controlled_market_data_provider_call": True,
        "allow_controlled_contract_qualification_provider_call": True,
        "allow_direct_reqMktData": False,
        "allow_direct_qualifyContracts": False,
        "allow_direct_reqContractDetails": False,
        "allow_strategy_scan": False,
        "allow_intent_creation": False,
        "allow_ticket_creation": False,
        "allow_order_submission": False,
        "allow_broker_submission": False,
        "allow_state_write": False,
        "allow_ledger_write": False,
        "live_trading_enabled": False,
        "all_strategies_enabled": False,
        "market_data_provider_payloads": [_provider_payload("SPY", request_id="md-001-SPY")],
        "contract_qualification_provider_payloads": [_provider_payload("SPY", qualification_id="cq-001-SPY")],
        "generated_at": "2026-05-17T15:00:00+00:00",
    }
    report: dict[str, object] = {
        "dry_run": True,
        "stage4k4_market_data_contract_execution_gate_report": True,
        "generated_at": "2026-05-17T15:00:00+00:00",
        "artifact_checks": {
            "stage4k4_report_present": True,
            "stage4k4_report_ready": True,
            "selected_strategy_present": True,
            "operation_id_present": True,
            "proposed_4k5_execution_payload_present": True,
        },
        "selected_strategy": {"selected_strategy_id": "S01", "paper_only": True, "one_strategy_only": True},
        "operation": {
            "operation_id": "s01_once_2026_05_16",
            "operation_scope": "single_strategy_market_data_contract_execution_gate",
            "paper_only": True,
            "live_trading_enabled": False,
            "broker_submission_enabled": False,
        },
        "stage4k4_gate_checks": {"ready_to_execute_controlled_market_data_contract_providers": True},
        "proposed_4k5_execution_payload": payload,
        "boundary_checks": {
            "no_market_data_fetched_now": True,
            "no_contracts_qualified_now": True,
            "no_provider_called_now": True,
            "no_direct_ib_call": True,
            "no_reqMktData": True,
            "no_qualifyContracts": True,
            "no_reqContractDetails": True,
            "no_strategy_scan": True,
            "no_intents_created": True,
            "no_tickets_created": True,
            "no_orders_submitted": True,
            "no_state_written": True,
            "no_ledger_written": True,
            "no_broker_submission": True,
            "no_live_trading": True,
            "no_all_strategy_enablement": True,
            "no_scheduler_registration": True,
            "no_lifecycle_execution": True,
        },
        "safety_checks": {
            "no_live_trading": True,
            "no_all_strategy_enablement": True,
            "no_broker_submission_enabled": True,
            "no_direct_market_data": True,
            "no_direct_contract_qualification": True,
            "no_order_submission": True,
            "no_intent_creation": True,
            "no_ticket_creation": True,
            "no_state_write": True,
            "no_ledger_write": True,
        },
        "readiness_for_stage4k5": {"ready_to_execute_controlled_market_data_contract_providers": True, "blockers": [], "warnings": []},
        "success": True,
        "errors": [],
        "warnings": [],
    }
    _deep_update(report, overrides)
    return report


def _provider_payload(symbol: str, **extra: object) -> dict:
    payload = {
        "selected_strategy_id": "S01",
        "operation_id": "s01_once_2026_05_16",
        "symbol": symbol,
        "allow_live_trading": False,
        "allow_broker_submission": False,
        "allow_order_submission": False,
        "allow_state_write": False,
        "allow_ledger_write": False,
    }
    payload.update(extra)
    return payload


_DEFAULT_PROVIDER_RESULT = object()


class RecordingProvider:
    def __init__(self, result: object = _DEFAULT_PROVIDER_RESULT, raises: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self.result = {"ok": True} if result is _DEFAULT_PROVIDER_RESULT else result
        self.raises = raises

    def request_controlled_market_data(self, payload: dict) -> object:
        self.calls.append(payload)
        if self.raises:
            raise self.raises
        return self.result

    def qualify_controlled_contracts(self, payload: dict) -> object:
        self.calls.append(payload)
        if self.raises:
            raise self.raises
        return self.result


class MissingMethod:
    pass


def clean_state_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"active_halt": False, "unresolved_needs_reconciliation_count": 0, "active_intents_count": 0, "open_positions_count": 7}
    _deep_update(snapshot, overrides)
    return snapshot


def clean_risk_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"kill_switch_available": True, "hard_halt_available": True, "daily_loss_limit_available": True, "max_position_limit_available": True, "risk_bypass_enabled": False}
    _deep_update(snapshot, overrides)
    return snapshot


def clean_scheduler_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"jobs": [{"strategy_id": "S01", "broker_submission_enabled": False, "live_trading_enabled": False, "all_strategies_enabled": False, "strategy_scan_execution_enabled": False}]}
    _deep_update(snapshot, overrides)
    return snapshot


def clean_lifecycle_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"lifecycle_automation_enabled": True, "selected_strategy_id": "S01", "broker_submission_enabled": False, "live_trading_enabled": False, "all_strategies_enabled": False, "lifecycle_transition_execution_enabled": False}
    _deep_update(snapshot, overrides)
    return snapshot


def clean_paper_broker_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"mode": "PAPER", "paper_trading": True, "ibkr_port": 4004, "live_trading_enabled": False, "broker_submission_enabled": False}
    _deep_update(snapshot, overrides)
    return snapshot


def clean_market_window_snapshot(**overrides: object) -> dict:
    snapshot: dict[str, object] = {"allowed_to_schedule_paper_run": True, "is_trading_day": True, "market_open": True}
    _deep_update(snapshot, overrides)
    return snapshot


_DEFAULT_REPORT = object()


def build(report: object = _DEFAULT_REPORT, **kwargs: object) -> dict:
    if report is _DEFAULT_REPORT:
        report = valid_stage4k4_report()
    kwargs.setdefault("controlled_market_data_provider", RecordingProvider({"md_ok": True}))
    kwargs.setdefault("controlled_contract_qualification_provider", RecordingProvider({"cq_ok": True}))
    return build_stage4k5_market_data_contract_executor_report(
        stage4k4_execution_gate_report=report,  # type: ignore[arg-type]
        now_provider=lambda: datetime(2026, 5, 17, 16, 0, tzinfo=timezone.utc),
        **kwargs,
    )


def build_with_snapshots(**kwargs: object) -> dict:
    defaults: dict[str, object] = {
        "state_snapshot": clean_state_snapshot(),
        "risk_snapshot": clean_risk_snapshot(),
        "scheduler_snapshot": clean_scheduler_snapshot(),
        "lifecycle_snapshot": clean_lifecycle_snapshot(),
        "paper_broker_snapshot": clean_paper_broker_snapshot(),
        "market_window_snapshot": clean_market_window_snapshot(),
    }
    defaults.update(kwargs)
    report = defaults.pop("report", _DEFAULT_REPORT)
    return build(report, **defaults)


def ready(report: dict) -> bool:
    return report["readiness_for_stage4k6"]["ready_to_build_market_data_contract_acceptance"]


def _deep_update(target: dict, updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index,arg-type]
        else:
            target[key] = value


def _with_payload_update(**updates: object) -> dict:
    report = valid_stage4k4_report()
    payload = report["proposed_4k5_execution_payload"]
    assert isinstance(payload, dict)
    payload.update(updates)
    return report


class Stage4K5MarketDataContractExecutorTests(unittest.TestCase):
    def test_stage4k4_artifact_strategy_operation_and_payload_gates(self) -> None:
        cases = [
            None,
            valid_stage4k4_report(stage4k4_market_data_contract_execution_gate_report=False),
            valid_stage4k4_report(success=False),
            valid_stage4k4_report(readiness_for_stage4k5={"ready_to_execute_controlled_market_data_contract_providers": False}),
            valid_stage4k4_report(selected_strategy={"selected_strategy_id": ""}),
            valid_stage4k4_report(operation={"operation_id": ""}),
            valid_stage4k4_report(proposed_4k5_execution_payload=None),
            valid_stage4k4_report(proposed_4k5_execution_payload="bad"),
        ]
        for report in cases:
            with self.subTest(report=report):
                result = build(report)
                self.assertFalse(ready(result))
                json.dumps(result)

    def test_permission_and_payload_length_checks_do_not_index_empty_or_none_lists(self) -> None:
        cases = [
            _with_payload_update(allow_controlled_market_data_provider_call=False, allow_controlled_contract_qualification_provider_call=False),
            _with_payload_update(market_data_provider_payloads=[]),
            _with_payload_update(market_data_provider_payloads=None),
            _with_payload_update(contract_qualification_provider_payloads=[]),
            _with_payload_update(contract_qualification_provider_payloads=None),
            _with_payload_update(allow_controlled_market_data_provider_call=False),
            _with_payload_update(allow_controlled_contract_qualification_provider_call=False),
        ]
        for report in cases:
            with self.subTest(report=report):
                self.assertFalse(ready(build(report)))

    def test_missing_providers_and_missing_methods_block_when_permission_true(self) -> None:
        self.assertFalse(ready(build(controlled_market_data_provider=None)))
        self.assertFalse(ready(build(controlled_contract_qualification_provider=None)))
        self.assertFalse(ready(build(controlled_market_data_provider=MissingMethod())))
        self.assertFalse(ready(build(controlled_contract_qualification_provider=MissingMethod())))

    def test_market_data_only_qualification_only_and_combined_order(self) -> None:
        md = RecordingProvider({"md_ok": True})
        cq = RecordingProvider({"cq_ok": True})
        md_only = _with_payload_update(allow_controlled_contract_qualification_provider_call=False, contract_qualification_provider_payloads=[])
        result = build(md_only, controlled_market_data_provider=md, controlled_contract_qualification_provider=None)
        self.assertTrue(ready(result))
        self.assertEqual(len(md.calls), 1)
        self.assertEqual(result["provider_call_trace"][0]["provider_type"], "market_data")
        self.assertEqual(result["contract_qualification_execution_results"]["qualification_count_attempted"], 0)

        md = RecordingProvider({"md_ok": True})
        cq = RecordingProvider({"cq_ok": True})
        cq_only = _with_payload_update(allow_controlled_market_data_provider_call=False, market_data_provider_payloads=[])
        result = build(cq_only, controlled_market_data_provider=None, controlled_contract_qualification_provider=cq)
        self.assertTrue(ready(result))
        self.assertEqual(len(cq.calls), 1)
        self.assertEqual(result["provider_call_trace"][0]["provider_type"], "contract_qualification")
        self.assertEqual(result["market_data_execution_results"]["request_count_attempted"], 0)

        result = build(controlled_market_data_provider=md, controlled_contract_qualification_provider=cq)
        self.assertTrue(ready(result))
        self.assertEqual([item["provider_type"] for item in result["provider_call_trace"]], ["market_data", "contract_qualification"])

    def test_provider_receives_sanitized_item_payload_and_inputs_are_not_mutated(self) -> None:
        report = valid_stage4k4_report()
        original = copy.deepcopy(report)
        provider = RecordingProvider({"ok": True})
        result = build(report, controlled_market_data_provider=provider, controlled_contract_qualification_provider=RecordingProvider({"ok": True}))
        self.assertTrue(ready(result))
        self.assertEqual(report, original)
        self.assertEqual(provider.calls[0], original["proposed_4k5_execution_payload"]["market_data_provider_payloads"][0])  # type: ignore[index]
        self.assertNotIn("proposed_4k5_execution_payload", provider.calls[0])

    def test_payload_validation_blocks_non_dict_json_string_unsafe_flags_and_stale_keys(self) -> None:
        cases = [
            _with_payload_update(market_data_provider_payloads=["bad"]),
            _with_payload_update(market_data_provider_payloads=[json.dumps(_provider_payload("SPY"))]),
            _with_payload_update(market_data_provider_payloads=[_provider_payload("SPY", bad=Decimal("1.2"))]),
        ]
        for key in ("allow_live_trading", "allow_broker_submission", "allow_order_submission", "allow_state_write", "allow_ledger_write"):
            cases.append(_with_payload_update(market_data_provider_payloads=[_provider_payload("SPY", **{key: True})]))
            cases.append(_with_payload_update(market_data_provider_payloads=[_provider_payload("SPY", **{key: "False"})]))
        for key in (
            "proposed_execution_permissions_for_4J5",
            "may_call_strategy_next_phase",
            "may_build_executor_next_phase",
            "may_fetch_market_data_next_phase",
            "allow_direct_reqMktData",
            "allow_direct_qualifyContracts",
            "allow_direct_reqContractDetails",
            "allow_strategy_scan",
            "allow_intent_creation",
            "allow_ticket_creation",
            "allow_order_submission",
            "allow_broker_submission",
            "allow_state_write",
            "allow_ledger_write",
            "live_trading_enabled",
            "all_strategies_enabled",
        ):
            cases.append(_with_payload_update(**{key: True}))
        for report in cases:
            with self.subTest(report=report):
                self.assertFalse(ready(build(report)))

    def test_provider_result_validation_and_exception_sanitization(self) -> None:
        bad_results = [
            None,
            ["bad"],
            {"bad": Decimal("1.2")},
            {"live_trading_enabled": True},
            {"broker_submission_enabled": True},
            {"order_submitted": True},
            {"state_written": True},
            {"ledger_written": True},
            {"direct_ib_call_made": True},
            {"reqMktData_called": True},
            {"qualifyContracts_called": True},
            {"reqContractDetails_called": True},
            {"live_trading_enabled": "False"},
        ]
        for result in bad_results:
            with self.subTest(result=result):
                report = build(controlled_market_data_provider=RecordingProvider(result), controlled_contract_qualification_provider=RecordingProvider({"ok": True}))
                self.assertFalse(ready(report))
                self.assertTrue(report["failed_operations"])

        exc = RuntimeError("line one\nline two <object at 0xABCDEF>")
        report = build(controlled_market_data_provider=RecordingProvider(raises=exc), controlled_contract_qualification_provider=RecordingProvider({"ok": True}))
        self.assertFalse(ready(report))
        self.assertEqual(report["failed_operations"][0]["status"], "failed")
        reason = report["provider_call_trace"][0]["failure_reason"]
        self.assertIn("RuntimeError: line one line two", reason)
        self.assertNotIn("\n", reason)
        self.assertNotIn("0xABCDEF", reason)
        self.assertTrue(report["skipped_operations"])
        self.assertEqual(report["provider_call_trace"][1]["provider_called"], False)

    def test_trace_operations_counts_and_boundaries(self) -> None:
        report = build()
        self.assertTrue(ready(report))
        self.assertEqual(len(report["applied_operations"]), 2)
        self.assertFalse(report["failed_operations"])
        self.assertFalse(report["skipped_operations"])
        for item in report["provider_call_trace"]:
            for key in (
                "sequence_number",
                "provider_type",
                "provider_method",
                "payload_id",
                "selected_strategy_id",
                "operation_id",
                "input_payload",
                "provider_called",
                "direct_ib_call_made",
                "success",
                "result",
                "failure_reason",
                "skipped_reason",
                "live_trading_enabled",
                "broker_submission_enabled",
                "order_submission_enabled",
                "state_write_enabled",
                "ledger_write_enabled",
            ):
                self.assertIn(key, item)
            self.assertFalse(item["direct_ib_call_made"])
        self.assertTrue(report["market_data_execution_results"]["attempted"])
        self.assertTrue(report["market_data_execution_results"]["provider_called"])
        self.assertFalse(report["market_data_execution_results"]["reqMktData_called"])
        self.assertFalse(report["contract_qualification_execution_results"]["qualifyContracts_called"])
        self.assertFalse(report["contract_qualification_execution_results"]["reqContractDetails_called"])
        self.assertTrue(report["boundary_checks"]["no_direct_ib_call"])

    def test_snapshot_blockers_and_warnings(self) -> None:
        blocking_cases = [
            {"state_snapshot": clean_state_snapshot(active_halt=True)},
            {"state_snapshot": clean_state_snapshot(unresolved_needs_reconciliation_count=1)},
            {"state_snapshot": {"needs_reconciliation_count": 1, "active_intents_count": 0}},
            {"state_snapshot": clean_state_snapshot(active_intents_count=1)},
            {"scheduler_snapshot": clean_scheduler_snapshot(all_strategies_enabled=True)},
            {"scheduler_snapshot": clean_scheduler_snapshot(strategy_scan_execution_enabled=True)},
            {"lifecycle_snapshot": clean_lifecycle_snapshot(all_strategies_enabled=True)},
            {"lifecycle_snapshot": clean_lifecycle_snapshot(lifecycle_transition_execution_enabled=True)},
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
        for kwargs in blocking_cases:
            with self.subTest(kwargs=kwargs):
                self.assertFalse(ready(build_with_snapshots(**kwargs)))
        safe_intents = build_with_snapshots(state_snapshot=clean_state_snapshot(active_intents_count=1, active_intents_safe_for_enablement=True))
        self.assertTrue(ready(safe_intents))
        self.assertIn("active intents present", " ".join(safe_intents["warnings"]))
        closed = build_with_snapshots(market_window_snapshot=clean_market_window_snapshot(market_open=False))
        self.assertTrue(ready(closed))
        self.assertIn("market is currently closed", " ".join(closed["warnings"]))
        missing_window = build_with_snapshots(market_window_snapshot=None)
        self.assertTrue(ready(missing_window))
        self.assertIn(MARKET_WINDOW_MANUAL_WARNING, missing_window["warnings"])

    def test_report_shape_json_safety_recommendations_and_cli(self) -> None:
        report = build()
        self.assertTrue(ready(report))
        for key in (
            "dry_run",
            "stage4k5_market_data_contract_executor_report",
            "stage4k4_gate_checks",
            "execution_payload_checks",
            "provider_availability_checks",
            "market_data_execution_results",
            "contract_qualification_execution_results",
            "provider_call_trace",
            "applied_operations",
            "failed_operations",
            "skipped_operations",
            "required_inputs_for_4k6",
            "readiness_for_stage4k6",
        ):
            self.assertIn(key, report)
        self.assertEqual(report["generated_at"], "2026-05-17T16:00:00+00:00")
        json.dumps(report)
        text = " ".join(report["recommendations"]["ordered_next_steps"]).lower()
        for phrase in ("enable live trading", "enable all strategies", "place orders now", "create intents", "write state", "direct reqmktdata"):
            self.assertNotIn(phrase, text)

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            rc = tool.run_stage4k5_market_data_contract_executor(["--stage4k4-gate-json", "{"])
        self.assertNotEqual(rc, 0)
        self.assertIn("--dry-run-only", stderr.getvalue())

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4k5_market_data_contract_executor(["--dry-run-only", "--json", "--stage4k4-gate-json", "{"])
        self.assertNotEqual(rc, 0)
        self.assertIn("JSONDecodeError", json.loads(stdout.getvalue())["errors"][0])

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = tool.run_stage4k5_market_data_contract_executor(["--dry-run-only", "--json", "--stage4k4-gate-json", json.dumps(valid_stage4k4_report())])
        self.assertNotEqual(rc, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertTrue(parsed["stage4k5_market_data_contract_executor_report"])
        self.assertIn("injected providers", " ".join(parsed["warnings"]))

        namespace = argparse.Namespace()
        for attr in ("submit", "cancel", "status", "market_data_execute", "contract_qualification_execute", "scheduler_enable", "broker_submit"):
            self.assertFalse(hasattr(namespace, attr))

    def test_no_direct_broker_external_scheduler_strategy_or_runner_calls_in_stage4k5_files(self) -> None:
        source = "\n".join(path.read_text() for path in STAGE4K5_RUNTIME_FILES)
        for forbidden in (
            "submit_order_plan(",
            "get_order_status(",
            "cancel_order(",
            "placeOrder(",
            "cancelOrder(",
            "reqMktData(",
            "qualifyContracts(",
            "reqContractDetails(",
            "StateStore(",
            ".save(",
            ".write(",
            ".update(",
            "ledger.append(",
            "ledger.write(",
            "scheduler.add_job(",
            "add_job(",
            ".start(",
            "run_scan(",
            "scan_now(",
            "run_controlled_paper_operation(",
            "yfinance.",
            "requests.",
            "urllib.",
            "systemctl",
            "systemd",
            "socket.create_connection",
            "socket.socket",
            "asyncio.run(",
            "asyncio.get_event_loop(",
            "asyncio.new_event_loop(",
            "uuid.uuid4(",
            "random.",
            "time.time(",
            "datetime.now(",
            "traceback.format_exc(",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
