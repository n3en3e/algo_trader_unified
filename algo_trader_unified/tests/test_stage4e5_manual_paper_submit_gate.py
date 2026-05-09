from __future__ import annotations

import copy
import inspect
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from algo_trader_unified.core.manual_paper_submit_gate import (
    build_manual_paper_submit_result,
)
from algo_trader_unified.tools import manual_paper_submit_gate as gate_tool


ROOT = Path(__file__).resolve().parents[1]
STAGE4E5_FILES = [
    ROOT / "core/manual_paper_submit_gate.py",
    ROOT / "tools/manual_paper_submit_gate.py",
    ROOT / "tests/test_stage4e5_manual_paper_submit_gate.py",
]
UNWIRED_RUNTIME_FILES = [
    ROOT / "tools/daemon.py",
    ROOT / "core/scheduler.py",
    ROOT / "core/scheduler_cadence.py",
    ROOT / "core/job_chain.py",
    ROOT / "jobs/submission.py",
    ROOT / "jobs/confirmation.py",
    ROOT / "jobs/fill_confirmation.py",
]


REQUIRED_ACKS = [
    "I understand this is PAPER only.",
    "I understand no live orders are allowed.",
    "I understand this ticket is not yet submitted.",
    "I understand scheduler/lifecycle wiring remains disabled.",
]


def valid_ticket(**overrides: object) -> dict[str, object]:
    ticket: dict[str, object] = {
        "dry_run": True,
        "paper_order_ticket_report": True,
        "generated_at": "2026-05-08T12:00:00+00:00",
        "intent": {
            "intent_id": "intent-stage4e5-001",
            "strategy_id": "S01_VOL_BASELINE",
            "symbol": "XSP",
            "side": "BUY",
            "quantity": 1,
            "order_type": "LIMIT",
            "valid": True,
        },
        "broker_order_request": {
            "available": True,
            "client_order_id": "intent-stage4e5-001",
            "strategy_id": "S01_VOL_BASELINE",
            "symbol": "XSP",
            "side": "BUY",
            "quantity": 1,
            "order_type": "LIMIT",
        },
        "ibkr_order_plan": {
            "available": True,
            "ready_for_submission": True,
            "client_order_id": "intent-stage4e5-001",
            "action": "BUY",
            "quantity": 1,
            "order_type": "LMT",
            "paper_only": True,
            "dry_run": True,
            "blockers": [],
            "ibkr_contract_hint": {"symbol": "XSP"},
            "ibkr_order_hint": {"orderRef": "intent-stage4e5-001"},
        },
        "submit_gate": {
            "eligible_for_future_manual_submit": True,
            "reasons": [],
            "required_operator_acknowledgements": list(REQUIRED_ACKS),
        },
        "safety": {
            "broker_calls_enabled": False,
            "order_submission_enabled": False,
            "cancel_enabled": False,
            "market_data_enabled": False,
            "contract_qualification_enabled": False,
            "live_orders_enabled": False,
            "scheduler_changes_enabled": False,
        },
        "success": True,
        "errors": [],
        "warnings": [],
    }
    ticket.update(overrides)
    return ticket


class FakeExecutionClient:
    def __init__(self, result: object | None = None, exc: Exception | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.result = result or SimpleNamespace(
            accepted=True,
            dry_run=True,
            broker_order_id="9001",
            client_order_id="intent-stage4e5-001",
            reason=None,
            raw={"operation": "fake_submit_order_plan"},
        )
        self.exc = exc

    def submit_order_plan(self, plan: dict[str, object]) -> object:
        self.calls.append(copy.deepcopy(plan))
        if self.exc is not None:
            raise self.exc
        return self.result


class ManualPaperSubmitGateCoreTests(unittest.TestCase):
    def build(
        self,
        ticket: dict | None = None,
        acknowledgements: list[str] | None = None,
        client: FakeExecutionClient | None = None,
    ) -> tuple[dict, FakeExecutionClient]:
        fake = client or FakeExecutionClient()
        result = build_manual_paper_submit_result(
            ticket_report=valid_ticket() if ticket is None else ticket,
            execution_client=fake,
            operator_acknowledgements=list(REQUIRED_ACKS)
            if acknowledgements is None
            else acknowledgements,
            now_provider=lambda: datetime(2026, 5, 8, 12, 30, tzinfo=timezone.utc),
        )
        return result, fake

    def assert_refused(self, ticket: dict, expected_reason: str) -> None:
        result, fake = self.build(ticket=ticket)

        self.assertFalse(result["submit_gate"]["passed"])
        self.assertFalse(result["submission"]["attempted"])
        self.assertFalse(result["submission"]["submitted"])
        self.assertEqual(fake.calls, [])
        self.assertIn(expected_reason, result["submit_gate"]["reasons"])

    def test_valid_ticket_and_exact_acknowledgements_call_fake_client_once(self) -> None:
        ticket = valid_ticket()
        result, fake = self.build(ticket=ticket)

        self.assertTrue(result["manual_paper_submit_gate"])
        self.assertTrue(result["submit_gate"]["passed"])
        self.assertTrue(result["submission"]["attempted"])
        self.assertTrue(result["submission"]["submitted"])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0], ticket["ibkr_order_plan"])
        self.assertEqual(result["submission"]["broker_order_id"], "9001")
        self.assertEqual(result["submission"]["client_order_id"], "intent-stage4e5-001")

    def test_missing_acknowledgement_refuses_submit(self) -> None:
        result, fake = self.build(acknowledgements=REQUIRED_ACKS[:-1])

        self.assertEqual(fake.calls, [])
        self.assertFalse(result["acknowledgements"]["exact_match"])
        self.assertEqual(result["acknowledgements"]["missing"], [REQUIRED_ACKS[-1]])
        self.assertFalse(result["submit_gate"]["passed"])

    def test_acknowledgement_matching_uses_exact_list_items(self) -> None:
        acknowledgements = [
            "I understand this is PAPER only. extra",
            *REQUIRED_ACKS[1:],
        ]
        result, fake = self.build(acknowledgements=acknowledgements)

        self.assertEqual(fake.calls, [])
        self.assertEqual(result["acknowledgements"]["missing"], [REQUIRED_ACKS[0]])

    def test_one_giant_acknowledgement_string_does_not_pass(self) -> None:
        result, fake = self.build(acknowledgements=[" ".join(REQUIRED_ACKS)])

        self.assertEqual(fake.calls, [])
        self.assertEqual(result["acknowledgements"]["missing"], REQUIRED_ACKS)

    def test_extra_acknowledgements_do_not_compensate_for_missing_required_ack(self) -> None:
        result, fake = self.build(
            acknowledgements=[REQUIRED_ACKS[0], REQUIRED_ACKS[1], "operator override"]
        )

        self.assertEqual(fake.calls, [])
        self.assertEqual(result["acknowledgements"]["missing"], REQUIRED_ACKS[2:])

    def test_whitespace_only_acknowledgement_differences_are_stripped(self) -> None:
        acknowledgements = [f"  {item}  " for item in REQUIRED_ACKS]
        result, fake = self.build(acknowledgements=acknowledgements)

        self.assertTrue(result["acknowledgements"]["exact_match"])
        self.assertEqual(len(fake.calls), 1)

    def test_ineligible_ticket_refuses_submit(self) -> None:
        ticket = valid_ticket(
            submit_gate={
                "eligible_for_future_manual_submit": False,
                "reasons": ["not eligible"],
                "required_operator_acknowledgements": list(REQUIRED_ACKS),
            }
        )

        self.assert_refused(ticket, "ticket is not eligible for future manual submit")

    def test_ready_for_submission_false_refuses_submit(self) -> None:
        ticket = valid_ticket()
        ticket["ibkr_order_plan"]["ready_for_submission"] = False

        self.assert_refused(ticket, "IBKR paper order plan is not ready for submission")

    def test_paper_only_false_refuses_submit(self) -> None:
        ticket = valid_ticket()
        ticket["ibkr_order_plan"]["paper_only"] = False

        self.assert_refused(ticket, "IBKR paper order plan is not marked paper_only")

    def test_dry_run_false_refuses_submit(self) -> None:
        ticket = valid_ticket()
        ticket["ibkr_order_plan"]["dry_run"] = False

        self.assert_refused(ticket, "IBKR paper order plan is not marked dry_run")

    def test_blockers_refuse_submit(self) -> None:
        ticket = valid_ticket()
        ticket["ibkr_order_plan"]["blockers"] = ["missing expiry"]

        self.assert_refused(ticket, "IBKR paper order plan has blockers")

    def test_live_orders_enabled_true_refuses_submit(self) -> None:
        ticket = valid_ticket()
        ticket["safety"]["live_orders_enabled"] = True

        self.assert_refused(ticket, "ticket safety live_orders_enabled must be False")

    def test_scheduler_changes_enabled_true_refuses_submit(self) -> None:
        ticket = valid_ticket()
        ticket["safety"]["scheduler_changes_enabled"] = True

        self.assert_refused(ticket, "ticket safety scheduler_changes_enabled must be False")

    def test_fake_submit_accepted_result_is_mapped(self) -> None:
        result, _fake = self.build(
            client=FakeExecutionClient(
                result={
                    "accepted": True,
                    "dry_run": True,
                    "broker_order_id": "BRK-1",
                    "client_order_id": "intent-stage4e5-001",
                    "reason": None,
                    "raw": {"accepted": True},
                }
            )
        )

        self.assertTrue(result["submission"]["submitted"])
        self.assertEqual(result["submission"]["broker_order_id"], "BRK-1")
        self.assertEqual(result["submission"]["raw"], {"accepted": True})

    def test_fake_submit_rejected_result_is_mapped(self) -> None:
        result, _fake = self.build(
            client=FakeExecutionClient(
                result=SimpleNamespace(
                    accepted=False,
                    dry_run=True,
                    broker_order_id=None,
                    client_order_id="intent-stage4e5-001",
                    reason="fake rejected",
                    raw={"accepted": False},
                )
            )
        )

        self.assertTrue(result["submission"]["attempted"])
        self.assertFalse(result["submission"]["submitted"])
        self.assertEqual(result["submission"]["reason"], "fake rejected")

    def test_fake_submit_exception_is_caught_and_reported(self) -> None:
        result, fake = self.build(client=FakeExecutionClient(exc=TimeoutError("too slow")))

        self.assertEqual(len(fake.calls), 1)
        self.assertFalse(result["submission"]["submitted"])
        self.assertIn("TimeoutError: too slow", result["submission"]["reason"])
        self.assertIn("TimeoutError: too slow", result["errors"])
        self.assertEqual(result["submission"]["raw"]["operation"], "fake_submit_order_plan")

    def test_input_ticket_report_is_not_mutated(self) -> None:
        ticket = valid_ticket()
        before = copy.deepcopy(ticket)

        self.build(ticket=ticket)

        self.assertEqual(ticket, before)

    def test_result_is_json_safe(self) -> None:
        result, _fake = self.build()

        json.dumps(result, sort_keys=True)

    def test_recommendations_are_deterministic_and_non_binding(self) -> None:
        first, _fake = self.build()
        second, _fake = self.build()

        self.assertEqual(first["recommendations"], second["recommendations"])
        joined = "\n".join(
            first["recommendations"]["ordered_next_steps"]
            + first["recommendations"]["do_not_do_yet"]
        )
        self.assertIn("Do not connect this command to real IBKR.", joined)


class ManualPaperSubmitGateCliTests(unittest.TestCase):
    def run_cli(
        self,
        argv: list[str],
        *,
        result_builder: object | None = None,
        execution_client: object | None = None,
    ) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        kwargs = {}
        if result_builder is not None:
            kwargs["result_builder"] = result_builder
        if execution_client is not None:
            kwargs["execution_client"] = execution_client
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = gate_tool.run_manual_paper_submit_gate(argv, **kwargs)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_requires_dry_run_only_before_load(self) -> None:
        def exploding_builder(**_kwargs: object) -> dict[str, object]:
            raise AssertionError("builder must not run before dry-run gate")

        code, stdout, stderr = self.run_cli(
            ["--ticket-json", json.dumps(valid_ticket())],
            result_builder=exploding_builder,
        )

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("--dry-run-only", stderr)

    def test_cli_json_stdout_is_strict_json_only(self) -> None:
        code, stdout, stderr = self.run_cli(
            [
                "--dry-run-only",
                "--json",
                "--ticket-json",
                json.dumps(valid_ticket()),
                *sum((["--ack", item] for item in REQUIRED_ACKS), []),
            ],
            execution_client=FakeExecutionClient(),
        )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertTrue(payload["manual_paper_submit_gate"])
        self.assertTrue(payload["safety"]["fake_client_only"])

    def test_cli_ack_uses_argparse_append(self) -> None:
        source = inspect.getsource(gate_tool.run_manual_paper_submit_gate)

        self.assertIn('parser.add_argument("--ack", action="append"', source)

    def test_cli_exposes_no_real_submit_cancel_market_data_or_qualification_actions(self) -> None:
        source = inspect.getsource(gate_tool.run_manual_paper_submit_gate)
        for token in ("--submit", "--cancel", "--market-data", "--qualify"):
            with self.subTest(token=token):
                self.assertNotIn(token, source)


class Stage4E5SafetyBoundaryTests(unittest.TestCase):
    def test_stage4e5_files_do_not_import_or_call_blocked_integrations(self) -> None:
        blocked_tokens = (
            "ib_" + "insync",
            "place" + "Order",
            "cancel" + "Order",
            "req" + "MktData",
            "qualify" + "Contracts",
            "y" + "finance",
            "requ" + "ests",
            "url" + "lib",
            "system" + "ctl",
            "system" + "d",
            "socket." + "create_connection",
            "socket." + "socket",
            "asyncio." + "run",
            "asyncio." + "get_event_loop",
            "asyncio." + "new_event_loop",
            "uuid." + "uuid4",
            "rand" + "om",
        )
        for path in STAGE4E5_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_stage4e5_files_do_not_wire_real_clients_scheduler_or_lifecycle(self) -> None:
        forbidden_tokens = (
            "IbkrPaper" + "ExecutionClient",
            "IbkrPaper" + "ReadOnlyClient",
            "PaperBroker" + "Adapter",
            "core." + "scheduler",
            "jobs." + "submission",
            "jobs." + "confirmation",
            "jobs." + "fill_confirmation",
        )
        for path in STAGE4E5_FILES:
            source = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_submit_order_plan_is_not_added_to_runtime_wiring(self) -> None:
        token = "submit_" + "order_plan"
        for path in UNWIRED_RUNTIME_FILES:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
