"""Run the Stage 4E-5 fake-client manual paper submit gate."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.manual_paper_submit_gate import (
    build_manual_paper_submit_result,
)


class _LocalFakeExecutionClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def submit_order_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(plan))
        return {
            "accepted": False,
            "dry_run": True,
            "broker_order_id": None,
            "client_order_id": plan.get("client_order_id"),
            "reason": "local fake client does not submit orders",
            "raw": {"operation": "local_fake_submit_order_plan"},
        }


def run_manual_paper_submit_gate(
    argv: list[str] | tuple[str, ...],
    *,
    result_builder: Callable[..., dict[str, Any]] = build_manual_paper_submit_result,
    execution_client: Any | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--ticket-json")
    parser.add_argument("--ack", action="append", default=[])
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print(
            "ERROR: manual paper submit gate requires --dry-run-only",
            file=sys.stderr,
        )
        return 1
    if not args.ticket_json:
        print("ERROR: provide --ticket-json", file=sys.stderr)
        return 1

    try:
        ticket_report = json.loads(args.ticket_json)
        if not isinstance(ticket_report, dict):
            raise ValueError("--ticket-json must decode to an object")
        client = execution_client if execution_client is not None else _LocalFakeExecutionClient()
        result = result_builder(
            ticket_report=ticket_report,
            execution_client=client,
            operator_acknowledgements=args.ack,
        )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        print(json.dumps(result, sort_keys=True))
    else:
        print(_format_human(result))
    return 0 if result.get("success") is True else 1


def main(argv: list[str] | None = None) -> int:
    return run_manual_paper_submit_gate(sys.argv[1:] if argv is None else argv)


def _format_human(result: dict[str, Any]) -> str:
    gate = result.get("submit_gate")
    gate = gate if isinstance(gate, dict) else {}
    submission = result.get("submission")
    submission = submission if isinstance(submission, dict) else {}
    lines = [
        "Manual paper submit gate",
        "mode: fake-client/manual-gate only",
        f"success: {result.get('success')}",
        f"gate_passed: {gate.get('passed')}",
        f"attempted: {submission.get('attempted')}",
        f"submitted: {submission.get('submitted')}",
        f"client_order_id: {submission.get('client_order_id')}",
    ]
    reasons = gate.get("reasons")
    if reasons:
        lines.append("reasons:")
        for reason in reasons:
            lines.append(f"  - {reason}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
