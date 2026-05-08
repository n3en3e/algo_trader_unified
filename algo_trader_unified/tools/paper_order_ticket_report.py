"""Print the Stage 4E-4 manual paper order ticket readiness report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from algo_trader_unified.core.paper_order_ticket_report import (
    build_paper_order_ticket_report,
)
from algo_trader_unified.core.state_store import StateStore, StateStoreCorruptError


def run_paper_order_ticket_report(
    argv: list[str] | tuple[str, ...],
    *,
    state_store_factory: Callable[[Path], Any] = StateStore,
    report_builder: Callable[..., dict[str, Any]] = build_paper_order_ticket_report,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--intent-id")
    parser.add_argument("--intent-json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4004)
    parser.add_argument("--client-id", type=int, default=7)
    parser.add_argument("--account-id")
    parser.add_argument("--trading-mode", default="PAPER")
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print(
            "ERROR: paper order ticket report requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    if bool(args.intent_id) == bool(args.intent_json):
        print(
            "ERROR: provide exactly one of --intent-id or --intent-json",
            file=sys.stderr,
        )
        return 1

    try:
        intent = _load_intent(args, state_store_factory=state_store_factory)
        config = {
            "host": args.host,
            "port": args.port,
            "client_id": args.client_id,
            "account_id": args.account_id,
            "trading_mode": args.trading_mode,
            "readonly": False,
        }
        report = report_builder(intent=intent, ibkr_config=config)
    except StateStoreCorruptError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    return 0 if report.get("success") is True else 1


def main(argv: list[str] | None = None) -> int:
    return run_paper_order_ticket_report(sys.argv[1:] if argv is None else argv)


def _load_intent(
    args: argparse.Namespace,
    *,
    state_store_factory: Callable[[Path], Any],
) -> dict[str, Any]:
    if args.intent_json:
        payload = json.loads(args.intent_json)
        if not isinstance(payload, dict):
            raise ValueError("--intent-json must decode to an object")
        return payload

    root = Path(args.root)
    state_store_path = root / "data" / "state" / "portfolio_state.json"
    if state_store_factory is StateStore and not state_store_path.exists():
        raise ValueError(f"StateStore does not exist: {state_store_path}")
    state_store = state_store_factory(state_store_path)
    intent = state_store.get_order_intent(args.intent_id)
    if intent is None:
        raise ValueError(f"order intent not found: {args.intent_id}")
    if not isinstance(intent, dict):
        raise ValueError(f"order intent is malformed: {args.intent_id}")
    return intent


def _format_human(report: dict[str, Any]) -> str:
    intent = report.get("intent")
    intent = intent if isinstance(intent, dict) else {}
    gate = report.get("submit_gate")
    gate = gate if isinstance(gate, dict) else {}
    plan = report.get("ibkr_order_plan")
    plan = plan if isinstance(plan, dict) else {}
    lines = [
        "Paper order ticket report",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        f"intent_id: {intent.get('intent_id')}",
        f"client_order_id: {plan.get('client_order_id')}",
        "eligible_for_future_manual_submit: "
        f"{gate.get('eligible_for_future_manual_submit')}",
    ]
    reasons = gate.get("reasons")
    if reasons:
        lines.append("reasons:")
        for reason in reasons:
            lines.append(f"  - {reason}")
    blockers = plan.get("blockers")
    if blockers:
        lines.append("blockers:")
        for blocker in blockers:
            lines.append(f"  - {blocker}")
    acknowledgements = gate.get("required_operator_acknowledgements")
    if acknowledgements:
        lines.append("required_operator_acknowledgements:")
        for acknowledgement in acknowledgements:
            lines.append(f"  - {acknowledgement}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
