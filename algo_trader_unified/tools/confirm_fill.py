"""Confirm a dry-run fill for a confirmed order intent."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from algo_trader_unified.core.execution import DryRunExecutionAdapter
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.order_intents import confirm_fill
from algo_trader_unified.core.state_store import StateStore


def _parse_filled_at(value: str) -> str:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--filled-at must be a parseable ISO timestamp: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(f"--filled-at must be timezone-aware: {value!r}")
    return parsed.isoformat()


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result_payload(intent: dict) -> dict:
    return {
        "intent_id": intent["intent_id"],
        "status": intent["status"],
        "simulated_order_id": intent["simulated_order_id"],
        "fill_id": intent["fill_id"],
        "fill_confirmed_event_id": intent["fill_confirmed_event_id"],
        "dry_run": intent["dry_run"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent-id", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--filled-at", type=_parse_filled_at)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Accepted for clarity; dry-run is the only supported fill mode.",
    )
    args = parser.parse_args(argv)

    root_dir = Path(args.root_dir)
    filled_at = args.filled_at or _current_timestamp()
    try:
        state_store = StateStore(root_dir / "data/state/portfolio_state.json")
        ledger = LedgerAppender(root_dir)
        updated = confirm_fill(
            state_store=state_store,
            ledger=ledger,
            execution_adapter=DryRunExecutionAdapter(),
            intent_id=args.intent_id,
            filled_at=filled_at,
        )
    except Exception as exc:
        print(f"confirm fill failed for {args.intent_id!r}: {exc}", file=sys.stderr)
        return 1

    payload = _result_payload(updated)
    if args.json_output:
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    else:
        print(
            "confirmed fill "
            f"intent_id={payload['intent_id']} "
            f"status={payload['status']} "
            f"simulated_order_id={payload['simulated_order_id']} "
            f"fill_id={payload['fill_id']} "
            f"fill_confirmed_event_id={payload['fill_confirmed_event_id']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
