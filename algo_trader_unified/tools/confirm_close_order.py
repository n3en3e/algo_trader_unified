"""Confirm a submitted dry-run close order."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from algo_trader_unified.core.close_intents import confirm_close_order
from algo_trader_unified.core.execution import DryRunExecutionAdapter
from algo_trader_unified.core.ledger import LedgerAppender, LedgerValidationError
from algo_trader_unified.core.state_store import CloseIntentTransitionError, StateStore


def _parse_confirmed_at(value: str) -> str:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--confirmed-at must be a parseable ISO timestamp: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(
            f"--confirmed-at must be timezone-aware: {value!r}"
        )
    return parsed.isoformat()


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result_payload(close_intent: dict) -> dict:
    return {
        "close_intent_id": close_intent["close_intent_id"],
        "position_id": close_intent["position_id"],
        "status": close_intent["status"],
        "simulated_close_order_id": close_intent["simulated_close_order_id"],
        "close_order_confirmed_event_id": close_intent[
            "close_order_confirmed_event_id"
        ],
        "dry_run": close_intent["dry_run"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--close-intent-id", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--confirmed-at", type=_parse_confirmed_at)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Accepted for clarity; dry-run is the only supported confirmation mode.",
    )
    args = parser.parse_args(argv)

    root_dir = Path(args.root_dir)
    confirmed_at = args.confirmed_at or _current_timestamp()
    try:
        state_store = StateStore(root_dir / "data/state/portfolio_state.json")
        ledger = LedgerAppender(root_dir)
        close_intent = confirm_close_order(
            state_store=state_store,
            ledger=ledger,
            execution_adapter=DryRunExecutionAdapter(),
            close_intent_id=args.close_intent_id,
            confirmed_at=confirmed_at,
        )
    except (ValueError, KeyError, CloseIntentTransitionError, LedgerValidationError) as exc:
        print(
            f"confirm close order failed for {args.close_intent_id!r}: {exc}",
            file=sys.stderr,
        )
        return 1

    payload = _result_payload(close_intent)
    if args.json_output:
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    else:
        print(
            "confirmed close order "
            f"close_intent_id={payload['close_intent_id']} "
            f"position_id={payload['position_id']} "
            f"status={payload['status']} "
            f"close_order_confirmed_event_id={payload['close_order_confirmed_event_id']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
