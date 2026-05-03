"""Close a dry-run position from a filled close intent."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from algo_trader_unified.core.ledger import LedgerAppender, LedgerValidationError
from algo_trader_unified.core.positions import close_position_from_filled_intent
from algo_trader_unified.core.state_store import CloseIntentTransitionError, StateStore


def _parse_closed_at(value: str) -> str:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--closed-at must be a parseable ISO timestamp: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(f"--closed-at must be timezone-aware: {value!r}")
    return parsed.isoformat()


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result_payload(position: dict) -> dict:
    return {
        "position_id": position["position_id"],
        "close_intent_id": position["close_intent_id"],
        "status": position["status"],
        "position_closed_event_id": position["position_closed_event_id"],
        "realized_pnl": position["realized_pnl"],
        "dry_run": position["dry_run"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--close-intent-id", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--closed-at", type=_parse_closed_at)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Accepted for clarity; dry-run is the only supported position-close mode.",
    )
    args = parser.parse_args(argv)

    root_dir = Path(args.root_dir)
    closed_at = args.closed_at or _current_timestamp()
    try:
        state_store = StateStore(root_dir / "data/state/portfolio_state.json")
        ledger = LedgerAppender(root_dir)
        position = close_position_from_filled_intent(
            state_store=state_store,
            ledger=ledger,
            close_intent_id=args.close_intent_id,
            closed_at=closed_at,
        )
    except (
        ValueError,
        KeyError,
        TypeError,
        CloseIntentTransitionError,
        LedgerValidationError,
    ) as exc:
        print(f"close position failed for {args.close_intent_id!r}: {exc}", file=sys.stderr)
        return 1

    payload = _result_payload(position)
    if args.json_output:
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    else:
        print(
            "closed position "
            f"position_id={payload['position_id']} "
            f"close_intent_id={payload['close_intent_id']} "
            f"status={payload['status']} "
            f"position_closed_event_id={payload['position_closed_event_id']} "
            f"realized_pnl={payload['realized_pnl']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
