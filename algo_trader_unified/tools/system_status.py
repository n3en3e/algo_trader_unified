"""Read-only system lifecycle status summary."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from typing import Any

from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.tools._formatting import dumps_json, state_path


ORDER_INTENT_STATUSES = [
    "created",
    "submitted",
    "confirmed",
    "filled",
    "position_opened",
    "expired",
    "cancelled",
]
POSITION_STATUSES = ["open"]
CLOSE_INTENT_STATUSES = ["created", "submitted", "confirmed"]


def _dict_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [deepcopy(record) for record in value.values() if isinstance(record, dict)]
    if isinstance(value, list):
        return [deepcopy(record) for record in value if isinstance(record, dict)]
    return []


def _load_records(
    root_dir: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    portfolio_state_path = state_path(root_dir)
    if not portfolio_state_path.exists():
        return [], [], []
    state_store = StateStore(portfolio_state_path)
    state = state_store.state
    return (
        _dict_records(state.get("order_intents", {})),
        _dict_records(state.get("positions", {})),
        _dict_records(state.get("close_intents", {})),
    )


def _counts_by_status(
    records: list[dict[str, Any]],
    default_statuses: list[str],
) -> dict[str, int]:
    counts = {status: 0 for status in default_statuses}
    for record in records:
        status = record.get("status")
        if not isinstance(status, str):
            continue
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_summary(
    order_intents: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    close_intents: list[dict[str, Any]] | None = None,
    *,
    strategy_id: str | None = None,
) -> dict[str, Any]:
    close_intents = close_intents or []
    if strategy_id is not None:
        order_intents = [
            record for record in order_intents if record.get("strategy_id") == strategy_id
        ]
        positions = [record for record in positions if record.get("strategy_id") == strategy_id]
        close_intents = [
            record for record in close_intents if record.get("strategy_id") == strategy_id
        ]

    order_counts = _counts_by_status(order_intents, ORDER_INTENT_STATUSES)
    position_counts = _counts_by_status(positions, POSITION_STATUSES)
    close_counts = _counts_by_status(close_intents, CLOSE_INTENT_STATUSES)

    created = order_counts["created"]
    submitted = order_counts["submitted"]
    confirmed = order_counts["confirmed"]
    filled = order_counts["filled"]
    position_opened = order_counts["position_opened"]
    stranded = submitted + confirmed + filled

    return {
        "order_intent_counts_by_status": order_counts,
        "position_counts_by_status": position_counts,
        "close_intent_counts_by_status": close_counts,
        "open_positions_count": position_counts["open"],
        "created_close_intents_count": close_counts["created"],
        "submitted_close_intents_count": close_counts["submitted"],
        "confirmed_close_intents_count": close_counts["confirmed"],
        "created_order_intents_count": created,
        "submitted_order_intents_count": submitted,
        "confirmed_order_intents_count": confirmed,
        "filled_order_intents_count": filled,
        "position_opened_intents_count": position_opened,
        "unresolved_order_intents_count": created + stranded,
        "stranded_order_intents_count": stranded,
        "stranded_filled_intents_count": filled,
        "stranded_submitted_intents_count": submitted,
        "stranded_confirmed_intents_count": confirmed,
        "total_order_intents_count": len(order_intents),
        "total_positions_count": len(positions),
        "total_close_intents_count": len(close_intents),
        "filters": {
            "strategy_id": strategy_id,
        },
    }


def format_human(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "System status",
            "Order intents:",
            f"  total: {summary['total_order_intents_count']}",
            f"  created: {summary['created_order_intents_count']}",
            f"  submitted: {summary['submitted_order_intents_count']}",
            f"  confirmed: {summary['confirmed_order_intents_count']}",
            f"  filled: {summary['filled_order_intents_count']}",
            f"  position_opened: {summary['position_opened_intents_count']}",
            f"  unresolved: {summary['unresolved_order_intents_count']}",
            "Close intents:",
            f"  total: {summary['total_close_intents_count']}",
            f"  created: {summary['created_close_intents_count']}",
            f"  submitted: {summary['submitted_close_intents_count']}",
            f"  confirmed: {summary['confirmed_close_intents_count']}",
            "Positions:",
            f"  total: {summary['total_positions_count']}",
            f"  open: {summary['open_positions_count']}",
            "Stranded:",
            f"  total: {summary['stranded_order_intents_count']}",
            f"  submitted: {summary['stranded_submitted_intents_count']}",
            f"  confirmed: {summary['stranded_confirmed_intents_count']}",
            f"  filled: {summary['stranded_filled_intents_count']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--strategy-id")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    try:
        order_intents, positions, close_intents = _load_records(args.root_dir)
        summary = build_summary(
            order_intents,
            positions,
            close_intents,
            strategy_id=args.strategy_id,
        )
    except Exception as exc:
        print(f"system status failed: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        print(dumps_json(summary))
        return 0

    print(format_human(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
