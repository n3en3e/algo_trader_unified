"""List positions from StateStore."""

from __future__ import annotations

import argparse
import sys

from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.tools._formatting import (
    apply_limit,
    compact_table,
    dumps_json,
    limit_arg,
    sort_records,
    state_path,
)


POSITION_COLUMNS = [
    "position_id",
    "strategy_id",
    "symbol",
    "status",
    "opened_at",
    "updated_at",
    "quantity",
    "entry_price",
    "dry_run",
]


def _load_records(args: argparse.Namespace) -> list[dict]:
    portfolio_state_path = state_path(args.root_dir)
    if not portfolio_state_path.exists():
        return []
    state_store = StateStore(portfolio_state_path)
    records = state_store.list_positions(strategy_id=args.strategy_id, status=args.status)
    if args.symbol is not None:
        records = [record for record in records if record.get("symbol") == args.symbol]
    records = sort_records(records, args.sort, reverse=args.reverse)
    return apply_limit(records, args.limit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--strategy-id")
    parser.add_argument("--status")
    parser.add_argument("--symbol")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--limit", type=limit_arg)
    parser.add_argument(
        "--sort",
        choices=("opened_at", "updated_at", "status", "strategy_id", "symbol"),
    )
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args(argv)

    try:
        records = _load_records(args)
    except Exception as exc:
        print(f"list positions failed: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        print(
            dumps_json(
                {
                    "positions": records,
                    "count": len(records),
                    "filters": {
                        "strategy_id": args.strategy_id,
                        "status": args.status,
                        "symbol": args.symbol,
                    },
                }
            )
        )
        return 0

    if not records:
        print("No positions found.")
        return 0

    print(compact_table(records, POSITION_COLUMNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
