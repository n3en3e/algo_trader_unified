"""Print StateStore status summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from algo_trader_unified.core.state_store import StateStore, StateStoreCorruptError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-path",
        default="data/state/portfolio_state.json",
        help="Path to StateStore JSON",
    )
    args = parser.parse_args(argv)
    path = Path(args.state_path)
    if not path.exists():
        print(f"StateStore not found: {path}")
        return 1
    try:
        store = StateStore(path)
    except StateStoreCorruptError as exc:
        print(f"StateStore corrupt: {exc}")
        return 1
    summary = store.summary()
    print(json.dumps(summary, indent=2, sort_keys=True))
    halt = summary.get("halt_state")
    if halt:
        print(f"halt scope={halt.get('scope')} tier={halt.get('tier')}")
    latest = summary.get("latest_reconciliation")
    if latest:
        print(f"latest reconciliation: {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

