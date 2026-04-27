"""Run the market-open readiness scan once with test-safe diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.readiness import ReadinessManager
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.jobs.readiness import all_clear_health_snapshot, market_open_scan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--state-path", default="data/state/portfolio_state.json")
    args = parser.parse_args(argv)

    try:
        root = Path(args.root)
        state_store = StateStore(root / args.state_path)
        ledger = LedgerAppender(root)
        manager = ReadinessManager(state_store, ledger)
        strategy_ids = (S01_VOL_BASELINE, S02_VOL_ENHANCED)
        result = market_open_scan(
            readiness_manager=manager,
            strategy_ids=strategy_ids,
            health_snapshot=all_clear_health_snapshot(strategy_ids),
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"market open scan failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
