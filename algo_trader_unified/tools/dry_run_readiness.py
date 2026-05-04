"""Operator dry-run readiness gate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from algo_trader_unified.core.ledger_reader import LedgerReader
from algo_trader_unified.core.readiness_report import build_dry_run_readiness_report
from algo_trader_unified.core.state_store import StateStore, StateStoreCorruptError


def parse_now(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--now must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def format_human(report: dict) -> str:
    summary = report["summary"]
    heading = report["status"].upper()
    return "\n".join(
        [
            f"{heading}",
            f"checked_at: {report['checked_at']}",
            f"blocking_issues: {len(report['blocking_issues'])}",
            f"warnings: {len(report['warnings'])}",
            "Lifecycle counts:",
            f"  order_intents: {summary['total_order_intents_count']}",
            f"  active_order_intents: {summary['active_order_intents_count']}",
            f"  close_intents: {summary['total_close_intents_count']}",
            f"  active_close_intents: {summary['active_close_intents_count']}",
            f"  positions: {summary['total_positions_count']}",
            f"  open_positions: {summary['open_positions_count']}",
            f"  closed_positions: {summary['closed_positions_count']}",
            f"next_action: {report['next_action']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--now")
    parser.add_argument("--dry-run", action="store_true", help="Accepted no-op; readiness is always dry-run.")
    args = parser.parse_args(argv)

    try:
        now = parse_now(args.now)
    except argparse.ArgumentTypeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    root = Path(args.root_dir)
    state_file = root / "data/state/portfolio_state.json"
    state_store = None
    if state_file.exists():
        try:
            state_store = StateStore(state_file)
        except StateStoreCorruptError as exc:
            state_store = exc

    report = build_dry_run_readiness_report(
        root_dir=root,
        state_store=state_store,
        ledger_reader=LedgerReader.from_root(root),
        now=now,
    )

    if args.json_output:
        print(json.dumps(report, separators=(",", ":"), sort_keys=True))
    else:
        print(format_human(report))
    return 1 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
