"""Print the dry-run strategy realism baseline report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.ledger_reader import LedgerReader
from algo_trader_unified.core.state_store import StateStore, StateStoreCorruptError
from algo_trader_unified.core.strategy_realism_report import build_strategy_realism_report
from algo_trader_unified.core.readiness_provider import DefaultHealthSnapshotProvider


def run_strategy_realism_report(
    argv: list[str] | tuple[str, ...],
    *,
    state_store_factory: Callable[[Path], Any] = StateStore,
    ledger_reader_factory: Callable[[Path], Any] = LedgerReader.from_root,
    readiness_provider_factory: Callable[..., Callable[[], Any]] = DefaultHealthSnapshotProvider,
    report_builder: Callable[..., dict[str, Any]] = build_strategy_realism_report,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--root", default=".")
    parser.add_argument("--session-date")
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print("ERROR: strategy realism report requires --dry-run-only", file=sys.stderr)
        return 1

    try:
        session = _parse_session_date(args.session_date)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    root = Path(args.root)
    snapshots_dir = root / "data" / "snapshots"
    halt_state_path = root / "data" / "state" / "halt_state.json"
    try:
        state_store_path = root / "data" / "state" / "portfolio_state.json"
        if state_store_factory is StateStore and not state_store_path.exists():
            state_store = _UnavailableStateStore(state_store_path)
        else:
            state_store = state_store_factory(state_store_path)
        ledger_reader = ledger_reader_factory(root)
        readiness_provider = readiness_provider_factory(
            state_store=state_store,
            halt_state_path=halt_state_path,
            snapshots_dir=snapshots_dir,
            max_staleness_minutes=15,
            strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
        )
        report = report_builder(
            ledger_reader=ledger_reader,
            state_store=state_store,
            readiness_provider=readiness_provider,
            snapshots_dir=snapshots_dir,
            halt_state_path=halt_state_path,
            strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
            session_date=session,
        )
    except StateStoreCorruptError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("success") is True else 1


def main(argv: list[str] | None = None) -> int:
    return run_strategy_realism_report(sys.argv[1:] if argv is None else argv)


def _parse_session_date(value: str | None) -> date | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed.date()


class _UnavailableStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.state = None

    def get_readiness(self, strategy_id: str) -> None:
        return None

    def list_positions(self) -> list[dict]:
        return []

    def list_order_intents(self) -> list[dict]:
        return []

    def list_close_intents(self) -> list[dict]:
        return []


if __name__ == "__main__":
    raise SystemExit(main())
