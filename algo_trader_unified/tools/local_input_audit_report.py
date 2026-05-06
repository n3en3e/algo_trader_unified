"""Print the dry-run local input audit report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.local_input_audit_report import build_local_input_audit_report
from algo_trader_unified.core.readiness_provider import DefaultHealthSnapshotProvider
from algo_trader_unified.core.state_store import StateStore, StateStoreCorruptError
from algo_trader_unified.tools.strategy_realism_report import _UnavailableStateStore


def run_local_input_audit_report(
    argv: list[str] | tuple[str, ...],
    *,
    state_store_factory: Callable[[Path], Any] = StateStore,
    readiness_provider_factory: Callable[..., Callable[[], Any]] = DefaultHealthSnapshotProvider,
    report_builder: Callable[..., dict[str, Any]] = build_local_input_audit_report,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print("ERROR: local input audit report requires --dry-run-only", file=sys.stderr)
        return 1

    root = Path(args.root)
    snapshots_dir = root / "data" / "snapshots"
    halt_state_path = root / "data" / "state" / "halt_state.json"
    state_store_path = root / "data" / "state" / "portfolio_state.json"
    try:
        if state_store_factory is StateStore and not state_store_path.exists():
            state_store = _UnavailableStateStore(state_store_path)
        else:
            state_store = state_store_factory(state_store_path)
        strategy_ids = [S01_VOL_BASELINE, S02_VOL_ENHANCED]
        readiness_provider = readiness_provider_factory(
            state_store=state_store,
            halt_state_path=halt_state_path,
            snapshots_dir=snapshots_dir,
            max_staleness_minutes=15,
            strategy_ids=strategy_ids,
        )
        report = report_builder(
            strategy_ids=strategy_ids,
            state_store=state_store,
            readiness_provider=readiness_provider,
            snapshots_dir=snapshots_dir,
            halt_state_path=halt_state_path,
            iv_store=root / "data" / "snapshots" / "iv_rank.json",
            vix_snapshot_path=root / "data" / "snapshots" / "vix.json",
            market_calendar_path=root / "data" / "snapshots" / "market_calendar.json",
        )
    except StateStoreCorruptError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    return 0 if report.get("success") is True else 1


def main(argv: list[str] | None = None) -> int:
    return run_local_input_audit_report(sys.argv[1:] if argv is None else argv)


def _format_human(report: dict[str, Any]) -> str:
    aggregate = report.get("aggregate") if isinstance(report.get("aggregate"), dict) else {}
    lines = [
        "Local input audit report",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        f"dominant_input_issue: {aggregate.get('dominant_input_issue')}",
        f"missing_input_count: {aggregate.get('missing_input_count')}",
        f"stale_input_count: {aggregate.get('stale_input_count')}",
    ]
    per_strategy = report.get("per_strategy")
    if isinstance(per_strategy, dict):
        lines.append("per_strategy:")
        for strategy_id in sorted(per_strategy):
            item = per_strategy[strategy_id]
            if not isinstance(item, dict):
                continue
            lines.append(
                f"  {strategy_id}: likely_input_issue={item.get('likely_input_issue')} "
                f"input_blockers={item.get('input_blockers')}"
            )
    recommendations = report.get("recommendations")
    steps = recommendations.get("ordered_next_steps") if isinstance(recommendations, dict) else []
    if steps:
        lines.append("ordered_next_steps:")
        for step in steps:
            lines.append(f"  - {step}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
