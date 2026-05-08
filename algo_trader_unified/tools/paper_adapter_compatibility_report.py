"""Print the Stage 4D-4 paper adapter compatibility report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from algo_trader_unified.core.paper_adapter_compatibility_report import (
    build_paper_adapter_compatibility_report,
)
from algo_trader_unified.core.state_store import StateStore, StateStoreCorruptError


def run_paper_adapter_compatibility_report(
    argv: list[str] | tuple[str, ...],
    *,
    state_store_factory: Callable[[Path], Any] = StateStore,
    report_builder: Callable[..., dict[str, Any]] = build_paper_adapter_compatibility_report,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print(
            "ERROR: paper adapter compatibility report requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    try:
        root = Path(args.root)
        state_store_path = root / "data" / "state" / "portfolio_state.json"
        if state_store_factory is StateStore and not state_store_path.exists():
            order_intents: list[dict] = []
        else:
            state_store = state_store_factory(state_store_path)
            order_intents = state_store.list_order_intents()
        strategy_ids = sorted(
            {
                intent.get("strategy_id")
                for intent in order_intents
                if isinstance(intent, dict)
                and isinstance(intent.get("strategy_id"), str)
                and intent.get("strategy_id").strip()
            }
        )
        report = report_builder(order_intents=order_intents, strategy_ids=strategy_ids)
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
    return run_paper_adapter_compatibility_report(sys.argv[1:] if argv is None else argv)


def _format_human(report: dict[str, Any]) -> str:
    aggregate = report.get("aggregate")
    aggregate = aggregate if isinstance(aggregate, dict) else {}
    recommendations = report.get("recommendations")
    recommendations = recommendations if isinstance(recommendations, dict) else {}
    lines = [
        "Paper adapter compatibility report",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        f"total_intents_seen: {aggregate.get('total_intents_seen')}",
        f"total_intents_valid: {aggregate.get('total_intents_valid')}",
        f"total_intents_invalid: {aggregate.get('total_intents_invalid')}",
        f"all_intents_compatible: {aggregate.get('all_intents_compatible')}",
    ]
    dominant = aggregate.get("dominant_invalid_reason")
    if dominant:
        lines.append(f"dominant_invalid_reason: {dominant}")
    steps = recommendations.get("ordered_next_steps")
    if steps:
        lines.append("ordered_next_steps:")
        for step in steps:
            lines.append(f"  - {step}")
    do_not = recommendations.get("do_not_do_yet")
    if do_not:
        lines.append("do_not_do_yet:")
        for item in do_not:
            lines.append(f"  - {item}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
