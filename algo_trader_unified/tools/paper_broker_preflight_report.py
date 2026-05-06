"""Print the Stage 4D-2 paper broker adapter preflight report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from algo_trader_unified.core.paper_broker_preflight_report import (
    build_paper_broker_preflight_report,
)
from algo_trader_unified.core.state_store import StateStore, StateStoreCorruptError


def run_paper_broker_preflight_report(
    argv: list[str] | tuple[str, ...],
    *,
    state_store_factory: Callable[[Path], Any] = StateStore,
    report_builder: Callable[..., dict[str, Any]] = build_paper_broker_preflight_report,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print(
            "ERROR: paper broker preflight report requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    try:
        root = Path(args.root)
        state_store_path = root / "data" / "state" / "portfolio_state.json"
        state_snapshot = None
        if state_store_factory is StateStore and not state_store_path.exists():
            state_snapshot = None
        else:
            state_store = state_store_factory(state_store_path)
            state = getattr(state_store, "state", None)
            state_snapshot = state if isinstance(state, dict) else None
        report = report_builder(state_snapshot=state_snapshot)
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
    return run_paper_broker_preflight_report(sys.argv[1:] if argv is None else argv)


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_next_phase")
    readiness = readiness if isinstance(readiness, dict) else {}
    recommendations = report.get("recommendations")
    recommendations = recommendations if isinstance(recommendations, dict) else {}
    lines = [
        "Paper broker preflight report",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        "ready_to_design_ibkr_paper_adapter: "
        f"{readiness.get('ready_to_design_ibkr_paper_adapter')}",
    ]
    blockers = readiness.get("blockers")
    if blockers:
        lines.append("blockers:")
        for blocker in blockers:
            lines.append(f"  - {blocker}")
    warnings = report.get("warnings")
    if warnings:
        lines.append("warnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")
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
