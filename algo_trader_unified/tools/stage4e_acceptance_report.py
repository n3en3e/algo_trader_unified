"""Print the Stage 4E paper-execution acceptance report."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4e_acceptance_report import (
    MODULE_CHECK_KEYS,
    SAFETY_CHECK_KEYS,
    build_stage4e_acceptance_report,
)


_MODULES = {
    "ibkr_paper_client_present": "algo_trader_unified.core.ibkr_paper_client",
    "ibkr_paper_readonly_preflight_present": (
        "algo_trader_unified.core.ibkr_paper_readonly_preflight"
    ),
    "ibkr_paper_execution_client_present": (
        "algo_trader_unified.core.ibkr_paper_execution_client"
    ),
    "paper_order_ticket_report_present": (
        "algo_trader_unified.core.paper_order_ticket_report"
    ),
    "manual_paper_submit_gate_present": (
        "algo_trader_unified.core.manual_paper_submit_gate"
    ),
}


def run_stage4e_acceptance_report(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4e_acceptance_report,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print(
            "ERROR: Stage 4E acceptance report requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    module_checks, errors = _module_checks()
    safety_checks = {key: True for key in SAFETY_CHECK_KEYS}
    report = report_builder(
        reports={},
        module_checks=module_checks,
        safety_checks=safety_checks,
        state_snapshot=None,
    )
    if errors:
        report["errors"] = list(report.get("errors", [])) + errors
        report["success"] = False

    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    return 0 if report.get("success") is True else 1


def main(argv: list[str] | None = None) -> int:
    return run_stage4e_acceptance_report(sys.argv[1:] if argv is None else argv)


def _module_checks() -> tuple[dict[str, bool], list[str]]:
    checks = {key: False for key in MODULE_CHECK_KEYS}
    errors: list[str] = []
    for key, module_name in _MODULES.items():
        try:
            checks[key] = importlib.util.find_spec(module_name) is not None
        except Exception as exc:  # noqa: BLE001 - report exceptions as data.
            checks[key] = False
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    return checks, errors


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_stage4f")
    readiness = readiness if isinstance(readiness, dict) else {}
    recommendations = report.get("recommendations")
    recommendations = recommendations if isinstance(recommendations, dict) else {}
    lines = [
        "Stage 4E acceptance report",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        "ready_to_begin_real_ibkr_paper_submit_planning: "
        f"{readiness.get('ready_to_begin_real_ibkr_paper_submit_planning')}",
    ]
    for key in ("blockers", "warnings"):
        values = readiness.get(key)
        if values:
            lines.append(f"{key}:")
            for value in values:
                lines.append(f"  - {value}")
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
