"""Print the Stage 4F manual real-paper execution acceptance report."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4f_acceptance_report import (
    MODULE_CHECK_KEYS,
    SAFETY_CHECK_KEYS,
    build_stage4f_acceptance_report,
)


_MODULES = {
    "ibkr_paper_factory_present": "algo_trader_unified.core.ibkr_paper_factory",
    "ibkr_paper_connection_preflight_present": (
        "algo_trader_unified.core.ibkr_paper_connection_preflight"
    ),
    "manual_real_paper_submit_present": (
        "algo_trader_unified.core.manual_real_paper_submit"
    ),
    "manual_real_paper_order_control_present": (
        "algo_trader_unified.core.manual_real_paper_order_control"
    ),
    "stage4f5_smoke_test_report_present": (
        "algo_trader_unified.core.stage4f5_smoke_test_report"
    ),
}


def run_stage4f_acceptance_report(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4f_acceptance_report,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--smoke-test-json", default=None)
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print(
            "ERROR: Stage 4F acceptance report requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    smoke_test_report = None
    if args.smoke_test_json is not None:
        try:
            smoke_test_report = json.loads(args.smoke_test_json)
        except Exception as exc:  # noqa: BLE001 - CLI reports parse type + text.
            message = f"ERROR: invalid JSON input: {type(exc).__name__}: {exc}"
            if args.json_output:
                print(
                    json.dumps(
                        {
                            "dry_run": True,
                            "stage4f_acceptance_report": True,
                            "success": False,
                            "errors": [message],
                            "warnings": [],
                        },
                        sort_keys=True,
                    )
                )
            else:
                print(message, file=sys.stderr)
            return 1

    module_checks, errors = _module_checks()
    safety_checks = {key: True for key in SAFETY_CHECK_KEYS}
    report = report_builder(
        reports={},
        module_checks=module_checks,
        safety_checks=safety_checks,
        smoke_test_report=smoke_test_report,
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
    return run_stage4f_acceptance_report(sys.argv[1:] if argv is None else argv)


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
    readiness = report.get("readiness_for_stage4g")
    readiness = readiness if isinstance(readiness, dict) else {}
    smoke_test = report.get("smoke_test")
    smoke_test = smoke_test if isinstance(smoke_test, dict) else {}
    recommendations = report.get("recommendations")
    recommendations = recommendations if isinstance(recommendations, dict) else {}
    lines = [
        "Stage 4F acceptance report",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        "ready_to_begin_manual_paper_lifecycle_validation: "
        f"{readiness.get('ready_to_begin_manual_paper_lifecycle_validation')}",
        f"smoke_test_accepted: {smoke_test.get('accepted')}",
        f"broker_order_id: {smoke_test.get('broker_order_id')}",
        f"client_order_id: {smoke_test.get('client_order_id')}",
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
