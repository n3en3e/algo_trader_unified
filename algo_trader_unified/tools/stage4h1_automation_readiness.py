"""Build the Stage 4H-1 controlled automation readiness report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4h1_automation_readiness import (
    build_stage4h1_automation_readiness_report,
)


def run_stage4h1_automation_readiness(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4h1_automation_readiness_report,
) -> int:
    if "--dry-run-only" not in argv:
        print(
            "ERROR: Stage 4H-1 automation readiness requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--stage4g-acceptance-json", required=True)
    parser.add_argument("--state-snapshot-json")
    parser.add_argument("--strategy-registry-json")
    parser.add_argument("--risk-snapshot-json")
    parser.add_argument("--module-checks-json")
    parser.add_argument("--safety-checks-json")
    args = parser.parse_args(argv)

    try:
        stage4g_acceptance_report = json.loads(args.stage4g_acceptance_json)
        state_snapshot = _loads_optional(args.state_snapshot_json)
        strategy_registry_snapshot = _loads_optional(args.strategy_registry_json)
        risk_snapshot = _loads_optional(args.risk_snapshot_json)
        module_checks = _loads_optional(args.module_checks_json)
        safety_checks = _loads_optional(args.safety_checks_json)
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports parse type + text.
        message = f"ERROR: invalid JSON input: {type(exc).__name__}: {exc}"
        if args.json_output:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "stage4h1_automation_readiness_report": True,
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

    report = report_builder(
        stage4g_acceptance_report=stage4g_acceptance_report,
        module_checks=module_checks,
        safety_checks=safety_checks,
        state_snapshot=state_snapshot,
        strategy_registry_snapshot=strategy_registry_snapshot,
        risk_snapshot=risk_snapshot,
    )
    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    readiness = report.get("readiness_for_stage4h2", {})
    return 0 if readiness.get("ready_to_build_automation_wiring_preview") is True else 1


def main(argv: list[str] | None = None) -> int:
    return run_stage4h1_automation_readiness(sys.argv[1:] if argv is None else argv)


def _loads_optional(value: str | None) -> Any:
    return json.loads(value) if value is not None else None


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_stage4h2", {})
    artifacts = report.get("artifact_checks", {})
    state = report.get("state_checks", {})
    strategies = report.get("strategy_candidate_checks", {})
    lines = [
        "Stage 4H-1 controlled automation readiness",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        "ready_to_build_automation_wiring_preview: "
        f"{readiness.get('ready_to_build_automation_wiring_preview')}",
        f"stage4g_acceptance_ready: {artifacts.get('stage4g_acceptance_ready')}",
        f"active_halt: {state.get('active_halt')}",
        f"paper_eligible_strategy_count: {strategies.get('paper_eligible_strategy_count')}",
    ]
    blockers = readiness.get("blockers")
    if blockers:
        lines.append("blockers:")
        for value in blockers:
            lines.append(f"  - {value}")
    for key in ("warnings", "errors"):
        values = report.get(key)
        if values:
            lines.append(f"{key}:")
            for value in values:
                lines.append(f"  - {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
