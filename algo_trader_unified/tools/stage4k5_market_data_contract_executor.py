"""Build the Stage 4K-5 controlled market data and contract qualification executor report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4k5_market_data_contract_executor import (
    build_stage4k5_market_data_contract_executor_report,
)


def run_stage4k5_market_data_contract_executor(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4k5_market_data_contract_executor_report,
) -> int:
    if "--dry-run-only" not in argv:
        print(
            "ERROR: Stage 4K-5 controlled executor CLI requires --dry-run-only and performs validation only without injected providers",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--stage4k4-gate-json", required=True)
    parser.add_argument("--scheduler-activation-snapshot-json")
    parser.add_argument("--lifecycle-activation-snapshot-json")
    parser.add_argument("--activation-snapshot-json")
    parser.add_argument("--state-snapshot-json")
    parser.add_argument("--risk-snapshot-json")
    parser.add_argument("--scheduler-snapshot-json")
    parser.add_argument("--lifecycle-snapshot-json")
    parser.add_argument("--paper-broker-snapshot-json")
    parser.add_argument("--market-window-snapshot-json")
    args = parser.parse_args(argv)

    try:
        stage4k4_gate_report = json.loads(args.stage4k4_gate_json)
        scheduler_activation_snapshot = _loads_optional(args.scheduler_activation_snapshot_json)
        lifecycle_activation_snapshot = _loads_optional(args.lifecycle_activation_snapshot_json)
        activation_snapshot = _loads_optional(args.activation_snapshot_json)
        state_snapshot = _loads_optional(args.state_snapshot_json)
        risk_snapshot = _loads_optional(args.risk_snapshot_json)
        scheduler_snapshot = _loads_optional(args.scheduler_snapshot_json)
        lifecycle_snapshot = _loads_optional(args.lifecycle_snapshot_json)
        paper_broker_snapshot = _loads_optional(args.paper_broker_snapshot_json)
        market_window_snapshot = _loads_optional(args.market_window_snapshot_json)
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports parse type + text.
        message = f"ERROR: invalid JSON input: {type(exc).__name__}: {exc}"
        if args.json_output:
            print(
                json.dumps(
                    {
                        "dry_run": False,
                        "stage4k5_market_data_contract_executor_report": True,
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
        stage4k4_execution_gate_report=stage4k4_gate_report,
        controlled_market_data_provider=None,
        controlled_contract_qualification_provider=None,
        scheduler_activation_snapshot=scheduler_activation_snapshot,
        lifecycle_activation_snapshot=lifecycle_activation_snapshot,
        activation_snapshot=activation_snapshot,
        state_snapshot=state_snapshot,
        risk_snapshot=risk_snapshot,
        scheduler_snapshot=scheduler_snapshot,
        lifecycle_snapshot=lifecycle_snapshot,
        paper_broker_snapshot=paper_broker_snapshot,
        market_window_snapshot=market_window_snapshot,
    )
    warning = "provider execution requires injected providers from Python/tests; CLI validation does not instantiate providers"
    if warning not in report.get("warnings", []):
        report = dict(report)
        report["warnings"] = [*report.get("warnings", []), warning]
        readiness = dict(report.get("readiness_for_stage4k6", {}))
        readiness["warnings"] = [*readiness.get("warnings", []), warning]
        report["readiness_for_stage4k6"] = readiness
    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    readiness = report.get("readiness_for_stage4k6", {})
    return 0 if isinstance(readiness, dict) and readiness.get("ready_to_build_market_data_contract_acceptance") is True else 1


def main(argv: list[str] | None = None) -> int:
    return run_stage4k5_market_data_contract_executor(sys.argv[1:] if argv is None else argv)


def _loads_optional(value: str | None) -> Any:
    return json.loads(value) if value is not None else None


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_stage4k6", {})
    selected = report.get("selected_strategy", {})
    operation = report.get("operation", {})
    market_data = report.get("market_data_execution_results", {})
    contract = report.get("contract_qualification_execution_results", {})
    lines = [
        "Stage 4K-5 controlled market data and contract qualification executor",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        f"ready_to_build_market_data_contract_acceptance: {readiness.get('ready_to_build_market_data_contract_acceptance')}",
        f"selected_strategy_id: {selected.get('selected_strategy_id')}",
        f"operation_id: {operation.get('operation_id')}",
        f"market_data_provider_called: {market_data.get('provider_called')}",
        f"contract_qualification_provider_called: {contract.get('provider_called')}",
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
