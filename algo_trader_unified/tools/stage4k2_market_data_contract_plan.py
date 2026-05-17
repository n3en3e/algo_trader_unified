"""Build the Stage 4K-2 market data and contract qualification plan report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4k2_market_data_contract_plan import (
    build_stage4k2_market_data_contract_plan_report,
)


def run_stage4k2_market_data_contract_plan(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4k2_market_data_contract_plan_report,
) -> int:
    if "--dry-run-only" not in argv:
        print(
            "ERROR: Stage 4K-2 market data and contract qualification plan requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--stage4k1-readiness-json", required=True)
    parser.add_argument("--strategy-contract-requirements-json")
    parser.add_argument("--market-data-capability-snapshot-json")
    parser.add_argument("--contract-qualification-capability-snapshot-json")
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
        stage4k1_readiness_report = json.loads(args.stage4k1_readiness_json)
        strategy_contract_requirements_snapshot = _loads_optional(args.strategy_contract_requirements_json)
        market_data_capability_snapshot = _loads_optional(args.market_data_capability_snapshot_json)
        contract_qualification_capability_snapshot = _loads_optional(args.contract_qualification_capability_snapshot_json)
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
                        "dry_run": True,
                        "stage4k2_market_data_contract_plan_report": True,
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
        stage4k1_readiness_report=stage4k1_readiness_report,
        strategy_contract_requirements_snapshot=strategy_contract_requirements_snapshot,
        market_data_capability_snapshot=market_data_capability_snapshot,
        contract_qualification_capability_snapshot=contract_qualification_capability_snapshot,
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
    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    readiness = report.get("readiness_for_stage4k3", {})
    return 0 if isinstance(readiness, dict) and readiness.get("ready_to_build_market_data_contract_dry_run") is True else 1


def main(argv: list[str] | None = None) -> int:
    return run_stage4k2_market_data_contract_plan(sys.argv[1:] if argv is None else argv)


def _loads_optional(value: str | None) -> Any:
    return json.loads(value) if value is not None else None


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_stage4k3", {})
    selected = report.get("selected_strategy", {})
    operation = report.get("operation", {})
    market_data_plan = report.get("market_data_plan", {})
    contract_plan = report.get("contract_qualification_plan", {})
    lines = [
        "Stage 4K-2 market data and contract qualification plan",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        f"ready_to_build_market_data_contract_dry_run: {readiness.get('ready_to_build_market_data_contract_dry_run') if isinstance(readiness, dict) else None}",
        f"selected_strategy_id: {selected.get('selected_strategy_id') if isinstance(selected, dict) else None}",
        f"operation_id: {operation.get('operation_id') if isinstance(operation, dict) else None}",
        f"may_fetch_market_data_in_4k2: {market_data_plan.get('may_fetch_market_data_in_4k2') if isinstance(market_data_plan, dict) else None}",
        f"may_qualify_contracts_in_4k2: {contract_plan.get('may_qualify_contracts_in_4k2') if isinstance(contract_plan, dict) else None}",
    ]
    blockers = readiness.get("blockers") if isinstance(readiness, dict) else None
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
