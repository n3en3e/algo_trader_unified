"""Run the Stage 4F-1 real IBKR paper factory preflight."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.ibkr_paper_factory import (
    build_ibkr_paper_factory_preflight_report,
)


def run_ibkr_paper_factory_preflight(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_ibkr_paper_factory_preflight_report,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4004)
    parser.add_argument("--client-id", type=int, default=7)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--trading-mode", default="PAPER")
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print(
            "ERROR: IBKR paper factory preflight requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    report = report_builder(
        config={
            "host": args.host,
            "port": args.port,
            "client_id": args.client_id,
            "account_id": args.account_id,
            "trading_mode": args.trading_mode,
            "readonly": True,
        }
    )

    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    return 0 if report.get("success") is True and not report.get("errors") else 1


def main(argv: list[str] | None = None) -> int:
    return run_ibkr_paper_factory_preflight(sys.argv[1:] if argv is None else argv)


def _format_human(report: dict[str, Any]) -> str:
    config = report.get("config", {})
    factory = report.get("factory", {})
    readiness = report.get("readiness_for_stage4f2", {})
    lines = [
        "IBKR paper factory preflight",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        f"trading_mode: {config.get('trading_mode')}",
        f"host: {config.get('host')}",
        f"port: {config.get('port')}",
        f"client_id: {config.get('client_id')}",
        f"account_id: {config.get('account_id')}",
        f"readonly: {config.get('readonly')}",
        f"paper_config_valid: {config.get('paper_config_valid')}",
        f"allow_real_ibkr_default: {factory.get('allow_real_ibkr_default')}",
        f"would_connect: {factory.get('would_connect')}",
        "ready_to_build_manual_real_paper_submit_command: "
        f"{readiness.get('ready_to_build_manual_real_paper_submit_command')}",
    ]
    for key in ("errors", "warnings"):
        values = report.get(key)
        if values:
            lines.append(f"{key}:")
            for value in values:
                lines.append(f"  - {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
