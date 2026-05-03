"""Scheduler wrapper for dry-run management scans."""

from __future__ import annotations

from typing import Any

from algo_trader_unified.core.management import (
    default_management_signal_provider,
    run_management_scan,
)


def run_management_scan_job(
    *,
    strategy_id: str,
    state_store,
    ledger,
    now,
    management_signal_provider=None,
) -> dict[str, Any]:
    provider = management_signal_provider or default_management_signal_provider
    return run_management_scan(
        state_store=state_store,
        ledger=ledger,
        strategy_id=strategy_id,
        management_signal_provider=provider,
        now=now,
    )
