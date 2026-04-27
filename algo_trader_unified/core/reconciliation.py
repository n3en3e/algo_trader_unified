"""Phase 1 reconciliation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReconciliationResult:
    clean: bool
    unknown_exposure: dict[str, float]
    dirty_strategies: list[str]
    event_id: str | None


def _broker_aggregate_exposure(broker_stub: Any) -> dict[str, float]:
    if hasattr(broker_stub, "get_aggregate_exposure"):
        return dict(broker_stub.get_aggregate_exposure())
    if hasattr(broker_stub, "aggregate_exposure"):
        return dict(broker_stub.aggregate_exposure)
    return {}


def reconcile_check(broker_stub: Any, state_store: Any, ledger_appender: Any) -> ReconciliationResult:
    broker_exposure = _broker_aggregate_exposure(broker_stub)
    bot_exposure = state_store.bot_attributed_exposure()
    unknown: dict[str, float] = {}
    for key, broker_qty in broker_exposure.items():
        residual = float(broker_qty) - float(bot_exposure.get(key, 0.0))
        if residual != 0:
            unknown[key] = residual
    dirty_strategies = []
    event_id = None
    if unknown:
        event = ledger_appender.append(
            event_type="RECONCILIATION_FAILED",
            strategy_id="ACCOUNT",
            execution_mode="disabled",
            source_module="core.reconciliation",
            position_id=None,
            opportunity_id=None,
            payload={
                "reason": "UNKNOWN_BROKER_EXPOSURE",
                "unknown_exposure": unknown,
                "broker_exposure": broker_exposure,
                "bot_attributed_exposure": bot_exposure,
            },
        )
        event_id = event.event_id
        dirty_strategies = ["ACCOUNT"]
    return ReconciliationResult(
        clean=not unknown,
        unknown_exposure=unknown,
        dirty_strategies=dirty_strategies,
        event_id=event_id,
    )

