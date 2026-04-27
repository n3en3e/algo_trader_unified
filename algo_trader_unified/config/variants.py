"""Immutable strategy variant configuration model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from algo_trader_unified.config.portfolio import (
    S01_VOL_BASELINE,
    S02_VOL_ENHANCED,
)


ExecutionMode = Literal[
    "disabled",
    "shadow",
    "manual_alert",
    "paper_only",
    "paper_proxy_for_live",
    "live_enabled",
]

VALID_EXECUTION_MODES = {
    "disabled",
    "shadow",
    "manual_alert",
    "paper_only",
    "paper_proxy_for_live",
    "live_enabled",
}


@dataclass(frozen=True)
class StrategyVariantConfig:
    strategy_id: str
    display_name: str
    legacy_source: str
    engine_type: str
    sleeve_id: str
    nominal_research_allocation: int
    execution_mode: ExecutionMode
    setup_modules: tuple[str, ...] = ()
    instruments: tuple[str, ...] = ()
    risk_profile: dict[str, Any] = field(default_factory=dict)
    sizing_profile: dict[str, Any] = field(default_factory=dict)
    entry_rules: dict[str, Any] = field(default_factory=dict)
    exit_rules: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    reporting_tags: tuple[str, ...] = ()
    allow_paper_proxy_fallback_sizing: bool = False

    def __post_init__(self) -> None:
        if self.execution_mode not in VALID_EXECUTION_MODES:
            raise ValueError(f"Invalid execution_mode: {self.execution_mode}")


S01_CONFIG = StrategyVariantConfig(
    strategy_id=S01_VOL_BASELINE,
    display_name="Vol Baseline",
    legacy_source="A1",
    engine_type="vol_selling",
    sleeve_id="VOL",
    nominal_research_allocation=90_000,
    execution_mode="paper_only",
    instruments=("XSP",),
    params={
        "iv_rank_min": 30,
        "vix_gate_min": None,
        "target_dte": 45,
        "profit_target_pct": 0.50,
        "stop_loss_mult": 2.0,
        "dte_close_threshold": 21,
    },
)

S02_CONFIG = StrategyVariantConfig(
    strategy_id=S02_VOL_ENHANCED,
    display_name="Vol Enhanced",
    legacy_source="B1",
    engine_type="vol_selling",
    sleeve_id="VOL",
    nominal_research_allocation=90_000,
    execution_mode="paper_only",
    instruments=("XSP",),
    params={
        "iv_rank_min": 30,
        "vix_gate_min": 13,
        "target_dte": 45,
        "profit_target_pct": 0.50,
        "stop_loss_mult": 2.0,
        "dte_close_threshold": 21,
    },
)
