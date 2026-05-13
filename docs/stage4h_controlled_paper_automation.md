# Stage 4H Controlled Paper Automation

Stage 4H is the controlled automated paper trading launch path. It is the first
stage that designs automation wiring, but it remains paper-only and explicitly
does not enable live trading.

Stage 4H-1 is readiness and reporting only. It consumes the accepted Stage 4G-6
manual lifecycle write acceptance report plus optional injected snapshots for
modules, safety flags, state, strategy registry, and risk controls. It answers
whether the repo is ready to start Stage 4H-2 automation wiring preview design.

Accepted Stage 4G is required before any Stage 4H automation work. The 4G-6
report must be present, successful, ready for Stage 4H, clean of executor
blockers, and free of rollback requirements.

4H should start with one strategy or sleeve only. 4H-1 may report sorted
paper-eligible candidates, but it does not select the first strategy to automate,
does not tune strategy configuration, and does not recommend enabling all
strategies at once.

No scheduler or lifecycle automation is enabled in 4H-1. This stage does not
wire daemon jobs, scheduler jobs, lifecycle jobs, broker submission, broker
status checks, cancellation, market data, contract qualification, StateStore
mutation, or ledger writes.

The strategy registry snapshot supports these read-only JSON shapes:

- `{"strategies": [{"strategy_id": "S01", "paper_eligible": true}]}`
- `{"candidates": [{"strategy_id": "S01", "paper_eligible": true}]}`
- `{"paper_eligible_strategy_ids": ["S01"]}`

When a risk snapshot is supplied, it must include truthy
`kill_switch_available`, `hard_halt_available`, and
`daily_loss_limit_available`. A supplied snapshot with missing core controls,
explicit false core controls, or `risk_bypass_enabled: true` blocks readiness.
Missing risk snapshots warn so 4H-2/4H-3 can verify risk controls before wiring.

Next stage: 4H-2 is automation wiring preview. It is still preview-only, still
does not submit orders, and still does not enable live trading.

```bash
python3 -m algo_trader_unified.tools.stage4h1_automation_readiness \
  --dry-run-only \
  --json \
  --stage4g-acceptance-json '{"stage4g6_lifecycle_write_acceptance_report":true,"artifact_checks":{"rollback_not_required":true},"readiness_for_stage4h":{"ready_to_begin_controlled_automated_paper_trading_launch":true,"blockers":[]},"success":true,"errors":[]}' \
  --strategy-registry-json '{"paper_eligible_strategy_ids":["S01_VOL_BASELINE"]}' \
  --risk-snapshot-json '{"kill_switch_available":true,"hard_halt_available":true,"daily_loss_limit_available":true}'
```
