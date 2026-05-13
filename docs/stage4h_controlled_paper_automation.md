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

Stage 4H-2 consumes the 4H-1 readiness report plus optional read-only strategy,
scheduler, lifecycle, risk, and state snapshots. It answers which exact
one-strategy or one-sleeve automated paper path would be previewed later, which
components would be involved, which gates must stay closed, and what must be
verified before Stage 4H-3.

4H-2 is automation wiring preview only. It does not enable scheduler automation,
does not enable lifecycle automation, does not register jobs, does not execute
lifecycle transitions, does not submit orders, does not call the broker, does
not request market data, does not qualify contracts, does not mutate state, does
not write ledger events, and does not enable live trading.

The 4H-2 preview must remain one strategy or sleeve only. If multiple
paper-eligible candidates are present, the report stays blocked until either
`explicit_preview_strategy_id` is passed separately or a read-only snapshot
`preview_strategy_id` is supplied. Passing `explicit_preview_strategy_id` does
not mutate the strategy registry snapshot.

The proposed scheduler and lifecycle wiring previews are structured JSON-safe
dictionaries/lists so Stage 4H-3 can parse them deterministically. They remain
disabled in 4H-2: proposed scheduler jobs use `disabled: true`,
`would_register: false`, and `would_execute: false`; proposed lifecycle flows
use `would_execute: false`.

Next stage after 4H-2: 4H-3 is automation wiring dry run. It is still no order
submission and still no live trading.

```bash
python3 -m algo_trader_unified.tools.stage4h2_automation_wiring_preview \
  --dry-run-only \
  --json \
  --stage4h1-readiness-json '{"stage4h1_automation_readiness_report":true,"strategy_candidate_checks":{"candidate_strategy_ids":["S01_VOL_BASELINE","S02_VOL_BASELINE"]},"readiness_for_stage4h2":{"ready_to_build_automation_wiring_preview":true,"blockers":[]},"success":true,"errors":[]}' \
  --strategy-registry-json '{"paper_eligible_strategy_ids":["S01_VOL_BASELINE","S02_VOL_BASELINE"]}' \
  --explicit-preview-strategy-id "S01_VOL_BASELINE" \
  --risk-snapshot-json '{"kill_switch_available":true,"hard_halt_available":true,"daily_loss_limit_available":true}'
```

```bash
python3 -m algo_trader_unified.tools.stage4h1_automation_readiness \
  --dry-run-only \
  --json \
  --stage4g-acceptance-json '{"stage4g6_lifecycle_write_acceptance_report":true,"artifact_checks":{"rollback_not_required":true},"readiness_for_stage4h":{"ready_to_begin_controlled_automated_paper_trading_launch":true,"blockers":[]},"success":true,"errors":[]}' \
  --strategy-registry-json '{"paper_eligible_strategy_ids":["S01_VOL_BASELINE"]}' \
  --risk-snapshot-json '{"kill_switch_available":true,"hard_halt_available":true,"daily_loss_limit_available":true}'
```
