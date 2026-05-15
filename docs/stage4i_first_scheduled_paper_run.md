# Stage 4I First Scheduled PAPER Run

Stage 4I begins the first controlled scheduled PAPER automation run work. The
stage is intentionally narrow: it moves from an accepted one-strategy PAPER
activation artifact toward a later run plan without enabling production
automation.

## Stage 4I-1 Readiness

Stage 4I-1 is readiness/reporting only. It reads the accepted Stage 4H-6
one-strategy activation acceptance report and optional read-only snapshots for
activation state, runtime state, risk, scheduler, lifecycle, paper broker
configuration, and the market window.

Stage 4I-1 does not submit orders, call a broker, scan strategies, register
scheduler jobs, register lifecycle jobs, execute lifecycle transitions, enable
live trading, or enable all strategies. Broker submission remains separately
gated until a later scheduled run phase explicitly permits it.

An accepted Stage 4H-6 activation report is required. The selected strategy ID
is read deterministically from the accepted 4H-6 activation payload/checks using
safe dictionary traversal, preferring
`activation_payload_checks.selected_strategy_id`. Stage 4I remains limited to
one selected strategy only.

Live trading and all-strategy automation remain blocked. If no market window
snapshot is supplied, the operator must manually verify exchange hours and
holiday schedules before proceeding to Stage 4I-2.

## Stage 4I-2 Run Plan

Stage 4I-2 is the first scheduled PAPER run planning report. It consumes the
accepted Stage 4I-1 readiness report and produces a deterministic, structured
plan for exactly one selected strategy. It is planning/reporting only.

Stage 4I-2 does not submit orders, call a broker, poll order status, cancel
orders, fetch market data, qualify contracts, scan strategies, register
scheduler jobs, register lifecycle jobs, execute lifecycle transitions, enable
live trading, or enable all-strategy automation. Broker submission remains
separately gated.

The report includes a structured `proposed_schedule` and
`proposed_execution_flow`. Both are disabled: schedule registration is false,
execution is false, scheduler job enablement is false, and every flow step is a
preview with `would_execute: false`. The broker-submission gate preview also
keeps `would_submit: false`.

Run-window inputs are operator planning hints only. Cadence validation trims
whitespace and compares case-insensitively, so values such as ` DAILY ` normalize
to `daily`. Unsupported cadences block readiness for the next dry-run stage.
The `proposed_run_id` is deterministic and avoids UUIDs, random values, and
high-granularity timestamp components.

Live trading and all-strategy automation remain blocked throughout Stage 4I-2.

## Next Stage

Stage 4I-3 is the scheduled run dry run. It still does not enable live trading.
Before 4I-3, operators must re-check runtime state, the activation artifact,
risk controls, paper broker config, scheduler state, lifecycle state, and the
market window.

## Example

```bash
python3 -m algo_trader_unified.tools.stage4i1_scheduled_paper_run_readiness \
  --dry-run-only \
  --json \
  --stage4h6-acceptance-json '{"stage4h6_one_strategy_activation_acceptance_report":true,"activation_payload_checks":{"selected_strategy_id":"S01","paper_only":true,"one_strategy_only":true,"live_trading_disabled":true,"all_strategies_disabled":true,"broker_submission_disabled":true},"readiness_for_next_phase":{"ready_to_build_first_scheduled_paper_automation_run":true},"success":true,"errors":[]}'
```

```bash
python3 -m algo_trader_unified.tools.stage4i2_scheduled_paper_run_plan \
  --dry-run-only \
  --json \
  --stage4i1-readiness-json '{"stage4i1_scheduled_paper_run_readiness_report":true,"selected_strategy":{"selected_strategy_id":"S01","paper_only":true,"one_strategy_only":true},"safety_checks":{"no_live_trading":true,"no_all_strategy_enablement":true,"no_broker_submission_enabled":true,"no_market_data":true,"no_contract_qualification":true,"no_order_submission":true},"readiness_for_stage4i2":{"ready_to_build_first_scheduled_run_plan":true},"success":true,"errors":[]}' \
  --run-window-config-json '{"cadence":" DAILY ","dry_run_only":true,"run_time":"09:45"}'
```
