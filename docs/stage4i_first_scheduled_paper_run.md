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

## Next Stage

Stage 4I-2 is the first scheduled PAPER run plan. It still does not enable live
trading. Before any run planning, it must re-check state, activation artifact,
risk controls, paper broker config, scheduler state, lifecycle state, and the
market window.

## Example

```bash
python3 -m algo_trader_unified.tools.stage4i1_scheduled_paper_run_readiness \
  --dry-run-only \
  --json \
  --stage4h6-acceptance-json '{"stage4h6_one_strategy_activation_acceptance_report":true,"activation_payload_checks":{"selected_strategy_id":"S01","paper_only":true,"one_strategy_only":true,"live_trading_disabled":true,"all_strategies_disabled":true,"broker_submission_disabled":true},"readiness_for_next_phase":{"ready_to_build_first_scheduled_paper_automation_run":true},"success":true,"errors":[]}'
```
