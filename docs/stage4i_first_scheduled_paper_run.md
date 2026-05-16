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

## Stage 4I-3 Scheduled Run Dry Run

Stage 4I-3 is scheduled run dry-run/reporting only. It consumes the accepted
Stage 4I-2 scheduled run plan and produces a deterministic `dry_run_trace` that
mirrors the 4I-2 `proposed_execution_flow`.

The `dry_run_trace` preserves the exact planned flow order. Each trace item
safely carries through the planned `target_function` or `target_component`
without resolving it to a callable and without calling it. Every step is marked
as simulated and keeps `would_execute: false`, `would_submit: false`,
`would_write_state: false`, `would_write_ledger: false`, and
`would_register_scheduler: false`.

The `pre_run_snapshot_check` uses a deterministic static checklist:
`state_snapshot`, `risk_snapshot`, `scheduler_snapshot`, `lifecycle_snapshot`,
`paper_broker_snapshot`, and `market_window_snapshot`.

Stage 4I-3 simulates each planned run segment only. `strategy_scan_preview`
does not call the strategy scan. `signal_to_intent_preview` does not create an
intent. `intent_to_ticket_preview` does not create a ticket.
`paper_order_submission_gate_preview` does not submit a paper order.
`state_ledger_tracking_preview` does not write state or ledger.
`alert_report_preview` does not emit real alerts unless a future phase
explicitly permits it. `post_run_reconciliation_preview` does not reconcile
broker truth.

Stage 4I-3 does not register scheduler jobs, execute lifecycle transitions,
wire into the daemon, call strategy scan or run methods, create real intents,
create real tickets, call a broker, submit paper orders, submit live orders,
fetch market data, qualify contracts, mutate `StateStore`, write ledger events,
enable live trading, or enable all strategies. Broker submission remains
separately gated. Live trading and all-strategy automation remain blocked.

## Stage 4I-4 Scheduled Run Activation Gate

Stage 4I-4 is the one-strategy scheduler/lifecycle activation gate. It consumes
the accepted Stage 4I-3 dry-run report and produces an activation gate report.
It is reporting only.

4I-4 requires these exact operator acknowledgements before readiness for 4I-5:

- `I understand this activates the scheduler and lifecycle for one PAPER strategy only.`
- `I understand this does not enable live trading.`
- `I understand this does not enable all strategies.`
- `I understand this does not enable broker order submission.`
- `I verified risk controls and kill switches are active.`
- `I verified PAPER broker configuration is active.`
- `I verified the market window is correct for the run.`

## Stage 4I-5 Scheduler/Lifecycle Activation Executor

Stage 4I-5 is the one-strategy scheduler/lifecycle activation executor. It
consumes the accepted Stage 4I-4 activation gate report and applies the
activation through injected writer abstractions only.

4I-5 is the first stage in 4I that may perform a write to activate scheduler
and lifecycle automation, but only for one strategy in PAPER mode, and only
after all gates pass. The core executor requires
`allow_scheduler_lifecycle_activation=true`, a valid and ready Stage 4I-4
report, clean safety snapshots, and the exact operator acknowledgements.

The core executor does not instantiate production scheduler, lifecycle, or
audit objects. Writes are allowed only through injected writer abstractions such
as `scheduler_activation_writer.activate_scheduler(...)`,
`lifecycle_activation_writer.activate_lifecycle(...)`, and an optional
`audit_writer.append_scheduler_lifecycle_activation_audit(...)`.

The activation payloads for the scheduler and lifecycle writers strictly
hardcode `strategy_scan_execution_enabled: false` and
`lifecycle_transition_execution_enabled: false`. The `applied_operations` in
the report include `target` labels (`scheduler_activation`,
`lifecycle_activation`, `audit`) to support manual rollback review. Partial
failure is reported transparently, and `rollback_required=true` is set if an
early write succeeds but a later one fails. No automated rollback is supported.

Stage 4I-5 does not submit orders, call a broker, scan strategies, execute
lifecycle transitions, enable live trading, or enable all strategies. Broker
submission remains separately gated until a later scheduled run phase
explicitly permits it.

## Stage 4I-6 Scheduler/Lifecycle Activation Acceptance

Stage 4I-6 is the scheduler/lifecycle activation acceptance report for the
first controlled scheduled PAPER automation run. It consumes the accepted
Stage 4I-5 activation executor report and optional read-only scheduler,
lifecycle, audit, activation, state, risk, paper broker, and market window
snapshots.

4I-6 is acceptance/reporting only. It verifies that the 4I-5 scheduler and
lifecycle activation payloads match the selected PAPER strategy, remain scoped
to a single strategy, and keep `live_trading_enabled`,
`all_strategies_enabled`, `broker_submission_enabled`,
`strategy_scan_execution_enabled`, and
`lifecycle_transition_execution_enabled` false. It also safely validates
`applied_operations` and `skipped_operations` target labels so malformed
operation records cannot crash the report.

Optional scheduler and lifecycle activation snapshots are checked only for
explicit contradictions. Optional audit snapshots may match the selected
strategy with `selected_strategy_id` at the event root or under nested
`payload` or `data` dictionaries. Unexpected audit schemas are ignored with
warnings rather than executed or trusted blindly.

Stage 4I-6 does not directly register APScheduler jobs, execute lifecycle
transitions, wire into the daemon, call strategy scans, call broker APIs, fetch
market data, qualify contracts, submit orders, poll order status, or cancel
orders. Broker submission remains separately gated. Live trading and
all-strategy automation remain blocked.

If accepted, Stage 4I is complete. The next decision point is whether to
proceed to the controlled scheduled PAPER operation / first actual scheduled
run path. That path is still PAPER only: no live trading, and broker submission
remains its own explicit phase.

```bash
python3 -m algo_trader_unified.tools.stage4i6_scheduler_lifecycle_activation_acceptance \
  --dry-run-only \
  --json \
  --stage4i5-executor-json '{...}' \
  --scheduler-activation-snapshot-json '{...}' \
  --lifecycle-activation-snapshot-json '{...}' \
  --audit-snapshot-json '{...}' \
  --activation-snapshot-json '{...}' \
  --state-snapshot-json '{...}' \
  --risk-snapshot-json '{...}' \
  --scheduler-snapshot-json '{...}' \
  --lifecycle-snapshot-json '{...}' \
  --paper-broker-snapshot-json '{...}' \
  --market-window-snapshot-json '{...}'
```

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

```bash
python3 -m algo_trader_unified.tools.stage4i3_scheduled_run_dry_run \
  --dry-run-only \
  --json \
  --stage4i2-plan-json '{...}' \
  --activation-snapshot-json '{...}' \
  --state-snapshot-json '{...}' \
  --risk-snapshot-json '{...}' \
  --scheduler-snapshot-json '{...}' \
  --lifecycle-snapshot-json '{...}' \
  --paper-broker-snapshot-json '{...}' \
  --market-window-snapshot-json '{...}'
```
