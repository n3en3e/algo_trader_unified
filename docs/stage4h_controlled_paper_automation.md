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

Stage 4H-3 consumes the 4H-2 automation wiring preview and produces a
controlled automation wiring dry-run report. It answers what exact one-strategy
paper automation wiring could later be activated by Stage 4H-4 while proving
nothing is enabled in 4H-3.

4H-3 is dry-run/reporting only. It does not register scheduler jobs, does not
wire lifecycle jobs, does not wire the daemon, does not scan strategies, does
not create real broker tickets, does not submit orders, does not call the
broker, does not request market data, does not qualify contracts, does not
mutate StateStore, does not write ledger events, and does not enable live
trading.

All 4H-3 dry-run operations must have `would_execute: false`. Scheduler dry-run
operations must also have `would_register: false` and remain disabled. Broker
or order-path dry-run operations must have `would_submit: false`. Every
operation includes a `target_function` or `target_component` string plus a
JSON-safe `payload` dictionary so Stage 4H-4 can map the dry-run packet to real
function pointers deterministically. 4H-3 never resolves those strings into
callables and never calls them.

The selected strategy is read directly from
`stage4h2_wiring_preview_report["strategy_selection"]["selected_preview_strategy_id"]`.
4H-3 does not re-derive that value from a strategy registry snapshot.

Stage 4H-4 consumes the 4H-3 wiring dry-run report and builds a one-strategy
automation enablement gate report. It answers whether the dry run proved enough
to prepare a tightly gated future Stage 4H-5 activation packet for exactly one
selected strategy or sleeve.

4H-4 is enablement-gate reporting only. It does not actually enable scheduler
automation, does not actually enable lifecycle automation, does not register
jobs, does not wire the daemon, does not execute lifecycle transitions, does
not scan strategies, does not submit orders, does not call the broker, does not
request market data, does not qualify contracts, does not mutate StateStore,
does not write ledger events, and does not enable live trading.

The selected strategy is read directly from the Stage 4H-3
`selected_strategy.selected_preview_strategy_id` field. The activation
candidate remains one strategy only and paper only. Live trading and
all-strategy automation remain blocked.

4H-4 requires these exact operator acknowledgements before readiness for 4H-5
can pass:

- `I understand this enables automated PAPER trading for one strategy only.`
- `I understand this does not enable live trading.`
- `I understand this does not enable all strategies.`
- `I verified risk controls and kill switches are available.`
- `I verified PAPER broker configuration is active.`
- `I understand scheduler/lifecycle activation must remain limited to the selected strategy.`

Omitted acknowledgements are treated as an empty list and block readiness
safely. A single combined string containing the acknowledgement text does not
pass; each acknowledgement must be supplied as an exact list item after
whitespace trimming.

A paper broker snapshot is required for readiness. It must confirm PAPER mode,
must not indicate `paper_trading: false`, must use the paper port convention,
and must not show live trading or broker submission automation already enabled.

If a state snapshot is supplied, an active halt or unresolved
`NEEDS_RECONCILIATION` blocks readiness. `active_intents_count > 0` also blocks
readiness unless `active_intents_safe_for_enablement: true` is explicitly
supplied, in which case the report may pass the state gate but still emits a
warning. `open_positions_count` alone is not treated as unknown exposure.

Next stage after 4H-4: 4H-5 is the one-strategy activation executor. It is
still no live trading. It may enable only the selected strategy in PAPER mode
if all gates pass immediately before activation.

```bash
python3 -m algo_trader_unified.tools.stage4h4_one_strategy_enablement_gate \
  --dry-run-only \
  --json \
  --stage4h3-dry-run-json '{"stage4h3_automation_wiring_dry_run_report":true,"selected_strategy":{"selected_preview_strategy_id":"S01_VOL_BASELINE"},"dry_run_packet":{"available":true,"scheduler_dry_run_operations":[{"target_component":"Scheduler","payload":{"strategy_id":"S01_VOL_BASELINE"},"would_execute":false,"would_register":false}],"lifecycle_dry_run_operations":[{"target_component":"LifecycleRouter","payload":{"strategy_id":"S01_VOL_BASELINE"},"would_execute":false}],"ticket_to_paper_submit_dry_run_operations":[{"target_function":"build_ibkr_paper_order_plan","payload":{"strategy_id":"S01_VOL_BASELINE"},"would_execute":false,"would_submit":false}]},"readiness_for_stage4h4":{"ready_to_build_one_strategy_automation_enablement_gate":true,"blockers":[]},"success":true,"errors":[]}' \
  --state-snapshot-json '{"active_halt":false,"unresolved_needs_reconciliation_count":0,"active_intents_count":0}' \
  --risk-snapshot-json '{"kill_switch_available":true,"hard_halt_available":true,"daily_loss_limit_available":true,"risk_bypass_enabled":false}' \
  --scheduler-snapshot-json '{}' \
  --lifecycle-snapshot-json '{}' \
  --paper-broker-snapshot-json '{"mode":"PAPER","paper_trading":true,"ibkr_port":4004,"live_trading_enabled":false,"broker_submission_enabled":false}' \
  --ack "I understand this enables automated PAPER trading for one strategy only." \
  --ack "I understand this does not enable live trading." \
  --ack "I understand this does not enable all strategies." \
  --ack "I verified risk controls and kill switches are available." \
  --ack "I verified PAPER broker configuration is active." \
  --ack "I understand scheduler/lifecycle activation must remain limited to the selected strategy."
```

```bash
python3 -m algo_trader_unified.tools.stage4h3_automation_wiring_dry_run \
  --dry-run-only \
  --json \
  --stage4h2-preview-json '{"stage4h2_automation_wiring_preview_report":true,"strategy_selection":{"selected_preview_strategy_id":"S01_VOL_BASELINE"},"wiring_preview":{"available":true,"proposed_scheduler_wiring_preview":{"jobs":[{"job_id":"stage4h3_dry_run_S01_VOL_BASELINE","strategy_id":"S01_VOL_BASELINE","disabled":true,"would_register":false,"would_execute":false,"paper_only":true}]},"proposed_lifecycle_wiring_preview":{"flows":[{"name":"signal_to_intent","would_execute":false,"paper_only":true}]}},"readiness_for_stage4h3":{"ready_to_build_automation_wiring_dry_run":true,"blockers":[]},"success":true,"errors":[]}' \
  --risk-snapshot-json '{"kill_switch_available":true,"hard_halt_available":true,"daily_loss_limit_available":true}' \
  --paper-broker-snapshot-json '{"mode":"PAPER","paper_trading":true,"ibkr_port":4004,"live_trading_enabled":false,"broker_submission_enabled":false}'
```

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
