# Stage 4J Controlled Scheduled PAPER Operation

Stage 4J starts the first controlled scheduled PAPER operation path after Stage 4I completion.
It does not enable live trading, all-strategy automation, broker submission, order submission,
strategy scans, market data, contract qualification, or lifecycle transition execution.

## Stage 4J-1 Readiness

Stage 4J-1 is readiness/reporting only. It consumes the accepted Stage 4I-6 scheduler/lifecycle
activation acceptance report and decides whether the project is clean enough to build the first
controlled scheduled PAPER operation plan for one selected strategy.

The selected strategy is read from:

```python
stage4i6_acceptance_report["selected_strategy"]["selected_strategy_id"]
```

using safe dictionary traversal. Stage 4I completion is extracted safely from:

```python
stage4i6_acceptance_report.get("readiness_for_next_phase", {}).get("ready_to_proceed_after_stage4i")
```

Missing or malformed nested dictionaries do not crash the report; they become blockers or warnings.

Stage 4J-1 confirms:

- Stage 4I-6 is present, successful, and ready to proceed after Stage 4I.
- The selected strategy remains exactly one strategy and PAPER-only.
- Scheduler/lifecycle activation artifacts are accepted.
- Live trading remains disabled.
- All-strategy automation remains disabled.
- Broker submission remains separately gated.
- Strategy scan execution remains disabled.
- Lifecycle transition execution remains disabled.
- Market data and contract qualification remain disabled.
- Optional state, risk, scheduler, lifecycle, paper broker, market window, and registry snapshots do not contradict the selected strategy or safety boundary.

## Registry Parsing

The optional strategy registry snapshot validates that the selected strategy is paper-eligible.
Stage 4J-1 safely accepts these shapes:

- `{"strategies": [{"strategy_id": "...", "paper_eligible": true}]}`
- `{"paper_eligible_strategy_ids": ["...", "..."]}`
- `["strategy_id_a", "strategy_id_b"]`

Malformed registry entries are ignored safely. Registry parsing is protected by a `TypeError`
barrier, so malformed structures do not crash the report. A parse failure records a warning and
does not prove paper eligibility.

## Hard Boundaries

Stage 4J-1 must not perform direct scheduler registration, lifecycle execution, broker calls,
order submission, strategy scans, market data calls, contract qualification, state writes, ledger
writes, production deployment, or live trading.

Broker submission remains separately gated until an explicit broker-submission phase. Live trading
and all-strategy automation remain blocked. The `strategy_scan_execution_enabled` and
`lifecycle_transition_execution_enabled` flags remain disabled in Stage 4J-1.

## Next Stage

Stage 4J-2 is the controlled scheduled PAPER operation plan. It is still a plan/report phase, not
live trading. Broker submission remains its own explicit future phase.

## Stage 4J-2 Operation Plan

Stage 4J-2 is controlled scheduled PAPER operation planning/reporting only. It consumes the
accepted Stage 4J-1 readiness report and creates a deterministic operation plan for exactly one
selected strategy.

The selected strategy is still read from the accepted readiness artifact:

```python
stage4j1_readiness_report.get("selected_strategy", {}).get("selected_strategy_id")
```

Stage 4J-2 does not re-select, rank, infer, or enable strategies. It does not enable all sleeves.

The plan creates a deterministic `operation_id` from the selected strategy id, normalized cadence,
and generated date string. It does not use random values, UUIDs, wall-clock timestamps, or
high-granularity time values for the id.

The `operation_window_config` is operator planning input only. Cadence normalization safely handles
missing or `None` values without calling string methods on `None`; missing cadence defaults
conservatively to `once`. Unsupported cadence labels block readiness. `dry_run_only=false` blocks
readiness.

The report includes:

- `controlled_operation_scope`
- `operation_plan.operation_id`
- `operation_plan.proposed_operation_window`
- `operation_plan.proposed_pre_operation_gates`
- `operation_plan.proposed_operation_flow`
- `operation_plan.proposed_post_operation_checks`
- `operation_plan.disabled_components`
- `operation_plan.required_inputs_for_4J3`
- `readiness_for_stage4j3`

The proposed operation flow is a disabled preview. Every step is structured and deterministic, but
no target component is called. Flow payloads must contain only primitive JSON-safe values: strings,
integers, floats, booleans, `null`, lists, and dictionaries. Payloads must not contain raw datetimes,
tuples, enums, custom objects, callables, or path objects.

Stage 4J-2 explicitly does not perform direct scheduler registration, lifecycle execution, broker
calls, order submission, strategy scans, market data, contract qualification, state mutation, ledger
writes, production deployment, or live trading. Broker submission remains separately gated. Live
trading and all-strategy automation remain blocked. `strategy_scan_execution_enabled` and
`lifecycle_transition_execution_enabled` remain disabled in Stage 4J-2.

Optional snapshots are read-only validation context. If supplied, they must not contradict the
selected strategy or safety boundary. Missing optional snapshots may leave readiness true with
warnings, because Stage 4J-3 must re-check activation artifacts, scheduler/lifecycle state, risk
controls, paper broker configuration, market window, and state reconciliation immediately before any
dry-run operation.

## Stage 4J-3

Stage 4J-3 is the controlled scheduled PAPER operation dry run. It is still not live trading, and
broker submission remains its own explicit phase.

## Example

```bash
python3 -m algo_trader_unified.tools.stage4j1_controlled_paper_operation_readiness \
  --dry-run-only \
  --json \
  --stage4i6-acceptance-json '{"dry_run": true, "stage4i6_scheduler_lifecycle_activation_acceptance_report": true, "selected_strategy": {"selected_strategy_id": "S01", "paper_only": true, "one_strategy_only": true}, "payload_checks": {"broker_submission_disabled": true, "strategy_scan_execution_disabled": true, "lifecycle_transition_execution_disabled": true, "market_data_disabled": true, "contract_qualification_disabled": true}, "safety_checks": {"no_live_trading": true, "no_all_strategy_enablement": true, "no_broker_submission_enabled": true, "no_market_data": true, "no_contract_qualification": true, "no_order_submission": true, "no_strategy_scan_execution": true, "no_lifecycle_transition_execution": true, "no_direct_scheduler_registration": true, "no_direct_lifecycle_execution": true, "no_state_write": true, "no_ledger_write": true}, "readiness_for_next_phase": {"ready_to_proceed_after_stage4i": true}, "success": true, "errors": [], "warnings": []}'
```

```bash
python3 -m algo_trader_unified.tools.stage4j2_controlled_paper_operation_plan \
  --dry-run-only \
  --json \
  --stage4j1-readiness-json '{"dry_run": true, "stage4j1_controlled_paper_operation_readiness_report": true, "selected_strategy": {"selected_strategy_id": "S01", "paper_only": true, "one_strategy_only": true}, "payload_checks": {"broker_submission_disabled": true, "strategy_scan_execution_disabled": true, "lifecycle_transition_execution_disabled": true, "market_data_disabled": true, "contract_qualification_disabled": true}, "safety_checks": {"no_live_trading": true, "no_all_strategy_enablement": true, "no_broker_submission_enabled": true, "no_market_data": true, "no_contract_qualification": true, "no_order_submission": true, "no_strategy_scan_execution": true, "no_lifecycle_transition_execution": true, "no_direct_scheduler_registration": true, "no_direct_lifecycle_execution": true, "no_state_write": true, "no_ledger_write": true}, "readiness_for_stage4j2": {"ready_to_build_controlled_paper_operation_plan": true}, "success": true, "errors": [], "warnings": []}' \
  --operation-window-config-json '{"cadence": "once", "dry_run_only": true}'
```
