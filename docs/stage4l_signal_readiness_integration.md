# Stage 4L Signal Readiness Integration

Stage 4L follows accepted Stage 4K market data and contract qualification. It prepares the accepted Stage 4K outputs for a future signal-readiness gate while keeping broker submission, order placement, intents, tickets, state writes, ledger writes, live trading, and all-strategy automation disabled.

## Stage 4L-1 Purpose

Stage 4L-1 is planning and reporting only. It consumes the accepted Stage 4K-6 market data and contract qualification acceptance report and decides whether those accepted outputs can become deterministic, read-only proposed inputs for a future single-strategy, PAPER-only signal-readiness validation phase.

Stage 4L-1 does not call providers. It does not fetch market data or qualify contracts. It does not call `request_controlled_market_data` or `qualify_controlled_contracts`, and direct IBKR methods remain forbidden:

- `reqMktData`
- `qualifyContracts`
- `reqContractDetails`

Accepted market data and contract qualification outputs become `proposed_signal_readiness_inputs`. These inputs preserve accepted result order, remain native JSON-safe dictionaries/lists, and keep all execution permissions disabled.

Stage 4L-1 does not run strategy scans or calculate signals. It does not create intents, tickets, orders, broker submissions, state writes, ledger writes, live trading, or all-strategy automation. Broker submission remains separately gated.

If `signal_schema_snapshot.expected_input_sections` is supplied, it must be a list before iteration. Non-list values or non-string section names block readiness safely instead of raising type errors.

## Proposed Inputs

`proposed_signal_readiness_inputs` contains the selected strategy id and operation id from Stage 4K-6, the accepted market data results, the accepted contract qualification results, and strict disabled flags for strategy scan, signal execution, intent creation, ticket creation, order submission, broker submission, state writes, ledger writes, live trading, and all-strategy enablement.

The payload is read-only and is intended only for future Stage 4L-2 validation. Stage 4L-1 readiness means ready to build the Stage 4L-2 signal readiness gate. It does not mean full PAPER trading is active.

## Proposed 4L-2 Validation Flow

Stage 4L-1 reports a deterministic `proposed_4l2_validation_flow` with exactly seven strictly ordered steps using `sequence_number` values 1 through 7:

1. `validate_stage4k_acceptance`
2. `validate_selected_strategy_scope`
3. `validate_accepted_market_data_inputs`
4. `validate_accepted_contract_qualification_inputs`
5. `validate_signal_schema_requirements`
6. `validate_no_execution_permissions`
7. `prepare_stage4l2_signal_readiness_gate_inputs`

Every step is non-executing: no strategy execution, signal calculation, intent creation, ticket creation, order submission, state write, ledger write, broker submission, or live trading.

## Stage 4L-2 Purpose

Stage 4L-2 is gate/reporting only. It consumes the accepted Stage 4L-1 plan and determines whether Stage 4L-3 may call an injected controlled signal-readiness validator for the selected strategy only.

Stage 4L-2 does not call validators, calculate signals, run strategy scans, call providers, fetch market data, qualify contracts, create intents or tickets, submit orders, write state or ledger, register scheduler jobs, execute lifecycle transitions, enable live trading, or enable all-strategy automation. Broker submission remains separately gated.

The inherited `proposed_4l2_validation_flow` is validated by exact step-name chain of custody, not just by length or sequence numbers. The expected Stage 4L-1 step names are:

1. `validate_stage4k_acceptance`
2. `validate_selected_strategy_scope`
3. `validate_accepted_market_data_inputs`
4. `validate_accepted_contract_qualification_inputs`
5. `validate_signal_schema_requirements`
6. `validate_no_execution_permissions`
7. `prepare_stage4l2_signal_readiness_gate_inputs`

Stage 4L-2 also validates that `proposed_signal_readiness_inputs` is read-only, single-strategy, PAPER-only, JSON-safe, and has at least one accepted input category. Disabled safety flags must be strict native boolean `false`; serialized strings such as `"False"` block readiness.

Stage 4L-2 does not call `request_controlled_market_data` or `qualify_controlled_contracts`, and direct IBKR methods remain forbidden:

- `reqMktData`
- `qualifyContracts`
- `reqContractDetails`

Operator acknowledgements are exact list-item strings. Missing acknowledgements block readiness, and extra strings warn but do not substitute for required acknowledgements.

If Stage 4L-2 passes, it emits `proposed_4l3_signal_readiness_payload` for Stage 4L-3 only. That payload allows only a future injected controlled signal-readiness validator call; it keeps strategy scan, signal execution, intent creation, ticket creation, order submission, broker submission, state writes, ledger writes, provider execution, direct market-data calls, direct contract-qualification calls, live trading, and all-strategy automation disabled.

Stage 4L-2 also emits a deterministic `proposed_4l3_validation_flow` with exactly seven strictly ordered steps using `sequence_number` values 1 through 7:

1. `validate_stage4l1_plan`
2. `validate_operator_acknowledgements`
3. `validate_signal_readiness_inputs`
4. `validate_strategy_registry_scope`
5. `validate_signal_schema_scope`
6. `validate_no_execution_permissions`
7. `prepare_stage4l3_controlled_validator_payload`

Stage 4L-2 readiness means ready to execute the controlled signal-readiness validator in Stage 4L-3. It does not mean full PAPER trading is active.

## Stage 4L-3 Purpose

Stage 4L-3 is a controlled signal-readiness validator report. It consumes the accepted Stage 4L-2 gate report and may call only the injected controlled signal-readiness validator for the selected strategy. The validator receives only the sanitized `proposed_4l3_signal_readiness_payload`, not the full Stage 4L-2 report and not runtime snapshots.

Stage 4L-3 does not run real strategy scans. It does not calculate executable signals, create intents, create tickets, submit orders, call providers, fetch market data, qualify contracts, write state, write ledger entries, register scheduler jobs, execute lifecycle transitions, enable live trading, or enable all-strategy automation. It does not call `request_controlled_market_data` or `qualify_controlled_contracts`, and direct IBKR methods remain forbidden:

- `reqMktData`
- `qualifyContracts`
- `reqContractDetails`

The inherited `proposed_4l3_validation_flow` is validated by exact step-name chain of custody from Stage 4L-2. Validation checks the exact seven expected names, strict sequence numbers, and non-execution flags. Malformed, shorter, or longer arrays are handled with safe iteration and bounds checks so bad upstream input blocks readiness without `IndexError`.

Validator output is treated as untrusted input. Required result safety flags must be native JSON booleans; a serialized string such as `"False"` blocks readiness. The validator result may safely conclude that the selected strategy is not ready for signal evaluation. That is not a code failure, but it blocks Stage 4L-4 readiness.

Validator exceptions are caught, flattened to `ExceptionType: message`, and sanitized to remove raw newline characters and memory-address-style object representations. Stage 4L-3 readiness means ready to build Stage 4L-4 signal readiness acceptance. It does not mean full PAPER trading is active.

```bash
python3 -m algo_trader_unified.tools.stage4l3_controlled_signal_readiness_validator \
  --dry-run-only \
  --json \
  --stage4l2-gate-json '{...}' \
  --strategy-registry-snapshot-json '{"candidate_strategy_ids": ["S01"]}' \
  --signal-schema-snapshot-json '{"selected_strategy_id": "S01", "expected_input_sections": ["selected_strategy_id", "operation_id"]}' \
  --state-snapshot-json '{"active_halt": false, "unresolved_needs_reconciliation_count": 0, "active_intents_count": 0}' \
  --risk-snapshot-json '{"kill_switch_available": true, "hard_halt_available": true, "daily_loss_limit_available": true, "risk_bypass_enabled": false}' \
  --paper-broker-snapshot-json '{"mode": "PAPER", "paper_trading": true, "ibkr_port": 4004, "live_trading_enabled": false, "broker_submission_enabled": false}'
```

The CLI is validation-only unless a safe injected validator is supplied by tests or callers using the core function directly. It does not fake validator success.

## Example

```bash
python3 -m algo_trader_unified.tools.stage4l1_signal_readiness_plan \
  --dry-run-only \
  --json \
  --stage4k6-acceptance-json '{"dry_run": true, "stage4k6_market_data_contract_acceptance_report": true, "selected_strategy": {"selected_strategy_id": "S01", "paper_only": true, "one_strategy_only": true}, "operation": {"operation_id": "s01_once_2026_05_16", "operation_scope": "single_strategy_market_data_contract_acceptance", "paper_only": true, "live_trading_enabled": false, "broker_submission_enabled": false}, "provider_result_acceptance": {"accepted": true}, "operation_audit": {"operation_audit_passed": true}, "accepted_market_data_outputs": {"available": true, "read_only_for_future_stages": true, "accepted_results": [{"symbol": "SPY", "live_trading_enabled": false, "broker_submission_enabled": false, "order_submission_enabled": false, "state_write_enabled": false, "ledger_write_enabled": false, "direct_ib_call_made": false, "reqMktData_called": false, "qualifyContracts_called": false, "reqContractDetails_called": false}]}, "accepted_contract_qualification_outputs": {"available": false, "read_only_for_future_stages": true, "accepted_results": []}, "boundary_checks": {"no_direct_ib_call": true, "no_direct_reqMktData": true, "no_direct_qualifyContracts": true, "no_direct_reqContractDetails": true, "no_strategy_scan": true, "no_intents_created": true, "no_tickets_created": true, "no_orders_submitted": true, "no_broker_submission": true, "no_state_written": true, "no_ledger_written": true, "no_live_trading": true, "no_all_strategy_enablement": true, "no_scheduler_registration": true, "no_lifecycle_execution": true}, "safety_checks": {"no_live_trading": true, "no_all_strategy_enablement": true, "no_broker_submission_enabled": true, "no_direct_market_data": true, "no_direct_contract_qualification": true, "no_strategy_scan": true, "no_signal_execution": true, "no_order_submission": true, "no_intent_creation": true, "no_ticket_creation": true, "no_state_write": true, "no_ledger_write": true}, "readiness_for_next_phase": {"stage4k_complete": true, "ready_to_proceed_after_stage4k": true}, "success": true, "errors": [], "warnings": []}'
```

```bash
python3 -m algo_trader_unified.tools.stage4l2_signal_readiness_gate \
  --dry-run-only \
  --json \
  --stage4l1-plan-json '{...}' \
  --ack ACK_4L2_SIGNAL_READINESS_GATE_ONLY \
  --ack ACK_NO_STRATEGY_SCAN \
  --ack ACK_NO_SIGNAL_EXECUTION \
  --ack ACK_NO_INTENT_OR_TICKET_CREATION \
  --ack ACK_NO_ORDER_SUBMISSION \
  --ack ACK_NO_BROKER_SUBMISSION \
  --ack ACK_NO_STATE_OR_LEDGER_WRITES \
  --ack ACK_LIVE_TRADING_DISABLED \
  --ack ACK_SINGLE_STRATEGY_ONLY \
  --strategy-registry-snapshot-json '{"candidate_strategy_ids": ["S01"]}' \
  --signal-schema-snapshot-json '{"selected_strategy_id": "S01", "expected_input_sections": ["selected_strategy_id", "operation_id"]}'
```
