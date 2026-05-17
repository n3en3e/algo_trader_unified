# Stage 4K Market Data And Contract Qualification

Stage 4K is the next controlled gate after Stage 4J. Its job is to prepare a selected single strategy for a future, explicitly gated market data and contract qualification flow while keeping broker submission, order placement, intents, tickets, state writes, ledger writes, live trading, and all-strategy automation disabled.

## Stage 4K-1 Purpose

Stage 4K-1 is readiness and reporting only. It consumes an accepted Stage 4J-6 controlled scheduled PAPER operation acceptance report and determines whether Stage 4J completed cleanly enough to design the next gate: Stage 4K-2, the market data and contract qualification plan.

Stage 4J completion validates the selected-strategy executor shell and its report-only safety boundaries. It does not enable full paper trading, broker submission, market data fetching, contract qualification, intent creation, ticket creation, state mutation, ledger writing, live trading, or all-strategy automation.

Stage 4K-1 does not fetch market data and does not qualify contracts. It does not call strategy scan/run methods, register scheduler or lifecycle jobs, instantiate broker clients, create intents or tickets, submit orders, write state, or write ledger entries.

## Snapshot Handling

The 4K-1 report accepts optional read-only snapshots for the strategy registry, market data capability, contract qualification capability, activation state, runtime scheduler/lifecycle state, state cleanliness, risk controls, paper broker config, and market window status.

Market data and contract qualification capability snapshots may expose fields at the root level or under nested `capabilities` or `config` dictionaries. Stage 4K-1 reads those structures with safe dictionary traversal and treats malformed or missing optional fields conservatively.

Stringified boolean values from external tools, such as `"True"` or `"False"`, are not treated as native booleans. Enablement flags must be native `False` when supplied. Ambiguous string values block readiness instead of being coerced by Python truthiness.

## Safety Boundary

Broker submission remains separately gated. Intents, tickets, state writes, and ledger writes remain separately gated. Live trading and all-strategy automation remain blocked.

The report may say the system is ready to build the 4K-2 plan, but it must still report:

- `may_fetch_market_data_now: false`
- `may_qualify_contracts_now: false`
- `may_submit_orders_now: false`
- `may_create_intents_now: false`
- `may_create_tickets_now: false`
- `may_write_state_now: false`
- `may_write_ledger_now: false`
- `live_trading_enabled: false`
- `all_strategies_enabled: false`
- `broker_submission_enabled: false`

## Next Phase

Stage 4K-2 is the market data and contract qualification plan. It should remain a plan-only phase until a later explicit controlled execution gate permits carefully scoped market data and contract qualification behavior.

## Example

```bash
python3 -m algo_trader_unified.tools.stage4k1_market_data_contract_readiness \
  --dry-run-only \
  --json \
  --stage4j6-acceptance-json '{"dry_run": true, "stage4j6_controlled_paper_operation_acceptance_report": true, "selected_strategy": {"selected_strategy_id": "S01", "paper_only": true, "one_strategy_only": true}, "operation": {"operation_id": "s01_once_2026_05_16", "operation_scope": "single_strategy_controlled_paper_operation_acceptance", "paper_only": true, "live_trading_enabled": false, "broker_submission_enabled": false}, "executor_acceptance": {"accepted": true}, "boundary_checks": {"no_market_data_requested": true, "no_contracts_qualified": true, "no_intents_created": true, "no_tickets_created": true, "no_orders_submitted": true, "no_state_written": true, "no_ledger_written": true, "no_live_trading": true, "no_all_strategy_enablement": true, "no_broker_submission": true}, "safety_checks": {"no_live_trading": true, "no_all_strategy_enablement": true, "no_broker_submission_enabled": true, "no_market_data": true, "no_contract_qualification": true, "no_order_submission": true, "no_intent_creation": true, "no_ticket_creation": true, "no_state_write": true, "no_ledger_write": true}, "readiness_for_stage4j_complete_or_next_gate": {"stage4j_complete": true, "ready_for_next_explicit_gate": true, "recommended_next_gate": "stage4k_market_data_and_contract_qualification_gate"}, "success": true, "errors": [], "warnings": []}' \
  --market-data-capability-snapshot-json '{"selected_strategy_id": "S01", "capabilities": {"market_data_provider_available": true, "paper_market_data_mode": true, "market_data_currently_enabled": false, "reqMktData_enabled": false}}' \
  --contract-qualification-capability-snapshot-json '{"selected_strategy_id": "S01", "config": {"contract_qualification_provider_available": true, "contract_qualification_currently_enabled": false, "qualifyContracts_enabled": false, "reqContractDetails_enabled": false}}'
```
