# Stage 4K Market Data And Contract Qualification

Stage 4K prepares one selected PAPER strategy for a future, explicitly gated market data and contract qualification flow. Broker submission, order placement, intents, tickets, state writes, ledger writes, live trading, and all-strategy automation remain disabled throughout these planning and dry-run phases.

## Stage 4K-1 Purpose

Stage 4K-1 is readiness and reporting only. It consumes an accepted Stage 4J-6 controlled scheduled PAPER operation acceptance report and determines whether Stage 4J completed cleanly enough to design the next gate: Stage 4K-2, the market data and contract qualification plan.

Stage 4K-1 does not fetch market data, qualify contracts, call strategy scan/run methods, register scheduler or lifecycle jobs, instantiate broker clients, create intents or tickets, submit orders, write state, or write ledger entries.

## Stage 4K-2 Purpose

Stage 4K-2 is plan and reporting only. It consumes an accepted Stage 4K-1 readiness report and builds the proposed selected-strategy market data and contract qualification plan. The plan includes proposed operation-flow steps and future provider payloads, but it still does not execute provider methods.

Stage 4K-2 does not fetch market data, qualify contracts, call strategy code, create intents or tickets, submit orders, write state, or write ledger entries. It validates the 4K-1 report, proposed 4K scope, and optional snapshots for capabilities, requirements, state, risk, scheduler, lifecycle, paper broker, and market window.

## Stage 4K-3 Purpose

Stage 4K-3 is dry-run/reporting only. It consumes the accepted Stage 4K-2 plan report and simulates the future controlled provider operation flow without executing any market data or contract qualification calls.

The 4K-3 report answers what would be passed to the future controlled market data and contract qualification providers, while confirming that every proposed step remains PAPER-only, selected-strategy-only, JSON-safe, deterministic, and non-executing.

Stage 4K-3 does not fetch market data or qualify contracts. It does not call `request_controlled_market_data` or `qualify_controlled_contracts`, and it does not call direct IBKR methods such as `reqMktData`, `qualifyContracts`, or `reqContractDetails`.

The dry run copies and sanitizes the Stage 4K-2 `proposed_provider_payloads` as native dictionaries. Payload list extraction is safe: missing `market_data_provider_payloads` or `contract_qualification_provider_payloads` defaults to an empty list and blocks readiness without raising a `KeyError`.

Any capability-style flag checks use safe traversal across root fields, `capabilities`, `config`, and module-specific keys such as `market_data` and `contract_qualification`. Stringified booleans such as `"True"` and `"False"` are not treated as native booleans; required-disabled fields must be native `False`.

The `dry_run_trace` is deterministic, JSON-safe, ordered, and non-executing. Trace items include native dict `input_payload` and native dict `simulated_result` values with flat placeholder fields.

## Safety Boundary

Broker submission remains separately gated. Intents, tickets, state writes, and ledger writes remain separately gated. Live trading and all-strategy automation remain blocked.

The reports may say the system is ready to build the next explicit gate, but they must still keep:

- `live_trading_enabled: false`
- `all_strategies_enabled: false`
- `broker_submission_enabled: false`
- `allow_order_submission: false`
- `allow_state_write: false`
- `allow_ledger_write: false`

## Next Phase

Stage 4K-4 is the market data and contract qualification execution gate. It consumes the accepted Stage 4K-3 dry-run report and produces a read-only permission packet for Stage 4K-5.

Stage 4K-4 is execution-gate/reporting only. It does not fetch market data, qualify contracts, call `request_controlled_market_data`, call `qualify_controlled_contracts`, call strategy scan/run methods, create intents or tickets, submit orders, write state, write ledger entries, register scheduler jobs, execute lifecycle transitions, or call direct IBKR methods such as `reqMktData`, `qualifyContracts`, or `reqContractDetails`.

The gate may permit Stage 4K-5 to call injected controlled provider abstractions only. Market-data-only, contract-qualification-only, and combined provider payload plans are valid as long as the non-present side remains disabled. Provider payload list extraction uses safe `dict.get(..., [])` traversal and treats present `None` lists as empty lists, so a missing or `None` side blocks only when both provider payload lists are empty after normalization.

Stage 4K-4 validates the Stage 4K-3 `dry_run_trace` with strict native booleans. Required disabled flags must be native `false`; string values such as `"False"` are rejected and do not pass as disabled flags.

The proposed Stage 4K-5 payload uses:

- `allow_controlled_market_data_provider_call`
- `allow_controlled_contract_qualification_provider_call`
- `allow_direct_reqMktData: false`
- `allow_direct_qualifyContracts: false`
- `allow_direct_reqContractDetails: false`
- `allow_strategy_scan: false`
- `allow_intent_creation: false`
- `allow_ticket_creation: false`
- `allow_order_submission: false`
- `allow_broker_submission: false`
- `allow_state_write: false`
- `allow_ledger_write: false`
- `live_trading_enabled: false`
- `all_strategies_enabled: false`

Stale Stage 4J executor fields are not part of Stage 4K-4 or Stage 4K-5:

- `proposed_execution_permissions_for_4J5`
- `may_call_strategy_next_phase`
- `may_build_executor_next_phase`
- `may_fetch_market_data_next_phase`

Required operator acknowledgements are exact strings:

- `ACK_4K4_MARKET_DATA_AND_CONTRACT_GATE_ONLY`
- `ACK_NO_ORDER_SUBMISSION`
- `ACK_NO_BROKER_SUBMISSION`
- `ACK_NO_STATE_OR_LEDGER_WRITES`
- `ACK_LIVE_TRADING_DISABLED`
- `ACK_SINGLE_STRATEGY_ONLY`

Broker submission remains separately gated. Intents, tickets, state writes, and ledger writes remain separately gated. Live trading and all-strategy automation remain blocked. The next phase is Stage 4K-5: the controlled market data and contract qualification executor.

## Examples

```bash
python3 -m algo_trader_unified.tools.stage4k1_market_data_contract_readiness \
  --dry-run-only \
  --json \
  --stage4j6-acceptance-json '{...}' \
  --market-data-capability-snapshot-json '{"selected_strategy_id": "S01", "capabilities": {"market_data_provider_available": true, "paper_market_data_mode": true, "market_data_currently_enabled": false, "reqMktData_enabled": false}}' \
  --contract-qualification-capability-snapshot-json '{"selected_strategy_id": "S01", "config": {"contract_qualification_provider_available": true, "contract_qualification_currently_enabled": false, "qualifyContracts_enabled": false, "reqContractDetails_enabled": false}}'
```

```bash
python3 -m algo_trader_unified.tools.stage4k2_market_data_contract_plan \
  --dry-run-only \
  --json \
  --stage4k1-readiness-json '{...}'
```

```bash
python3 -m algo_trader_unified.tools.stage4k3_market_data_contract_dry_run \
  --dry-run-only \
  --json \
  --stage4k2-plan-json '{...}' \
  --state-snapshot-json '{...}' \
  --risk-snapshot-json '{...}' \
  --scheduler-snapshot-json '{...}' \
  --lifecycle-snapshot-json '{...}' \
  --paper-broker-snapshot-json '{...}' \
  --market-window-snapshot-json '{...}'
```

```bash
python3 -m algo_trader_unified.tools.stage4k4_market_data_contract_execution_gate \
  --dry-run-only \
  --json \
  --stage4k3-dry-run-json '{...}' \
  --ack ACK_4K4_MARKET_DATA_AND_CONTRACT_GATE_ONLY \
  --ack ACK_NO_ORDER_SUBMISSION \
  --ack ACK_NO_BROKER_SUBMISSION \
  --ack ACK_NO_STATE_OR_LEDGER_WRITES \
  --ack ACK_LIVE_TRADING_DISABLED \
  --ack ACK_SINGLE_STRATEGY_ONLY \
  --state-snapshot-json '{...}' \
  --risk-snapshot-json '{...}' \
  --scheduler-snapshot-json '{...}' \
  --lifecycle-snapshot-json '{...}' \
  --paper-broker-snapshot-json '{...}' \
  --market-window-snapshot-json '{...}'
```

```bash
python3 -m algo_trader_unified.tools.stage4k3_market_data_contract_dry_run \
  --dry-run-only \
  --json \
  --stage4k2-plan-json '{...}' \
  --state-snapshot-json '{...}' \
  --risk-snapshot-json '{...}' \
  --scheduler-snapshot-json '{...}' \
  --lifecycle-snapshot-json '{...}' \
  --paper-broker-snapshot-json '{...}' \
  --market-window-snapshot-json '{...}'
```
