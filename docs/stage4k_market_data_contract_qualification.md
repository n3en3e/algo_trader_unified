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

Required operator acknowledgements are exact strings:

- `ACK_4K4_MARKET_DATA_AND_CONTRACT_GATE_ONLY`
- `ACK_NO_ORDER_SUBMISSION`
- `ACK_NO_BROKER_SUBMISSION`
- `ACK_NO_STATE_OR_LEDGER_WRITES`
- `ACK_LIVE_TRADING_DISABLED`
- `ACK_SINGLE_STRATEGY_ONLY`

Broker submission remains separately gated. Intents, tickets, state writes, and ledger writes remain separately gated. Live trading and all-strategy automation remain blocked. The next phase is Stage 4K-5: the controlled market data and contract qualification executor.

## Stage 4K-5 Purpose

Stage 4K-5 is the controlled provider executor. It consumes the accepted Stage 4K-4 execution gate report and may execute only the explicitly permitted injected controlled provider abstractions for the selected strategy.

Stage 4K-5 may call `request_controlled_market_data` only through the injected controlled market data provider and only when the 4K-4 payload sets `allow_controlled_market_data_provider_call: true`. It may call `qualify_controlled_contracts` only through the injected controlled contract qualification provider and only when `allow_controlled_contract_qualification_provider_call: true`.

The executor never calls direct IBKR methods. These remain forbidden in Stage 4K-5:

- `reqMktData`
- `qualifyContracts`
- `reqContractDetails`

Market-data-only, contract-qualification-only, and combined provider payload plans are valid. Provider payload lists are extracted with safe `dict.get(..., [])` handling, present `None` values are treated as empty lists, and list checks use safe length checks before execution. Stage 4K-5 must not use unsafe index access when deciding whether payloads exist.

Provider payloads and provider outputs are untrusted. Payloads must be native dictionaries, JSON-safe, selected-strategy-only, operation-matched, and must keep live trading, broker submission, order submission, state writes, and ledger writes disabled. Provider outputs are copied into the report only after validation; they must not claim direct IBKR calls, order submission, broker submission, state writes, ledger writes, or live trading.

Provider exceptions are caught and flattened into JSON-safe strings in the form `ExceptionType: message`. Raw newlines and memory-address-style object repr fragments are removed before the failure is placed in `provider_call_trace`, `failed_operations`, or `skipped_operations`.

The report records:

- `provider_call_trace`
- `applied_operations`
- `failed_operations`
- `skipped_operations`
- market data execution results
- contract qualification execution results

If a market data provider call fails, subsequent provider execution is skipped and recorded. Successful provider calls are listed in `applied_operations`; failed calls are listed in `failed_operations`; payloads not attempted after a failure or validation block are listed in `skipped_operations`.

Stage 4K-5 still does not run strategy scans, create intents, create tickets, submit orders, enable broker submission, write state, write ledgers, enable live trading, or enable all-strategy automation. Broker submission remains separately gated. Controlled provider results are read-only inputs for future stages.

The next phase is Stage 4K-6: market data and contract qualification acceptance.

## Stage 4K-6 Purpose

Stage 4K-6 is acceptance and reporting only. It consumes the accepted Stage 4K-5 executor report and decides whether the selected strategy's controlled market data and/or contract qualification capability is complete enough to use as read-only input for the next phase.

Stage 4K-6 does not call providers again. It does not call `request_controlled_market_data` or `qualify_controlled_contracts`, and direct IBKR methods remain forbidden:

- `reqMktData`
- `qualifyContracts`
- `reqContractDetails`

The accepted outputs are intentionally narrow:

- `accepted_market_data_outputs`
- `accepted_contract_qualification_outputs`

Each accepted output list includes only successful, JSON-safe, safe provider results from the Stage 4K-5 `provider_call_trace`. The order preserves the Stage 4K-5 trace order. These outputs are read-only for future stages; Stage 4K-6 does not write files, mutate state, append ledgers, create intents, create tickets, submit orders, or enable broker submission.

Stage 4K-6 validates the operation audit by comparing successful `provider_call_trace` entries to `applied_operations` with stable IDs such as `payload_id`, `provider_type`, `provider_method`, `selected_strategy_id`, and `operation_id`. It does not rely on brittle list-index alignment. The audit safely handles skipped, failed, malformed, and `None` list entries by blocking readiness with clear reasons rather than crashing.

Nested `provider_call_trace.result` safety fields use strict native booleans. Required-disabled fields must be native `false`; a string value such as `"False"` is rejected because it does not prove the Stage 4K-5 executor serialized a native boolean.

Stage 4K-6 still does not run strategy scans, create intents, create tickets, submit orders, enable broker submission, write state, write ledgers, enable live trading, enable all-strategy automation, register scheduler jobs, or execute lifecycle transitions. Broker submission remains separately gated.

When all acceptance gates pass, Stage 4K-6 reports `readiness_for_next_phase.stage4k_complete: true` and `readiness_for_next_phase.ready_to_proceed_after_stage4k: true`. Stage 4K completion means the controlled market data and contract qualification capability is complete for the selected strategy. It does not mean full PAPER trading is active. The next phase is Stage 4L signal/readiness integration, or the next defined post-4K phase, which should consume accepted 4K outputs as read-only inputs while broker submission remains separately gated.

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
python3 -m algo_trader_unified.tools.stage4k5_market_data_contract_executor \
  --dry-run-only \
  --json \
  --stage4k4-gate-json '{...}' \
  --state-snapshot-json '{...}' \
  --risk-snapshot-json '{...}' \
  --scheduler-snapshot-json '{...}' \
  --lifecycle-snapshot-json '{...}' \
  --paper-broker-snapshot-json '{...}' \
  --market-window-snapshot-json '{...}'
```

The Stage 4K-5 CLI is validation-only unless injected providers are supplied from Python/tests. It does not instantiate production providers, connect to IBKR, register scheduler or lifecycle jobs, submit orders, or expose submit/cancel/status actions.

```bash
python3 -m algo_trader_unified.tools.stage4k6_market_data_contract_acceptance \
  --dry-run-only \
  --json \
  --stage4k5-executor-json '{...}' \
  --state-snapshot-json '{...}' \
  --risk-snapshot-json '{...}' \
  --scheduler-snapshot-json '{...}' \
  --lifecycle-snapshot-json '{...}' \
  --paper-broker-snapshot-json '{...}' \
  --market-window-snapshot-json '{...}'
```

The Stage 4K-6 CLI is acceptance/reporting only. It does not instantiate providers, connect to IBKR, expose provider execution actions, register scheduler or lifecycle jobs, submit orders, or expose submit/cancel/status actions.
