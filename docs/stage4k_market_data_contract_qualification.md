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

Stage 4K-4 is the market data and contract qualification execution gate. It is the next gate after a clean 4K-3 dry-run report.

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
