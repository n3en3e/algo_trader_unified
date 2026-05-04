# Dry-Run Lifecycle Runbook

## Purpose

Use this runbook to validate the complete dry-run lifecycle in a temporary or test root. The flow exercises scheduler job wrappers and existing lifecycle helpers only.

## Preconditions

- The repo compiles.
- The test suite passes.
- Use paper/dry-run state only.
- Do not use production state for smoke validation.
- No broker, market data, deployment service, or live scheduler is required.

## Verification Commands

```bash
python3 -m py_compile $(find algo_trader_unified -name '*.py' -not -path '*/__pycache__/*')
python3 -m unittest discover -s algo_trader_unified/tests
python3 -m unittest algo_trader_unified.tests.test_phase3w_e2e_dry_run_lifecycle
```

## Lifecycle Sequence

Entry:

1. Run S01 vol scan with an injected test signal provider.
2. Run intent submission.
3. Run intent confirmation.
4. Run intent fill confirmation.
5. Run position transitions to open the position.

Exit:

1. Run management scan with an injected close signal provider.
2. Run intent submission.
3. Run intent confirmation.
4. Run intent fill confirmation.
5. Run position transitions to close the position.

Expected lifecycle events:

- Order ledger lifecycle: `ORDER_INTENT_CREATED`, `ORDER_SUBMITTED`, `ORDER_CONFIRMED`, `FILL_CONFIRMED`, `CLOSE_INTENT_CREATED`, `CLOSE_ORDER_SUBMITTED`, `CLOSE_ORDER_CONFIRMED`, `CLOSE_FILL_CONFIRMED`
- Execution ledger lifecycle: `POSITION_OPENED`, `POSITION_CLOSED`

## Inspecting State

Useful read-only tools:

```bash
python3 -m algo_trader_unified.tools.system_status --root-dir <test-root> --json
python3 -m algo_trader_unified.tools.list_order_intents --root-dir <test-root>
python3 -m algo_trader_unified.tools.list_positions --root-dir <test-root>
```

Ledger files to inspect under the selected test root:

- `data/ledger/order_ledger.jsonl`
- `data/ledger/execution_ledger.jsonl`

## Safety Notes

- Keep smoke validation on temp or test state.
- Do not run against production state.
- Do not start a live scheduler.
- Do not add deployment service changes for this smoke.
- Do not use broker credentials or live market data.
