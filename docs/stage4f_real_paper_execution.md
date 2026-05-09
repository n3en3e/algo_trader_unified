# Stage 4F Real Paper Execution

Stage 4F introduces real IBKR paper execution capabilities in small, manual
operator-gated steps. It remains paper-only and is not wired into daemon,
scheduler, lifecycle, or automated submission flows.

## Stage 4F-1 Factory Preflight

Stage 4F-1 adds only a preflight and factory seam for real IBKR paper execution.
It remains dry-run, paper-only, and unwired from daemon, scheduler, lifecycle,
and operator submission flows.

The preflight report can be run with:

```bash
python3 -m algo_trader_unified.tools.ibkr_paper_factory_preflight --dry-run-only --json
```

The core report validates an injected read-only PAPER config on port 4004 and
checks `ib_insync` availability with `importlib.util.find_spec` or an injected
probe. It does not import `ib_insync` at module import, instantiate IB, connect,
submit orders, cancel orders, request market data, or qualify contracts.

The optional `create_real_ibkr_paper_ib` helper defaults closed unless
`allow_real_ibkr=True` is passed. Even then it validates config first, imports
`ib_insync` only inside that gated function, instantiates `IB()`, and returns the
unconnected object without any further broker interaction.

## Stage 4F-2 Connection Preflight

Stage 4F-2 adds a manual real IBKR paper connection preflight. Its purpose is to
prove that an operator can explicitly connect to IBKR paper Gateway, perform
read-only checks, disconnect cleanly, and inspect a JSON-safe report before any
future real paper submit phase.

This preflight is real IBKR paper connection preflight only. It is read-only and
requires the explicit `--allow-real-ibkr` flag before it attempts a real
connection. Without that flag, the tool reports that the real connection was not
attempted.

The connection path validates PAPER configuration first and connects only to the
IBKR paper port, `4004`. LIVE mode and live port `4002` are rejected before
factory/client creation.

The Stage 4F-2 read-only checks are:

- connect
- connection status
- current time
- account snapshot
- open orders
- positions
- disconnect

The Stage 4F-2 preflight does not submit orders, cancel orders, request market
data, or qualify contracts. It does not call order submission, cancellation,
market-data, or contract-qualification APIs. It is not wired into daemon,
scheduler, lifecycle jobs, readiness mutation, ledgers, StateStore, or any live
trading path.

Run a dry report without attempting a real connection:

```bash
python3 -m algo_trader_unified.tools.ibkr_paper_connection_preflight --dry-run-only --json
```

Run the manual real paper connection preflight against paper Gateway:

```bash
python3 -m algo_trader_unified.tools.ibkr_paper_connection_preflight --dry-run-only --json --allow-real-ibkr --host 127.0.0.1 --port 4004 --client-id <ID> --trading-mode PAPER
```

If connection fails, check whether the configured `client_id` is already in use
by a zombie process or another active daemon. This is a diagnostic step only;
the actual report reason remains the source of truth for the failure.

Stage 4F-2 is a manual operator gate before any future real paper submit phase.
Stage 4F-3 remains separate and must still be explicitly approved before any
paper submit command or order submission path is added.
