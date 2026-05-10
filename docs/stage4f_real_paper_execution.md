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

## Stage 4F-3 Manual Real Paper Submit

Stage 4F-3 adds one manual, operator-gated real IBKR paper submit command for
exactly one pre-approved ticket. It combines a Stage 4E-4 paper order ticket
report, a Stage 4F-2 connection preflight report, explicit acknowledgement
strings, PAPER/4004 submit config validation, and injected factory/client
dependencies.

The core report builder does not instantiate IB directly, import `ib_insync`,
read config.py, read or write ledgers, touch StateStore, request market data,
qualify contracts, or wire into daemon, scheduler, lifecycle, or live trading.
If any gate fails, it refuses before calling the IB factory, execution-client
factory, or submit method.

The manual CLI requires both real-paper allow flags and keeps `--dry-run-only`
mandatory:

```bash
python3 -m algo_trader_unified.tools.manual_real_paper_submit \
  --dry-run-only \
  --json \
  --allow-real-ibkr \
  --allow-real-paper-submit \
  --ticket-json '{...}' \
  --preflight-json '{...}' \
  --ack "I understand this is IBKR PAPER only." \
  --ack "I understand this will submit one real paper order." \
  --ack "I understand no live orders are allowed." \
  --ack "I understand scheduler/lifecycle automation remains disabled." \
  --ack "I reviewed the ticket and preflight report."
```

Acknowledgements are exact list items after trimming outer whitespace. A single
combined string does not satisfy the gate, and extra acknowledgements do not
replace missing required text.

Stage 4F-3 remains paper-only and single-ticket only. It does not add automated
paper trading, live order submission, market data, contract qualification,
scheduler cadence changes, lifecycle execution changes, strategy threshold
changes, sizing changes, deployment, or systemd edits.

## Stage 4F-4 Manual Real Paper Status/Cancel

Stage 4F-4 may add manual, operator-gated status and cancel actions for already
submitted IBKR paper orders. These actions remain paper-only, dry-run-only from
the project perspective, and unwired from daemon, scheduler, lifecycle, ledgers,
StateStore, or automated submission flows.

The tool must not call `submit_order_plan`, submit new orders, request market
data, qualify contracts, change strategy thresholds, change sizing, or enable
live trading paths. Its only execution-client actions are status lookup and
cancel for explicitly supplied broker order ids.

### Manual status gates

- Require `action="status"` and `allow_real_paper_status is True`.
- Ensure the action strictly aligns with the flags: when `action="status"`,
  `allow_real_paper_cancel` must be `False` to reject mixed-intent payloads.
- Require `allow_real_ibkr is True`, PAPER mode, explicit port `4004`, and a
  non-empty broker order id.
- Refuse before factory/client creation if any gate fails.

### Manual cancel gates

- Require `action="cancel"` and `allow_real_paper_cancel is True`.
- Ensure the action strictly aligns with the flags: when `action="cancel"`,
  `allow_real_paper_status` must be `False` to reject mixed-intent payloads.
- Require `allow_real_ibkr is True`, PAPER mode, explicit port `4004`, a
  non-empty broker order id, and exact operator acknowledgements for cancel.
- Refuse before factory/client creation if any gate fails.

### Failure behavior

Status and cancel failures should be reported in the JSON-safe report without
crashing the tool boundary. To prevent disconnect failures from masking
status/cancel exceptions, wrap the `execution_client.disconnect()` call inside
its own `try`/`except` block within the main `finally` block. Catch the
disconnect exception, log it to the report's warnings or errors, and allow the
original result or exception report to return safely.

### Review tasks

- Grep for `disconnect()` and verify the call inside the `finally` block is
  protected by its own nested `try`/`except`, so a disconnect crash cannot
  overwrite or mask the primary status/cancel exception.
- Grep for `submit_order_plan` and verify Stage 4F-4 does not call it.
- Verify the status and cancel gates enforce mutually exclusive allow flags for
  the selected action.
- Verify the tool does not read or write ledgers or StateStore and remains a
  stateless manual probe/trigger.
