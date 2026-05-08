# Stage 4D Paper Broker Adapter Contract

Stage 4D-3 adds a translation and validation layer only. The
`PaperBrokerAdapter` converts internal order-intent dictionaries into normalized
`BrokerOrderRequest` values, validates paper-mode safety gates, and returns the
Stage 4D-1 broker result dataclasses.

The adapter accepts an injected fake/test client only in this phase. It does not
import IBKR libraries, does not make broker calls, does not fetch market data,
and is not wired into daemon, scheduler, lifecycle jobs, CLI order submission,
or any live/paper execution path.

Safety rules:

- `DRY_RUN` and `PAPER` modes are allowed.
- `LIVE` and unknown modes are rejected through `validate_broker_mode`.
- `allow_live` does not enable `LIVE`.
- `client_order_id` is deterministic and based on `intent_id`.
- Invalid order intents fail closed and do not call the injected client.
- Raw client responses are recursively sanitized to JSON-safe primitives before
  being returned through result `raw` fields.
- Metadata copied onto `BrokerOrderRequest` is recursively sanitized.

A later phase may add an IBKR paper client implementation behind this adapter,
still protected by explicit mode and wiring gates.

## Stage 4D-5 IBKR Paper Order Mapping

Stage 4D-5 adds IBKR paper order mapping and config validation only. The mapper
converts a normalized `BrokerOrderRequest` into a deterministic JSON-safe
`IbkrPaperOrderPlan`; it does not import `ib_insync`, make IBKR calls, qualify
contracts, submit orders, or wire anything into daemon, scheduler, lifecycle
jobs, or adapter behavior.

4D-5 is paper-only:

- `trading_mode` must be exactly `PAPER`.
- Port `4004` is the paper-only gate for this phase.
- Port `4002`, `LIVE`, and unknown trading modes are rejected.
- `readonly` must be `False` for future paper submission planning, though this
  phase still performs no submission.
- TIF mapping is limited to `DAY` and `GTC`; absent TIF defaults to `DAY`.
- OPTION contract hints are JSON-only metadata hints (`expiry`, `strike`,
  `right`), not qualified IBKR contracts.

The next phase may add an injected IBKR paper client behind this mapping. That
client must remain behind explicit paper gates and must not be wired into
scheduler or lifecycle execution until an explicit 4E gate.
