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
