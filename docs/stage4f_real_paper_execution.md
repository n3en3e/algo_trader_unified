# Stage 4F Real Paper Execution

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
