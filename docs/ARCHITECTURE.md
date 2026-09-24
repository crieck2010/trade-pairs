# Architecture — trade-pairs

## Layout

```
src/trade_pairs/
  __init__.py     public API surface (versioned exports)
  __main__.py     python -m trade_pairs
  models.py       dataclasses: ADFResult, PairCandidate, PairSignal, PairsBacktest
  stats.py        pure-Python stats: OLS, ADF, Engle-Granger, half-life, z-scores
  screening.py    universe screen: correlation pre-filter → EG test → ranking
  signals.py      z-score entry/exit state machine (causal, no lookahead)
  backtest.py     lightweight spread backtester (Sharpe / max DD / win rate)
  pipeline.py     analyze_pair(): one-call workup returning a JSON-safe report
  adapters.py     lazy bridges: to_agent_ideas, spread_to_dict_bars, register_strategy
  cli.py          screen / analyze / license / update-check
  licensing.py    license-key + update-check hooks (suite convention)
```

## Design decisions

- **Engine-first, stdlib-only.** No numpy/scipy/statsmodels: the statistics
  are implemented from normal equations + Gaussian elimination so the math is
  auditable and the package installs anywhere. Sibling data engines
  (`trade-data-equities`, …) supply prices; this module only consumes plain
  `list[float]` closes.
- **Plain data at every boundary.** Candidates, signals, and backtest results
  are dataclasses with `to_dict()`; `analyze_pair` returns a JSON-serializable
  report. This keeps the door open to process fan-out (screening is
  embarrassingly parallel) and to agent consumption without importing the
  engine.
- **Lazy interop.** `adapters.py` imports siblings inside functions only, so
  `trade-pairs` is importable and testable standalone. The bridges:
  - `to_agent_ideas` → `trade-agents` researcher briefs (explicit legs +
    conviction from |z|);
  - `spread_to_dict_bars` → `trade-backtest` via its structural
    `normalize_bar` (spread as one synthetic instrument);
  - `register_strategy` → `trade-strategies` registry descriptor for the
    `pairs_zscore` strategy.
- **Honest statistics.** The cointegration decision uses Engle-Granger
  critical values (stricter than ADF) — see `docs/METHODOLOGY.md`. The
  trade-off is documented, not hidden.

## Scaling notes

- Screening is O(P²) in universe size for the correlation pass, then O(P)
  EG tests on the shortlist. For universes beyond a few hundred symbols,
  shard the pair combinations across processes — `find_pairs` takes plain
  dicts, so chunking the symbol list per worker is trivial.
- The ADF inner loop is the hot spot (OLS per candidate lag); a numpy port
  of `stats.py` behind the same function signatures is the planned speedup
  if profiling demands it. The public API would not change.
