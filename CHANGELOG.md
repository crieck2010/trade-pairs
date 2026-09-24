# Changelog — trade-pairs

## v0.1.0 — 2026-09-24

Initial release: pairs trading / statistical-arbitrage engine for the
trade-suite.

**Screening & estimation**
- `find_pairs()`: correlation pre-filter over a price universe, then
  Engle-Granger two-step cointegration tests on the top candidates;
  cointegrated pairs rank first by ADF strength.
- Pure-Python statistics core (`stats.py`): OLS via normal equations,
  Augmented Dickey-Fuller test with AIC lag selection and MacKinnon (1996)
  critical values, Engle-Granger cointegration judged against the stricter
  asymptotic Engle-Granger critical values (5%: −3.34), half-life of
  mean reversion, causal rolling z-scores.

**Signals & backtesting**
- `generate_signals()`: z-score entry (|z| ≥ 2.0) / exit (|z| ≤ 0.5) state
  machine with position flips; no lookahead.
- `backtest_spread()`: unit-position spread walk reporting total return,
  annualized Sharpe, max drawdown, win rate and profit factor over round
  trips, with optional per-side costs.
- `analyze_pair()`: one-call JSON-serializable workup (estimate + live
  z-score + latest signal + backtest).

**Interop**
- `adapters.to_agent_ideas()`: pair + signal → agent idea dicts with explicit
  legs and |z|-scaled conviction (trade-agents seam).
- `adapters.spread_to_dict_bars()`: spread as synthetic OHLC dict-bars for
  trade-backtest's structural `normalize_bar`.
- `adapters.register_strategy()`: `pairs_zscore` descriptor for the
  trade-strategies registry.

**CLI & packaging**
- `trade-pairs screen` (universe screening, CSV or synthetic demo),
  `trade-pairs analyze A B` (full workup, `--ideas` for agent JSON),
  `license` and `update-check` hooks per suite convention.
- Stdlib-only, MIT licensed, 21 tests.
