# trade-pairs

Pairs trading / statistical-arbitrage engine for the **trade-suite**: it screens
a universe for cointegrated pairs, estimates hedge ratios, trades the spread on
z-score signals, and backtests the result.

> **AAA/AAB** — cointegrated (ADF −4.39 vs 5% crit −3.34), β=1.53, half-life
> 8.4 bars — current z-score −2.31 → **LONG spread: buy AAA, short 1.53× AAB**.

Part of the [trade-suite](https://github.com/crieck2010/trade-suite) algorithmic
and agentic trading system. Research/backtesting/paper-trading only — never live
trading, never personalized investment advice.

## Install

```bash
pip install git+https://github.com/crieck2010/trade-pairs.git
```

Requires Python 3.10+. No third-party dependencies — the statistics core
(OLS, Augmented Dickey-Fuller, Engle-Granger) is implemented in pure Python so
the math is auditable without numpy/scipy/statsmodels.

## Quick start

```bash
# screen the built-in synthetic universe (2 cointegrated pairs + noise)
trade-pairs screen

# full workup of one pair
trade-pairs analyze AAA AAB

# JSON report + agent ideas
trade-pairs analyze AAA AAB --format json
trade-pairs analyze AAA AAB --ideas

# your own data: wide CSV with a date column then one column per symbol
trade-pairs screen --csv prices.csv --lookback 252 --min-correlation 0.7
```

```python
from trade_pairs import find_pairs, analyze_pair

prices = {"AAA": [...], "AAB": [...], ...}  # each a list of closes
pairs = find_pairs(prices, lookback=252)
for p in pairs:
    if p.cointegrated:
        print(p.symbol_a, p.symbol_b, p.hedge_ratio, p.adf.stat)

report = analyze_pair("AAA", "AAB", prices["AAA"], prices["AAB"])
print(report["backtest_summary"])
```

## How it works

1. **Screen** — rank all pairs by |correlation| of log prices over the lookback
   window, then run the Engle-Granger two-step cointegration test on the top
   candidates. Cointegrated pairs rank first, ordered by ADF strength.
2. **Estimate** — hedge ratio β from the cointegrating regression
   (A = α + β·B); the tradable spread is `A − β·B`. Half-life comes from OLS
   of Δspread on lagged spread.
3. **Trade** — z-score the spread on a rolling window (causal, no lookahead):
   enter long/short spread at |z| ≥ 2.0, exit at |z| ≤ 0.5.
4. **Backtest** — walk the signals in spread units and report total return,
   annualized Sharpe, max drawdown, and win rate over round trips.

See `docs/METHODOLOGY.md` for the exact math (including why the cointegration
bar uses Engle-Granger critical values, not plain ADF ones) and
`docs/ARCHITECTURE.md` for the module layout and suite interop.

## Suite interop

| Sibling | Bridge |
|---|---|
| `trade-agents` | `adapters.to_agent_ideas()` — pair + signal → idea dicts with explicit legs, conviction from \|z\| |
| `trade-backtest` | `adapters.spread_to_dict_bars()` — spread as synthetic OHLC dict-bars via its structural `normalize_bar` |
| `trade-strategies` | `adapters.register_strategy()` — `pairs_zscore` descriptor for the strategy registry |
| `trade-data-equities` | feed `get_bars` closes straight into `find_pairs` / `analyze_pair` (no import needed) |

## Status

v0.1.0 — cointegration screening, pair analysis, z-score signals, spread
backtester, CLI, and agent/backtest/strategy adapters. See `CHANGELOG.md`.
