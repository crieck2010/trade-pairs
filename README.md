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

## The maths

**What you learn.** Which stock pairs move together *in the strong sense* —
not just correlated, but cointegrated, so their spread wobbles around a
fixed mean instead of wandering off. For each pair you get the hedge ratio,
the half-life of a dislocation, and a live z-score that says "long the
spread", "short the spread", or "flat".

**Why it matters.** Correlation is cheap and treacherous: two stocks can be
95% correlated and still drift apart forever, which is exactly how naive
pairs trades blow up. Cointegration tests the tradable claim — stationarity
of the spread — and the hedge ratio turns it into a market-neutral position.

**The maths.**

- *Engle-Granger two-step*: (1) OLS cointegrating regression A_t = α + β·B_t
  + ε_t — β is the hedge ratio, residuals ε are the spread; (2) ADF test on
  ε: Δε_t = α + γ·ε_{t-1} + Σᵢ δᵢ·Δε_{t-i} + u_t, rejecting the unit-root null
  on a sufficiently negative t-stat of γ. Augmentation lag chosen by AIC
  (Schwert's rule).
- *Critical values*: the stricter Engle-Granger values for 2 variables with
  a constant (5%: −3.34, vs −2.86 for plain ADF) — because β is estimated,
  residuals look more stationary than they are, and plain ADF over-rejects.
- *Half-life*: from Δs_t = a + λ·s_{t-1}, half-life = −ln(2)/λ bars; λ ≥ 0
  means no mean reversion and a poor z-score candidate.
- *Signals*: rolling causal z-score of the spread — enter at |z| ≥ 2.0 (long
  spread when z ≤ −2, short when z ≥ +2), exit at |z| ≤ 0.5; opposite entry
  flips the position.
- *Backtest*: P&L = position × Δspread in unit spread positions, with
  optional per-side costs; reports total return, annualized Sharpe, max
  drawdown, win rate and profit factor.

**Honest limitations.**

- The correlation pre-filter is a heuristic and can miss cointegrated pairs
  that don't show high log-price correlation.
- β is assumed constant over the lookback — regime breaks invalidate old
  hedges; re-estimate on rolling windows for live use.
- Asymptotic EG critical values with no finite-sample correction: marginal
  rejections on short lookbacks deserve skepticism.
- Screening many pairs and backtesting the winners overstates expected
  performance — walk-forward validation is the honest next step.
