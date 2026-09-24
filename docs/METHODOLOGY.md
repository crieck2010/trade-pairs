# Methodology — trade-pairs v0.1.0

How the engine decides two stocks are a tradable pair, and how it trades them.
Written to be read by a quant, not just a user.

## 1. Cointegration (Engle-Granger two-step)

Two price series can be highly correlated and still drift apart forever.
Cointegration is the stronger claim: some linear combination of the two is
*stationary* — it wobbles around a fixed mean instead of wandering off. That
wobble is what we trade.

**Step 1 — the cointegrating regression.** OLS of A on B:

    A_t = α + β·B_t + ε_t

β is the hedge ratio: hold 1 unit of A against β units of B and the portfolio
is (approximately) market-neutral to their shared drift. The residuals ε are
the spread.

**Step 2 — ADF on the residuals.** The Augmented Dickey-Fuller test checks
whether ε has a unit root (H0: non-stationary, i.e. *not* a tradable pair):

    Δε_t = α + γ·ε_{t-1} + Σᵢ δᵢ·Δε_{t-i} + u_t

A sufficiently negative t-statistic on γ rejects H0.

### Why Engle-Granger critical values, not ADF ones

The spread ε is *estimated* (β and α come from the data), which makes the
residuals look more stationary than they are. Judging them against plain ADF
critical values therefore **over-rejects** — it finds "cointegration" too
often. The engine instead uses the stricter asymptotic Engle-Granger critical
values for 2 variables with a constant (5%: **−3.34** vs the ADF −2.86), in
the spirit of MacKinnon (2010).

Caveat: v0.1.0 uses *asymptotic* EG values with no finite-sample correction,
so marginal rejections (stat just below −3.34 on short lookbacks) deserve
skepticism. The report always prints both the stat and the bar.

### Lag selection

The augmentation lag is chosen by AIC over 0..maxlag (Schwert's rule,
⌊12·(n/100)^¼⌋ by default) — enough lags to whiten the residuals without
overfitting.

## 2. Half-life

Regress Δspread on lagged spread: Δs_t = a + λ·s_{t-1}. If λ < 0 the spread
mean-reverts with half-life **−ln(2)/λ** bars — the expected time for a
dislocation to decay halfway. No mean reversion (λ ≥ 0) returns `None`, and
the pair is a poor z-score candidate however cointegrated it looks.

## 3. Z-score signals

The spread is z-scored on a rolling window using only data up to each bar
(no lookahead):

- |z| ≥ `entry_z` (default 2.0) → enter: **long spread** (buy A, short β·B)
  when z ≤ −2, **short spread** when z ≥ +2
- |z| ≤ `exit_z` (default 0.5) → exit
- an opposite-side entry flips the position (exit + new entry, same bar)

## 4. Backtest

Signals are walked in unit spread positions. P&L = position × Δspread, scaled
so equity stays in sensible units; an optional per-side cost keeps results
honest. Reported: total return, annualized Sharpe (√252), max drawdown,
win rate and profit factor over round trips. Dangling positions close at the
last bar.

## 5. Known limitations

- **Correlation pre-filter can miss pairs** whose cointegration doesn't show
  up as high log-price correlation; the screen is a heuristic, not exhaustive.
- **β is assumed constant** over the lookback. Regime breaks (mergers,
  sector rotations) invalidate old hedges — re-estimate on rolling windows
  for live use.
- **No borrow-cost, dividend, or corporate-action handling** in the spread
  backtest; treat it as a research screen, not a P&L forecast.
- **Survivorship/data-snooping**: screening many pairs and backtesting the
  winners overstates expected performance. Walk-forward validation is the
  honest next step (planned).
