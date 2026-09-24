"""Pure-Python statistics for pairs trading: OLS, ADF unit-root tests,
Engle-Granger cointegration, half-life, and rolling z-scores.

Everything here is stdlib-only and deterministic.  The implementations are
deliberately transparent (normal equations + Gaussian elimination) so the
math is auditable without pulling in numpy/scipy/statsmodels.
"""

from __future__ import annotations

import math

from .models import ADFResult

# ---------------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------------


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def variance(xs: list[float], ddof: int = 1) -> float:
    n = len(xs)
    if n <= ddof:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - ddof)


def stdev(xs: list[float], ddof: int = 1) -> float:
    return math.sqrt(max(variance(xs, ddof), 0.0))


def covariance(xs: list[float], ys: list[float], ddof: int = 1) -> float:
    n = len(xs)
    if n != len(ys) or n <= ddof:
        return 0.0
    mx, my = mean(xs), mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - ddof)


def correlation(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation; 0.0 when either leg has no variance."""
    sx, sy = stdev(xs), stdev(ys)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return covariance(xs, ys) / (sx * sy)


def diff(xs: list[float]) -> list[float]:
    return [b - a for a, b in zip(xs, xs[1:])]


# ---------------------------------------------------------------------------
# linear algebra (normal equations + Gaussian elimination)
# ---------------------------------------------------------------------------


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [bi] for row, bi in zip(A, b)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            raise ValueError("singular matrix in OLS solve")
        M[col], M[piv] = M[piv], M[col]
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for k in range(col, n + 1):
                M[row][k] -= factor * M[col][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def ols_regression(X: list[list[float]], y: list[float]) -> dict:
    """OLS via normal equations.  Returns coefficients, residuals, R² and
    standard errors (homoskedastic).  ``X`` rows are observations."""
    n, k = len(X), len(X[0])
    XtX = [[sum(X[r][i] * X[r][j] for r in range(n)) for j in range(k)] for i in range(k)]
    Xty = [sum(X[r][i] * y[r] for r in range(n)) for i in range(k)]
    beta = _solve(XtX, b=Xty)
    resid = [y[r] - sum(X[r][i] * beta[i] for i in range(k)) for r in range(n)]
    sse = sum(r * r for r in resid)
    dof = max(n - k, 1)
    s2 = sse / dof
    # (X'X)^{-1} via solving against each unit vector
    XtX_inv = [_solve(XtX, [1.0 if i == j else 0.0 for i in range(k)]) for j in range(k)]
    XtX_inv = [[XtX_inv[j][i] for j in range(k)] for i in range(k)]
    se = [math.sqrt(max(s2 * XtX_inv[i][i], 0.0)) for i in range(k)]
    ybar = mean(y)
    sst = sum((v - ybar) ** 2 for v in y)
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    return {"beta": beta, "se": se, "residuals": resid, "r_squared": r2, "sse": sse}


def ols_slope_intercept(x: list[float], y: list[float]) -> tuple[float, float, list[float]]:
    """Regress y on x with intercept.  Returns (slope, intercept, residuals)."""
    X = [[1.0, xi] for xi in x]
    res = ols_regression(X, y)
    intercept, slope = res["beta"]
    return slope, intercept, res["residuals"]


# ---------------------------------------------------------------------------
# Augmented Dickey-Fuller test
# ---------------------------------------------------------------------------

# MacKinnon (1996) response-surface coefficients for ADF critical values:
# crit(p) = b_inf + b1/T + b2/T^2, per regression type and tail probability.
_MACKINNON = {
    "c": {  # constant, no trend
        0.01: (-3.4336, -5.999, -29.25),
        0.05: (-2.8621, -2.738, -8.36),
        0.10: (-2.5671, -1.438, -4.48),
    },
    "ct": {  # constant + trend
        0.01: (-3.9638, -8.353, -47.44),
        0.05: (-3.4126, -4.039, -17.83),
        0.10: (-3.1279, -2.418, -7.58),
    },
}

# Asymptotic Engle-Granger critical values for the ADF regression run on the
# *residuals* of a cointegrating regression (constant, no trend, 2 variables).
# These are the standard textbook values in the spirit of MacKinnon (2010).
# They are deliberately more negative than the plain ADF critical values
# above: the spread is estimated, so the residual ADF test needs a stricter
# bar.  v0.1.0 uses asymptotic values (no finite-sample correction); see
# docs/METHODOLOGY.md for the caveat on marginal rejections.
_EG_CRIT = {0.01: -3.90, 0.05: -3.34, 0.10: -3.04}


def _adf_critical(regression: str, n: int, tail: float) -> float:
    b_inf, b1, b2 = _MACKINNON[regression][tail]
    return b_inf + b1 / n + b2 / (n * n)


def adf_test(
    series: list[float],
    regression: str = "c",
    max_lag: int | None = None,
) -> ADFResult:
    """Augmented Dickey-Fuller test of ``series`` for a unit root.

    H0: the series has a unit root (non-stationary).  A sufficiently negative
    ``stat`` rejects H0 in favour of (trend-)stationarity.

    ``regression`` is ``"c"`` (constant) or ``"ct"`` (constant + trend).
    Lag length is chosen by AIC over 0..max_lag (Schwert's rule by default).
    """
    if regression not in _MACKINNON:
        raise ValueError("regression must be 'c' or 'ct'")
    y = [float(v) for v in series]
    n = len(y)
    if n < 10:
        raise ValueError("adf_test needs at least 10 observations")
    if max_lag is None:
        max_lag = max(1, int(12.0 * (n / 100.0) ** 0.25))
    max_lag = min(max_lag, n // 3)

    dy = diff(y)

    def _fit(lag: int) -> dict:
        # regress dy[lag:] on [1, (trend,), y[lag:-1] or y[lag:], dy lags...]
        # careful with alignment: dy has n-1 obs; y level lagged once.
        rows: list[list[float]] = []
        target: list[float] = []
        for t in range(lag, n - 1):
            row = [1.0]
            if regression == "ct":
                row.append(float(t + 1))  # trend 1..T
            row.append(y[t])  # y_{t} aligns with dy_t = y_{t+1} - y_t
            for j in range(1, lag + 1):
                row.append(dy[t - j])
            rows.append(row)
            target.append(dy[t])
        return ols_regression(rows, target)

    best, best_aic = None, math.inf
    for lag in range(max_lag + 1):
        try:
            fit = _fit(lag)
        except ValueError:
            continue
        k = len(fit["beta"])
        nobs = len(fit["residuals"])
        aic = nobs * math.log(max(fit["sse"] / nobs, 1e-300)) + 2 * k
        if aic < best_aic:
            best_aic, best = aic, (lag, fit)

    if best is None:
        raise ValueError("adf_test: no lag specification could be fit")
    lag, fit = best
    gamma_idx = 2 if regression == "ct" else 1
    gamma, se_gamma = fit["beta"][gamma_idx], fit["se"][gamma_idx]
    stat = gamma / se_gamma if se_gamma > 0 else 0.0
    nobs = len(fit["residuals"])
    return ADFResult(
        stat=stat,
        lags_used=lag,
        nobs=nobs,
        crit_1pct=_adf_critical(regression, nobs, 0.01),
        crit_5pct=_adf_critical(regression, nobs, 0.05),
        crit_10pct=_adf_critical(regression, nobs, 0.10),
        regression=regression,
    )


# ---------------------------------------------------------------------------
# cointegration (Engle-Granger two-step) and pair analytics
# ---------------------------------------------------------------------------


def engle_granger(
    price_a: list[float],
    price_b: list[float],
    regression: str = "c",
) -> dict:
    """Engle-Granger two-step cointegration test.

    Step 1: OLS of ``price_a`` on ``price_b`` (the cointegrating regression).
    Step 2: ADF test on the residuals, judged against Engle-Granger critical
    values (stricter than plain ADF values — see ``_EG_CRIT``).

    Returns hedge_ratio, intercept, residuals (= the spread), the raw ADF
    result, the EG 5% critical value, and the cointegration decision.
    """
    if len(price_a) != len(price_b):
        raise ValueError("price legs must have equal length")
    a = [float(v) for v in price_a]
    b = [float(v) for v in price_b]
    hedge_ratio, intercept, residuals = ols_slope_intercept(b, a)
    adf = adf_test(residuals, regression=regression)
    eg_crit_5pct = _EG_CRIT[0.05]
    return {
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "residuals": residuals,
        "spread": residuals,
        "adf": adf,
        "eg_crit_5pct": eg_crit_5pct,
        "cointegrated": adf.stat < eg_crit_5pct,
    }


def spread_series(
    price_a: list[float], price_b: list[float], hedge_ratio: float, intercept: float = 0.0
) -> list[float]:
    """The tradable spread: A - beta * B - intercept."""
    return [a - hedge_ratio * b - intercept for a, b in zip(price_a, price_b)]


def half_life(spread: list[float]) -> float | None:
    """Mean-reversion half-life in bars via OLS of Δspread on lagged spread.

    Returns None when the spread shows no mean reversion (slope >= 0).
    """
    s = [float(v) for v in spread]
    if len(s) < 10:
        return None
    y = diff(s)
    x = s[:-1]
    try:
        X = [[1.0, xi] for xi in x]
        fit = ols_regression(X, y)
    except ValueError:
        return None
    lam = fit["beta"][1]
    if lam >= 0:
        return None
    return -math.log(2.0) / lam


def rolling_zscore(series: list[float], window: int) -> list[float | None]:
    """Rolling z-score using only data up to each bar (no lookahead)."""
    out: list[float | None] = [None] * len(series)
    for i in range(window - 1, len(series)):
        w = series[i - window + 1 : i + 1]
        sd = stdev(w)
        out[i] = (series[i] - mean(w)) / sd if sd > 0 else 0.0
    return out
