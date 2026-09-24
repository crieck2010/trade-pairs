"""Pairs trading / statistical-arbitrage engine for the trade-suite.

Screens a universe for cointegrated pairs (Engle-Granger), estimates hedge
ratios, trades the spread on z-score signals, and backtests the result.
Stdlib-only, deterministic, and JSON-serializable end to end.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .backtest import backtest_spread
from .models import ADFResult, PairCandidate, PairSignal, PairsBacktest
from .pipeline import analyze_pair
from .screening import find_pairs
from .signals import current_zscore, generate_signals
from .stats import (
    adf_test,
    correlation,
    engle_granger,
    half_life,
    ols_slope_intercept,
    rolling_zscore,
    spread_series,
)

__all__ = [
    "ADFResult",
    "PairCandidate",
    "PairSignal",
    "PairsBacktest",
    "adf_test",
    "analyze_pair",
    "backtest_spread",
    "correlation",
    "current_zscore",
    "engle_granger",
    "find_pairs",
    "generate_signals",
    "half_life",
    "ols_slope_intercept",
    "rolling_zscore",
    "spread_series",
    "__version__",
]
