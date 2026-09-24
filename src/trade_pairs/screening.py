"""Universe screening: find cointegrated pairs worth trading.

Pipeline: correlation pre-filter over the lookback window → Engle-Granger
cointegration test on the top candidates → rank by ADF strength.
"""

from __future__ import annotations

import itertools
import math

from .models import PairCandidate
from .stats import correlation, engle_granger, half_life


def _log_prices(prices: list[float]) -> list[float]:
    return [math.log(max(p, 1e-12)) for p in prices]


def find_pairs(
    prices: dict[str, list[float]],
    lookback: int = 252,
    min_correlation: float = 0.70,
    max_candidates: int = 40,
    max_pairs: int = 15,
) -> list[PairCandidate]:
    """Screen a ``{symbol: price_history}`` universe for tradable pairs.

    1. Take the last ``lookback`` bars of each symbol.
    2. Rank all pairs by |correlation| of log prices; keep the top
       ``max_candidates`` above ``min_correlation``.
    3. Run the Engle-Granger cointegration test on each candidate.
    4. Return up to ``max_pairs`` candidates, cointegrated first, then by
       ADF statistic (most negative = strongest mean reversion).
    """
    syms = [s for s, p in prices.items() if len(p) >= lookback]
    if len(syms) < 2:
        return []
    windowed = {s: _log_prices(prices[s][-lookback:]) for s in syms}

    scored: list[tuple[float, str, str]] = []
    for a, b in itertools.combinations(sorted(syms), 2):
        corr = correlation(windowed[a], windowed[b])
        if abs(corr) >= min_correlation:
            scored.append((abs(corr), a, b))
    scored.sort(reverse=True)
    scored = scored[:max_candidates]

    out: list[PairCandidate] = []
    for _, a, b in scored:
        pa = [float(v) for v in prices[a][-lookback:]]
        pb = [float(v) for v in prices[b][-lookback:]]
        eg = engle_granger(pa, pb)
        out.append(
            PairCandidate(
                symbol_a=a,
                symbol_b=b,
                correlation=correlation(windowed[a], windowed[b]),
                hedge_ratio=eg["hedge_ratio"],
                intercept=eg["intercept"],
                adf=eg["adf"],
                half_life_bars=half_life(eg["spread"]),
                lookback=lookback,
                cointegrated=eg["cointegrated"],
                eg_crit_5pct=eg["eg_crit_5pct"],
            )
        )
    out.sort(key=lambda c: (not c.cointegrated, c.adf.stat))
    return out[:max_pairs]
