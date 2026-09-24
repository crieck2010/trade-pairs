"""End-to-end pair analysis pipeline: screen → estimate → signal."""

from __future__ import annotations

from .backtest import backtest_spread
from .models import PairCandidate
from .signals import current_zscore, generate_signals
from .stats import engle_granger, half_life, spread_series


def analyze_pair(
    symbol_a: str,
    symbol_b: str,
    price_a: list[float],
    price_b: list[float],
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    window: int = 60,
    run_backtest: bool = True,
) -> dict:
    """Full workup of one pair.  Returns a JSON-serializable report dict."""
    pa = [float(v) for v in price_a]
    pb = [float(v) for v in price_b]
    eg = engle_granger(pa, pb)
    spread = spread_series(pa, pb, eg["hedge_ratio"], eg["intercept"])
    z = current_zscore(spread, window)
    signals = generate_signals(spread, entry_z, exit_z, window)
    live = signals[-1] if signals else None

    candidate = PairCandidate(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        correlation=0.0,  # single-pair path; screening fills this in
        hedge_ratio=eg["hedge_ratio"],
        intercept=eg["intercept"],
        adf=eg["adf"],
        half_life_bars=half_life(spread),
        lookback=len(pa),
        cointegrated=eg["cointegrated"],
        eg_crit_5pct=eg["eg_crit_5pct"],
    )
    # correlation needs log prices; compute cheaply here for completeness
    from .stats import correlation
    import math
    la = [math.log(max(v, 1e-12)) for v in pa]
    lb = [math.log(max(v, 1e-12)) for v in pb]
    candidate.correlation = correlation(la, lb)

    report = {
        "pair": candidate.to_dict(),
        "current_zscore": z,
        "n_signals": len(signals),
        "latest_signal": live.to_dict() if live else None,
        "latest_signal_text": live.describe(symbol_a, symbol_b, eg["hedge_ratio"]) if live else None,
    }
    if run_backtest:
        bt = backtest_spread(pa, pb, eg["hedge_ratio"], symbol_a, symbol_b,
                             entry_z, exit_z, window)
        report["backtest"] = bt.to_dict()
        report["backtest_summary"] = bt.summary()
    return report
