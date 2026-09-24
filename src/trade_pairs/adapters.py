"""Interop adapters: plain-data bridges to sibling suite modules.

Every adapter is lazy — sibling packages are imported inside the function
so ``trade-pairs`` stays importable and testable on its own.
"""

from __future__ import annotations

from .models import PairCandidate, PairSignal


def to_agent_ideas(
    candidate: PairCandidate,
    signal: PairSignal | None = None,
    min_conviction: float = 0.0,
) -> list[dict]:
    """Convert a pair (and optional live signal) into agent idea dicts.

    A long/short *spread* becomes a pair idea with explicit legs, so a
    downstream researcher (e.g. a trade-agents pairs scout) can turn it
    into two orders.  Conviction scales with |z|/entry distance.
    """
    ideas: list[dict] = []
    base = {
        "strategy": "pairs",
        "symbol_a": candidate.symbol_a,
        "symbol_b": candidate.symbol_b,
        "hedge_ratio": candidate.hedge_ratio,
        "correlation": candidate.correlation,
        "cointegrated": candidate.cointegrated,
        "adf_stat": candidate.adf.stat,
        "half_life_bars": candidate.half_life_bars,
        "source": "trade-pairs",
    }
    if signal is None:
        idea = dict(base)
        idea.update({"direction": "watch", "conviction": 0.0,
                     "thesis": "Cointegrated pair under watch; no entry signal."})
        return [idea]
    conviction = min(abs(signal.zscore) / 3.0, 1.0)
    if conviction < min_conviction:
        return []
    direction = {"long_spread": "long_spread", "short_spread": "short_spread",
                 "exit": "exit"}[signal.action]
    idea = dict(base)
    idea.update({
        "direction": direction,
        "conviction": round(conviction, 3),
        "zscore": signal.zscore,
        "thesis": signal.describe(candidate.symbol_a, candidate.symbol_b,
                                  candidate.hedge_ratio),
    })
    ideas.append(idea)
    return ideas


def spread_to_dict_bars(
    price_a: list[float],
    price_b: list[float],
    hedge_ratio: float,
    symbol_a: str = "A",
    symbol_b: str = "B",
) -> list[dict]:
    """Represent the spread as synthetic OHLC dict-bars.

    ``trade-backtest``'s structural ``normalize_bar`` adapter accepts plain
    dicts, so the spread can be simulated as a single synthetic instrument
    without importing trade-backtest here.
    """
    from .stats import spread_series

    spread = spread_series(price_a, price_b, hedge_ratio)
    bars = []
    for i, s in enumerate(spread):
        prev = spread[i - 1] if i else s
        bars.append({
            "time": i,
            "symbol": f"{symbol_a}/{symbol_b}",
            "open": prev,
            "high": max(prev, s),
            "low": min(prev, s),
            "close": s,
            "volume": 0.0,
        })
    return bars


def register_strategy() -> dict:
    """Describe this engine's strategy for the trade-strategies registry.

    Returns a plain descriptor (no import of trade-strategies needed); the
    registry side can wrap ``generate_signals`` via this metadata.
    """
    from . import signals as _sig  # noqa: F401  (keeps the seam explicit)

    return {
        "name": "pairs_zscore",
        "family": "statistical_arbitrage",
        "engine": "trade-pairs",
        "description": (
            "Engle-Granger cointegration screen + z-score spread trading: "
            "long/short the A - beta*B spread on |z| >= entry, exit on snap-back."
        ),
        "params": {"entry_z": 2.0, "exit_z": 0.5, "window": 60},
        "signal_fn": "trade_pairs.signals.generate_signals",
    }
