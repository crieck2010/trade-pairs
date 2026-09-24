"""Data models for the trade-pairs engine.

All models are plain dataclasses with JSON-friendly ``to_dict`` helpers so
results cross process and agent boundaries without the engine leaking in.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ADFResult:
    """Outcome of an Augmented Dickey-Fuller unit-root test."""

    stat: float
    lags_used: int
    nobs: int
    crit_1pct: float
    crit_5pct: float
    crit_10pct: float
    regression: str = "c"

    @property
    def reject_5pct(self) -> bool:
        """True when the unit-root null is rejected at the 5% level."""
        return self.stat < self.crit_5pct

    def to_dict(self) -> dict:
        return {
            "stat": self.stat,
            "lags_used": self.lags_used,
            "nobs": self.nobs,
            "crit_1pct": self.crit_1pct,
            "crit_5pct": self.crit_5pct,
            "crit_10pct": self.crit_10pct,
            "regression": self.regression,
            "reject_5pct": self.reject_5pct,
        }


@dataclass
class PairCandidate:
    """A screened pair, ranked by cointegration strength."""

    symbol_a: str
    symbol_b: str
    correlation: float
    hedge_ratio: float
    intercept: float
    adf: ADFResult
    half_life_bars: float | None
    lookback: int
    cointegrated: bool = False  # Engle-Granger decision (stricter than ADF)
    eg_crit_5pct: float = -3.34  # the EG bar the decision was judged against

    def to_dict(self) -> dict:
        return {
            "symbol_a": self.symbol_a,
            "symbol_b": self.symbol_b,
            "correlation": self.correlation,
            "hedge_ratio": self.hedge_ratio,
            "intercept": self.intercept,
            "half_life_bars": self.half_life_bars,
            "lookback": self.lookback,
            "cointegrated": self.cointegrated,
            "eg_crit_5pct": self.eg_crit_5pct,
            "adf": self.adf.to_dict(),
        }


@dataclass
class PairSignal:
    """One trading instruction on a pair's spread."""

    index: int
    action: str  # "long_spread" | "short_spread" | "exit"
    zscore: float
    spread: float

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "action": self.action,
            "zscore": self.zscore,
            "spread": self.spread,
        }

    def describe(self, symbol_a: str, symbol_b: str, hedge_ratio: float) -> str:
        if self.action == "long_spread":
            return (
                f"LONG spread {symbol_a}/{symbol_b}: buy {symbol_a}, "
                f"short {hedge_ratio:.3f}x {symbol_b} (z={self.zscore:+.2f})"
            )
        if self.action == "short_spread":
            return (
                f"SHORT spread {symbol_a}/{symbol_b}: short {symbol_a}, "
                f"buy {hedge_ratio:.3f}x {symbol_b} (z={self.zscore:+.2f})"
            )
        return f"EXIT spread {symbol_a}/{symbol_b} (z={self.zscore:+.2f})"


@dataclass
class PairsBacktest:
    """Performance summary of a spread backtest."""

    symbol_a: str
    symbol_b: str
    hedge_ratio: float
    n_bars: int
    n_round_trips: int
    total_return: float
    sharpe_annualized: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    final_equity: float
    equity_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "equity_curve"}
        d["equity_curve"] = self.equity_curve
        return d

    def summary(self) -> str:
        return (
            f"{self.symbol_a}/{self.symbol_b} spread backtest: "
            f"return {self.total_return:+.2%}, Sharpe {self.sharpe_annualized:.2f}, "
            f"max DD {self.max_drawdown:.2%}, win rate {self.win_rate:.1%} "
            f"over {self.n_round_trips} round trips"
        )
