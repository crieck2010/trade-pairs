"""Lightweight spread backtester.

Trades the spread in unit positions (+1 long spread, -1 short, 0 flat) and
reports the metrics the suite cares about: total return, annualized Sharpe,
max drawdown, and win rate over round trips.  Commissions/slippage are
modelled as a per-side cost in spread units so results stay honest.

This is intentionally small: for portfolio-level simulation with lot
matching and multi-asset books, pipe the signals into ``trade-backtest``
via ``adapters.spread_to_dict_bars``.
"""

from __future__ import annotations

import math

from .models import PairSignal, PairsBacktest
from .signals import generate_signals
from .stats import spread_series


def backtest_spread(
    price_a: list[float],
    price_b: list[float],
    hedge_ratio: float,
    symbol_a: str = "A",
    symbol_b: str = "B",
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    window: int = 60,
    cost_per_side: float = 0.0,
    annualize: int = 252,
) -> PairsBacktest:
    """Backtest z-score signals on the A − β·B spread."""
    a = [float(v) for v in price_a]
    b = [float(v) for v in price_b]
    if len(a) != len(b) or len(a) < window + 2:
        raise ValueError("legs must have equal length and exceed the z window")
    spread = spread_series(a, b, hedge_ratio)
    signals = generate_signals(spread, entry_z, exit_z, window)

    # index → net position change implied by signals on that bar
    pos, equity = 0, 1.0
    equity_curve = [equity]
    sig_by_idx: dict[int, list[PairSignal]] = {}
    for s in signals:
        sig_by_idx.setdefault(s.index, []).append(s)

    round_trips: list[float] = []
    open_equity = 0.0
    wins = 0

    for i in range(1, len(spread)):
        # mark-to-market FIRST on the move into bar i, using the position
        # held from the previous bar; signals firing at bar i (computed from
        # data through bar i's close) take effect for the i -> i+1 move.
        # This ordering is what keeps the backtest free of lookahead bias.
        scale = max(abs(spread[i - 1]), 1e-9)
        equity += pos * (spread[i] - spread[i - 1]) / scale * 0.01
        equity_curve.append(equity)
        for s in sig_by_idx.get(i, []):
            if s.action == "long_spread":
                pos = 1
                equity -= cost_per_side
                open_equity = equity
            elif s.action == "short_spread":
                pos = -1
                equity -= cost_per_side
                open_equity = equity
            else:  # exit
                if pos != 0:
                    rt = equity - open_equity
                    round_trips.append(rt)
                    if rt > 0:
                        wins += 1
                pos = 0
                equity -= cost_per_side

    if pos != 0:  # close dangling position at the end
        rt = equity - open_equity
        round_trips.append(rt)
        if rt > 0:
            wins += 1

    total_return = equity - 1.0
    rets = [e2 - e1 for e1, e2 in zip(equity_curve, equity_curve[1:])]
    mu = sum(rets) / len(rets) if rets else 0.0
    sd = math.sqrt(sum((r - mu) ** 2 for r in rets) / max(len(rets) - 1, 1))
    sharpe = (mu / sd * math.sqrt(annualize)) if sd > 0 else 0.0

    peak, max_dd = equity_curve[0], 0.0
    for e in equity_curve:
        peak = max(peak, e)
        max_dd = min(max_dd, (e - peak) / peak if peak else 0.0)

    n_rt = len(round_trips)
    gross_win = sum(r for r in round_trips if r > 0)
    gross_loss = -sum(r for r in round_trips if r < 0)
    return PairsBacktest(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        hedge_ratio=hedge_ratio,
        n_bars=len(spread),
        n_round_trips=n_rt,
        total_return=total_return,
        sharpe_annualized=sharpe,
        max_drawdown=max_dd,
        win_rate=(wins / n_rt) if n_rt else 0.0,
        avg_win=(gross_win / wins) if wins else 0.0,
        avg_loss=(gross_loss / (n_rt - wins)) if n_rt - wins else 0.0,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else math.inf,
        final_equity=equity,
        equity_curve=equity_curve,
    )
