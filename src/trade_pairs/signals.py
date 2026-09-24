"""Signal generation on a pair's spread.

Classic z-score rules: enter when the spread is stretched (|z| >= entry_z),
exit when it snaps back (|z| <= exit_z).  All statistics are rolling and
causal — no lookahead.
"""

from __future__ import annotations

from .models import PairSignal
from .stats import rolling_zscore


def generate_signals(
    spread: list[float],
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    window: int = 60,
) -> list[PairSignal]:
    """Walk the spread and emit long_spread / short_spread / exit signals.

    Position state machine: flat → long/short on entry → flat on exit.
    A fresh entry signal in the opposite direction flips the position
    (exit + new entry are emitted as two signals on the same bar).
    """
    if entry_z <= exit_z:
        raise ValueError("entry_z must exceed exit_z")
    zs = rolling_zscore([float(v) for v in spread], window)
    signals: list[PairSignal] = []
    pos = 0  # +1 long spread, -1 short spread, 0 flat
    for i, z in enumerate(zs):
        if z is None:
            continue
        if pos == 0:
            if z <= -entry_z:
                pos = 1
                signals.append(PairSignal(i, "long_spread", z, spread[i]))
            elif z >= entry_z:
                pos = -1
                signals.append(PairSignal(i, "short_spread", z, spread[i]))
        elif pos == 1:
            if abs(z) <= exit_z:
                pos = 0
                signals.append(PairSignal(i, "exit", z, spread[i]))
            elif z >= entry_z:  # flip
                pos = -1
                signals.append(PairSignal(i, "exit", z, spread[i]))
                signals.append(PairSignal(i, "short_spread", z, spread[i]))
        else:  # pos == -1
            if abs(z) <= exit_z:
                pos = 0
                signals.append(PairSignal(i, "exit", z, spread[i]))
            elif z <= -entry_z:  # flip
                pos = 1
                signals.append(PairSignal(i, "exit", z, spread[i]))
                signals.append(PairSignal(i, "long_spread", z, spread[i]))
    return signals


def current_zscore(spread: list[float], window: int = 60) -> float | None:
    """The latest rolling z-score, or None when history is too short."""
    zs = rolling_zscore([float(v) for v in spread], window)
    return zs[-1] if zs else None
