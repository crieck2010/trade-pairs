"""Tests for trade-pairs.  All stochastic fixtures use fixed seeds."""

from __future__ import annotations

import json
import math
import random

import pytest

from trade_pairs import (
    adf_test,
    analyze_pair,
    backtest_spread,
    correlation,
    current_zscore,
    engle_granger,
    find_pairs,
    generate_signals,
    half_life,
    ols_slope_intercept,
    rolling_zscore,
    spread_series,
)
from trade_pairs.adapters import (
    register_strategy,
    spread_to_dict_bars,
    to_agent_ideas,
)
from trade_pairs.cli import demo_universe
from trade_pairs.models import PairCandidate


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _rng(seed: int = 11) -> random.Random:
    return random.Random(seed)


def random_walk(rng: random.Random, n: int, start: float = 100.0, vol: float = 0.01):
    p, out = start, []
    for _ in range(n):
        p = max(p * (1 + rng.gauss(0, vol)), 0.01)
        out.append(p)
    return out


def coint_pair(rng: random.Random, n: int, beta: float, spread_sd: float = 0.8):
    b = random_walk(rng, n, start=100.0)
    a, s = [], 0.0
    for pb in b:
        s = 0.9 * s + rng.gauss(0, spread_sd)
        a.append(max(beta * pb + s, 0.01))
    return a, b


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------

def test_ols_recovers_known_line():
    rng = _rng()
    x = [float(i) for i in range(100)]
    y = [3.0 * xi + 7.0 + rng.gauss(0, 0.5) for xi in x]
    slope, intercept, resid = ols_slope_intercept(x, y)
    assert slope == pytest.approx(3.0, abs=0.05)
    assert intercept == pytest.approx(7.0, abs=0.5)
    assert abs(sum(resid)) < 1e-6  # residuals sum to ~0 with intercept


def test_correlation_basics():
    x = [float(i) for i in range(50)]
    assert correlation(x, x) == pytest.approx(1.0)
    assert correlation(x, [-v for v in x]) == pytest.approx(-1.0)
    assert correlation([1.0] * 50, x) == 0.0  # no variance -> 0


def test_adf_rejects_stationary_ar1():
    rng = _rng(21)
    y, v = [], 0.0
    for _ in range(300):
        v = 0.5 * v + rng.gauss(0, 1.0)
        y.append(v)
    res = adf_test(y)
    assert res.reject_5pct, f"AR(1) should reject unit root, stat={res.stat:.2f}"
    assert res.stat < res.crit_5pct < 0


def test_adf_keeps_random_walk():
    rng = _rng(22)
    y = random_walk(rng, 300, vol=0.02)
    res = adf_test(y)
    assert not res.reject_5pct, f"random walk should keep H0, stat={res.stat:.2f}"


def test_adf_result_serializes():
    rng = _rng(23)
    y = random_walk(rng, 120)
    d = adf_test(y).to_dict()
    json.dumps(d)
    assert set(d) >= {"stat", "crit_5pct", "reject_5pct"}


def test_engle_granger_finds_cointegration():
    rng = _rng(31)
    a, b = coint_pair(rng, 300, beta=1.5)
    eg = engle_granger(a, b)
    assert eg["cointegrated"], f"stat={eg['adf'].stat:.2f}"
    assert eg["hedge_ratio"] == pytest.approx(1.5, abs=0.15)


def test_engle_granger_rejects_independent_walks():
    rng = _rng(32)
    a = random_walk(rng, 300, vol=0.015)
    b = random_walk(random.Random(99), 300, vol=0.015)
    eg = engle_granger(a, b)
    assert not eg["cointegrated"], f"stat={eg['adf'].stat:.2f}"
    # EG bar is stricter than the plain ADF bar:
    assert eg["eg_crit_5pct"] < eg["adf"].crit_5pct


def test_half_life_recovers_ar1():
    rng = _rng(41)
    # AR(1) with phi=0.9 -> half-life = -ln2/ln(0.9) ≈ 6.58
    s, v = [], 0.0
    for _ in range(2000):
        v = 0.9 * v + rng.gauss(0, 1.0)
        s.append(v)
    hl = half_life(s)
    assert hl is not None
    assert hl == pytest.approx(6.58, rel=0.25)


def test_half_life_none_or_huge_without_mean_reversion():
    # a random walk has no true mean reversion: OLS may still fit a tiny
    # negative slope by chance, but the implied half-life must be enormous
    rng = _rng(42)
    hl = half_life(random_walk(rng, 400, vol=0.02))
    assert hl is None or hl > 100


def test_rolling_zscore_no_lookahead():
    s = [float(i) for i in range(100)]
    zs = rolling_zscore(s, window=20)
    assert all(z is None for z in zs[:19])
    assert all(z is not None for z in zs[19:])
    # strictly increasing series -> constant positive z within window
    assert zs[19] == pytest.approx(zs[50])


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------

def test_generate_signals_entry_exit():
    rng = _rng(51)
    # spread: flat, then a spike down (long entry), then reversion (exit)
    spread = [rng.gauss(0, 1.0) for _ in range(80)]
    spread += [-6.0] * 5
    spread += [rng.gauss(0, 1.0) for _ in range(40)]
    sigs = generate_signals(spread, entry_z=2.0, exit_z=0.5, window=60)
    actions = [s.action for s in sigs]
    assert "long_spread" in actions
    assert "exit" in actions
    # entries reference the bar they fired on
    for s in sigs:
        assert s.spread == pytest.approx(spread[s.index])


def test_generate_signals_bad_thresholds():
    with pytest.raises(ValueError):
        generate_signals([1.0] * 100, entry_z=0.5, exit_z=2.0)


def test_current_zscore_none_when_short():
    assert current_zscore([1.0, 2.0, 3.0], window=60) is None


# --------------------------------------------------------------------------
# screening + pipeline
# --------------------------------------------------------------------------

def test_find_pairs_discovers_embedded_pairs():
    uni = demo_universe(n=300, seed=2)
    pairs = find_pairs(uni, lookback=250, max_pairs=10)
    found = {(p.symbol_a, p.symbol_b) for p in pairs if p.cointegrated}
    assert ("AAA", "AAB") in found
    assert ("CCC", "CCD") in found
    # cointegrated pairs rank before non-cointegrated ones
    flags = [p.cointegrated for p in pairs]
    assert flags == sorted(flags, reverse=True)


def test_find_pairs_needs_two_symbols():
    assert find_pairs({"A": [1.0] * 300}) == []


def test_analyze_pair_report_is_json_serializable():
    uni = demo_universe(n=300, seed=2)
    rep = analyze_pair("AAA", "AAB", uni["AAA"], uni["AAB"])
    json.dumps(rep)
    assert rep["pair"]["cointegrated"] is True
    assert rep["pair"]["hedge_ratio"] == pytest.approx(1.5, abs=0.2)
    assert "backtest_summary" in rep


def test_backtest_spread_metrics_sane():
    uni = demo_universe(n=400, seed=2)
    bt = backtest_spread(uni["AAA"], uni["AAB"], hedge_ratio=1.5,
                         symbol_a="AAA", symbol_b="AAB")
    assert bt.n_bars == 400
    assert bt.n_round_trips > 0
    assert 0.0 <= bt.win_rate <= 1.0
    assert bt.max_drawdown <= 0.0
    assert math.isfinite(bt.sharpe_annualized)
    assert len(bt.equity_curve) == 400
    json.dumps(bt.to_dict())


def test_backtest_profitable_on_clean_cycle():
    # deterministic mean-reverting spread (sine wave): the z-score strategy
    # should capture the oscillations with a healthy win rate
    import math as _math
    cyc = [10 * _math.sin(2 * _math.pi * i / 40) for i in range(400)]
    pa = [100.0 + s for s in cyc]
    pb = [100.0] * 400
    bt = backtest_spread(pa, pb, hedge_ratio=1.0, entry_z=1.0, exit_z=0.25,
                         window=30)
    assert bt.n_round_trips >= 5
    assert bt.total_return > 0
    assert bt.win_rate > 0.5


def test_backtest_with_costs_is_not_better():
    uni = demo_universe(n=400, seed=2)
    free = backtest_spread(uni["AAA"], uni["AAB"], 1.5)
    paid = backtest_spread(uni["AAA"], uni["AAB"], 1.5, cost_per_side=0.01)
    assert paid.final_equity <= free.final_equity + 1e-9


# --------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------

def test_to_agent_ideas_shapes():
    uni = demo_universe(n=300, seed=2)
    rep = analyze_pair("AAA", "AAB", uni["AAA"], uni["AAB"], run_backtest=False)
    p = rep["pair"]
    from trade_pairs.models import ADFResult, PairSignal
    adf = ADFResult(**{k: v for k, v in p["adf"].items() if k != "reject_5pct"})
    cand = PairCandidate("AAA", "AAB", p["correlation"], p["hedge_ratio"],
                         p["intercept"], adf, p["half_life_bars"],
                         p["lookback"], p["cointegrated"], p["eg_crit_5pct"])
    watch = to_agent_ideas(cand, None)
    assert watch[0]["direction"] == "watch"
    sig = PairSignal(10, "long_spread", -2.5, 1.0)
    ideas = to_agent_ideas(cand, sig)
    assert ideas[0]["direction"] == "long_spread"
    assert 0.0 < ideas[0]["conviction"] <= 1.0
    assert "thesis" in ideas[0]
    json.dumps(ideas)


def test_spread_to_dict_bars_shape():
    bars = spread_to_dict_bars([100.0, 101.0, 102.0], [50.0, 50.5, 51.0], 2.0)
    assert len(bars) == 3
    b = bars[0]
    assert set(b) >= {"open", "high", "low", "close", "symbol", "time"}
    assert b["high"] >= b["low"]


def test_register_strategy_descriptor():
    d = register_strategy()
    assert d["family"] == "statistical_arbitrage"
    assert d["engine"] == "trade-pairs"
    json.dumps(d)
