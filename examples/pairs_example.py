"""End-to-end example: screen a universe, analyze the best pair, emit ideas."""

from trade_pairs import analyze_pair, find_pairs
from trade_pairs.adapters import to_agent_ideas
from trade_pairs.cli import demo_universe
from trade_pairs.models import ADFResult, PairCandidate, PairSignal

prices = demo_universe(n=300, seed=2)

print("=== screening ===")
pairs = find_pairs(prices, lookback=250, max_pairs=5)
for p in pairs:
    flag = "COINT" if p.cointegrated else "     "
    print(f"[{flag}] {p.symbol_a}/{p.symbol_b} beta={p.hedge_ratio:.3f} "
          f"adf={p.adf.stat:.2f}")

best = next(p for p in pairs if p.cointegrated)
print(f"\n=== analyzing {best.symbol_a}/{best.symbol_b} ===")
report = analyze_pair(best.symbol_a, best.symbol_b,
                      prices[best.symbol_a], prices[best.symbol_b])
print(report["backtest_summary"])
print("live signal:", report["latest_signal_text"])

print("\n=== agent ideas ===")
p = report["pair"]
adf = ADFResult(**{k: v for k, v in p["adf"].items() if k != "reject_5pct"})
cand = PairCandidate(best.symbol_a, best.symbol_b, p["correlation"],
                     p["hedge_ratio"], p["intercept"], adf,
                     p["half_life_bars"], p["lookback"], p["cointegrated"],
                     p["eg_crit_5pct"])
sig = None
if report["latest_signal"]:
    s = report["latest_signal"]
    sig = PairSignal(s["index"], s["action"], s["zscore"], s["spread"])
for idea in to_agent_ideas(cand, sig):
    print(idea["direction"], f"{idea['conviction']:.2f}", "-", idea["thesis"])
