"""Command-line interface for trade-pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys

from . import __version__
from .adapters import to_agent_ideas
from .licensing import check_license, check_update
from .pipeline import analyze_pair
from .screening import find_pairs


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def load_csv(path: str) -> dict[str, list[float]]:
    """Load a wide CSV: first column date (ignored), rest are symbols."""
    out: dict[str, list[float]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = [f for f in (reader.fieldnames or []) if f]
        syms = fields[1:]
        series = {s: [] for s in syms}
        for row in reader:
            for s in syms:
                try:
                    series[s].append(float(row[s]))
                except (ValueError, TypeError):
                    pass
        out = {s: v for s, v in series.items() if v}
    return out


def demo_universe(n: int = 300, seed: int = 2) -> dict[str, list[float]]:
    """Synthetic universe with two embedded cointegrated pairs + noise."""
    rng = random.Random(seed)

    def random_walk(start: float, vol: float) -> list[float]:
        p, out = start, []
        for _ in range(n):
            p = max(p * (1 + rng.gauss(0, vol)), 0.01)
            out.append(p)
        return out

    def coint_pair(start_a: float, beta: float, vol: float, spread_sd: float):
        b = random_walk(start_a / beta, vol)
        a, spread = [], 0.0
        for pb in b:
            spread = 0.9 * spread + rng.gauss(0, spread_sd)
            a.append(max(beta * pb + spread, 0.01))
        return a, b

    uni: dict[str, list[float]] = {}
    a1, b1 = coint_pair(100.0, 1.5, 0.01, 0.8)
    uni["AAA"], uni["AAB"] = a1, b1
    a2, b2 = coint_pair(50.0, 0.7, 0.012, 0.5)
    uni["CCC"], uni["CCD"] = a2, b2
    for sym in ("EEE", "FFF", "GGG", "HHH"):
        uni[sym] = random_walk(80.0, 0.015)
    return uni


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_screen(args: argparse.Namespace) -> int:
    prices = load_csv(args.csv) if args.csv else demo_universe()
    pairs = find_pairs(
        prices,
        lookback=args.lookback,
        min_correlation=args.min_correlation,
        max_pairs=args.top,
    )
    rows = [p.to_dict() for p in pairs]
    if args.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        if not rows:
            print("No pairs passed the screen.")
            return 0
        for p in pairs:
            flag = "COINT" if p.cointegrated else "     "
            hl = f"{p.half_life_bars:.1f}" if p.half_life_bars else "n/a"
            print(f"[{flag}] {p.symbol_a}/{p.symbol_b}  corr={p.correlation:+.3f} "
                  f"beta={p.hedge_ratio:.3f} adf={p.adf.stat:.2f} "
                  f"(EG 5% crit {p.eg_crit_5pct:.2f}) half-life={hl} bars")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    prices = load_csv(args.csv) if args.csv else demo_universe()
    a, b = args.pair
    if a not in prices or b not in prices:
        print(f"symbols {a!r}/{b!r} not in data", file=sys.stderr)
        return 2
    report = analyze_pair(a, b, prices[a], prices[b],
                          entry_z=args.entry_z, exit_z=args.exit_z,
                          window=args.window)
    if args.format == "json":
        print(json.dumps(report, indent=2))
        return 0
    p = report["pair"]
    print(f"Pair {a}/{b}  (n={p['lookback']})")
    print(f"  correlation      : {p['correlation']:+.3f}")
    print(f"  hedge ratio (β)  : {p['hedge_ratio']:.4f}")
    print(f"  ADF stat         : {p['adf']['stat']:.3f}  (EG 5% crit {p['eg_crit_5pct']:.2f})")
    print(f"  cointegrated     : {p['cointegrated']}")
    hl = p["half_life_bars"]
    print(f"  half-life        : {f'{hl:.1f} bars' if hl else 'no mean reversion'}")
    z = report["current_zscore"]
    print(f"  current z-score  : {z:+.2f}" if z is not None else "  current z-score  : n/a")
    if report["latest_signal_text"]:
        print(f"  latest signal    : {report['latest_signal_text']}")
    if "backtest_summary" in report:
        print(f"  {report['backtest_summary']}")
    if args.ideas:
        from .models import PairCandidate, ADFResult, PairSignal
        adf = ADFResult(**{k: v for k, v in p["adf"].items() if k != "reject_5pct"})
        cand = PairCandidate(a, b, p["correlation"], p["hedge_ratio"],
                             p["intercept"], adf, p["half_life_bars"],
                             p["lookback"], p["cointegrated"],
                             p["eg_crit_5pct"])
        sig = None
        if report["latest_signal"]:
            s = report["latest_signal"]
            sig = PairSignal(s["index"], s["action"], s["zscore"], s["spread"])
        print(json.dumps(to_agent_ideas(cand, sig), indent=2))
    return 0


def cmd_license(_args: argparse.Namespace) -> int:
    print(json.dumps(check_license(), indent=2))
    return 0


def cmd_update_check(_args: argparse.Namespace) -> int:
    print(json.dumps(check_update(), indent=2))
    return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="trade-pairs",
                                 description="Pairs trading / stat-arb engine")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("screen", help="screen a universe for cointegrated pairs")
    p.add_argument("--csv", help="wide CSV (date,sym1,sym2,...); default: synthetic demo")
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--min-correlation", type=float, default=0.70)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("analyze", help="full workup of one pair")
    p.add_argument("pair", nargs=2, metavar=("A", "B"))
    p.add_argument("--csv", help="wide CSV; default: synthetic demo")
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--exit-z", type=float, default=0.5)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--ideas", action="store_true", help="also emit agent ideas JSON")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("license", help="check the license-key hook")
    p.set_defaults(func=cmd_license)
    p = sub.add_parser("update-check", help="check for a newer release")
    p.set_defaults(func=cmd_update_check)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
