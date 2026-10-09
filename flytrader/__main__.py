"""flytrader: a FlyWire fruit-fly brain paper-trading Solana memecoins.

  python -m flytrader backtest --mint <MINT> --hours 6 [--train-hours 12]
  python -m flytrader live --mint <MINT>
  python -m flytrader demo            # synthetic market, no API key needed
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import market, paths, trader
from .circuits import Circuits
from .fly import LESIONS, Fly

OUT = paths.OUT


def printer(t, price, th):
    icon = {"BUY": "🪰🍬 BUY ", "SELL": "🪰💨 SELL"}[th.action]
    print(f"  {t:%m-%d %H:%M}  {icon}  @ {price:.3e}  {th.reason}")


def evaluate(name: str, fly: Fly, df: pd.DataFrame, start_ts: int | None, verbose: bool,
             fee: float = 0.0125, report: bool = True) -> dict:
    b = market.bars(df)
    f = market.features(b)
    if start_ts is not None:
        # train on everything before start_ts (learning online), then score only the test window
        t0 = pd.to_datetime(start_ts, unit="s")
        train_b, train_f = b[b.index < t0], f[f.index < t0]
        if len(train_b) > 20:
            print(f"training on {len(train_b)} minutes before the test window...")
            p_train, _ = trader.run(fly, train_b, train_f, paper=trader.Paper(fee=fee))
            print(f"  training PnL ${p_train.cash:+.2f} over {len(p_train.trades) // 2} trades")
        b, f = b[b.index >= t0], f[f.index >= t0]
        df = df[df.ts >= start_ts]
    print(f"testing on {len(b)} minutes, {len(df)} swaps")
    paper, log = trader.run(fly, b, f, paper=trader.Paper(fee=fee), warmup=15 if start_ts is None else 0,
                            log=printer if verbose else None)
    sells = [t for t in paper.trades if t["side"] == "sell"]
    fly_roi = paper.cash / paper.spent if paper.spent else 0.0
    w = market.wallet_pnl(df, b.close.iloc[-1], exit_fee=fee)
    hu = w[~w.bot]
    hold = int(np.mean([(pd.Timestamp(s["t"]) - pd.Timestamp(bu["t"])).total_seconds() / 60
                        for bu, s in zip(paper.trades[::2], paper.trades[1::2], strict=False)])) if sells else 1
    monkeys = [trader.random_trader(b, len(sells), max(hold, 1), s, paper.fee) for s in range(200)]
    res = {
        "name": name, "minutes": len(b), "swaps": len(df),
        "fly": {"trades": len(sells), "pnl_usd": paper.cash, "roi": fly_roi,
                "win_rate": float(np.mean([s["ret"] > 0 for s in sells])) if sells else 0.0,
                "avg_hold_min": hold},
        "wallets": wallet_stats(w, fly_roi),
        "humans": wallet_stats(hu, fly_roi),
        "momentum_rule_roi": trader.rule_trader(b, f, "momentum", fee),
        "dip_rule_roi": trader.rule_trader(b, f, "dip", fee),
        "fee": fee,
        "buy_and_hold_roi": float(b.close.iloc[-1] / b.open.iloc[0] * (1 - paper.fee) ** 2 - 1),
        "random_trader_roi_median": float(np.median(monkeys)),
        "fly_beats_random_pct": float(np.mean(np.array(monkeys) < fly_roi)),
    }
    if report:
        write_report(res, b, log, paper, hu if len(hu) else w)
    return res


def wallet_stats(w: pd.DataFrame, fly_roi: float) -> dict:
    if not len(w):
        return {"n": 0, "median_roi": None, "pct_profitable": None, "fly_beats_pct": None}
    return {"n": len(w), "median_roi": float(w.roi.median()), "pct_profitable": float((w.pnl > 0).mean()),
            "fly_beats_pct": float((w.roi < fly_roi).mean())}


def show(res: dict):
    fl, w, hu = res["fly"], res["wallets"], res["humans"]
    print("\n" + "=" * 60)
    print(f"🪰 {res['name']}: {res['minutes']} min, {res['swaps']} swaps")
    print(f"   fly:       {fl['trades']} trades, ROI {fl['roi']:+.1%}, PnL {fl['pnl_usd']:+.2f} USD, "
          f"win rate {fl['win_rate']:.0%}, avg hold {fl['avg_hold_min']}m")
    if w["n"]:
        print(f"   humans:    {w['n']} wallets, median ROI {w['median_roi']:+.1%}, "
              f"{w['pct_profitable']:.0%} profitable")
        print(f"   ➜ the fly beat {w['fly_beats_pct']:.0%} of all wallets", end="")
        if hu["n"]:
            print(f", {hu['fly_beats_pct']:.0%} of the {hu['n']} human-looking ones (median {hu['median_roi']:+.1%})")
        else:
            print()
    print(f"   buy&hold:  {res['buy_and_hold_roi']:+.1%}   random monkey (median): "
          f"{res['random_trader_roi_median']:+.1%}  (fly beats {res['fly_beats_random_pct']:.0%} of monkeys)")
    print(f"   rules:     momentum {res['momentum_rule_roi']:+.1%}/trade, "
          f"dip-buy {res['dip_rule_roi']:+.1%}/trade  (fees {res['fee']:.2%}/side)")
    print("=" * 60)


def write_report(res, b, log, paper, w):
    OUT.mkdir(exist_ok=True)
    slug = res["name"].replace(" ", "_").replace("/", "_")
    data = {
        "res": res,
        "t": [str(t) for t in b.index], "close": b.close.tolist(),
        "trades": [{**x, "t": str(x["t"])} for x in paper.trades],
        "brain": ({k: log[k].round(1).tolist() for k in ("mn9", "gf", "sugar", "bitter", "loom", "valence")}
                  if len(log) else {}),
        "brain_t": [str(t) for t in log.index] if len(log) else [],
        "wallet_roi": w.roi.clip(-1, 5).round(4).tolist(),
    }
    tpl = (Path(__file__).parent / "report.html").read_text()
    (OUT / f"{slug}.html").write_text(tpl.replace("/*DATA*/null", json.dumps(data)))
    (OUT / f"{slug}.json").write_text(json.dumps(res, indent=2))
    print(f"report: {OUT / (slug + '.html')}")


def cmd_backtest(a):
    from .helius import fetch_swaps
    end = a.end or int(time.time()) - 120
    start = end - int(a.hours * 3600)
    fetch_from = start - int(a.train_hours * 3600)
    print(f"fetching swaps for {a.mint} via Helius ({a.train_hours}h train + {a.hours}h test)...")
    swaps = fetch_swaps(a.mint, fetch_from, end)
    if len(swaps) < 50:
        raise SystemExit(f"only {len(swaps)} swaps found; pick a busier token or a longer window")
    fly = Fly(Circuits.load(), seed=a.seed, taste=a.taste, lesion=a.lesion, fee=a.fee)
    res = evaluate(a.name or a.mint[:8], fly, market.swaps_df(swaps), start if a.train_hours else None,
                   a.verbose, fee=a.fee)
    show(res)


def cmd_study(a):
    from .helius import fetch_swaps
    from .study import study
    end = a.end or (int(time.time()) // 3600) * 3600
    print(f"fetching {a.hours}h of swaps for {a.mint} via Helius...")
    df = market.swaps_df(fetch_swaps(a.mint, end - int(a.hours * 3600), end, verbose=False, chunk_s=a.chunk))
    print(f"{len(df)} swaps; walk-forward: {a.burn_in}h burn-in, then {a.window}h windows")
    variants = [(t, "none") for t in a.tastes.split(",")]
    if a.lesions:
        variants += [(a.tastes.split(",")[0], l) for l in LESIONS if l != "none"]
    s = study(a.name or a.mint[:8], df, a.fee, a.window, a.burn_in, variants, list(range(a.seeds)))
    pd.set_option("display.width", 200)
    print(s.groupby(["taste", "lesion"]).agg(trades=("trades", "mean"), roi_per_trade=("roi_per_trade", "mean"),
          beats_humans=("median_beats_humans", "mean"), beats_monkeys=("median_beats_monkeys", "mean"),
          windows_positive=("pct_windows_fly_positive", "mean")).round(4).to_string())


def cmd_demo(a):
    from .synthetic import make_swaps
    print("SYNTHETIC market (demo only, not real data)")
    df = market.swaps_df(make_swaps(a.hours * 2, seed=a.seed))
    fly = Fly(Circuits.load(), seed=a.seed, taste=a.taste, lesion=a.lesion)
    mid = int(df.ts.iloc[0] + (df.ts.iloc[-1] - df.ts.iloc[0]) / 2)
    show(evaluate("synthetic demo", fly, df, mid, a.verbose))


def cmd_live(a):
    from .helius import poll_new
    fly = Fly(Circuits.load(), seed=a.seed, taste=a.taste, fee=a.fee)
    ses = trader.Session(fly, trader.Paper(size=a.size, fee=a.fee))
    since = int(time.time()) - 3 * 3600
    print(f"🪰 live paper trading {a.mint}; warming up on the last 3h of swaps (learning, no orders)...")
    seen: set[str] = set()
    swaps = poll_new(a.mint, seen, since)
    df = market.swaps_df(swaps)
    b, f = market.bars(df), market.features(market.bars(df))
    trader.run(fly, b.iloc[:-1], f.iloc[:-1], paper=trader.Paper(fee=a.fee))  # learn from recent history
    state_file = OUT / f"live_{a.mint[:8]}.json"
    OUT.mkdir(exist_ok=True)
    last_bar = b.index[-2] if len(b) > 1 else None
    print("warm-up done; trading live (orders fill at the latest swap price)\n")
    while True:
        time.sleep(a.poll)
        since = max(since, int(time.time()) - 300)  # re-scan the last 5 min; `seen` drops repeats
        swaps = [s for s in swaps + poll_new(a.mint, seen, since) if s.ts >= int(time.time()) - 3 * 3600]
        if len(swaps) < 20:
            continue
        df = market.swaps_df(swaps)
        b = market.bars(df)
        f = market.features(b)
        closed = b.index[-2] if len(b) > 1 else None  # last fully closed minute
        if closed is None or closed == last_bar:
            continue
        last_bar = closed
        px = float(df.price.iloc[-1])
        th = ses.decide(closed, b.close.loc[closed], f.loc[closed].to_dict())
        ses.fill(pd.Timestamp.now(tz="UTC").tz_localize(None), px)
        p = ses.paper
        status = (f"{closed:%H:%M} ${px:.4g} | sugar {th.senses.sugar_hz:3.0f} bitter {th.senses.bitter_hz:3.0f} "
                  f"loom {th.senses.loom_hz:3.0f} | MN9 {th.mn9_hz:3.0f} GF {th.gf_hz:3.0f} smell {th.valence:+.2f} "
                  f"| {th.action:4s} {th.reason} | PnL ${p.equity(px):+.2f}")
        print(status, flush=True)
        state_file.write_text(json.dumps({
            "mint": a.mint, "pnl_usd": p.equity(px), "spent": p.spent, "in_position": p.in_position,
            "last": status, "trades": [{**x, "t": str(x["t"])} for x in p.trades]}, indent=2))


def main():
    ap = argparse.ArgumentParser(prog="flytrader")
    sub = ap.add_subparsers(dest="cmd", required=True)
    bt = sub.add_parser("backtest", help="replay real Helius swap history")
    bt.add_argument("--mint", required=True)
    bt.add_argument("--name")
    bt.add_argument("--hours", type=float, default=6)
    bt.add_argument("--train-hours", type=float, default=0, help="let the fly learn on the preceding N hours first")
    bt.add_argument("--end", type=int, help="unix end time (default: now)")
    bt.add_argument("--fee", type=float, default=0.0125,
                    help="cost per side incl. slippage (deep pools ~0.003, pump.fun curve ~0.015)")
    st = sub.add_parser("study", help="walk-forward over many windows, with lesion controls")
    st.add_argument("--mint", required=True)
    st.add_argument("--name")
    st.add_argument("--hours", type=float, default=24)
    st.add_argument("--end", type=int)
    st.add_argument("--window", type=float, default=2, help="hours per scored window")
    st.add_argument("--burn-in", type=float, default=6, help="hours of learning before scoring starts")
    st.add_argument("--fee", type=float, default=0.0125)
    st.add_argument("--tastes", default="auto,momentum,dip")
    st.add_argument("--lesions", action="store_true", help="also run lesioned / shuffled-wiring flies")
    st.add_argument("--seeds", type=int, default=1)
    st.add_argument("--chunk", type=int, default=120, help="seconds per parallel Helius fetch slice")
    lv = sub.add_parser("live", help="paper trade in real time")
    lv.add_argument("--mint", required=True)
    lv.add_argument("--size", type=float, default=100.0)
    lv.add_argument("--poll", type=float, default=10)
    lv.add_argument("--fee", type=float, default=0.0125)
    dm = sub.add_parser("demo", help="synthetic market, no API key")
    dm.add_argument("--hours", type=float, default=6)
    for p in (bt, lv, dm):
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--taste", default="auto", choices=["auto", "momentum", "dip"])
    for p in (bt, dm):
        p.add_argument("--lesion", default="none", choices=LESIONS)
        p.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    {"backtest": cmd_backtest, "live": cmd_live, "demo": cmd_demo, "study": cmd_study}[a.cmd](a)


if __name__ == "__main__":
    main()
