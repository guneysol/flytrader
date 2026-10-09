"""Walk-forward study: one continuously learning fly, scored window by window.

The fly only ever learns from the past (online, causal), so every window after
the burn-in is out-of-sample. Each variant (taste, lesion) is a separate fly
over the same data. Results are a distribution over windows, not one lucky run.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import market, paths, trader
from .circuits import Circuits
from .fly import Fly

OUT = paths.OUT


def _run_variant(args):
    df, fee, taste, lesion, seed = args
    b = market.bars(df)
    f = market.features(b)
    fly = Fly(Circuits.load(), seed=seed, taste=taste, lesion=lesion, fee=fee)
    paper, log = trader.run(fly, b, f, paper=trader.Paper(fee=fee))
    return (taste, lesion, seed), paper.trades


def score_windows(df: pd.DataFrame, trades: list[dict], fee: float, window_h: float, burn_in_h: float) -> pd.DataFrame:
    b = market.bars(df)
    t_first = b.index[0] + pd.Timedelta(hours=burn_in_h)
    edges = pd.date_range(t_first, b.index[-1], freq=f"{int(window_h * 60)}min")
    pairs = [(bu, s) for bu, s in zip(trades[::2], trades[1::2], strict=False)]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mine = [s["ret"] for bu, s in pairs if lo <= pd.Timestamp(bu["t"]) < hi]
        fly_roi = float(np.mean(mine)) if mine else 0.0
        wdf = df[(df.ts >= lo.timestamp()) & (df.ts < hi.timestamp())]
        wb = b[(b.index >= lo) & (b.index < hi)]
        if len(wdf) < 20 or len(wb) < 10:
            continue
        w = market.wallet_pnl(wdf, wb.close.iloc[-1], exit_fee=fee)
        hu = w[~w.bot]
        monkeys = [trader.random_trader(wb, max(len(mine), 1), 10, s, fee) for s in range(100)]
        rows.append({
            "window": f"{lo:%m-%d %H:%M}", "fly_trades": len(mine), "fly_roi": fly_roi,
            "humans": len(hu), "human_median_roi": float(hu.roi.median()) if len(hu) else np.nan,
            "beats_humans": float((hu.roi < fly_roi).mean()) if len(hu) else np.nan,
            "beats_monkeys": float(np.mean(np.array(monkeys) < fly_roi)),
            "buy_hold": float(wb.close.iloc[-1] / wb.open.iloc[0] * (1 - fee) ** 2 - 1),
        })
    return pd.DataFrame(rows)


def study(name: str, df: pd.DataFrame, fee: float, window_h: float, burn_in_h: float,
          variants: list[tuple[str, str]], seeds: list[int]) -> pd.DataFrame:
    jobs = [(df, fee, taste, lesion, s) for taste, lesion in variants for s in seeds]
    summary = []
    with ProcessPoolExecutor(max_workers=min(len(jobs), 6)) as ex:
        for (taste, lesion, seed), trades in ex.map(_run_variant, jobs):
            w = score_windows(df, trades, fee, window_h, burn_in_h)
            traded = w[w.fly_trades > 0]
            summary.append({
                "taste": taste, "lesion": lesion, "seed": seed, "windows": len(w), "windows_traded": len(traded),
                "trades": int(w.fly_trades.sum()),
                "roi_per_trade": float(np.average(w.fly_roi, weights=w.fly_trades)) if w.fly_trades.sum() else 0.0,
                "median_beats_humans": float(w.beats_humans.median()),
                "beats_humans_iqr": f"{w.beats_humans.quantile(.25):.0%}–{w.beats_humans.quantile(.75):.0%}",
                "median_beats_monkeys": float(w.beats_monkeys.median()),
                "human_median_roi": float(w.human_median_roi.median()),
                "pct_windows_fly_positive": float((traded.fly_roi > 0).mean()) if len(traded) else 0.0,
            })
            OUT.mkdir(exist_ok=True)
            w.to_csv(OUT / f"study_{name}_{taste}_{lesion}_{seed}.csv", index=False)
            print(f"  done {taste}/{lesion}/seed{seed}: {summary[-1]['trades']} trades, "
                  f"ROI/trade {summary[-1]['roi_per_trade']:+.2%}, beats humans (median window) "
                  f"{summary[-1]['median_beats_humans']:.0%}", flush=True)
    s = pd.DataFrame(summary)
    (OUT / f"study_{name}.json").write_text(s.to_json(orient="records", indent=2))
    return s
