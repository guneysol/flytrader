"""Fair comparison for a video clip: the fly vs real wallets that bought during the clip.

Both are measured on the same horizon, from the clip start to the end of the episode, with each
wallet's real sells and open bags marked at the final price. Writes the result into the episode
JSON (under "clips") for the viewer's end card.

  python viz/clip_stats.py --episode trump_launch --mint <MINT> --from-bar 700 --to-bar 812
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from flytrader import market
from flytrader.helius import fetch_swaps

VIZ = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--mint", required=True)
    ap.add_argument("--from-bar", type=int, required=True)
    ap.add_argument("--to-bar", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=120)
    a = ap.parse_args()

    path = VIZ / f"{a.episode}.json"
    ep = json.loads(path.read_text())
    t_start, t_end = pd.Timestamp(ep["start"], unit="s"), pd.Timestamp(ep["end"], unit="s")
    clip_start = t_start + pd.Timedelta(minutes=a.from_bar)
    clip_end = t_start + pd.Timedelta(minutes=a.to_bar + 1)

    df = market.swaps_df(fetch_swaps(a.mint, int(clip_start.timestamp()), int(t_end.timestamp()),
                                     verbose=False, chunk_s=a.chunk))
    final_px = market.bars(df).close.iloc[-1]
    buyers = set(df[(df.side == "buy") & (df.ts < clip_end.timestamp())].wallet)
    w = market.wallet_pnl(df, final_px, exit_fee=ep["fee"])
    w = w[w.index.isin(buyers) & ~w.bot]

    # the fly over the same hours, same metric (PnL / capital spent, fixed size per entry)
    rets = [t["ret"] for t in ep["trades"] if t["side"] == "sell" and t["i"] >= a.from_bar]
    fly = float(np.mean(rets)) if rets else 0.0

    ep.setdefault("clips", {})[f"{a.from_bar}-{a.to_bar}"] = {
        "buyers": int(len(w)), "buyers_median_roi": float(w.roi.median()),
        "buyers_profitable": float((w.pnl > 0).mean()), "fly_roi": fly, "fly_trades": len(rets),
        "fly_beats": float((w.roi < fly).mean()),
    }
    path.write_text(json.dumps(ep, separators=(",", ":")))
    print(json.dumps(ep["clips"][f"{a.from_bar}-{a.to_bar}"], indent=2))


if __name__ == "__main__":
    main()
