"""Run the fly over a market window and export everything the 3D viewer replays.

  python viz/export_episode.py --mint <MINT> --start <unix> --end <unix> [--train-from <unix>] --name trump
  python viz/export_episode.py --demo --name demo

Writes viz/<name>.json (bars, thoughts, trades, KC patterns) and viz/<name>_spikes.bin
(per-frame spiking neuron indices for each distinct brain simulation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from flytrader import market, trader
from flytrader.circuits import Circuits
from flytrader.fly import Fly

VIZ = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mint")
    ap.add_argument("--symbol", default="TOKEN")
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--train-from", type=int)
    ap.add_argument("--fee", type=float, default=0.0125)
    ap.add_argument("--taste", default="momentum")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--name", default="episode")
    ap.add_argument("--frame-ms", type=float, default=5.0)
    ap.add_argument("--chunk", type=int, default=120)
    a = ap.parse_args()

    if a.demo:
        from flytrader.synthetic import make_swaps
        df = market.swaps_df(make_swaps(6, seed=3))
        mid = int(df.ts.iloc[0] + (df.ts.iloc[-1] - df.ts.iloc[0]) / 2)
        start = mid
        a.symbol = "SYNTH (fake data)"
    else:
        from flytrader.helius import fetch_swaps
        train_from = a.train_from or a.start
        df = market.swaps_df(fetch_swaps(a.mint, train_from, a.end, chunk_s=a.chunk))
        start = a.start

    b = market.bars(df)
    f = market.features(b)
    fly = Fly(Circuits.load(), taste=a.taste, record_ms=a.frame_ms, fee=a.fee)
    t0 = pd.to_datetime(start, unit="s")
    if (b.index < t0).sum() > 20:
        print("training...")
        trader.run(fly, b[b.index < t0], f[f.index < t0], paper=trader.Paper(fee=a.fee))
    b, f = b[b.index >= t0], f[f.index >= t0]
    df = df[df.ts >= start]

    ses = trader.Session(fly, trader.Paper(fee=a.fee))
    rows, kc_rows, trades = [], [], []
    n_tr = 0
    for i, (t, bar) in enumerate(b.iterrows()):
        if bar.trades > 0:
            ses.fill(t, bar.open)
        while n_tr < len(ses.paper.trades):
            x = ses.paper.trades[n_tr]
            trades.append({"i": i, "d": max(i - 1, 0), "side": x["side"], "price": x["price"],
                           "reason": x.get("reason", ""), "ret": x.get("ret"),
                           "pnl": (x["usd"] - ses.paper.size) if x["side"] == "sell" else 0})
            n_tr += 1
        th = ses.decide(t, bar.close, f.loc[t].to_dict())
        rows.append({"t": f"{t:%m-%d %H:%M}", "c": bar.close, "k": ",".join(map(str, th.key)),
                     "s": round(th.senses.sugar_hz), "b": round(th.senses.bitter_hz), "l": round(th.senses.loom_hz),
                     "m": round(th.mn9_hz), "g": round(th.gf_hz), "v": round(th.valence, 3), "h": round(th.hunger, 3),
                     "p": int(ses.paper.in_position), "r": th.reason})
        kc = np.flatnonzero(th.kc_pattern) if th.kc_pattern is not None else []
        kc_rows.append([int(fly.c.kc[j]) for j in kc])  # brain-wide neuron indices
    ses.close_out(b.index[-1], b.close.iloc[-1])
    if len(ses.paper.trades) > n_tr:
        x = ses.paper.trades[-1]
        trades.append({"i": len(b) - 1, "d": len(b) - 1, "side": "sell", "price": x["price"], "reason": x["reason"],
                       "ret": x["ret"], "pnl": x["usd"] - ses.paper.size})

    # spikes: concatenate every recorded sim, frame by frame
    chunks, offsets, cur = [], {}, 0
    for key, frames in fly.frames.items():
        o = [cur]
        for fr in frames:
            chunks.append(fr.astype(np.uint32))
            cur += len(fr)
            o.append(cur)
        offsets[",".join(map(str, key))] = o
    spikes = np.concatenate(chunks) if chunks else np.zeros(0, np.uint32)
    spikes.tofile(VIZ / f"{a.name}_spikes.bin")

    w = market.wallet_pnl(df, b.close.iloc[-1], exit_fee=a.fee)
    hu = w[~w.bot]
    p = ses.paper
    fly_roi = p.cash / p.spent if p.spent else 0.0
    beat = float((hu.roi < fly_roi).mean()) if len(hu) else 0.0
    ep = {
        "symbol": a.symbol,
        "subtitle": (f"FlyWire connectome · 138,639 neurons · {a.symbol} · {len(df):,} real swaps via Helius"
                     if not a.demo else "FlyWire connectome · 138,639 neurons · synthetic market (fake data)"),
        "bars": rows, "trades": trades, "kc": kc_rows, "sim_frames": offsets,
        "compare_label": "BEATS HUMANS", "compare_value": f"{beat:.0%}",
        "stats": {"fly_roi": fly_roi, "pnl": p.cash, "humans": len(hu), "beat_humans": beat,
                  "human_median_roi": float(hu.roi.median()) if len(hu) else None},
    }
    (VIZ / f"{a.name}.json").write_text(json.dumps(ep, separators=(",", ":")))
    print(f"{a.name}: {len(rows)} bars, {len(trades)} fills, {len(offsets)} brain sims, "
          f"{spikes.nbytes / 1e6:.1f} MB spikes; fly ROI {fly_roi:+.2%}, beats {beat:.0%} of {len(hu)} humans")


if __name__ == "__main__":
    main()
