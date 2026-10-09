"""Fake memecoin market for development/demo only (never reported as real results)."""

from __future__ import annotations

import numpy as np

from .helius import Swap


def make_swaps(hours: float = 8, seed: int = 0) -> list[Swap]:
    rng = np.random.default_rng(seed)
    t0 = 1_760_000_000
    n_min = int(hours * 60)
    # regime-switching log price: chop, pumps, slow bleeds and sudden dumps
    p = np.empty(n_min)
    lp, regime = np.log(1e-5), "chop"
    for i in range(n_min):
        if rng.random() < 0.03:
            regime = rng.choice(["chop", "pump", "bleed", "dump"], p=[0.45, 0.25, 0.2, 0.1])
        drift = {"chop": 0.0, "pump": 0.012, "bleed": -0.006, "dump": -0.05}[regime]
        lp += drift + rng.normal(0, 0.015)
        if regime == "dump" and rng.random() < 0.4:
            regime = "chop"
        p[i] = np.exp(lp)
    wallets = [f"W{i:04d}" for i in range(600)]
    style = rng.choice(["fomo", "paperhands", "diamond", "sniper"], size=len(wallets), p=[0.45, 0.25, 0.2, 0.1])
    holdings: dict[str, float] = {}
    out: list[Swap] = []
    for i in range(1, n_min):
        r5 = np.log(p[i] / p[max(i - 5, 0)])
        n = rng.poisson(8 + 60 * max(r5, 0) + 20 * max(-r5, 0))
        for _ in range(n):
            w = wallets[rng.integers(len(wallets))]
            s = style[wallets.index(w)]
            held = holdings.get(w, 0.0)
            px = p[i] * np.exp(rng.normal(0, 0.003))
            want_buy = {"fomo": r5 > 0.03, "paperhands": rng.random() < 0.5,
                        "diamond": rng.random() < 0.3, "sniper": r5 < -0.05}[s]
            want_sell = held > 0 and {"fomo": r5 < -0.08, "paperhands": rng.random() < 0.5,
                                      "diamond": rng.random() < 0.03, "sniper": r5 > 0.05}[s]
            ts = t0 + i * 60 + int(rng.integers(60))
            if want_sell:
                frac = 1.0 if rng.random() < 0.6 else 0.5
                tok = held * frac
                holdings[w] = held - tok
                out.append(Swap(ts, f"s{len(out)}", w, "sell", "USD", tok * px * 0.99 * 150, tok))
            elif want_buy or rng.random() < 0.15:
                sol = float(np.exp(rng.normal(-0.5, 1.0)))
                tok = sol * 0.99 / px
                holdings[w] = held + tok
                out.append(Swap(ts, f"s{len(out)}", w, "buy", "USD", sol * 150, tok))
    out.sort(key=lambda s: s.ts)
    return out
