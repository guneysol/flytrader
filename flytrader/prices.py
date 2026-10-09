"""SOL/USD 1-minute candles, so SOL-quoted and USDC-quoted swaps share one price."""

from __future__ import annotations

import json

import numpy as np
import requests

from . import paths

CACHE = paths.DATA / "solusd"


def sol_usd(start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """(minute timestamps, close prices) covering start..end."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{start // 60}_{end // 60}.json"
    if f.exists():
        d = json.loads(f.read_text())
        return np.array(d["t"]), np.array(d["p"])
    ts, px = [], []
    cur = start - 120
    while cur < end + 60:
        try:
            r = requests.get("https://api.binance.com/api/v3/klines", timeout=20, params={
                "symbol": "SOLUSDT", "interval": "1m", "startTime": cur * 1000, "limit": 1000})
            r.raise_for_status()
            rows = [(k[0] // 1000, float(k[4])) for k in r.json()]
        except requests.RequestException:
            r = requests.get("https://api.exchange.coinbase.com/products/SOL-USD/candles", timeout=20,
                             params={"granularity": 60, "start": cur, "end": min(cur + 300 * 60, end + 60)})
            r.raise_for_status()
            rows = sorted((int(k[0]), float(k[4])) for k in r.json())
        if not rows:
            break
        ts += [t for t, _ in rows]
        px += [p for _, p in rows]
        cur = rows[-1][0] + 60
    f.write_text(json.dumps({"t": ts, "p": px}))
    return np.array(ts), np.array(px)
