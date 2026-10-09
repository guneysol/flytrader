"""Swaps -> 1-minute bars -> features the fly can sense; real-wallet PnL."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .helius import Swap


def swaps_df(swaps: list[Swap], min_usd: float = 10.0) -> pd.DataFrame:
    """One row per swap, valued in USD (column `usd`, price in USD per token)."""
    df = pd.DataFrame([s.__dict__ for s in swaps])
    df["usd"] = df.amount
    if (df.quote == "SOL").any():
        from .prices import sol_usd
        t, p = sol_usd(int(df.ts.min()), int(df.ts.max()))
        i = np.clip(np.searchsorted(t, df.ts.values, side="right") - 1, 0, len(p) - 1)
        df.loc[df.quote == "SOL", "usd"] = df.amount * p[i]
    df = df[df.usd >= min_usd]
    df["price"] = df.usd / df.tokens
    # Below ~$10, account rent and tips distort the effective price; above it, drop prints more than
    # 2x away from the median of the previous fills (dust spam, routing artifacts). Trailing only:
    # no lookahead.
    med = df.price.rolling(51, min_periods=1).median().shift(1).fillna(df.price)
    df = df[(df.price / med).between(0.5, 2)].reset_index(drop=True)
    # "mid": trailing median of the last 15 fills; individual fills carry slippage, sandwiches and tips
    df["mid"] = df.price.rolling(15, min_periods=1).median()
    return df


def bars(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    d = df.assign(t=pd.to_datetime(df.ts, unit="s"), buy_usd=np.where(df.side == "buy", df.usd, 0.0),
                  sell_usd=np.where(df.side == "sell", df.usd, 0.0))
    first_seen = d.groupby("wallet").ts.transform("min")
    d["new_wallet"] = (d.ts == first_seen) & (d.side == "buy")
    g = d.set_index("t").groupby(pd.Grouper(freq=freq))
    b = pd.DataFrame({
        "open": g.mid.first(), "high": g.mid.max(), "low": g.mid.min(), "close": g.mid.last(),
        "vwap": g.usd.sum() / g.tokens.sum(),
        "volume": g.usd.sum(), "buy_usd": g.buy_usd.sum(), "sell_usd": g.sell_usd.sum(),
        "trades": g.usd.count(),
        "buyers": d[d.side == "buy"].set_index("t").groupby(pd.Grouper(freq=freq)).wallet.nunique(),
        "new_wallets": g.new_wallet.sum(), "max_sell": g.sell_usd.max(),
    })
    b["close"] = b.close.ffill()
    for c in ("open", "high", "low", "vwap"):
        b[c] = b[c].fillna(b.close)
    return b.fillna(0.0)


def features(b: pd.DataFrame) -> pd.DataFrame:
    lc = np.log(b.close)
    f = pd.DataFrame(index=b.index)
    for n in (1, 5, 15, 60):
        f[f"r{n}"] = lc.diff(n).fillna(0)
    f["drawdown"] = 1 - b.close / b.close.rolling(30, min_periods=1).max()
    tot = b.buy_usd + b.sell_usd
    f["buy_ratio"] = (b.buy_usd.rolling(3, min_periods=1).sum() / tot.rolling(3, min_periods=1).sum()).fillna(0.5)
    avg_trade = (b.volume / b.trades.replace(0, np.nan)).rolling(30, min_periods=1).mean()
    f["big_sell"] = (b.max_sell / avg_trade).fillna(0)
    f["new_wallet_frac"] = (b.new_wallets / b.trades.replace(0, np.nan)).fillna(0)
    f["volatility"] = lc.diff().rolling(10, min_periods=2).std().fillna(0)
    # distinct buyers per trade: wash/bot volume inflates trades, not wallets
    f["uniq_ratio"] = (b.buyers / b.trades.replace(0, np.nan)).fillna(0)
    recent = b.buyers.rolling(3, min_periods=1).sum()
    f["buyers_accel"] = (recent - recent.shift(3)).fillna(0)

    def z(x: pd.Series) -> pd.Series:
        m, s = x.rolling(60, min_periods=5).mean(), x.rolling(60, min_periods=5).std()
        return ((x - m) / s.replace(0, np.nan)).fillna(0).clip(-5, 5)

    f["vol_z"] = z(np.log1p(b.volume)).where(f.uniq_ratio >= 0.3, 0.0)  # ignore likely wash volume
    f["buyers_z"] = z(b.buyers)
    f["trade_size_z"] = z(np.log1p(b.volume / b.trades.replace(0, np.nan)).fillna(0))
    for c in ("r1", "r5", "r15", "r60", "drawdown", "buy_ratio", "new_wallet_frac", "big_sell", "volatility",
              "uniq_ratio", "buyers_accel"):
        f[c + "_z"] = z(f[c])
    return f


def wallet_pnl(df: pd.DataFrame, final_price: float, min_spent: float = 20.0, exit_fee: float = 0.0) -> pd.DataFrame:
    """Realized + unrealized PnL of every wallet that bought in the window, with a bot flag.

    Open bags are marked at the last price net of what it would cost to sell them.
    """
    buys, sells = df[df.side == "buy"], df[df.side == "sell"]
    srt = df.sort_values(["wallet", "ts"])
    gap = srt.ts.diff().where(srt.wallet.eq(srt.wallet.shift()))
    g = srt.groupby("wallet")
    w = pd.DataFrame({
        "spent": buys.groupby("wallet").usd.sum(), "bought": buys.groupby("wallet").tokens.sum(),
        "received": sells.groupby("wallet").usd.sum(), "sold": sells.groupby("wallet").tokens.sum(),
        "trades": g.usd.count(),
        "median_gap_s": gap.groupby(srt.wallet).median(),
        "span_h": (g.ts.max() - g.ts.min()) / 3600,
    })
    w["median_gap_s"] = w.median_gap_s.fillna(np.inf)
    w = w.fillna(0.0)
    w = w[w.spent >= min_spent]
    # sells of tokens bought before the window would look like free money: only count
    # the share of sells covered by in-window buys
    covered = np.minimum(w.sold, w.bought)
    w["received"] = w.received * (covered / w.sold.replace(0, np.nan)).fillna(0)
    w["tokens"] = w.bought - covered
    w["pnl"] = w.received + w.tokens * final_price * (1 - exit_fee) - w.spent
    w["roi"] = w.pnl / w.spent
    turnover = (w.spent + w.received) / w.spent
    per_hour = w.trades / w.span_h.clip(lower=0.25)
    # bots/MM/arb: machine-gun cadence, very high trade rate, or big churn for ~zero edge
    w["bot"] = (((w.trades >= 10) & (w.median_gap_s < 20)) | (per_hour > 40)
                | ((turnover > 4) & (w.roi.abs() < 0.01)))
    return w.sort_values("roi", ascending=False)
