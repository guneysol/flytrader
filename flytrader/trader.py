"""Paper trading loop shared by backtests and live mode."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fly import Fly, Thought


@dataclass
class Paper:
    size: float = 100.0  # USD per buy
    fee: float = 0.0125  # per side: ~1% DEX/launchpad fee + slippage + priority fee
    cash: float = 0.0  # running PnL in USD
    tokens: float = 0.0
    entry_cost: float = 0.0
    spent: float = 0.0
    trades: list[dict] = field(default_factory=list)

    @property
    def in_position(self) -> bool:
        return self.tokens > 0

    def buy(self, t, price: float):
        self.tokens = self.size * (1 - self.fee) / price
        self.cash -= self.size
        self.spent += self.size
        self.entry_cost = self.size
        self.trades.append({"t": t, "side": "buy", "price": price, "usd": self.size})

    def sell(self, t, price: float, reason: str) -> float:
        got = self.tokens * price * (1 - self.fee)
        ret = got / self.entry_cost - 1
        self.cash += got
        self.trades.append({"t": t, "side": "sell", "price": price, "usd": got, "ret": ret, "reason": reason})
        self.tokens = 0.0
        return ret

    def equity(self, price: float) -> float:
        return self.cash + self.tokens * price * (1 - self.fee)


@dataclass
class Session:
    """One fly + one paper account, stepped bar by bar (backtest and live share this)."""
    fly: Fly
    paper: Paper = field(default_factory=Paper)
    pending: tuple | None = None
    last_close: float | None = None

    def fill(self, t, open_px: float):
        """Execute the order decided on the previous bar at this bar's open."""
        if not self.pending:
            return
        side, reason = self.pending
        if side == "BUY" and not self.paper.in_position:
            self.paper.buy(t, open_px)
        elif side == "SELL" and self.paper.in_position:
            self.fly.end_trade(self.paper.sell(t, open_px, reason))
        self.pending = None

    def decide(self, t, close: float, feat: dict) -> Thought:
        bar_ret = float(np.log(close / self.last_close)) if self.last_close else 0.0
        self.last_close = close
        p = self.paper
        unreal = p.tokens * close * (1 - p.fee) / p.entry_cost - 1 if p.in_position else 0.0
        th = self.fly.think(feat, p.in_position, unreal, bar_ret)
        if th.action in ("BUY", "SELL"):
            self.pending = (th.action, th.reason)
        return th

    def close_out(self, t, price: float):
        if self.paper.in_position:
            self.fly.end_trade(self.paper.sell(t, price, "end of window"))


def thought_row(t, close: float, th: Thought, paper: Paper) -> dict:
    return {"t": t, "close": close, "action": th.action, "reason": th.reason,
            "sugar": th.senses.sugar_hz, "bitter": th.senses.bitter_hz, "loom": th.senses.loom_hz,
            "mn9": th.mn9_hz, "gf": th.gf_hz, "mdn": th.mdn_hz, "valence": th.valence,
            "novelty": th.novelty, "hunger": th.hunger, "evidence": th.evidence, "kc": th.kc_active,
            "equity": paper.equity(close), "in_pos": paper.in_position}


def run(fly: Fly, bars: pd.DataFrame, feats: pd.DataFrame, paper: Paper | None = None,
        warmup: int = 15, log=None, close_at_end: bool = True) -> tuple[Paper, pd.DataFrame]:
    """Fly decides at each bar close; orders fill at the next bar's open (no lookahead)."""
    ses = Session(fly, paper or Paper())
    rows = []
    for i, (t, b) in enumerate(bars.iterrows()):
        if b.trades > 0:
            ses.fill(t, b.open)
        if i < warmup:
            ses.last_close = b.close
            continue
        th = ses.decide(t, b.close, feats.loc[t].to_dict())
        if log and th.action in ("BUY", "SELL"):
            log(t, b.close, th)
        rows.append(thought_row(t, b.close, th, ses.paper))
    if close_at_end:
        ses.close_out(bars.index[-1], bars.close.iloc[-1])
    return ses.paper, pd.DataFrame(rows).set_index("t") if rows else pd.DataFrame()


def rule_trader(bars: pd.DataFrame, feats: pd.DataFrame, kind: str, fee: float, hold: int = 15) -> float:
    """Dumb baselines under the same costs: 5-min momentum or 5-min dip-buying, fixed hold."""
    sig = feats.r5 > 0.02 if kind == "momentum" else feats.r5 < -0.02
    opens = bars.open.values
    rets, i = [], 0
    s = sig.values
    while i < len(s) - hold - 1:
        if s[i]:
            rets.append(opens[i + 1 + hold] / opens[i + 1] * (1 - fee) ** 2 - 1)
            i += hold + 1
        else:
            i += 1
    return float(np.mean(rets)) if rets else 0.0


def random_trader(bars: pd.DataFrame, n_trades: int, hold: int, seed: int, fee: float) -> float:
    """ROI of a monkey: same number of trades and average hold, random entries."""
    rng = np.random.default_rng(seed)
    opens = bars.open.values
    if n_trades == 0 or len(opens) < hold + 2:
        return 0.0
    starts = np.sort(rng.choice(len(opens) - hold - 1, size=min(n_trades, len(opens) - hold - 1), replace=False))
    rets = [(opens[s + hold] / opens[s]) * (1 - fee) ** 2 - 1 for s in starts]
    return float(np.mean(rets))
