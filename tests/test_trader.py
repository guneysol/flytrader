from types import SimpleNamespace

import pandas as pd

from flytrader import trader


class ScriptedFly:
    """Stands in for the brain: buys on the first bar, sells on the third."""

    def __init__(self):
        self.calls, self.rewards = 0, []

    def think(self, f, in_position, unrealized, bar_ret):
        self.calls += 1
        action = {1: "BUY", 3: "SELL"}.get(self.calls, "HOLD" if in_position else "WAIT")
        return SimpleNamespace(action=action, reason="scripted")

    def end_trade(self, ret):
        self.rewards.append(ret)


def test_orders_fill_at_next_bar_open_with_fees():
    fly = ScriptedFly()
    ses = trader.Session(fly, trader.Paper(size=100, fee=0.01))
    bars = pd.DataFrame({"open": [1.0, 2.0, 3.0, 4.0], "close": [1.5, 2.5, 3.5, 4.5]},
                        index=pd.date_range("2025-01-01", periods=4, freq="1min"))
    for t, b in bars.iterrows():
        ses.fill(t, b.open)
        ses.decide(t, b.close, {})
    buy, sell = ses.paper.trades
    assert buy["price"] == 2.0  # decided on bar 0, filled at bar 1 open
    assert sell["price"] == 4.0  # decided on bar 2, filled at bar 3 open
    tokens = 100 * 0.99 / 2.0
    assert abs(sell["usd"] - tokens * 4.0 * 0.99) < 1e-9
    assert fly.rewards == [sell["ret"]]
