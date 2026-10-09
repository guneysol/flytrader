import numpy as np
import pandas as pd

from flytrader import market
from flytrader.helius import Swap


def df_of(rows):
    return pd.DataFrame(rows, columns=["ts", "wallet", "side", "usd", "tokens"])


def test_wallet_pnl_realized_plus_marked_bag():
    d = df_of([(0, "A", "buy", 100.0, 100.0), (60, "A", "sell", 75.0, 50.0)])
    w = market.wallet_pnl(d, final_price=2.0, min_spent=1)
    assert w.loc["A", "pnl"] == 75 + 50 * 2 - 100


def test_wallet_pnl_ignores_inventory_bought_before_window():
    # sells 30 tokens but only bought 10 inside the window: only 1/3 of the sale counts
    d = df_of([(0, "B", "buy", 10.0, 10.0), (60, "B", "sell", 60.0, 30.0)])
    w = market.wallet_pnl(d, final_price=1.0, min_spent=1)
    assert w.loc["B", "pnl"] == 20 - 10


def test_open_bags_are_marked_net_of_exit_fee():
    d = df_of([(0, "C", "buy", 100.0, 100.0)])
    w = market.wallet_pnl(d, final_price=1.0, min_spent=1, exit_fee=0.01)
    assert np.isclose(w.loc["C", "pnl"], -1.0)


def test_machine_gun_wallets_are_flagged_as_bots():
    rows = [(i * 5, "bot", "buy" if i % 2 else "sell", 50.0, 50.0) for i in range(20)]
    rows += [(0, "human", "buy", 50.0, 50.0), (3600, "human", "sell", 60.0, 50.0)]
    w = market.wallet_pnl(df_of(rows), final_price=1.0, min_spent=1)
    assert w.loc["bot", "bot"] and not w.loc["human", "bot"]


def synthetic_swaps(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    px = np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    return [Swap(i * 7, f"s{i}", f"w{rng.integers(200)}", "buy" if rng.random() < 0.5 else "sell",
                 "USD", float(rng.uniform(5, 500)), 0.0) for i in range(n)], px


def test_features_never_look_ahead():
    swaps, px = synthetic_swaps()
    for s, p in zip(swaps, px, strict=True):
        s.tokens = s.amount / p
    full = market.features(market.bars(market.swaps_df(swaps)))
    cut = swaps[: len(swaps) // 2]
    part = market.features(market.bars(market.swaps_df(cut)))
    common = part.index[:-1]  # last bar of the truncated set is incomplete
    pd.testing.assert_frame_equal(full.loc[common], part.loc[common])
