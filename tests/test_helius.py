from flytrader.helius import STABLES, WSOL, parse_swap

MINT = "TokenMint1111111111111111111111111111111111"
USDC = next(iter(sorted(STABLES)))
PAYER = "Payer11111111111111111111111111111111111111"
POOL = "Pool111111111111111111111111111111111111111"


def tx(changes, sol_change=0, fee=5000, err=None):
    """A getTransactionsForAddress item where the fee payer's balances move by `changes` [(mint, raw, dec)]."""
    pre, post = [], []
    for i, (mint, raw, dec) in enumerate(changes, start=1):
        base = 10**12
        pre.append({"accountIndex": i, "mint": mint, "owner": PAYER,
                    "uiTokenAmount": {"amount": str(base), "decimals": dec}})
        post.append({"accountIndex": i, "mint": mint, "owner": PAYER,
                     "uiTokenAmount": {"amount": str(base + raw), "decimals": dec}})
    # the pool's balances move too, but they aren't the payer's and must be ignored
    pre.append({"accountIndex": 9, "mint": MINT, "owner": POOL, "uiTokenAmount": {"amount": "5", "decimals": 6}})
    post.append({"accountIndex": 9, "mint": MINT, "owner": POOL, "uiTokenAmount": {"amount": "1", "decimals": 6}})
    return {"blockTime": 1, "transaction": {"signatures": ["sig"], "message": {"accountKeys": [PAYER, POOL]}},
            "meta": {"err": err, "fee": fee, "preBalances": [10**10, 0], "postBalances": [10**10 + sol_change, 0],
                     "preTokenBalances": pre, "postTokenBalances": post}}


def test_buy_with_usdc():
    s = parse_swap(tx([(MINT, 10_000_000, 6), (USDC, -100_000_000, 6)], sol_change=-5000), MINT)
    assert s and s.side == "buy" and s.quote == "USD"
    assert s.amount == 100 and s.tokens == 10 and s.wallet == PAYER


def test_sell_for_sol_ignores_network_fee():
    s = parse_swap(tx([(MINT, -5_000_000, 6)], sol_change=1_000_000_000 - 5000), MINT)
    assert s and s.side == "sell" and s.quote == "SOL"
    assert abs(s.amount - 1.0) < 1e-12 and s.tokens == 5


def test_wrapped_sol_counts_as_sol():
    s = parse_swap(tx([(MINT, 2_000_000, 6), (WSOL, -500_000_000, 9)], sol_change=-5000), MINT)
    assert s and s.side == "buy" and s.quote == "SOL" and abs(s.amount - 0.5) < 1e-12


def test_round_trip_arbitrage_is_not_a_swap():
    # bought and sold the token inside one tx: net zero token change
    assert parse_swap(tx([(MINT, 0, 6), (USDC, 7_306, 6)]), MINT) is None


def test_transfers_and_failed_txs_are_skipped():
    assert parse_swap(tx([(MINT, 10_000_000, 6)], sol_change=-2_039_280 - 5000), MINT) is None  # rent only
    assert parse_swap(tx([(MINT, 10_000_000, 6), (USDC, -100_000_000, 6)], err={"x": 1}), MINT) is None


def test_quote_must_move_opposite_to_token():
    # received both token and USDC (e.g. a claim) or removed liquidity is not a trade
    assert parse_swap(tx([(MINT, 10_000_000, 6), (USDC, 50_000_000, 6)]), MINT) is None
