"""Pull every swap of a token from Helius and turn it into trades.

Uses Helius `getTransactionsForAddress` on the token mint: up to 1,000 full
transactions per call, filtered by block time, oldest first. It works the same
for pump.fun, PumpSwap, Raydium, Meteora, Orca and Jupiter routes, because we
only look at how the fee payer's SOL, USDC/USDT and token balances moved.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

import requests
from dotenv import load_dotenv

from .paths import DATA, ENV_FILE

load_dotenv(ENV_FILE)

RPC = "https://mainnet.helius-rpc.com/"
WSOL = "So11111111111111111111111111111111111111112"
STABLES = {"EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
           "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"}  # USDT
CACHE = DATA / "swaps"


@dataclass
class Swap:
    ts: int
    sig: str
    wallet: str
    side: str  # "buy" | "sell"
    quote: str  # "SOL" | "USD": what the trader paid / received
    amount: float  # in quote units
    tokens: float


def api_key() -> str:
    key = os.environ.get("HELIUS_API_KEY")
    if not key:
        raise SystemExit(f"HELIUS_API_KEY is not set. Put it in {ENV_FILE} (see .env.example) or export it.")
    return key


def _redact(text: str) -> str:
    """Never let the API key reach logs: request URLs carry it, and errors echo URLs."""
    key = os.environ.get("HELIUS_API_KEY")
    return text.replace(key, "***") if key else text


def rpc(method: str, params: list) -> dict:
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    last = ""
    for attempt in range(6):
        try:
            r = requests.post(RPC, params={"api-key": api_key()}, json=body, timeout=120)
        except requests.RequestException as e:
            last = _redact(f"{type(e).__name__}: {e}")
            time.sleep(1 + attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:  # rate limited / transient: back off and retry
            last = f"{r.status_code}"
            time.sleep(1.5 * (attempt + 1))
            continue
        if not r.ok:  # bad key / bad request: retrying won't help
            raise RuntimeError(_redact(f"Helius request failed: {r.status_code} {r.text[:200]}"))
        out = r.json()
        if "error" in out:
            raise RuntimeError(_redact(f"Helius {method} error: {out['error']}"))
        return out["result"]
    raise RuntimeError(f"Helius request failed after retries: {last}")


def parse_swap(item: dict, mint: str) -> Swap | None:
    """Net balance change of the fee payer: +token/-quote is a buy, the reverse a sell.

    `item` is one full transaction from getTransactionsForAddress (getTransaction shape).
    Arbitrage bots that round-trip the token inside one tx net to zero and are skipped,
    as are transfers, airdrops and liquidity moves (token and quote move the same way).
    """
    meta = item.get("meta") or {}
    if meta.get("err") is not None:
        return None
    tx = item["transaction"]
    payer = tx["message"]["accountKeys"][0]  # the fee payer is always the first account
    sol = (meta["postBalances"][0] - meta["preBalances"][0] + meta.get("fee", 0)) / 1e9  # fee isn't price
    delta: dict[str, float] = defaultdict(float)
    for sign, balances in ((-1, meta.get("preTokenBalances") or []), (1, meta.get("postTokenBalances") or [])):
        for b in balances:
            if b.get("owner") == payer:
                amt = b["uiTokenAmount"]
                delta[b["mint"]] += sign * int(amt["amount"]) / 10 ** int(amt["decimals"])
    tok = delta.get(mint, 0.0)
    sol += delta.get(WSOL, 0.0)
    usd = sum(v for m, v in delta.items() if m in STABLES)
    if tok == 0:
        return None
    # pick the quote leg that moved opposite to the token
    if abs(usd) >= 0.5 and (usd > 0) != (tok > 0):
        quote, amount = "USD", abs(usd)
    elif abs(sol) >= 0.003 and (sol > 0) != (tok > 0):
        quote, amount = "SOL", abs(sol)
    else:
        return None
    side = "buy" if tok > 0 else "sell"
    return Swap(int(item["blockTime"]), tx["signatures"][0], payer, side, quote, amount, abs(tok))


def _page(mint: str, filters: dict, token: str | None) -> tuple[list[dict], str | None]:
    opts = {"transactionDetails": "full", "sortOrder": "asc", "limit": 1000, "encoding": "json",
            "maxSupportedTransactionVersion": 1, "commitment": "confirmed",
            "filters": {"status": "succeeded", **filters}}
    if token:
        opts["paginationToken"] = token
    res = rpc("getTransactionsForAddress", [mint, opts])
    return res.get("data") or [], res.get("paginationToken")


def _cache_file(mint: str, start: int, end: int):
    return CACHE / mint / f"gtfa_{start}_{end}.jsonl"


def _load_chunk(f) -> list[Swap]:
    return [Swap(**json.loads(line)) for line in f.read_text().splitlines() if line]


def _fetch_chunk(mint: str, start: int, end: int) -> list[Swap]:
    """One time slice [start, end), cached on disk so interrupted downloads resume."""
    f = _cache_file(mint, start, end)
    if f.exists():
        return _load_chunk(f)
    swaps: list[Swap] = []
    token = None
    while True:
        items, token = _page(mint, {"blockTime": {"gte": start, "lt": end}}, token)
        swaps += [s for s in (parse_swap(i, mint) for i in items) if s]
        if not token or not items:
            break
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".part")
    tmp.write_text("\n".join(json.dumps(asdict(s)) for s in swaps))
    tmp.rename(f)
    return swaps


def fetch_swaps(mint: str, start: int, end: int, verbose: bool = True,
                chunk_s: int = 120, workers: int = 32) -> list[Swap]:
    """All swaps of `mint` with start <= time < end, oldest first.

    The window is split into short time slices fetched in parallel worker processes (decoding
    ~1 MB of JSON per 1,000 transactions is CPU-bound, so threads would serialize on the GIL).
    Each slice is cached on disk.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    slices = [(t, min(t + chunk_s, end)) for t in range(start - start % chunk_s, end, chunk_s)]
    out: list[Swap] = []
    todo = []
    for a, b in slices:
        f = _cache_file(mint, a, b)
        if f.exists():
            out += _load_chunk(f)
        else:
            todo.append((a, b))
    if todo:
        with ProcessPoolExecutor(min(workers, len(todo))) as ex:
            futs = [ex.submit(_fetch_chunk, mint, a, b) for a, b in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                out += fut.result()
                if verbose:
                    print(f"  {i}/{len(todo)} slices downloaded, {len(out)} swaps", flush=True)
    seen: set[str] = set()
    uniq = [s for s in sorted(out, key=lambda s: s.ts) if not (s.sig in seen or seen.add(s.sig))]
    return [s for s in uniq if start <= s.ts < end]


def poll_new(mint: str, seen: set[str], since: int) -> list[Swap]:
    """Swaps since `since` that aren't in `seen` (for live paper trading). Updates `seen`."""
    out: list[Swap] = []
    token = None
    while True:
        items, token = _page(mint, {"blockTime": {"gte": since}}, token)
        for i in items:
            s = parse_swap(i, mint)
            if s and s.sig not in seen:
                seen.add(s.sig)
                out.append(s)
        if not token or not items:
            break
    return out
