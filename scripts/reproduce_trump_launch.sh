#!/bin/sh
# Reproduces the published $TRUMP launch-weekend numbers and video.
#
#   needs: HELIUS_API_KEY in .env. Downloads ~3.7M transactions with getTransactionsForAddress
#   (~370k credits, ~12 min), then ~15 min of simulation. For the video: ffmpeg,
#   `pip install -e ".[video]"` and `playwright install chromium`.
#
#   window: fly learns on 2025-01-19 04:00-08:00 UTC, then is scored on the next 24h
#   fees:   1% per side (launch-day pool fees + slippage)
set -e
cd "$(dirname "$0")/.."
MINT=6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN
LEARN=1737259200   # 2025-01-19 04:00 UTC
START=1737273600   # 2025-01-19 08:00 UTC
END=1737360000     # 2025-01-20 08:00 UTC

# 1) the headline: one fly over the full 24h, vs every human wallet that traded it
python viz/export_episode.py --mint $MINT --symbol '$TRUMP' --train-from $LEARN --start $START --end $END \
  --fee 0.01 --taste auto --name trump_launch

# 2) robustness: 4h windows x 3 seeds, with lesion controls
python -m flytrader study --mint $MINT --name TRUMP_launch --hours 28 --end $END \
  --fee 0.01 --window 4 --burn-in 4 --tastes auto --lesions --seeds 3

# 3) the video: buy at $65, giant-fiber exit at $75, MELANIA crash, 2025-01-19 19:40-21:32 UTC
#    (needs the viewer server running:
#    python -m http.server 8765 --directory viz)
if command -v ffmpeg >/dev/null && curl -s localhost:8765 >/dev/null; then
  python viz/clip_stats.py --episode trump_launch --mint $MINT --from-bar 700 --to-bar 812
  python viz/record.py --episode trump_launch --from-bar 700 --bars 113 --fpb 4 --slow 772-790:10,791-812:6 --hold 4 --out out/trump_launch.mp4
fi
