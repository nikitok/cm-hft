#!/usr/bin/env bash
# Record 8 x 1h sessions of Bybit BTC+ETH market data.
# Each session produces timestamped files:
#   testdata/bybit_btcusdt_2026-02-26_22:00.jsonl.gz
#   testdata/bybit_ethusdt_2026-02-26_22:00.jsonl.gz
#
# Usage: ./scripts/record-series.sh [sessions] [duration]
#   sessions: number of recording sessions (default: 8)
#   duration: duration per session (default: 1h)

set -euo pipefail
cd "$(dirname "$0")/.."

SESSIONS="${1:-8}"
DURATION="${2:-1h}"
SYMBOLS="BTCUSDT,ETHUSDT"
OUTPUT="testdata"

echo "═══════════════════════════════════════════════════"
echo "  Recording $SESSIONS x $DURATION sessions"
echo "  Symbols: $SYMBOLS"
echo "  Output:  $OUTPUT/"
echo "═══════════════════════════════════════════════════"
echo

for i in $(seq 1 "$SESSIONS"); do
    echo ">>> Session $i/$SESSIONS starting at $(date '+%Y-%m-%d %H:%M:%S')"
    cargo run --release --bin cm-record -- \
        --symbols "$SYMBOLS" \
        --duration "$DURATION" \
        --output "$OUTPUT" \
        --timestamp
    echo "<<< Session $i/$SESSIONS complete at $(date '+%Y-%m-%d %H:%M:%S')"
    echo
done

echo "All $SESSIONS sessions complete."
echo "Files:"
ls -lh "$OUTPUT"/bybit_*_????-??-??_??:??.jsonl.gz 2>/dev/null || echo "  (none found)"
