#!/usr/bin/env bash
# Record consecutive sessions of Bybit market data to gzipped JSONL.
# Resumable: on restart, recovers orphaned .jsonl from crashed sessions,
# counts valid completed files, and continues from where it left off.
#
# Usage:
#   ./scripts/record-series.sh [options]
#
# Options (positional, all optional):
#   $1  symbols   Comma-separated pairs (default: BTCUSDT,ETHUSDT)
#   $2  sessions  Total number of sessions to collect (default: 8)
#   $3  duration  Duration per session (default: 1h)
#
# Examples:
#   ./scripts/record-series.sh                                  # 8x1h BTC+ETH
#   ./scripts/record-series.sh SOLUSDT,DOGEUSDT                 # 8x1h SOL+DOGE
#   ./scripts/record-series.sh SOLUSDT,DOGEUSDT,SUIUSDT 12 1h  # 12x1h SOL+DOGE+SUI
#   ./scripts/record-series.sh BTCUSDT 4 2h                    # 4x2h BTC only
#
# Crash recovery:
#   Just re-run the same command. The script will:
#   1. Compress any orphaned .jsonl files from a crashed session
#   2. Remove any corrupted .jsonl.gz files
#   3. Count valid completed sessions
#   4. Continue recording the remaining ones

set -euo pipefail
cd "$(dirname "$0")/.."

SYMBOLS="${1:-BTCUSDT,ETHUSDT}"
SESSIONS="${2:-8}"
DURATION="${3:-1h}"
OUTPUT="testdata"

mkdir -p "$OUTPUT"

# ── Use first symbol to track session count ──
# All symbols are recorded together, so counting one is enough.
IFS=',' read -ra SYM_ARRAY <<< "$SYMBOLS"
TRACK_SYM="${SYM_ARRAY[0],,}"  # lowercase first symbol

# ── Step 1: Recover orphaned .jsonl files from crashed sessions ──
# cm-record writes to .jsonl during recording and compresses to .jsonl.gz
# on clean shutdown. If the process was killed, the .jsonl remains.
recovered=0
for f in "$OUTPUT"/bybit_*_????-??-??_??:??.jsonl; do
    [ -f "$f" ] || continue
    gz="${f}.gz"
    lines=$(wc -l < "$f" 2>/dev/null || echo 0)
    if [ "$lines" -gt 0 ]; then
        echo "  Recovering crashed session: $(basename "$f") ($lines lines)"
        gzip -f "$f"
        ((recovered++)) || true
    else
        echo "  Removing empty orphan: $(basename "$f")"
        rm -f "$f"
    fi
done
if [ "$recovered" -gt 0 ]; then
    echo "  Recovered $recovered file(s) from previous crashed session(s)."
    echo
fi

# ── Step 2: Clean up corrupted .jsonl.gz files ──
corrupted=0
for f in "$OUTPUT"/bybit_*_????-??-??_??:??.jsonl.gz; do
    [ -f "$f" ] || continue
    if ! gzip -t "$f" 2>/dev/null; then
        echo "  Removing corrupted file: $(basename "$f")"
        rm -f "$f"
        ((corrupted++)) || true
    fi
done
if [ "$corrupted" -gt 0 ]; then
    echo "  Cleaned $corrupted corrupted file(s)."
    echo
fi

# ── Step 3: Count valid completed sessions ──
completed=0
for f in "$OUTPUT"/bybit_"${TRACK_SYM}"_????-??-??_??:??.jsonl.gz; do
    [ -f "$f" ] && ((completed++)) || true
done

remaining=$((SESSIONS - completed))

if [ "$remaining" -le 0 ]; then
    echo "═══════════════════════════════════════════════════"
    echo "  All $SESSIONS sessions already recorded."
    echo "  Symbols: $SYMBOLS"
    echo "  Output:  $OUTPUT/"
    echo "═══════════════════════════════════════════════════"
    echo
    echo "Files for ${TRACK_SYM^^}:"
    ls -lh "$OUTPUT"/bybit_"${TRACK_SYM}"_????-??-??_??:??.jsonl.gz 2>/dev/null
    exit 0
fi

echo "═══════════════════════════════════════════════════"
echo "  Target:    $SESSIONS sessions x $DURATION"
echo "  Completed: $completed"
echo "  Remaining: $remaining"
echo "  Symbols:   $SYMBOLS"
echo "  Output:    $OUTPUT/"
echo "═══════════════════════════════════════════════════"
echo

# ── Step 4: Record remaining sessions ──
for i in $(seq 1 "$remaining"); do
    session_num=$((completed + i))
    echo ">>> Session $session_num/$SESSIONS starting at $(date '+%Y-%m-%d %H:%M:%S')"
    cargo run --release --bin cm-record -- \
        --symbols "$SYMBOLS" \
        --duration "$DURATION" \
        --output "$OUTPUT" \
        --timestamp
    echo "<<< Session $session_num/$SESSIONS complete at $(date '+%Y-%m-%d %H:%M:%S')"
    echo
done

echo "All $SESSIONS sessions complete."
echo "Files:"
ls -lh "$OUTPUT"/bybit_*_????-??-??_??:??.jsonl.gz 2>/dev/null || echo "  (none found)"
