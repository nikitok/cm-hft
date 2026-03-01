#!/usr/bin/env bash
# =============================================================================
# record-and-upload.sh — Entrypoint for cm-record K8s CronJob
# =============================================================================
# 1. Configure MinIO client alias from env vars
# 2. Run cm-record to capture market data
# 3. Compress any orphaned .jsonl files (crash recovery)
# 4. Upload all .jsonl.gz to S3
# 5. Clean up local files after successful upload
# =============================================================================

set -euo pipefail

# Required env vars
: "${RECORD_SYMBOLS:?RECORD_SYMBOLS not set}"
: "${RECORD_DURATION:?RECORD_DURATION not set}"
: "${S3_ENDPOINT:?S3_ENDPOINT not set}"
: "${S3_ACCESS_KEY:?S3_ACCESS_KEY not set}"
: "${S3_SECRET_KEY:?S3_SECRET_KEY not set}"
: "${S3_BUCKET:?S3_BUCKET not set}"

RECORD_EXCHANGE="${RECORD_EXCHANGE:-bybit}"
S3_PREFIX="${S3_PREFIX:-recordings}"
DATA_DIR="/data"

echo "=== cm-record entrypoint ==="
echo "  Exchange: ${RECORD_EXCHANGE}"
echo "  Symbols:  ${RECORD_SYMBOLS}"
echo "  Duration: ${RECORD_DURATION}"
echo "  S3:       ${S3_ENDPOINT}/${S3_BUCKET}/${S3_PREFIX}/"

# ── Step 1: Configure MinIO client ──
mc alias set s3 "${S3_ENDPOINT}" "${S3_ACCESS_KEY}" "${S3_SECRET_KEY}" --api S3v4

# ── Step 2: Record market data ──
echo ">>> Starting recording at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
cm-record \
    --exchange "${RECORD_EXCHANGE}" \
    --symbols "${RECORD_SYMBOLS}" \
    --duration "${RECORD_DURATION}" \
    --output "${DATA_DIR}" \
    --timestamp \
    --hour-align
echo "<<< Recording finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# ── Step 3: Compress orphaned .jsonl files (crash recovery) ──
orphaned=0
for f in "${DATA_DIR}"/*.jsonl; do
    [ -f "$f" ] || continue
    lines=$(wc -l < "$f" 2>/dev/null || echo 0)
    if [ "$lines" -gt 0 ]; then
        echo "  Compressing orphaned: $(basename "$f") ($lines lines)"
        gzip -f "$f"
        ((orphaned++)) || true
    else
        echo "  Removing empty orphan: $(basename "$f")"
        rm -f "$f"
    fi
done
[ "$orphaned" -gt 0 ] && echo "  Compressed ${orphaned} orphaned file(s)."

# ── Step 4: Upload .jsonl.gz to S3 ──
uploaded=0
failed=0
for f in "${DATA_DIR}"/*.jsonl.gz; do
    [ -f "$f" ] || continue
    dest="s3/${S3_BUCKET}/${S3_PREFIX}/$(basename "$f")"
    echo "  Uploading: $(basename "$f") → ${dest}"
    if mc cp "$f" "${dest}"; then
        ((uploaded++)) || true
        # ── Step 5: Remove local file after successful upload ──
        rm -f "$f"
    else
        echo "  ERROR: Failed to upload $(basename "$f")"
        ((failed++)) || true
    fi
done

echo "=== Upload complete: ${uploaded} uploaded, ${failed} failed ==="

if [ "$failed" -gt 0 ]; then
    echo "ERROR: ${failed} file(s) failed to upload"
    exit 1
fi

exit 0
