#!/usr/bin/env bash
set -euo pipefail

# Wait for MinIO to be ready
until mc alias set local http://localhost:9100 minioadmin minioadmin 2>/dev/null; do
  echo "Waiting for MinIO..."
  sleep 2
done

# Create buckets
mc mb --ignore-existing local/market-data
mc mb --ignore-existing local/backtest-results
mc mb --ignore-existing local/model-artifacts

echo "MinIO buckets initialized."
