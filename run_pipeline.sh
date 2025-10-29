#!/usr/bin/env bash
set -euo pipefail

OUT_PATH="data/processed/merged.parquet"
START="${1:-2015-01}"
END="${2:-2025-12}"

python src/etl_zillow.py --start "$START" --end "$END" --out "$OUT_PATH"
echo "Done. Output at $OUT_PATH and ${OUT_PATH%.parquet}.csv"
