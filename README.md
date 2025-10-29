# Zillow Housing ETL (Polars + FRED)

This module builds an analysis-ready dataset from the Hugging Face *Zillow Viewer* parquet tables,
and optionally merges macro series from FRED (30Y mortgage rate & CPI).

## Quickstart

```bash
# 1) Create venv and install deps
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) (Optional) set FRED API key for macros
export FRED_API_KEY=YOUR_KEY

# 3) Run ETL
python src/etl_zillow.py --start 2015-01 --end 2025-12 --out data/processed/merged.parquet

# Outputs:
# - data/processed/merged.parquet
# - data/processed/merged.csv
```

## Notes
- If FRED API key is not set, the script will skip macro merge gracefully.
- The script tolerates small schema variations (e.g., different column names for price cuts).
- Date filters are inclusive and monthly.
