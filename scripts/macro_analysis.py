#!/usr/bin/env python3
"""
Step 2: Make needed transformations on the gold FRED monthly sheet.

Reads:  data/gold/fred/unified_monthly.parquet
Writes: data/gold/fred/unified_monthly_enriched.parquet and .csv

Adds columns (YoY as fractions unless noted):
- cpi_yoy                = CPI / CPI_12m_ago - 1
- real_mortgage          = mortgage_rate - (cpi_yoy * 100)  [percent points]
- permits_yoy            = building_permits / building_permits_12m_ago - 1
- starts_yoy             = housing_starts / housing_starts_12m_ago - 1
- natl_price_yoy         = price / price_12m_ago - 1 (prefers monthly existing-home price)
- income_yoy (optional)  = income_median / income_median_12m_ago - 1
"""

from pathlib import Path
import shutil
import sys
import polars as pl


GOLD_DIR = Path("data/gold/fred")
SRC = GOLD_DIR / "unified_monthly.parquet"
OUT_PQ = GOLD_DIR / "unified_monthly_enriched.parquet"
OUT_CSV = GOLD_DIR / "unified_monthly_enriched.csv"


def copy_fred_unified():
    src = Path("data/silver/fred/unified_monthly.parquet")
    dst_dir = Path("data/gold/fred")
    if not src.exists():
        print(f"Source not found: {src}")
        sys.exit(1)

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")


def yoy(expr: pl.Expr) -> pl.Expr:
    return (expr / expr.shift(12) - 1)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Source not found: {SRC}")

    df = pl.read_parquet(str(SRC))

    # Ensure datetime and sort
    if df["date"].dtype == pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Datetime))
    df = df.sort("date")

    cols = set(df.columns)

    # Choose price column preference
    price_col = None
    if "median_sales_price_existing" in cols:
        price_col = "median_sales_price_existing"
    elif "median_sales_price" in cols:
        price_col = "median_sales_price"

    new_cols = []

    # CPI YoY (fraction) and real mortgage (pp)
    if "cpi" in cols:
        cpi_yoy_expr = yoy(pl.col("cpi")).alias("cpi_yoy")
        new_cols.append(cpi_yoy_expr)
        if "mortgage_rate" in cols:
            # real mortgage in percent points: nominal % - (cpi_yoy * 100)
            new_cols.append((pl.col("mortgage_rate") - (yoy(pl.col("cpi")) * 100)).alias("real_mortgage"))

    # Permits/Starts YoY
    if "building_permits" in cols:
        new_cols.append(yoy(pl.col("building_permits")).alias("permits_yoy"))
    if "housing_starts" in cols:
        new_cols.append(yoy(pl.col("housing_starts")).alias("starts_yoy"))

    # National price YoY
    if price_col is not None:
        new_cols.append(yoy(pl.col(price_col)).alias("natl_price_yoy"))

    # Income YoY (optional)
    if "income_median" in cols:
        new_cols.append(yoy(pl.col("income_median")).alias("income_yoy"))

    if new_cols:
        df = df.with_columns(new_cols)

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(OUT_PQ))
    df.write_csv(str(OUT_CSV))
    print(f"Wrote: {OUT_PQ}")
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    copy_fred_unified()
    main()