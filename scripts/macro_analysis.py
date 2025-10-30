#!/usr/bin/env python3
"""Macro analysis helpers for national Zillow aggregates.

Builds a national monthly file with:
- dom_nat: national median DOM (Median Days on Pending)
- pct_price_cut_nat: national median % listings with price cut
- median_sale_price_nat: national median sale price (prefers Smoothed/SA)

Inputs (silver):
- data/silver/zillow/days_on_market/wide.parquet
- data/silver/zillow/sales/wide.parquet

Output (gold):
- data/gold/zillow/national_monthly.parquet
"""

from pathlib import Path
import sys
from typing import Optional, List
import polars as pl
import shutil



SILVER = Path("data/silver/zillow")
GOLD = Path("data/gold/zillow")


def _read_parquet_safe(path: Path) -> Optional[pl.DataFrame]:
    if not path.exists():
        return None
    try:
        return pl.read_parquet(str(path))
    except Exception:
        return None


def _prefer(cols: List[str], available: List[str]) -> Optional[str]:
    for c in cols:
        if c in available:
            return c
    return None


def _filter_state_level(df: pl.DataFrame) -> pl.DataFrame:
    # Prefer explicit region type filter if available; otherwise keep as-is
    if "Region Type Name" in df.columns:
        return df.filter(pl.col("Region Type Name").str.to_lowercase() == "state")
    return df

def _ensure_month_date(df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """Ensure a monthly Date column named 'date' of type Date at month start."""
    d = df
    if date_col not in d.columns and "Date" in d.columns:
        d = d.with_columns(pl.col("Date").alias(date_col))
    # Cast to datetime then rebuild as Date first of month
    d = d.with_columns([
        pl.col(date_col).cast(pl.Datetime).alias(date_col)
    ])
    d = d.with_columns([
        pl.date(pl.col(date_col).dt.year().cast(pl.Int32),
                pl.col(date_col).dt.month().cast(pl.Int32),
                1).alias(date_col)
    ])
    return d


def build_national_zillow() -> Path:
    GOLD.mkdir(parents=True, exist_ok=True)

    # Load days_on_market for DOM and % price cut
    dom_path = SILVER / "days_on_market" / "wide.parquet"
    df_dom = _read_parquet_safe(dom_path)
    dom_nat = None
    if df_dom is not None:
        df_dom = _filter_state_level(df_dom)
        # Ensure monthly date
        df_dom = _ensure_month_date(df_dom, "date" if "date" in df_dom.columns else "Date")
        # Candidate columns
        dom_candidates = [
            "Median Days on Pending (Smoothed)",
            "Median Days on Pending",
        ]
        cuts_candidates = [
            "Percent Listings Price Cut (Smoothed)",
            "Percent Listings Price Cut",
        ]
        dom_col = _prefer(dom_candidates, df_dom.columns)
        cuts_col = _prefer(cuts_candidates, df_dom.columns)
        selects = [c for c in ["date", dom_col, cuts_col] if c]
        if "date" in selects:
            grp = df_dom.select(selects).group_by("date").agg([
                pl.col(dom_col).median().alias("dom_nat") if dom_col else pl.lit(None).alias("dom_nat"),
                pl.col(cuts_col).median().alias("pct_price_cut_nat") if cuts_col else pl.lit(None).alias("pct_price_cut_nat"),
            ])
            dom_nat = grp

    # Load sales for median sale price
    sales_path = SILVER / "sales" / "wide.parquet"
    df_sales = _read_parquet_safe(sales_path)
    price_nat = None
    if df_sales is not None:
        df_sales = _filter_state_level(df_sales)
        df_sales = _ensure_month_date(df_sales, "date" if "date" in df_sales.columns else "Date")
        price_candidates = [
            "Median Sale Price (Smoothed) (Seasonally Adjusted)",
            "Median Sale Price (Smoothed)",
            "Median Sale Price",
        ]
        pcol = _prefer(price_candidates, df_sales.columns)
        if pcol:
            price_nat = df_sales.select(["date", pcol]).group_by("date").agg([
                pl.col(pcol).median().alias("median_sale_price_nat")
            ])

    # Join available pieces
    if dom_nat is None and price_nat is None:
        print("No national Zillow inputs found; expected days_on_market/sales in silver.")
        sys.exit(1)

    if dom_nat is not None and price_nat is not None:
        # Ensure both have Date type for join
        dom_nat = _ensure_month_date(dom_nat, "date")
        price_nat = _ensure_month_date(price_nat, "date")
        out = dom_nat.join(price_nat, on=["date"], how="full")
        # Coalesce any duplicate date columns (e.g., date_right) then drop
        if "date_right" in out.columns:
            out = out.with_columns(pl.coalesce([pl.col("date"), pl.col("date_right")]).alias("date")).drop("date_right")
    else:
        out = dom_nat if dom_nat is not None else price_nat

    out = out.sort("date")
    dst = GOLD / "national_monthly.parquet"
    out.write_parquet(str(dst))
    print(f"Saved national Zillow aggregates to {dst}")
    return dst

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


if __name__ == "__main__":
    copy_fred_unified()
    build_national_zillow()



    