#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETL for Zillow Viewer + Macros (FRED).

- One-time snapshots the Hugging Face Parquet branch locally, then reads parquet from disk
  to avoid HTTP 429 rate limits.
- Computes: YoY growth (home & rent), price cut ratio, rent-to-price ratio, etc.
- Optionally merges FRED (30Y mortgage rate, CPI).
- Robust to small schema variations (Title-case vs lower-case).

Usage:
  python src/etl_zillow.py --out data/processed/merged.parquet --start 2015-01 --end 2025-12

Env (optional):
  FRED_API_KEY=xxxxx
  HF_HUB_ENABLE_HF_TRANSFER=1   # speeds up snapshot on first run
  ZILLOW_CACHE_DIR=./.cache/zillow-viewer  # where to put local snapshot
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List

import polars as pl

# --- Optional deps
try:
    from fredapi import Fred  # type: ignore
except Exception:
    Fred = None  # type: ignore

try:
    from huggingface_hub import snapshot_download  # type: ignore
except Exception:
    snapshot_download = None  # type: ignore

HF_DATASET = "misikoff/zillow-viewer"
HF_PARQUET_REV = "refs/convert/parquet"  # aka @~parquet on the web UI

# Configs we use
CONFIG_HOME = "home_values"         # ZHVI (home values)
CONFIG_RENT = "rentals"             # ZORI (rental values)
CONFIG_SALE = "for_sale_listings"   # median list price, inventory
CONFIG_DOM  = "days_on_market"      # DOM, price-cut shares


@dataclass
class Config:
    start: Optional[str] = None   # 'YYYY-MM'
    end: Optional[str] = None     # 'YYYY-MM'
    out: str = "merged.parquet"
    add_fred: bool = True
    fred_api_key: Optional[str] = None
    fred_series: Dict[str, str] = None  # {"mortgage_rate":"MORTGAGE30US", "cpi":"CPIAUCSL"}
    region_priority: List[str] = None   # columns to use as region key
    cache_dir: str = os.getenv("ZILLOW_CACHE_DIR", os.path.join(".cache", "zillow-viewer"))

    def __post_init__(self):
        if self.fred_series is None:
            self.fred_series = {"mortgage_rate": "MORTGAGE30US", "cpi": "CPIAUCSL"}
        if self.region_priority is None:
            # include Title-case variants & common ids
            self.region_priority = [
                "region", "Region", "state", "State", "Region ID", "region_id", "Region Name", "region_name"
            ]


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Zillow Viewer ETL (Polars)")
    p.add_argument("--start", type=str, default=None, help="Filter start month YYYY-MM (inclusive)")
    p.add_argument("--end", type=str, default=None, help="Filter end month YYYY-MM (inclusive)")
    p.add_argument("--out", type=str, default="data/processed/merged.parquet", help="Output parquet path")
    p.add_argument("--no-fred", action="store_true", help="Disable FRED merge even if API key exists")
    args = p.parse_args()

    return Config(
        start=args.start,
        end=args.end,
        out=args.out,
        add_fred=not args.no_fred,
        fred_api_key=os.getenv("FRED_API_KEY"),
    )


# ---------- Utilities ----------
def _month_floor(expr: pl.Expr) -> pl.Expr:
    return expr.dt.truncate("1mo")


def _coalesce_first(df: pl.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pick_region_col(df: pl.DataFrame, priority: List[str]) -> str:
    c = _coalesce_first(df, priority)
    if c:
        return c
    # fallback: heuristic
    for col in df.columns:
        lc = col.lower()
        if any(k in lc for k in ("region", "state", "county", "metro", "msa", "cbsa")):
            return col
    raise ValueError(f"No region-identifying column found in columns: {df.columns}")


def _ensure_numeric(df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
    casts = [pl.col(c).cast(pl.Float64) for c in cols if c in df.columns]
    return df.with_columns(casts) if casts else df


def _yoy(df: pl.DataFrame, value_col: str, group_col: str) -> pl.DataFrame:
    return (
        df.sort(["date"])
        .group_by(group_col, maintain_order=True)
        .agg([pl.col(value_col), pl.col("date")])
        .explode([value_col, "date"])
        .with_columns((pl.col(value_col) / pl.col(value_col).shift(12) - 1).alias(f"{value_col}_yoy"))
    )


def _apply_date_filter(df: pl.DataFrame, start: Optional[str], end: Optional[str]) -> pl.DataFrame:
    if start:
        start_dt = datetime.strptime(start, "%Y-%m")
        df = df.filter(pl.col("date") >= pl.lit(start_dt))
    if end:
        end_dt = datetime.strptime(end, "%Y-%m")
        df = df.filter(pl.col("date") <= pl.lit(end_dt))
    return df


# ---------- Snapshot + load ----------
def _ensure_snapshot(cfg: Config) -> str:
    """
    Downloads (or reuses) a local snapshot of the Parquet branch.
    Returns the local path to the snapshot root.
    """
    if snapshot_download is None:
        logging.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(2)

    os.makedirs(cfg.cache_dir, exist_ok=True)
    logging.info("Ensuring local Parquet snapshot (first run may take a minute)...")
    # Download only the Parquet branch; reuse cache if already present.
    local_dir = snapshot_download(
        repo_id=HF_DATASET,
        revision=HF_PARQUET_REV,
        cache_dir=cfg.cache_dir,
        local_files_only=False,
        allow_patterns=[
            f"{CONFIG_HOME}/train/*.parquet",
            f"{CONFIG_RENT}/train/*.parquet",
            f"{CONFIG_SALE}/train/*.parquet",
            f"{CONFIG_DOM}/train/*.parquet",
        ],
    )
    return local_dir


def _glob_local(snapshot_root: str, config: str) -> str:
    # Local path to shards
    return os.path.join(snapshot_root, config, "train", "*.parquet")


def _collect_normalized(path_glob: str, name: str) -> Optional[pl.DataFrame]:
    try:
        lf = pl.scan_parquet(path_glob)
        df = lf.collect()
        # Date column is Title-case in HF (e.g., 'Date'); include both
        dcol = _coalesce_first(df, ["date", "Date", "period"])
        if dcol is None:
            raise ValueError(f"[{name}] no date column among ['date','Date','period']")
        df = (
            df.with_columns(pl.col(dcol).cast(pl.Datetime).alias("_date"))
              .with_columns(_month_floor(pl.col("_date")).alias("date"))
              .drop([dcol, "_date"])
        )
        return df
    except Exception as e:
        logging.warning(f"Failed to load {name}: {e}")
        return None


def load_zillow_tables(cfg: Config) -> Dict[str, pl.DataFrame]:
    root = _ensure_snapshot(cfg)

    home = _collect_normalized(_glob_local(root, CONFIG_HOME), CONFIG_HOME)
    rent = _collect_normalized(_glob_local(root, CONFIG_RENT), CONFIG_RENT)
    sale = _collect_normalized(_glob_local(root, CONFIG_SALE), CONFIG_SALE)
    dom  = _collect_normalized(_glob_local(root, CONFIG_DOM),  CONFIG_DOM)

    out = {}
    if home is not None: out["home_values"] = _apply_date_filter(home, cfg.start, cfg.end)
    if rent is not None: out["rental_values"] = _apply_date_filter(rent, cfg.start, cfg.end)
    if sale is not None: out["for_sale_listings"] = _apply_date_filter(sale, cfg.start, cfg.end)
    if dom  is not None: out["days_on_market"]    = _apply_date_filter(dom,  cfg.start, cfg.end)
    return out


# ---------- Feature engineering ----------
def derive_features(tables: Dict[str, pl.DataFrame], cfg: Config) -> pl.DataFrame:
    home = tables.get("home_values")
    rent = tables.get("rental_values")
    if home is None or rent is None:
        raise ValueError("home_values and rental_values are required (check snapshot).")

    region_home = _pick_region_col(home, cfg.region_priority)
    region_rent = _pick_region_col(rent, cfg.region_priority)

    # Value column heuristics
    home_val_col = _coalesce_first(home, ["value", "home_value", "zhvi", "median_value", "zhvi_median", "ZHVI"])
    rent_val_col = _coalesce_first(rent, ["value", "rent_value", "zori", "median_rent", "zori_median", "ZORI"])

    if home_val_col is None or rent_val_col is None:
        # As a last resort, pick any single float column as value
        float_cols_home = [c for c, dt in zip(home.columns, home.dtypes) if dt.is_float()]
        float_cols_rent = [c for c, dt in zip(rent.columns, rent.dtypes) if dt.is_float()]
        home_val_col = home_val_col or (float_cols_home[0] if float_cols_home else None)
        rent_val_col = rent_val_col or (float_cols_rent[0] if float_cols_rent else None)
        if home_val_col is None or rent_val_col is None:
            raise ValueError("Could not identify value columns in home_values/rental_values.")

    home_base = (
        home.select([region_home, "date", home_val_col])
            .pipe(_ensure_numeric, [home_val_col])
            .rename({region_home: "region_key", home_val_col: "home_value"})
    )
    rent_base = (
        rent.select([region_rent, "date", rent_val_col])
            .pipe(_ensure_numeric, [rent_val_col])
            .rename({region_rent: "region_key", rent_val_col: "rent_value"})
    )

    home_yoy = (
        home_base.select(["region_key", "date", "home_value"])
                 .pipe(_yoy, "home_value", "region_key")
                 .select(["region_key", "date", "home_value_yoy"])
                 .rename({"home_value_yoy": "home_yoy"})
    )
    rent_yoy = (
        rent_base.select(["region_key", "date", "rent_value"])
                 .pipe(_yoy, "rent_value", "region_key")
                 .select(["region_key", "date", "rent_value_yoy"])
                 .rename({"rent_value_yoy": "rent_yoy"})
    )

    merged = (
        home_base.join(rent_base, on=["region_key", "date"], how="outer")
                 .join(home_yoy, on=["region_key", "date"], how="left")
                 .join(rent_yoy, on=["region_key", "date"], how="left")
                 .with_columns(((pl.col("rent_value") * 12) / pl.col("home_value")).alias("rent_to_price_ratio"))
    )

    # For-sale listings (inventory / list price)
    sale = tables.get("for_sale_listings")
    if sale is not None and sale.height > 0:
        region_sale = _pick_region_col(sale, cfg.region_priority)
        num_listings_col = _coalesce_first(sale, ["num_of_listings", "listings", "active_listings", "inventory"])
        median_list_price = _coalesce_first(sale, ["median_list_price", "list_price_median", "list_price"])

        agg_exprs = []
        if median_list_price:
            agg_exprs += [pl.median(pl.col(median_list_price)).alias("median_list_price")]
        if num_listings_col:
            agg_exprs += [pl.sum(pl.col(num_listings_col)).alias("active_listings")]

        if agg_exprs:
            sale_agg = (
                sale.group_by([region_sale, "date"])
                    .agg(agg_exprs)
                    .rename({region_sale: "region_key"})
            )
            merged = merged.join(sale_agg, on=["region_key", "date"], how="left")

    # Days on market (DOM) & price-cut share
    dom = tables.get("days_on_market")
    if dom is not None and dom.height > 0:
        region_dom = _pick_region_col(dom, cfg.region_priority)
        dom_col = _coalesce_first(dom, ["median_days_on_market", "dom", "days_on_market", "median_dom",
                                        "Median Days on Market", "Median Days on Pending"])
        price_cut_share_col = _coalesce_first(dom, [
            "percent_listings_price_cut_smoothed",
            "percent_listings_price_cut",
            "price_cut_share",
            "pct_price_cut",
            "pct_listings_price_cut",
            "Percent Listings Price Cut (Smoothed)",  # Title-case variant on HF
            "Percent Listings Price Cut",
        ])

        agg_exprs = []
        if dom_col:
            agg_exprs += [pl.median(pl.col(dom_col)).alias("median_dom")]
        if price_cut_share_col:
            agg_exprs += [pl.mean(pl.col(price_cut_share_col)).alias("price_cut_ratio")]

        if agg_exprs:
            dom_agg = (
                dom.group_by([region_dom, "date"])
                   .agg(agg_exprs)
                   .rename({region_dom: "region_key"})
            )
            merged = merged.join(dom_agg, on=["region_key", "date"], how="left")

    return merged


# ---------- FRED ----------
def _fred_series_to_pl(name: str, series, monthly: bool = True) -> pl.DataFrame:
    import pandas as pd
    s = series.dropna()
    if monthly:
        s = s.resample("MS").mean()
    df = pl.from_pandas(pd.DataFrame({"date": s.index, name: s.values}))
    return df.with_columns(pl.col("date").cast(pl.Datetime), pl.col(name).cast(pl.Float64))


def merge_fred(merged: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    if not cfg.add_fred:
        logging.info("Skipping FRED merge: disabled via --no-fred.")
        return merged
    if Fred is None:
        logging.warning("fredapi not installed; skipping FRED merge.")
        return merged
    api_key = cfg.fred_api_key or os.getenv("FRED_API_KEY")
    if not api_key:
        logging.warning("FRED_API_KEY not provided; skipping FRED merge.")
        return merged

    fred = Fred(api_key=api_key)
    frames = []
    for name, sid in cfg.fred_series.items():
        try:
            series = fred.get_series(sid)
            fdf = _fred_series_to_pl(name, series)
            frames.append(fdf)
            logging.info(f"Fetched FRED series {sid} as {name} ({len(fdf)} rows).")
        except Exception as e:
            logging.warning(f"Failed to fetch FRED series {sid}: {e}")

    if not frames:
        return merged

    fred_df = frames[0]
    for f in frames[1:]:
        fred_df = fred_df.join(f, on="date", how="outer")

    if "cpi" in fred_df.columns:
        fred_df = fred_df.with_columns((pl.col("cpi") / pl.col("cpi").shift(12) - 1).alias("cpi_yoy"))

    merged = merged.join(fred_df, on="date", how="left")

    if "home_yoy" in merged.columns and "cpi_yoy" in merged.columns:
        merged = merged.with_columns((pl.col("home_yoy") - pl.col("cpi_yoy")).alias("home_yoy_real"))

    return merged


# ---------- Output ----------
def finalize_and_write(df: pl.DataFrame, cfg: Config) -> None:
    order_cols = [c for c in [
        "region_key", "date", "home_value", "rent_value",
        "home_yoy", "rent_yoy", "home_yoy_real",
        "price_cut_ratio", "median_dom", "median_list_price",
        "rent_to_price_ratio", "active_listings",
        "mortgage_rate", "cpi", "cpi_yoy"
    ] if c in df.columns]

    df = df.select(order_cols + [c for c in df.columns if c not in order_cols])

    out_parquet = cfg.out
    out_csv = os.path.splitext(out_parquet)[0] + ".csv"
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)

    logging.info(f"Writing parquet -> {out_parquet}")
    df.write_parquet(out_parquet)
    logging.info(f"Writing csv -> {out_csv}")
    df.write_csv(out_csv)


def main():
    # _setup_logging()
    # cfg = _parse_args()
    # logging.info(f"Config: {cfg}")

    # tables = load_zillow_tables(cfg)
    # if not tables:
    #     logging.error("No Zillow tables loaded (check snapshot).")
    #     sys.exit(2)

    # merged = derive_features(tables, cfg)
    # merged = merge_fred(merged, cfg)
    # finalize_and_write(merged, cfg)

    # logging.info("ETL complete.")
    # logging.info(f"Rows: {merged.height:,} | Columns: {merged.width}")
    # logging.info(f"Columns: {merged.columns}")
    from datasets import load_dataset

    ds = load_dataset("misikoff/zillow-viewer", "days_on_market")


if __name__ == "__main__":
    main()
