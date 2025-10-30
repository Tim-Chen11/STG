#!/usr/bin/env python3
"""
Enhanced Transform Module for Zillow + FRED ETL Pipeline
Preserves all data richness with multi-level aggregation
"""

import json
import pathlib
from typing import Dict
import datetime as dt

import polars as pl


# =========================
# Configuration
# =========================
BASE_OUT = pathlib.Path("data")
BRONZE = BASE_OUT / "raw"
SILVER = BASE_OUT / "silver"

# ---- Zillow schema & label maps (from dataset card) ----
ZILLOW_BASE_COLS = {
    "days_on_market": {"Region ID","Size Rank","Region","Region Type","State","Home Type","Date"},
    "for_sale_listings": {"Region ID","Size Rank","Region","Region Type","State","Home Type","Date"},
    "home_values": {"Region ID","Size Rank","Region","Region Type","State","Home Type","Bedroom Count","Date"},
    "home_values_forecasts": {"Region ID","Size Rank","Region","Region Type","State","City","Metro","County","Home Type","Date"},
    "new_construction": {"Region ID","Size Rank","Region","Region Type","State","Home Type","Date"},
    "rentals": {"Region ID","Size Rank","Region","Region Type","State","Home Type","Date"},
    "sales": {"Region ID","Size Rank","Region","Region Type","State","Home Type","Date"},
}

# Region Type encodings differ per config (per dataset card)
ZILLOW_REGION_TYPE = {
    "days_on_market": {0:"zip",1:"city",2:"county",3:"msa",4:"state",5:"country"},
    "for_sale_listings": {0:"zip",1:"city",2:"county",3:"msa",4:"state"},
    "home_values": {0:"zip",1:"city",2:"county",3:"msa",4:"state",5:"country"},
    "home_values_forecasts": {0:"county",1:"city",2:"zip",3:"country",4:"msa"},
    "new_construction": {0:"county",1:"city",2:"zip",3:"country",4:"msa"},
    "rentals": {0:"county",1:"city",2:"zip",3:"country",4:"msa"},
    "sales": {0:"county",1:"city",2:"zip",3:"country",4:"msa"},
}

# Home Type encodings differ per config (per dataset card)
ZILLOW_HOME_TYPE = {
    # days_on_market & home_values:
    "_type_A": {0:"multifamily",1:"condo/co-op",2:"SFR",3:"all homes",4:"all homes + multifamily"},
    # for_sale_listings, new_construction, rentals, sales:
    "_type_B": {0:"all homes",1:"all homes + multifamily",2:"SFR",3:"condo/co-op",4:"multifamily"},
}
ZILLOW_HOME_TYPE_BY_CONFIG = {
    "days_on_market": ZILLOW_HOME_TYPE["_type_A"],
    "home_values": ZILLOW_HOME_TYPE["_type_A"],
    "for_sale_listings": ZILLOW_HOME_TYPE["_type_B"],
    "new_construction": ZILLOW_HOME_TYPE["_type_B"],
    "rentals": ZILLOW_HOME_TYPE["_type_B"],
    "sales": ZILLOW_HOME_TYPE["_type_B"],
    # forecasts uses free-text Home Type; no mapping needed
}

# Bedroom labels (home_values only)
ZILLOW_BEDROOM_LABELS = {
    0:"1-Bedroom", 1:"2-Bedrooms", 2:"3-Bedrooms",
    3:"4-Bedrooms", 4:"5+-Bedrooms", 5:"All Bedrooms"
}


# Removed unused SchemaManager and related helpers

def transform_zillow_dataset(config_name: str, bronze_path: pathlib.Path) -> tuple[Dict[str, pl.DataFrame], Dict]:
    """
    Zillow Transform (Enhanced but Faithful)
    - Keeps all raw columns exactly as-is.
    - Adds: standardized datetime column, readable region/home type labels, date parts.
    - Does NOT remove or rename any original field.
    - Produces both 'wide' (original) and 'long' normalized forms.
    """

    # 1. Locate parquet shards
    shards = list(bronze_path.glob("*.parquet"))
    if not shards:
        shards = list((bronze_path / "train").glob("*.parquet"))
    if not shards:
        raise ValueError(f"No parquet files found in {bronze_path}")

    # 2. Load
    df = pl.scan_parquet([str(p) for p in shards]).collect()

    # 3. Normalize dataset key
    cfg = config_name.strip().lower()
    if cfg not in ZILLOW_BASE_COLS:
        for k in ZILLOW_BASE_COLS:
            if cfg.startswith(k):
                cfg = k
                break
    if cfg not in ZILLOW_BASE_COLS:
        raise ValueError(f"Unknown Zillow config '{config_name}'")

    # 4. Standardize Date -> 'date' (Datetime)
    if "Date" in df.columns:
        dtype = df["Date"].dtype
        if dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
            df = df.with_columns(pl.from_epoch(pl.col("Date").cast(pl.Int64), unit="ms").alias("date"))
        elif dtype == pl.Utf8:
            df = df.with_columns(pl.col("Date").str.strptime(pl.Datetime, strict=False).alias("date"))
        elif dtype in (pl.Datetime, pl.Date):
            df = df.with_columns(pl.col("Date").cast(pl.Datetime).alias("date"))
        else:
            df = df.with_columns(pl.from_epoch(pl.col("Date").cast(pl.Int64), unit="ms").alias("date"))
    elif "date" not in df.columns:
        raise ValueError(f"{config_name}: missing Date column")

    # 5. Add readable region type if present
    if "Region Type" in df.columns:
        region_map = ZILLOW_REGION_TYPE.get(cfg, {})
        df = df.with_columns(
            pl.col("Region Type")
              .cast(pl.Int64, strict=False)
              .map_elements(lambda x: region_map.get(x, None), return_dtype=pl.Utf8)
              .alias("Region Type Name")
        )

    # 6. Add readable home type if present
    if "Home Type" in df.columns:
        home_map = ZILLOW_HOME_TYPE_BY_CONFIG.get(cfg)
        if home_map:
            df = df.with_columns(
                pl.col("Home Type")
                  .cast(pl.Int64, strict=False)
                  .map_elements(lambda x: home_map.get(x, None), return_dtype=pl.Utf8)
                  .alias("Home Type Name")
            )
        else:
            # leave text-based types as is
            df = df.with_columns(pl.col("Home Type").cast(pl.Utf8).alias("Home Type Name"))

    # 7. Decode Bedroom Count if available
    if "Bedroom Count" in df.columns:
        df = df.with_columns(
            pl.col("Bedroom Count")
              .cast(pl.Int64, strict=False)
              .map_elements(lambda x: ZILLOW_BEDROOM_LABELS.get(x, None), return_dtype=pl.Utf8)
              .alias("Bedroom Label")
        )

    # 8. Add time decomposition columns
    if "date" in df.columns:
        df = df.with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.quarter().alias("quarter")
        ])

    # 9. Keep all columns (wide version)
    wide = df

    # 10. Create long version for metrics
    id_cols = [c for c in df.columns if c.lower() in (
        "region id", "region", "region type", "region type name",
        "home type", "home type name", "bedroom count", "bedroom label",
        "date", "year", "month", "quarter", "size rank", "state", "city", "metro", "county"
    )]
    value_cols = [c for c in df.columns if c not in id_cols]

    long = df.melt(id_vars=id_cols, value_vars=value_cols, variable_name="metric", value_name="value")

    return {"wide": wide, "long": long}, {}



def save_silver_data(config_name: str, results: Dict[str, pl.DataFrame], 
                    metrics: Dict, silver_path: pathlib.Path):
    """Save transformed data to silver layer with metadata"""
    
    config_path = silver_path / config_name
    config_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    for level, df in results.items():
        if df is not None and df.height > 0:
            file_path = config_path / f"{level}.parquet"
            df.write_parquet(str(file_path))
            saved_files[level] = {
                "path": str(file_path),
                "rows": df.height,
                "columns": df.columns,
                "size_bytes": file_path.stat().st_size
            }
            print(f"    Saved {level}: {df.height:,} rows to {file_path.name}")
    
    # Save metadata
    metadata = {
        "dataset": config_name,
        "transformed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "levels": saved_files,
        "metrics": metrics,
        "total_metrics": len(metrics),
        "metric_categories": {}
    }
    
    # Count metrics by category
    for meta in metrics.values():
        if isinstance(meta, dict) and meta.get("category"):
            cat = meta["category"]
            if cat not in metadata["metric_categories"]:
                metadata["metric_categories"][cat] = 0
            metadata["metric_categories"][cat] += 1
    
    metadata_path = config_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return saved_files


def transform_fred_data(bronze_path: pathlib.Path, silver_path: pathlib.Path) -> Dict:
    """Transform FRED data preserving all series and frequencies"""
    
    fred_files = list(bronze_path.glob("*.parquet"))
    if not fred_files:
        fred_files = list(bronze_path.glob("*.csv"))
    
    print(f"Processing FRED data: {len(fred_files)} series found")
    
    all_series = {}
    series_metadata = {}
    
    for file_path in fred_files:
        series_name = file_path.stem
        
        # Read data
        if file_path.suffix == '.parquet':
            df = pl.read_parquet(str(file_path))
        else:
            df = pl.read_csv(str(file_path))
        
        print(f"  {series_name}: {df.height} observations")
        
        # Ensure date column
        date_col = None
        for col in df.columns:
            if col.lower() in ['date', 'observation_date']:
                date_col = col
                break
        
        if not date_col:
            continue
        
        # Rename columns
        df = df.rename({date_col: "date"})
        
        # Ensure datetime
        if df["date"].dtype == pl.Utf8:
            df = df.with_columns([
                pl.col("date").str.strptime(pl.Datetime, strict=False)
            ])
        
        # Detect frequency
        if df.height >= 2:
            date_diffs = df["date"].sort().diff().drop_nulls()
            median_diff = date_diffs.median()
            
            if median_diff is not None:
                days = median_diff.days
                if days <= 1:
                    frequency = "daily"
                elif days <= 7:
                    frequency = "weekly"
                elif days <= 31:
                    frequency = "monthly"
                elif days <= 93:
                    frequency = "quarterly"
                else:
                    frequency = "annual"
            else:
                frequency = "unknown"
        else:
            frequency = "unknown"
        
        # Add time features
        df = df.with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.quarter().alias("quarter")
        ])
        
        # Save original and resampled versions
        series_path = silver_path / "fred" / series_name
        series_path.mkdir(parents=True, exist_ok=True)
        
        # Original frequency
        original_path = series_path / "original.parquet"
        df.write_parquet(str(original_path))
        
        saved = {"original": str(original_path)}
        
        # Create monthly version for easier joining
        if frequency == "monthly":
            # Already monthly, just save as is
            monthly_path = series_path / "monthly.parquet"
            df.write_parquet(str(monthly_path))
            saved["monthly"] = str(monthly_path)
        elif frequency != "unknown":
            # Resample to monthly
            value_col = [c for c in df.columns if c not in ["date", "year", "month", "quarter"]][0]
            
            if frequency == "daily" or frequency == "weekly":
                # Aggregate to monthly mean
                df_monthly = df.group_by(["year", "month"]).agg([
                    pl.col("date").min().alias("date"),
                    pl.col(value_col).mean().alias(f"{value_col}_mean"),
                    pl.col(value_col).min().alias(f"{value_col}_min"),
                    pl.col(value_col).max().alias(f"{value_col}_max"),
                    pl.col(value_col).std().alias(f"{value_col}_std"),
                    pl.col(value_col).count().alias("observations")
                ])
                # Normalize date to first day of month and add quarter column
                df_monthly = df_monthly.with_columns([
                    pl.date(pl.col("year").cast(pl.Int32), pl.col("month").cast(pl.Int32), 1).alias("date"),
                    pl.col("date").dt.quarter().alias("quarter")
                ])
                # Sort by date to ensure chronological order
                df_monthly = df_monthly.sort("date")
            else:  # quarterly or annual
                # Forward fill to monthly - only interpolate the value column, not date components
                df_monthly = df.set_sorted("date").upsample("date", every="1mo")
                # Forward fill the value column
                df_monthly = df_monthly.with_columns([
                    pl.col(value_col).forward_fill()
                ])
                # Recalculate date components from the upsampled date
                df_monthly = df_monthly.with_columns([
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.month().alias("month"),
                    pl.col("date").dt.quarter().alias("quarter")
                ])
            
            monthly_path = series_path / "monthly.parquet"
            df_monthly.write_parquet(str(monthly_path))
            saved["monthly"] = str(monthly_path)
        
        all_series[series_name] = saved
        series_metadata[series_name] = {
            "frequency": frequency,
            "observations": df.height,
            "start_date": str(df["date"].min()),
            "end_date": str(df["date"].max()),
            "columns": df.columns
        }
    
    # Create unified monthly dataset
    monthly_frames = []
    for series_name, paths in all_series.items():
        monthly_path = paths.get("monthly", paths.get("original"))
        if monthly_path:
            df = pl.read_parquet(monthly_path)
            # Get value column - use the first one that matches the series name pattern
            value_cols = [c for c in df.columns if c not in ["date", "year", "month", "quarter", "observations"]]
            if value_cols:
                # For aggregated data, prefer the _mean column
                value_col = None
                for c in value_cols:
                    if f"{series_name}_mean" == c:
                        value_col = c
                        break
                if not value_col:
                    value_col = value_cols[0]
                
                # Take just date and value column, rename to series name
                df_series = df.select(["date", value_col])
                if value_col != series_name:
                    df_series = df_series.rename({value_col: series_name})
                monthly_frames.append(df_series)
    
    if monthly_frames:
        # Combine all dataframes by concatenating horizontally after ensuring all have same dates
        # First get all unique dates - convert to datetime for consistency
        all_dates = set()
        for frame in monthly_frames:
            # Ensure date column is datetime
            if frame["date"].dtype == pl.Date:
                frame = frame.with_columns(pl.col("date").cast(pl.Datetime))
            all_dates.update(frame["date"].to_list())
        
        # Create base dataframe with all dates
        unified = pl.DataFrame({"date": sorted(list(all_dates))})
        
        # Join each series
        for frame in monthly_frames:
            # Ensure date column is datetime for joining
            if frame["date"].dtype == pl.Date:
                frame = frame.with_columns(pl.col("date").cast(pl.Datetime))
            # Get the non-date column name
            value_col = [c for c in frame.columns if c != "date"][0]
            # Join with suffix handling
            unified = unified.join(frame, on="date", how="left")
        
        unified = unified.sort("date")
        unified_path = silver_path / "fred" / "unified_monthly.parquet"
        unified.write_parquet(str(unified_path))
        print(f"  Saved unified monthly FRED data: {unified.height} rows, {len(unified.columns)} columns")
    
    # Save metadata
    metadata = {
        "transformed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "series": series_metadata,
        "total_series": len(all_series)
    }
    
    metadata_path = silver_path / "fred" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return all_series


def main():
    """Run the enhanced transform pipeline"""
    
    print("Enhanced Transform Pipeline")
    print("=" * 50)
    
    # Ensure directories
    SILVER.mkdir(parents=True, exist_ok=True)
    (SILVER / "zillow").mkdir(parents=True, exist_ok=True)
    (SILVER / "fred").mkdir(parents=True, exist_ok=True)
    
    # Process Zillow data
    zillow_bronze = BRONZE / "zillow"
    if zillow_bronze.exists():
        zillow_configs = [d for d in zillow_bronze.iterdir() if d.is_dir()]
        print(f"\nFound {len(zillow_configs)} Zillow datasets")
        
        transform_summary = {}
        
        for config_dir in zillow_configs:
            config_name = config_dir.name
            print(f"\nTransforming: {config_name}")
            
            try:
                results, metrics = transform_zillow_dataset(config_name, config_dir)
                saved = save_silver_data(config_name, results, metrics, SILVER / "zillow")
                
                transform_summary[config_name] = {
                    "status": "success",
                    "levels": list(results.keys()),
                    "metrics_count": len(metrics),
                    "files": saved
                }
                
            except Exception as e:
                import traceback
                print(f"  Error: {e}")
                print(f"  Traceback: {traceback.format_exc()}")
                transform_summary[config_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Save overall Zillow summary
        summary_path = SILVER / "zillow" / "transform_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "transformed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "datasets": transform_summary
            }, f, indent=2)
        
        print(f"\nZillow transform complete. Summary saved to {summary_path}")
    
    # Process FRED data
    fred_bronze = BRONZE / "macro"
    if fred_bronze.exists():
        print(f"\nTransforming FRED data")
        fred_results = transform_fred_data(fred_bronze, SILVER)
        print(f"FRED transform complete. {len(fred_results)} series processed")
    
    print("\n" + "=" * 50)
    print("Transform pipeline complete!")
    print(f"Silver data location: {SILVER}")


if __name__ == "__main__":
    main()
