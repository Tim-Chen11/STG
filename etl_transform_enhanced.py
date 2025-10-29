#!/usr/bin/env python3
"""
Enhanced Transform Module for Zillow + FRED ETL Pipeline
Preserves all data richness with multi-level aggregation
"""

import os
import json
import pathlib
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import datetime as dt

import pandas as pd
import polars as pl
import pyarrow.parquet as pq


# =========================
# Configuration
# =========================
BASE_OUT = pathlib.Path("data")
BRONZE = BASE_OUT / "raw"
SILVER = BASE_OUT / "silver"

# Region hierarchy levels
REGION_LEVELS = {
    "nation": ["usa", "united states", "national"],
    "state": ["state"],
    "metro": ["msa", "cbsa", "metro", "metropolitan"],
    "county": ["county"],
    "city": ["city", "place", "town"],
    "zip": ["zip", "zipcode", "zip code", "postal"]
}

# Metric categories for better organization
METRIC_CATEGORIES = {
    "price": ["price", "value", "zhvi", "zori", "cost"],
    "inventory": ["inventory", "listings", "supply", "active", "new", "pending"],
    "time": ["days", "dom", "time", "duration"],
    "volume": ["sales", "sold", "count", "transactions"],
    "ratio": ["ratio", "percent", "pct", "share"],
    "forecast": ["forecast", "prediction", "projected", "expected"],
    "growth": ["yoy", "mom", "growth", "change", "appreciation"]
}


# =========================
# Data Schema Manager
# =========================
@dataclass
class ColumnMetadata:
    """Metadata for a single column"""
    original_name: str
    clean_name: str
    data_type: str
    category: Optional[str]
    description: Optional[str]
    unit: Optional[str]
    frequency: Optional[str]
    is_seasonal_adjusted: bool = False
    is_smoothed: bool = False
    percentile: Optional[int] = None  # For tier data (e.g., bottom 33%, top 67%)


class SchemaManager:
    """Manages column schemas and metadata"""
    
    def __init__(self):
        self.schemas: Dict[str, List[ColumnMetadata]] = {}
        self.load_known_schemas()
    
    def load_known_schemas(self):
        """Load predefined schema knowledge about Zillow columns"""
        # This would ideally load from a config file
        self.known_patterns = {
            "zhvi": {"category": "price", "unit": "USD", "description": "Zillow Home Value Index"},
            "zori": {"category": "price", "unit": "USD", "description": "Zillow Observed Rent Index"},
            "median_sale_price": {"category": "price", "unit": "USD", "description": "Median sale price"},
            "median_list_price": {"category": "price", "unit": "USD", "description": "Median listing price"},
            "median_dom": {"category": "time", "unit": "days", "description": "Median days on market"},
            "inventory": {"category": "inventory", "unit": "count", "description": "Active inventory count"},
            "new_listings": {"category": "inventory", "unit": "count", "description": "New listings added"},
            "price_cut": {"category": "ratio", "unit": "percent", "description": "Share of listings with price cut"},
            "_sa": {"is_seasonal_adjusted": True},
            "_smoothed": {"is_smoothed": True},
            "_raw": {"is_smoothed": False},
            "bottom_tier": {"percentile": 33},
            "middle_tier": {"percentile": 50},
            "top_tier": {"percentile": 67}
        }
    
    def analyze_column(self, col_name: str, sample_data: Optional[pl.Series] = None) -> ColumnMetadata:
        """Analyze a column and extract metadata"""
        clean_name = self._clean_column_name(col_name)
        
        # Determine data type
        data_type = "unknown"
        if sample_data is not None:
            dtype = sample_data.dtype
            if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                data_type = "numeric"
            elif dtype in [pl.Utf8, pl.Categorical]:
                data_type = "string"
            elif dtype in [pl.Date, pl.Datetime]:
                data_type = "datetime"
            else:
                data_type = str(dtype)
        
        # Extract metadata from column name
        meta = ColumnMetadata(
            original_name=col_name,
            clean_name=clean_name,
            data_type=data_type,
            category=self._detect_category(col_name),
            description=None,
            unit=self._detect_unit(col_name),
            frequency=None,
            is_seasonal_adjusted=self._is_seasonal_adjusted(col_name),
            is_smoothed=self._is_smoothed(col_name),
            percentile=self._detect_percentile(col_name)
        )
        
        # Apply known patterns
        for pattern, attrs in self.known_patterns.items():
            if pattern.lower() in col_name.lower():
                for key, value in attrs.items():
                    if hasattr(meta, key) and getattr(meta, key) is None:
                        setattr(meta, key, value)
        
        return meta
    
    def _clean_column_name(self, name: str) -> str:
        """Clean column name to standard format"""
        import re
        # Remove special characters and normalize
        name = re.sub(r'[^\w]+', '_', name.lower())
        name = re.sub(r'_+', '_', name)
        return name.strip('_')
    
    def _detect_category(self, col_name: str) -> Optional[str]:
        """Detect metric category from column name"""
        col_lower = col_name.lower()
        for category, patterns in METRIC_CATEGORIES.items():
            if any(p in col_lower for p in patterns):
                return category
        return None
    
    def _detect_unit(self, col_name: str) -> Optional[str]:
        """Detect unit from column name"""
        col_lower = col_name.lower()
        if any(x in col_lower for x in ["price", "value", "zhvi", "zori", "cost"]):
            return "USD"
        elif any(x in col_lower for x in ["percent", "pct", "share", "ratio"]):
            return "percent"
        elif any(x in col_lower for x in ["days", "dom"]):
            return "days"
        elif any(x in col_lower for x in ["count", "inventory", "listings"]):
            return "count"
        return None
    
    def _is_seasonal_adjusted(self, col_name: str) -> bool:
        """Check if column is seasonally adjusted"""
        col_lower = col_name.lower()
        return "_sa" in col_lower or "seasonally_adjusted" in col_lower
    
    def _is_smoothed(self, col_name: str) -> bool:
        """Check if column is smoothed"""
        col_lower = col_name.lower()
        return "smoothed" in col_lower or "_sm" in col_lower
    
    def _detect_percentile(self, col_name: str) -> Optional[int]:
        """Detect if column represents a percentile tier"""
        col_lower = col_name.lower()
        if "bottom" in col_lower or "low" in col_lower:
            return 33
        elif "middle" in col_lower or "mid" in col_lower:
            return 50
        elif "top" in col_lower or "high" in col_lower:
            return 67
        return None


# =========================
# Transform Functions
# =========================

def detect_region_columns(df: pl.DataFrame) -> Dict[str, Optional[str]]:
    """Detect all region-related columns in a dataframe"""
    cols = df.columns
    cols_lower = {c.lower(): c for c in cols}
    
    result = {
        "region_id": None,
        "region_name": None,
        "region_type": None,
        "state": None,
        "state_code": None,
        "county": None,
        "city": None,
        "metro": None,
        "zip": None
    }
    
    # Direct matches
    patterns = {
        "region_id": ["region_id", "regionid", "geoid", "fips"],
        "region_name": ["region_name", "regionname", "region", "name"],
        "region_type": ["region_type", "regiontype", "type"],
        "state": ["state", "state_name"],
        "state_code": ["state_code", "stateabbr", "state_abbr"],
        "county": ["county", "county_name"],
        "city": ["city", "city_name", "place"],
        "metro": ["metro", "msa", "cbsa", "metro_name"],
        "zip": ["zip", "zipcode", "zip_code", "postal_code"]
    }
    
    for key, patterns_list in patterns.items():
        for pattern in patterns_list:
            if pattern in cols_lower:
                result[key] = cols_lower[pattern]
                break
    
    return result


def detect_date_columns(df: pl.DataFrame) -> List[str]:
    """Detect all date-related columns"""
    date_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        # Check column name
        if any(d in col_lower for d in ["date", "time", "period", "month", "year", "week"]):
            date_cols.append(col)
            continue
        
        # Check data type
        if df[col].dtype in [pl.Date, pl.Datetime]:
            date_cols.append(col)
    
    return date_cols


def extract_all_metrics(df: pl.DataFrame, schema_mgr: SchemaManager) -> Dict[str, ColumnMetadata]:
    """Extract all numeric columns as metrics with metadata"""
    metrics = {}
    
    for col in df.columns:
        # Skip region and date columns
        col_lower = col.lower()
        if any(x in col_lower for x in ["region", "state", "county", "city", "zip", "date", "time", "period"]):
            continue
        
        # Check if numeric
        if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            sample = df[col].head(100) if df.height > 100 else df[col]
            metadata = schema_mgr.analyze_column(col, sample)
            metrics[col] = metadata
        
        # Also handle string columns that might be categorical metrics
        elif df[col].dtype in [pl.Utf8, pl.Categorical]:
            # Check if it's a metric encoded as string (e.g., "increasing", "decreasing")
            unique_vals = df[col].unique().to_list()
            if len(unique_vals) < 20:  # Likely categorical
                metadata = schema_mgr.analyze_column(col, df[col])
                metadata.data_type = "categorical"
                metrics[col] = metadata
    
    return metrics


def create_region_hierarchy(df: pl.DataFrame, region_cols: Dict[str, Optional[str]]) -> pl.DataFrame:
    """Add region hierarchy columns to dataframe"""
    df_with_hierarchy = df
    
    # Ensure we have a region level column
    if "region_level" not in df.columns:
        # Determine region level from region_type or infer from available columns
        if region_cols["region_type"]:
            df_with_hierarchy = df_with_hierarchy.with_columns([
                pl.col(region_cols["region_type"]).alias("region_level")
            ])
        else:
            # Infer from what columns we have
            if region_cols["zip"]:
                level = "zip"
            elif region_cols["city"]:
                level = "city"
            elif region_cols["county"]:
                level = "county"
            elif region_cols["metro"]:
                level = "metro"
            elif region_cols["state"]:
                level = "state"
            else:
                level = "unknown"
            
            df_with_hierarchy = df_with_hierarchy.with_columns([
                pl.lit(level).alias("region_level")
            ])
    
    # Create unified region key for joining
    region_key_expr = None
    if region_cols.get("region_id"):
        region_key_expr = pl.col(region_cols["region_id"]).cast(pl.Utf8)
    elif region_cols.get("region_name"):
        region_key_expr = pl.col(region_cols["region_name"]).cast(pl.Utf8)
    else:
        # Concatenate available region info
        parts = []
        for key in ["state", "county", "city", "metro", "zip"]:
            if region_cols.get(key):
                parts.append(pl.col(region_cols[key]).cast(pl.Utf8))
        if parts:
            region_key_expr = pl.concat_str(parts, separator="_")
    
    if region_key_expr is not None:
        df_with_hierarchy = df_with_hierarchy.with_columns([
            region_key_expr.alias("region_key")
        ])
    
    return df_with_hierarchy


def standardize_dates(df: pl.DataFrame, date_cols: List[str]) -> pl.DataFrame:
    """Standardize all date columns to consistent format"""
    df_std = df
    
    # Find primary date column
    primary_date = None
    for col in date_cols:
        col_lower = col.lower()
        if col_lower in ["date", "period", "month"]:
            primary_date = col
            break
    
    if not primary_date and date_cols:
        primary_date = date_cols[0]
    
    if primary_date and primary_date in df.columns:
        # Convert to datetime and add standard date columns
        date_expr = pl.col(primary_date)
        
        # Check current dtype
        current_dtype = df[primary_date].dtype
        
        if current_dtype == pl.Utf8:
            date_expr = date_expr.str.strptime(pl.Datetime, strict=False)
        elif current_dtype not in [pl.Date, pl.Datetime]:
            date_expr = date_expr.cast(pl.Datetime)
        
        # Only alias to 'date' if it's not already called 'date'
        new_cols = []
        if primary_date.lower() != "date":
            new_cols.append(date_expr.alias("date"))
        
        # Only add date components if they don't already exist
        if "year" not in df.columns:
            new_cols.append(date_expr.dt.year().alias("year"))
        if "month" not in df.columns:
            new_cols.append(date_expr.dt.month().alias("month"))
        if "quarter" not in df.columns:
            new_cols.append(date_expr.dt.quarter().alias("quarter"))
        if "week" not in df.columns:
            new_cols.append(date_expr.dt.week().alias("week"))
        
        if new_cols:
            df_std = df_std.with_columns(new_cols)
    
    return df_std


def transform_zillow_dataset(config_name: str, bronze_path: pathlib.Path, schema_mgr: SchemaManager) -> Dict[str, pl.DataFrame]:
    """
    Transform a single Zillow dataset into multiple silver tables
    Returns dict of {level: dataframe} for different aggregation levels
    """
    
    # Read raw data
    parquet_files = list(bronze_path.glob("*.parquet"))
    if not parquet_files:
        # Check for train subdirectory (Hugging Face structure)
        parquet_files = list((bronze_path / "train").glob("*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {bronze_path}")
    
    # Load data
    df = pl.scan_parquet([str(f) for f in parquet_files]).collect()
    
    print(f"  Processing {config_name}: {df.height:,} rows, {len(df.columns)} columns")
    
    # Detect column types
    region_cols = detect_region_columns(df)
    date_cols = detect_date_columns(df)
    
    # Standardize dates
    if date_cols:
        df = standardize_dates(df, date_cols)
    
    # Add region hierarchy
    df = create_region_hierarchy(df, region_cols)
    
    # Extract all metrics with metadata
    metrics = extract_all_metrics(df, schema_mgr)
    
    # Create column groups for organized output
    id_cols = ["region_key", "region_level"]
    if "region_id" in df.columns:
        id_cols.append("region_id")
    
    geo_cols = []
    for col in ["state", "state_code", "county", "city", "metro", "zip"]:
        if region_cols.get(col) is not None:
            if region_cols[col] in df.columns:
                geo_cols.append(region_cols[col])
    
    date_cols_final = ["date", "year", "month", "quarter", "week"]
    date_cols_final = [c for c in date_cols_final if c in df.columns]
    
    metric_cols = list(metrics.keys())
    
    # Build final dataframe with all columns organized - ensure uniqueness
    all_cols = id_cols + geo_cols + date_cols_final + metric_cols
    # Remove duplicates while preserving order
    seen = set()
    unique_cols = []
    for col in all_cols:
        if col not in seen and col in df.columns:
            seen.add(col)
            unique_cols.append(col)
    
    df_final = df.select(unique_cols)
    
    # Create aggregated views for different region levels
    results = {}
    
    # Original granularity
    results["original"] = df_final
    
    # Aggregate by different levels if data supports it
    has_state = region_cols.get("state") is not None and region_cols["state"] in df.columns
    if has_state or "state" in df.columns:
        # State level aggregation
        state_col = region_cols["state"] if region_cols.get("state") else "state"
        if state_col in df.columns and "date" in df.columns:
            agg_exprs = []
            for metric in metric_cols:
                if df[metric].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    agg_exprs.extend([
                        pl.col(metric).mean().alias(f"{metric}_mean"),
                        pl.col(metric).median().alias(f"{metric}_median"),
                        pl.col(metric).std().alias(f"{metric}_std"),
                        pl.col(metric).min().alias(f"{metric}_min"),
                        pl.col(metric).max().alias(f"{metric}_max"),
                        pl.col(metric).count().alias(f"{metric}_count")
                    ])
            
            if agg_exprs:
                df_state = df.group_by([state_col, "date"]).agg(agg_exprs)
                results["state"] = df_state
    
    # Metro level if available
    if region_cols.get("metro") is not None and region_cols["metro"] in df.columns and "date" in df.columns:
        metro_col = region_cols["metro"]
        agg_exprs = []
        for metric in metric_cols:
            if df[metric].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                agg_exprs.append(pl.col(metric).mean().alias(f"{metric}_mean"))
        
        if agg_exprs:
            # Include state if available for hierarchy
            group_cols = [metro_col, "date"]
            if region_cols.get("state") is not None and region_cols["state"] in df.columns:
                group_cols.insert(0, region_cols["state"])
            
            df_metro = df.group_by(group_cols).agg(agg_exprs)
            results["metro"] = df_metro
    
    # County level if available
    if region_cols.get("county") is not None and region_cols["county"] in df.columns and "date" in df.columns:
        county_col = region_cols["county"]
        agg_exprs = []
        for metric in metric_cols:
            if df[metric].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                agg_exprs.append(pl.col(metric).mean().alias(f"{metric}_mean"))
        
        if agg_exprs:
            group_cols = [county_col, "date"]
            if region_cols.get("state") is not None and region_cols["state"] in df.columns:
                group_cols.insert(0, region_cols["state"])
            
            df_county = df.group_by(group_cols).agg(agg_exprs)
            results["county"] = df_county
    
    return results, metrics


def save_silver_data(config_name: str, results: Dict[str, pl.DataFrame], 
                    metrics: Dict[str, ColumnMetadata], silver_path: pathlib.Path):
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
        "metrics": {
            col: asdict(meta) for col, meta in metrics.items()
        },
        "total_metrics": len(metrics),
        "metric_categories": {}
    }
    
    # Count metrics by category
    for meta in metrics.values():
        if meta.category:
            if meta.category not in metadata["metric_categories"]:
                metadata["metric_categories"][meta.category] = 0
            metadata["metric_categories"][meta.category] += 1
    
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
    
    # Initialize schema manager
    schema_mgr = SchemaManager()
    
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
                results, metrics = transform_zillow_dataset(config_name, config_dir, schema_mgr)
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