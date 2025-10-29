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
    Transform ONE Zillow config folder into:
      - 'wide' (cleaned base columns + all value columns)
      - 'long' (normalized: one row per metric per date per region/home-type)
    We DO NOT aggregate or guess region level; we map label-encodings per config.
    """
    # 1) locate parquet shards
    shards = list(bronze_path.glob("*.parquet"))
    if not shards:
        shards = list((bronze_path / "train").glob("*.parquet"))
    if not shards:
        raise ValueError(f"No parquet files found in {bronze_path}")

    # 2) load
    df = pl.scan_parquet([str(p) for p in shards]).collect()

    # 3) normalize config key
    cfg = config_name.strip().lower()
    if cfg not in ZILLOW_BASE_COLS:
        for k in ZILLOW_BASE_COLS:
            if cfg.startswith(k):
                cfg = k
                break
    if cfg not in ZILLOW_BASE_COLS:
        raise ValueError(f"Unknown Zillow config '{config_name}'")

    base_cols = ZILLOW_BASE_COLS[cfg]

    # 4) Date → 'date' (datetime)
    if "Date" in df.columns:
        if df["Date"].dtype in (pl.Int64, pl.Int32):
            df = df.with_columns(pl.from_epoch(pl.col("Date").cast(pl.Int64), unit="ms").alias("date"))
        elif df["Date"].dtype == pl.Utf8:
            df = df.with_columns(pl.col("Date").str.strptime(pl.Datetime, strict=False).alias("date"))
        elif df["Date"].dtype in (pl.Datetime, pl.Date):
            df = df.with_columns(pl.col("Date").cast(pl.Datetime).alias("date"))
        else:
            df = df.with_columns(pl.from_epoch(pl.col("Date").cast(pl.Int64), unit="ms").alias("date"))
    else:
        if "date" not in df.columns:
            raise ValueError(f"{config_name}: no Date column")

    out = df

    # 5) region_id / region_name
    if "Region ID" in out.columns:
        out = out.with_columns(pl.col("Region ID").cast(pl.Utf8).alias("region_id"))
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias("region_id"))

    if "Region" in out.columns:
        out = out.with_columns(pl.col("Region").cast(pl.Utf8).alias("region_name"))
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias("region_name"))

    # 6) Region Type mapping (fix: no Expr.dtype; cast first, then branch on nulls)
    if "Region Type" in out.columns:
        map_dict = ZILLOW_REGION_TYPE.get(cfg, {})
        out = out.with_columns([
            pl.col("Region Type").cast(pl.Int64, strict=False).alias("_rt_code"),
            pl.col("Region Type").cast(pl.Utf8).alias("_rt_str"),
        ])
        out = out.with_columns(
            pl.when(pl.col("_rt_code").is_not_null())
              .then(pl.col("_rt_code").map_elements(lambda v: map_dict.get(int(v)) if v is not None else None, return_dtype=pl.Utf8))
              .otherwise(pl.col("_rt_str"))
              .alias("region_level")
        ).drop(["_rt_code", "_rt_str"])
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias("region_level"))

    # 7) Home Type mapping (same pattern)
    if "Home Type" in out.columns:
        if cfg in ZILLOW_HOME_TYPE_BY_CONFIG:
            ht_map = ZILLOW_HOME_TYPE_BY_CONFIG[cfg]
            out = out.with_columns([
                pl.col("Home Type").cast(pl.Int64, strict=False).alias("_ht_code"),
                pl.col("Home Type").cast(pl.Utf8).alias("_ht_str"),
            ])
            out = out.with_columns(
                pl.when(pl.col("_ht_code").is_not_null())
                  .then(pl.col("_ht_code").map_elements(lambda v: ht_map.get(int(v)) if v is not None else None, return_dtype=pl.Utf8))
                  .otherwise(pl.col("_ht_str"))
                  .alias("home_type")
            ).drop(["_ht_code", "_ht_str"])
        else:
            # forecasts: keep as text
            out = out.with_columns(pl.col("Home Type").cast(pl.Utf8).alias("home_type"))
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias("home_type"))

    # 8) Bedroom Count (home_values only) with safe cast
    if cfg == "home_values" and "Bedroom Count" in out.columns:
        out = out.with_columns([
            pl.col("Bedroom Count").cast(pl.Int64, strict=False).alias("_bed_code")
        ])
        out = out.with_columns(
            pl.col("_bed_code").map_elements(
                lambda v: ZILLOW_BEDROOM_LABELS.get(int(v)) if v is not None else None,
                return_dtype=pl.Utf8
            ).alias("bedrooms")
        ).drop("_bed_code")
    else:
        if "Bedroom Count" in out.columns:
            out = out.with_columns(pl.col("Bedroom Count").cast(pl.Utf8).alias("bedrooms"))
        else:
            out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias("bedrooms"))

    # 9) time parts
    out = out.with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.quarter().alias("quarter"),
    ])

    # 10) passthrough dims for forecasts if present
    dim_passthrough = [c for c in ("City","County","Metro","State") if c in out.columns]

    # 11) value columns = numeric columns NOT in base + not in our helper IDs
    present_base = {c for c in base_cols if c in out.columns}
    helper_cols = {
        "region_id","region_name","region_level","home_type","bedrooms",
        "date","year","month","quarter","Size Rank"
    }.union(dim_passthrough)

    candidate_vals = [c for c in out.columns if c not in present_base and c not in helper_cols]
    value_cols = []
    schema_map = out.schema  # faster than out[c].dtype repeatedly
    for c in candidate_vals:
        dtp = schema_map[c]
        if dtp.is_numeric() or dtp in (pl.Float32, pl.Float64, pl.Int64, pl.Int32):
            value_cols.append(c)

    # 12) wide
    keep_cols = [c for c in ["region_id","region_name","region_level","home_type","bedrooms",
                             "date","year","month","quarter","Size Rank"] if c in out.columns]
    keep_cols += dim_passthrough
    wide = out.select(keep_cols + value_cols).sort(
        [c for c in ["region_level","region_name","home_type","bedrooms","date"] if c in keep_cols]
    )

    # 13) long
    id_cols = [c for c in (keep_cols) if c in wide.columns]
    if value_cols:
        long = wide.melt(id_vars=id_cols, value_vars=value_cols, variable_name="metric", value_name="value")
        long = long.with_columns([
            pl.col("metric").str.contains("Smoothed").alias("is_smoothed"),
            pl.col("metric").str.contains("Seasonally Adjusted").alias("is_seasonally_adjusted"),
            pl.col("metric").str.contains(r"\(Raw\)").alias("is_raw"),
        ])
    else:
        long = wide.with_columns([
            pl.lit(None).alias("metric"),
            pl.lit(None, dtype=pl.Float64).alias("value"),
            pl.lit(False).alias("is_smoothed"),
            pl.lit(False).alias("is_seasonally_adjusted"),
            pl.lit(False).alias("is_raw"),
        ])

    # 14) return two tables; your saver will write them out
    return {"wide": wide, "long": long}, {}



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