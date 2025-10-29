#!/usr/bin/env python3
"""
ETL Extract Module - Bronze Layer
Handles data extraction from Zillow (Hugging Face) and FRED
"""

import os
import io
import json
import pathlib
import shutil
import datetime as dt
from typing import Dict, Optional

import requests
import pandas as pd
from huggingface_hub import snapshot_download, get_token


# =========================
# Configuration
# =========================
BASE_OUT = pathlib.Path("data")
BRONZE = BASE_OUT / "raw"

# Zillow configuration
ZILLOW_ID = "misikoff/zillow-viewer"
ZILLOW_REV = "~parquet"  # auto-parquet branch

# FRED series (validated IDs)
FRED_SERIES: Dict[str, str] = {
    "mortgage_rate": "MORTGAGE30US",   # weekly, %
    "cpi": "CPIAUCSL",                 # monthly, index
    "income_median": "MEHOINUSA672N",  # annual, $
    "unemployment_rate": "UNRATE",     # monthly, %
    "building_permits": "PERMIT",      # monthly, SAAR
    "housing_starts": "HOUST",         # monthly, SAAR
    "median_sales_price": "MSPUS",     # quarterly, $
}
FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Data quality thresholds
MIN_ROWS_PER_ZILLOW_CONFIG = 1


# =========================
# Utils
# =========================
def log(msg: str):
    """Log message with timestamp"""
    ts = dt.datetime.now().strftime("%H:%M:%S")
    try:
        print(f"{ts} | {msg}")
    except UnicodeEncodeError:
        msg_safe = msg.encode('ascii', 'replace').decode('ascii')
        print(f"{ts} | {msg_safe}")


def ensure_dirs():
    """Create necessary directories"""
    (BRONZE / "zillow").mkdir(parents=True, exist_ok=True)
    (BRONZE / "macro").mkdir(parents=True, exist_ok=True)


def write_json(path: pathlib.Path, obj):
    """Write JSON file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parquet_rows(p: pathlib.Path) -> int:
    """Count rows in parquet file"""
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(p).metadata.num_rows
    except Exception:
        return -1


def fred_url(series_id: str, start: Optional[str] = None, end: Optional[str] = None) -> str:
    """Build FRED download URL"""
    qs = f"id={series_id}"
    if start:
        qs += f"&cosd={start}"
    if end:
        qs += f"&coed={end}"
    return f"{FRED_CSV_BASE}?{qs}"


# =========================
# Zillow Extraction
# =========================
def extract_zillow_bronze(force_redownload: bool = False) -> dict:
    """
    Extract Zillow data from Hugging Face dataset.
    Downloads all parquet shards and saves them to bronze layer.
    
    Args:
        force_redownload: If True, always download. If False, skip if data exists.
    
    Returns:
        dict: Manifest of extracted data
    """
    z_base = BRONZE / "zillow"
    manifest_path = z_base / "manifest.json"
    
    # Check if data already exists
    if not force_redownload and manifest_path.exists():
        log("Zillow bronze data already exists. Use --force-extract to re-download.")
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    log("Checking Hugging Face token...")
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        log("Warning: No HF token detected. Run `huggingface-cli login` or set HF_TOKEN.")
    
    log(f"Downloading {ZILLOW_ID}@{ZILLOW_REV} (parquet shards)...")
    cache_dir = snapshot_download(
        repo_id=ZILLOW_ID,
        repo_type="dataset",
        revision=ZILLOW_REV,
        allow_patterns=["*/train/*.parquet"],
        local_dir_use_symlinks=False,
        max_workers=4
    )
    cache = pathlib.Path(cache_dir)
    
    # Find all configurations (datasets)
    configs = []
    for p in cache.iterdir():
        if p.is_dir() and (p / "train").exists() and list((p / "train").glob("*.parquet")):
            configs.append(p.name)
    configs = sorted(configs)
    log(f"Found Zillow configs: {configs}")
    
    man = {
        "dataset": f"{ZILLOW_ID}@{ZILLOW_REV}",
        "extracted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "configs": {}
    }
    
    # Copy each configuration's data
    for cfg in configs:
        src_dir = cache / cfg / "train"
        dst_dir = z_base / cfg
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        files_meta = []
        total_rows = 0
        
        for src in sorted(src_dir.glob("*.parquet")):
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            
            rows = parquet_rows(dst)
            total_rows += max(rows, 0)
            files_meta.append({
                "file": dst.as_posix(),
                "rows": rows,
                "bytes": dst.stat().st_size
            })
        
        # Get sample metadata
        sample_cols = []
        date_range = {"min": None, "max": None}
        if files_meta:
            try:
                import polars as pl
                # Read just schema from first file
                sample_df = pl.read_parquet(files_meta[0]["file"], n_rows=1)
                sample_cols = sample_df.columns
                
                # Try to get date range
                date_col = None
                for col in sample_cols:
                    if col.lower() in ["date", "period"]:
                        date_col = col
                        break
                
                if date_col:
                    all_files = dst_dir / "*.parquet"
                    try:
                        df_dates = pl.scan_parquet(str(all_files)).select(date_col).collect()
                        date_range["min"] = str(df_dates[date_col].min())
                        date_range["max"] = str(df_dates[date_col].max())
                    except:
                        pass
            except:
                pass
        
        man["configs"][cfg] = {
            "dest_dir": dst_dir.as_posix(),
            "num_files": len(files_meta),
            "total_rows": total_rows if total_rows > 0 else None,
            "columns_example": sample_cols,
            "date_range": date_range,
            "files": files_meta
        }
        
        if len(files_meta) < MIN_ROWS_PER_ZILLOW_CONFIG:
            log(f"Warning: {cfg} appears empty; check your HF auth/rate limits.")
        else:
            log(f"  {cfg}: {len(files_meta)} files, {total_rows:,} rows")
    
    write_json(manifest_path, man)
    log(f"Zillow bronze manifest saved to {manifest_path}")
    return man


# =========================
# FRED Extraction
# =========================
def extract_fred_bronze(
    start: Optional[str] = None, 
    end: Optional[str] = None, 
    force_redownload: bool = False
) -> dict:
    """
    Extract FRED economic data series.
    Downloads CSV data and saves as both CSV and Parquet.
    
    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        force_redownload: If True, always download. If False, skip if data exists.
    
    Returns:
        dict: Manifest of extracted data
    """
    m_base = BRONZE / "macro"
    m_base.mkdir(parents=True, exist_ok=True)
    manifest_path = m_base / "manifest.json"
    
    # Check if data already exists
    if not force_redownload and manifest_path.exists():
        log("FRED bronze data already exists. Use --force-extract to re-download.")
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    man = {
        "source": "FRED (fredgraph.csv)",
        "downloaded_at": dt.datetime.utcnow().isoformat() + "Z",
        "series": {}
    }
    
    for name, sid in FRED_SERIES.items():
        url = fred_url(sid, start, end)
        log(f"Downloading FRED {name} ({sid})...")
        
        csv_path = m_base / f"{name}.csv"
        pq_path = m_base / f"{name}.parquet"
        
        try:
            # Download data
            r = requests.get(url, timeout=45)
            if r.status_code != 200:
                log(f"  HTTP {r.status_code} - skip {sid}")
                man["series"][name] = {
                    "id": sid,
                    "status": f"http_{r.status_code}",
                    "url": url
                }
                continue
            
            # Parse CSV
            df = pd.read_csv(io.StringIO(r.text))
            if "observation_date" not in df.columns or sid not in df.columns:
                raise ValueError(f"Unexpected columns for {sid}: {df.columns.tolist()}")
            
            # Rename columns
            df.rename(columns={"observation_date": "date", sid: name}, inplace=True)
            
            # Save both formats
            df.to_csv(csv_path, index=False)
            df.to_parquet(pq_path, index=False)
            
            # Get date range
            d0 = pd.to_datetime(df["date"], errors="coerce")
            
            man["series"][name] = {
                "id": sid,
                "status": "ok",
                "url": url,
                "rows": int(df.shape[0]),
                "min_date": d0.min().strftime("%Y-%m-%d") if len(d0) else None,
                "max_date": d0.max().strftime("%Y-%m-%d") if len(d0) else None,
                "csv": csv_path.as_posix(),
                "parquet": pq_path.as_posix()
            }
            log(f"  Saved {csv_path.name} | {pq_path.name} ({df.shape[0]} rows)")
            
        except Exception as e:
            log(f"  Failed {sid}: {e}")
            man["series"][name] = {
                "id": sid,
                "status": "error",
                "error": str(e),
                "url": url
            }
    
    write_json(manifest_path, man)
    log(f"FRED bronze manifest saved to {manifest_path}")
    return man


# =========================
# Main Extraction Pipeline
# =========================
def extract_all(
    force_redownload: bool = False,
    fred_start: Optional[str] = None,
    fred_end: Optional[str] = None
) -> dict:
    """
    Run complete extraction pipeline.
    
    Args:
        force_redownload: Force re-download even if data exists
        fred_start: FRED start date
        fred_end: FRED end date
    
    Returns:
        dict: Combined manifest of all extracted data
    """
    ensure_dirs()
    
    log("=" * 50)
    log("ETL Extraction Pipeline (Bronze Layer)")
    log("=" * 50)
    
    results = {}
    
    # Extract Zillow
    log("\n[1/2] Extracting Zillow data...")
    try:
        zillow_manifest = extract_zillow_bronze(force_redownload)
        results["zillow"] = {
            "status": "success",
            "configs": list(zillow_manifest.get("configs", {}).keys()),
            "manifest_path": str(BRONZE / "zillow" / "manifest.json")
        }
    except Exception as e:
        log(f"Zillow extraction failed: {e}")
        results["zillow"] = {"status": "error", "error": str(e)}
    
    # Extract FRED
    log("\n[2/2] Extracting FRED data...")
    try:
        fred_manifest = extract_fred_bronze(fred_start, fred_end, force_redownload)
        results["fred"] = {
            "status": "success",
            "series": list(fred_manifest.get("series", {}).keys()),
            "manifest_path": str(BRONZE / "macro" / "manifest.json")
        }
    except Exception as e:
        log(f"FRED extraction failed: {e}")
        results["fred"] = {"status": "error", "error": str(e)}
    
    log("\n" + "=" * 50)
    log("Extraction complete!")
    log(f"Bronze data location: {BRONZE}")
    
    return results


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ETL Extract Module - Bronze Layer")
    parser.add_argument("--force-extract", action="store_true",
                       help="Force re-download even if data exists")
    parser.add_argument("--fred-start", type=str, default=None,
                       help="FRED start date (YYYY-MM-DD)")
    parser.add_argument("--fred-end", type=str, default=None,
                       help="FRED end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    extract_all(
        force_redownload=args.force_extract,
        fred_start=args.fred_start,
        fred_end=args.fred_end
    )


if __name__ == "__main__":
    main()