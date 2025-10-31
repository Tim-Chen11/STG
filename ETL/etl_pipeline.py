#!/usr/bin/env python3
"""
ETL Pipeline Orchestrator
Calls etl_extract and etl_transform modules to run the complete pipeline
"""

import sys
import pathlib
import argparse
import datetime as dt
from typing import Optional

# Add current directory to path to import sibling modules
current_dir = pathlib.Path(__file__).parent
sys.path.insert(0, str(current_dir))

import etl_extract
import etl_transform


def log(msg: str):
    """Log message with timestamp"""
    ts = dt.datetime.now().strftime("%H:%M:%S")
    try:
        print(f"{ts} | {msg}")
    except UnicodeEncodeError:
        msg_safe = msg.encode('ascii', 'replace').decode('ascii')
        print(f"{ts} | {msg_safe}")


def main():
    """Run the complete ETL pipeline"""
    parser = argparse.ArgumentParser(
        description="ETL Pipeline - Orchestrates extract and transform stages"
    )
    
    # Extract arguments
    parser.add_argument("--force-extract", action="store_true",
                       help="Force re-download even if data exists")
    parser.add_argument("--fred-start", type=str, default=None,
                       help="FRED start date (YYYY-MM-DD)")
    parser.add_argument("--fred-end", type=str, default=None,
                       help="FRED end date (YYYY-MM-DD)")
    
    # Pipeline control arguments
    parser.add_argument("--skip-extract", action="store_true",
                       help="Skip extraction stage (use existing bronze data)")
    parser.add_argument("--skip-transform", action="store_true",
                       help="Skip transform stage (extract only)")
    
    args = parser.parse_args()
    
    log("=" * 60)
    log("ETL Pipeline - Complete Data Processing")
    log("=" * 60)
    
    # Stage 1: Extract (Bronze Layer)
    if not args.skip_extract:
        log("\nStage 1: Extraction (Bronze Layer)")
        log("-" * 40)
        try:
            extract_results = etl_extract.extract_all(
                force_redownload=args.force_extract,
                fred_start=args.fred_start,
                fred_end=args.fred_end
            )
            
            # Report results
            if extract_results.get("zillow", {}).get("status") == "success":
                configs = extract_results["zillow"].get("configs", [])
                log(f"✓ Zillow: {len(configs)} datasets extracted")
            else:
                log(f"✗ Zillow: {extract_results.get('zillow', {}).get('error', 'Unknown error')}")
            
            if extract_results.get("fred", {}).get("status") == "success":
                series = extract_results["fred"].get("series", [])
                log(f"✓ FRED: {len(series)} series extracted")
            else:
                log(f"✗ FRED: {extract_results.get('fred', {}).get('error', 'Unknown error')}")
                
        except Exception as e:
            log(f"✗ Extraction failed: {e}")
            if not args.skip_transform:
                log("Stopping pipeline due to extraction failure")
                return 1
    else:
        log("\nSkipping extraction stage (using existing bronze data)")
    
    # Stage 2: Transform (Silver Layer)
    if not args.skip_transform:
        log("\nStage 2: Transformation (Silver Layer)")
        log("-" * 40)
        try:
            # Run transform main function
            etl_transform.main()
            log("✓ Transformation complete")
        except Exception as e:
            log(f"✗ Transformation failed: {e}")
            return 1
    else:
        log("\nSkipping transformation stage")
    
    log("\n" + "=" * 60)
    log("ETL Pipeline Complete!")
    log("=" * 60)
    
    # Report data locations
    data_dir = pathlib.Path("data")
    if (data_dir / "raw").exists():
        log(f"Bronze data: {data_dir / 'raw'}")
    if (data_dir / "silver").exists():
        log(f"Silver data: {data_dir / 'silver'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())