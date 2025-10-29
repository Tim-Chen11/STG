#!/usr/bin/env python3
# etl_full.py — Zillow (HF) + FRED, Bronze→Silver→Gold with QA
from __future__ import annotations

import os, io, re, json, glob, shutil, pathlib
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple

import requests
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download, get_token

# =========================
# Configuration (tweak here)
# =========================
BASE_OUT   = pathlib.Path("data")
BRONZE     = BASE_OUT / "raw"
SILVER     = BASE_OUT / "silver"
GOLD       = BASE_OUT / "gold"
ZILLOW_ID  = "misikoff/zillow-viewer"
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

# Data quality thresholds (lightweight)
MIN_ROWS_PER_ZILLOW_CONFIG = 1
ALLOW_MISSING_RATIO = 0.3  # warn if key columns exceed this missingness


# ==========
# CLI config
# ==========
@dataclass
class ETLConfig:
    start: Optional[str] = None     # YYYY-MM (inclusive)
    end: Optional[str] = None       # YYYY-MM (inclusive)
    region_level: str = "state"     # current gold target
    skip_extract: bool = False      # if True, assume bronze exists
    skip_transform: bool = False    # if True, only extract
    skip_gold: bool = False         # if True, stop at silver

    def window_as_dt(self) -> Tuple[Optional[dt.datetime], Optional[dt.datetime]]:
        s = dt.datetime.strptime(self.start, "%Y-%m") if self.start else None
        e = dt.datetime.strptime(self.end, "%Y-%m") if self.end else None
        return s, e


# ===========
# Utils
# ===========
def log(msg: str):
    ts = dt.datetime.now().strftime("%H:%M:%S")
    try:
        print(f"{ts} | {msg}")
    except UnicodeEncodeError:
        # Fallback for Windows console that doesn't support Unicode
        msg_safe = msg.encode('ascii', 'replace').decode('ascii')
        print(f"{ts} | {msg_safe}")

def ensure_dirs():
    (BRONZE / "zillow").mkdir(parents=True, exist_ok=True)
    (BRONZE / "macro").mkdir(parents=True, exist_ok=True)
    (SILVER / "zillow").mkdir(parents=True, exist_ok=True)
    (SILVER / "macro").mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)

def write_json(path: pathlib.Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def to_snake(s: str) -> str:
    s = re.sub(r"[^\w]+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()

def detect_date_col(cols: List[str]) -> Optional[str]:
    for c in cols:
        cl = c.lower()
        if cl in ("date", "period", "observation_date", "ds", "time"):
            return c
    return None

def parquet_rows(p: pathlib.Path) -> int:
    try:
        return pq.ParquetFile(p).metadata.num_rows
    except Exception:
        return -1

def fred_url(series_id: str, start: Optional[str]=None, end: Optional[str]=None) -> str:
    qs = f"id={series_id}"
    if start: qs += f"&cosd={start}"
    if end:   qs += f"&coed={end}"
    return f"{FRED_CSV_BASE}?{qs}"

def month_floor_expr(e: pl.Expr) -> pl.Expr:
    return e.dt.truncate("1mo")

def parse_month(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m")


# =================
# Bronze — Extract
# =================
def extract_zillow_bronze(force_redownload: bool = False) -> dict:
    """
    Snapshot all parquet shards for misikoff/zillow-viewer@~parquet and copy into
    data/raw/zillow/<config>/ as-is. Write a manifest with columns, rowcounts, date ranges.
    
    Args:
        force_redownload: If True, always download. If False, skip if data exists.
    """
    z_base = BRONZE / "zillow"
    manifest_path = z_base / "manifest.json"
    
    # Check if data already exists
    if not force_redownload and manifest_path.exists():
        log("✅ Zillow bronze data already exists. Use --force-extract to re-download.")
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    log("🔐 Checking Hugging Face token…")
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        log("⚠️  No HF token detected. Run `huggingface-cli login` or set HF_TOKEN.")

    log(f"⬇️  Snapshot {ZILLOW_ID}@{ZILLOW_REV} (parquet shards)…")
    cache_dir = snapshot_download(
        repo_id=ZILLOW_ID,
        repo_type="dataset",
        revision=ZILLOW_REV,
        allow_patterns=["*/train/*.parquet"],
        local_dir_use_symlinks=False,
        max_workers=4
    )
    cache = pathlib.Path(cache_dir)

    configs = []
    for p in cache.iterdir():
        if p.is_dir() and (p / "train").exists() and list((p / "train").glob("*.parquet")):
            configs.append(p.name)
    configs = sorted(configs)
    log(f"📦 Found Zillow configs: {configs}")

    man = {
        "dataset": f"{ZILLOW_ID}@{ZILLOW_REV}",
        "extracted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "configs": {}
    }

    for cfg in configs:
        src_dir = (cache / cfg / "train")
        dst_dir = z_base / cfg
        dst_dir.mkdir(parents=True, exist_ok=True)

        files_meta = []
        total = 0
        sample_file: Optional[pathlib.Path] = None

        for src in sorted(src_dir.glob("*.parquet")):
            if sample_file is None:
                sample_file = src
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            rows = parquet_rows(dst)
            total += max(rows, 0)
            files_meta.append({
                "file": dst.as_posix(),
                "rows": rows,
                "bytes": dst.stat().st_size
            })

        # columns / date range from sample + lazy scan
        cols = []
        date_col = None
        min_date = max_date = None
        if sample_file is not None:
            try:
                df0 = pl.read_parquet(sample_file.as_posix(), n_rows=0)
                cols = df0.columns
                date_col = detect_date_col(cols)
            except Exception:
                pass
            if date_col:
                try:
                    lf = pl.scan_parquet((dst_dir / "*.parquet").as_posix())
                    rng = (
                        lf.select([
                            pl.col(date_col).cast(pl.Utf8).str.strptime(pl.Datetime, strict=False).alias("_d")
                        ])
                        .select([pl.col("_d").min().alias("min"), pl.col("_d").max().alias("max")])
                        .collect()
                    )
                    def fmt(x):
                        return None if x is None else pl.Series([x]).dt.strftime("%Y-%m-%d")[0]
                    min_date = fmt(rng["min"][0])
                    max_date = fmt(rng["max"][0])
                except Exception:
                    pass

        man["configs"][cfg] = {
            "dest_dir": (z_base / cfg).as_posix(),
            "num_files": len(files_meta),
            "total_rows_sum": None if total <= 0 else total,
            "columns_example": cols,
            "date_column": date_col,
            "min_date": min_date,
            "max_date": max_date,
            "files": files_meta
        }

        if len(files_meta) < MIN_ROWS_PER_ZILLOW_CONFIG:
            log(f"⚠️  {cfg}: appears empty; check your HF auth/rate limits.")

    write_json(z_base / "manifest.json", man)
    log(f"📝 Zillow bronze manifest → {z_base / 'manifest.json'}")
    return man


def extract_fred_bronze(start: Optional[str], end: Optional[str], force_redownload: bool = False) -> dict:
    """
    Download validated FRED series via public CSV endpoint (no API key),
    save CSV + Parquet into data/raw/macro, with a manifest.
    
    Args:
        force_redownload: If True, always download. If False, skip if data exists.
    """
    m_base = BRONZE / "macro"; m_base.mkdir(parents=True, exist_ok=True)
    manifest_path = m_base / "manifest.json"
    
    # Check if data already exists
    if not force_redownload and manifest_path.exists():
        log("✅ FRED bronze data already exists. Use --force-extract to re-download.")
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    man = {
        "source": "FRED (fredgraph.csv)",
        "downloaded_at": dt.datetime.utcnow().isoformat() + "Z",
        "series": {}
    }
    for name, sid in FRED_SERIES.items():
        url = fred_url(sid, start, end)
        log(f"⬇️  FRED {name} ({sid}) → {url}")
        csv_path = (m_base / f"{name}.csv")
        pq_path  = (m_base / f"{name}.parquet")
        try:
            r = requests.get(url, timeout=45)
            if r.status_code != 200:
                log(f"  ❌ HTTP {r.status_code} — skip {sid}")
                man["series"][name] = {"id": sid, "status": f"http_{r.status_code}", "url": url}
                continue
            df = pd.read_csv(io.StringIO(r.text))
            if "observation_date" not in df.columns or sid not in df.columns:
                raise ValueError(f"Unexpected columns for {sid}: {df.columns.tolist()}")
            df.rename(columns={"observation_date":"date", sid:name}, inplace=True)
            df.to_csv(csv_path, index=False)
            df.to_parquet(pq_path, index=False)
            d0 = pd.to_datetime(df["date"], errors="coerce")
            man["series"][name] = {
                "id": sid, "status": "ok", "url": url,
                "rows": int(df.shape[0]), "min_date": d0.min().strftime("%Y-%m-%d") if len(d0) else None,
                "max_date": d0.max().strftime("%Y-%m-%d") if len(d0) else None,
                "csv": csv_path.as_posix(), "parquet": pq_path.as_posix()
            }
            log(f"  ✅ Saved {csv_path.name} | {pq_path.name}")
        except Exception as e:
            log(f"  ❌ Failed {sid}: {e}")
            man["series"][name] = {"id": sid, "status": "error", "error": str(e), "url": url}
    write_json(m_base / "manifest.json", man)
    log(f"📝 FRED bronze manifest → {m_base / 'manifest.json'}")
    return man


# =================
# Silver — Standardize
# =================
def _read_zillow_config_raw(config_name: str) -> pl.LazyFrame:
    path_glob = (BRONZE / "zillow" / config_name / "*.parquet").as_posix()
    return pl.scan_parquet(path_glob)

def _normalize_zillow_columns(lf: pl.LazyFrame) -> Tuple[pl.LazyFrame, Dict[str, str]]:
    """Lower/snake case; return mapping {old:new} to keep track."""
    # pull schema
    cols = lf.collect_schema().names()
    mapping = {c: to_snake(c) for c in cols}
    lf2 = lf.rename(mapping)
    return lf2, mapping

def _pick_state_col(cols: List[str]) -> Optional[str]:
    for c in cols:
        if c.lower() == "state":
            return c
    # fallback: sometimes 'state_name'
    for c in cols:
        if c.lower() in ("state_name", "state_abbrev", "state_code"):
            return c
    return None

def _pick_region_cols(cols: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    rid = None; rname = None; rtype = None
    for c in cols:
        lc = c.lower()
        if rid is None and lc in ("region_id", "regionid", "regioncode"):
            rid = c
        if rname is None and lc in ("region", "region_name", "city", "county", "metro"):
            rname = c
        if rtype is None and lc in ("region_type", "regiontype"):
            rtype = c
    return rid, rname, rtype

def _coalesce_first(cols: List[str], options: List[str]) -> Optional[str]:
    lcset = {c.lower(): c for c in cols}
    for want in options:
        if want.lower() in lcset:
            return lcset[want.lower()]
    return None

def _normalize_date(lf: pl.LazyFrame, date_cols: List[str]) -> pl.LazyFrame:
    for cand in date_cols:
        if cand in lf.collect_schema().names():
            return (
                lf.with_columns(
                    pl.col(cand)
                      .cast(pl.Utf8)
                      .str.strptime(pl.Datetime, strict=False)
                      .alias("_d")
                )
                .with_columns(month_floor_expr(pl.col("_d")).alias("date"))
                .drop([cand, "_d"])
            )
    # as a last resort, try any 'date'-ish column
    cols = lf.collect_schema().names()
    pick = detect_date_col(cols)
    if pick:
        return _normalize_date(lf, [pick])
    raise ValueError("No date-like column found for Zillow subset.")

def standardize_zillow_subset(config_name: str) -> pl.DataFrame:
    """
    Return a standardized, month-grained DF with columns:
    - state (region_key for gold)
    - date
    - metrics (best-effort heuristic per subset)
    """
    lf = _read_zillow_config_raw(config_name)
    lf, mapping = _normalize_zillow_columns(lf)
    cols = lf.collect_schema().names()

    # date
    lf = _normalize_date(lf, ["date", "period"])

    # pick region fields
    rid, rname, rtype = _pick_region_cols(cols)
    state_col = _pick_state_col(cols)

    # Use state as region_key (gold target); if missing, fall back to region (string)
    region_key = state_col or rname or rid
    if region_key is None:
        raise ValueError(f"[{config_name}] cannot identify region key among {cols}")

    # Best-effort metric detection per subset
    lower_cols = [c.lower() for c in cols]

    metrics: List[pl.Expr] = []
    renames: Dict[str, str] = {}

    def add_metric(alias: str, candidates: List[str]):
        c = _coalesce_first(cols, candidates)
        if c:
            metrics.append(pl.col(c).cast(pl.Float64).alias(alias))
            renames[c] = alias

    if config_name == "home_values":
        add_metric("home_value", [
            "zhvi", "zhvi_median", "value", "home_value", "mid_tier_zhvi"
        ])

    elif config_name == "rentals":
        add_metric("rent_value", [
            "rent", "zori", "rent_smoothed", "rent_value", "rent__smoothed_seasonally_adjusted"
        ])

    elif config_name == "for_sale_listings":
        add_metric("median_list_price", ["median_listing_price", "median_list_price", "list_price_median"])
        add_metric("new_listings", ["new_listings", "new_listings_smoothed"])
        add_metric("active_listings", ["active_listings", "inventory", "num_of_listings"])

    elif config_name == "days_on_market":
        add_metric("median_dom", [
            "median_days_on_market", "median_days_on_pending", "median_dom",
            "median_days_on_pending_smoothed"
        ])
        add_metric("price_cut_ratio", [
            "percent_listings_price_cut", "percent_listings_price_cut_smoothed",
            "price_cut_share", "pct_listings_price_cut"
        ])

    elif config_name in ("sales", "new_construction"):
        add_metric("median_sale_price", ["median_sale_price"])
        add_metric("median_sale_price_per_sqft", ["median_sale_price_per_sqft"])
        add_metric("sales_count", ["sales_count"])

    elif config_name == "home_values_forecasts":
        # keep raw forecast fields if present (optional)
        add_metric("home_yoy_forecast_raw", ["year_over_year___raw", "year_over_year"])
        add_metric("home_yoy_forecast_smooth", ["year_over_year___smoothed", "year_over_year_smoothed"])

    # Always retain region key and date
    selected = [pl.col(region_key).alias("state"), pl.col("date")]
    selected += metrics if metrics else []
    df = (
        lf.select(selected)
          .with_columns(pl.col("state").cast(pl.Utf8).str.strip())
          .collect()
    )
    return df


def silver_zillow_all(config: ETLConfig) -> Dict[str, str]:
    """
    Writes standardized silver tables per Zillow subset into data/silver/zillow/<subset>.parquet
    applies date window if provided.
    """
    z_configs = discover_zillow_configs()
    outputs: Dict[str, str] = {}
    s_dt, e_dt = config.window_as_dt()

    for cfg in z_configs:
        try:
            log(f"🔧 Standardizing Zillow subset: {cfg}")
            df = standardize_zillow_subset(cfg)

            # window
            if s_dt:
                df = df.filter(pl.col("date") >= pl.lit(s_dt))
            if e_dt:
                df = df.filter(pl.col("date") <= pl.lit(e_dt))

            # basic QA
            n = df.height
            if n == 0:
                log(f"  ⚠️  {cfg}: empty after window; writing anyway")
            if "state" in df.columns:
                miss = df["state"].null_count() / max(n,1)
                if miss > ALLOW_MISSING_RATIO:
                    log(f"  ⚠️  {cfg}: >{ALLOW_MISSING_RATIO:.0%} missing in state")

            out = (SILVER / "zillow" / f"{cfg}_std.parquet")
            out.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out.as_posix())
            outputs[cfg] = out.as_posix()
            log(f"  ✅ {cfg} → {out}")
        except Exception as e:
            log(f"  ❌ {cfg}: {e}")

    # manifest
    man = {
        "written_at": dt.datetime.utcnow().isoformat() + "Z",
        "outputs": outputs
    }
    write_json(SILVER / "zillow" / "manifest.json", man)
    return outputs


def discover_zillow_configs() -> List[str]:
    # based on bronze/zillow/<subset> presence
    base = (BRONZE / "zillow")
    configs = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and list(d.glob("*.parquet")):
            configs.append(d.name)
        elif d.is_dir() and (d / "train").exists() and list((d / "train").glob("*.parquet")):
            configs.append(d.name)
    # common expected names, keep order if present
    order = [
        "home_values", "rentals", "for_sale_listings", "days_on_market",
        "sales", "new_construction", "home_values_forecasts"
    ]
    configs = [c for c in order if c in configs] + [c for c in configs if c not in order]
    return configs


def silver_fred_all(config: ETLConfig) -> str:
    """
    Load FRED CSVs from bronze, standardize to monthly DataFrame with columns:
    date + each series (monthly-mean for weekly; ffill for quarterly/annual).
    """
    s_dt, e_dt = config.window_as_dt()
    frames: List[pd.DataFrame] = []
    for name in FRED_SERIES.keys():
        path = (BRONZE / "macro" / f"{name}.csv")
        if not path.exists():
            log(f"  ⚠️  FRED series missing in bronze: {name} (skip)")
            continue
        df = pd.read_csv(path)
        if "date" not in df.columns or name not in df.columns:
            log(f"  ⚠️  Unexpected FRED file shape for {name}: {df.columns.tolist()}")
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # frequency handling
        # heuristic by name:
        if name in ("mortgage_rate",):      # weekly → monthly mean
            df = (df.set_index("date")
                    .resample("MS")
                    .mean()
                    .reset_index())
        elif name in ("median_sales_price",):  # quarterly → monthly ffill
            df = (df.set_index("date")
                    .resample("QS")  # quarter start
                    .mean()
                    .resample("MS")
                    .ffill()
                    .reset_index())
        elif name in ("income_median",):   # annual → monthly ffill
            df = (df.set_index("date")
                    .resample("YS")
                    .mean()
                    .resample("MS")
                    .ffill()
                    .reset_index())
        else:
            # monthly already
            pass

        if s_dt:
            df = df[df["date"] >= s_dt]
        if e_dt:
            df = df[df["date"] <= e_dt]
        frames.append(df[["date", name]])

    # merge on date
    if not frames:
        log("  ⚠️  No FRED frames assembled.")
        return ""
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="date", how="outer")
    merged = merged.sort_values("date")
    out = (SILVER / "macro" / "fred_monthly.parquet")
    pl.from_pandas(merged).write_parquet(out.as_posix())
    log(f"  ✅ Silver macro → {out}")
    return out.as_posix()


# =================
# Gold — Conformed panel (state × month)
# =================
def yoy_by_group(df: pl.DataFrame, group_col: str, value_col: str, out_name: str) -> pl.DataFrame:
    return (
        df.sort(["date"])
          .group_by(group_col, maintain_order=True)
          .agg([pl.col(value_col), pl.col("date")])
          .explode([value_col, "date"])
          .with_columns(((pl.col(value_col) / pl.col(value_col).shift(12)) - 1.0).alias(out_name))
          .select([group_col, "date", out_name])
    )

def load_silver_if_exists(name: str) -> Optional[pl.DataFrame]:
    p = (SILVER / "zillow" / f"{name}_std.parquet")
    return pl.read_parquet(p.as_posix()) if p.exists() else None

def build_gold_state_panel(config: ETLConfig) -> str:
    """
    Join standardized Zillow silver tables on state+date, then join monthly FRED.
    Compute basic derived fields and write a gold parquet & csv.
    """
    s_dt, e_dt = config.window_as_dt()

    home = load_silver_if_exists("home_values")
    rent = load_silver_if_exists("rentals")
    sale = load_silver_if_exists("for_sale_listings")
    dom  = load_silver_if_exists("days_on_market")

    if home is None:
        raise RuntimeError("Gold requires silver/home_values_std.parquet — not found.")

    # Apply window here as well (defensive)
    def w(df: Optional[pl.DataFrame]) -> Optional[pl.DataFrame]:
        if df is None: return None
        if s_dt: df = df.filter(pl.col("date") >= pl.lit(s_dt))
        if e_dt: df = df.filter(pl.col("date") <= pl.lit(e_dt))
        return df

    home = w(home)
    rent = w(rent)
    sale = w(sale)
    dom  = w(dom)

    # Start from home frame
    merged = home

    # Join rent
    if rent is not None and "rent_value" in rent.columns:
        merged = merged.join(
            rent.select(["state","date","rent_value"]),
            on=["state","date"], how="outer"
        )

    # Join for-sale listings
    if sale is not None:
        keep = [c for c in ["state","date","median_list_price","active_listings","new_listings"] if c in sale.columns]
        if len(keep) > 2:
            merged = merged.join(sale.select(keep), on=["state","date"], how="left")

    # Join DOM / price cut
    if dom is not None:
        keep = [c for c in ["state","date","median_dom","price_cut_ratio"] if c in dom.columns]
        if len(keep) > 2:
            merged = merged.join(dom.select(keep), on=["state","date"], how="left")

    # Derived: rent-to-price
    if set(["rent_value","home_value"]).issubset(set(merged.columns)):
        merged = merged.with_columns(((pl.col("rent_value") * 12.0) / pl.col("home_value")).alias("rent_to_price_ratio"))

    # YoY features
    if "home_value" in merged.columns:
        merged = merged.join(yoy_by_group(merged.select(["state","date","home_value"]), "state", "home_value", "home_yoy"),
                             on=["state","date"], how="left")
    if "rent_value" in merged.columns:
        merged = merged.join(yoy_by_group(merged.select(["state","date","rent_value"]), "state", "rent_value", "rent_yoy"),
                             on=["state","date"], how="left")

    # Join macro monthly on date
    fred_path = (SILVER / "macro" / "fred_monthly.parquet")
    if fred_path.exists():
        fred = pl.read_parquet(fred_path.as_posix())
        merged = merged.join(fred, on="date", how="left")

        # Real growth adjuster (if CPI YoY available)
        # Compute CPI YoY quickly:
        if "cpi" in fred.columns:
            cpi_yoy = (
                fred.select([pl.col("date"), (pl.col("cpi") / pl.col("cpi").shift(12) - 1.0).alias("cpi_yoy")])
            )
            merged = merged.join(cpi_yoy, on="date", how="left")
            if "home_yoy" in merged.columns:
                merged = merged.with_columns((pl.col("home_yoy") - pl.col("cpi_yoy")).alias("home_yoy_real"))

    # Order columns
    head = [c for c in [
        "state","date",
        "home_value","rent_value","home_yoy","rent_yoy","home_yoy_real",
        "rent_to_price_ratio",
        "median_list_price","active_listings","new_listings",
        "median_dom","price_cut_ratio",
        "mortgage_rate","unemployment_rate","cpi","building_permits","housing_starts","median_sales_price","income_median","cpi_yoy"
    ] if c in merged.columns]
    merged = merged.select(head + [c for c in merged.columns if c not in head]).sort(["state","date"])

    # QA: rows / missingness
    n = merged.height
    miss_rate = None
    if "home_value" in merged.columns and n > 0:
        miss_rate = float(merged["home_value"].null_count()) / n
        if miss_rate > ALLOW_MISSING_RATIO:
            log(f"⚠️  Gold: home_value missingness {miss_rate:.1%} > {ALLOW_MISSING_RATIO:.0%} threshold")

    # Write outputs
    out_pq = (GOLD / "housing_panel_state.parquet")
    out_csv = (GOLD / "housing_panel_state.csv")
    merged.write_parquet(out_pq.as_posix())
    merged.write_csv(out_csv.as_posix())
    log(f"🏆 Gold panel → {out_pq} ({n:,} rows)")

    # Gold manifest
    man = {
        "written_at": dt.datetime.utcnow().isoformat() + "Z",
        "rows": n,
        "columns": merged.columns,
        "start": merged["date"].min().strftime("%Y-%m-%d") if n else None,
        "end": merged["date"].max().strftime("%Y-%m-%d") if n else None,
        "missing_home_value_ratio": miss_rate
    }
    write_json(GOLD / "manifest.json", man)
    return out_pq.as_posix()


# ===========
# Main runner
# ===========
def parse_args() -> Tuple[ETLConfig, bool]:
    import argparse
    p = argparse.ArgumentParser(description="Zillow + FRED ETL (Bronze→Silver→Gold)")
    p.add_argument("--start", type=str, default=None, help="Start month YYYY-MM (inclusive)")
    p.add_argument("--end", type=str, default=None, help="End month YYYY-MM (inclusive)")
    p.add_argument("--region-level", type=str, default="state", choices=["state"], help="Gold panel grain")
    p.add_argument("--skip-extract", action="store_true", help="Skip Bronze (use existing raw)")
    p.add_argument("--skip-transform", action="store_true", help="Skip Silver")
    p.add_argument("--skip-gold", action="store_true", help="Skip Gold panel")
    p.add_argument("--force-extract", action="store_true", help="Force re-download even if data exists")
    a = p.parse_args()
    config = ETLConfig(start=a.start, end=a.end, region_level=a.region_level,
                      skip_extract=a.skip_extract, skip_transform=a.skip_transform, skip_gold=a.skip_gold)
    return config, a.force_extract


def main():
    ensure_dirs()
    cfg, force_extract = parse_args()
    log(f"Config: {asdict(cfg)}")
    if force_extract:
        log("🔄 Force extract enabled - will re-download all data")

    # Bronze
    if not cfg.skip_extract:
        extract_zillow_bronze(force_redownload=force_extract)
        extract_fred_bronze(cfg.start, cfg.end, force_redownload=force_extract)
    else:
        log("⏩ Skipping Extract (Bronze).")

    # Silver
    if not cfg.skip_transform:
        silver_zillow_all(cfg)
        silver_fred_all(cfg)
    else:
        log("⏩ Skipping Transform (Silver).")

    # Gold
    if not cfg.skip_gold:
        build_gold_state_panel(cfg)
    else:
        log("⏩ Skipping Gold.")

    log("✅ ETL complete.")

if __name__ == "__main__":
    main()
