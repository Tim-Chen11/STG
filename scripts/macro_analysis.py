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
FIG_A = GOLD_DIR / "macro_index_regime.png"
FIG_B = GOLD_DIR / "mortgage_vs_price_yoy.png"


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

    # --- 1. Prep: datetime + sort ------------------------------------------------
    if df["date"].dtype == pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Datetime))
    df = df.sort("date")

    # --- 2. Derive YoY columns (Step 2) -----------------------------------------
    price_col = (
        "median_sales_price_existing" if "median_sales_price_existing" in df.columns
        else "median_sales_price" if "median_sales_price" in df.columns
        else None
    )

    new_cols = []
    if "cpi" in df.columns:
        cpi_yoy = yoy(pl.col("cpi")).alias("cpi_yoy")
        new_cols.append(cpi_yoy)
        if "mortgage_rate" in df.columns:
            new_cols.append(
                (pl.col("mortgage_rate") - (yoy(pl.col("cpi")) * 100)).alias("real_mortgage")
            )

    for raw, yoy_name in [
        ("building_permits", "permits_yoy"),
        ("housing_starts",   "starts_yoy"),
        ("income_median",    "income_yoy"),
    ]:
        if raw in df.columns:
            new_cols.append(yoy(pl.col(raw)).alias(yoy_name))

    if price_col:
        new_cols.append(yoy(pl.col(price_col)).alias("natl_price_yoy"))

    if new_cols:
        df = df.with_columns(new_cols)

    # --- 3. Macro Index: ONE TIME ONLY (2018 鈫?latest) ---------------------------
    window_df = df.filter(pl.col("date") >= pl.datetime(2015, 1, 1))

    good = [c for c in ["permits_yoy", "starts_yoy", "natl_price_yoy"] if c in df.columns]
    bad  = [c for c in ["real_mortgage", "unemployment_rate", "cpi_yoy", "median_days_on_market"]
            if c in df.columns]

    z_exprs = []
    z_cols = []  # <-- collect column names here

    def z(col_name: str) -> pl.Expr:
        mu = window_df[col_name].mean()
        sd = window_df[col_name].std()
        return (pl.col(col_name) - pl.lit(mu)) / pl.lit(sd).fill_null(1.0)

    # GOOD variables 鈫?keep +z
    for c in good:
        expr = z(c).alias(f"z_{c}")
        z_exprs.append(expr)
        z_cols.append(f"z_{c}")

    # BAD variables 鈫?flip sign
    for c in bad:
        expr = (-z(c)).alias(f"z_{c}")
        z_exprs.append(expr)
        z_cols.append(f"z_{c}")

    if z_exprs: 
        df = df.with_columns(z_exprs)                # ← z‑scores are now in df

        # --------------------------------------------------------------
        # *** INSERT THE SAFE‑MACRO‑INDEX BLOCK HERE ***
        # --------------------------------------------------------------
        # Count how many of the z‑columns are NOT null in each row
        z_present = pl.sum_horizontal(
            [pl.col(c).is_not_null().cast(pl.Int8) for c in z_cols]
        )
        min_req = len(z_cols) // 2                     # at least 50 % of the series

        df = df.with_columns(
            pl.when(z_present >= min_req)
            .then(pl.sum_horizontal(z_cols) / len(z_cols))
            .otherwise(pl.lit(None))
            .alias("macro_index")
        )

        df = df.with_columns(
            pl.when(pl.col("macro_index").is_null())
            .then(pl.lit(None))
            .otherwise(
                pl.when(pl.col("macro_index") > 0.5).then(pl.lit("Expansion"))
                .when(pl.col("macro_index") < -0.5).then(pl.lit("Correction"))
                .otherwise(pl.lit("Neutral"))
            )
            .alias("macro_regime")
        )

    # --- 4. Save gold ------------------------------------------------------------
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(OUT_PQ))
    print(f"Wrote: {OUT_PQ}")
    print(f"Wrote: {OUT_CSV}")

    # --- Step 4: Charts ----------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Patch

        # Filter plot to same window as z-scores: 2015-01 onward
        plot_df = df.filter(pl.col("date") >= pl.datetime(2015, 1, 1))
        dates = plot_df["date"].to_list()

        # Chart A: Macro Index with regime shading
        if "macro_index" in plot_df.columns and "macro_regime" in plot_df.columns:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(dates, plot_df["macro_index"].to_list(), color="#1f77b4", lw=1.8, label="Macro Index")

            # Shading — null-safe
            regimes = plot_df["macro_regime"].to_list()
            colors = {"Expansion": "#2ca02c", "Neutral": "#7f7f7f", "Correction": "#d62728"}
            start_idx = 0
            for i in range(1, len(regimes)):
                if regimes[i] != regimes[i - 1]:
                    if regimes[i - 1] in colors:
                        ax.axvspan(dates[start_idx], dates[i - 1], color=colors[regimes[i - 1]], alpha=0.15)
                    start_idx = i
            # Final segment
            if start_idx < len(regimes) and regimes[-1] in colors:
                ax.axvspan(dates[start_idx], dates[-1], color=colors[regimes[-1]], alpha=0.15)

            ax.axhline(0.5, color="g", ls="--", lw=1, alpha=0.7)
            ax.axhline(-0.5, color="r", ls="--", lw=1, alpha=0.7)
            ax.set_title("Macro Index with Housing Cycle Regimes (2015 → latest)", fontsize=14)
            ax.set_ylabel("Z-Score Average")
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            fig.autofmt_xdate()

            legend_patches = [Patch(facecolor=c, alpha=0.3, label=r) for r, c in colors.items()]
            ax.legend(handles=legend_patches, loc="upper left")

            plt.tight_layout()
            fig.savefig(FIG_A, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Chart A saved: {FIG_A}")

        # Chart B: Mortgage Rate vs Price YoY
        if "mortgage_rate" in plot_df.columns and "natl_price_yoy" in plot_df.columns:
            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(dates, plot_df["mortgage_rate"].to_list(), color="#9467bd", lw=1.8, label="30-Yr Mortgage Rate (%)")
            ax1.set_ylabel("Mortgage Rate (%)", color="#9467bd")
            ax1.tick_params(axis="y", labelcolor="#9467bd")

            ax2 = ax1.twinx()
            price_yoy_pct = (plot_df["natl_price_yoy"] * 100).to_list()
            ax2.plot(dates, price_yoy_pct, color="#ff7f0e", lw=1.8, label="National Price YoY (%)")
            ax2.set_ylabel("Price YoY (%)", color="#ff7f0e")
            ax2.tick_params(axis="y", labelcolor="#ff7f0e")

            ax1.set_title("Mortgage Rates vs Home Price Growth (2015 → latest)", fontsize=14)
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            fig.autofmt_xdate()

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

            plt.tight_layout()
            fig.savefig(FIG_B, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Chart B saved: {FIG_B}")

    except Exception as e:
        print(f"Charting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    copy_fred_unified()
    main()
