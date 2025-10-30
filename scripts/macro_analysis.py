#!/usr/bin/env python3

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

    # --------------------------------------------------------------
    # 6. QUICK AFFORDABILITY GAUGES
    # --------------------------------------------------------------
    import math

    # Use forward-filled slow series to avoid N/A in latest month
    price_source = "median_sales_price"
    if "median_sales_price_existing" in df.columns:
        price_source = "median_sales_price_existing"

    if price_source in df.columns and "income_median" in df.columns:
        df = df.with_columns([
            pl.col(price_source).forward_fill().alias("price_ff"),
            pl.col("income_median").forward_fill().alias("income_ff")
        ])

        # 1) Price-to-Income
        df = df.with_columns((pl.col("price_ff") / pl.col("income_ff")).alias("price_to_income"))

        # 2) Monthly payment from mortgage rate and forward-filled price
        def mortgage_payment(price, rate_percent, years=30):
            if price is None or rate_percent is None:
                return None
            r = rate_percent / 100 / 12
            n = years * 12
            if r == 0:
                return price / n
            return price * r * (1 + r) ** n / ((1 + r) ** n - 1)

        df = df.with_columns(
            pl.struct(["price_ff", "mortgage_rate"])  
              .map_elements(lambda s: mortgage_payment(s["price_ff"], s["mortgage_rate"]), return_dtype=pl.Float64)
              .alias("monthly_payment")
        )

        # 3) Payment-to-Income
        df = df.with_columns((pl.col("monthly_payment") / (pl.col("income_ff") / 12)).alias("payment_to_income"))

        # 4) Baseline window AFTER all columns exist
        afford_df = df.filter(pl.col("date") >= pl.datetime(2015, 1, 1))

        # --- P/I Stats ---
        pi_window = afford_df.filter(pl.col("price_to_income").is_not_null())
        pi_mean = float(pi_window["price_to_income"].mean() or 0.0)
        pi_std = float(pi_window["price_to_income"].std() or 1.0)

        df = df.with_columns(((pl.col("price_to_income") - pi_mean) / pi_std).alias("pi_z"))

        # --- Pay/I Stats ---
        payi_window = afford_df.filter(pl.col("payment_to_income").is_not_null())
        payi_mean = float(payi_window["payment_to_income"].mean() or 0.0)
        payi_std = float(payi_window["payment_to_income"].std() or 1.0)

        df = df.with_columns(((pl.col("payment_to_income") - payi_mean) / payi_std).alias("payi_z"))

        # --- Status ---
        df = df.with_columns(
            pl.when(pl.col("pi_z") <= -0.5).then(pl.lit("Below Trend (Green)"))
              .when(pl.col("pi_z").abs() <= 0.5).then(pl.lit("At Trend (Yellow)"))
              .otherwise(pl.lit("Above Trend (Red)"))
              .alias("pi_status")
        )
        df = df.with_columns(
            pl.when(pl.col("payi_z") <= -0.5).then(pl.lit("Below Trend (Green)"))
              .when(pl.col("payi_z").abs() <= 0.5).then(pl.lit("At Trend (Yellow)"))
              .otherwise(pl.lit("Above Trend (Red)"))
              .alias("payi_status")
        )

        df = df.drop(["pi_z", "payi_z"])  # keep clean outputs

    # --- Print (null-safe) ---
    latest = df.tail(1).select([
        "date",
        pl.col("price_to_income"),
        pl.col("payment_to_income"),
        "pi_status",
        "payi_status"
    ]).row(0, named=True)

    def _fmt_float(v, d=2):
        try:
            return f"{float(v):.{d}f}"
        except Exception:
            return "N/A"

    def _fmt_pct(v, d=1):
        try:
            return f"{float(v):.{d}%}"
        except Exception:
            return "N/A"

    print("\n=== AFFORDABILITY GAUGES (latest month) ===")
    print(f"Date               : {latest.get('date')}")
    print(f"Price-to-Income    : {_fmt_float(latest.get('price_to_income'), 2)}    {latest.get('pi_status') or 'N/A'}")
    print(f"Payment-to-Income  : {_fmt_pct(latest.get('payment_to_income'), 1)}    {latest.get('payi_status') or 'N/A'}")
    print("============================================\n")
    
    # --- 4. Save gold ------------------------------------------------------------
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(OUT_PQ))
    print(f"Wrote: {OUT_PQ}")

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

            # Affordability Summary Box
            if "pi_status" in plot_df.columns and "payi_status" in plot_df.columns:
                latest = plot_df.tail(1)
                pi_val = latest["price_to_income"].item()
                payi_val = latest["payment_to_income"].item()
                pi_status = latest["pi_status"].item() or "N/A"
                payi_status = latest["payi_status"].item() or "N/A"

                if pi_val is not None and payi_val is not None:
                    summary_text = (
                        f"House Price/Income: {pi_val:.2f} → {pi_status.split()[0]}\n"
                        f"Mortgage payment/Income: {payi_val:.1%} → {payi_status.split()[0]}"
                    )
                    # Color-code the box
                    pi_color = {"Green": "lightgreen", "Yellow": "wheat", "Red": "lightcoral"}.get(
                        pi_status.split()[0], "gray"
                    )
                    payi_color = {"Green": "lightgreen", "Yellow": "wheat", "Red": "lightcoral"}.get(
                        payi_status.split()[0], "gray"
                    )

                    # Place in top-right
                    ax.text(0.98, 0.98, summary_text,
                            transform=ax.transAxes,
                            fontsize=10, fontweight='bold',
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95, edgecolor="black"))
                    
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

