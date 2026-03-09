"""
Script 04: Exploratory Analysis — Figures and Tables

Reads the analysis-ready dataset (with marker scores + surprise) and generates:

Figures:
  1. Marker rate time trends by field
  2. LLM flag rate over time (aggregate + by field)
  3. Top 30 subfields by LLM marker rate
  4. Author count vs LLM usage
  5. Institutional / country diversity vs LLM usage
  6. Pre/post era distribution shifts
  7. LLM usage vs citation surprise (quintile plot)
  8. Surprise by LLM marker quartile per field
  9. Surprise time series: LLM-flagged vs non-flagged
  10. Coefficient plot from OLS regressions
  11. Field-level scatter: Δmarker rate vs Δsurprise

Tables:
  - field_summary.csv
  - era_summary.csv
  - llm_era_summary.csv
  - subfield_llm_rates.csv
  - surprise_regression_table.csv
  - field_delta_table.csv

Output directory: output/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import DATA_DIR, OUTPUT_DIR, FIELDS, FIELD_SHORT


def load_data():
    """Load the analysis-ready dataset."""
    path = os.path.join(DATA_DIR, "analysis_ready.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No analysis-ready data at {path}. Run 01-03 first.")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    return df


def _find_month_index(df, year_month_prefix):
    months = sorted(df["year_month"].unique())
    for i, m in enumerate(months):
        if str(m).startswith(year_month_prefix):
            return i
    return 0


def _set_month_xticks(ax, months):
    tick_positions = list(range(0, len(months), 6))
    tick_labels = [months[i] for i in tick_positions if i < len(months)]
    ax.set_xticks(tick_positions[:len(tick_labels)])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)


def _add_era_lines(ax, df):
    """Add ChatGPT and GPT-4o vertical lines."""
    ax.axvline(x=_find_month_index(df, "2022-12"), color="gray",
               linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=_find_month_index(df, "2024-05"), color="red",
               linestyle="--", alpha=0.5, linewidth=1)


# ═══════════════════════════════════════════════════════════════════════════

def fig1_marker_rate_trends(df):
    """Figure 1: Marker word rate over time by field."""
    print("  [1/11] Marker rate time trends...")
    has_abs = df[df["has_abstract"] == 1]

    fig, ax = plt.subplots(figsize=(14, 7))
    months = sorted(has_abs["year_month"].dropna().unique())

    for field in sorted(has_abs["field"].unique()):
        subset = has_abs[has_abs["field"] == field]
        monthly = subset.groupby("year_month")["marker_rate"].mean()
        # Align to month index
        vals = [monthly.get(m, np.nan) for m in months]
        label = FIELD_SHORT.get(field, field)
        ax.plot(range(len(months)), vals, alpha=0.7, label=label, linewidth=1.2)

    _add_era_lines(ax, has_abs)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Marker Word Rate (per word)")
    ax.set_title("LLM Marker Word Rate Over Time by Field")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    _set_month_xticks(ax, months)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig01_marker_rate_trends.png"), dpi=200)
    plt.close()


def fig2_llm_flag_over_time(df):
    """Figure 2: LLM flag rate over time."""
    print("  [2/11] LLM flag rate over time...")
    has_abs = df[df["has_abstract"] == 1]
    months = sorted(has_abs["year_month"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Aggregate
    monthly_agg = has_abs.groupby("year_month")["llm_flag"].mean()
    vals = [monthly_agg.get(m, np.nan) for m in months]
    axes[0].plot(range(len(months)), vals, color="steelblue", linewidth=2)
    _add_era_lines(axes[0], has_abs)
    axes[0].set_title("Panel A: Share of Papers Flagged as LLM-Assisted (All Fields)")
    axes[0].set_ylabel("Share flagged (top 25% marker rate)")
    _set_month_xticks(axes[0], months)

    # Panel B: By field
    for field in sorted(has_abs["field"].unique()):
        subset = has_abs[has_abs["field"] == field]
        monthly_f = subset.groupby("year_month")["llm_flag"].mean()
        vals = [monthly_f.get(m, np.nan) for m in months]
        axes[1].plot(range(len(months)), vals, alpha=0.7,
                     label=FIELD_SHORT.get(field, field), linewidth=1)
    _add_era_lines(axes[1], has_abs)
    axes[1].set_title("Panel B: LLM Flag Rate by Field")
    axes[1].legend(fontsize=6, ncol=2, loc="upper left")
    _set_month_xticks(axes[1], months)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig02_llm_flag_over_time.png"), dpi=200)
    plt.close()


def fig3_subfield_rates(df):
    """Figure 3: Top 30 subfields by LLM marker rate."""
    print("  [3/11] Subfield LLM usage rates...")
    has_abs = df[df["has_abstract"] == 1]
    post = has_abs[has_abs["date"] >= "2023-06-01"]

    subfield_rates = (post.groupby("subfield")
                      .agg(mean_marker_rate=("marker_rate", "mean"),
                           llm_pct=("llm_flag", "mean"),
                           n_papers=("paper_id", "count"),
                           field=("field", "first"))
                      .reset_index())
    subfield_rates = subfield_rates[subfield_rates["n_papers"] >= 20]
    top30 = subfield_rates.nlargest(30, "mean_marker_rate")

    fig, ax = plt.subplots(figsize=(10, 10))
    field_colors = {}
    cmap = plt.cm.tab20
    unique_fields = sorted(top30["field"].unique())
    for i, f in enumerate(unique_fields):
        field_colors[f] = cmap(i / max(len(unique_fields) - 1, 1))

    colors = [field_colors[row["field"]] for _, row in top30.iterrows()]
    y_pos = range(len(top30))
    ax.barh(y_pos, top30["mean_marker_rate"].values, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top30["subfield"].values, fontsize=8)
    ax.set_xlabel("Mean Marker Word Rate")
    ax.set_title("Top 30 Subfields by LLM Marker Word Rate (Post-Jun 2023)")
    ax.invert_yaxis()

    legend_elements = [Patch(facecolor=field_colors[f], label=FIELD_SHORT.get(f, f))
                       for f in unique_fields]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig03_subfield_rates.png"), dpi=200)
    plt.close()

    # Save full table
    subfield_rates.sort_values("mean_marker_rate", ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, "subfield_llm_rates.csv"), index=False)


def fig4_authors_vs_llm(df):
    """Figure 4: Author count vs LLM usage."""
    print("  [4/11] Author count vs LLM usage...")
    has_abs = df[(df["has_abstract"] == 1) & (df["date"] >= "2023-06-01")].copy()

    bins = [0, 1, 2, 3, 5, 10, 50]
    labels = ["1", "2", "3", "4-5", "6-10", "11+"]
    has_abs["author_bin"] = pd.cut(has_abs["num_authors"], bins=bins, labels=labels, right=True)

    rates = has_abs.groupby("author_bin", observed=True).agg(
        mean_rate=("marker_rate", "mean"),
        llm_pct=("llm_flag", "mean"),
        n=("paper_id", "count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(rates["author_bin"].astype(str), rates["mean_rate"],
                color="steelblue", alpha=0.8)
    axes[0].set_xlabel("Number of Authors")
    axes[0].set_ylabel("Mean Marker Word Rate")
    axes[0].set_title("Panel A: Marker Rate by Author Count")

    axes[1].bar(rates["author_bin"].astype(str), rates["llm_pct"] * 100,
                color="coral", alpha=0.8)
    axes[1].set_xlabel("Number of Authors")
    axes[1].set_ylabel("% Papers Flagged as LLM-Assisted")
    axes[1].set_title("Panel B: LLM Flag Rate by Author Count")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig04_authors_vs_llm.png"), dpi=200)
    plt.close()


def fig5_institutions_countries(df):
    """Figure 5: Institutional / country diversity vs LLM usage."""
    print("  [5/11] Institutional diversity vs LLM usage...")
    has_abs = df[(df["has_abstract"] == 1) & (df["date"] >= "2023-06-01")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Institutions
    has_abs["inst_bin"] = pd.cut(
        has_abs["institutions_distinct_count"],
        bins=[0, 1, 2, 3, 5, 100], labels=["1", "2", "3", "4-5", "6+"], right=True)
    inst = has_abs.groupby("inst_bin", observed=True)["marker_rate"].mean().reset_index()
    axes[0].bar(inst["inst_bin"].astype(str), inst["marker_rate"],
                color="mediumpurple", alpha=0.8)
    axes[0].set_xlabel("Distinct Institutions")
    axes[0].set_ylabel("Mean Marker Word Rate")
    axes[0].set_title("Panel A: Marker Rate by Institutional Diversity")

    # Countries
    has_abs["country_bin"] = pd.cut(
        has_abs["countries_distinct_count"],
        bins=[0, 1, 2, 3, 100], labels=["1", "2", "3", "4+"], right=True)
    country = has_abs.groupby("country_bin", observed=True)["marker_rate"].mean().reset_index()
    axes[1].bar(country["country_bin"].astype(str), country["marker_rate"],
                color="seagreen", alpha=0.8)
    axes[1].set_xlabel("Distinct Countries")
    axes[1].set_ylabel("Mean Marker Word Rate")
    axes[1].set_title("Panel B: Marker Rate by Country Diversity")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig05_institutions_countries.png"), dpi=200)
    plt.close()


def fig6_era_comparison(df):
    """Figure 6: Pre/post era distribution shifts."""
    print("  [6/11] Pre/post era comparison...")
    has_abs = df[df["has_abstract"] == 1].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Distribution of marker rates by era
    for era, color in [("Pre-ChatGPT", "steelblue"),
                        ("Post-ChatGPT", "coral"),
                        ("Post-4o", "firebrick")]:
        subset = has_abs[has_abs["era"] == era]["marker_rate"]
        if len(subset) > 0:
            axes[0].hist(subset, bins=50, alpha=0.5, color=color,
                        label=era, density=True, range=(0, 0.05))
    axes[0].set_xlabel("Marker Word Rate")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Panel A: Distribution of Marker Rates by Era")
    axes[0].legend()

    # Panel B: Mean marker rate by field × era
    era_field = has_abs.groupby(["field", "era"])["marker_rate"].mean().unstack()
    era_order = ["Pre-ChatGPT", "Post-ChatGPT", "Post-4o"]
    era_field = era_field[[c for c in era_order if c in era_field.columns]]

    field_labels = [FIELD_SHORT.get(f, f) for f in era_field.index]
    x = np.arange(len(field_labels))
    width = 0.25
    colors_era = ["steelblue", "coral", "firebrick"]

    for i, (col, c) in enumerate(zip(era_field.columns, colors_era)):
        axes[1].bar(x + i * width, era_field[col].values, width,
                    label=col, color=c, alpha=0.8)

    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(field_labels, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Mean Marker Word Rate")
    axes[1].set_title("Panel B: Marker Rate by Field and Era")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig06_era_comparison.png"), dpi=200)
    plt.close()


def fig7_surprise_quintiles(df):
    """Figure 7: LLM usage vs citation surprise — quintile plot."""
    print("  [7/11] LLM usage vs citation surprise (quintiles)...")
    valid = df[(df["has_abstract"] == 1) & (df["surprise_kl"].notna()) &
               (df["surprise_kl"] > 0)].copy()
    valid["log_surprise"] = np.log(valid["surprise_kl"] + 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Quintile bars
    try:
        valid["marker_quintile"] = pd.qcut(
            valid["marker_rate"], 5,
            labels=["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"],
            duplicates="drop")
    except ValueError:
        valid["marker_quintile"] = pd.qcut(
            valid["marker_rate"], 4,
            labels=["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"],
            duplicates="drop")

    qm = valid.groupby("marker_quintile", observed=True).agg(
        mean_surprise=("surprise_kl", "mean"),
        mean_log=("log_surprise", "mean"),
        n=("paper_id", "count"),
    ).reset_index()

    axes[0].bar(qm["marker_quintile"].astype(str), qm["mean_surprise"],
                color="steelblue", alpha=0.8)
    axes[0].set_xlabel("Marker Word Rate Quintile")
    axes[0].set_ylabel("Mean Citation Surprise (KL)")
    axes[0].set_title("Panel A: Citation Surprise by LLM Marker Quintile")

    # Panel B: By era
    for era, color in [("Pre-ChatGPT", "steelblue"),
                        ("Post-ChatGPT", "coral"),
                        ("Post-4o", "firebrick")]:
        subset = valid[valid["era"] == era]
        if len(subset) > 100:
            eq = subset.groupby("marker_quintile", observed=True)["surprise_kl"].mean()
            axes[1].plot(eq.index.astype(str), eq.values,
                        marker="o", label=era, color=color, linewidth=2)

    axes[1].set_xlabel("Marker Word Rate Quintile")
    axes[1].set_ylabel("Mean Citation Surprise (KL)")
    axes[1].set_title("Panel B: Surprise × LLM Marker Quintile by Era")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig07_surprise_quintiles.png"), dpi=200)
    plt.close()


def fig8_surprise_by_field(df):
    """Figure 8: Surprise by LLM marker quartile per field."""
    print("  [8/11] Surprise by marker quartile per field...")
    valid = df[(df["has_abstract"] == 1) & (df["surprise_kl"].notna()) &
               (df["surprise_kl"] > 0)].copy()

    fields_sorted = sorted(valid["field"].unique())
    n_fields = len(fields_sorted)
    ncols = min(4, n_fields)
    nrows = (n_fields + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_fields == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, field in enumerate(fields_sorted):
        ax = axes[idx]
        subset = valid[valid["field"] == field].copy()
        if len(subset) < 50:
            ax.set_title(FIELD_SHORT.get(field, field) + " (n<50)")
            continue
        try:
            subset["mq"] = pd.qcut(subset["marker_rate"], 4,
                                    labels=["Q1", "Q2", "Q3", "Q4"],
                                    duplicates="drop")
            qm = subset.groupby("mq", observed=True)["surprise_kl"].mean()
            ax.bar(qm.index.astype(str), qm.values, color="steelblue", alpha=0.8)
        except ValueError:
            ax.text(0.5, 0.5, "Low variation", transform=ax.transAxes, ha="center")
        ax.set_title(FIELD_SHORT.get(field, field), fontsize=10)
        ax.set_ylabel("Mean KL Surprise")

    for idx in range(n_fields, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Citation Surprise by LLM Marker Quartile (per Field)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig08_surprise_by_field.png"), dpi=200)
    plt.close()


def fig9_surprise_timeseries(df):
    """Figure 9: Surprise time series for LLM-flagged vs non-flagged."""
    print("  [9/11] Surprise time series by LLM flag...")
    valid = df[(df["has_abstract"] == 1) & (df["surprise_kl"].notna()) &
               (df["surprise_kl"] > 0)].copy()
    valid["log_surprise"] = np.log(valid["surprise_kl"] + 1e-6)
    months = sorted(valid["year_month"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for flag_val, label, color in [(0, "Non-flagged", "steelblue"),
                                    (1, "LLM-flagged", "coral")]:
        subset = valid[valid["llm_flag"] == flag_val]
        monthly = subset.groupby("year_month")["surprise_kl"].mean()
        vals = [monthly.get(m, np.nan) for m in months]
        axes[0].plot(range(len(months)), vals, alpha=0.8, label=label,
                     color=color, linewidth=1.5)

    _add_era_lines(axes[0], valid)
    axes[0].set_title("Panel A: Mean KL Surprise Over Time")
    axes[0].set_ylabel("Mean Surprise (KL)")
    axes[0].legend()
    _set_month_xticks(axes[0], months)

    for flag_val, label, color in [(0, "Non-flagged", "steelblue"),
                                    (1, "LLM-flagged", "coral")]:
        subset = valid[valid["llm_flag"] == flag_val]
        monthly = subset.groupby("year_month")["log_surprise"].mean()
        vals = [monthly.get(m, np.nan) for m in months]
        axes[1].plot(range(len(months)), vals, alpha=0.8, label=label,
                     color=color, linewidth=1.5)

    _add_era_lines(axes[1], valid)
    axes[1].set_title("Panel B: Mean Log Surprise Over Time")
    axes[1].set_ylabel("Mean Log(Surprise)")
    axes[1].legend()
    _set_month_xticks(axes[1], months)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig09_surprise_timeseries.png"), dpi=200)
    plt.close()


def fig10_ols_regressions(df):
    """Figure 10: OLS regressions of surprise on marker rate + controls."""
    print("  [10/11] OLS regressions...")

    valid = df[(df["has_abstract"] == 1) & (df["surprise_kl"].notna()) &
               (df["surprise_kl"] > 0)].copy()
    valid["log_surprise"] = np.log(valid["surprise_kl"] + 1e-6)
    valid = valid.dropna(subset=["marker_rate", "num_authors", "n_references"])
    valid = valid[np.isfinite(valid["log_surprise"])]

    y = valid["log_surprise"].values

    def ols_hc1(X, y):
        n, k = X.shape
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        XtX_inv = np.linalg.inv(X.T @ X)
        meat = (X * (resid ** 2)[:, None]).T @ X
        V = (n / (n - k)) * XtX_inv @ meat @ XtX_inv
        se = np.sqrt(np.maximum(np.diag(V), 0))
        return beta, se, r2, n

    def pval(t):
        x = abs(t)
        b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
        p_coeff = 0.2316419
        t_val = 1.0 / (1.0 + p_coeff * x)
        poly = t_val * (b1 + t_val * (b2 + t_val * (b3 + t_val * (b4 + t_val * b5))))
        phi = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x * x)
        return 2 * phi * poly

    # Model A: marker_rate only
    X_a = np.column_stack([np.ones(len(valid)), valid["marker_rate"].values])
    b_a, se_a, r2_a, n_a = ols_hc1(X_a, y)

    # Model B: + field FE
    fd = pd.get_dummies(valid["field"], drop_first=True, dtype=float)
    X_b = np.column_stack([np.ones(len(valid)), valid["marker_rate"].values, fd.values])
    b_b, se_b, r2_b, n_b = ols_hc1(X_b, y)

    # Model C: + field FE + time FE + controls
    td = pd.get_dummies(valid["year_month"], drop_first=True, dtype=float)
    controls = valid[["num_authors", "n_references", "institutions_distinct_count"]].values
    X_c = np.column_stack([np.ones(len(valid)), valid["marker_rate"].values,
                           fd.values, td.values, controls])
    b_c, se_c, r2_c, n_c = ols_hc1(X_c, y)

    # Model D: binary llm_flag + full controls
    X_d = np.column_stack([np.ones(len(valid)), valid["llm_flag"].values,
                           fd.values, td.values, controls])
    b_d, se_d, r2_d, n_d = ols_hc1(X_d, y)

    results = []
    for label, b, se, r2, n in [
        ("A: Marker rate only", b_a, se_a, r2_a, n_a),
        ("B: + Field FE", b_b, se_b, r2_b, n_b),
        ("C: + Field & Time FE + Controls", b_c, se_c, r2_c, n_c),
        ("D: LLM flag (binary) + Full", b_d, se_d, r2_d, n_d),
    ]:
        t = b[1] / se[1] if se[1] > 0 else 0
        p = pval(t)
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        results.append({
            "Model": label, "beta": round(b[1], 4), "se": round(se[1], 4),
            "t_stat": round(t, 3), "p_value": round(p, 4), "sig": stars,
            "R2": round(r2, 4), "N": n,
        })

    reg_df = pd.DataFrame(results)
    reg_df.to_csv(os.path.join(OUTPUT_DIR, "surprise_regression_table.csv"), index=False)

    # Print
    print("\n  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │  OLS: DV = log(surprise_kl)                                    │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    for _, r in reg_df.iterrows():
        sig = r["sig"]
        print(f"  │  {r['Model']:42s}  β={r['beta']:+.4f}{sig:3s}  SE={r['se']:.4f}  R²={r['R2']:.4f}  │")
    print("  └─────────────────────────────────────────────────────────────────┘")

    # Write text results
    with open(os.path.join(OUTPUT_DIR, "surprise_regression_results.txt"), "w") as f:
        f.write("=" * 70 + "\n")
        f.write("OLS: Marker Word Rate → Citation Surprise\n")
        f.write("DV: log(surprise_kl + 1e-6)\n")
        f.write("=" * 70 + "\n\n")
        for _, r in reg_df.iterrows():
            f.write(f"Model {r['Model']}\n")
            f.write(f"  β = {r['beta']:+.4f} {r['sig']}\n")
            f.write(f"  SE = {r['se']:.4f}\n")
            f.write(f"  t = {r['t_stat']:.3f}, p = {r['p_value']:.4f}\n")
            f.write(f"  R² = {r['R2']:.4f}, N = {r['N']}\n\n")
        f.write("Notes: HC1 robust SEs. Controls = num_authors, n_references, institutions.\n")
        f.write("*** p<0.01, ** p<0.05, * p<0.10\n")

    # Coefficient plot
    fig, ax = plt.subplots(figsize=(8, 5))
    betas = reg_df["beta"].values
    ses = reg_df["se"].values
    y_pos = np.arange(len(reg_df))

    ax.errorbar(betas, y_pos, xerr=1.96 * ses, fmt="o", color="steelblue",
                capsize=5, markersize=8, linewidth=2)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(reg_df["Model"].values, fontsize=9)
    ax.set_xlabel("Coefficient (β ± 1.96 SE)")
    ax.set_title("Effect of LLM Marker Usage on Log Citation Surprise")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig10_coefficient_plot.png"), dpi=200)
    plt.close()


def fig11_field_delta_scatter(df):
    """Figure 11: Field-level Δmarker rate vs Δsurprise."""
    print("  [11/11] Field-level delta scatter...")
    has_abs = df[(df["has_abstract"] == 1) & (df["surprise_kl"].notna())].copy()

    pre = has_abs[has_abs["era"] == "Pre-ChatGPT"]
    post = has_abs[has_abs["era"].isin(["Post-ChatGPT", "Post-4o"])]

    fp = pre.groupby("field").agg(mr_pre=("marker_rate", "mean"),
                                   surp_pre=("surprise_kl", "mean"))
    fq = post.groupby("field").agg(mr_post=("marker_rate", "mean"),
                                    surp_post=("surprise_kl", "mean"))
    delta = fp.join(fq).dropna()
    delta["delta_mr"] = delta["mr_post"] - delta["mr_pre"]
    delta["delta_surp"] = delta["surp_post"] - delta["surp_pre"]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(delta["delta_mr"], delta["delta_surp"], s=100, color="steelblue", zorder=5)

    for field_name, row in delta.iterrows():
        ax.annotate(FIELD_SHORT.get(field_name, field_name),
                    (row["delta_mr"], row["delta_surp"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)

    # Fit line
    x_vals = delta["delta_mr"].values
    y_vals = delta["delta_surp"].values
    if len(x_vals) >= 3:
        coeffs = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), "r--", alpha=0.6, linewidth=1.5)
        corr = np.corrcoef(x_vals, y_vals)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=11, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Δ Marker Word Rate (Post − Pre ChatGPT)")
    ax.set_ylabel("Δ Citation Surprise (Post − Pre ChatGPT)")
    ax.set_title("Field-Level: Change in LLM Usage vs Change in Citation Surprise")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig11_field_delta_scatter.png"), dpi=200)
    plt.close()

    delta.round(6).to_csv(os.path.join(OUTPUT_DIR, "field_delta_table.csv"))


def generate_summary_tables(df):
    """Generate summary CSVs."""
    print("\n  Generating summary tables...")
    has_abs = df[df["has_abstract"] == 1].copy()

    # Field summary
    fs = has_abs.groupby("field").agg(
        n_papers=("paper_id", "count"),
        mean_marker_rate=("marker_rate", "mean"),
        median_marker_rate=("marker_rate", "median"),
        pct_llm=("llm_flag", "mean"),
        pct_llm_strict=("llm_flag_strict", "mean"),
        mean_authors=("num_authors", "mean"),
        mean_institutions=("institutions_distinct_count", "mean"),
        mean_countries=("countries_distinct_count", "mean"),
        mean_surprise=("surprise_kl", "mean"),
        median_surprise=("surprise_kl", "median"),
    ).reset_index()
    fs["pct_llm"] = (fs["pct_llm"] * 100).round(1)
    fs["pct_llm_strict"] = (fs["pct_llm_strict"] * 100).round(1)
    fs.sort_values("mean_marker_rate", ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, "field_summary.csv"), index=False)

    # Era summary
    es = has_abs.groupby("era").agg(
        n_papers=("paper_id", "count"),
        mean_marker_rate=("marker_rate", "mean"),
        pct_llm=("llm_flag", "mean"),
        mean_surprise=("surprise_kl", "mean"),
    ).reset_index()
    es["pct_llm"] = (es["pct_llm"] * 100).round(1)
    es.to_csv(os.path.join(OUTPUT_DIR, "era_summary.csv"), index=False)

    # LLM flag × era
    le = has_abs.groupby(["era", "llm_flag"]).agg(
        n_papers=("paper_id", "count"),
        mean_marker_rate=("marker_rate", "mean"),
        mean_surprise=("surprise_kl", "mean"),
        mean_authors=("num_authors", "mean"),
    ).reset_index()
    le["llm_flag"] = le["llm_flag"].map({0: "Non-flagged", 1: "LLM-flagged"})
    le.to_csv(os.path.join(OUTPUT_DIR, "llm_era_summary.csv"), index=False)


def main():
    print("=" * 70)
    print("EXPLORATORY 04: Analysis & Figures")
    print("=" * 70)

    df = load_data()
    print(f"  Loaded {len(df)} papers")
    print(f"  Fields: {df['field'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    fig1_marker_rate_trends(df)
    fig2_llm_flag_over_time(df)
    fig3_subfield_rates(df)
    fig4_authors_vs_llm(df)
    fig5_institutions_countries(df)
    fig6_era_comparison(df)
    fig7_surprise_quintiles(df)
    fig8_surprise_by_field(df)
    fig9_surprise_timeseries(df)
    fig10_ols_regressions(df)
    fig11_field_delta_scatter(df)
    generate_summary_tables(df)

    print("\n" + "=" * 70)
    print("EXPLORATORY ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
