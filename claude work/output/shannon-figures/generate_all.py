"""
Generate all 10 Shannon entropy figures for the paper.

Figures:
  1. AI Propensity Scores by Field (horizontal bar)
  2. Marker Word Frequency: Pre vs Post GPT-3.5 (paired dots)
  3. Shannon Entropy Over Time: High vs Low Propensity (time series)
  4. Event Study: Shannon Entropy (coefficient plot)
  5. Shannon Entropy Distributions: Pre vs Post by Propensity Group
  6. Treatment × Time Trend: Shannon Entropy (residualized scatter)
  7. Field-Level Scatter: Propensity vs Δ Entropy
  8. Within-Field Share Over Time: High vs Low Propensity
  9. Coefficient Comparison Across DVs (forest plot)
 10. Parallel Trends: Shannon Entropy (pre-period test)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from config import DATA_DIR, OUTPUT_DIR, POST_4O_DATE, FIELD_SHORT

FIG_DIR = os.path.dirname(__file__)

# ── Load data ────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_DIR, "analysis_dataset.csv"))
df["log_surprise"] = np.log(df["surprise_kl"] + 1e-6)
df["pub_date"] = pd.to_datetime(df["publication_date"])
df["date"] = pd.to_datetime(df["year_month"] + "-15")

prop_df = pd.read_csv(os.path.join(DATA_DIR, "propensity_scores.csv"))

# Propensity groups
prop = df.groupby("field")["propensity_ratio"].first()
median_prop = prop.median()
high_fields = prop[prop >= median_prop].index.tolist()
low_fields = prop[prop < median_prop].index.tolist()
df["prop_group"] = df["field"].apply(lambda f: "High AI Propensity" if f in high_fields else "Low AI Propensity")

cutoff = pd.to_datetime(POST_4O_DATE)
df["post_4o"] = (df["pub_date"] >= cutoff).astype(int)

# Short field names
def short(f):
    return FIELD_SHORT.get(f, f)

# Colors
HIGH_COLOR = "#d62728"
LOW_COLOR = "#1f77b4"
ACCENT = "#2ca02c"

print(f"Loaded {len(df)} obs, {df['field'].nunique()} fields")
print(f"High propensity ({len(high_fields)}): {[short(f) for f in sorted(high_fields)]}")
print(f"Low propensity ({len(low_fields)}): {[short(f) for f in sorted(low_fields)]}")


# ── OLS Engine ───────────────────────────────────────────────────────────
def norm_cdf(x):
    a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
    a4, a5, p = -1.453152027, 1.061405429, 0.3275911
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x / 2)
    return 0.5 * (1.0 + sign * y)

def run_ols(y, X, names):
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    e2 = resid**2
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = (X * e2[:, None]).T @ X
    V = (n / (n - k)) * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    t_stat = beta / se
    p_values = 2 * (1 - norm_cdf(np.abs(t_stat)))
    results = {}
    for i, name in enumerate(names):
        results[name] = {"coef": beta[i], "se": se[i], "t": t_stat[i], "p": p_values[i]}
    return results, r2, resid


def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""


# ════════════════════════════════════════════════════════════════════════
# FIGURE 1: AI Propensity Scores by Field
# ════════════════════════════════════════════════════════════════════════
print("\n── Figure 1: Propensity Scores ──")
fig, ax = plt.subplots(figsize=(8, 7))

prop_sorted = prop.sort_values()
colors = [HIGH_COLOR if f in high_fields else LOW_COLOR for f in prop_sorted.index]
labels = [short(f) for f in prop_sorted.index]

bars = ax.barh(range(len(prop_sorted)), prop_sorted.values, color=colors, edgecolor="white", height=0.7)
ax.set_yticks(range(len(prop_sorted)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("AI Propensity Score (Marker Word Ratio)", fontsize=11)
ax.set_title("AI Propensity by Field\n(Kobak et al. marker word increase, GPT-3.5 era)", fontsize=13, fontweight="bold")
ax.axvline(median_prop, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.text(median_prop + 0.005, -0.8, f"Median = {median_prop:.3f}", fontsize=9, color="gray")

# Add value labels on bars
for i, (val, f) in enumerate(zip(prop_sorted.values, prop_sorted.index)):
    ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=9, color="gray")

# Legend
from matplotlib.patches import Patch
ax.legend([Patch(facecolor=HIGH_COLOR), Patch(facecolor=LOW_COLOR)],
          ["High Propensity (≥ median)", "Low Propensity (< median)"],
          loc="lower right", fontsize=9)
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig01_propensity_scores.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig01_propensity_scores.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 2: Marker Word Frequency Pre vs Post
# ════════════════════════════════════════════════════════════════════════
print("\n── Figure 2: Marker Words Pre vs Post ──")
fig, ax = plt.subplots(figsize=(8, 7))

prop_df_sorted = prop_df.sort_values("propensity_ratio")
y_pos = range(len(prop_df_sorted))
labels2 = [short(f) for f in prop_df_sorted["field"]]

for i, (_, row) in enumerate(prop_df_sorted.iterrows()):
    pre_val = row["avg_markers_pre"]
    post_val = row["avg_markers_post"]
    color = HIGH_COLOR if row["field"] in high_fields else LOW_COLOR

    # Dots
    ax.scatter(pre_val, i, color=color, s=60, zorder=5, marker="o")
    ax.scatter(post_val, i, color=color, s=60, zorder=5, marker="D")

    # Arrow connecting them
    ax.annotate("", xy=(post_val, i), xytext=(pre_val, i),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.6))

ax.set_yticks(list(y_pos))
ax.set_yticklabels(labels2, fontsize=10)
ax.set_xlabel("Avg. Marker Words per Abstract", fontsize=11)
ax.set_title("Marker Word Usage: Before vs. After GPT-3.5\n(○ = Pre-period, ◇ = Post-period)", fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.2)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="gray", markersize=8, linestyle="None", label="Pre (Jan–Oct 2022)"),
    Line2D([0], [0], marker="D", color="gray", markersize=8, linestyle="None", label="Post (Apr 2023–Apr 2024)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig02_marker_words_pre_post.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig02_marker_words_pre_post.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 3: Shannon Entropy Over Time
# ════════════════════════════════════════════════════════════════════════
print("\n── Figure 3: Entropy Time Series ──")
fig, ax = plt.subplots(figsize=(12, 6))

monthly = df.groupby(["year_month", "prop_group"])["surprise_entropy"].mean().reset_index()
monthly["date"] = pd.to_datetime(monthly["year_month"] + "-15")
monthly = monthly.sort_values("date")

for group, color in [("High AI Propensity", HIGH_COLOR), ("Low AI Propensity", LOW_COLOR)]:
    gdf = monthly[monthly["prop_group"] == group]
    ax.scatter(gdf["date"], gdf["surprise_entropy"], color=color, alpha=0.45, s=30, zorder=3)

    for period, mask in [("pre", gdf["date"] < cutoff), ("post", gdf["date"] >= cutoff)]:
        sub = gdf[mask]
        if len(sub) < 2:
            continue
        x_num = (sub["date"] - sub["date"].min()).dt.days.values.astype(float)
        y_vals = sub["surprise_entropy"].values
        coeffs = np.polyfit(x_num, y_vals, 1)
        x_fit = np.linspace(x_num.min(), x_num.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)
        dates_fit = sub["date"].min() + pd.to_timedelta(x_fit, unit="D")
        ls = "-" if period == "pre" else "--"
        label = group if period == "pre" else None
        ax.plot(dates_fit, y_fit, color=color, linewidth=2.5, linestyle=ls, label=label, zorder=4)

ax.axvline(cutoff, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)
ax.text(cutoff + pd.Timedelta(days=15), ax.get_ylim()[1], "Post-4o cutoff",
        fontsize=9, color="gray", va="top", ha="left")

ax.set_xlabel("Publication Month", fontsize=12)
ax.set_ylabel("Mean Shannon Entropy (nats)", fontsize=12)
ax.set_title("Citation Diversity (Shannon Entropy) Over Time\nHigh vs. Low AI Propensity Fields",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="lower left")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30, ha="right")
ax.grid(axis="y", alpha=0.3)
ax.text(0.98, 0.02, "Solid = pre-period fit, Dashed = post-period fit",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom", color="gray", style="italic")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig03_entropy_time_series.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig03_entropy_time_series.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 4: Event Study — Shannon Entropy
# ════════════════════════════════════════════════════════════════════════
print("\n── Figure 4: Event Study ──")

df["quarter"] = df["pub_date"].dt.to_period("Q").astype(str)
quarters_all = sorted(df["quarter"].unique())
omit_q = "2024Q3"
interact_quarters = [q for q in quarters_all if q != omit_q]

def build_es_X(data):
    parts, names = [], []
    for q in interact_quarters:
        indicator = (data["quarter"] == q).astype(float).values
        parts.append(data["propensity_ratio"].values * indicator)
        names.append(f"prop_x_{q}")
    for c in ["num_authors", "institutions_distinct_count", "countries_distinct_count"]:
        parts.append(data[c].values)
        names.append(c)
    fields = sorted(data["field"].unique())
    for f in fields[1:]:
        parts.append((data["field"] == f).astype(float).values)
        names.append(f"fe_{f}")
    months = sorted(data["year_month"].unique())
    for m in months[1:]:
        parts.append((data["year_month"] == m).astype(float).values)
        names.append(f"ym_{m}")
    parts.append(np.ones(len(data)))
    names.append("const")
    return np.column_stack(parts), names

y_es = df["surprise_entropy"].values
X_es, names_es = build_es_X(df)
res_es, r2_es, _ = run_ols(y_es, X_es, names_es)

coefs = []
for q in interact_quarters:
    c = res_es[f"prop_x_{q}"]
    coefs.append({"quarter": q, "coef": c["coef"], "se": c["se"], "p": c["p"]})
coefs.append({"quarter": omit_q, "coef": 0.0, "se": 0.0, "p": 1.0})
cdf = pd.DataFrame(coefs).sort_values("quarter").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 5.5))
x_pos = np.arange(len(cdf))
cutoff_idx = cdf[cdf["quarter"] == omit_q].index[0]

ax.errorbar(x_pos, cdf["coef"], yerr=1.96 * cdf["se"],
            fmt="o", color="black", ecolor="gray", elinewidth=1.2,
            capsize=3, markersize=5, zorder=5)

# Color significant post-period dots
for i, row in cdf.iterrows():
    if row["p"] < 0.05 and row["quarter"] != omit_q:
        c = HIGH_COLOR if i > cutoff_idx else LOW_COLOR
        ax.scatter([i], [row["coef"]], color=c, s=80, zorder=6, edgecolors="black", linewidth=0.5)

ax.axvspan(cutoff_idx + 0.5, len(cdf) - 0.5, alpha=0.08, color="red")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(cutoff_idx, color="gray", linewidth=1, linestyle=":", alpha=0.7)
ax.text(cutoff_idx + 0.3, ax.get_ylim()[1] * 0.9, "Ref.\nperiod", fontsize=8, color="gray", ha="left")

ax.set_xticks(x_pos)
ax.set_xticklabels(cdf["quarter"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("β (AI Propensity × Quarter)", fontsize=11)
ax.set_title("Event Study: Shannon Entropy\n(Reference period: 2024-Q3, just before post-4o cutoff)",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.text(0.98, 0.02, "Filled dots = p < 0.05. Pink shading = post-treatment.",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom", color="gray", style="italic")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig04_event_study_entropy.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig04_event_study_entropy.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 5: Entropy Distributions Pre vs Post
# ════════════════════════════════════════════════════════════════════════
print("\n── Figure 5: Entropy Distributions ──")
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for idx, (group, color) in enumerate([("High AI Propensity", HIGH_COLOR), ("Low AI Propensity", LOW_COLOR)]):
    ax = axes[idx]
    gdf = df[df["prop_group"] == group]
    pre = gdf[gdf["post_4o"] == 0]["surprise_entropy"]
    post = gdf[gdf["post_4o"] == 1]["surprise_entropy"]

    bins = np.linspace(0, 2.5, 50)
    ax.hist(pre, bins=bins, density=True, alpha=0.5, color="steelblue", label=f"Pre-4o (N={len(pre):,})")
    ax.hist(post, bins=bins, density=True, alpha=0.5, color="tomato", label=f"Post-4o (N={len(post):,})")

    ax.axvline(pre.mean(), color="steelblue", linestyle="--", linewidth=1.5)
    ax.axvline(post.mean(), color="tomato", linestyle="--", linewidth=1.5)

    delta = post.mean() - pre.mean()
    ax.set_title(f"{group}\nΔ mean = {delta:+.4f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Shannon Entropy (nats)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)

axes[0].set_ylabel("Density", fontsize=11)
fig.suptitle("Shannon Entropy Distribution: Before vs. After GPT-4o Cutoff", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig05_entropy_distributions.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig05_entropy_distributions.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 6: Treatment × Time Trend (residualized)
# ════════════════════════════════════════════════════════════════════════
print("\n── Figure 6: Residualized Trend ──")

# Residualize entropy against field FE + time FE + controls (no treatment)
def build_residualize_X(data):
    parts, names = [], []
    for c in ["num_authors", "institutions_distinct_count", "countries_distinct_count"]:
        parts.append(data[c].values)
        names.append(c)
    fields = sorted(data["field"].unique())
    for f in fields[1:]:
        parts.append((data["field"] == f).astype(float).values)
        names.append(f"fe_{f}")
    months = sorted(data["year_month"].unique())
    for m in months[1:]:
        parts.append((data["year_month"] == m).astype(float).values)
        names.append(f"ym_{m}")
    parts.append(np.ones(len(data)))
    names.append("const")
    return np.column_stack(parts), names

X_r, n_r = build_residualize_X(df)
_, _, resid_entropy = run_ols(df["surprise_entropy"].values, X_r, n_r)
df["entropy_resid"] = resid_entropy

# Monthly mean of residuals by propensity group — post period only
post_df = df[df["post_4o"] == 1].copy()
post_df["months_post"] = np.maximum(
    ((post_df["pub_date"] - cutoff).dt.days / 30.44).round(), 0
)

monthly_resid = post_df.groupby(["months_post", "prop_group"])["entropy_resid"].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))

for group, color in [("High AI Propensity", HIGH_COLOR), ("Low AI Propensity", LOW_COLOR)]:
    gdf = monthly_resid[monthly_resid["prop_group"] == group]
    ax.scatter(gdf["months_post"], gdf["entropy_resid"], color=color, s=50, alpha=0.7, zorder=3)

    # Fit line
    x = gdf["months_post"].values
    y = gdf["entropy_resid"].values
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), color=color, linewidth=2.5,
                label=f"{group} (slope={coeffs[0]:+.4f}/mo)", zorder=4)

ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
ax.set_xlabel("Months After Post-4o Cutoff", fontsize=12)
ax.set_ylabel("Residualized Shannon Entropy", fontsize=12)
ax.set_title("Post-Period Trend in Residualized Entropy\n(after removing Field FE + Time FE + controls)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig06_residualized_trend.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig06_residualized_trend.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 7: Field-Level Scatter — Propensity vs Δ Entropy
# ════════════════════════════════════════════════════════════════════════
print("\n── Figure 7: Field Scatter ──")

field_pre = df[df["post_4o"] == 0].groupby("field")["surprise_entropy"].mean()
field_post = df[df["post_4o"] == 1].groupby("field")["surprise_entropy"].mean()
delta_entropy = field_post - field_pre

scatter_df = pd.DataFrame({
    "propensity": prop,
    "delta_entropy": delta_entropy
}).dropna()

fig, ax = plt.subplots(figsize=(9, 7))

for _, row in scatter_df.iterrows():
    f = row.name
    color = HIGH_COLOR if f in high_fields else LOW_COLOR
    ax.scatter(row["propensity"], row["delta_entropy"], color=color, s=80, zorder=5, edgecolors="black", linewidth=0.5)
    ax.annotate(short(f), (row["propensity"], row["delta_entropy"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8, color="gray")

# Fit line
x_s = scatter_df["propensity"].values
y_s = scatter_df["delta_entropy"].values
coeffs_s = np.polyfit(x_s, y_s, 1)
x_fit_s = np.linspace(x_s.min() - 0.02, x_s.max() + 0.02, 50)
ax.plot(x_fit_s, np.polyval(coeffs_s, x_fit_s), color="black", linewidth=1.5, linestyle="--", alpha=0.6)

# Correlation
corr = np.corrcoef(x_s, y_s)[0, 1]
ax.text(0.03, 0.97, f"r = {corr:.3f}\nSlope = {coeffs_s[0]:.4f}",
        transform=ax.transAxes, fontsize=10, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

ax.axhline(0, color="gray", linewidth=0.5, linestyle="-", alpha=0.5)
ax.set_xlabel("AI Propensity Score", fontsize=12)
ax.set_ylabel("Δ Mean Shannon Entropy (Post − Pre)", fontsize=12)
ax.set_title("Cross-Sectional Variation:\nAI Propensity vs. Change in Citation Diversity",
             fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig07_field_scatter.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig07_field_scatter.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 8: Within-Field Share Over Time
# ════════════════════════════════════════════════════════════════════════
print("\n── Figure 8: Within-Field Share Time Series ──")
fig, ax = plt.subplots(figsize=(12, 6))

monthly_wfs = df.groupby(["year_month", "prop_group"])["within_field_share"].mean().reset_index()
monthly_wfs["date"] = pd.to_datetime(monthly_wfs["year_month"] + "-15")
monthly_wfs = monthly_wfs.sort_values("date")

for group, color in [("High AI Propensity", HIGH_COLOR), ("Low AI Propensity", LOW_COLOR)]:
    gdf = monthly_wfs[monthly_wfs["prop_group"] == group]
    ax.scatter(gdf["date"], gdf["within_field_share"], color=color, alpha=0.45, s=30, zorder=3)

    for period, mask in [("pre", gdf["date"] < cutoff), ("post", gdf["date"] >= cutoff)]:
        sub = gdf[mask]
        if len(sub) < 2:
            continue
        x_num = (sub["date"] - sub["date"].min()).dt.days.values.astype(float)
        y_vals = sub["within_field_share"].values
        coeffs = np.polyfit(x_num, y_vals, 1)
        x_fit = np.linspace(x_num.min(), x_num.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)
        dates_fit = sub["date"].min() + pd.to_timedelta(x_fit, unit="D")
        ls = "-" if period == "pre" else "--"
        label = group if period == "pre" else None
        ax.plot(dates_fit, y_fit, color=color, linewidth=2.5, linestyle=ls, label=label, zorder=4)

ax.axvline(cutoff, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)
ax.text(cutoff + pd.Timedelta(days=15), ax.get_ylim()[1], "Post-4o cutoff",
        fontsize=9, color="gray", va="top", ha="left")

ax.set_xlabel("Publication Month", fontsize=12)
ax.set_ylabel("Mean Within-Field Citation Share", fontsize=12)
ax.set_title("Citation Insularity (Within-Field Share) Over Time\nHigh vs. Low AI Propensity Fields",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30, ha="right")
ax.grid(axis="y", alpha=0.3)
ax.text(0.98, 0.02, "Solid = pre-period fit, Dashed = post-period fit",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom", color="gray", style="italic")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig08_within_field_share.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig08_within_field_share.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 9: Forest Plot — Treatment × Trend Across DVs
# ════════════════════════════════════════════════════════════════════════
print("\n── Figure 9: Forest Plot ──")

DV_SPECS = [
    ("surprise_entropy",       "Shannon Entropy",     "nats"),
    ("n_distinct_fields_cited", "Distinct Fields",    "count"),
    ("within_field_share",     "Within-Field Share",  "proportion"),
    ("citation_hhi",           "Citation HHI",        "index"),
    ("n_references",           "Reference Count",     "count"),
    ("other_field_share",      "Other-Field Share",   "proportion"),
    ("log_surprise",           "Log Surprise (KL)",   "log units"),
]

# Compute treatment × trend for each DV
df["months_post"] = np.maximum(
    ((df["pub_date"] - cutoff).dt.days / 30.44).round(), 0
).astype(float)
df["treatment_x_trend"] = df["treatment"] * df["months_post"]

def build_trend_X(data):
    fields = sorted(data["field"].unique())
    months = sorted(data["year_month"].unique())
    parts = [data["treatment"].values, data["treatment_x_trend"].values]
    names = ["treatment", "treatment_x_trend"]
    for c in ["num_authors", "institutions_distinct_count", "countries_distinct_count"]:
        parts.append(data[c].values)
        names.append(c)
    for f in fields[1:]:
        parts.append((data["field"] == f).astype(float).values)
        names.append(f"fe_{f}")
    for m in months[1:]:
        parts.append((data["year_month"] == m).astype(float).values)
        names.append(f"ym_{m}")
    parts.append(np.ones(len(data)))
    names.append("const")
    return np.column_stack(parts), names

forest_data = []
for y_col, label, unit in DV_SPECS:
    data = df.dropna(subset=[y_col])
    y = data[y_col].values
    X, names = build_trend_X(data)
    res, r2, _ = run_ols(y, X, names)
    tc = res["treatment_x_trend"]
    forest_data.append({
        "label": label, "unit": unit,
        "coef": tc["coef"], "se": tc["se"], "p": tc["p"],
        "ci_lo": tc["coef"] - 1.96 * tc["se"],
        "ci_hi": tc["coef"] + 1.96 * tc["se"],
    })

fdf = pd.DataFrame(forest_data)

fig, ax = plt.subplots(figsize=(10, 5.5))
y_pos = np.arange(len(fdf))

for i, row in fdf.iterrows():
    color = HIGH_COLOR if row["p"] < 0.05 else "gray"
    ax.errorbar(row["coef"], i, xerr=1.96 * row["se"],
                fmt="o", color=color, ecolor="lightgray", elinewidth=2,
                capsize=4, markersize=8, zorder=5)
    sig = stars(row["p"])
    ax.text(row["ci_hi"] + 0.001, i, f" {row['coef']:+.4f}{sig} (p={row['p']:.3f})",
            va="center", fontsize=9, color=color)

ax.axvline(0, color="black", linewidth=1, linestyle="-")
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{row['label']}\n({row['unit']})" for _, row in fdf.iterrows()], fontsize=10)
ax.set_xlabel("Treatment × Time Trend Coefficient (per month)", fontsize=11)
ax.set_title("Treatment Effect Growth Rate Across All DVs\n(Model 2 + Treatment × Months Since Cutoff)",
             fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.text(0.98, 0.02, "Red = p < 0.05, Gray = not significant",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom", color="gray", style="italic")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig09_forest_plot.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig09_forest_plot.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 10: Parallel Trends — Shannon Entropy (Pre-Period)
# ════════════════════════════════════════════════════════════════════════
print("\n── Figure 10: Parallel Trends (18-month window) ──")

# Use 18-month pre-period window (May 2023 – Oct 2024)
pt_start = cutoff - pd.DateOffset(months=18)
pre_df = df[(df["post_4o"] == 0) & (df["pub_date"] >= pt_start)].copy()
pre_monthly = pre_df.groupby(["year_month", "prop_group"])["surprise_entropy"].mean().reset_index()
pre_monthly["date"] = pd.to_datetime(pre_monthly["year_month"] + "-15")
pre_monthly = pre_monthly.sort_values("date")

# Run formal test: propensity × trend in pre-period with field FE
pre_df["months_since_start"] = ((pre_df["pub_date"] - pre_df["pub_date"].min()).dt.days / 30.44).round()
pre_df["prop_x_trend"] = pre_df["propensity_ratio"] * pre_df["months_since_start"]

field_dummies_pt = pd.get_dummies(pre_df["field"], drop_first=True, dtype=float)
field_fe_names = [f"fe_{c}" for c in field_dummies_pt.columns]

X_pt_parts = [
    pre_df["prop_x_trend"].values[:, None],
    pre_df["months_since_start"].values[:, None],
    pre_df["propensity_ratio"].values[:, None],
    pre_df["num_authors"].values[:, None],
    pre_df["institutions_distinct_count"].values[:, None],
    pre_df["countries_distinct_count"].values[:, None],
    field_dummies_pt.values,
    np.ones((len(pre_df), 1)),
]
X_pt = np.hstack(X_pt_parts)
names_pt = ["prop_x_trend", "months", "propensity", "num_auth", "inst", "countries"] + field_fe_names + ["const"]
res_pt, r2_pt, _ = run_ols(pre_df["surprise_entropy"].values, X_pt, names_pt)
pt_coef = res_pt["prop_x_trend"]

fig, ax = plt.subplots(figsize=(12, 6))

for group, color in [("High AI Propensity", HIGH_COLOR), ("Low AI Propensity", LOW_COLOR)]:
    gdf = pre_monthly[pre_monthly["prop_group"] == group]
    ax.scatter(gdf["date"], gdf["surprise_entropy"], color=color, alpha=0.5, s=35, zorder=3)

    x_num = (gdf["date"] - gdf["date"].min()).dt.days.values.astype(float)
    y_vals = gdf["surprise_entropy"].values
    coeffs = np.polyfit(x_num, y_vals, 1)
    x_fit = np.linspace(x_num.min(), x_num.max(), 100)
    dates_fit = gdf["date"].min() + pd.to_timedelta(x_fit, unit="D")
    ax.plot(dates_fit, np.polyval(coeffs, x_fit), color=color, linewidth=2.5,
            label=f"{group} (slope={coeffs[0]*30.44:+.5f}/mo)", zorder=4)

ax.set_xlabel("Publication Month", fontsize=12)
ax.set_ylabel("Mean Shannon Entropy (nats)", fontsize=12)
ax.set_title("Parallel Trends Check: Shannon Entropy (18-Month Pre-Period, Field FE)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="lower left")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30, ha="right")
ax.grid(axis="y", alpha=0.3)

# Add test result
pass_fail = "PASS" if pt_coef["p"] > 0.1 else "FAIL"
ax.text(0.98, 0.97,
        f"Parallel trends test (18-mo window, field FE):\n"
        f"Propensity × Trend = {pt_coef['coef']:.6f}\n"
        f"p = {pt_coef['p']:.4f} → {pass_fail}",
        transform=ax.transAxes, fontsize=10, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow" if pass_fail == "PASS" else "mistyrose",
                  edgecolor="gray", alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig10_parallel_trends.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig10_parallel_trends.png")


print("\n" + "=" * 50)
print("All 10 figures saved to shannon-figures/")
print("=" * 50)
