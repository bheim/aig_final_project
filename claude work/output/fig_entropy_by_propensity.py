"""
Figure: Shannon Entropy over time — High vs Low AI Propensity fields.
Median split on propensity_ratio. Fitted lines pre- and post-4o on each group.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from config import DATA_DIR, OUTPUT_DIR, POST_4O_DATE

# ── Load data ────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_DIR, "analysis_dataset.csv"))

# Median split on propensity
prop = df.groupby("field")["propensity_ratio"].first()
median_prop = prop.median()
high_fields = prop[prop >= median_prop].index.tolist()
low_fields = prop[prop < median_prop].index.tolist()

df["propensity_group"] = df["field"].apply(
    lambda f: "High AI Propensity" if f in high_fields else "Low AI Propensity"
)

# Monthly means
monthly = (
    df.groupby(["year_month", "propensity_group"])["surprise_entropy"]
    .mean()
    .reset_index()
)
monthly["date"] = pd.to_datetime(monthly["year_month"] + "-15")
monthly = monthly.sort_values("date")

cutoff = pd.to_datetime(POST_4O_DATE)

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

colors = {"High AI Propensity": "#d62728", "Low AI Propensity": "#1f77b4"}

for group in ["High AI Propensity", "Low AI Propensity"]:
    gdf = monthly[monthly["propensity_group"] == group].copy()
    color = colors[group]

    # Scatter points
    ax.scatter(gdf["date"], gdf["surprise_entropy"], color=color, alpha=0.5, s=30, zorder=3)

    # Fit lines: pre and post
    for period_label, mask in [("pre", gdf["date"] < cutoff), ("post", gdf["date"] >= cutoff)]:
        sub = gdf[mask]
        if len(sub) < 2:
            continue
        x_num = (sub["date"] - sub["date"].min()).dt.days.values.astype(float)
        y_vals = sub["surprise_entropy"].values
        # Linear fit
        coeffs = np.polyfit(x_num, y_vals, 1)
        x_fit = np.linspace(x_num.min(), x_num.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)
        dates_fit = sub["date"].min() + pd.to_timedelta(x_fit, unit="D")

        linestyle = "-" if period_label == "pre" else "--"
        label = f"{group}" if period_label == "pre" else None
        ax.plot(dates_fit, y_fit, color=color, linewidth=2.5, linestyle=linestyle,
                label=label, zorder=4)

# Vertical cutoff line
ax.axvline(cutoff, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)
ax.text(cutoff + pd.Timedelta(days=15), ax.get_ylim()[1] * 0.99, "Post-4o\ncutoff",
        fontsize=9, color="gray", va="top", ha="left")

# Formatting
ax.set_xlabel("Publication Month", fontsize=12)
ax.set_ylabel("Mean Shannon Entropy (nats)", fontsize=12)
ax.set_title("Citation Diversity (Shannon Entropy) Over Time\nHigh vs. Low AI Propensity Fields",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="lower left")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30, ha="right")
ax.grid(axis="y", alpha=0.3)

# Add note about line styles
ax.text(0.98, 0.02, "Solid = pre-period fit, Dashed = post-period fit",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
        color="gray", style="italic")

plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "fig_entropy_by_propensity.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved: {out_path}")

# Also print the group compositions
print(f"\nHigh propensity fields (>= {median_prop:.4f}):")
for f in sorted(high_fields):
    print(f"  {f}: {prop[f]:.4f}")
print(f"\nLow propensity fields (< {median_prop:.4f}):")
for f in sorted(low_fields):
    print(f"  {f}: {prop[f]:.4f}")
