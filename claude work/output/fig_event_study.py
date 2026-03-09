"""
Event-study figure: estimate separate treatment effects by quarter.

Specification:
  Y_ift = alpha + SUM_q [ beta_q * (AIPropensity_f x D_t^{quarter=q}) ]
          + gamma * X_ift + delta_f + theta_t + epsilon_ift

The omitted quarter is the one just before the post-4o cutoff (2024-Q3),
so all coefficients are relative to that period.

Generates coefficient plots for Shannon Entropy and n_distinct_fields_cited.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from config import DATA_DIR, OUTPUT_DIR, POST_4O_DATE

# ── OLS engine (from rerun_all.py) ──────────────────────────────────────
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
    return results, r2, n

# ── Load data ────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_DIR, "analysis_dataset.csv"))
df["log_surprise"] = np.log(df["surprise_kl"] + 1e-6)
df["pub_date"] = pd.to_datetime(df["publication_date"])

# Create quarterly periods
df["quarter"] = df["pub_date"].dt.to_period("Q").astype(str)

quarters = sorted(df["quarter"].unique())
print(f"Quarters: {quarters}")
print(f"Total: {len(quarters)} quarters")

# Omitted (reference) quarter: 2024Q3 — just before post-4o cutoff (Nov 2024)
omit_q = "2024Q3"
print(f"Reference quarter: {omit_q}")

interact_quarters = [q for q in quarters if q != omit_q]

# ── Build event-study design matrix ─────────────────────────────────────
def build_event_study_X(data):
    parts = []
    names = []

    # Propensity × quarter interactions
    for q in interact_quarters:
        indicator = (data["quarter"] == q).astype(float).values
        interaction = data["propensity_ratio"].values * indicator
        parts.append(interaction)
        names.append(f"prop_x_{q}")

    # Controls
    for c in ["num_authors", "institutions_distinct_count", "countries_distinct_count"]:
        parts.append(data[c].values)
        names.append(c)

    # Field FE
    fields = sorted(data["field"].unique())
    for f in fields[1:]:
        parts.append((data["field"] == f).astype(float).values)
        names.append(f"fe_{f}")

    # Time FE (monthly)
    months = sorted(data["year_month"].unique())
    for m in months[1:]:
        parts.append((data["year_month"] == m).astype(float).values)
        names.append(f"ym_{m}")

    # Constant
    parts.append(np.ones(len(data)))
    names.append("const")

    return np.column_stack(parts), names


# ── Run event studies for key DVs ────────────────────────────────────────
DVs = [
    ("surprise_entropy", "Shannon Entropy", "nats"),
    ("n_distinct_fields_cited", "Distinct Fields Cited", "count"),
    ("citation_hhi", "Citation HHI", "index"),
    ("within_field_share", "Within-Field Share", "proportion"),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
axes = axes.flatten()

for idx, (y_col, label, unit) in enumerate(DVs):
    ax = axes[idx]

    data = df.dropna(subset=[y_col])
    y = data[y_col].values
    X, names = build_event_study_X(data)
    res, r2, n = run_ols(y, X, names)

    # Extract interaction coefficients
    coefs = []
    for q in interact_quarters:
        key = f"prop_x_{q}"
        c = res[key]
        coefs.append({
            "quarter": q,
            "coef": c["coef"],
            "se": c["se"],
            "p": c["p"],
            "ci_lo": c["coef"] - 1.96 * c["se"],
            "ci_hi": c["coef"] + 1.96 * c["se"],
        })

    # Add the omitted quarter (zero by definition)
    coefs.append({
        "quarter": omit_q,
        "coef": 0.0, "se": 0.0, "p": 1.0,
        "ci_lo": 0.0, "ci_hi": 0.0,
    })

    cdf = pd.DataFrame(coefs).sort_values("quarter").reset_index(drop=True)
    x_pos = np.arange(len(cdf))

    # Color: pre-period blue, post-period red
    cutoff_idx = cdf[cdf["quarter"] == omit_q].index[0]
    colors = ["#1f77b4" if i <= cutoff_idx else "#d62728" for i in range(len(cdf))]

    # Plot
    ax.errorbar(x_pos, cdf["coef"], yerr=1.96 * cdf["se"],
                fmt="o", color="black", ecolor="gray", elinewidth=1.2,
                capsize=3, markersize=5, zorder=5)

    # Shade post period
    post_start = cutoff_idx + 0.5
    ax.axvspan(post_start, len(cdf) - 0.5, alpha=0.08, color="red")

    # Zero line and reference quarter marker
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.axvline(cutoff_idx, color="gray", linewidth=1, linestyle=":", alpha=0.7)

    # Mark significant coefficients
    for i, row in cdf.iterrows():
        if abs(row["p"]) < 0.05 and row["quarter"] != omit_q:
            ax.scatter([i], [row["coef"]], color="#d62728" if i > cutoff_idx else "#1f77b4",
                      s=80, zorder=6, edgecolors="black", linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cdf["quarter"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"β (effect on {unit})", fontsize=10)
    ax.set_title(f"{label}", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Print key results
    post_qs = cdf[cdf.index > cutoff_idx]
    n_sig = (post_qs["p"] < 0.05).sum()
    print(f"\n{label}:")
    print(f"  R² = {r2:.4f}, N = {n}")
    print(f"  Post-period significant quarters: {n_sig}/{len(post_qs)}")
    for _, row in cdf.iterrows():
        marker = "***" if row["p"] < 0.01 else "**" if row["p"] < 0.05 else "*" if row["p"] < 0.1 else ""
        ref = " (ref)" if row["quarter"] == omit_q else ""
        print(f"    {row['quarter']}: β={row['coef']:+.4f} (SE={row['se']:.4f}) p={row['p']:.4f} {marker}{ref}")

fig.suptitle("Event Study: AI Propensity × Quarter Interactions\n"
             "(Reference period: 2024-Q3, just before post-4o cutoff)",
             fontsize=14, fontweight="bold", y=1.02)

plt.figtext(0.5, -0.02,
            "Dots = point estimates, whiskers = 95% CI. Filled dots = p < 0.05. "
            "Pink shading = post-treatment period.",
            ha="center", fontsize=9, color="gray", style="italic")

plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "fig_event_study.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nSaved: {out_path}")
