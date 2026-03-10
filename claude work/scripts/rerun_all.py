"""
Regenerate regression and robustness output files using numpy OLS.

Seven DVs organized into separate robustness tables:
  Table A: Citation Surprise (KL divergence) — log-transformed
  Table B: Citation Diversity (n_distinct_fields_cited)
  Table C: Citation Insularity (within_field_share)
  Table D: Citation Concentration (HHI)
  Table E: Reference Count (n_references)
  Table F: Citation Entropy (Shannon)
  Table G: Outside-Field Share (other_field_share)

Model 2 (field + time FE) is the preferred specification throughout.
"""

import pandas as pd
import numpy as np
import os

from config import DATA_DIR, OUTPUT_DIR

df = pd.read_csv(os.path.join(DATA_DIR, "analysis_dataset.csv"))
df["log_surprise"] = np.log(df["surprise_kl"] + 1e-6)

# Backward compat: ensure new columns exist
for col in ["within_field_share", "citation_hhi", "other_field_share"]:
    if col not in df.columns:
        print(f"WARNING: {col} not found. Run 05_compute_surprise.py first.")
        df[col] = np.nan

print(f"Loaded {len(df)} observations, {df['field'].nunique()} fields")

# ── Restrict to 18-month pre-period window ──
# We use only 18 months before the GPT-4o cutoff (where parallel trends hold, p=0.56)
# plus all post-period data.
from config import POST_4O_DATE
_cutoff = pd.to_datetime(POST_4O_DATE)
_window_start = _cutoff - pd.DateOffset(months=18)  # 2023-05-01
df["pub_date_dt"] = pd.to_datetime(df["publication_date"])
n_before = len(df)
df = df[df["pub_date_dt"] >= _window_start].reset_index(drop=True)
print(f"Restricted to 18-month pre-period window (>= {_window_start.date()}): "
      f"{n_before} -> {len(df)} observations")

# Create treatment × time_trend variable
# time_trend = months since post-4o cutoff (0 for pre, 1,2,3... for post months)
cutoff_dt = _cutoff
df["months_post"] = np.maximum(
    ((df["pub_date_dt"] - cutoff_dt).dt.days / 30.44).round(), 0
).astype(float)
df["treatment_x_trend"] = df["treatment"] * df["months_post"]
print(f"Post-period months range: {df.loc[df['months_post']>0, 'months_post'].min():.0f} "
      f"to {df['months_post'].max():.0f}")


# ════════════════════════════════════════════════════════════════════════
# OLS ENGINE
# ════════════════════════════════════════════════════════════════════════

def norm_cdf(x):
    a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
    a4, a5, p = -1.453152027, 1.061405429, 0.3275911
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x / 2)
    return 0.5 * (1.0 + sign * y)


def build_model1_X(data):
    return np.column_stack([
        data["treatment"].values,
        data["propensity_ratio"].values,
        data["post_4o"].values,
        data["num_authors"].values,
        data["institutions_distinct_count"].values,
        data["countries_distinct_count"].values,
        np.ones(len(data)),
    ]), ["treatment", "propensity_ratio", "post_4o", "num_authors",
         "institutions_distinct_count", "countries_distinct_count", "const"]


def build_model2_X(data, treatment_col="treatment", include_trend=False):
    """Build Model 2 design matrix. If include_trend=True, adds treatment × time_trend."""
    fields = sorted(data["field"].unique())
    months = sorted(data["year_month"].unique())
    parts = [data[treatment_col].values]
    names = [treatment_col]
    if include_trend:
        parts.append(data["treatment_x_trend"].values)
        names.append("treatment_x_trend")
    for c in ["num_authors", "institutions_distinct_count", "countries_distinct_count"]:
        parts.append(data[c].values)
        names.append(c)
    for f in fields[1:]:
        parts.append((data["field"] == f).astype(float).values)
        names.append(f"fe_field_{f}")
    for m in months[1:]:
        parts.append((data["year_month"] == m).astype(float).values)
        names.append(f"fe_ym_{m}")
    parts.append(np.ones(len(data)))
    names.append("const")
    return np.column_stack(parts), names


def run_ols(y, X, names):
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
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
    return results, r2, r2_adj, n


def stars(p):
    if np.isnan(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""


def run_robustness(data, y_col="log_surprise", treatment_col="treatment", label=""):
    yv = data[y_col].values
    X, names = build_model2_X(data, treatment_col=treatment_col)
    res, r2v, r2av, nv = run_ols(yv, X, names)
    tc = res[treatment_col]
    return {"label": label, "N": nv, "beta1": tc["coef"], "se": tc["se"],
            "t_stat": tc["t"], "p_value": tc["p"], "r2": r2v, "r2_adj": r2av}


# ════════════════════════════════════════════════════════════════════════
# STANDARD ROBUSTNESS BATTERY (applied to each DV)
# ════════════════════════════════════════════════════════════════════════

def run_standard_battery(df, y_col, dv_label):
    """Run the full robustness battery for a given DV."""
    results = []

    # 1. Baseline
    results.append(run_robustness(df, y_col=y_col, label="Baseline"))

    # 2. Binary propensity
    df_bin = df.copy()
    df_bin["treatment_binary"] = df_bin["propensity_binary"] * df_bin["post_4o"]
    results.append(run_robustness(df_bin, y_col=y_col,
                                   treatment_col="treatment_binary",
                                   label="Binary propensity"))

    # 3-5. Drop fields
    results.append(run_robustness(
        df[df["field"] != "Arts and Humanities"], y_col=y_col,
        label="Drop Arts & Hum."))
    results.append(run_robustness(
        df[df["field"] != "Psychology"], y_col=y_col,
        label="Drop Psychology"))
    results.append(run_robustness(
        df[~df["field"].isin(["Arts and Humanities", "Psychology"])],
        y_col=y_col, label="Drop A&H + Psych"))

    # 6-7. Cutoff sensitivity
    df_early = df.copy()
    df_early["post_4o"] = (df_early["publication_date"] >= "2024-08-01").astype(int)
    df_early["treatment"] = df_early["propensity_ratio"] * df_early["post_4o"]
    results.append(run_robustness(df_early, y_col=y_col, label="Cutoff: Aug 2024"))

    df_late = df.copy()
    df_late["post_4o"] = (df_late["publication_date"] >= "2025-02-01").astype(int)
    df_late["treatment"] = df_late["propensity_ratio"] * df_late["post_4o"]
    results.append(run_robustness(df_late, y_col=y_col, label="Cutoff: Feb 2025"))

    # 8-9. Reference thresholds
    results.append(run_robustness(
        df[df["n_references_resolved"] >= 5], y_col=y_col, label="Min 5 refs"))
    results.append(run_robustness(
        df[df["n_references_resolved"] >= 10], y_col=y_col, label="Min 10 refs"))

    return results


def format_battery(results, dv_label, dv_description):
    """Format a robustness battery into text and CSV."""
    rdf = pd.DataFrame(results)
    rdf["sig"] = rdf["p_value"].apply(stars)

    header = f"ROBUSTNESS: {dv_label} ({dv_description})"
    width = 90
    lines = []
    lines.append("=" * width)
    lines.append(header)
    lines.append("=" * width)
    lines.append(f"{'Specification':<25s} {'N':>7s} {'beta1':>10s} {'SE':>10s} "
                 f"{'p-value':>10s} {'':>4s} {'R2':>8s}")
    lines.append("-" * width)
    for _, row in rdf.iterrows():
        lines.append(f"{row['label']:<25s} {row['N']:>7,.0f} {row['beta1']:>10.4f} "
                     f"{row['se']:>10.4f} {row['p_value']:>10.4f} {row['sig']:>4s} "
                     f"{row['r2']:>8.4f}")
    lines.append("-" * width)
    lines.append("HC1 robust SEs. All models: Field FE + Time FE.")
    lines.append("*** p<0.01, ** p<0.05, * p<0.10")

    sig_pos = ((rdf["beta1"] > 0) & (rdf["p_value"] < 0.05)).sum()
    sig_neg = ((rdf["beta1"] < 0) & (rdf["p_value"] < 0.05)).sum()
    insig = len(rdf) - sig_pos - sig_neg
    lines.append(f"\nSignificant positive: {sig_pos}/{len(rdf)}")
    lines.append(f"Significant negative: {sig_neg}/{len(rdf)}")
    lines.append(f"Insignificant:        {insig}/{len(rdf)}")

    return "\n".join(lines), rdf


# ════════════════════════════════════════════════════════════════════════
# PART 1: MAIN REGRESSION TABLE (all DVs side by side)
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("MAIN REGRESSION RESULTS")
print("=" * 80)

# Define all DVs
DV_SPECS = [
    ("log_surprise",           "Log Surprise",       "log(KL + 1e-6)"),
    ("n_distinct_fields_cited", "Distinct Fields",    "count"),
    ("within_field_share",     "Within-Field Share",  "proportion"),
    ("citation_hhi",           "Citation HHI",        "concentration"),
    ("n_references",           "Reference Count",     "count"),
    ("surprise_entropy",       "Shannon Entropy",     "nats"),
    ("other_field_share",      "Other-Field Share",   "proportion"),
]

# Run Model 2 (level) and Model 2+ (with trend) for each DV
main_results = {}
trend_results = {}
for y_col, label, unit in DV_SPECS:
    if y_col not in df.columns or df[y_col].isna().all():
        print(f"  SKIP: {label} (not available)")
        continue
    data = df.dropna(subset=[y_col])
    y = data[y_col].values

    # Model 2: level treatment effect
    X, names = build_model2_X(data)
    res, r2v, r2av, nv = run_ols(y, X, names)
    tc = res["treatment"]
    sig = stars(tc["p"])
    print(f"  {label:25s}  beta1 = {tc['coef']:+.4f}{sig:3s}  "
          f"(SE={tc['se']:.4f}, p={tc['p']:.4f})  R2={r2v:.4f}  N={nv}")
    main_results[label] = {
        "y_col": y_col, "unit": unit, "res": res, "r2": r2v, "r2_adj": r2av, "n": nv
    }

    # Model 2+: with treatment × time_trend
    Xt, namest = build_model2_X(data, include_trend=True)
    rest, r2t, r2at, nvt = run_ols(y, Xt, namest)
    tct = rest["treatment"]
    trend_c = rest["treatment_x_trend"]
    print(f"  {label+' (trend)':25s}  beta1 = {tct['coef']:+.4f}{stars(tct['p']):3s}  "
          f"trend = {trend_c['coef']:+.4f}{stars(trend_c['p']):3s}  "
          f"(SE={trend_c['se']:.4f}, p={trend_c['p']:.4f})")
    trend_results[label] = {
        "y_col": y_col, "unit": unit, "res": rest, "r2": r2t, "r2_adj": r2at, "n": nvt
    }

# Also run Model 1 (no FE) for log_surprise
y_ls = df["log_surprise"].values
X1, n1 = build_model1_X(df)
r1, r2_1, r2a_1, nobs1 = run_ols(y_ls, X1, n1)

# Build the main regression table
# Structure: (model_name, results_dict, r2, r2_adj, n, y_col, has_trend)
models_for_table = [("(1) No FE", r1, r2_1, r2a_1, nobs1, "log_surprise", False)]
model_num = 2
for label, info in main_results.items():
    models_for_table.append((
        f"({model_num}) {label}",
        info["res"], info["r2"], info["r2_adj"], info["n"], info["y_col"], False
    ))
    model_num += 1
    # Add the trend version right after
    tinfo = trend_results[label]
    models_for_table.append((
        f"({model_num}) {label}+T",
        tinfo["res"], tinfo["r2"], tinfo["r2_adj"], tinfo["n"], tinfo["y_col"], True
    ))
    model_num += 1

lines = []
lines.append("=" * 140)
lines.append("REGRESSION RESULTS: Impact of AI Adoption on Citation Behavior")
lines.append("=" * 140)
lines.append("")

header = f"{'Variable':35s}"
for name, *_ in models_for_table:
    header += f"{name:>18s}"
lines.append(header)
lines.append("-" * 140)

# Treatment coefficient (level)
row = f"{'AIPropensity x Post4o':35s}"
se_row = f"{'':35s}"
for name, res, *_ in models_for_table:
    if "treatment" in res:
        c = res["treatment"]
        row += f"{c['coef']:>15.4f}{stars(c['p']):>3s}"
        se_val = f"({c['se']:.4f})"
        se_row += f"{se_val:>18s}"
    else:
        row += f"{'--':>18s}"
        se_row += f"{'':>18s}"
lines.append(row)
lines.append(se_row)
lines.append("")

# Treatment × Time Trend coefficient
row = f"{'Treatment x Time Trend':35s}"
se_row = f"{'':35s}"
for name, res, _, _, _, _, has_trend in models_for_table:
    if has_trend and "treatment_x_trend" in res:
        c = res["treatment_x_trend"]
        row += f"{c['coef']:>15.4f}{stars(c['p']):>3s}"
        se_val = f"({c['se']:.4f})"
        se_row += f"{se_val:>18s}"
    else:
        row += f"{'':>18s}"
        se_row += f"{'':>18s}"
lines.append(row)
lines.append(se_row)
lines.append("")

# Controls
for var, var_label in [("num_authors", "Num authors"),
                        ("institutions_distinct_count", "Distinct institutions"),
                        ("countries_distinct_count", "Distinct countries")]:
    row = f"{var_label:35s}"
    se_row = f"{'':35s}"
    for name, res, *_ in models_for_table:
        if var in res:
            c = res[var]
            row += f"{c['coef']:>15.4f}{stars(c['p']):>3s}"
            se_val = f"({c['se']:.4f})"
            se_row += f"{se_val:>18s}"
        else:
            row += f"{'--':>18s}"
            se_row += f"{'':>18s}"
    lines.append(row)
    lines.append(se_row)
lines.append("")

lines.append("-" * 140)
row = f"{'N':35s}" + "".join(f"{n:>18d}" for _, _, _, _, n, _, _ in models_for_table)
lines.append(row)
row = f"{'R-squared':35s}" + "".join(f"{r:>18.4f}" for _, _, r, _, _, _, _ in models_for_table)
lines.append(row)
lines.append("-" * 140)
lines.append("HC1 robust SEs. *** p<0.01, ** p<0.05, * p<0.10")
lines.append("'+T' models include Treatment x Time Trend (months since cutoff)")
lines.append("")

# FE and DV indicators
lines.append("Fixed Effects:")
fe_f = f"{'  Field FE':35s}{'No':>18s}" + "".join(f"{'Yes':>18s}" for _ in models_for_table[1:])
fe_t = f"{'  Time FE':35s}{'No':>18s}" + "".join(f"{'Yes':>18s}" for _ in models_for_table[1:])
lines.append(fe_f)
lines.append(fe_t)

dv_row = f"{'Dependent Variable':35s}"
for name, _, _, _, _, y_col, _ in models_for_table:
    short = {"log_surprise": "log(surp)", "n_distinct_fields_cited": "n_fields",
             "within_field_share": "w/in share", "citation_hhi": "HHI",
             "n_references": "n_refs", "surprise_entropy": "entropy",
             "other_field_share": "other share"}.get(y_col, y_col)
    dv_row += f"{short:>18s}"
lines.append(dv_row)
lines.append("=" * 140)

table_text = "\n".join(lines)
print("\n" + table_text)

with open(os.path.join(OUTPUT_DIR, "regression_results.txt"), "w") as f:
    f.write(table_text)
print(f"\nSaved regression_results.txt")

# Save CSV
coef_rows = []
for name, res, r2v, r2av, nv, y_col, has_trend in models_for_table:
    key_vars = ["treatment", "num_authors", "institutions_distinct_count",
                "countries_distinct_count"]
    if has_trend:
        key_vars.insert(1, "treatment_x_trend")
    for var in key_vars:
        if var in res:
            c = res[var]
            coef_rows.append({"model": name, "dv": y_col, "variable": var,
                              "coefficient": c["coef"], "std_error": c["se"],
                              "t_stat": c["t"], "p_value": c["p"]})
pd.DataFrame(coef_rows).to_csv(os.path.join(OUTPUT_DIR, "regression_table.csv"), index=False)
print("Saved regression_table.csv")


# ════════════════════════════════════════════════════════════════════════
# PART 2: PER-DV ROBUSTNESS BATTERIES
# ════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("ROBUSTNESS CHECKS — SEPARATE BATTERY PER DV")
print("=" * 90)

all_robustness_text = []
all_robustness_dfs = []

for y_col, label, unit in DV_SPECS:
    if y_col not in df.columns or df[y_col].isna().all():
        continue

    data = df.dropna(subset=[y_col])
    print(f"\n── {label} ({y_col}) ──")

    # Run the standard battery
    battery = run_standard_battery(data, y_col, label)

    # Add DV-specific specs for log_surprise
    if y_col == "log_surprise":
        # Raw (untransformed)
        battery.append(run_robustness(data, y_col="surprise_kl", label="Raw (no log)"))
        # Winsorized
        d95 = data.copy()
        p95 = d95["surprise_kl"].quantile(0.95)
        d95["_win95"] = np.log(d95["surprise_kl"].clip(upper=p95) + 1e-6)
        battery.append(run_robustness(d95, y_col="_win95", label="Winsorized (95th)"))
        d99 = data.copy()
        p99 = d99["surprise_kl"].quantile(0.99)
        d99["_win99"] = np.log(d99["surprise_kl"].clip(upper=p99) + 1e-6)
        battery.append(run_robustness(d99, y_col="_win99", label="Winsorized (99th)"))

    text, rdf = format_battery(battery, label, f"DV = {y_col} [{unit}]")
    rdf["dv"] = y_col
    print(text)
    all_robustness_text.append(text)
    all_robustness_dfs.append(rdf)

# Save combined robustness
combined_text = "\n\n".join(all_robustness_text)
with open(os.path.join(OUTPUT_DIR, "robustness_results.txt"), "w") as f:
    f.write(combined_text)
print(f"\nSaved robustness_results.txt")

combined_df = pd.concat(all_robustness_dfs, ignore_index=True)
combined_df.to_csv(os.path.join(OUTPUT_DIR, "robustness_table.csv"), index=False)
print("Saved robustness_table.csv")

# ════════════════════════════════════════════════════════════════════════
# PART 3: EVENT STUDY (Treatment × Quarter interactions)
# ════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("EVENT STUDY: AI Propensity × Quarter Interactions")
print("=" * 90)

df["pub_date"] = pd.to_datetime(df["publication_date"])
df["quarter"] = df["pub_date"].dt.to_period("Q").astype(str)

quarters_all = sorted(df["quarter"].unique())
omit_q = "2024Q3"  # reference quarter: just before post-4o cutoff
interact_quarters = [q for q in quarters_all if q != omit_q]
print(f"Reference quarter: {omit_q}")
print(f"Total quarters: {len(quarters_all)}")


def build_event_study_X(data):
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


event_study_all = []
for y_col, label, unit in DV_SPECS:
    if y_col not in df.columns or df[y_col].isna().all():
        continue
    data = df.dropna(subset=[y_col])
    y = data[y_col].values
    X, names = build_event_study_X(data)
    res, r2v, _, nv = run_ols(y, X, names)

    print(f"\n── {label} ({y_col}) ──  R²={r2v:.4f}, N={nv}")
    for q in interact_quarters:
        c = res[f"prop_x_{q}"]
        sig_str = stars(c["p"])
        print(f"    {q}: β={c['coef']:+.4f} (SE={c['se']:.4f}) p={c['p']:.4f} {sig_str}")
        event_study_all.append({
            "dv": y_col, "dv_label": label, "quarter": q,
            "coef": c["coef"], "se": c["se"], "p_value": c["p"],
            "sig": sig_str, "r2": r2v, "N": nv,
        })
    # Add omitted quarter row
    event_study_all.append({
        "dv": y_col, "dv_label": label, "quarter": omit_q,
        "coef": 0.0, "se": 0.0, "p_value": 1.0,
        "sig": "(ref)", "r2": r2v, "N": nv,
    })

es_df = pd.DataFrame(event_study_all)
es_df.to_csv(os.path.join(OUTPUT_DIR, "event_study_table.csv"), index=False)
print(f"\nSaved event_study_table.csv")

# Write text version
es_lines = []
es_lines.append("=" * 100)
es_lines.append("EVENT STUDY: AI Propensity × Quarter Interactions")
es_lines.append(f"Reference quarter: {omit_q} (just before post-4o cutoff)")
es_lines.append("=" * 100)
for y_col, label, unit in DV_SPECS:
    sub = es_df[es_df["dv"] == y_col].sort_values("quarter")
    if len(sub) == 0:
        continue
    r2v = sub["r2"].iloc[0]
    nv = sub["N"].iloc[0]
    es_lines.append(f"\n{'─' * 80}")
    es_lines.append(f"{label} (DV = {y_col} [{unit}])  R²={r2v:.4f}, N={int(nv)}")
    es_lines.append(f"{'─' * 80}")
    es_lines.append(f"{'Quarter':>10s} {'beta':>10s} {'SE':>10s} {'p-value':>10s} {'':>6s}")
    for _, row in sub.iterrows():
        es_lines.append(f"{row['quarter']:>10s} {row['coef']:>10.4f} {row['se']:>10.4f} "
                        f"{row['p_value']:>10.4f} {row['sig']:>6s}")
es_lines.append("\n" + "=" * 100)
es_lines.append("HC1 robust SEs. All models: Field FE + Monthly Time FE.")
es_lines.append("*** p<0.01, ** p<0.05, * p<0.10")

with open(os.path.join(OUTPUT_DIR, "event_study_results.txt"), "w") as f:
    f.write("\n".join(es_lines))
print("Saved event_study_results.txt")


# ════════════════════════════════════════════════════════════════════════
# PART 4: SUMMARY — which DVs show significant treatment effects?
# ════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("SUMMARY: Treatment Effects Across All DVs")
print("=" * 90)

summary_rows = []
for y_col, label, unit in DV_SPECS:
    if y_col not in df.columns or df[y_col].isna().all():
        continue
    data = df.dropna(subset=[y_col])
    y = data[y_col].values
    X, names = build_model2_X(data)
    res, r2v, _, nv = run_ols(y, X, names)
    tc = res["treatment"]
    sig = stars(tc["p"])
    summary_rows.append({
        "DV": label, "y_col": y_col, "unit": unit,
        "beta1": tc["coef"], "se": tc["se"], "t_stat": tc["t"],
        "p_value": tc["p"], "sig": sig, "r2": r2v, "N": nv,
    })

sdf = pd.DataFrame(summary_rows)
print(f"\n{'DV':25s} {'beta1':>10s} {'SE':>10s} {'p':>10s} {'':>4s} {'R2':>8s}")
print("-" * 75)
for _, row in sdf.iterrows():
    print(f"{row['DV']:25s} {row['beta1']:>10.4f} {row['se']:>10.4f} "
          f"{row['p_value']:>10.4f} {row['sig']:>4s} {row['r2']:>8.4f}")

sdf.to_csv(os.path.join(OUTPUT_DIR, "dv_summary.csv"), index=False)
print(f"\nSaved dv_summary.csv")

print("\nAll output files regenerated.")
