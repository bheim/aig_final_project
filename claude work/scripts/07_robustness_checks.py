"""
Script 07: Robustness checks on the main regression result.

Uses Model 2 (field FE + time FE) as the preferred specification.
Model 3 (field×time FE) is NOT used because the treatment variable has
zero within-cell variation, making beta1 unidentified.

Uses log(surprise) as the main dependent variable.
Tests whether the null finding (beta1 ≈ -0.10, n.s.) is stable across:
  1. Raw (untransformed) surprise
  2. Winsorizing surprise at 95th percentile (then log)
  3. Winsorizing surprise at 99th percentile (then log)
  4. Using binary propensity score instead of ratio
  5. Dropping Arts & Humanities (thin sample)
  6. Dropping Psychology (negative raw diff)
  7. Dropping both
  8. Using Shannon entropy as alternative surprise measure
  9-10. Varying the post-4o cutoff date (+/- 3 months)
  11-12. Minimum reference thresholds (5, 10)

Output: output/robustness_results.txt, output/robustness_table.csv
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from config import DATA_DIR, OUTPUT_DIR, POST_4O_DATE


def load_data():
    """Load and prepare the analysis dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, "analysis_dataset.csv"))
    return df


def run_model2(df, y_col="log_surprise", treatment_col="treatment", label=""):
    """
    Run Model 2 (field FE + time FE) — the preferred specification.
    Treatment is identified from cross-field variation in propensity
    interacted with the post-4o indicator, controlling for additive
    field and time effects.
    """
    y = df[y_col].copy()

    field_dummies = pd.get_dummies(df["field"], prefix="field", drop_first=True, dtype=float)
    time_dummies = pd.get_dummies(df["year_month"], prefix="ym", drop_first=True, dtype=float)

    X = pd.concat([
        df[[treatment_col, "num_authors", "institutions_distinct_count",
            "countries_distinct_count"]],
        field_dummies,
        time_dummies,
    ], axis=1)
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit(cov_type="HC1")
        return {
            "label": label,
            "N": int(model.nobs),
            "beta1": model.params[treatment_col],
            "se": model.bse[treatment_col],
            "t_stat": model.tvalues[treatment_col],
            "p_value": model.pvalues[treatment_col],
            "r2": model.rsquared,
            "r2_adj": model.rsquared_adj,
        }
    except Exception as e:
        print(f"  ERROR in '{label}': {e}")
        return {
            "label": label, "N": len(df), "beta1": float("nan"),
            "se": float("nan"), "t_stat": float("nan"),
            "p_value": float("nan"), "r2": float("nan"), "r2_adj": float("nan"),
        }


def main():
    print("=" * 70)
    print("SCRIPT 07: Robustness Checks (Model 2 — Field + Time FE)")
    print("=" * 70)

    df = load_data()
    df["log_surprise"] = np.log(df["surprise_kl"] + 1e-6)
    results = []

    # ── 0. Baseline (log surprise, Model 2) ──────────────────────────
    print("\n  [0] Baseline (log surprise)...")
    results.append(run_model2(df, label="Baseline (log surprise)"))

    # ── 1. Raw (untransformed) surprise ──────────────────────────────
    print("  [1] Raw surprise (no log)...")
    results.append(run_model2(df, y_col="surprise_kl", label="Raw surprise (no log)"))

    # ── 2. Winsorize at 95th percentile (then log) ───────────────────
    print("  [2] Winsorized (95th, then log)...")
    df_win = df.copy()
    p95 = df_win["surprise_kl"].quantile(0.95)
    df_win["log_s95"] = np.log(df_win["surprise_kl"].clip(upper=p95) + 1e-6)
    results.append(run_model2(df_win, y_col="log_s95", label="Winsorized (95th)"))

    # ── 3. Winsorize at 99th percentile (then log) ───────────────────
    print("  [3] Winsorized (99th, then log)...")
    df_win99 = df.copy()
    p99 = df_win99["surprise_kl"].quantile(0.99)
    df_win99["log_s99"] = np.log(df_win99["surprise_kl"].clip(upper=p99) + 1e-6)
    results.append(run_model2(df_win99, y_col="log_s99", label="Winsorized (99th)"))

    # ── 4. Binary propensity score ───────────────────────────────────
    print("  [4] Binary propensity score...")
    df_bin = df.copy()
    df_bin["treatment_binary"] = df_bin["propensity_binary"] * df_bin["post_4o"]
    results.append(run_model2(df_bin, treatment_col="treatment_binary",
                              label="Binary propensity"))

    # ── 5. Drop Arts & Humanities ────────────────────────────────────
    print("  [5] Drop Arts & Humanities...")
    df_no_ah = df[df["field"] != "Arts and Humanities"].copy()
    results.append(run_model2(df_no_ah, label="Drop Arts & Hum."))

    # ── 6. Drop Psychology ───────────────────────────────────────────
    print("  [6] Drop Psychology...")
    df_no_psy = df[df["field"] != "Psychology"].copy()
    results.append(run_model2(df_no_psy, label="Drop Psychology"))

    # ── 7. Drop both Arts & Humanities and Psychology ────────────────
    print("  [7] Drop Arts & Hum. + Psychology...")
    df_drop2 = df[~df["field"].isin(["Arts and Humanities", "Psychology"])].copy()
    results.append(run_model2(df_drop2, label="Drop A&H + Psych"))

    # ── 8. Shannon entropy as outcome ────────────────────────────────
    print("  [8] Shannon entropy...")
    if "surprise_entropy" in df.columns:
        results.append(run_model2(df, y_col="surprise_entropy",
                                  label="Shannon entropy"))
    else:
        print("    Skipped (column not found)")

    # ── 9. Earlier post-4o cutoff (Aug 2024) ─────────────────────────
    print("  [9] Earlier cutoff (Aug 2024)...")
    df_early = df.copy()
    df_early["post_4o"] = (df_early["publication_date"] >= "2024-08-01").astype(int)
    df_early["treatment"] = df_early["propensity_ratio"] * df_early["post_4o"]
    results.append(run_model2(df_early, label="Cutoff: Aug 2024"))

    # ── 10. Later post-4o cutoff (Feb 2025) ──────────────────────────
    print("  [10] Later cutoff (Feb 2025)...")
    df_late = df.copy()
    df_late["post_4o"] = (df_late["publication_date"] >= "2025-02-01").astype(int)
    df_late["treatment"] = df_late["propensity_ratio"] * df_late["post_4o"]
    results.append(run_model2(df_late, label="Cutoff: Feb 2025"))

    # ── 11. Minimum 5 resolved references ────────────────────────────
    print("  [11] Min 5 resolved refs...")
    df_min5 = df[df["n_references_resolved"] >= 5].copy()
    results.append(run_model2(df_min5, label="Min 5 refs"))

    # ── 12. Minimum 10 resolved references ───────────────────────────
    print("  [12] Min 10 resolved refs...")
    df_min10 = df[df["n_references_resolved"] >= 10].copy()
    results.append(run_model2(df_min10, label="Min 10 refs"))

    # ── 13. DV: n_distinct_fields_cited ─────────────────────────────
    print("  [13] DV: n_distinct_fields_cited...")
    results.append(run_model2(df, y_col="n_distinct_fields_cited",
                              label="DV: n_distinct_fields"))

    # ── 14. DV: within_field_share ──────────────────────────────────
    if "within_field_share" in df.columns and df["within_field_share"].notna().sum() > 0:
        print("  [14] DV: within_field_share...")
        df_wfs = df.dropna(subset=["within_field_share"]).copy()
        results.append(run_model2(df_wfs, y_col="within_field_share",
                                  label="DV: within_field_share"))

    # ── Format results ───────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    def stars(p):
        if pd.isna(p):
            return ""
        if p < 0.01:
            return "***"
        if p < 0.05:
            return "**"
        if p < 0.1:
            return "*"
        return ""

    results_df["sig"] = results_df["p_value"].apply(stars)

    # Print table
    print("\n" + "=" * 90)
    print("ROBUSTNESS CHECK RESULTS (Model 2: Field + Time FE)")
    print("=" * 90)
    print(f"{'Specification':<25s} {'N':>7s} {'beta1':>10s} {'SE':>10s} {'p-value':>10s} {'':>4s} {'R²':>8s}")
    print("-" * 90)
    for _, row in results_df.iterrows():
        print(f"{row['label']:<25s} {row['N']:>7,.0f} {row['beta1']:>10.4f} "
              f"{row['se']:>10.4f} {row['p_value']:>10.4f} {row['sig']:>4s} "
              f"{row['r2']:>8.4f}")
    print("-" * 90)
    print("Robust (HC1) standard errors. All models include field FE + time FE.")
    print("*** p<0.01, ** p<0.05, * p<0.1")
    print("=" * 90)

    # Count outcomes
    sig_positive = ((results_df["beta1"] > 0) & (results_df["p_value"] < 0.05)).sum()
    sig_negative = ((results_df["beta1"] < 0) & (results_df["p_value"] < 0.05)).sum()
    insig = len(results_df) - sig_positive - sig_negative
    total = len(results_df)
    print(f"\nSignificant positive (p<0.05): {sig_positive}/{total}")
    print(f"Significant negative (p<0.05): {sig_negative}/{total}")
    print(f"Insignificant:                 {insig}/{total}")

    # Save
    txt_path = os.path.join(OUTPUT_DIR, "robustness_results.txt")
    csv_path = os.path.join(OUTPUT_DIR, "robustness_table.csv")

    results_df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    with open(txt_path, "w") as f:
        f.write("ROBUSTNESS CHECK RESULTS (Model 2: Field + Time FE)\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Specification':<25s} {'N':>7s} {'beta1':>10s} {'SE':>10s} "
                f"{'p-value':>10s} {'':>4s} {'R²':>8s}\n")
        f.write("-" * 90 + "\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['label']:<25s} {row['N']:>7,.0f} {row['beta1']:>10.4f} "
                    f"{row['se']:>10.4f} {row['p_value']:>10.4f} {row['sig']:>4s} "
                    f"{row['r2']:>8.4f}\n")
        f.write("-" * 90 + "\n")
        f.write("Robust (HC1) standard errors. All models include field FE + time FE.\n")
        f.write("*** p<0.01, ** p<0.05, * p<0.1\n")
        f.write(f"\nSignificant positive (p<0.05): {sig_positive}/{total}\n")
        f.write(f"Significant negative (p<0.05): {sig_negative}/{total}\n")
        f.write(f"Insignificant:                 {insig}/{total}\n")
    print(f"  Report saved: {txt_path}")

    print("\n" + "=" * 70)
    print("Done. Robustness checks complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
