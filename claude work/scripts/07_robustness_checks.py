"""
Script 07: Robustness checks on the main regression result.

Tests whether the main finding (beta1 > 0 in Model 3) survives:
  1. Log-transforming surprise (addresses extreme right skew)
  2. Winsorizing surprise at 95th percentile
  3. Using binary propensity score instead of ratio
  4. Dropping Arts & Humanities (thin sample, <50 obs/month)
  5. Dropping Psychology (negative raw diff, potential outlier)
  6. Using Shannon entropy as alternative surprise measure
  7. Varying the post-4o cutoff date (+/- 3 months)

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


def run_model3(df, y_col="surprise_kl", treatment_col="treatment", label=""):
    """
    Run Model 3 (field x time FE) and return key results.
    This is the preferred specification from Script 06.
    """
    # Build field_time if not present
    if "field_time" not in df.columns:
        df = df.copy()
        df["field_time"] = df["field"] + "_" + df["year_month"]

    y = df[y_col].copy()
    ft_dummies = pd.get_dummies(df["field_time"], prefix="ft", drop_first=True, dtype=float)

    X = pd.concat([
        df[[treatment_col, "num_authors", "institutions_distinct_count",
            "countries_distinct_count"]],
        ft_dummies,
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
    print("SCRIPT 07: Robustness Checks")
    print("=" * 70)

    df = load_data()
    df["field_time"] = df["field"] + "_" + df["year_month"]
    results = []

    # ── 0. Baseline (reproduce main result) ──────────────────────────────
    print("\n  [0] Baseline (main result)...")
    results.append(run_model3(df, label="Baseline"))

    # ── 1. Log-transform surprise ────────────────────────────────────────
    print("  [1] Log(surprise)...")
    df_log = df.copy()
    df_log["log_surprise"] = np.log(df_log["surprise_kl"] + 1e-6)
    results.append(run_model3(df_log, y_col="log_surprise", label="Log(surprise)"))

    # ── 2. Winsorize surprise at 95th percentile ─────────────────────────
    print("  [2] Winsorized surprise (95th pctile)...")
    df_win = df.copy()
    p95 = df_win["surprise_kl"].quantile(0.95)
    df_win["surprise_win"] = df_win["surprise_kl"].clip(upper=p95)
    results.append(run_model3(df_win, y_col="surprise_win", label="Winsorized (95th)"))

    # ── 3. Winsorize at 99th percentile ──────────────────────────────────
    print("  [3] Winsorized surprise (99th pctile)...")
    df_win99 = df.copy()
    p99 = df_win99["surprise_kl"].quantile(0.99)
    df_win99["surprise_win99"] = df_win99["surprise_kl"].clip(upper=p99)
    results.append(run_model3(df_win99, y_col="surprise_win99", label="Winsorized (99th)"))

    # ── 4. Binary propensity score ───────────────────────────────────────
    print("  [4] Binary propensity score...")
    df_bin = df.copy()
    df_bin["treatment_binary"] = df_bin["propensity_binary"] * df_bin["post_4o"]
    results.append(run_model3(df_bin, treatment_col="treatment_binary",
                              label="Binary propensity"))

    # ── 5. Drop Arts & Humanities ────────────────────────────────────────
    print("  [5] Drop Arts & Humanities...")
    df_no_ah = df[df["field"] != "Arts and Humanities"].copy()
    df_no_ah["field_time"] = df_no_ah["field"] + "_" + df_no_ah["year_month"]
    results.append(run_model3(df_no_ah, label="Drop Arts & Hum."))

    # ── 6. Drop Psychology ───────────────────────────────────────────────
    print("  [6] Drop Psychology...")
    df_no_psy = df[df["field"] != "Psychology"].copy()
    df_no_psy["field_time"] = df_no_psy["field"] + "_" + df_no_psy["year_month"]
    results.append(run_model3(df_no_psy, label="Drop Psychology"))

    # ── 7. Drop both Arts & Humanities and Psychology ────────────────────
    print("  [7] Drop Arts & Hum. + Psychology...")
    df_drop2 = df[~df["field"].isin(["Arts and Humanities", "Psychology"])].copy()
    df_drop2["field_time"] = df_drop2["field"] + "_" + df_drop2["year_month"]
    results.append(run_model3(df_drop2, label="Drop A&H + Psych"))

    # ── 8. Shannon entropy as outcome ────────────────────────────────────
    print("  [8] Shannon entropy...")
    if "surprise_entropy" in df.columns:
        results.append(run_model3(df, y_col="surprise_entropy",
                                  label="Shannon entropy"))
    else:
        print("    Skipped (column not found)")

    # ── 9. Earlier post-4o cutoff (Aug 2024) ─────────────────────────────
    print("  [9] Earlier cutoff (Aug 2024)...")
    df_early = df.copy()
    df_early["post_4o"] = (df_early["publication_date"] >= "2024-08-01").astype(int)
    df_early["treatment"] = df_early["propensity_ratio"] * df_early["post_4o"]
    df_early["field_time"] = df_early["field"] + "_" + df_early["year_month"]
    results.append(run_model3(df_early, label="Cutoff: Aug 2024"))

    # ── 10. Later post-4o cutoff (Feb 2025) ──────────────────────────────
    print("  [10] Later cutoff (Feb 2025)...")
    df_late = df.copy()
    df_late["post_4o"] = (df_late["publication_date"] >= "2025-02-01").astype(int)
    df_late["treatment"] = df_late["propensity_ratio"] * df_late["post_4o"]
    df_late["field_time"] = df_late["field"] + "_" + df_late["year_month"]
    results.append(run_model3(df_late, label="Cutoff: Feb 2025"))

    # ── 11. Minimum 5 resolved references ────────────────────────────────
    print("  [11] Min 5 resolved refs...")
    df_min5 = df[df["n_references_resolved"] >= 5].copy()
    df_min5["field_time"] = df_min5["field"] + "_" + df_min5["year_month"]
    results.append(run_model3(df_min5, label="Min 5 refs"))

    # ── 12. Minimum 10 resolved references ───────────────────────────────
    print("  [12] Min 10 resolved refs...")
    df_min10 = df[df["n_references_resolved"] >= 10].copy()
    df_min10["field_time"] = df_min10["field"] + "_" + df_min10["year_month"]
    results.append(run_model3(df_min10, label="Min 10 refs"))

    # ── Format results ───────────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    # Significance stars
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
    print("ROBUSTNESS CHECK RESULTS")
    print("=" * 90)
    print(f"{'Specification':<25s} {'N':>7s} {'beta1':>10s} {'SE':>10s} {'p-value':>10s} {'':>4s} {'R²':>8s}")
    print("-" * 90)
    for _, row in results_df.iterrows():
        print(f"{row['label']:<25s} {row['N']:>7,.0f} {row['beta1']:>10.4f} "
              f"{row['se']:>10.4f} {row['p_value']:>10.4f} {row['sig']:>4s} "
              f"{row['r2']:>8.4f}")
    print("-" * 90)
    print("Robust (HC1) standard errors. All models include field × time FE.")
    print("*** p<0.01, ** p<0.05, * p<0.1")
    print("=" * 90)

    # Count how many specs have significant positive beta1
    sig_positive = ((results_df["beta1"] > 0) & (results_df["p_value"] < 0.05)).sum()
    sig_negative = ((results_df["beta1"] < 0) & (results_df["p_value"] < 0.05)).sum()
    total = len(results_df)
    print(f"\nSignificant positive (p<0.05): {sig_positive}/{total}")
    print(f"Significant negative (p<0.05): {sig_negative}/{total}")
    print(f"Insignificant:                 {total - sig_positive - sig_negative}/{total}")

    # Save
    txt_path = os.path.join(OUTPUT_DIR, "robustness_results.txt")
    csv_path = os.path.join(OUTPUT_DIR, "robustness_table.csv")

    results_df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    with open(txt_path, "w") as f:
        f.write("ROBUSTNESS CHECK RESULTS\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Specification':<25s} {'N':>7s} {'beta1':>10s} {'SE':>10s} "
                f"{'p-value':>10s} {'':>4s} {'R²':>8s}\n")
        f.write("-" * 90 + "\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['label']:<25s} {row['N']:>7,.0f} {row['beta1']:>10.4f} "
                    f"{row['se']:>10.4f} {row['p_value']:>10.4f} {row['sig']:>4s} "
                    f"{row['r2']:>8.4f}\n")
        f.write("-" * 90 + "\n")
        f.write("Robust (HC1) standard errors. All models include field × time FE.\n")
        f.write("*** p<0.01, ** p<0.05, * p<0.1\n")
        f.write(f"\nSignificant positive (p<0.05): {sig_positive}/{total}\n")
        f.write(f"Significant negative (p<0.05): {sig_negative}/{total}\n")
        f.write(f"Insignificant:                 {total - sig_positive - sig_negative}/{total}\n")
    print(f"  Report saved: {txt_path}")

    print("\n" + "=" * 70)
    print("Done. Robustness checks complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
