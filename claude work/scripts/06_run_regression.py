"""
Script 06: Run the diff-in-diff regression.

Specification:
  Y_ift = alpha + beta1 * (AIPropensity_f x D_t^{after 4o})
          + gamma * X_ift + delta_f + theta_t + epsilon_ift

Where:
  - Y_ift = outcome variable
  - delta_f = field fixed effects
  - theta_t = month-year fixed effects
  - X_ift = paper-level controls (num_authors, institutions, countries)

Seven dependent variables:
  1. log(surprise_kl + 1e-6)   — citation novelty (KL divergence)
  2. n_distinct_fields_cited    — citation diversity (count)
  3. within_field_share         — citation insularity (0-1)
  4. citation_hhi               — citation concentration (Herfindahl)
  5. n_references               — reference count
  6. surprise_entropy           — Shannon entropy (nats)
  7. other_field_share          — share of refs outside tracked fields

Model 2 (field + time FE) is the preferred specification.
beta1 is the coefficient of interest.

Output: output/regression_results.txt, output/regression_table.csv
"""

import os
import pandas as pd
import numpy as np
from config import DATA_DIR, OUTPUT_DIR, POST_4O_DATE


def load_data():
    """Load surprise data and propensity scores, merge them."""
    surprise_path = os.path.join(DATA_DIR, "phase2_with_surprise.csv")
    df = pd.read_csv(surprise_path)

    prop_path = os.path.join(DATA_DIR, "propensity_scores.csv")
    prop_df = pd.read_csv(prop_path)

    df = df.merge(
        prop_df[["field", "propensity_ratio", "propensity_binary"]],
        on="field",
        how="left",
    )

    print(f"  Loaded {len(df)} observations")
    print(f"  Fields: {df['field'].nunique()}")
    print(f"  Date range: {df['publication_date'].min()} to {df['publication_date'].max()}")

    return df


def prepare_regression_data(df):
    """Create regression variables."""
    df["post_4o"] = (df["publication_date"] >= POST_4O_DATE).astype(int)
    df["treatment"] = df["propensity_ratio"] * df["post_4o"]
    df["field_time"] = df["field"] + "_" + df["year_month"]

    # Log-transform surprise (main dependent variable)
    df["log_surprise"] = np.log(df["surprise_kl"] + 1e-6)

    # Ensure new DV columns exist (backward compat)
    for col in ["within_field_share", "citation_hhi", "other_field_share", "surprise_entropy"]:
        if col not in df.columns:
            print(f"  WARNING: {col} not in data. Run 05_compute_surprise.py first.")
            df[col] = np.nan

    required_cols = ["surprise_kl", "propensity_ratio", "num_authors",
                     "institutions_distinct_count", "countries_distinct_count"]
    before = len(df)
    df = df.dropna(subset=required_cols)
    print(f"  Dropped {before - len(df)} rows with missing values")
    print(f"  Final sample: {len(df)} observations")

    print(f"\n  Post-4o observations: {df['post_4o'].sum()}")
    print(f"  Pre-4o observations: {(1 - df['post_4o']).sum()}")
    print(f"  Treatment (mean): {df['treatment'].mean():.4f}")

    return df


def run_ols_with_dummies(df):
    """
    Run OLS regressions with explicit dummy variables for fixed effects.
    Runs Model 1 (no FE) for log_surprise, then Model 2 for all 7 DVs.
    """
    import statsmodels.api as sm

    results = {}

    field_dummies = pd.get_dummies(df["field"], prefix="field", drop_first=True, dtype=float)
    time_dummies = pd.get_dummies(df["year_month"], prefix="ym", drop_first=True, dtype=float)

    X2 = pd.concat([
        df[["treatment", "num_authors", "institutions_distinct_count",
            "countries_distinct_count"]],
        field_dummies,
        time_dummies,
    ], axis=1)
    X2 = sm.add_constant(X2)

    # Also build Model 1 (no FE) for log_surprise only
    X1 = df[["treatment", "propensity_ratio", "post_4o",
             "num_authors", "institutions_distinct_count",
             "countries_distinct_count"]].copy()
    X1 = sm.add_constant(X1)

    # ── DV 1: log(surprise) — Model 1 (no FE) ───────────────────────────
    print("\n  Running DV: log(surprise) — No FE...")
    y_logsurp = df["log_surprise"]

    model1 = sm.OLS(y_logsurp, X1).fit(cov_type="HC1")
    results["(1) Log Surprise — No FE"] = model1
    print(f"    Model 1: beta1 = {model1.params['treatment']:.6f} "
          f"(SE={model1.bse['treatment']:.6f}, p={model1.pvalues['treatment']:.4f})")

    # ── DV 2: log(surprise) — Model 2 (Field+Time FE) ───────────────────
    print("\n  Running DV: log(surprise) — Field+Time FE...")
    model2 = sm.OLS(y_logsurp, X2).fit(cov_type="HC1")
    results["(2) Log Surprise — Field+Time FE"] = model2
    print(f"    Model 2: beta1 = {model2.params['treatment']:.6f} "
          f"(SE={model2.bse['treatment']:.6f}, p={model2.pvalues['treatment']:.4f})")

    # ── All 7 DVs with Model 2 ───────────────────────────────────────────
    DV_LIST = [
        ("n_distinct_fields_cited", "Distinct Fields"),
        ("within_field_share",      "Within-Field Share"),
        ("citation_hhi",            "Citation HHI"),
        ("n_references",            "Reference Count"),
        ("surprise_entropy",        "Shannon Entropy"),
        ("other_field_share",       "Other-Field Share"),
    ]

    model_num = 3
    for dv_col, dv_label in DV_LIST:
        if dv_col not in df.columns or df[dv_col].isna().all():
            print(f"\n  Skipping DV: {dv_col} (not available)")
            continue

        print(f"\n  Running DV: {dv_col}...")
        df_dv = df.dropna(subset=[dv_col])
        X2_dv = pd.concat([
            df_dv[["treatment", "num_authors", "institutions_distinct_count",
                     "countries_distinct_count"]],
            pd.get_dummies(df_dv["field"], prefix="field", drop_first=True, dtype=float),
            pd.get_dummies(df_dv["year_month"], prefix="ym", drop_first=True, dtype=float),
        ], axis=1)
        X2_dv = sm.add_constant(X2_dv)
        y_dv = df_dv[dv_col]

        model_dv = sm.OLS(y_dv, X2_dv).fit(cov_type="HC1")
        results[f"({model_num}) {dv_label} — Field+Time FE"] = model_dv
        print(f"    Model 2: beta1 = {model_dv.params['treatment']:.6f} "
              f"(SE={model_dv.bse['treatment']:.6f}, p={model_dv.pvalues['treatment']:.4f})")
        model_num += 1

    return results


def format_results_table(results):
    """Format regression results into a clean comparison table."""
    lines = []
    lines.append("=" * 65)
    lines.append("REGRESSION RESULTS: Impact of AI Adoption on Research Surprise")
    lines.append("=" * 65)
    lines.append("Dependent variable: Surprise (KL divergence)")
    lines.append("")

    model_names = list(results.keys())
    header = f"{'Variable':40s}" + "".join(f"{name:>20s}" for name in model_names)
    lines.append(header)
    lines.append("-" * 65)

    # Key coefficient: treatment
    row = f"{'AIPropensity × Post4o':40s}"
    for name in model_names:
        model = results[name]
        coef = model.params.get("treatment", float("nan"))
        pval = model.pvalues.get("treatment", 1.0)
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        row += f"{coef:>17.4f}{sig:>3s}"
    lines.append(row)

    row = f"{'':40s}"
    for name in model_names:
        model = results[name]
        se = model.bse.get("treatment", float("nan"))
        se_str = f"({se:.4f})"
        row += f"{se_str:>20s}"
    lines.append(row)
    lines.append("")

    # Controls
    for var, label in [
        ("num_authors", "Num authors"),
        ("institutions_distinct_count", "Distinct institutions"),
        ("countries_distinct_count", "Distinct countries"),
    ]:
        row = f"{label:40s}"
        for name in model_names:
            model = results[name]
            coef = model.params.get(var, float("nan"))
            pval = model.pvalues.get(var, 1.0)
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            row += f"{coef:>17.4f}{sig:>3s}"
        lines.append(row)

        row = f"{'':40s}"
        for name in model_names:
            model = results[name]
            se = model.bse.get(var, float("nan"))
            se_str = f"({se:.4f})"
            row += f"{se_str:>20s}"
        lines.append(row)
    lines.append("")

    # Model statistics
    lines.append("-" * 65)
    row = f"{'N':40s}"
    for name in model_names:
        row += f"{int(results[name].nobs):>20d}"
    lines.append(row)

    row = f"{'R-squared':40s}"
    for name in model_names:
        row += f"{results[name].rsquared:>20.4f}"
    lines.append(row)

    row = f"{'Adj. R-squared':40s}"
    for name in model_names:
        row += f"{results[name].rsquared_adj:>20.4f}"
    lines.append(row)

    lines.append("-" * 65)
    lines.append("Robust standard errors in parentheses")
    lines.append("*** p<0.01, ** p<0.05, * p<0.1")
    lines.append("")

    # Fixed effects indicator
    lines.append("Fixed Effects:")
    row = f"{'  Field FE':40s}"
    row += f"{'No':>20s}{'Yes':>20s}"
    lines.append(row)
    row = f"{'  Time FE':40s}"
    row += f"{'No':>20s}{'Yes':>20s}"
    lines.append(row)

    lines.append("=" * 65)
    return "\n".join(lines)


def parallel_trends_check(df):
    """
    Check parallel pre-trends: estimate surprise trends by field before
    the 4o shock, interacted with AI propensity.
    """
    print("\n  Running parallel trends check...")
    import statsmodels.api as sm

    pre_df = df[df["post_4o"] == 0].copy()

    pre_df["date_numeric"] = pd.to_datetime(pre_df["publication_date"])
    min_date = pre_df["date_numeric"].min()
    pre_df["months_since_start"] = (
        (pre_df["date_numeric"] - min_date).dt.days / 30.44
    ).round()

    pre_df["propensity_x_trend"] = pre_df["propensity_ratio"] * pre_df["months_since_start"]

    X = pre_df[["propensity_x_trend", "months_since_start", "propensity_ratio",
                "num_authors", "institutions_distinct_count", "countries_distinct_count"]]
    X = sm.add_constant(X)
    y = pre_df["surprise_kl"]

    model = sm.OLS(y, X).fit(cov_type="HC1")

    coef = model.params["propensity_x_trend"]
    pval = model.pvalues["propensity_x_trend"]
    print(f"    Propensity × Trend coefficient: {coef:.6f} (p={pval:.4f})")

    if pval > 0.1:
        print("    PASS: No significant differential pre-trend (p > 0.10)")
    else:
        print("    WARNING: Significant differential pre-trend detected. "
              "Parallel trends assumption may be violated.")

    return model


def main():
    print("=" * 60)
    print("SCRIPT 06: Diff-in-Diff Regression")
    print("=" * 60)

    df = load_data()
    df = prepare_regression_data(df)

    # Run regressions
    results = run_ols_with_dummies(df)

    # Format and save results
    table = format_results_table(results)
    print("\n" + table)

    table_path = os.path.join(OUTPUT_DIR, "regression_results.txt")
    with open(table_path, "w") as f:
        f.write(table)
    print(f"\n  Results saved: {table_path}")

    # Save coefficient table as CSV
    coef_rows = []
    for name, model in results.items():
        for var in ["treatment", "num_authors", "institutions_distinct_count",
                     "countries_distinct_count"]:
            if var in model.params:
                coef_rows.append({
                    "model": name,
                    "variable": var,
                    "coefficient": model.params[var],
                    "std_error": model.bse[var],
                    "t_stat": model.tvalues[var],
                    "p_value": model.pvalues[var],
                })
    coef_df = pd.DataFrame(coef_rows)
    coef_path = os.path.join(OUTPUT_DIR, "regression_table.csv")
    coef_df.to_csv(coef_path, index=False)
    print(f"  Coefficient table saved: {coef_path}")

    # Parallel trends check
    trends_model = parallel_trends_check(df)

    trends_path = os.path.join(OUTPUT_DIR, "parallel_trends_check.txt")
    with open(trends_path, "w") as f:
        f.write("Parallel Trends Check\n")
        f.write("=" * 40 + "\n")
        f.write(f"Propensity × Trend coefficient: "
                f"{trends_model.params['propensity_x_trend']:.6f}\n")
        f.write(f"p-value: {trends_model.pvalues['propensity_x_trend']:.4f}\n")
        f.write("\nFull model summary:\n")
        f.write(str(trends_model.summary()))
    print(f"  Parallel trends check saved: {trends_path}")

    # Save the analysis-ready dataset
    analysis_path = os.path.join(DATA_DIR, "analysis_dataset.csv")
    df.to_csv(analysis_path, index=False)
    print(f"  Analysis dataset saved: {analysis_path}")

    print("\n" + "=" * 60)
    print("Done. Regression analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
