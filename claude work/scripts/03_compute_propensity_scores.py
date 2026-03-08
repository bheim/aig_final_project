"""
Script 03: Compute AI propensity scores for each field.

Uses the Phase 1 abstracts (from Script 02) and the marker word list
(from Script 01) to compute the ratio of post/pre marker word frequency
for each of the 8 fields.

Output: data/propensity_scores.csv, output/propensity_scores_report.txt
"""

import os
import json
import pandas as pd
from config import DATA_DIR, OUTPUT_DIR, FIELDS
from utils import count_marker_words_in_abstract, has_any_marker_word


def load_marker_words():
    """Load the filtered marker word list."""
    path = os.path.join(DATA_DIR, "marker_words.txt")
    with open(path, "r") as f:
        words = set(line.strip().lower() for line in f if line.strip())
    print(f"  Loaded {len(words)} marker words.")
    return words


def load_phase1_abstracts():
    """Load the Phase 1 abstracts collected in Script 02."""
    path = os.path.join(DATA_DIR, "phase1_abstracts.json")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def compute_scores(abstracts, marker_words):
    """
    Compute AI propensity scores for each field.

    Returns a DataFrame with columns:
    - field
    - n_pre, n_post (sample sizes)
    - avg_markers_pre, avg_markers_post (avg distinct marker words per abstract)
    - frac_any_pre, frac_any_post (fraction of abstracts with >=1 marker word)
    - propensity_ratio (main score: (post - pre) / pre)
    - propensity_binary (alternative: (frac_post - frac_pre) / frac_pre)
    """
    rows = []

    for field in FIELDS:
        pre_papers = abstracts.get("pre", {}).get(field, [])
        post_papers = abstracts.get("post", {}).get(field, [])

        if not pre_papers or not post_papers:
            print(f"  WARNING: No data for {field}. Skipping.")
            continue

        # Count marker words in each abstract
        pre_counts = [
            count_marker_words_in_abstract(p.get("abstract_inverted_index"), marker_words)
            for p in pre_papers
        ]
        post_counts = [
            count_marker_words_in_abstract(p.get("abstract_inverted_index"), marker_words)
            for p in post_papers
        ]

        # Binary: does the abstract contain at least one marker word?
        pre_any = [
            has_any_marker_word(p.get("abstract_inverted_index"), marker_words)
            for p in pre_papers
        ]
        post_any = [
            has_any_marker_word(p.get("abstract_inverted_index"), marker_words)
            for p in post_papers
        ]

        n_pre = len(pre_counts)
        n_post = len(post_counts)
        avg_pre = sum(pre_counts) / n_pre
        avg_post = sum(post_counts) / n_post
        frac_pre = sum(pre_any) / n_pre
        frac_post = sum(post_any) / n_post

        # Propensity score: (post - pre) / pre
        if avg_pre > 0:
            propensity_ratio = (avg_post - avg_pre) / avg_pre
        else:
            propensity_ratio = float("nan")

        if frac_pre > 0:
            propensity_binary = (frac_post - frac_pre) / frac_pre
        else:
            propensity_binary = float("nan")

        rows.append({
            "field": field,
            "n_pre": n_pre,
            "n_post": n_post,
            "avg_markers_pre": round(avg_pre, 4),
            "avg_markers_post": round(avg_post, 4),
            "frac_any_pre": round(frac_pre, 4),
            "frac_any_post": round(frac_post, 4),
            "propensity_ratio": round(propensity_ratio, 4),
            "propensity_binary": round(propensity_binary, 4),
        })

    return pd.DataFrame(rows)


def validation_checks(scores_df):
    """Run the validation checks from the implementation guide."""
    report = []
    report.append("=" * 60)
    report.append("VALIDATION REPORT: AI Propensity Scores")
    report.append("=" * 60)

    # 4a. Intuitive ordering
    report.append("\n4a. Field Rankings (by propensity_ratio, descending):")
    ranked = scores_df.sort_values("propensity_ratio", ascending=False)
    for _, row in ranked.iterrows():
        report.append(f"  {row['field']:30s}  {row['propensity_ratio']:+.4f}")

    report.append("\n  Expected high: Computer Science, Business & Management, Mathematics")
    report.append("  Expected low:  Chemistry, Physics & Astronomy")

    # 4c. Variation across fields
    scores = scores_df["propensity_ratio"]
    report.append(f"\n4c. Cross-field variation:")
    report.append(f"  Min:    {scores.min():.4f}")
    report.append(f"  Max:    {scores.max():.4f}")
    report.append(f"  Range:  {scores.max() - scores.min():.4f}")
    report.append(f"  Std:    {scores.std():.4f}")
    report.append(f"  Mean:   {scores.mean():.4f}")

    if scores.max() - scores.min() < 0.05:
        report.append("  WARNING: Very little cross-field variation. "
                       "Consider expanding the marker word list.")
    else:
        report.append("  OK: Sufficient variation for regression identification.")

    # Comparison of ratio vs binary measures
    report.append(f"\n4d. Ratio vs Binary consistency:")
    corr = scores_df["propensity_ratio"].corr(scores_df["propensity_binary"])
    report.append(f"  Correlation between ratio and binary scores: {corr:.4f}")

    rank_ratio = scores_df["propensity_ratio"].rank()
    rank_binary = scores_df["propensity_binary"].rank()
    rank_corr = rank_ratio.corr(rank_binary)
    report.append(f"  Rank correlation (Spearman): {rank_corr:.4f}")

    if rank_corr > 0.8:
        report.append("  OK: Rankings are consistent across methods.")
    else:
        report.append("  WARNING: Rankings differ. Investigate field-specific word list issues.")

    return "\n".join(report)


def main():
    print("=" * 60)
    print("SCRIPT 03: Compute AI Propensity Scores")
    print("=" * 60)

    marker_words = load_marker_words()
    abstracts = load_phase1_abstracts()

    print("\n  Computing propensity scores...")
    scores_df = compute_scores(abstracts, marker_words)

    # Save scores
    csv_path = os.path.join(DATA_DIR, "propensity_scores.csv")
    scores_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Print scores
    print("\n  AI Propensity Scores:")
    print(scores_df.to_string(index=False))

    # Run validation
    report = validation_checks(scores_df)
    print("\n" + report)

    report_path = os.path.join(OUTPUT_DIR, "propensity_scores_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Validation report saved: {report_path}")

    print("\n" + "=" * 60)
    print("Done. Propensity scores computed and validated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
