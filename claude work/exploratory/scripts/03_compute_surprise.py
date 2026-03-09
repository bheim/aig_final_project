"""
Script 03: Compute citation surprise (KL divergence) for each paper.

Uses the same methodology as the causal analysis:
1. Build baseline citation distribution P_f(g) from pre-ChatGPT papers
2. For each paper, compute its reference field distribution q_i(g)
3. Compute KL divergence: Surprise_i = D_KL(q_i || P_f)
4. Merge surprise scores into the scored dataset

Output: data/analysis_ready.csv (scored_papers + surprise scores)
"""

import os
import math
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime

from config import DATA_DIR, FIELDS, BASELINE_CUTOFF
from utils import load_checkpoint


ALL_FIELDS = FIELDS + ["Other"]


def map_field(field_name):
    """Map a field name to one of our tracked fields or 'Other'."""
    if field_name in FIELDS:
        return field_name
    return "Other"


def build_baseline_distribution(papers_dict, ref_fields, cutoff_date):
    """
    Build baseline co-citation distribution P_f(g) using papers
    published before the cutoff date.
    """
    print(f"  Building baseline distributions (papers before {cutoff_date})...")

    ref_counts = defaultdict(lambda: Counter())
    n_baseline = 0
    cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")

    for key, papers in papers_dict.items():
        for paper in papers:
            pub_date = paper.get("publication_date", "")
            if not pub_date:
                continue
            try:
                paper_dt = datetime.strptime(pub_date[:10], "%Y-%m-%d")
            except ValueError:
                continue
            if paper_dt >= cutoff_dt:
                continue

            focal_field = paper.get("field")
            if not focal_field or focal_field not in FIELDS:
                continue

            refs = paper.get("referenced_works", [])
            if not refs:
                continue

            n_baseline += 1
            for ref_id in refs:
                ref_field_raw = ref_fields.get(ref_id)
                if ref_field_raw is None or ref_field_raw == "Unknown":
                    continue
                ref_field = map_field(ref_field_raw)
                ref_counts[focal_field][ref_field] += 1

    print(f"  Baseline papers used: {n_baseline}")

    # Normalize to probabilities
    baseline = {}
    for focal_field in FIELDS:
        counts = ref_counts[focal_field]
        total = sum(counts.values())
        if total == 0:
            print(f"  WARNING: No baseline references for {focal_field}")
            baseline[focal_field] = {g: 1.0 / len(ALL_FIELDS) for g in ALL_FIELDS}
        else:
            baseline[focal_field] = {g: counts.get(g, 0) / total for g in ALL_FIELDS}

    return baseline


def compute_kl_divergence(q, p, fields):
    """KL divergence D_KL(q || p) with Laplace smoothing."""
    epsilon = 1e-8

    q_smooth = {g: q.get(g, 0) + epsilon for g in fields}
    p_smooth = {g: p.get(g, 0) + epsilon for g in fields}

    q_total = sum(q_smooth.values())
    p_total = sum(p_smooth.values())

    q_norm = {g: q_smooth[g] / q_total for g in fields}
    p_norm = {g: p_smooth[g] / p_total for g in fields}

    kl = 0.0
    for g in fields:
        if q_norm[g] > 0:
            kl += q_norm[g] * math.log(q_norm[g] / p_norm[g])
    return kl


def compute_shannon_entropy(q, fields):
    """Shannon entropy of distribution q."""
    epsilon = 1e-8
    q_smooth = {g: q.get(g, 0) + epsilon for g in fields}
    total = sum(q_smooth.values())
    q_norm = {g: q_smooth[g] / total for g in fields}

    entropy = 0.0
    for g in fields:
        if q_norm[g] > 0:
            entropy -= q_norm[g] * math.log(q_norm[g])
    return entropy


def compute_surprise_scores(papers_dict, ref_fields, baseline):
    """Compute surprise scores for all papers."""
    print("\n  Computing surprise scores...")

    results = {}  # paper_id -> {surprise_kl, surprise_entropy, n_refs_resolved, n_distinct_fields}

    for key, papers in papers_dict.items():
        for paper in papers:
            pid = paper.get("id")
            focal_field = paper.get("field")
            if not pid or not focal_field or focal_field not in FIELDS:
                continue

            refs = paper.get("referenced_works", [])

            # Build paper-level reference distribution
            ref_field_counts = Counter()
            n_refs_resolved = 0
            for ref_id in refs:
                ref_field_raw = ref_fields.get(ref_id)
                if ref_field_raw is None or ref_field_raw == "Unknown":
                    continue
                ref_field = map_field(ref_field_raw)
                ref_field_counts[ref_field] += 1
                n_refs_resolved += 1

            # Need at least 3 resolvable references
            if n_refs_resolved < 3:
                results[pid] = {
                    "surprise_kl": np.nan,
                    "surprise_entropy": np.nan,
                    "n_references_resolved": n_refs_resolved,
                    "n_distinct_fields_cited": 0,
                }
                continue

            q_i = {g: ref_field_counts.get(g, 0) / n_refs_resolved for g in ALL_FIELDS}
            p_f = baseline.get(focal_field, {})

            kl = compute_kl_divergence(q_i, p_f, ALL_FIELDS)
            entropy = compute_shannon_entropy(q_i, ALL_FIELDS)
            n_distinct = sum(1 for g in ALL_FIELDS if ref_field_counts.get(g, 0) > 0)

            # Within-field citation share
            own_field_mapped = map_field(focal_field)
            within_field_refs = ref_field_counts.get(own_field_mapped, 0)
            within_field_share = within_field_refs / n_refs_resolved

            results[pid] = {
                "surprise_kl": round(kl, 6),
                "surprise_entropy": round(entropy, 6),
                "n_references_resolved": n_refs_resolved,
                "n_distinct_fields_cited": n_distinct,
                "within_field_share": round(within_field_share, 6),
            }

    return results


def main():
    print("=" * 70)
    print("EXPLORATORY 03: Compute Citation Surprise (KL Divergence)")
    print("=" * 70)

    # Load papers
    papers_path = os.path.join(DATA_DIR, "papers_checkpoint.json")
    papers_dict = load_checkpoint(papers_path)
    if papers_dict is None:
        raise FileNotFoundError(f"No papers at {papers_path}. Run 01 first.")

    # Load reference fields
    refs_path = os.path.join(DATA_DIR, "reference_fields_checkpoint.json")
    ref_fields = load_checkpoint(refs_path)
    if ref_fields is None:
        raise FileNotFoundError(f"No reference fields at {refs_path}. Run 01 first.")

    total_papers = sum(len(p) for p in papers_dict.values())
    print(f"  Papers loaded: {total_papers}")
    print(f"  Reference field mappings: {len(ref_fields)}")

    # Build baseline
    baseline = build_baseline_distribution(papers_dict, ref_fields, BASELINE_CUTOFF)

    # Compute surprise
    surprise_dict = compute_surprise_scores(papers_dict, ref_fields, baseline)
    n_scored = sum(1 for v in surprise_dict.values() if not np.isnan(v.get("surprise_kl", np.nan)))
    print(f"  Papers with surprise scores: {n_scored}")

    # Merge into scored dataset
    scored_path = os.path.join(DATA_DIR, "scored_papers.csv")
    if not os.path.exists(scored_path):
        raise FileNotFoundError(f"No scored papers at {scored_path}. Run 02 first.")

    df = pd.read_csv(scored_path)
    print(f"\n  Merging surprise scores into scored dataset ({len(df)} papers)...")

    # Create surprise columns
    surprise_df = pd.DataFrame.from_dict(surprise_dict, orient="index")
    surprise_df.index.name = "paper_id"
    surprise_df = surprise_df.reset_index()

    df = df.merge(surprise_df, on="paper_id", how="left")

    # Log surprise
    df["log_surprise"] = np.log(df["surprise_kl"] + 1e-6)

    # Summary stats
    valid = df.dropna(subset=["surprise_kl"])
    print(f"  Papers with valid surprise: {len(valid)}")
    print(f"\n  Surprise (KL) by field:")
    field_stats = valid.groupby("field")["surprise_kl"].agg(["mean", "median", "count"])
    print(field_stats.to_string())

    print(f"\n  Surprise (KL) by era:")
    era_stats = valid.groupby("era")["surprise_kl"].agg(["mean", "median", "count"])
    print(era_stats.to_string())

    # Save
    output_path = os.path.join(DATA_DIR, "analysis_ready.csv")
    save_cols = [c for c in df.columns if c != "date"]
    df[save_cols].to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path} ({len(df)} rows)")

    print("\n" + "=" * 70)
    print("Done. Dataset ready for exploratory analysis.")
    print("=" * 70)


if __name__ == "__main__":
    main()
