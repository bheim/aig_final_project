"""
Script 05: Compute surprise scores (KL divergence) for each paper.

1. Builds a baseline co-citation distribution P_f(g) using pre-Nov 2022 papers.
2. For each focal paper, computes its reference field distribution q_i(g).
3. Computes KL divergence: Surprise_i = D_KL(q_i || P_f).

Output: data/phase2_with_surprise.csv
"""

import os
import json
import math
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from config import DATA_DIR, FIELDS


# All fields we track (our 8 + a catch-all for others)
ALL_FIELDS = FIELDS + ["Other"]


def load_phase2_data():
    """Load Phase 2 papers and reference field mappings."""
    with open(os.path.join(DATA_DIR, "phase2_papers.json"), "r") as f:
        papers_dict = json.load(f)
    with open(os.path.join(DATA_DIR, "reference_fields.json"), "r") as f:
        ref_fields = json.load(f)
    return papers_dict, ref_fields


def flatten_papers(papers_dict):
    """Flatten the field_year_month keyed dict into a list of papers."""
    all_papers = []
    for key, papers in papers_dict.items():
        for p in papers:
            all_papers.append(p)
    return all_papers


def map_field(field_name):
    """Map a field name to one of our 8 fields or 'Other'."""
    if field_name in FIELDS:
        return field_name
    return "Other"


def build_baseline_distribution(all_papers, ref_fields, cutoff_date="2022-11-01"):
    """
    Build the baseline co-citation distribution P_f(g) using papers
    published before the cutoff date.

    For each focal field f, P_f(g) is the share of references that
    point to papers in field g.

    Returns: dict[field_f] -> dict[field_g] -> probability
    """
    print(f"  Building baseline distributions (papers before {cutoff_date})...")

    # Count references by focal field -> referenced field
    ref_counts = defaultdict(lambda: Counter())
    n_baseline = 0

    cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")

    for paper in all_papers:
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
            # Uniform fallback
            baseline[focal_field] = {g: 1.0 / len(ALL_FIELDS) for g in ALL_FIELDS}
        else:
            baseline[focal_field] = {}
            for g in ALL_FIELDS:
                baseline[focal_field][g] = counts.get(g, 0) / total

    # Print baseline summary
    print("\n  Baseline distributions (top 3 cited fields per focal field):")
    for f in FIELDS:
        sorted_fields = sorted(baseline[f].items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{g}: {p:.3f}" for g, p in sorted_fields)
        print(f"    {f:30s} -> {top_str}")

    return baseline


def compute_kl_divergence(q, p, fields):
    """
    Compute KL divergence D_KL(q || p) over the given field categories.

    Uses Laplace smoothing to handle zero probabilities:
    - Add a small epsilon to both distributions and renormalize.
    """
    epsilon = 1e-8

    # Smooth and normalize
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
    """Compute Shannon entropy of distribution q as an alternative surprise measure."""
    epsilon = 1e-8
    q_smooth = {g: q.get(g, 0) + epsilon for g in fields}
    total = sum(q_smooth.values())
    q_norm = {g: q_smooth[g] / total for g in fields}

    entropy = 0.0
    for g in fields:
        if q_norm[g] > 0:
            entropy -= q_norm[g] * math.log(q_norm[g])
    return entropy


def compute_surprise_for_papers(all_papers, ref_fields, baseline):
    """
    For each paper, compute its reference field distribution and surprise score.

    Returns a list of dicts ready for DataFrame conversion.
    """
    print("\n  Computing surprise scores for all papers...")
    rows = []

    for paper in all_papers:
        focal_field = paper.get("field")
        if not focal_field or focal_field not in FIELDS:
            continue

        refs = paper.get("referenced_works", [])
        pub_date = paper.get("publication_date", "")

        # Paper-level reference distribution q_i(g)
        ref_field_counts = Counter()
        n_refs_resolved = 0
        for ref_id in refs:
            ref_field_raw = ref_fields.get(ref_id)
            if ref_field_raw is None or ref_field_raw == "Unknown":
                continue  # skip unresolved references
            ref_field = map_field(ref_field_raw)
            ref_field_counts[ref_field] += 1
            n_refs_resolved += 1

        n_total_refs = len(refs)

        # Skip papers with no resolvable references (need at least 3)
        if n_refs_resolved < 3:
            continue

        # Convert to probability distribution
        q_i = {g: ref_field_counts.get(g, 0) / n_refs_resolved for g in ALL_FIELDS}

        # Get baseline for this field
        p_f = baseline.get(focal_field, {})

        # Compute surprise measures
        kl_surprise = compute_kl_divergence(q_i, p_f, ALL_FIELDS)
        entropy = compute_shannon_entropy(q_i, ALL_FIELDS)

        # Count distinct fields cited (simple diversity measure)
        n_distinct_fields = sum(1 for g in ALL_FIELDS if ref_field_counts.get(g, 0) > 0)

        # Within-field citation share: fraction of resolved refs pointing
        # to papers in the same field as the focal paper
        own_field_mapped = map_field(focal_field)
        within_field_refs = ref_field_counts.get(own_field_mapped, 0)
        within_field_share = within_field_refs / n_refs_resolved

        # HHI (Herfindahl) of citation field distribution — measures
        # concentration. HHI=1 means all refs in one field, lower = more diverse.
        hhi = sum(s ** 2 for s in q_i.values())

        # Share of references pointing to "Other" (outside our tracked fields)
        other_share = q_i.get("Other", 0)

        rows.append({
            "paper_id": paper.get("id"),
            "publication_date": pub_date,
            "year_month": pub_date[:7] if pub_date else "",
            "field": focal_field,
            "subfield": paper.get("subfield", ""),
            "num_authors": paper.get("num_authors", 0),
            "institutions_distinct_count": paper.get("institutions_distinct_count", 0),
            "countries_distinct_count": paper.get("countries_distinct_count", 0),
            "type": paper.get("type", ""),
            "language": paper.get("language", ""),
            "n_references": n_total_refs,
            "n_references_resolved": n_refs_resolved,
            "n_distinct_fields_cited": n_distinct_fields,
            "within_field_share": round(within_field_share, 6),
            "citation_hhi": round(hhi, 6),
            "other_field_share": round(other_share, 6),
            "surprise_kl": round(kl_surprise, 6),
            "surprise_entropy": round(entropy, 6),
        })

    return rows


def main():
    print("=" * 60)
    print("SCRIPT 05: Compute Surprise Scores")
    print("=" * 60)

    papers_dict, ref_fields = load_phase2_data()
    all_papers = flatten_papers(papers_dict)
    print(f"  Total papers loaded: {len(all_papers)}")
    print(f"  Reference field mappings: {len(ref_fields)}")

    # Build baseline
    baseline = build_baseline_distribution(all_papers, ref_fields)

    # Compute surprise
    rows = compute_surprise_for_papers(all_papers, ref_fields, baseline)
    df = pd.DataFrame(rows)

    # Summary statistics
    print(f"\n  Papers with surprise scores: {len(df)}")
    print(f"\n  Surprise (KL divergence) summary by field:")
    summary = df.groupby("field")["surprise_kl"].agg(["mean", "std", "median", "count"])
    print(summary.to_string())

    # Save
    output_path = os.path.join(DATA_DIR, "phase2_with_surprise.csv")
    df.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")

    print("\n" + "=" * 60)
    print("Done. Surprise scores computed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
