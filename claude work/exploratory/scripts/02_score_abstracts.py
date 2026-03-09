"""
Script 02: Download Kobak marker words and score all abstracts.

1. Downloads excess_words.csv from Kobak et al. (2025) GitHub
2. Filters to LLM style words (post-2022)
3. Scores each paper's abstract for marker word usage
4. Outputs a flat CSV with paper metadata + marker scores

Output: data/scored_papers.csv, data/marker_words.txt
"""

import os
import re
import pandas as pd
import numpy as np
import requests
from collections import Counter

from config import DATA_DIR, FIELDS, FIELD_SHORT, KOBAK_URL, BASELINE_CUTOFF
from utils import load_checkpoint


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Download and filter Kobak marker words
# ═══════════════════════════════════════════════════════════════════════════

def get_marker_words():
    """Download and filter Kobak et al. excess words to LLM style markers."""
    marker_path = os.path.join(DATA_DIR, "marker_words.txt")

    # Check if already downloaded
    if os.path.exists(marker_path):
        with open(marker_path, "r") as f:
            words = [line.strip().lower() for line in f if line.strip()]
        print(f"  Loaded {len(words)} cached marker words")
        return set(words)

    print("  Downloading excess_words.csv from Kobak et al. GitHub...")
    resp = requests.get(KOBAK_URL, timeout=30)
    resp.raise_for_status()

    raw_path = os.path.join(DATA_DIR, "excess_words_raw.csv")
    with open(raw_path, "wb") as f:
        f.write(resp.content)

    df = pd.read_csv(raw_path)
    print(f"  Raw excess words: {len(df)} rows")

    # Find annotation column (contains "style")
    annotation_col = None
    for col in df.columns:
        unique_vals = df[col].astype(str).str.lower().unique()
        if any("style" in v for v in unique_vals):
            annotation_col = col
            break

    # Find year column
    year_col = None
    for col in df.columns:
        if "year" in col.lower():
            year_col = col
            break

    # Find word column
    word_col = None
    for candidate in ["word", "term", "token"]:
        if candidate in [c.lower() for c in df.columns]:
            word_col = [c for c in df.columns if c.lower() == candidate][0]
            break
    if word_col is None:
        word_col = df.columns[0]

    # Filter: style words, year >= 2023
    mask = pd.Series([True] * len(df))
    if annotation_col:
        mask = mask & df[annotation_col].astype(str).str.lower().str.contains("style")
    if year_col:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        mask = mask & (df[year_col] >= 2023)

    filtered = df[mask]
    marker_words = sorted(filtered[word_col].str.lower().str.strip().unique())

    # Save
    with open(marker_path, "w") as f:
        for w in marker_words:
            f.write(w + "\n")

    print(f"  Filtered to {len(marker_words)} LLM style marker words")
    return set(marker_words)


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Score abstracts
# ═══════════════════════════════════════════════════════════════════════════

def score_abstract(text, marker_words):
    """
    Score an abstract for LLM marker word usage.

    Returns: (marker_count, marker_types, word_count, marker_rate)
    """
    if not text or not text.strip():
        return 0, 0, 0, 0.0

    words = re.findall(r'[a-z]+', text.lower())
    word_count = len(words)
    if word_count == 0:
        return 0, 0, 0, 0.0

    found = [w for w in words if w in marker_words]
    marker_count = len(found)
    marker_types = len(set(found))
    marker_rate = marker_count / word_count

    return marker_count, marker_types, word_count, marker_rate


def build_scored_dataset(marker_words):
    """
    Load collected papers, score abstracts, and build flat CSV dataset.
    """
    # Load papers
    cp_path = os.path.join(DATA_DIR, "papers_checkpoint.json")
    papers_dict = load_checkpoint(cp_path)
    if papers_dict is None:
        raise FileNotFoundError(
            f"No papers found at {cp_path}. Run 01_collect_papers.py first.")

    print(f"  Loading papers from checkpoint...")
    total = sum(len(p) for p in papers_dict.values())
    print(f"  Total papers: {total}")

    # Track which marker words appear
    all_markers_found = Counter()

    rows = []
    for key, papers in papers_dict.items():
        for p in papers:
            abstract = p.get("abstract", "")
            mc, mt, wc, mr = score_abstract(abstract, marker_words)

            # Track found markers
            if abstract:
                words = re.findall(r'[a-z]+', abstract.lower())
                for w in words:
                    if w in marker_words:
                        all_markers_found[w] += 1

            pub_date = p.get("publication_date", "")
            rows.append({
                "paper_id": p.get("id"),
                "publication_date": pub_date,
                "year_month": pub_date[:7] if pub_date else "",
                "field": p.get("field", ""),
                "subfield": p.get("subfield", ""),
                "abstract": abstract,
                "num_authors": p.get("num_authors", 0),
                "institutions_distinct_count": p.get("institutions_distinct_count", 0),
                "countries_distinct_count": p.get("countries_distinct_count", 0),
                "type": p.get("type", ""),
                "language": p.get("language", ""),
                "n_references": len(p.get("referenced_works", [])),
                "referenced_works": "|".join(p.get("referenced_works", [])),
                "marker_count": mc,
                "marker_types": mt,
                "word_count": wc,
                "marker_rate": mr,
                "has_abstract": 1 if abstract else 0,
            })

    df = pd.DataFrame(rows)

    # Add derived columns
    df["date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["year"] = df["date"].dt.year

    # Era classification
    df["era"] = "Pre-ChatGPT"
    df.loc[df["date"] >= "2023-06-01", "era"] = "Post-ChatGPT"
    df.loc[df["date"] >= "2024-11-01", "era"] = "Post-4o"

    # LLM flags based on marker rate distribution (papers with abstracts only)
    has_abs = df[df["has_abstract"] == 1]
    if len(has_abs) > 0:
        p75 = has_abs["marker_rate"].quantile(0.75)
        p90 = has_abs["marker_rate"].quantile(0.90)
        df["llm_flag"] = (df["marker_rate"] >= p75).astype(int)
        df["llm_flag_strict"] = (df["marker_rate"] >= p90).astype(int)
        print(f"\n  Papers with abstracts: {len(has_abs)}")
        print(f"  Mean marker rate: {has_abs['marker_rate'].mean():.5f}")
        print(f"  75th percentile threshold: {p75:.5f}")
        print(f"  Papers flagged (top 25%): {df['llm_flag'].sum()}")
        print(f"  Papers flagged strict (top 10%): {df['llm_flag_strict'].sum()}")
    else:
        df["llm_flag"] = 0
        df["llm_flag_strict"] = 0

    # Save top marker words found
    top_markers = pd.DataFrame(
        all_markers_found.most_common(50),
        columns=["word", "occurrences"]
    )
    top_markers.to_csv(os.path.join(DATA_DIR, "top_marker_words.csv"), index=False)
    print(f"\n  Top 10 marker words found:")
    for _, row in top_markers.head(10).iterrows():
        print(f"    {row['word']:20s}  {row['occurrences']:,}")

    return df


def main():
    print("=" * 70)
    print("EXPLORATORY 02: Score Abstracts with Kobak Marker Words")
    print("=" * 70)

    marker_words = get_marker_words()
    df = build_scored_dataset(marker_words)

    # Save (without abstract text to keep file size manageable)
    output_path = os.path.join(DATA_DIR, "scored_papers.csv")
    save_cols = [c for c in df.columns if c not in ["abstract", "referenced_works", "date"]]
    df[save_cols].to_csv(output_path, index=False)
    print(f"\n  Saved scored dataset: {output_path} ({len(df)} rows)")

    # Also save with abstracts for potential text analysis
    full_path = os.path.join(DATA_DIR, "scored_papers_with_abstracts.csv")
    save_cols_full = [c for c in df.columns if c not in ["date"]]
    df[save_cols_full].to_csv(full_path, index=False)
    print(f"  Saved full dataset (with abstracts): {full_path}")

    print("\n" + "=" * 70)
    print("Done. Abstracts scored.")
    print("=" * 70)


if __name__ == "__main__":
    main()
