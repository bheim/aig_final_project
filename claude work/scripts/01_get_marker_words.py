"""
Script 01: Download and filter the Kobak et al. (2025) marker word list.

Downloads excess_words.csv from the berenslab GitHub repo, filters for
LLM style words (post-2022), and saves the cleaned marker word list.

Output: data/marker_words.csv, data/marker_words.txt
"""

import os
import pandas as pd
import requests
from config import DATA_DIR

EXCESS_WORDS_URL = (
    "https://raw.githubusercontent.com/berenslab/llm-excess-vocab/"
    "main/results/excess_words.csv"
)


def download_excess_words():
    """Download the full excess words CSV from Kobak et al. GitHub."""
    print("Downloading excess_words.csv from berenslab/llm-excess-vocab...")
    resp = requests.get(EXCESS_WORDS_URL, timeout=30)
    resp.raise_for_status()

    raw_path = os.path.join(DATA_DIR, "excess_words_raw.csv")
    with open(raw_path, "wb") as f:
        f.write(resp.content)
    print(f"  Saved raw file: {raw_path}")
    return raw_path


def filter_marker_words(raw_path):
    """
    Filter the excess words list to keep only LLM style markers.

    Filtering criteria (from Kobak_Implementation_Guide.md):
    1. Keep only words annotated as 'style' (not content, not ambiguous)
    2. Keep only words whose excess year is 2023 or 2024
    """
    df = pd.read_csv(raw_path)

    print(f"\n  Raw excess words: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # --- Explore the data structure ---
    # Column names may vary; let's be flexible
    # Expected columns: word, annotation/category (style/content), year

    # Identify the annotation column
    annotation_col = None
    for col in df.columns:
        unique_vals = df[col].astype(str).str.lower().unique()
        if any("style" in v for v in unique_vals):
            annotation_col = col
            break

    if annotation_col is None:
        # Fallback: check for common column names
        for candidate in ["annotation", "category", "type", "label", "class"]:
            if candidate in df.columns:
                annotation_col = candidate
                break

    if annotation_col:
        print(f"  Annotation column: '{annotation_col}'")
        print(f"  Unique values: {df[annotation_col].unique()}")
    else:
        print("  WARNING: Could not identify annotation column.")
        print(f"  Available columns: {list(df.columns)}")
        print("  Printing first 5 rows for manual inspection:")
        print(df.head())
        print("\n  Proceeding without annotation filter (will use all words).")

    # Identify the year column
    year_col = None
    for col in df.columns:
        if "year" in col.lower():
            year_col = col
            break
    # Fallback: find a column with values in 2013-2024 range
    if year_col is None:
        for col in df.columns:
            try:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals) > 0 and vals.min() >= 2010 and vals.max() <= 2025:
                    year_col = col
                    break
            except Exception:
                continue

    if year_col:
        print(f"  Year column: '{year_col}'")
    else:
        print("  WARNING: Could not identify year column.")

    # Identify the word column
    word_col = None
    for candidate in ["word", "term", "token", "excess_word"]:
        if candidate in [c.lower() for c in df.columns]:
            word_col = [c for c in df.columns if c.lower() == candidate][0]
            break
    if word_col is None:
        # Use first column as default
        word_col = df.columns[0]
    print(f"  Word column: '{word_col}'")

    # --- Apply filters ---
    mask = pd.Series([True] * len(df))

    # Filter 1: style words only
    if annotation_col:
        style_mask = df[annotation_col].astype(str).str.lower().str.contains("style")
        mask = mask & style_mask
        print(f"\n  After style filter: {mask.sum()} words")

    # Filter 2: year >= 2023 (excess words that emerged after ChatGPT)
    if year_col:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        year_mask = df[year_col] >= 2023
        mask = mask & year_mask
        print(f"  After year >= 2023 filter: {mask.sum()} words")

    filtered = df[mask].copy()
    marker_words = filtered[word_col].str.lower().str.strip().unique().tolist()
    marker_words.sort()

    print(f"\n  Final marker word count: {len(marker_words)}")
    print(f"  Sample words: {marker_words[:20]}")

    return marker_words, filtered, word_col


def save_marker_words(marker_words, filtered_df, word_col):
    """Save the filtered marker word list."""
    # Save as simple text file (one word per line)
    txt_path = os.path.join(DATA_DIR, "marker_words.txt")
    with open(txt_path, "w") as f:
        for word in marker_words:
            f.write(word + "\n")
    print(f"\n  Saved marker word list: {txt_path}")

    # Save the filtered DataFrame with annotations
    csv_path = os.path.join(DATA_DIR, "marker_words.csv")
    filtered_df.to_csv(csv_path, index=False)
    print(f"  Saved annotated marker words: {csv_path}")

    return txt_path, csv_path


def main():
    print("=" * 60)
    print("SCRIPT 01: Get Kobak et al. Marker Words")
    print("=" * 60)

    raw_path = download_excess_words()
    marker_words, filtered_df, word_col = filter_marker_words(raw_path)
    save_marker_words(marker_words, filtered_df, word_col)

    print("\n" + "=" * 60)
    print(f"Done. {len(marker_words)} marker words ready for Phase 1.")
    print("=" * 60)


if __name__ == "__main__":
    main()
