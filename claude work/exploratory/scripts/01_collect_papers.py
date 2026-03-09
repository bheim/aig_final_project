"""
Script 01: Collect papers from OpenAlex with abstracts and references.

For each field × month, samples PAPERS_PER_FIELD_MONTH random English journal
articles. Collects:
  - Paper metadata (authors, institutions, countries, subfield)
  - Abstract (inverted index → reconstructed text)
  - Referenced works (for computing citation surprise later)

Checkpointed: safe to interrupt and resume.

Output: data/papers_checkpoint.json
"""

import os
import json
import time
from datetime import datetime, timedelta
from tqdm import tqdm

from config import (
    DATA_DIR, FIELDS, FIELD_IDS,
    SAMPLE_START, SAMPLE_END,
    PAPERS_PER_FIELD_MONTH, BATCH_SIZE,
)
from utils import openalex_get, save_checkpoint, load_checkpoint


def generate_months(start_date, end_date):
    """Generate (year, month) tuples between start and end dates."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    months = []
    current = start.replace(day=1)
    while current <= end:
        months.append((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def last_day_of_month(year, month):
    if month == 12:
        return f"{year}-12-31"
    next_month = datetime(year, month + 1, 1)
    last = next_month - timedelta(days=1)
    return last.strftime("%Y-%m-%d")


def collect_papers_for_field_month(field, year, month, sample_size):
    """
    Collect a random sample of English journal articles for a given
    field and month. Includes abstract_inverted_index and referenced_works.
    """
    start_date = f"{year}-{month:02d}-01"
    end_date = last_day_of_month(year, month)

    field_id = FIELD_IDS[field]
    filter_str = (
        f"primary_topic.field.id:{field_id},"
        f"from_publication_date:{start_date},"
        f"to_publication_date:{end_date},"
        f"language:en,"
        f"type:article,"
        f"has_abstract:true"  # Only papers with abstracts
    )

    select_fields = (
        "id,publication_date,primary_topic,"
        "abstract_inverted_index,"
        "referenced_works,"
        "authorships,institutions_distinct_count,countries_distinct_count,"
        "type,language"
    )

    seed_val = year * 100 + month
    data = openalex_get("works", {
        "filter": filter_str,
        "select": select_fields,
        "sample": sample_size,
        "seed": seed_val,
        "per_page": sample_size,
    })

    papers = []
    if data and "results" in data:
        for item in data["results"]:
            authorships = item.get("authorships", [])
            num_authors = len(authorships) if authorships else 0

            # Reconstruct abstract from inverted index
            aii = item.get("abstract_inverted_index")
            abstract = _reconstruct_abstract(aii)

            papers.append({
                "id": item.get("id"),
                "publication_date": item.get("publication_date"),
                "field": (item.get("primary_topic") or {}).get("field", {}).get("display_name"),
                "subfield": (item.get("primary_topic") or {}).get("subfield", {}).get("display_name"),
                "abstract": abstract,
                "referenced_works": item.get("referenced_works", []),
                "num_authors": num_authors,
                "institutions_distinct_count": item.get("institutions_distinct_count", 0),
                "countries_distinct_count": item.get("countries_distinct_count", 0),
                "type": item.get("type"),
                "language": item.get("language"),
            })

    return papers


def _reconstruct_abstract(inverted_index):
    """Convert OpenAlex abstract_inverted_index to plain text."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join([w for _, w in word_positions])


def collect_all_papers():
    """Collect all papers across fields and months with checkpointing."""
    checkpoint_path = os.path.join(DATA_DIR, "papers_checkpoint.json")
    collected = load_checkpoint(checkpoint_path)
    if collected is None:
        collected = {}

    months = generate_months(SAMPLE_START, SAMPLE_END)
    total_tasks = len(FIELDS) * len(months)
    completed = sum(1 for f in FIELDS for m in months
                    if f"{f}_{m[0]}_{m[1]:02d}" in collected)
    print(f"  Progress: {completed}/{total_tasks} field-months already collected.\n")

    for field in FIELDS:
        for year, month in tqdm(months, desc=f"  {field}", leave=True):
            key = f"{field}_{year}_{month:02d}"
            if key in collected:
                continue

            papers = collect_papers_for_field_month(
                field, year, month, PAPERS_PER_FIELD_MONTH
            )
            collected[key] = papers

            # Checkpoint every 10 field-months
            if sum(1 for k in collected) % 10 == 0:
                save_checkpoint(collected, checkpoint_path)

    # Final save
    save_checkpoint(collected, checkpoint_path)
    return collected


def resolve_reference_fields(papers_dict):
    """
    Resolve the field classification for all referenced works.
    Checkpointed separately since this is the most API-intensive step.
    """
    checkpoint_path = os.path.join(DATA_DIR, "reference_fields_checkpoint.json")
    ref_fields = load_checkpoint(checkpoint_path)
    if ref_fields is None:
        ref_fields = {}

    # Gather all unique reference IDs
    all_refs = set()
    for key, papers in papers_dict.items():
        for p in papers:
            for ref_id in p.get("referenced_works", []):
                all_refs.add(ref_id)

    # Filter out already resolved
    unresolved = [rid for rid in all_refs if rid not in ref_fields]
    print(f"\n  Total unique references: {len(all_refs)}")
    print(f"  Already resolved: {len(ref_fields)}")
    print(f"  Remaining: {len(unresolved)}")

    if not unresolved:
        print("  All references already resolved!")
        return ref_fields

    # Batch resolve
    total_batches = (len(unresolved) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(unresolved), BATCH_SIZE):
        batch = unresolved[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        short_ids = [rid.replace("https://openalex.org/", "") for rid in batch]
        id_filter = "|".join(short_ids)

        data = openalex_get("works", {
            "filter": f"openalex:{id_filter}",
            "select": "id,primary_topic",
            "per_page": BATCH_SIZE,
        })

        if data and "results" in data:
            for item in data["results"]:
                wid = item.get("id", "")
                pt = item.get("primary_topic") or {}
                field_name = pt.get("field", {}).get("display_name", "Unknown")
                ref_fields[wid] = field_name

        # Mark missing as Unknown
        for rid in batch:
            if rid not in ref_fields:
                ref_fields[rid] = "Unknown"

        if batch_num % 50 == 0 or batch_num == total_batches:
            save_checkpoint(ref_fields, checkpoint_path)
            print(f"  Batch {batch_num}/{total_batches} — "
                  f"{len(ref_fields)} references resolved")

    save_checkpoint(ref_fields, checkpoint_path)
    return ref_fields


def main():
    print("=" * 70)
    print("EXPLORATORY 01: Collect Papers with Abstracts and References")
    print("=" * 70)

    papers_dict = collect_all_papers()

    total_papers = sum(len(p) for p in papers_dict.values())
    abstracts_found = sum(
        1 for papers in papers_dict.values()
        for p in papers if p.get("abstract")
    )
    print(f"\n  Total papers collected: {total_papers}")
    print(f"  Papers with abstracts: {abstracts_found}")

    # Resolve reference fields for surprise computation
    print("\n── Resolving reference fields ──")
    ref_fields = resolve_reference_fields(papers_dict)

    print(f"\n  Reference fields resolved: {len(ref_fields)}")

    print("\n" + "=" * 70)
    print("Done. Data ready for scoring and analysis.")
    print("=" * 70)


if __name__ == "__main__":
    main()
