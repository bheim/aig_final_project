"""
Script 04: Collect Phase 2 papers from OpenAlex for the regression.

Samples 100 papers per field per month from Jan 2021 through Dec 2025,
collecting paper metadata and referenced works. Then resolves the field
classification for each referenced work.

This is the most API-intensive script. It uses checkpointing and batch
resolution to minimize API calls.

Output: data/phase2_papers.json, data/reference_fields.json
"""

import os
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
from config import (
    DATA_DIR, FIELDS, FIELD_IDS,
    REGRESSION_START, REGRESSION_END,
    PHASE2_PAPERS_PER_MONTH, BATCH_SIZE,
)
from utils import (
    paginate_openalex, openalex_get,
    save_checkpoint, load_checkpoint,
)


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
    """Return the last day of a given month as YYYY-MM-DD."""
    if month == 12:
        return f"{year}-12-31"
    next_month = datetime(year, month + 1, 1)
    last = next_month - timedelta(days=1)
    return last.strftime("%Y-%m-%d")


def collect_papers_for_field_month(field, year, month, sample_size):
    """
    Collect a random sample of English journal articles for a given field and
    month. Uses OpenAlex `sample` parameter (single page, up to 200 results).
    Since sample_size=100, this fits in one request.
    """
    start_date = f"{year}-{month:02d}-01"
    end_date = last_day_of_month(year, month)

    field_id = FIELD_IDS[field]
    filter_str = (
        f"primary_topic.field.id:{field_id},"
        f"from_publication_date:{start_date},"
        f"to_publication_date:{end_date},"
        f"language:en,"
        f"type:article"
    )

    select_fields = (
        "id,publication_date,primary_topic,"
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

            papers.append({
                "id": item.get("id"),
                "publication_date": item.get("publication_date"),
                "field": (item.get("primary_topic") or {}).get("field", {}).get("display_name"),
                "subfield": (item.get("primary_topic") or {}).get("subfield", {}).get("display_name"),
                "referenced_works": item.get("referenced_works", []),
                "num_authors": num_authors,
                "institutions_distinct_count": item.get("institutions_distinct_count", 0),
                "countries_distinct_count": item.get("countries_distinct_count", 0),
                "type": item.get("type"),
                "language": item.get("language"),
            })

    return papers


def collect_all_papers():
    """Collect all Phase 2 papers across fields and months."""
    checkpoint_path = os.path.join(DATA_DIR, "phase2_papers_checkpoint.json")
    collected = load_checkpoint(checkpoint_path)
    if collected is None:
        collected = {}

    months = generate_months(REGRESSION_START, REGRESSION_END)
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
                field, year, month, PHASE2_PAPERS_PER_MONTH
            )
            collected[key] = papers

            # Checkpoint every 10 field-months
            if sum(1 for k in collected) % 10 == 0:
                save_checkpoint(collected, checkpoint_path)

    # Final checkpoint
    save_checkpoint(collected, checkpoint_path)
    return collected


def gather_all_reference_ids(papers_dict):
    """Extract all unique referenced work IDs across all papers."""
    all_refs = set()
    for key, papers in papers_dict.items():
        for paper in papers:
            for ref_id in paper.get("referenced_works", []):
                all_refs.add(ref_id)
    return all_refs


def resolve_reference_fields(all_ref_ids):
    """
    Resolve the field classification for all referenced works using batch queries.
    Returns dict mapping reference_id -> field_display_name.
    """
    checkpoint_path = os.path.join(DATA_DIR, "reference_fields_checkpoint.json")
    resolved = load_checkpoint(checkpoint_path)
    if resolved is None:
        resolved = {}

    # Filter out already-resolved IDs
    unresolved = [rid for rid in all_ref_ids if rid not in resolved]
    print(f"  Total unique references: {len(all_ref_ids)}")
    print(f"  Already resolved: {len(resolved)}")
    print(f"  To resolve: {len(unresolved)}")

    if not unresolved:
        return resolved

    # Process in batches
    batches = [unresolved[i:i+BATCH_SIZE] for i in range(0, len(unresolved), BATCH_SIZE)]
    consecutive_failures = 0

    for batch_idx, batch in enumerate(tqdm(batches, desc="  Resolving references")):
        # Build pipe-separated filter
        short_ids = [rid.replace("https://openalex.org/", "") for rid in batch]
        id_filter = "|".join(short_ids)

        data = openalex_get("works", {
            "filter": f"openalex:{id_filter}",
            "select": "id,primary_topic",
            "per_page": BATCH_SIZE,
        })

        if data is None:
            # API call failed (rate limit, network error, etc.)
            # Do NOT mark these as Unknown — leave them for retry on next run
            consecutive_failures += 1
            if consecutive_failures >= 5:
                # Sustained rate limit — wait 5 minutes then try again
                print(f"\n  Hit sustained rate limit ({consecutive_failures} consecutive failures).")
                print(f"  Saving checkpoint and waiting 5 minutes...")
                save_checkpoint(resolved, checkpoint_path)
                time.sleep(300)
                consecutive_failures = 0
            continue

        consecutive_failures = 0  # Reset on success

        if "results" in data:
            for item in data["results"]:
                work_id = item.get("id")
                topic = item.get("primary_topic") or {}
                field_info = topic.get("field", {})
                field_name = field_info.get("display_name", "Unknown")
                resolved[work_id] = field_name

            # Only mark IDs as Unknown if the API call SUCCEEDED but the
            # work wasn't in the results (meaning the work genuinely doesn't exist)
            returned_ids = set(item.get("id") for item in data["results"])
            for rid in batch:
                if rid not in resolved:
                    resolved[rid] = "Unknown"

        # Checkpoint periodically
        if (batch_idx + 1) % 50 == 0:
            save_checkpoint(resolved, checkpoint_path)

    # Final save
    save_checkpoint(resolved, checkpoint_path)
    return resolved


def main():
    print("=" * 60)
    print("SCRIPT 04: Collect Phase 2 Papers from OpenAlex")
    print("=" * 60)

    # Step 1: Collect papers
    print("\nStep 1: Collecting papers...")
    papers_dict = collect_all_papers()

    total_papers = sum(len(v) for v in papers_dict.values())
    print(f"\n  Total papers collected: {total_papers}")

    # Step 2: Resolve reference fields
    print("\nStep 2: Resolving reference field classifications...")
    all_ref_ids = gather_all_reference_ids(papers_dict)
    ref_fields = resolve_reference_fields(all_ref_ids)

    # Save final outputs
    papers_path = os.path.join(DATA_DIR, "phase2_papers.json")
    with open(papers_path, "w") as f:
        json.dump(papers_dict, f)
    print(f"\n  Papers saved: {papers_path}")

    ref_path = os.path.join(DATA_DIR, "reference_fields.json")
    with open(ref_path, "w") as f:
        json.dump(ref_fields, f)
    print(f"  Reference fields saved: {ref_path}")

    # Summary
    print("\n  Summary by field:")
    for field in FIELDS:
        n = sum(len(papers_dict[k]) for k in papers_dict if k.startswith(field))
        print(f"    {field}: {n} papers")

    ref_known = sum(1 for v in ref_fields.values() if v != "Unknown")
    print(f"\n  References resolved: {ref_known}/{len(ref_fields)}")

    print("\n" + "=" * 60)
    print("Done. Phase 2 data collection complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
