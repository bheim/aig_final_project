"""
Script 02: Collect abstracts from OpenAlex for Phase 1 (AI propensity scores).

For each of the 8 fields, in both pre-period (Jan-Oct 2022) and post-period
(Apr 2023 - Apr 2024), randomly samples papers and retrieves their abstracts
(as inverted indices) for marker word counting.

Uses the OpenAlex `sample` parameter with multiple calls and deduplication
to achieve true random sampling beyond the 200-per-page limit.

Output: data/phase1_abstracts.json
"""

import os
import json
from config import (
    DATA_DIR, FIELDS, FIELD_IDS,
    PROPENSITY_PRE_START, PROPENSITY_PRE_END,
    PROPENSITY_POST_START, PROPENSITY_POST_END,
    PHASE1_SAMPLE_SIZE,
)
from utils import openalex_get, save_checkpoint, load_checkpoint


def collect_abstracts_for_field(field, date_start, date_end, sample_size):
    """
    Collect a random sample of English journal articles from OpenAlex.

    The `sample` parameter returns a single page of random results (max 200).
    To get more, we make multiple calls with different seeds and deduplicate.
    """
    field_id = FIELD_IDS[field]

    filter_str = (
        f"primary_topic.field.id:{field_id},"
        f"from_publication_date:{date_start},"
        f"to_publication_date:{date_end},"
        f"language:en,"
        f"type:article,"
        f"has_abstract:true"
    )

    seen_ids = set()
    papers = []
    per_page = 200
    max_attempts = (sample_size // per_page) * 3 + 5  # extra attempts for dedup

    for attempt in range(max_attempts):
        if len(papers) >= sample_size:
            break

        seed_val = attempt * 7919 + hash(f"{field}_{date_start}") % 100000
        remaining = sample_size - len(papers)

        data = openalex_get("works", {
            "filter": filter_str,
            "select": "id,publication_date,primary_topic,abstract_inverted_index,language",
            "sample": min(remaining, per_page),
            "seed": seed_val,
            "per_page": min(remaining, per_page),
        })

        if not data or "results" not in data or not data["results"]:
            print(f"    Attempt {attempt}: no results returned")
            if attempt == 0:
                # First attempt failed — likely a real problem
                print(f"    DEBUG: filter={filter_str}")
                print(f"    DEBUG: response meta={data.get('meta') if data else 'None'}")
            continue

        new_count = 0
        for item in data["results"]:
            work_id = item.get("id")
            if work_id not in seen_ids:
                seen_ids.add(work_id)
                papers.append({
                    "id": work_id,
                    "publication_date": item.get("publication_date"),
                    "field": field,
                    "abstract_inverted_index": item.get("abstract_inverted_index"),
                })
                new_count += 1

        print(f"    Attempt {attempt}: got {len(data['results'])} results, "
              f"{new_count} new (total: {len(papers)})")

        # If we got very few new results, the pool may be exhausted
        if new_count < 5 and attempt > 2:
            print(f"    Pool likely exhausted. Stopping at {len(papers)} papers.")
            break

    return papers


def main():
    print("=" * 60)
    print("SCRIPT 02: Collect Phase 1 Abstracts from OpenAlex")
    print("=" * 60)

    checkpoint_path = os.path.join(DATA_DIR, "phase1_abstracts_checkpoint.json")
    output_path = os.path.join(DATA_DIR, "phase1_abstracts.json")

    # Try to resume from checkpoint
    collected = load_checkpoint(checkpoint_path)
    if collected is None:
        collected = {"pre": {}, "post": {}}

    windows = {
        "pre": (PROPENSITY_PRE_START, PROPENSITY_PRE_END),
        "post": (PROPENSITY_POST_START, PROPENSITY_POST_END),
    }

    total_tasks = len(FIELDS) * 2
    completed = sum(
        1 for w in windows for f in FIELDS
        if len(collected.get(w, {}).get(f, [])) > 0
    )
    print(f"  Progress: {completed}/{total_tasks} field-windows already collected.\n")

    for window_name, (date_start, date_end) in windows.items():
        if window_name not in collected:
            collected[window_name] = {}

        for field in FIELDS:
            existing = collected.get(window_name, {}).get(field, [])
            if len(existing) > 0:
                print(f"  [{window_name}] {field}: already collected ({len(existing)} papers). Skipping.")
                continue

            print(f"  [{window_name}] {field}: collecting up to {PHASE1_SAMPLE_SIZE} random papers...")
            papers = collect_abstracts_for_field(
                field, date_start, date_end, PHASE1_SAMPLE_SIZE
            )
            collected[window_name][field] = papers
            print(f"    -> Final: {len(papers)} papers")

            if len(papers) == 0:
                print(f"    WARNING: 0 papers returned! Check API connectivity and field ID.")

            save_checkpoint(collected, checkpoint_path)

    # Save final output
    with open(output_path, "w") as f:
        json.dump(collected, f)
    print(f"\n  Final output saved: {output_path}")

    # Summary
    print("\n  Summary:")
    for window_name in ["pre", "post"]:
        print(f"  {window_name.upper()} period:")
        for field in FIELDS:
            n = len(collected.get(window_name, {}).get(field, []))
            print(f"    {field}: {n} papers")

    print("\n" + "=" * 60)
    print("Done. Phase 1 abstracts collected.")
    print("=" * 60)


if __name__ == "__main__":
    main()
