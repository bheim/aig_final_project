"""
Shared utility functions for OpenAlex API interaction and data processing.
"""

import time
import requests
import json
import os
from config import OPENALEX_EMAIL, OPENALEX_API_KEY, OPENALEX_BASE, API_DELAY

# ─── Session Setup ──────────────────────────────────────────────────────────
_session = requests.Session()
_session.headers.update({"User-Agent": f"AIResearchProject/1.0 (mailto:{OPENALEX_EMAIL})"})


def openalex_params(**kwargs):
    """Build query params dict with polite pool email and optional API key."""
    params = dict(kwargs)
    params["mailto"] = OPENALEX_EMAIL
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    return params


def openalex_get(endpoint, params=None, max_retries=5):
    """
    Make a GET request to the OpenAlex API with rate limiting and retries.
    Returns the JSON response or None on failure.
    """
    url = f"{OPENALEX_BASE}/{endpoint}" if not endpoint.startswith("http") else endpoint
    if params is None:
        params = {}
    params = openalex_params(**params)

    for attempt in range(max_retries):
        try:
            time.sleep(API_DELAY)
            resp = _session.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                meta = data.get("meta", {})
                count = meta.get("count", "?")
                n_results = len(data.get("results", []))
                # Log first call per endpoint for debugging
                if attempt == 0 and n_results == 0:
                    print(f"  DEBUG: 0 results from {resp.url[:120]}...")
                    print(f"  DEBUG: meta={meta}")
                return data
            elif resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            elif resp.status_code == 403:
                print(f"  403 Forbidden: {resp.url[:200]}")
                print(f"  Body: {resp.text[:300]}")
                return None
            else:
                print(f"  HTTP {resp.status_code}: {resp.url[:200]}")
                print(f"  Body: {resp.text[:300]}")
                if attempt < max_retries - 1:
                    print(f"  Retrying ({attempt+1}/{max_retries})...")
                time.sleep(1)
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}. Retrying ({attempt+1}/{max_retries})...")
            time.sleep(2 ** attempt)

    print(f"  Failed after {max_retries} retries: {url}")
    return None


def paginate_openalex(endpoint, params, max_results=None, per_page=200):
    """
    Paginate through OpenAlex results using cursor-based pagination.
    Yields individual result items.
    """
    params = dict(params)
    params["per_page"] = per_page
    params["cursor"] = "*"
    total_yielded = 0

    while True:
        data = openalex_get(endpoint, params)
        if data is None:
            break

        results = data.get("results", [])
        if not results:
            break

        for item in results:
            yield item
            total_yielded += 1
            if max_results and total_yielded >= max_results:
                return

        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        params["cursor"] = next_cursor


def resolve_works_batch(work_ids, select_fields="id,primary_topic"):
    """
    Resolve a batch of OpenAlex work IDs in a single API call using the
    pipe-separated filter. Returns a dict mapping work_id -> work data.

    work_ids: list of OpenAlex work IDs (full URLs like 'https://openalex.org/W...')
    """
    if not work_ids:
        return {}

    # Extract short IDs
    short_ids = []
    for wid in work_ids:
        if isinstance(wid, str):
            short_ids.append(wid.replace("https://openalex.org/", ""))

    # OpenAlex filter supports pipe-separated IDs
    id_filter = "|".join(short_ids)
    results = {}

    for item in paginate_openalex("works", {
        "filter": f"openalex:{id_filter}",
        "select": select_fields,
    }, max_results=len(short_ids)):
        results[item["id"]] = item

    return results


def reconstruct_abstract(inverted_index):
    """Reconstruct abstract text from OpenAlex's inverted index format."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join([word for _, word in word_positions])


def count_marker_words_in_abstract(inverted_index, marker_words_set):
    """
    Count distinct marker words present in an abstract's inverted index.
    Much faster than reconstructing the full text.
    """
    if not inverted_index:
        return 0
    abstract_words = set(w.lower() for w in inverted_index.keys())
    return len(abstract_words.intersection(marker_words_set))


def has_any_marker_word(inverted_index, marker_words_set):
    """Check if abstract contains at least one marker word (binary measure)."""
    if not inverted_index:
        return False
    abstract_words = set(w.lower() for w in inverted_index.keys())
    return bool(abstract_words.intersection(marker_words_set))


def save_checkpoint(data, filepath):
    """Save intermediate results as JSON for resumability."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Checkpoint saved: {filepath}")


def load_checkpoint(filepath):
    """Load intermediate results from a checkpoint file."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


def get_field_filter(field_name):
    """Convert field display name to OpenAlex filter format."""
    return f"primary_topic.field.display_name:{field_name}"
