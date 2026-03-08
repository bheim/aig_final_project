"""
Diagnostic script: Test OpenAlex API connectivity and verify filters work.
Run this to debug data collection issues.
"""

import requests
import json
from config import OPENALEX_EMAIL, OPENALEX_API_KEY, OPENALEX_BASE, FIELD_IDS, FIELDS


def api_get(params):
    """Quick helper for test requests."""
    url = f"{OPENALEX_BASE}/works"
    params["mailto"] = OPENALEX_EMAIL
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    resp = requests.get(url, params=params, timeout=30)
    return resp


def test_basic_connectivity():
    print("=" * 60)
    print("TEST 1: Basic API connectivity")
    print("=" * 60)
    resp = api_get({"per_page": 1})
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Total works: {data.get('meta', {}).get('count', '?')}")
        print("  PASS")
    else:
        print(f"  FAIL: {resp.text[:300]}")
    print()


def test_field_id_formats():
    """Test different field ID format variants to find what works."""
    print("=" * 60)
    print("TEST 2: Field ID format variants")
    print("=" * 60)

    # Try multiple formats for Computer Science (field 17)
    variants = [
        ("Full URL", "primary_topic.field.id:https://openalex.org/fields/17"),
        ("Short ID (fields/17)", "primary_topic.field.id:fields/17"),
        ("Number only (17)", "primary_topic.field.id:17"),
        ("F-prefixed (F17)", "primary_topic.field.id:F17"),
        ("topics.field.id full URL", "topics.field.id:https://openalex.org/fields/17"),
        ("topics.field.id number", "topics.field.id:17"),
        ("primary_topic.domain.id (1)", "primary_topic.domain.id:https://openalex.org/domains/1"),
        ("primary_topic.domain.id number", "primary_topic.domain.id:1"),
    ]

    for label, filter_val in variants:
        full_filter = f"{filter_val},from_publication_date:2022-01-01,to_publication_date:2022-10-31,language:en,type:article"
        resp = api_get({"filter": full_filter, "per_page": 1})
        if resp.status_code == 200:
            count = resp.json().get("meta", {}).get("count", 0)
            status = "OK" if count > 0 else "EMPTY"
            print(f"  {status:5s} | {label:40s} | count={count:>10,}")
        else:
            print(f"  ERR  | {label:40s} | HTTP {resp.status_code}: {resp.text[:100]}")
    print()


def test_simple_topic_filter():
    """Test the simplest possible topic filter."""
    print("=" * 60)
    print("TEST 3: Simple filters (one at a time)")
    print("=" * 60)

    filters = [
        ("type:article", "type only"),
        ("language:en", "language only"),
        ("has_abstract:true", "has_abstract only"),
        ("from_publication_date:2022-01-01,to_publication_date:2022-10-31", "date range only"),
        ("primary_topic.field.id:https://openalex.org/fields/17", "field ID (full URL) only"),
        ("primary_topic.field.id:17", "field ID (number) only"),
    ]

    for filter_str, label in filters:
        resp = api_get({"filter": filter_str, "per_page": 1})
        if resp.status_code == 200:
            count = resp.json().get("meta", {}).get("count", 0)
            status = "OK" if count > 0 else "EMPTY"
            print(f"  {status:5s} | {label:40s} | count={count:>10,}")
        else:
            print(f"  ERR  | {label:40s} | HTTP {resp.status_code}: {resp.text[:150]}")
    print()


def test_display_name_filter():
    """Test filtering by display_name instead of ID."""
    print("=" * 60)
    print("TEST 4: Filter by display_name (alternative approach)")
    print("=" * 60)

    for field in FIELDS:
        # Try display_name search
        filter_str = f"primary_topic.field.display_name.search:{field},from_publication_date:2022-01-01,to_publication_date:2022-10-31,language:en,type:article"
        resp = api_get({"filter": filter_str, "per_page": 1})
        if resp.status_code == 200:
            count = resp.json().get("meta", {}).get("count", 0)
            status = "OK" if count > 0 else "EMPTY"
            print(f"  {status:5s} | {field:45s} | count={count:>10,}")
        else:
            # Try without .search
            filter_str2 = f"primary_topic.field.display_name:{field},from_publication_date:2022-01-01,to_publication_date:2022-10-31,language:en,type:article"
            resp2 = api_get({"filter": filter_str2, "per_page": 1})
            if resp2.status_code == 200:
                count = resp2.json().get("meta", {}).get("count", 0)
                print(f"  {'OK' if count > 0 else 'EMPTY':5s} | {field:45s} | count={count:>10,} (no .search)")
            else:
                print(f"  ERR  | {field:45s} | HTTP {resp.status_code}/{resp2.status_code}")
    print()


def test_list_fields():
    """List all available fields from OpenAlex."""
    print("=" * 60)
    print("TEST 5: List all OpenAlex fields")
    print("=" * 60)

    url = f"{OPENALEX_BASE}/fields"
    params = {"mailto": OPENALEX_EMAIL, "per_page": 50}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        for item in data.get("results", []):
            fid = item.get("id", "")
            name = item.get("display_name", "")
            count = item.get("works_count", 0)
            ours = " <-- OURS" if name in FIELDS else ""
            print(f"  {fid:45s} | {name:50s} | {count:>10,} works{ours}")
    else:
        print(f"  FAIL: HTTP {resp.status_code}: {resp.text[:300]}")
    print()


def main():
    print("\nOpenAlex API Diagnostic Tool v2")
    print(f"Email: {OPENALEX_EMAIL}")
    print(f"API Key: {'***' + OPENALEX_API_KEY[-4:] if OPENALEX_API_KEY else 'not set'}")
    print(f"Base URL: {OPENALEX_BASE}\n")

    test_basic_connectivity()
    test_list_fields()
    test_field_id_formats()
    test_simple_topic_filter()
    test_display_name_filter()

    print("=" * 60)
    print("Diagnostics complete. Use the working filter format above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
