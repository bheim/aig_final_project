"""
Script 00: Verify that the OpenAlex field IDs in config.py are correct.

Queries the /fields endpoint and compares display names against our mapping.
Run this once before starting data collection to ensure correctness.

Output: prints verification results
"""

from config import FIELDS, FIELD_IDS, FIELD_SHORT
from utils import openalex_get


def main():
    print("=" * 60)
    print("SCRIPT 00: Verify OpenAlex Field IDs")
    print("=" * 60)

    # Fetch all fields from OpenAlex
    data = openalex_get("fields", {"per_page": 50})
    if not data or "results" not in data:
        print("  ERROR: Could not fetch fields from OpenAlex API.")
        print("  Check your internet connection and API configuration.")
        return

    # Build lookup from API
    api_fields = {}
    for item in data["results"]:
        api_fields[item["id"]] = item["display_name"]

    print(f"\n  OpenAlex has {len(api_fields)} fields total.\n")

    # Verify our mappings
    all_ok = True
    for field_name, field_id in FIELD_IDS.items():
        api_name = api_fields.get(field_id)
        if api_name is None:
            print(f"  FAIL: {field_id} not found in OpenAlex")
            all_ok = False
        elif api_name != field_name:
            print(f"  MISMATCH: config says '{field_name}' but API says '{api_name}' for {field_id}")
            print(f"    -> Update config.py FIELDS and FIELD_IDS accordingly")
            all_ok = False
        else:
            short = FIELD_SHORT.get(field_name, field_name)
            print(f"  OK: {field_id} -> '{field_name}' (short: '{short}')")

    if all_ok:
        print("\n  All field IDs verified successfully!")
    else:
        print("\n  WARNING: Some field IDs need correction in config.py")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
