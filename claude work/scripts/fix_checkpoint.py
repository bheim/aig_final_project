"""
One-time fix: Remove "Unknown" entries from the reference_fields checkpoint
that were incorrectly written during rate-limit failures.

The old code marked references as "Unknown" even when the API call failed
(429 rate limit), not just when the work genuinely didn't exist. This script
removes those entries so they'll be retried on the next run.

Run this ONCE before resuming Script 04.
"""

import json
import os
from config import DATA_DIR

checkpoint_path = os.path.join(DATA_DIR, "reference_fields_checkpoint.json")

print("Loading checkpoint...")
with open(checkpoint_path, "r") as f:
    resolved = json.load(f)

total = len(resolved)
unknowns = sum(1 for v in resolved.values() if v == "Unknown")
known = total - unknowns

print(f"  Total entries: {total:,}")
print(f"  Known (real fields): {known:,}")
print(f"  Unknown: {unknowns:,}")

# We can't tell which Unknowns are real vs rate-limit artifacts.
# But we know the rate limit started around batch 18981 (of 22776).
# Each batch = 50 IDs, so ~18981*50 = 949,050 IDs were processed before failure.
# Conservatively: remove ALL Unknowns and let the API re-resolve them.
# Real unknowns (deleted works, etc.) are a small fraction and will just
# get re-marked as Unknown on the next run.

print(f"\nRemoving all {unknowns:,} Unknown entries for re-resolution...")
cleaned = {k: v for k, v in resolved.items() if v != "Unknown"}

print(f"  Cleaned entries: {len(cleaned):,}")

# Save backup first
backup_path = checkpoint_path + ".backup"
with open(backup_path, "w") as f:
    json.dump(resolved, f)
print(f"  Backup saved: {backup_path}")

# Save cleaned checkpoint
with open(checkpoint_path, "w") as f:
    json.dump(cleaned, f)
print(f"  Cleaned checkpoint saved: {checkpoint_path}")

print(f"\nDone. {unknowns:,} entries will be re-resolved on next run.")
print("Now run: python run_all.py --from 4")
