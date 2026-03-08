"""
Master runner: Execute all pipeline scripts in sequence.

Usage:
    python run_all.py             # Run everything
    python run_all.py --from 3    # Start from script 03
    python run_all.py --only 5    # Run only script 05
"""

import sys
import subprocess
import time
import os

SCRIPTS = [
    ("00", "00_verify_field_ids.py", "Verify OpenAlex field IDs"),
    ("01", "01_get_marker_words.py", "Download & filter Kobak marker words"),
    ("02", "02_collect_phase1_abstracts.py", "Collect Phase 1 abstracts from OpenAlex"),
    ("03", "03_compute_propensity_scores.py", "Compute AI propensity scores"),
    ("04", "04_collect_phase2_papers.py", "Collect Phase 2 papers + references"),
    ("05", "05_compute_surprise.py", "Compute surprise (KL divergence)"),
    ("06", "06_run_regression.py", "Run diff-in-diff regression"),
]


def run_script(script_path, description):
    """Run a single script and report timing."""
    print(f"\n{'#' * 70}")
    print(f"# Running: {description}")
    print(f"# Script:  {script_path}")
    print(f"{'#' * 70}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nERROR: {script_path} exited with code {result.returncode}")
        print(f"Elapsed: {elapsed:.1f}s")
        sys.exit(1)

    print(f"\nCompleted in {elapsed:.1f}s")
    return elapsed


def main():
    start_from = 0
    only = None

    # Parse args
    args = sys.argv[1:]
    if "--from" in args:
        idx = args.index("--from")
        start_from = int(args[idx + 1])
    if "--only" in args:
        idx = args.index("--only")
        only = int(args[idx + 1])

    print("=" * 70)
    print("AI-SURPRISE RESEARCH PIPELINE")
    print("=" * 70)

    total_time = 0
    for num, script, desc in SCRIPTS:
        script_num = int(num)
        if only and script_num != only:
            continue
        if script_num < start_from:
            print(f"  Skipping {num}: {desc}")
            continue

        elapsed = run_script(script, desc)
        total_time += elapsed

    print(f"\n{'=' * 70}")
    print(f"ALL DONE. Total elapsed: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
