"""
Run the full exploratory analysis pipeline.

Usage:
  python run_all.py           # Run all steps
  python run_all.py --from 2  # Start from step 2
  python run_all.py --only 4  # Only run step 4
"""

import sys
import subprocess

SCRIPTS = [
    ("01", "01_collect_papers.py",    "Collect papers with abstracts & references"),
    ("02", "02_score_abstracts.py",   "Score abstracts with Kobak marker words"),
    ("03", "03_compute_surprise.py",  "Compute citation surprise (KL divergence)"),
    ("04", "04_analyze.py",           "Exploratory analysis & figures"),
]


def main():
    from_step = 1
    only_step = None

    for arg in sys.argv[1:]:
        if arg.startswith("--from"):
            idx = sys.argv.index(arg)
            from_step = int(sys.argv[idx + 1])
        elif arg.startswith("--only"):
            idx = sys.argv.index(arg)
            only_step = int(sys.argv[idx + 1])

    for num, script, desc in SCRIPTS:
        step_num = int(num)
        if only_step is not None and step_num != only_step:
            continue
        if step_num < from_step:
            print(f"  Skipping step {num}: {desc}")
            continue

        print(f"\n{'='*70}")
        print(f"  STEP {num}: {desc}")
        print(f"{'='*70}\n")

        result = subprocess.run(
            [sys.executable, script],
            cwd=__import__("os").path.dirname(__import__("os").path.abspath(__file__))
        )
        if result.returncode != 0:
            print(f"\n  ERROR: Step {num} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print("\n" + "=" * 70)
    print("  ALL STEPS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
