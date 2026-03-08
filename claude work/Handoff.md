# Project Handoff: AI Adoption & Research Surprise

## What This Project Does

Diff-in-diff estimating the impact of AI adoption on the "surprise" (citation novelty) of academic papers across 8 fields. Two temporal shocks: GPT-3.5 (Nov 2022) gives us field-level AI propensity scores, GPT-4o (May 2024) is the treatment event.

The full methodology is documented in:

- `Directions_for_OpenAlex_Data_Collection_Refined_1.md` — regression spec, variable definitions, identification strategy
- `Kobak_Implementation_Guide.md` — how the marker word / propensity score method works

---

## Folder Structure

```
claude work/
├── Handoff.md                ← you are here
├── Directions_for_...md      ← methodology doc (regression spec)
├── Kobak_Implementation...md ← propensity score methodology
├── scripts/                  ← all pipeline code (Python)
├── data/                     ← raw + intermediate data files
├── output/                   ← final results (regression tables, reports)
└── venv/                     ← Python virtual environment
```

---

## Key Results (Already Computed)

Everything has been run end-to-end. Here's where to find results:

| File | What It Contains |
|------|-----------------|
| `output/regression_results.txt` | Main regression table (3 models) |
| `output/regression_table.csv` | Same coefficients in CSV format |
| `output/parallel_trends_check.txt` | Pre-trend test (p=0.454, passes) |
| `output/propensity_scores_report.txt` | Validation of propensity scores |
| `data/propensity_scores.csv` | AI propensity score per field |
| `data/analysis_dataset.csv` | Full paper-level dataset (33,149 obs) ready for custom analysis |
| `data/phase2_with_surprise.csv` | Same data, before merging propensity scores |

### Headline Numbers

- **beta1 (preferred model, field×time FE):** 0.5649 (p<0.01) — higher-AI-exposure fields see significantly more surprising citation patterns post-4o
- **Parallel trends:** p=0.454, no differential pre-trend detected
- **N:** 33,149 papers across 8 fields, Jan 2021–Dec 2025

### Propensity Score Rankings (highest AI adoption → lowest)

1. Computer Science (0.41)
2. Business, Management and Accounting (0.32)
3. Agricultural and Biological Sciences (0.25)
4. Chemistry (0.23)
5. Arts and Humanities (0.19)
6. Physics and Astronomy (0.19)
7. Mathematics (0.19)
8. Psychology (0.16)

---

## Scripts

All scripts live in `scripts/`. They run sequentially and checkpoint progress, so any step can be re-run independently.

| Script | What It Does | Runtime |
|--------|-------------|---------|
| `00_verify_field_ids.py` | Validates OpenAlex field IDs | ~1s |
| `01_get_marker_words.py` | Downloads Kobak et al. marker word list, filters to ~434 LLM style words | ~5s |
| `02_collect_phase1_abstracts.py` | Samples 1000 abstracts/field/window from OpenAlex for propensity scores | ~10s |
| `03_compute_propensity_scores.py` | Computes AI propensity ratio + validation | ~1s |
| `04_collect_phase2_papers.py` | Samples 100 papers/field/month (Jan 2021–Dec 2025) + resolves reference fields | ~2-3 hrs |
| `05_compute_surprise.py` | Builds baseline co-citation distribution, computes KL divergence per paper | ~2s |
| `06_run_regression.py` | Runs 3 diff-in-diff models + parallel trends check | ~5s |

Run everything: `python run_all.py`
Run from a specific step: `python run_all.py --from 5`
Run one step only: `python run_all.py --only 6`

### Config

All settings are in `scripts/config.py`:

- `OPENALEX_API_KEY` — API key (required)
- `FIELDS` / `FIELD_IDS` — the 8 fields and their OpenAlex numeric IDs
- `PHASE1_SAMPLE_SIZE` — abstracts per field per window (currently 1000)
- `PHASE2_PAPERS_PER_MONTH` — papers per field per month (currently 100)
- `POST_4O_DATE` — treatment cutoff (2024-11-01, with 6-month pub lag)

---

## Data Files

| File | Size | Description |
|------|------|-------------|
| `data/marker_words.txt` | 4 KB | 434 LLM style words (one per line) |
| `data/marker_words.csv` | 13 KB | Same with Kobak annotations |
| `data/propensity_scores.csv` | <1 KB | 8 rows, one per field |
| `data/phase1_abstracts.json` | 33 MB | 16,000 abstracts (8 fields × 2 windows × 1000) |
| `data/phase2_papers.json` | 63 MB | ~48,000 papers with metadata + reference lists |
| `data/reference_fields.json` | 67 MB | ~1.1M reference → field mappings |
| `data/phase2_with_surprise.csv` | 4.6 MB | Paper-level dataset with surprise scores |
| `data/analysis_dataset.csv` | 6.2 MB | Final regression-ready dataset (with propensity merged in) |

---

## If You Want to Re-Run or Modify

**Change the regression spec:** Edit `06_run_regression.py` and run `python run_all.py --only 6`. The analysis dataset is already built — no API calls needed.

**Change surprise computation:** Edit `05_compute_surprise.py` and run `python run_all.py --from 5`.

**Change sample sizes or fields:** Edit `config.py`, delete relevant checkpoint files in `data/`, and re-run from the appropriate step.

**Python environment:** Activate with `source venv/bin/activate`. Dependencies are in `scripts/requirements.txt`.

---

## Known Limitations

- OpenAlex field names are their full ASJC names (e.g., "Agricultural and Biological Sciences" not "Biology"). The `FIELD_SHORT` dict in config.py maps to shorter labels.
- ~15% of references resolve to "Unknown" (works deleted or not in OpenAlex). These are excluded from surprise computation.
- Papers with fewer than 3 resolvable references are dropped from the surprise calculation.
- The Kobak marker word list was developed on biomedical abstracts — may not transfer perfectly to all fields, but the ratio-based propensity score accounts for field-specific baselines.
