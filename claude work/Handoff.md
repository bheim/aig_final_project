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

Everything has been run end-to-end, including robustness checks. Here's where to find results:

| File | What It Contains |
|------|-----------------|
| `output/regression_results.txt` | Main regression table (3 models) |
| `output/regression_table.csv` | Same coefficients in CSV format |
| `output/robustness_results.txt` | 13 robustness checks (all significant) |
| `output/robustness_table.csv` | Same in CSV format |
| `output/parallel_trends_check.txt` | Pre-trend test (p=0.454, passes) |
| `output/propensity_scores_report.txt` | Validation of propensity scores |
| `data/propensity_scores.csv` | AI propensity score per field |
| `data/analysis_dataset.csv` | Full paper-level dataset (33,149 obs) ready for custom analysis |
| `data/phase2_with_surprise.csv` | Same data, before merging propensity scores |

### Headline Numbers

- **beta1 (preferred model, field×time FE):** 0.5649 (p<0.01) — higher-AI-exposure fields see significantly more surprising citation patterns post-4o
- **Parallel trends:** p=0.454, no differential pre-trend detected
- **N:** 33,149 papers across 8 fields, Jan 2021–Dec 2025
- **Robustness:** 13/13 specifications return significant positive beta1

### Propensity Score Rankings (highest AI adoption → lowest)

1. Computer Science (0.41)
2. Business, Management and Accounting (0.32)
3. Agricultural and Biological Sciences (0.25)
4. Chemistry (0.23)
5. Arts and Humanities (0.19)
6. Physics and Astronomy (0.19)
7. Mathematics (0.19)
8. Psychology (0.16)

### Robustness Summary

All 13 specifications significant at p<0.05, 12 at p<0.01:

| Specification | beta1 | p-value |
|---|---|---|
| Baseline | 0.5649 | <0.001 |
| Log(surprise) | 1.8461 | <0.001 |
| Winsorized (95th) | 0.4380 | <0.001 |
| Winsorized (99th) | 0.5303 | <0.001 |
| Binary propensity | 0.6093 | <0.001 |
| Drop Arts & Humanities | 0.5236 | <0.001 |
| Drop Psychology | 0.5624 | <0.001 |
| Drop both | 0.5183 | <0.001 |
| Shannon entropy | 0.2670 | 0.016 |
| Cutoff: Aug 2024 | 0.5681 | <0.001 |
| Cutoff: Feb 2025 | 0.5434 | <0.001 |
| Min 5 refs | 0.5156 | <0.001 |
| Min 10 refs | 0.5231 | <0.001 |

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
| `07_robustness_checks.py` | Runs 13 robustness specifications (no API calls) | ~30s |

Utility scripts:

| Script | Purpose |
|--------|---------|
| `run_all.py` | Orchestrates scripts 00-06 sequentially |
| `config.py` | All settings (API key, fields, time windows, sample sizes) |
| `utils.py` | Shared helpers (API calls, pagination, checkpointing) |
| `debug_api.py` | Diagnostic tool for OpenAlex API connectivity issues |
| `fix_checkpoint.py` | One-time fix for rate-limit-poisoned checkpoint entries |

Run everything: `python run_all.py`
Run from a specific step: `python run_all.py --from 5`
Run one step only: `python run_all.py --only 6`
Run robustness checks: `python 07_robustness_checks.py`

### Config

All settings are in `scripts/config.py`:

- `OPENALEX_API_KEY` — set via environment variable: `export OPENALEX_API_KEY=your_key`
- `FIELDS` / `FIELD_IDS` — the 8 fields and their OpenAlex numeric IDs (use numbers only, not full URLs)
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

Note: The large JSON files are gitignored. Regenerate them by running the pipeline from the relevant step.

---

## If You Want to Re-Run or Modify

**Run robustness checks:** `python 07_robustness_checks.py` — no API calls, uses existing data.

**Change the regression spec:** Edit `06_run_regression.py` and run `python run_all.py --only 6`. The analysis dataset is already built — no API calls needed.

**Change surprise computation:** Edit `05_compute_surprise.py` and run `python run_all.py --from 5`.

**Change sample sizes or fields:** Edit `config.py`, delete relevant checkpoint files in `data/`, and re-run from the appropriate step.

**Python environment:** Activate with `source venv/bin/activate`. Dependencies are in `scripts/requirements.txt`.

---

## Interpretive Notes

**The sign flip between models:** beta1 is insignificant and slightly negative in Models 1-2 (no FE / field+time FE), but flips to significantly positive in Model 3 (field×time FE). This means the effect is identified from within-field-month variation after absorbing field-specific trends, not from raw cross-field comparisons. Worth discussing in the paper.

**Raw surprise diffs vs regression:** The simple pre/post surprise difference by field is actually weakly negatively correlated with propensity (-0.20). Mathematics had the largest raw surprise increase despite low propensity. The regression result emerges after controlling for field-specific time trends.

**Surprise distribution:** Highly right-skewed (skewness 3.5, kurtosis 20). The log-transform robustness check (beta=1.85, p<0.001) confirms the result isn't driven by outliers.

**Arts & Humanities sample:** Averages only 34 obs/month (target was 100). Dropping it doesn't change the result (beta=0.52 vs 0.56).

**Propensity score measurement:** The ratio and binary propensity measures have low rank correlation (0.52), but the result holds with either measure.

---

## Known Limitations

- OpenAlex field names are their full ASJC names (e.g., "Agricultural and Biological Sciences" not "Biology"). The `FIELD_SHORT` dict in config.py maps to shorter labels.
- ~7.4% of references resolve to "Unknown" (works deleted or not in OpenAlex). These are excluded from surprise computation.
- Papers with fewer than 3 resolvable references are dropped from the surprise calculation.
- The Kobak marker word list was developed on biomedical abstracts — may not transfer perfectly to all fields, but the ratio-based propensity score accounts for field-specific baselines.
- R² is low (3.7% in Model 3). Surprise is inherently noisy — most variation is idiosyncratic to individual papers.
