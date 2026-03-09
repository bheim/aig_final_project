# Project Handoff: AI Adoption & Citation Behavior

## What This Project Does

Diff-in-diff estimating the impact of AI adoption on citation behavior of academic papers across 16 fields. Two temporal shocks: GPT-3.5 (Nov 2022) gives us field-level AI propensity scores, GPT-4o (May 2024) is the treatment event.

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
├── output/                   ← final results (regression tables, reports, figures)
├── exploratory/              ← standalone exploratory pipeline (Kobak marker word analysis)
└── venv/                     ← Python virtual environment
```

---

## Key Results (67,008 observations, 16 fields)

### Headline Findings

**Citation surprise (KL divergence) shows no treatment effect** — beta1 = +0.05, p = 0.52. However, **citation diversity measures show significant effects that grow over time:**

| DV | Level Effect (beta1) | p-value | Trend (per month) | Trend p |
|---|---|---|---|---|
| Log Surprise (KL) | +0.050 | 0.517 | +0.015 | 0.359 |
| Distinct Fields Cited | +1.072*** | <0.001 | +0.105** | 0.028 |
| Within-Field Share | −0.071** | 0.040 | −0.014** | 0.046 |
| Citation HHI | −0.097*** | <0.001 | −0.015*** | 0.005 |
| Reference Count | +10.10*** | 0.005 | +0.670 | 0.307 |
| Shannon Entropy | +0.209*** | <0.001 | +0.030*** | 0.005 |
| Other-Field Share | +0.042*** | 0.008 | −0.0004 | 0.887 |

**Interpretation:** AI-heavy fields are diversifying their citation patterns — citing more distinct fields, distributing citations more evenly, and becoming less insular. But they're branching into fields already somewhat represented in their baseline (hence no KL surprise effect). The diversity effect is growing over time, consistent with gradual diffusion of AI tools into research workflows.

### Output Files

| File | What It Contains |
|------|-----------------|
| `output/regression_results.txt` | Main regression table — 15 models (7 DVs × level/trend + 1 no-FE) |
| `output/regression_table.csv` | Same coefficients in CSV format |
| `output/robustness_results.txt` | Per-DV robustness batteries (9 specs each, +3 for log surprise) |
| `output/robustness_table.csv` | Same in CSV format |
| `output/event_study_results.txt` | Propensity × quarter interactions for all 7 DVs |
| `output/event_study_table.csv` | Same in CSV format |
| `output/dv_summary.csv` | One-line-per-DV summary |
| `output/parallel_trends_check.txt` | Pre-trend test |
| `output/propensity_scores_report.txt` | Validation of propensity scores |
| `output/fig_entropy_by_propensity.png` | Shannon entropy over time: high vs low propensity fields |
| `output/fig_event_study.png` | Event study coefficient plots (4 key DVs) |
| `data/propensity_scores.csv` | AI propensity score per field |
| `data/analysis_dataset.csv` | Full paper-level dataset (67,008 obs) |

### Propensity Score Rankings (highest AI adoption → lowest)

1. Computer Science (0.41)
2. Materials Science (0.35)
3. Business, Management and Accounting (0.32)
4. Economics, Econometrics and Finance (0.31)
5. Engineering (0.25)
6. Agricultural and Biological Sciences (0.25)
7. Environmental Science (0.24)
8. Neuroscience (0.23)
9. Chemistry (0.23)
10. Social Sciences (0.22)
11. Arts and Humanities (0.19)
12. Physics and Astronomy (0.19)
13. Mathematics (0.19)
14. Psychology (0.16)
15. Medicine (0.15)
16. Nursing (0.14)

---

## Scripts

All scripts live in `scripts/`. They run sequentially and checkpoint progress.

| Script | What It Does | Runtime |
|--------|-------------|---------|
| `00_verify_field_ids.py` | Validates OpenAlex field IDs | ~1s |
| `01_get_marker_words.py` | Downloads Kobak et al. marker word list | ~5s |
| `02_collect_phase1_abstracts.py` | Samples 1000 abstracts/field/window for propensity | ~10s |
| `03_compute_propensity_scores.py` | Computes AI propensity ratio + validation | ~1s |
| `04_collect_phase2_papers.py` | Samples papers + resolves reference fields | ~2-3 hrs |
| `05_compute_surprise.py` | Builds baseline, computes KL divergence + all DVs per paper | ~2s |
| `06_run_regression.py` | Runs diff-in-diff models for all 7 DVs + parallel trends check | ~5s |
| `07_robustness_checks.py` | Runs robustness specs using Model 2 | ~30s |
| `rerun_all.py` | Numpy-based output generator — all tables, robustness, event study, trend models | ~30s |

Utility scripts:

| Script | Purpose |
|--------|---------|
| `run_all.py` | Orchestrates scripts 00-06 sequentially |
| `config.py` | All settings (API key, fields, time windows, sample sizes) |
| `utils.py` | Shared helpers (API calls, pagination, checkpointing) |
| `debug_api.py` | Diagnostic tool for OpenAlex API connectivity |
| `fix_checkpoint.py` | One-time fix for rate-limit-poisoned checkpoint entries |

Figure generation scripts (in `output/`):

| Script | Output |
|--------|--------|
| `fig_entropy_by_propensity.py` | Shannon entropy time series, high vs low propensity |
| `fig_event_study.py` | Event study coefficient plots for 4 key DVs |

Run everything: `python run_all.py`
Run from a specific step: `python run_all.py --from 5`
Run one step only: `python run_all.py --only 6`
Regenerate output tables: `python rerun_all.py`

### Config

All settings are in `scripts/config.py`:

- `OPENALEX_API_KEY` — set via environment variable: `export OPENALEX_API_KEY=your_key`
- `FIELDS` / `FIELD_IDS` — the 16 fields and their OpenAlex numeric IDs
- `PHASE1_SAMPLE_SIZE` — abstracts per field per window (currently 1000)
- `PHASE2_PAPERS_PER_MONTH` — papers per field per month (currently 100)
- `POST_4O_DATE` — treatment cutoff (2024-11-01, with 6-month pub lag)

---

## Seven Dependent Variables

| DV | Column | Description |
|---|---|---|
| Log Surprise | `log_surprise` | log(KL divergence + 1e-6) — how unusual citations are relative to field baseline |
| Distinct Fields | `n_distinct_fields_cited` | Count of unique fields cited — extensive margin diversity |
| Within-Field Share | `within_field_share` | Fraction of refs to own field — citation insularity |
| Citation HHI | `citation_hhi` | Herfindahl index of citation field distribution — concentration |
| Reference Count | `n_references` | Total number of references |
| Shannon Entropy | `surprise_entropy` | Expected surprise of citation distribution — intensive margin diversity |
| Other-Field Share | `other_field_share` | Fraction of refs outside our 16 tracked fields |

---

## Data Files

| File | Description |
|------|-------------|
| `data/marker_words.txt` | 434 LLM style words |
| `data/propensity_scores.csv` | 16 rows, one per field |
| `data/phase1_abstracts.json` | ~32,000 abstracts for propensity scoring |
| `data/phase2_papers.json` | ~96,000 papers with metadata + references |
| `data/reference_fields.json` | Reference → field mappings |
| `data/phase2_with_surprise.csv` | Paper-level dataset with all DV columns |
| `data/analysis_dataset.csv` | Final regression-ready dataset (67,008 obs) |

Note: Large JSON files are gitignored. Regenerate by running the pipeline.

---

## Exploratory Pipeline

A separate standalone pipeline in `exploratory/` uses Kobak marker words to identify LLM-written abstracts and explore patterns across subfields, authors, institutions, etc. This is a non-causal descriptive analysis.

| Script | What It Does |
|--------|-------------|
| `exploratory/scripts/01_collect_papers.py` | Collects papers with abstracts + reference fields |
| `exploratory/scripts/02_score_abstracts.py` | Scores abstracts using Kobak marker words |
| `exploratory/scripts/03_compute_surprise.py` | Independent KL divergence computation |
| `exploratory/scripts/04_analyze.py` | Generates 11 figures + tables |
| `exploratory/scripts/run_all.py` | Orchestrator with --from and --only flags |

---

## If You Want to Re-Run or Modify

**Regenerate output tables:** `python rerun_all.py` — no API calls, reads analysis_dataset.csv.

**Change the regression spec:** Edit `06_run_regression.py` or `rerun_all.py` and run.

**Change surprise computation:** Edit `05_compute_surprise.py` and run `python run_all.py --from 5`.

**Change sample sizes or fields:** Edit `config.py`, delete relevant checkpoint files, re-run.

**Python environment:** `source venv/bin/activate`. Dependencies in `scripts/requirements.txt`.

---

## Known Limitations

- Kobak marker words developed on biomedical abstracts — may not transfer perfectly to all fields
- ~7.4% of references resolve to "Unknown" (excluded from surprise computation)
- Papers with <3 resolvable references are dropped
- AI propensity is measured at the field level, not paper level
- R² is modest (~7% for surprise, ~12-14% for diversity measures) — paper-level variation is inherently noisy
