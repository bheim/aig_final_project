# Directions for OpenAlex Data Collection

## Introduction

We are performing a regression analysis to estimate the impact of AI adoption on the **surprise level** of academic research output.

Our identification strategy uses two temporal shocks:

1. **GPT-3.5 release (November 2022):** We use the Kobak et al. (2025) method to measure the uptick in LLM-associated marker words in published papers before and after this release. This gives us a field-level **AI propensity score** capturing how heavily each field adopted AI tools.
2. **GPT-4o release (May 2024):** We use this as our treatment event. With AI propensity scores already measured from the 3.5 era, we study how the surprise level of papers shifts after 4o's introduction in fields with higher versus lower AI exposure.

By measuring AI exposure in the 3.5 window and estimating outcomes in the 4o window, we avoid the simultaneity problem of using post-treatment data to define treatment intensity.

---

## Outcome Variable: Surprise

**Surprise** measures how unlikely the combination of cited fields is for a given paper. The intuition: if a computer science paper cites references from English, anthropology, and mathematics, that is a highly unusual combination and thus a "surprising" paper. If it only cites other CS papers, that is unsurprising.

### Formal Definition

We operationalize surprise using **Kullback-Leibler divergence** of a paper's reference field distribution relative to a pre-shock baseline for its field.

**Step 1 — Construct a baseline co-citation distribution.** Using all papers published before November 2022 (pre-shock), compute, for each field $f$, the empirical probability distribution over fields of cited references. That is, for field $f$, calculate $P_f(g)$ = the share of references in field-$f$ papers that point to papers in field $g$. This is the expected citation profile for a paper in field $f$.

**Step 2 — For each focal paper $i$ in field $f$, compute its reference field distribution.** Let $q_i(g)$ be the observed share of paper $i$'s references that belong to field $g$.

**Step 3 — Compute surprise.** Calculate the Kullback-Leibler divergence (or, alternatively, Shannon entropy) of the focal paper's reference distribution relative to the baseline:

$$
\text{Surprise}_i = D_{KL}(q_i \| P_f) = \sum_g q_i(g) \log \frac{q_i(g)}{P_f(g)}
$$

Higher values indicate that the paper's citation pattern is more unusual relative to what is typical for its field.

**Important:** The baseline distribution $P_f(g)$ must be computed using only pre-shock (pre-November 2022) data so that it is not contaminated by AI-influenced citation behavior.

---

## Fields

We study the following 8 fields, defined using the `primary_topic.field.display_name` variable in OpenAlex:

1. Arts & Humanities
2. Biology
3. Business & Management
4. Chemistry
5. Computer Science
6. Mathematics
7. Physics & Astronomy
8. Psychology

---

## Phase 1: Estimating the AI Propensity Score

### Purpose

Produce a single scalar for each of the 8 fields capturing how heavily that field adopted LLM tools after GPT-3.5.

### Time Windows

We apply a **6-month publication lag** to account for the delay between paper submission and publication:

- **Pre-period:** January 2022 – October 2022 (papers unlikely to reflect GPT-3.5 use)
- **Post-period:** April 2023 – April 2024 (papers plausibly written with GPT-3.5 access, but before 4o)

### Method

Following Kobak et al. (2025), identify a set of marker words whose usage frequency increased sharply in LLM-generated text (e.g., "delve," "intricate," "furthermore," "noteworthy" — use the specific word list from Kobak et al.).

For each field $f$:

1. Compute the average frequency of marker words per paper in the **pre-period**: $\bar{w}_f^{\text{pre}}$
2. Compute the average frequency of marker words per paper in the **post-period**: $\bar{w}_f^{\text{post}}$
3. Compute the AI propensity score as the ratio:

$$
\text{AIPropensity}_f = \frac{\bar{w}_f^{\text{post}} - \bar{w}_f^{\text{pre}}}{\bar{w}_f^{\text{pre}}}
$$

A value of 0 means no change in marker word usage (the field shows no detectable AI adoption); a value of 0.5 means a 50% increase. This centering at zero ensures that the interaction term $\text{AIPropensity}_f \times D_t^{\text{after 4o}}$ contributes nothing to the regression for fields unaffected by AI.

### Data Requirements for This Phase

For each field, in each time window, collect a sample of papers. For each paper, we need:

- `primary_topic.field.display_name` (to assign field)
- Full text or abstract text (to compute marker word frequencies)
- `publication_date` (to assign to the correct time window)

### Note on Full Text Availability

If full text is not reliably available through OpenAlex, abstracts can be used instead. The key is consistency: use the same text source (abstract or full text) across all fields and time periods so the ratio is not biased by differential text availability.

---

## Phase 2: The Regression

### Sample

For each of the 8 fields, sample **100 random papers per month** from **January 2021 through December 2025**. This yields approximately 48,000 observations (8 fields × 60 months × 100 papers).

Restrict the sample to journal articles (i.e., `type` = "article") to maintain comparability across fields and time periods.

### Treatment Variable

The treatment is the interaction of field-level AI propensity with a post-4o indicator:

$$
\text{Treatment}_{ft} = \text{AIPropensity}_f \times D_{t}^{\text{after 4o}}
$$

where $D_{t}^{\text{after 4o}} = 1$ for papers published after **November 2024** (applying the same 6-month publication lag used in the propensity score construction; GPT-4o was released May 2024).

### Regression Specification

$$
\text{Surprise}_{ift} = \alpha + \beta_1 (\text{AIPropensity}_f \times D_t^{\text{after 4o}}) + \gamma X_{ift} + \delta_f + \theta_t + \phi_{ft} + \varepsilon_{ift}
$$

Where:

- $\text{Surprise}_{ift}$ = surprise score for paper $i$ in field $f$ at time $t$
- $\beta_1$ = **coefficient of interest** — captures how surprise shifts in higher-exposure fields after 4o, relative to lower-exposure fields
- $X_{ift}$ = paper-level controls (see below)
- $\delta_f$ = field fixed effects
- $\theta_t$ = time (month-year) fixed effects
- $\phi_{ft}$ = field × time fixed effects (absorb field-specific trends)
- $\varepsilon_{ift}$ = error term

**Note on fixed effects:** Including both $\delta_f$, $\theta_t$, and $\phi_{ft}$ means the field and time main effects are absorbed. The field × time interactions are important because they control for field-specific trends in surprise that might exist independent of AI (e.g., if CS was already becoming more interdisciplinary over time). With $\phi_{ft}$ in the model, the AIPropensity main effect is absorbed, so $\beta_1$ is identified purely from the differential shift at the 4o threshold.

### Paper-Level Controls ($X_{ift}$)

Include only variables that are **predetermined or not causally affected by AI adoption**:

| Variable | OpenAlex Field | Rationale |
|---|---|---|
| Number of authors | `num_authors` (derived from `authorships`) | Team size affects novelty; plausibly predetermined |
| Number of distinct institutions | `institutions_distinct_count` | Institutional collaboration breadth |
| Number of distinct countries | `countries_distinct_count` | International collaboration |
| Publication type | `type` | If not restricting to articles only |
| Language | `language` | Controls for accessibility differences |

### Variables to Exclude from Controls

Do **not** control for any of the following, as they are post-treatment outcomes or could be causally downstream of AI use:

| Variable | Reason to Exclude |
|---|---|
| `cited_by_count` | This is an outcome, not a control |
| `fwci` | Derived from citations; outcome |
| `citation_normalized_percentile.value` | Derived from citations; outcome |
| `open_access.is_oa` / `open_access.oa_status` | AI might change where researchers submit |
| `referenced_works_count` | AI might change how many references a paper includes |
| `has_fulltext` | Correlated with OA status and venue choice |

---

## Phase 2 Data Requirements (Per Paper)

For each of the sampled papers, collect:

| Variable | OpenAlex Field | Purpose |
|---|---|---|
| Paper ID | `id` | Unique identifier |
| Publication date | `publication_date` | Time assignment, constructing $D_t^{\text{after 4o}}$ |
| Field | `primary_topic.field.display_name` | Field assignment |
| Subfield | `primary_topic.subfield.display_name` | Optional robustness check |
| Referenced works | `referenced_works` | **Critical:** needed to compute surprise |
| Number of authors | `num_authors` | Control |
| Distinct institutions | `institutions_distinct_count` | Control |
| Distinct countries | `countries_distinct_count` | Control |
| Publication type | `type` | Control / sample restriction |
| Language | `language` | Control |

**For each referenced work** (i.e., each item in `referenced_works`), we also need:

| Variable | OpenAlex Field | Purpose |
|---|---|---|
| Referenced paper ID | `id` | Link to the reference |
| Field of referenced paper | `primary_topic.field.display_name` | **Critical:** needed to compute surprise |

This is the most data-intensive part of the collection. Each focal paper may have 20–50+ references, and for each reference we need its field classification. Plan API calls accordingly.

---

## Summary of Steps

1. **Collect marker word data** for all 8 fields in the pre-period (Jan–Oct 2022) and post-period (Apr 2023–Apr 2024). Compute AI propensity scores as the ratio of marker word frequency.
2. **Construct the baseline co-citation distribution** using all pre-November 2022 papers across the 8 fields.
3. **Sample 100 papers per field per month** from Jan 2021–Dec 2025. For each paper, collect the variables listed above.
4. **For each sampled paper, retrieve the field classification of every referenced work** to compute the surprise measure.
5. **Compute surprise** for each paper using KL divergence against the pre-shock baseline.
6. **Run the regression** with the specification above.

---

## Limitations to Acknowledge

- **AI propensity is measured at the field level, not the paper level.** We are estimating the effect of being in a high-exposure field, not the direct effect of an individual paper using AI.
- **The propensity score is static.** We measure it once (in the 3.5 window) and hold it fixed. In reality, field-level AI adoption evolves.
- **Parallel trends assumption.** We assume that, absent 4o, surprise would have evolved similarly across high- and low-exposure fields. This should be validated by showing pre-4o trends in surprise are parallel across fields with different AI propensity scores.
- **4o is not the only shock in mid-2024.** Other AI models (Gemini, Llama 3, etc.) were also released around this time. Our estimate captures the combined effect of the capability jump in LLMs, not 4o specifically.
- **Citation lag for surprise.** Surprise is computed from references, which are determined at submission time. The 6-month lag accounts for this, but there is inherent imprecision.
