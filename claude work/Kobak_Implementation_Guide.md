# Implementing the Kobak Measure for AI Propensity Scores

## Overview

This document provides implementation instructions for computing field-level AI propensity scores using the excess vocabulary method from Kobak et al. (2025). The goal is to measure how heavily each of our 8 fields adopted LLM tools after GPT-3.5, producing a single scalar per field that will serve as the cross-sectional variation in our diff-in-diff regression.

**Reference:** Kobak, D., González-Márquez, R., Horvát, E.-Á., & Lause, J. (2025). Delving into LLM-assisted writing in biomedical publications through excess vocabulary. *Science Advances*, 11(27), eadt3813.

**Code and data:** https://github.com/berenslab/llm-excess-vocab — the full list of 900 excess words with annotations is available at `results/excess_words.csv`.

---

## Background on the Method

Kobak et al. studied vocabulary changes across 15+ million PubMed abstracts from 2010–2024. Their key insight is that certain **style words** — words unrelated to scientific content — sharply increased in frequency after ChatGPT's release. These words serve as LLM fingerprints because they reflect writing style preferences baked into the models, not changes in research topics.

The method distinguishes between two types of excess words:

- **Content words** (e.g., "covid," "pandemic," "remdesivir"): these spike when a topic becomes popular and are *not* useful as LLM markers.
- **Style words** (e.g., "delve," "intricate," "meticulous," "showcasing," "pivotal"): these are topic-independent and spiked specifically in 2023–2024, making them reliable LLM markers.

Kobak et al. identified **379 excess style words** in 2024 with highly elevated frequencies. We will use a curated subset of these style words as our marker word list.

---

## Step 1: Obtain the Marker Word List

Download the excess words list from the Kobak et al. GitHub repository:

```
https://github.com/berenslab/llm-excess-vocab/blob/main/results/excess_words.csv
```

This CSV contains all 900 excess words identified from 2013 to 2024, along with annotations classifying each word as **content** or **style** and the year in which the word became an excess word.

**Filter the list as follows:**

1. Keep only words annotated as **style** (not content, not ambiguous).
2. Keep only words whose excess year is **2023 or 2024** (i.e., words that became excess after ChatGPT's release).

This filtered list is our set of LLM marker words. It should contain approximately 200–379 words. Examples of words that should appear on the list include: delve, intricate, meticulously, pivotal, showcasing, realm, underscore, noteworthy, commendable, grappling, multifaceted, nuanced, paramount, comprehensive, foster, harness, navigate, crucial, elevate, illuminate, embark, emphasize, facilitate, bolster, surpass, landscape, tapestry, testament, vibrant, compelling, innovative, transformative, underscore, leveraging, streamline, robust, advent, burgeoning.

**Important:** The Kobak word list was developed on biomedical (PubMed) abstracts. Some words may not transfer perfectly to all 8 of our fields (e.g., Arts & Humanities may have different baseline usage of words like "intricate" or "nuanced"). This is an acknowledged limitation, but because we are computing a *ratio* of post/pre frequencies within each field, field-specific baseline differences are accounted for — what matters is the *change*.

---

## Step 2: Collect Abstract Data from OpenAlex

### Fields

Collect data for each of the following 8 fields, identified by the `primary_topic.field.display_name` variable in OpenAlex:

1. Arts & Humanities
2. Biology
3. Business & Management
4. Chemistry
5. Computer Science
6. Mathematics
7. Physics & Astronomy
8. Psychology

### Time Windows

We use two time windows with a 6-month publication lag to account for the delay between writing/submission and publication:

| Window | Date Range | Rationale |
|---|---|---|
| **Pre-period** | January 2022 – October 2022 | Papers written and submitted before GPT-3.5 existed |
| **Post-period** | April 2023 – April 2024 | Papers plausibly written with GPT-3.5 access, before GPT-4o |

### Sampling

For each field × time window combination, collect a sample of papers. We recommend sampling **at least 500 papers per field per window** (more is better for precision). If the field has fewer than 500 papers in a window, take all available papers.

For each paper, retrieve:

| Variable | OpenAlex Field | Purpose |
|---|---|---|
| Paper ID | `id` | Unique identifier |
| Publication date | `publication_date` | Assign to correct time window |
| Field | `primary_topic.field.display_name` | Assign to correct field |
| Abstract | `abstract_inverted_index` | Text for marker word counting |
| Language | `language` | Filter to English only |

### Filtering

- **English only:** Restrict to papers where `language` = "en". Non-English abstracts will have different word frequency distributions and will bias the marker word counts.
- **Articles only:** Restrict to `type` = "article" for consistency (exclude reviews, preprints, editorials, etc., as these may have different writing patterns).

### Note on OpenAlex Abstracts

OpenAlex stores abstracts as an **inverted index** (a dictionary mapping words to their positions in the abstract). To reconstruct the full abstract text, you need to invert this index. Example:

```python
def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join([word for _, word in word_positions])
```

For our purposes — counting whether specific marker words appear — you can skip reconstruction and simply check whether each marker word exists as a key in the inverted index. This is computationally much faster:

```python
def check_marker_words(inverted_index, marker_words):
    """Returns the count of distinct marker words present in the abstract."""
    if not inverted_index:
        return 0
    abstract_words = set(w.lower() for w in inverted_index.keys())
    return len(abstract_words.intersection(marker_words))
```

---

## Step 3: Compute the AI Propensity Score

### Method

For each field $f$, compute:

**3a. Pre-period marker word frequency.** For all sampled papers in field $f$ in the pre-period (Jan–Oct 2022), compute the average number of distinct marker words per abstract:

$$
\bar{w}_f^{\text{pre}} = \frac{1}{N_f^{\text{pre}}} \sum_{i \in \text{pre}} m_i
$$

where $m_i$ is the number of distinct marker words found in abstract $i$, and $N_f^{\text{pre}}$ is the number of pre-period papers in field $f$.

**3b. Post-period marker word frequency.** Same computation for the post-period (Apr 2023–Apr 2024):

$$
\bar{w}_f^{\text{post}} = \frac{1}{N_f^{\text{post}}} \sum_{i \in \text{post}} m_i
$$

**3c. Compute the ratio.**

$$
\text{AIPropensity}_f = \frac{\bar{w}_f^{\text{post}} - \bar{w}_f^{\text{pre}}}{\bar{w}_f^{\text{pre}}}
$$

### Interpretation

- A value of **0** means no change in marker word usage — the field shows no detectable LLM adoption.
- A value of **0.3** means marker word frequency increased by 30% — moderate LLM adoption.
- A value of **1.0** means marker word frequency doubled — heavy LLM adoption.

This centering at zero is important for the regression. Because the propensity score enters the regression as an interaction ($\text{AIPropensity}_f \times D_t^{\text{after 4o}}$), a field with zero AI adoption produces an interaction term of 0, so $\beta_1$ has no impact on predicted surprise for that field. If we used a raw ratio (where no change = 1.0), the interaction term would be nonzero even for unaffected fields, biasing the estimate.

### Alternative: Binary Frequency

Instead of counting distinct marker words per abstract, you could compute the **fraction of abstracts containing at least one marker word** in each period and take the ratio. This aligns more closely with Kobak et al.'s frequency gap approach, which uses binary occurrence (does the abstract contain the word or not?). We recommend computing both and checking whether field rankings are consistent. Use whichever produces more intuitive variation across fields.

---

## Step 4: Validation Checks

Before using the propensity scores in the regression, run the following sanity checks:

### 4a. Do the scores make intuitive sense?

We would expect fields like Computer Science, Mathematics, and Business & Management to have higher AI propensity scores (more text-generation-compatible work), while fields like Chemistry and Physics & Astronomy (more lab/experiment-dependent) should have lower scores. Arts & Humanities could go either way. If the rankings are counterintuitive, investigate whether the marker word list is picking up field-specific jargon rather than LLM usage.

### 4b. Pre-trend stability

Compute marker word frequencies for 2020 and 2021 (well before ChatGPT). The pre-period frequency $\bar{w}_f^{\text{pre}}$ should be roughly stable from 2020 to 2022 within each field. If there's already a rising trend in marker words before ChatGPT, that suggests the words are picking up a secular change in writing style, not LLM usage.

### 4c. Variation across fields

Ensure there is meaningful variation in the propensity scores across the 8 fields. If all fields have scores clustered tightly around, say, 0.05–0.10, there may not be enough cross-sectional variation to power the diff-in-diff. If variation is too low, consider expanding the marker word list or using a different aggregation method (e.g., weighting words by their excess frequency ratio from Kobak et al.).

### 4d. Robustness to word list choice

Try computing scores using subsets of the marker word list (e.g., top 50 words by excess frequency, top 100, all 379) and check that the relative ranking of fields is stable across these choices.

---

## Summary

| Step | Input | Output |
|---|---|---|
| 1. Get marker words | Kobak et al. GitHub CSV | Filtered list of ~200–379 LLM style words |
| 2. Collect abstracts | OpenAlex API | ~500+ abstracts per field × 2 time windows |
| 3. Compute scores | Abstracts + marker words | 8 scalar AI propensity scores (one per field) |
| 4. Validate | Propensity scores | Sanity checks on rankings, variation, and stability |

The output of this process — 8 AI propensity scores — feeds directly into the main regression specification described in `Directions_for_OpenAlex_Data_Collection_Refined.md`.
