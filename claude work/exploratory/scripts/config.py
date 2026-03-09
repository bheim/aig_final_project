"""
Configuration for the Exploratory LLM Usage Analysis.

This is a standalone analysis pipeline — separate from the diff-in-diff
causal analysis. It uses Kobak et al. (2025) marker words to identify
LLM-assisted papers and explores patterns across fields, subfields,
author characteristics, and citation surprise (KL divergence).
"""

import os

# ─── OpenAlex API ───────────────────────────────────────────────────────────
OPENALEX_EMAIL = "bheim@uchicago.edu"
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "")

OPENALEX_BASE = "https://api.openalex.org"

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Kobak Marker Words ────────────────────────────────────────────────────
# URL for the excess_words.csv from Kobak et al. (2025) GitHub repo
KOBAK_URL = (
    "https://raw.githubusercontent.com/berenslab/llm-excess-vocab/"
    "main/results/excess_words.csv"
)

# ─── Fields of Study ───────────────────────────────────────────────────────
# Same 16 fields as the causal analysis for consistency
FIELDS = [
    "Arts and Humanities",
    "Agricultural and Biological Sciences",
    "Business, Management and Accounting",
    "Chemistry",
    "Computer Science",
    "Mathematics",
    "Physics and Astronomy",
    "Psychology",
    "Medicine",
    "Engineering",
    "Social Sciences",
    "Economics, Econometrics and Finance",
    "Environmental Science",
    "Neuroscience",
    "Materials Science",
    "Nursing",
]

FIELD_SHORT = {
    "Arts and Humanities": "Arts & Humanities",
    "Agricultural and Biological Sciences": "Ag & Bio Sciences",
    "Business, Management and Accounting": "Business & Mgmt",
    "Chemistry": "Chemistry",
    "Computer Science": "Computer Science",
    "Mathematics": "Mathematics",
    "Physics and Astronomy": "Physics & Astro",
    "Psychology": "Psychology",
    "Medicine": "Medicine",
    "Engineering": "Engineering",
    "Social Sciences": "Social Sciences",
    "Economics, Econometrics and Finance": "Econ & Finance",
    "Environmental Science": "Environmental Sci",
    "Neuroscience": "Neuroscience",
    "Materials Science": "Materials Science",
    "Nursing": "Nursing",
}

FIELD_IDS = {
    "Arts and Humanities": "12",
    "Agricultural and Biological Sciences": "11",
    "Business, Management and Accounting": "14",
    "Chemistry": "16",
    "Computer Science": "17",
    "Mathematics": "26",
    "Physics and Astronomy": "31",
    "Psychology": "32",
    "Medicine": "27",
    "Engineering": "22",
    "Social Sciences": "33",
    "Economics, Econometrics and Finance": "20",
    "Environmental Science": "23",
    "Neuroscience": "28",
    "Materials Science": "25",
    "Nursing": "29",
}

# ─── Time Windows ──────────────────────────────────────────────────────────
# We collect papers from 2021-01 through 2025-12 to have good pre/post coverage
SAMPLE_START = "2021-01-01"
SAMPLE_END = "2025-12-31"

# Key dates for era classification
CHATGPT_DATE = "2022-12-01"       # ChatGPT public release
GPT4O_DATE = "2024-05-01"         # GPT-4o release

# Pre-ChatGPT baseline cutoff for citation surprise baseline
BASELINE_CUTOFF = "2022-11-01"

# ─── Sampling ──────────────────────────────────────────────────────────────
# Papers per field per month (random sample)
PAPERS_PER_FIELD_MONTH = 100

# ─── API Rate Limiting ─────────────────────────────────────────────────────
API_DELAY = 0.12   # ~8 requests per second
BATCH_SIZE = 50     # IDs per batch request
