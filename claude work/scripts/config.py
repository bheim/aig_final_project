"""
Shared configuration for the AI-Surprise research pipeline.
Edit the settings below before running any scripts.
"""

import os

# ─── OpenAlex API ───────────────────────────────────────────────────────────
# Polite pool: include your email for higher rate limits.
# If you have a premium API key, set it here as well.
OPENALEX_EMAIL = "bheim@uchicago.edu"
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "")  # Set via: export OPENALEX_API_KEY=your_key

# Base URL for the OpenAlex API
OPENALEX_BASE = "https://api.openalex.org"

# ─── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "..", "output")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Fields of Study ───────────────────────────────────────────────────────
# Display names we use internally (must match OpenAlex display_name output)
FIELDS = [
    # ── Original 8 fields ──
    "Arts and Humanities",
    "Agricultural and Biological Sciences",
    "Business, Management and Accounting",
    "Chemistry",
    "Computer Science",
    "Mathematics",
    "Physics and Astronomy",
    "Psychology",
    # ── Added 8 fields (for cross-sectional power) ──
    "Medicine",
    "Engineering",
    "Social Sciences",
    "Economics, Econometrics and Finance",
    "Environmental Science",
    "Neuroscience",
    "Materials Science",
    "Nursing",
]

# Short labels for display / regression output
FIELD_SHORT = {
    "Arts and Humanities": "Arts & Humanities",
    "Agricultural and Biological Sciences": "Biology",
    "Business, Management and Accounting": "Business & Management",
    "Chemistry": "Chemistry",
    "Computer Science": "Computer Science",
    "Mathematics": "Mathematics",
    "Physics and Astronomy": "Physics & Astronomy",
    "Psychology": "Psychology",
    "Medicine": "Medicine",
    "Engineering": "Engineering",
    "Social Sciences": "Social Sciences",
    "Economics, Econometrics and Finance": "Economics & Finance",
    "Environmental Science": "Environmental Sci.",
    "Neuroscience": "Neuroscience",
    "Materials Science": "Materials Science",
    "Nursing": "Nursing",
}

# OpenAlex field IDs — use NUMBER ONLY (not full URL)
# Full URLs like "https://openalex.org/fields/17" return 0 results in filters.
# Verify at runtime with: https://api.openalex.org/fields
FIELD_IDS = {
    # ── Original 8 ──
    "Arts and Humanities": "12",
    "Agricultural and Biological Sciences": "11",
    "Business, Management and Accounting": "14",
    "Chemistry": "16",
    "Computer Science": "17",
    "Mathematics": "26",
    "Physics and Astronomy": "31",
    "Psychology": "32",
    # ── Added 8 ──
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
# Phase 1: AI propensity score windows
PROPENSITY_PRE_START = "2022-01-01"
PROPENSITY_PRE_END = "2022-10-31"
PROPENSITY_POST_START = "2023-04-01"
PROPENSITY_POST_END = "2024-04-30"

# Phase 2: Regression sample window
REGRESSION_START = "2021-01-01"
REGRESSION_END = "2025-12-31"

# Post-4o indicator date (with 6-month publication lag from May 2024)
POST_4O_DATE = "2024-11-01"

# ─── Sampling ──────────────────────────────────────────────────────────────
# Phase 1: abstracts per field per time window
PHASE1_SAMPLE_SIZE = 1000  # >=500 recommended

# Phase 2: papers per field per month
PHASE2_PAPERS_PER_MONTH = 100

# ─── API Rate Limiting ─────────────────────────────────────────────────────
# Seconds to wait between API requests (polite pool allows ~10 req/s)
API_DELAY = 0.12  # ~8 requests per second
BATCH_SIZE = 50    # Number of work IDs to resolve per batch request
