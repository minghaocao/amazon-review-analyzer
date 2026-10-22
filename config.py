"""项目级路径与运行配置。"""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_NAME = "amazon_review_analyzer"
APP_VERSION = "1.0.1"
ANALYSIS_CANDIDATE_VERSION = "generalization_v2_candidate"
TAXONOMY_VERSION = ANALYSIS_CANDIDATE_VERSION
DEFAULT_RUNTIME_MODE = "stable"
DEFAULT_WORKERS = 1
DEFAULT_MODEL_PROVIDER = "ollama"
DEFAULT_MODEL_NAME = "qwen3:8b"
SUPPORTED_LANGUAGES = {
    "English": "en",
    "中文": "zh",
}
SUPPORTED_INPUT_MODES = (
    "demo",
    "amazon_quick_scan",
    "upload",
    "paste",
)
DEFAULT_LANGUAGE = "en"
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_REVIEWS_PATH = DATA_DIR / "raw_reviews.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ANALYZED_REVIEWS_PATH = OUTPUTS_DIR / "analyzed_reviews.csv"
ANALYSIS_CHECKPOINT_PATH = OUTPUTS_DIR / "analyzed_reviews_checkpoint.csv"
ANALYSIS_CHECKPOINT_META_PATH = OUTPUTS_DIR / "analyzed_reviews_checkpoint.meta.json"
ASPECT_SUMMARY_PATH = OUTPUTS_DIR / "aspect_summary.csv"
ASPECT_EVIDENCE_PATH = OUTPUTS_DIR / "aspect_evidence.json"
BUSINESS_SUMMARY_PATH = OUTPUTS_DIR / "business_summary.json"
KEYWORD_SUMMARY_PATH = OUTPUTS_DIR / "keyword_summary.csv"
SENTIMENT_KEYWORDS_PATH = OUTPUTS_DIR / "sentiment_keywords.json"
ASPECT_KEYWORDS_PATH = OUTPUTS_DIR / "aspect_keywords.json"
SESSION_CURRENT_DIR = OUTPUTS_DIR / "session" / "current"
AMAZON_CURRENT_DIR = OUTPUTS_DIR / "amazon" / "current"
COMPARISON_CURRENT_DIR = OUTPUTS_DIR / "comparison" / "current"
REPORTS_DIR = OUTPUTS_DIR / "reports"
REPORT_TEMPLATE_VERSION = "step14.report.v1"
TESTS_DIR = PROJECT_ROOT / "tests"
TAXONOMIES_DIR = PROJECT_ROOT / "taxonomies"

# The built-in interview dataset is frozen. Step 10B import and analysis code
# must reject every path in this collection as a write target.
DEMO_OUTPUT_PATHS = frozenset(
    {
        RAW_REVIEWS_PATH.resolve(),
        ANALYZED_REVIEWS_PATH.resolve(),
        ASPECT_SUMMARY_PATH.resolve(),
        BUSINESS_SUMMARY_PATH.resolve(),
        ASPECT_EVIDENCE_PATH.resolve(),
        KEYWORD_SUMMARY_PATH.resolve(),
        SENTIMENT_KEYWORDS_PATH.resolve(),
        ASPECT_KEYWORDS_PATH.resolve(),
    }
)

REQUIRED_DIRECTORIES = (
    DATA_DIR,
    OUTPUTS_DIR,
    TESTS_DIR,
    TAXONOMIES_DIR,
)


def _env_value(name: str, default: str) -> str:
    """读取非空环境变量，空值仍回退到项目默认值。"""

    return os.getenv(name, default).strip() or default


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive integer setting and fall back safely."""

    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


LLM_PROVIDER = _env_value("LLM_PROVIDER", DEFAULT_MODEL_PROVIDER).casefold()
OLLAMA_MODEL = _env_value("OLLAMA_MODEL", DEFAULT_MODEL_NAME)
OLLAMA_HOST = _env_value("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_KEEP_ALIVE = _env_value("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_NUM_CTX = _positive_int_env("OLLAMA_NUM_CTX", 4096)
BATCH_CHECKPOINT_INTERVAL = _positive_int_env("BATCH_CHECKPOINT_INTERVAL", 10)
PRODUCTION_CHECKPOINT_INTERVAL = _positive_int_env(
    "PRODUCTION_CHECKPOINT_INTERVAL",
    1,
)
MAX_CONSECUTIVE_LLM_FAILURES = _positive_int_env(
    "MAX_CONSECUTIVE_LLM_FAILURES",
    3,
)
MIN_ASPECT_REVIEWS_FOR_RANKING = _positive_int_env(
    "MIN_ASPECT_REVIEWS_FOR_RANKING",
    3,
)

# Step 12 Business Intelligence thresholds.  These labels communicate sample
# size and deterministic prioritization only; they are not statistical
# confidence intervals or significance tests.
BI_EVIDENCE_MEDIUM_MIN_REVIEWS = 3
BI_EVIDENCE_HIGH_MIN_REVIEWS = 10
BI_QUICK_SCAN_MAX_REVIEWS = 9
BI_FULL_INSIGHT_MIN_REVIEWS = 30
BI_HIGH_REVIEW_SHARE_PERCENT = 20.0
BI_HIGH_POSITIVE_RATE_PERCENT = 60.0
BI_HIGH_NEGATIVE_RATE_PERCENT = 50.0
BI_OPPORTUNITY_MEDIUM_SCORE = 8.0
BI_OPPORTUNITY_HIGH_SCORE = 20.0
BI_DEFAULT_EVIDENCE_LIMIT = 5
BI_EXPANDED_EVIDENCE_LIMIT = 25

# Step 13 deterministic competitor-comparison rules.  These thresholds are
# business heuristics for directional comparison, not statistical
# significance tests.  The comparison layer never invokes an LLM.
COMPARISON_MIN_SUPPORT = 3
COMPARISON_SLIGHT_GAP_PERCENT = 8.0
COMPARISON_CLEAR_GAP_PERCENT = 20.0
COMPARISON_UNEVEN_SAMPLE_RATIO = 3.0
COMPARISON_EARLY_SAMPLE_MAX_REVIEWS = 9
COMPARISON_EVIDENCE_MEDIUM_MIN_REVIEWS = 3
COMPARISON_EVIDENCE_HIGH_MIN_REVIEWS = 10
COMPARISON_WHITE_SPACE_NEGATIVE_RATE = 50.0
COMPARISON_LANGUAGE_MIN_REVIEW_COUNT = 2
COMPARISON_LANGUAGE_DISTINCTIVE_SHARE_GAP = 5.0
COMPARISON_DEFAULT_EVIDENCE_LIMIT = 5
COMPARISON_EXPANDED_EVIDENCE_LIMIT = 25
KEYWORD_MIN_REVIEW_COUNT = _positive_int_env(
    "KEYWORD_MIN_REVIEW_COUNT",
    2,
)

_DEFAULT_BENCHMARK_MODELS = ("qwen3:8b", "qwen3:4b-instruct")
OLLAMA_BENCHMARK_MODELS = tuple(
    model.strip()
    for model in os.getenv(
        "OLLAMA_BENCHMARK_MODELS",
        ",".join(_DEFAULT_BENCHMARK_MODELS),
    ).split(",")
    if model.strip()
) or _DEFAULT_BENCHMARK_MODELS

OPENAI_MODEL = _env_value("OPENAI_MODEL", "gpt-6-astra")
# 保留旧名称，避免现有外部调用在本阶段突然失效。
LLM_MODEL = OPENAI_MODEL

try:
    LLM_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1200"))
except ValueError:
    LLM_MAX_OUTPUT_TOKENS = 1200
