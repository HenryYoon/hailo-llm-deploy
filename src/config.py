"""
Centralized path configuration for the legal chatbot project.
"""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA = DATA_ROOT / "raw"
PROCESSED_DATA = DATA_ROOT / "processed"
METADATA = PROCESSED_DATA / "metadata"
EXTERNAL_DATA = DATA_ROOT / "external"
EXTERNAL_STATUTES = EXTERNAL_DATA / "statutes"
EXTERNAL_CASES = EXTERNAL_DATA / "cases"

# Raw data files
RAW_CSV = RAW_DATA / "law_qa_v1.csv"

# Processed data files
SAMPLED_TRAIN = PROCESSED_DATA / "sampled_train.csv"
SAMPLED_VAL = PROCESSED_DATA / "sampled_val.csv"
SAMPLED_TEST = PROCESSED_DATA / "sampled_test.csv"
TRAIN_DATASET = PROCESSED_DATA / "train_dataset.jsonl"
VAL_DATASET = PROCESSED_DATA / "val_dataset.jsonl"
TEST_DATASET = PROCESSED_DATA / "test_dataset.jsonl"

# Metadata files
SAMPLING_STATS = METADATA / "sampling_stats.json"
REFERENCE_EXTRACTION = METADATA / "reference_extraction.json"
CHUNK_DATABASE = METADATA / "chunk_database.json"

# Model directories
MODEL_ROOT = PROJECT_ROOT / "models"
CHECKPOINTS = MODEL_ROOT / "checkpoints"
LORA_ADAPTERS = MODEL_ROOT / "lora_adapters"
MERGED_MODELS = MODEL_ROOT / "merged"
ONNX_MODELS = MODEL_ROOT / "onnx"

# Infrastructure directories
INFRA_ROOT = PROJECT_ROOT / "infra"
DOCKER_CONFIG = INFRA_ROOT / "docker" / "docker-compose.yml"
ENV_FILE = INFRA_ROOT / "config" / ".env"
HAILO_DIR = INFRA_ROOT / "hailo"

# Development directories
NOTEBOOKS = PROJECT_ROOT / "notebooks"
LOGS = PROJECT_ROOT / "logs"

# Valid major labels (20 categories from KLAC dataset)
VALID_MAJOR_LABELS = [
    '노동', '주택임대차', '상가임대차', '손해배상', '민사일반', '물권', '채권', '계약', '상사',
    '민사소송', '친족', '상속', '가족관계등록', '민사집행', '보전처분', '개인회생, 파산 및 면책',
    '형법', '형사소송', '행정', '헌법'
]

# Sampling parameters
TRAIN_SIZE = 1000
VAL_SIZE = 100
TEST_SIZE = 100
RANDOM_SEED = 42

# Chunking parameters
CHUNK_SIZE = 300       # characters (Korean)
CHUNK_OVERLAP = 50     # characters

# RAFT parameters
DOCS_PER_SAMPLE = 5
MAX_ORACLE_DOCS = 2
DISTRACTOR_SAME_LABEL_RATIO = 0.6  # 60% same major_label

# API configuration (set via environment variable or .env)
LAW_API_OC = "yoonihg"  # 국가법령정보센터 OC (user ID)
API_DELAY = 0.5  # seconds between API requests
