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
TRIAL1_DATA = PROCESSED_DATA / "trial1"
TRIAL2_DATA = PROCESSED_DATA / "trial2"

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
