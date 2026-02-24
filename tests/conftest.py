"""Shared fixtures for hailo_llm_deploy tests."""

import json
from pathlib import Path

import pytest

from hailo_llm_deploy.config import PipelineConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_config() -> PipelineConfig:
    """Return a PipelineConfig with all defaults."""
    return PipelineConfig()


@pytest.fixture
def korean_legal_yaml() -> Path:
    """Return path to the real korean_legal.yaml config."""
    return PROJECT_ROOT / "configs" / "examples" / "korean_legal.yaml"


@pytest.fixture
def default_yaml() -> Path:
    """Return path to the real default.yaml config."""
    return PROJECT_ROOT / "configs" / "default.yaml"


@pytest.fixture
def sample_yaml(tmp_path) -> Path:
    """Write a minimal YAML config file and return its path."""
    content = "model:\n  name: test-model\n  max_seq_length: 512\n"
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def sample_test_data() -> list[dict]:
    """Return a small list of test data dicts."""
    return [
        {
            "instruction": "You are a helpful assistant.",
            "input": "What is AI?",
            "output": "AI is artificial intelligence.",
        },
        {
            "instruction": "You are a helpful assistant.",
            "input": "What is ML?",
            "output": "ML is machine learning.",
        },
        {
            "instruction": "You are a helpful assistant.",
            "input": "What is NLP?",
            "output": "NLP is natural language processing.",
        },
    ]


@pytest.fixture
def sample_jsonl(tmp_path, sample_test_data) -> Path:
    """Write sample test data as JSONL and return the path."""
    path = tmp_path / "test_data.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for item in sample_test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def mock_trial_dirs(tmp_path) -> Path:
    """Create trial subdirectories and return the parent path."""
    (tmp_path / "trial1").mkdir()
    (tmp_path / "trial2").mkdir()
    (tmp_path / "notrial").mkdir()
    return tmp_path
