"""Tests for hailo_llm_deploy.config — Pydantic models and YAML loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from hailo_llm_deploy.config import (
    CompileConfig,
    DataConfig,
    DeployConfig,
    EvaluateConfig,
    ExportConfig,
    LoraConfig,
    ModelConfig,
    OutputConfig,
    PipelineConfig,
    TrainingConfig,
)


class TestModelConfig:
    def test_defaults(self):
        c = ModelConfig()
        assert c.name == "Qwen/Qwen2.5-3B-Instruct"
        assert c.max_seq_length == 2048
        assert c.load_in_4bit is True


class TestLoraConfig:
    def test_defaults(self):
        c = LoraConfig()
        assert c.r == 16
        assert c.alpha == 16
        assert c.dropout == 0
        assert len(c.target_modules) == 7
        assert "q_proj" in c.target_modules
        assert "down_proj" in c.target_modules


class TestTrainingConfig:
    def test_defaults(self):
        c = TrainingConfig()
        assert c.epochs == 10
        assert c.batch_size == 2
        assert c.gradient_accumulation == 4
        assert c.learning_rate == 2e-4
        assert c.warmup_ratio == 0.03
        assert c.scheduler == "cosine"
        assert c.weight_decay == 0.01
        assert c.prompt_format == "chatml"
        assert c.seed == 42


class TestDataConfig:
    def test_defaults(self):
        c = DataConfig()
        assert c.train_path is None
        assert c.val_path is None
        assert c.test_path is None
        assert c.instruction is None


class TestExportConfig:
    def test_defaults(self):
        c = ExportConfig()
        assert c.format == "onnx"
        assert c.dtype == "float16"
        assert c.output_dir == Path("./output")


class TestCompileConfig:
    def test_defaults(self):
        c = CompileConfig()
        assert c.har_path is None
        assert c.alls_path is None
        assert c.hw_arch == "hailo10h"
        assert c.adapter_name == "lora_adapter"
        assert c.lora_weights_path is None
        assert c.calibration_data_path is None
        assert c.calibration_size == 64
        assert c.output_dir == Path("./models/compiled")
        assert c.output_name == "model"


class TestDeployConfig:
    def test_defaults(self):
        c = DeployConfig()
        assert c.target == "hailo-10h"
        assert c.port == 8000


class TestEvaluateConfig:
    def test_defaults(self):
        c = EvaluateConfig()
        assert c.metrics == ["rouge_l", "bertscore"]
        assert c.llm_judge is False
        assert c.prompt_format == "chatml"
        assert c.max_new_tokens == 512
        assert c.temperature == 0.1
        assert c.top_p == 0.9
        assert c.repetition_penalty == 1.1


class TestPipelineConfig:
    def test_defaults(self):
        c = PipelineConfig()
        assert isinstance(c.model, ModelConfig)
        assert isinstance(c.lora, LoraConfig)
        assert isinstance(c.training, TrainingConfig)
        assert isinstance(c.data, DataConfig)
        assert isinstance(c.export, ExportConfig)
        assert isinstance(c.compile, CompileConfig)
        assert isinstance(c.deploy, DeployConfig)
        assert isinstance(c.evaluate, EvaluateConfig)
        assert isinstance(c.output, OutputConfig)

    def test_from_yaml_default(self, default_yaml):
        c = PipelineConfig.from_yaml(default_yaml)
        assert c.model.name == "Qwen/Qwen2.5-3B-Instruct"
        assert c.deploy.target == "hailo-10h"
        assert c.compile.hw_arch == "hailo10h"
        assert c.compile.har_path is None

    def test_from_yaml_korean_legal(self, korean_legal_yaml):
        c = PipelineConfig.from_yaml(korean_legal_yaml)
        assert c.model.name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert c.data.train_path == Path("data/processed/train_dataset.jsonl")
        assert c.compile.har_path == Path("models/hailo/qwen2_1.5b_instruct.q.har")
        assert c.compile.adapter_name == "korean_legal_lora"
        assert c.compile.output_name == "qwen2_1.5b_korean_legal"
        assert c.output.lora_dir == Path("models/lora_adapters/trial2.1")

    def test_from_yaml_empty(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        c = PipelineConfig.from_yaml(path)
        assert c.model.name == "Qwen/Qwen2.5-3B-Instruct"
        assert c.compile.calibration_size == 64

    def test_from_yaml_partial(self, sample_yaml):
        c = PipelineConfig.from_yaml(sample_yaml)
        assert c.model.name == "test-model"
        assert c.model.max_seq_length == 512
        assert c.lora.r == 16  # default

    def test_validation_error(self):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"lora": {"r": "not_an_int"}})
