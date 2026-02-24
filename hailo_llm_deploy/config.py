"""Pydantic configuration models for the pipeline. Loaded from YAML files."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-3B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True


class LoraConfig(BaseModel):
    r: int = 16
    alpha: int = 16
    dropout: float = 0
    target_modules: list[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


class TrainingConfig(BaseModel):
    epochs: int = 10
    batch_size: int = 2
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    eval_steps: int = 50
    save_steps: int = 50
    seed: int = 42
    prompt_format: str = "chatml"


class DataConfig(BaseModel):
    train_path: Path | None = None
    val_path: Path | None = None
    test_path: Path | None = None
    instruction: str | None = None


class ExportConfig(BaseModel):
    format: str = "onnx"
    dtype: str = "float16"
    output_dir: Path = Path("./output")


class CompileConfig(BaseModel):
    """Hailo NPU compilation configuration."""
    har_path: Path | None = None
    alls_path: Path | None = None
    hw_arch: str = "hailo10h"
    adapter_name: str = "lora_adapter"
    lora_weights_path: Path | None = None
    calibration_data_path: Path | None = None
    calibration_size: int = 64
    output_dir: Path = Path("./models/compiled")
    output_name: str = "model"


class DeployConfig(BaseModel):
    target: str = "hailo-10h"
    port: int = 8000


class EvaluateConfig(BaseModel):
    metrics: list[str] = ["rouge_l", "bertscore"]
    llm_judge: bool = False
    prompt_format: str = "chatml"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.1


class OutputConfig(BaseModel):
    checkpoint_dir: Path | None = None
    lora_dir: Path | None = None
    merged_dir: Path | None = None
    results_dir: Path | None = None


class PipelineConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    lora: LoraConfig = LoraConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    export: ExportConfig = ExportConfig()
    compile: CompileConfig = CompileConfig()
    deploy: DeployConfig = DeployConfig()
    evaluate: EvaluateConfig = EvaluateConfig()
    output: OutputConfig = OutputConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw or {})
