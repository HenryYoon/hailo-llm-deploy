"""Tests for hailo_llm_deploy.compile — Hailo NPU compilation logic."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock hailo_sdk_client (not pip-installable)
sys.modules.setdefault("hailo_sdk_client", MagicMock())
sys.modules.setdefault("hailo_sdk_client.runner", MagicMock())
sys.modules.setdefault("hailo_sdk_client.runner.client_runner", MagicMock())

from hailo_llm_deploy.compile import HailoCompiler  # noqa: E402
from hailo_llm_deploy.config import PipelineConfig  # noqa: E402


@pytest.fixture
def compiler():
    return HailoCompiler(PipelineConfig())


@pytest.fixture
def configured_compiler(tmp_path):
    """Compiler with har/alls/lora paths configured."""
    cfg = PipelineConfig()
    cfg.compile.har_path = tmp_path / "model.har"
    cfg.compile.alls_path = tmp_path / "model.alls"
    cfg.compile.lora_weights_path = tmp_path / "adapter_model.safetensors"
    cfg.compile.output_dir = tmp_path / "output"
    cfg.compile.output_name = "test_model"
    cfg.compile.adapter_name = "test_adapter"
    return HailoCompiler(cfg)


class TestHailoCompilerInit:
    def test_init(self, compiler):
        assert compiler.runner is None
        assert compiler.hn_dict is None
        assert isinstance(compiler.config, PipelineConfig)


class TestResolveLoraPath:
    def test_explicit_path(self, compiler, tmp_path):
        compiler.config.compile.lora_weights_path = tmp_path / "lora.safetensors"
        result = compiler._resolve_lora_path()
        assert result == tmp_path / "lora.safetensors"

    def test_fallback_to_output_lora_dir(self, compiler, tmp_path):
        compiler.config.compile.lora_weights_path = None
        compiler.config.output.lora_dir = tmp_path / "adapters"
        result = compiler._resolve_lora_path()
        assert result == tmp_path / "adapters" / "adapter_model.safetensors"

    def test_error_when_both_none(self, compiler):
        compiler.config.compile.lora_weights_path = None
        compiler.config.output.lora_dir = None
        with pytest.raises(ValueError, match="No LoRA weights path"):
            compiler._resolve_lora_path()


class TestResolveCalibrationPath:
    def test_explicit_path(self, compiler, tmp_path):
        calib_file = tmp_path / "calib.jsonl"
        calib_file.write_text("{}\n", encoding="utf-8")
        compiler.config.compile.calibration_data_path = calib_file
        result = compiler._resolve_calibration_path()
        assert result == calib_file

    def test_fallback_to_train_path(self, compiler, tmp_path):
        train_file = tmp_path / "train.jsonl"
        train_file.write_text("{}\n", encoding="utf-8")
        compiler.config.compile.calibration_data_path = None
        compiler.config.data.train_path = train_file
        result = compiler._resolve_calibration_path()
        assert result == train_file

    def test_error_when_both_none(self, compiler):
        compiler.config.compile.calibration_data_path = None
        compiler.config.data.train_path = None
        with pytest.raises(ValueError, match="No calibration data path"):
            compiler._resolve_calibration_path()

    def test_error_when_file_missing(self, compiler, tmp_path):
        compiler.config.compile.calibration_data_path = tmp_path / "missing.jsonl"
        with pytest.raises(FileNotFoundError):
            compiler._resolve_calibration_path()


class TestGetCacheSize:
    def test_from_hn_dict(self, compiler):
        compiler.hn_dict = {"net_params": {"cache_size": 1024}}
        assert compiler._get_cache_size() == 1024

    def test_fallback_to_config(self, compiler):
        compiler.hn_dict = None
        compiler.config.model.max_seq_length = 4096
        assert compiler._get_cache_size() == 4096

    def test_fallback_when_no_cache_size(self, compiler):
        compiler.hn_dict = {"net_params": {}}
        assert compiler._get_cache_size() == 2048  # default


class TestFormatCalibrationText:
    def test_chatml(self):
        sample = {"instruction": "Sys", "input": "Q", "output": "A"}
        result = HailoCompiler._format_calibration_text(sample, "chatml")
        assert "<|im_start|>system\nSys<|im_end|>" in result
        assert "<|im_start|>user\nQ<|im_end|>" in result
        assert "<|im_start|>assistant\nA<|im_end|>" in result

    def test_alpaca(self):
        sample = {"instruction": "Sys", "input": "Q", "output": "A"}
        result = HailoCompiler._format_calibration_text(sample, "alpaca")
        assert "### Instruction:\nSys" in result
        assert "### Input:\nQ" in result
        assert "### Response:\nA" in result

    def test_missing_keys(self):
        sample = {}
        result = HailoCompiler._format_calibration_text(sample, "chatml")
        assert "<|im_start|>system\n<|im_end|>" in result


class TestLoadJsonlSamples:
    def test_loads_up_to_max(self, sample_jsonl):
        samples = HailoCompiler._load_jsonl_samples(sample_jsonl, max_samples=2)
        assert len(samples) == 2
        assert samples[0]["input"] == "What is AI?"

    def test_loads_all_when_fewer(self, sample_jsonl):
        samples = HailoCompiler._load_jsonl_samples(sample_jsonl, max_samples=100)
        assert len(samples) == 3


class TestLoadHar:
    def test_file_not_found(self, configured_compiler):
        with pytest.raises(FileNotFoundError, match="HAR file not found"):
            configured_compiler.load_har()

    def test_success(self, configured_compiler, tmp_path):
        har_file = tmp_path / "model.har"
        har_file.write_bytes(b"fake_har")
        configured_compiler.config.compile.har_path = har_file

        mock_runner = MagicMock()
        MockClientRunner = MagicMock(return_value=mock_runner)

        with patch.object(HailoCompiler, "_import_sdk", return_value=MockClientRunner):
            configured_compiler.load_har()

        MockClientRunner.assert_called_once_with(hw_arch="hailo10h")
        mock_runner.load_har.assert_called_once_with(str(har_file))
        assert configured_compiler.runner is mock_runner


class TestAttachLora:
    def test_attach(self, configured_compiler, tmp_path):
        lora_file = tmp_path / "adapter_model.safetensors"
        lora_file.write_bytes(b"fake_lora")
        configured_compiler.config.compile.lora_weights_path = lora_file
        configured_compiler.runner = MagicMock()

        configured_compiler.attach_lora()

        configured_compiler.runner.load_lora_weights.assert_called_once_with(
            lora_weights_path=str(lora_file),
            lora_adapter_name="test_adapter",
        )


class TestCompile:
    def test_saves_hef_and_har(self, configured_compiler, tmp_path):
        configured_compiler.config.compile.output_dir = tmp_path / "compiled"
        configured_compiler.runner = MagicMock()
        configured_compiler.runner.compile.return_value = b"fake_hef_binary"

        hef_path = configured_compiler.compile()

        assert hef_path == tmp_path / "compiled" / "test_model.hef"
        assert hef_path.read_bytes() == b"fake_hef_binary"
        configured_compiler.runner.save_har.assert_called_once_with(
            str(tmp_path / "compiled" / "test_model.compiled.har"),
            compilation_only=True,
        )


class TestRun:
    def test_call_order(self, compiler):
        with patch.object(compiler, "load_har") as m1, \
             patch.object(compiler, "attach_lora") as m2, \
             patch.object(compiler, "load_model_script") as m3, \
             patch.object(compiler, "optimize") as m4, \
             patch.object(compiler, "compile", return_value=Path("out.hef")) as m5:

            result = compiler.run()

            m1.assert_called_once()
            m2.assert_called_once()
            m3.assert_called_once()
            m4.assert_called_once()
            m5.assert_called_once()
            assert result == Path("out.hef")
