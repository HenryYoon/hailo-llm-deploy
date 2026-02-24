"""Tests for hailo_llm_deploy.finetune — prompt formatters and orchestration."""

import sys
from unittest.mock import MagicMock, patch, call

import pytest

# Mock heavy dependencies before importing the module
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())
sys.modules.setdefault("datasets", MagicMock())
sys.modules.setdefault("trl", MagicMock())

from hailo_llm_deploy.finetune import FineTuner  # noqa: E402
from hailo_llm_deploy.config import PipelineConfig  # noqa: E402


@pytest.fixture
def finetuner():
    return FineTuner(PipelineConfig())


class TestFormatChatml:
    def test_single_sample(self):
        examples = {
            "instruction": ["Be helpful."],
            "input": ["Hello"],
            "output": ["Hi there!"],
        }
        result = FineTuner._format_chatml(examples)
        assert "text" in result
        assert len(result["text"]) == 1
        text = result["text"][0]
        assert "<|im_start|>system\nBe helpful.<|im_end|>" in text
        assert "<|im_start|>user\nHello<|im_end|>" in text
        assert "<|im_start|>assistant\nHi there!<|im_end|>" in text

    def test_multiple_samples(self):
        examples = {
            "instruction": ["A", "B"],
            "input": ["X", "Y"],
            "output": ["1", "2"],
        }
        result = FineTuner._format_chatml(examples)
        assert len(result["text"]) == 2


class TestFormatAlpaca:
    def test_single_sample(self):
        examples = {
            "instruction": ["Do task"],
            "input": ["Some input"],
            "output": ["Result"],
        }
        result = FineTuner._format_alpaca(examples)
        text = result["text"][0]
        assert "### Instruction:\nDo task" in text
        assert "### Input:\nSome input" in text
        assert "### Response:\nResult" in text


class TestGetFormatFn:
    def test_chatml(self):
        fn = FineTuner._get_format_fn("chatml")
        assert fn == FineTuner._format_chatml

    def test_alpaca(self):
        fn = FineTuner._get_format_fn("alpaca")
        assert fn == FineTuner._format_alpaca

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt format"):
            FineTuner._get_format_fn("xyz")


class TestFineTunerInit:
    def test_init(self, finetuner):
        assert finetuner.model is None
        assert finetuner.tokenizer is None
        assert isinstance(finetuner.config, PipelineConfig)


class TestFineTunerSave:
    def test_save_lora(self, finetuner, tmp_path):
        finetuner.config.output.lora_dir = tmp_path / "lora"
        finetuner.config.output.merged_dir = None
        finetuner.model = MagicMock()
        finetuner.tokenizer = MagicMock()

        finetuner.save()

        finetuner.model.save_pretrained.assert_called_once_with(str(tmp_path / "lora"))
        finetuner.tokenizer.save_pretrained.assert_called_once_with(str(tmp_path / "lora"))

    def test_save_skips_when_no_dirs(self, finetuner):
        finetuner.config.output.lora_dir = None
        finetuner.config.output.merged_dir = None
        finetuner.model = MagicMock()
        finetuner.tokenizer = MagicMock()

        finetuner.save()

        finetuner.model.save_pretrained.assert_not_called()


class TestFineTunerRun:
    def test_run_call_order(self, finetuner):
        with patch.object(finetuner, "load_model") as m_load, \
             patch.object(finetuner, "apply_lora") as m_lora, \
             patch.object(finetuner, "load_data", return_value="dataset") as m_data, \
             patch.object(finetuner, "train") as m_train, \
             patch.object(finetuner, "save") as m_save:

            finetuner.run()

            m_load.assert_called_once()
            m_lora.assert_called_once()
            m_data.assert_called_once()
            m_train.assert_called_once_with("dataset")
            m_save.assert_called_once()
