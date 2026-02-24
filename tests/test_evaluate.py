"""Tests for hailo_llm_deploy.evaluate — evaluation logic."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock heavy dependencies before importing
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())
sys.modules.setdefault("tqdm", MagicMock(tqdm=lambda x, **kw: x))
_mock_rouge = MagicMock()
sys.modules.setdefault("rouge_score", _mock_rouge)
sys.modules.setdefault("rouge_score.rouge_scorer", _mock_rouge.rouge_scorer)
_mock_bert = MagicMock()
sys.modules.setdefault("bert_score", _mock_bert)

from hailo_llm_deploy.evaluate import Evaluator  # noqa: E402
from hailo_llm_deploy.config import PipelineConfig  # noqa: E402


@pytest.fixture
def evaluator():
    return Evaluator(PipelineConfig())


class TestEvaluatorInit:
    def test_init(self, evaluator):
        assert evaluator.model is None
        assert evaluator.tokenizer is None
        assert evaluator.llm_client is None
        assert evaluator.test_data == []


class TestBuildPrompt:
    def test_chatml(self, evaluator):
        evaluator.config.evaluate.prompt_format = "chatml"
        sample = {"instruction": "Be helpful.", "input": "Hello"}
        result = evaluator.build_prompt(sample)
        assert "<|im_start|>system\nBe helpful.<|im_end|>" in result
        assert "<|im_start|>user\nHello<|im_end|>" in result
        assert result.endswith("<|im_start|>assistant\n")

    def test_alpaca(self, evaluator):
        evaluator.config.evaluate.prompt_format = "alpaca"
        sample = {"instruction": "Do task", "input": "Data"}
        result = evaluator.build_prompt(sample)
        assert "### Instruction:\nDo task" in result
        assert "### Input:\nData" in result
        assert result.endswith("### Response:\n")


class TestDiscoverTrials:
    def test_finds_trials(self, evaluator, mock_trial_dirs):
        trials = evaluator.discover_trials(mock_trial_dirs)
        assert trials == ["trial1", "trial2"]

    def test_empty_dir(self, evaluator, tmp_path):
        trials = evaluator.discover_trials(tmp_path)
        assert trials == []

    def test_nonexistent_dir(self, evaluator, tmp_path):
        trials = evaluator.discover_trials(tmp_path / "nope")
        assert trials == []


class TestLoadTestData:
    def test_loads_jsonl(self, evaluator, sample_jsonl):
        evaluator.config.data.test_path = sample_jsonl
        evaluator.load_test_data()
        assert len(evaluator.test_data) == 3
        assert evaluator.test_data[0]["input"] == "What is AI?"

    def test_missing_file(self, evaluator):
        evaluator.config.data.test_path = Path("/nonexistent/file.jsonl")
        with pytest.raises(FileNotFoundError):
            evaluator.load_test_data()

    def test_none_path(self, evaluator):
        evaluator.config.data.test_path = None
        with pytest.raises(FileNotFoundError):
            evaluator.load_test_data()


class TestPrintResults:
    def test_single_result(self, capsys):
        results = [{
            "trial": "trial1",
            "rouge_l": {"rouge_l_precision": 0.5, "rouge_l_recall": 0.4, "rouge_l_f1": 0.45},
            "bertscore": {"bertscore_precision": 0.7, "bertscore_recall": 0.6, "bertscore_f1": 0.65},
        }]
        Evaluator.print_results(results)
        output = capsys.readouterr().out
        assert "trial1" in output
        assert "0.4500" in output

    def test_multiple_results(self, capsys):
        results = [
            {
                "trial": "trial1",
                "rouge_l": {"rouge_l_precision": 0.3, "rouge_l_recall": 0.2, "rouge_l_f1": 0.25},
                "bertscore": {"bertscore_precision": 0.6, "bertscore_recall": 0.5, "bertscore_f1": 0.55},
            },
            {
                "trial": "trial2",
                "rouge_l": {"rouge_l_precision": 0.5, "rouge_l_recall": 0.4, "rouge_l_f1": 0.45},
                "bertscore": {"bertscore_precision": 0.7, "bertscore_recall": 0.6, "bertscore_f1": 0.65},
            },
        ]
        Evaluator.print_results(results)
        output = capsys.readouterr().out
        assert "*" in output  # best marker
        assert "trial1" in output
        assert "trial2" in output

    def test_with_llm_judge(self, capsys):
        results = [{
            "trial": "trial1",
            "rouge_l": {"rouge_l_precision": 0.5, "rouge_l_recall": 0.4, "rouge_l_f1": 0.45},
            "bertscore": {"bertscore_precision": 0.7, "bertscore_recall": 0.6, "bertscore_f1": 0.65},
            "llm_judge": {"correctness": 3.5, "completeness": 3.0, "faithfulness": 4.0},
        }]
        Evaluator.print_results(results)
        output = capsys.readouterr().out
        assert "Judge" in output
