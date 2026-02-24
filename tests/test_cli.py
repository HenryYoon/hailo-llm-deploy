"""Tests for hailo_llm_deploy.cli — Typer CLI commands."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock heavy dependencies that get imported via module chains
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())
sys.modules.setdefault("datasets", MagicMock())
sys.modules.setdefault("trl", MagicMock())
sys.modules.setdefault("optimum", MagicMock())
sys.modules.setdefault("optimum.exporters", MagicMock())
sys.modules.setdefault("optimum.exporters.onnx", MagicMock())
sys.modules.setdefault("rouge_score", MagicMock())
sys.modules.setdefault("rouge_score.rouge_scorer", MagicMock())
sys.modules.setdefault("bert_score", MagicMock())
sys.modules.setdefault("tqdm", MagicMock(tqdm=lambda x, **kw: x))
sys.modules.setdefault("hailo_sdk_client", MagicMock())
sys.modules.setdefault("hailo_sdk_client.runner", MagicMock())
sys.modules.setdefault("hailo_sdk_client.runner.client_runner", MagicMock())
sys.modules.setdefault("numpy", MagicMock())

from typer.testing import CliRunner  # noqa: E402
from hailo_llm_deploy.cli import app  # noqa: E402

runner = CliRunner()


class TestHelp:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "hailo-llm-deploy" in result.output

    def test_finetune_help(self):
        result = runner.invoke(app, ["finetune", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_export_help(self):
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output

    def test_evaluate_help(self):
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_compile_help(self):
        result = runner.invoke(app, ["compile", "--help"])
        assert result.exit_code == 0
        assert "--har" in result.output
        assert "--alls" in result.output
        assert "--lora" in result.output


class TestCompileCommand:
    def test_blocked_without_force(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("compile:\n  har_path: /some/file.har\n  alls_path: /some/file.alls\n")
        result = runner.invoke(app, ["compile", "--config", str(config_path)])
        assert result.exit_code == 1
        assert "Blocked" in result.output

    def test_missing_har_with_force(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("compile:\n  har_path: null\n  alls_path: null\n")
        result = runner.invoke(app, ["compile", "--config", str(config_path), "--force"])
        assert result.exit_code == 1

    def test_missing_alls_with_force(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "compile:\n  har_path: /some/file.har\n  alls_path: null\n"
        )
        result = runner.invoke(app, [
            "compile", "--config", str(config_path),
            "--force", "--har", "/some/file.har",
        ])
        assert result.exit_code == 1
