"""Tests for hailo_llm_deploy.export — ONNX model export."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock heavy dependency before importing
sys.modules.setdefault("optimum", MagicMock())
sys.modules.setdefault("optimum.exporters", MagicMock())
sys.modules.setdefault("optimum.exporters.onnx", MagicMock())

from hailo_llm_deploy.export import ModelExporter  # noqa: E402


class TestModelExporterInit:
    def test_stores_params(self, tmp_path):
        exporter = ModelExporter(
            model_path=Path("/models/test"),
            output_dir=tmp_path / "output",
            dtype="float32",
        )
        assert exporter.model_path == Path("/models/test")
        assert exporter.output_dir == tmp_path / "output"
        assert exporter.dtype == "float32"

    def test_default_dtype(self, tmp_path):
        exporter = ModelExporter(model_path=Path("/m"), output_dir=tmp_path)
        assert exporter.dtype == "float16"


class TestModelExporterExport:
    def test_export_calls_main_export(self, tmp_path):
        exporter = ModelExporter(model_path=Path("/models/test"), output_dir=tmp_path / "out")

        with patch("hailo_llm_deploy.export.main_export") as mock_export:
            exporter.export()

            mock_export.assert_called_once_with(
                model_name_or_path="/models/test",
                output=str(tmp_path / "out"),
                task="text-generation",
                trust_remote_code=True,
            )

    def test_export_creates_output_dir(self, tmp_path):
        output = tmp_path / "nested" / "output"
        exporter = ModelExporter(model_path=Path("/models/test"), output_dir=output)

        with patch("hailo_llm_deploy.export.main_export"):
            exporter.export()

        assert output.exists()
