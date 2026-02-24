"""ONNX model export."""

import logging
from pathlib import Path

from optimum.exporters.onnx import main_export

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export a HuggingFace model to ONNX format."""

    def __init__(self, model_path: Path, output_dir: Path, dtype: str = "float16"):
        self.model_path = model_path
        self.output_dir = output_dir
        self.dtype = dtype

    def export(self):
        """Run the ONNX export."""
        logger.info("Exporting model: %s", self.model_path)
        logger.info("Output: %s", self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        main_export(
            model_name_or_path=str(self.model_path),
            output=str(self.output_dir),
            task="text-generation",
            trust_remote_code=True,
        )

        logger.info("Export complete.")
