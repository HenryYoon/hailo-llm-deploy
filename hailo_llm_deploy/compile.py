"""Hailo NPU compilation — attach LoRA to pre-compiled HAR and produce HEF."""

import json
import logging
from pathlib import Path

import numpy as np

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class HailoCompiler:
    """Compile a LoRA-adapted LLM for Hailo-10H NPU.

    The flow is:
        1. Load pre-optimized HAR from Hailo GenAI Model Zoo
        2. Attach LoRA adapter weights (safetensors)
        3. Load model script (ALLS)
        4. Optimize with calibration data
        5. Compile and save HAR + HEF

    Requires hailo_sdk_client (installed via Hailo AI SW Suite, not pip).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.runner = None
        self.hn_dict: dict | None = None

    def load_har(self) -> None:
        """Step 1: Load pre-optimized HAR file."""
        cc = self.config.compile
        har_path = Path(cc.har_path)
        if not har_path.exists():
            raise FileNotFoundError(
                f"HAR file not found: {har_path}. "
                "Download it from the Hailo GenAI Model Zoo."
            )

        ClientRunner = self._import_sdk()
        logger.info("Creating ClientRunner for hw_arch=%s", cc.hw_arch)
        self.runner = ClientRunner(hw_arch=cc.hw_arch)

        logger.info("Loading HAR: %s", har_path)
        self.runner.load_har(str(har_path))

    def attach_lora(self) -> None:
        """Step 2: Attach LoRA adapter weights to the HAR model."""
        cc = self.config.compile
        lora_path = self._resolve_lora_path()

        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

        logger.info("Attaching LoRA adapter: %s (name=%s)", lora_path, cc.adapter_name)
        self.runner.load_lora_weights(
            lora_weights_path=str(lora_path),
            lora_adapter_name=cc.adapter_name,
        )

    def load_model_script(self) -> None:
        """Step 3: Load ALLS model script."""
        cc = self.config.compile
        alls_path = Path(cc.alls_path)
        if not alls_path.exists():
            raise FileNotFoundError(
                f"ALLS file not found: {alls_path}. "
                "Download it from the Hailo GenAI Model Zoo."
            )

        logger.info("Loading model script: %s", alls_path)
        self.hn_dict = self.runner.load_model_script(str(alls_path))

    def optimize(self) -> None:
        """Step 4: Optimize with calibration data."""
        cc = self.config.compile
        input_dict = self._prepare_calibration_data()

        logger.info(
            "Optimizing with %d calibration samples (max_length=%d)...",
            cc.calibration_size,
            self._get_cache_size(),
        )
        self.runner.optimize(input_dict)

    def compile(self) -> Path:
        """Step 5: Compile to HEF and save compiled HAR."""
        cc = self.config.compile
        output_dir = Path(cc.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Compiling model...")
        hef = self.runner.compile()

        hef_path = output_dir / f"{cc.output_name}.hef"
        with open(hef_path, "wb") as f:
            f.write(hef)
        logger.info("HEF saved: %s", hef_path)

        har_path = output_dir / f"{cc.output_name}.compiled.har"
        self.runner.save_har(str(har_path), compilation_only=True)
        logger.info("Compiled HAR saved: %s", har_path)

        return hef_path

    def run(self) -> Path:
        """Full compilation pipeline: load -> lora -> script -> optimize -> compile."""
        self.load_har()
        self.attach_lora()
        self.load_model_script()
        self.optimize()
        return self.compile()

    # -- Private helpers --

    @staticmethod
    def _import_sdk():
        """Lazily import hailo_sdk_client with a clear error on failure."""
        try:
            from hailo_sdk_client.runner.client_runner import ClientRunner
            return ClientRunner
        except ImportError:
            raise ImportError(
                "hailo_sdk_client is not installed. "
                "Install it via Hailo AI SW Suite (DFC v5.2.0+). "
                "It is NOT available via pip. "
                "See: https://hailo.ai/developer-zone/software-downloads/"
            )

    def _resolve_lora_path(self) -> Path:
        """Resolve LoRA safetensors path from compile config or output config."""
        cc = self.config.compile
        if cc.lora_weights_path:
            return Path(cc.lora_weights_path)
        if self.config.output.lora_dir:
            return Path(self.config.output.lora_dir) / "adapter_model.safetensors"
        raise ValueError(
            "No LoRA weights path specified. Set compile.lora_weights_path "
            "or output.lora_dir in the config."
        )

    def _resolve_calibration_path(self) -> Path:
        """Resolve calibration data path from compile config or data config."""
        cc = self.config.compile
        if cc.calibration_data_path:
            path = Path(cc.calibration_data_path)
        elif self.config.data.train_path:
            path = Path(self.config.data.train_path)
        else:
            raise ValueError(
                "No calibration data path. Set compile.calibration_data_path "
                "or data.train_path in the config."
            )
        if not path.exists():
            raise FileNotFoundError(f"Calibration data not found: {path}")
        return path

    def _get_cache_size(self) -> int:
        """Get the model's cache_size from hn_dict (set after load_model_script)."""
        if self.hn_dict and "net_params" in self.hn_dict:
            return self.hn_dict["net_params"].get("cache_size", 2048)
        return self.config.model.max_seq_length

    def _prepare_calibration_data(self) -> dict:
        """Tokenize calibration samples into the input_dict format required by DFC.

        Returns:
            dict with keys "{adapter_name}/input_layer{1-6}" and numpy array values.
        """
        cc = self.config.compile
        adapter_name = cc.adapter_name
        max_length = self._get_cache_size()
        calibset_size = cc.calibration_size

        calib_path = self._resolve_calibration_path()
        samples = self._load_jsonl_samples(calib_path, calibset_size)

        tokenizer = self._load_tokenizer()
        input_ids, current_position = self._tokenize_samples(
            samples, tokenizer, max_length, calibset_size
        )

        input_dict = {f"{adapter_name}/input_layer1": input_ids}
        for i in range(2, 7):
            input_dict[f"{adapter_name}/input_layer{i}"] = current_position

        logger.info(
            "Calibration data prepared: input_ids shape=%s, position shape=%s",
            input_ids.shape,
            current_position.shape,
        )
        return input_dict

    @staticmethod
    def _load_jsonl_samples(path: Path, max_samples: int) -> list[dict]:
        """Load up to max_samples from a JSONL file."""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                samples.append(json.loads(line))
        logger.info("Loaded %d calibration samples from %s", len(samples), path)
        return samples

    def _load_tokenizer(self):
        """Load the tokenizer from the LoRA adapter directory or base model."""
        from transformers import AutoTokenizer

        lora_path = self._resolve_lora_path().parent
        tokenizer_config = lora_path / "tokenizer_config.json"
        if tokenizer_config.exists():
            logger.info("Loading tokenizer from: %s", lora_path)
            return AutoTokenizer.from_pretrained(str(lora_path))

        logger.info("Loading tokenizer from base model: %s", self.config.model.name)
        return AutoTokenizer.from_pretrained(self.config.model.name)

    def _tokenize_samples(
        self,
        samples: list[dict],
        tokenizer,
        max_length: int,
        calibset_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize samples into padded input_ids and current_position arrays.

        Returns:
            Tuple of (input_ids, current_position) as numpy int32 arrays.
        """
        prompt_format = self.config.training.prompt_format

        all_input_ids = []
        all_positions = []

        for sample in samples:
            text = self._format_calibration_text(sample, prompt_format)
            encoded = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            ids = encoded["input_ids"][0]
            non_pad_mask = ids != tokenizer.pad_token_id
            last_pos = int(np.where(non_pad_mask)[0][-1]) if non_pad_mask.any() else 0
            all_input_ids.append(ids)
            all_positions.append(last_pos)

        input_ids = np.stack(all_input_ids).astype(np.int32)
        current_position = np.array(all_positions, dtype=np.int32).reshape(-1, 1)
        return input_ids, current_position

    @staticmethod
    def _format_calibration_text(sample: dict, prompt_format: str) -> str:
        """Format a single sample into a full prompt+response text for calibration."""
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")

        if prompt_format == "chatml":
            return (
                f"<|im_start|>system\n{instruction}<|im_end|>\n"
                f"<|im_start|>user\n{input_text}<|im_end|>\n"
                f"<|im_start|>assistant\n{output_text}<|im_end|>"
            )
        else:
            return (
                "Below is an instruction that describes a task, paired with an "
                "input that provides further context. Write a response that "
                "appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output_text}"
            )
