"""LoRA fine-tuning via Unsloth. Generic pipeline — no domain-specific logic."""

import logging
from pathlib import Path

import torch
from unsloth import FastLanguageModel  # must be imported before trl
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class FineTuner:
    """Generic LoRA fine-tuner using Unsloth + SFTTrainer.

    Splits the monolithic training flow into composable steps:
    load_model -> apply_lora -> load_data -> train -> save.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load base model and tokenizer via Unsloth."""
        mc = self.config.model
        logger.info("Loading model: %s", mc.name)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=mc.name,
            max_seq_length=mc.max_seq_length,
            dtype=None,
            load_in_4bit=mc.load_in_4bit,
        )
        logger.info("GPU memory: %.2f GB", torch.cuda.memory_allocated() / 1024**3)

    def apply_lora(self):
        """Apply LoRA adapter to the loaded model."""
        lc = self.config.lora
        tc = self.config.training
        logger.info("Applying LoRA (r=%d, alpha=%d)", lc.r, lc.alpha)
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lc.r,
            target_modules=lc.target_modules,
            lora_alpha=lc.alpha,
            lora_dropout=lc.dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=tc.seed,
        )
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info("Trainable: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}",
                     100 * trainable / total)

    def load_data(self):
        """Load and format training/validation datasets."""
        dc = self.config.data
        tc = self.config.training

        data_files = {}
        if dc.train_path:
            data_files["train"] = str(dc.train_path)
        if dc.val_path:
            data_files["validation"] = str(dc.val_path)

        dataset = load_dataset("json", data_files=data_files)
        logger.info("Train: %d samples", len(dataset.get("train", [])))
        if "validation" in dataset:
            logger.info("Val: %d samples", len(dataset["validation"]))

        format_fn = self._get_format_fn(tc.prompt_format)
        dataset = dataset.map(format_fn, batched=True)
        return dataset

    def train(self, dataset):
        """Run SFTTrainer."""
        tc = self.config.training
        mc = self.config.model
        oc = self.config.output

        checkpoint_dir = str(oc.checkpoint_dir) if oc.checkpoint_dir else "./checkpoints"

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            args=SFTConfig(
                dataset_text_field="text",
                packing=False,
                eos_token=self.tokenizer.eos_token,
                output_dir=checkpoint_dir,
                num_train_epochs=tc.epochs,
                per_device_train_batch_size=tc.batch_size,
                gradient_accumulation_steps=tc.gradient_accumulation,
                learning_rate=tc.learning_rate,
                warmup_ratio=tc.warmup_ratio,
                lr_scheduler_type=tc.scheduler,
                optim="adamw_8bit",
                weight_decay=tc.weight_decay,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10,
                eval_strategy="steps" if dataset.get("validation") else "no",
                eval_steps=tc.eval_steps,
                save_strategy="steps",
                save_steps=tc.save_steps,
                save_total_limit=3,
                load_best_model_at_end=bool(dataset.get("validation")),
                metric_for_best_model="eval_loss",
                seed=tc.seed,
                report_to="tensorboard",
                dataloader_num_workers=0,
            ),
        )

        logger.info("Starting training...")
        stats = trainer.train()
        logger.info("Training complete. Final loss: %.4f", stats.training_loss)
        return stats

    def save(self):
        """Save LoRA adapter and merged model."""
        oc = self.config.output

        if oc.lora_dir:
            lora_dir = Path(oc.lora_dir)
            lora_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(lora_dir))
            self.tokenizer.save_pretrained(str(lora_dir))
            logger.info("LoRA adapter saved: %s", lora_dir)

        if oc.merged_dir:
            merged_dir = Path(oc.merged_dir)
            merged_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained_merged(
                str(merged_dir), self.tokenizer, save_method="merged_16bit"
            )
            logger.info("Merged model saved: %s", merged_dir)

    def run(self):
        """Full pipeline: load -> lora -> data -> train -> save."""
        self.load_model()
        self.apply_lora()
        dataset = self.load_data()
        self.train(dataset)
        self.save()

    @staticmethod
    def _get_format_fn(prompt_format: str):
        """Return a dataset mapping function for the given prompt format."""
        if prompt_format == "chatml":
            return FineTuner._format_chatml
        elif prompt_format == "alpaca":
            return FineTuner._format_alpaca
        else:
            raise ValueError(f"Unknown prompt format: {prompt_format}")

    @staticmethod
    def _format_chatml(examples: dict) -> dict:
        """Format dataset rows into ChatML."""
        texts = []
        for instruction, input_text, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            text = (
                f"<|im_start|>system\n{instruction}<|im_end|>\n"
                f"<|im_start|>user\n{input_text}<|im_end|>\n"
                f"<|im_start|>assistant\n{output}<|im_end|>"
            )
            texts.append(text)
        return {"text": texts}

    @staticmethod
    def _format_alpaca(examples: dict) -> dict:
        """Format dataset rows into Alpaca style."""
        texts = []
        for instruction, input_text, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            text = (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. Write a response that appropriately "
                "completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
            texts.append(text)
        return {"text": texts}
