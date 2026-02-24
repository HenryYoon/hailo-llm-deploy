"""
Trial 2: RAFT Legal QA Fine-tuning with Unsloth
================================================
- Base model: Qwen/Qwen2.5-3B-Instruct (4-bit)
- Method: LoRA via Unsloth
- Data: RAFT format (1000 train / 100 val / 100 test)
- Format: ChatML (Qwen native)
- Target GPU: RTX 3060 12GB

Usage:
    python src/train_trial2.py
"""

import torch
from pathlib import Path
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ========================================
# Paths
# ========================================
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / "data" / "processed" / "train_dataset.jsonl"
VAL_DATA = PROJECT_ROOT / "data" / "processed" / "val_dataset.jsonl"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "trial2.1"
LORA_DIR = PROJECT_ROOT / "models" / "lora_adapters" / "trial2.1"
MERGED_DIR = PROJECT_ROOT / "models" / "merged" / "trial2.1"

# ========================================
# Model config
# ========================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 2048
DTYPE = None  # auto-detect (bf16 on Ampere+)
LOAD_IN_4BIT = True
SEED = 42

# ========================================
# LoRA config
# ========================================
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ========================================
# Training config
# ========================================
NUM_EPOCHS = 10
BATCH_SIZE = 2
GRAD_ACCUM = 4          # effective batch size = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
EVAL_STEPS = 50
SAVE_STEPS = 50


def format_chatml(examples):
    """Format dataset rows into ChatML for Qwen.

    Maps RAFT fields to ChatML roles:
        system  = instruction (시스템 지시 + [D1]~[D5] 문서 컨텍스트)
        user    = input (질문)
        assistant = output (답변 + 참고문헌)
    """
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


def main():
    print("=" * 50)
    print("Trial 2: RAFT Legal QA Fine-tuning")
    print("=" * 50)

    # ------ Model ------
    print("\n[1/6] Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    print(f"  Model: {MODEL_NAME}")
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # ------ LoRA ------
    print("\n[2/6] Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ------ Dataset ------
    print("\n[3/6] Loading dataset...")
    dataset = load_dataset("json", data_files={
        "train": str(TRAIN_DATA),
        "validation": str(VAL_DATA),
    })
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Val:   {len(dataset['validation'])} samples")

    dataset = dataset.map(format_chatml, batched=True)

    # Token length check
    sample = tokenizer(dataset["train"][0]["text"], return_length=True)
    print(f"  Sample token length: {sample['length'][0]}")

    # ------ Trainer ------
    print("\n[4/6] Setting up trainer...")

    steps_per_epoch = len(dataset["train"]) // (BATCH_SIZE * GRAD_ACCUM)
    total_steps = steps_per_epoch * NUM_EPOCHS
    print(f"  Steps/epoch: {steps_per_epoch}, Total: {total_steps}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        args=SFTConfig(
            output_dir=str(CHECKPOINT_DIR),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=WEIGHT_DECAY,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=LOGGING_STEPS,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            seed=SEED,
            report_to="tensorboard",
            dataloader_num_workers=0,
        ),
    )

    # ------ Train ------
    print("\n" + "=" * 50)
    print("[5/6] Training...")
    print("=" * 50)

    stats = trainer.train()

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"  Final loss: {stats.training_loss:.4f}")
    print(f"  Runtime: {stats.metrics['train_runtime'] / 3600:.2f}h")
    print("=" * 50)

    # ------ Save ------
    print("\n[6/6] Saving model...")

    LORA_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(LORA_DIR))
    tokenizer.save_pretrained(str(LORA_DIR))
    print(f"  LoRA adapter: {LORA_DIR}")

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(MERGED_DIR), tokenizer, save_method="merged_16bit"
    )
    print(f"  Merged 16-bit: {MERGED_DIR}")

    print("\nDone!")


if __name__ == "__main__":
    main()
