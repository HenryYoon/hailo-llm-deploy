# %%
# notebooks/01_stage1_legal_expert.ipynb

"""
=================================================
Stage 1: Legal Expert Fine-tuning
- 목표: 법률 전문 지식 학습
- 데이터: 16.5K 샘플
- 예상 시간: ~18시간 (RTX 3060 12GB)
=================================================
"""

# ========================================
# 1. 라이브러리 임포트
# ========================================
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ========================================
# 2. 설정값
# ========================================
# 모델 설정
max_seq_length = 1024
dtype = None  # Auto-detect (BF16 for Ampere+)
load_in_4bit = True
seed = 3407

# LoRA 설정
lora_r = 16
lora_alpha = 16
lora_dropout = 0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# 학습 설정
output_dir = "../../models/checkpoints/stage1"
num_train_epochs = 1
per_device_train_batch_size = 2
gradient_accumulation_steps = 1
learning_rate = 2e-4
logging_steps = 10

# 데이터 경로
train_data_path = "../../data/processed/trial1/train_dataset_trial1.json"
val_data_path = "../../data/processed/trial1/val_dataset_trial1.json"

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen2.5-3B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        device_map="balanced",
    )

model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        target_modules = target_modules,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        bias = "none",
        use_gradient_checkpointing = "unsloth",  # 30% 더 빠른 체크포인팅
        random_state = seed,
        use_rslora = False,  # Rank-Stabilized LoRA
        loftq_config = None,
    )

if __name__ == "__main__":

    # ========================================
    # 3. 모델 로드
    # ========================================
    print("📥 모델 로딩 중...")

    
    print(f"✅ 모델 로드 완료: Qwen2.5-7B-Instruct")
    print(f"📊 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # ========================================
    # 4. LoRA 설정
    # ========================================
    print("🔧 LoRA 설정 중...")

    

    print(f"✅ LoRA 설정 완료 (r={lora_r}, alpha={lora_alpha})")

    # 학습 가능한 파라미터 확인
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 학습 가능 파라미터: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # ========================================
    # 5. 데이터셋 준비
    # ========================================
    print("📚 데이터셋 로딩 중...")

    dataset = load_dataset("json", data_files={
        "train": train_data_path,
        "validation": val_data_path
    })

    print(f"✅ 데이터셋 로드 완료")
    print(f"  - Train: {len(dataset['train'])} 샘플")
    print(f"  - Validation: {len(dataset['validation'])} 샘플")

    # 샘플 확인
    print("\n📝 데이터 샘플 (첫 번째):")
    print(dataset['train'][0])

    # ========================================
    # 6. 프롬프트 포맷팅
    # ========================================
    # ChatML 포맷
    

    def formatting_prompts_func(examples):
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""
        """데이터를 학습 포맷으로 변환"""
        instructions = examples["instruction"]
        inputs = examples["input"] if "input" in examples else [""] * len(instructions)
        outputs = examples["output"]
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            # input이 없으면 단순화
            if input_text.strip() == "":
                text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Response:
                {output}"""
            else:
                text = alpaca_prompt.format(instruction, input_text, output)
            
            text += tokenizer.eos_token
            texts.append(text)
        
        return {"text": texts}

    # 데이터셋 변환
    dataset = dataset.map(formatting_prompts_func, batched=True)

    print("✅ 프롬프트 포맷팅 완료")


    # ========================================
    # 7. Trainer 설정
    # ========================================
    print("🏋️ Trainer 설정 중...")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["validation"],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        packing = False,  # Stage 1에서는 False
        args = SFTConfig(
            per_device_train_batch_size = per_device_train_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            warmup_steps = 10,
            num_train_epochs = num_train_epochs, # Set this for 1 full training run.
            learning_rate = learning_rate,
            logging_steps = logging_steps,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = seed,
            output_dir = output_dir,
            report_to = "tensorboard", # Use TrackIO/WandB etc
            eval_strategy='steps',
            eval_steps = 1000,
            save_strategy ='best',
            dataloader_num_workers = 8,   # 중요 (윈도우에서 워커=spawn)
            load_best_model_at_end=True
        ),
        )


    print("✅ Trainer 준비 완료")

    # ========================================
    # 8. 학습 시작
    # ========================================
    print("\n" + "="*50)
    print("🚀 Stage 1 학습 시작!")
    print("="*50)
    print(f"📊 총 학습 스텝: {len(dataset['train']) // (per_device_train_batch_size * gradient_accumulation_steps) * num_train_epochs}")
    print(f"⏱️  예상 소요 시간: ~18시간 (RTX 3060 12GB)")
    print("="*50 + "\n")

    # 학습 실행
    trainer_stats = trainer.train()

    print("\n" + "="*50)
    print("✅ Stage 1 학습 완료!")
    print("="*50)
    print(f"📊 최종 손실(Loss): {trainer_stats.training_loss:.4f}")
    print(f"⏱️  실제 소요 시간: {trainer_stats.metrics['train_runtime'] / 3600:.2f}시간")
    print("="*50 + "\n")

    # ========================================
    # 9. 모델 저장
    # ========================================
    print("💾 모델 저장 중...")

    # LoRA 어댑터만 저장
    model.save_pretrained("../../models/lora_adapters/stage1")
    tokenizer.save_pretrained("../../models/lora_adapters/stage1")
    print("✅ LoRA 어댑터 저장 완료: ../../models/lora_adapters/stage1")

    # 16-bit 병합 모델 저장
    model.save_pretrained_merged(
        "../../models/merged/stage1_16bit",
        tokenizer,
        save_method = "merged_16bit",
    )
    print("✅ 병합 모델 저장 완료: ../../models/merged/stage1_16bit")

    # 4-bit GGUF 저장 (선택)
    # model.save_pretrained_gguf(
    #     "./models/stage1_gguf",
    #     tokenizer,
    #     quantization_method = "q4_k_m"
    # )
    # print("✅ GGUF 모델 저장 완료: ./models/stage1_gguf")

    print("\n🎉 Stage 1 완료!")




# %%
