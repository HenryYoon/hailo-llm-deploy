"""
Legal QA Model Evaluation
==========================
Auto-detect and evaluate all fine-tuned models under models/merged/trial*.

Metrics:
    1. ROUGE-L  — n-gram overlap (핵심 법률 용어 포함 여부)
    2. BERTScore — semantic similarity (의미적 유사도)
    3. LLM-as-a-Judge (optional) — GPT-based quality scoring

Usage:
    # Evaluate all detected trials
    python src/evaluate_trial2.py

    # Evaluate specific trials only
    python src/evaluate_trial2.py --trial trial2 trial3

    # With LLM-as-a-Judge
    python src/evaluate_trial2.py --llm-judge

    # Override prompt format (default: chatml)
    python src/evaluate_trial2.py --trial trial1 --prompt-format alpaca
"""

import json
import re
import argparse
from pathlib import Path

import torch
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from unsloth import FastLanguageModel

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# ========================================
# Paths
# ========================================
PROJECT_ROOT = Path(__file__).parent.parent
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test_dataset.jsonl"
MERGED_DIR = PROJECT_ROOT / "models" / "merged"
RESULTS_DIR = PROJECT_ROOT / "results"

# ========================================
# Generation config
# ========================================
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
TOP_P = 0.9
REPETITION_PENALTY = 1.1


def discover_trials() -> list[str]:
    """Auto-detect trial directories under models/merged/trial*."""
    if not MERGED_DIR.exists():
        return []
    trials = sorted(
        [d.name for d in MERGED_DIR.iterdir() if d.is_dir() and d.name.startswith("trial")],
        key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0,
    )
    return trials


def load_model(model_path: Path):
    """Load a fine-tuned model from the given path."""
    print(f"  Loading model from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def build_prompt(sample: dict, prompt_format: str) -> str:
    """Build prompt from a test sample based on the format."""
    if prompt_format == "chatml":
        return (
            f"<|im_start|>system\n{sample['instruction']}<|im_end|>\n"
            f"<|im_start|>user\n{sample['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:  # alpaca
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Input:\n{sample['input']}\n\n"
            "### Response:\n"
        )


def generate_answer(model, tokenizer, prompt: str) -> str:
    """Generate a single answer from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
    precision = np.mean([s["rougeL"].precision for s in scores])
    recall = np.mean([s["rougeL"].recall for s in scores])
    f1 = np.mean([s["rougeL"].fmeasure for s in scores])
    return {"rouge_l_precision": precision, "rouge_l_recall": recall, "rouge_l_f1": f1}


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    """Compute BERTScore using multilingual model."""
    P, R, F1 = bert_score(
        predictions,
        references,
        lang="ko",
        verbose=True,
    )
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


def llm_judge_score(
    question: str, reference: str, prediction: str, client
) -> dict:
    """Use GPT as a judge to evaluate the prediction quality."""
    prompt = f"""당신은 한국 법률 QA 모델의 답변 품질을 평가하는 전문 평가관입니다.
아래의 질문, 정답, 모델 답변을 보고 3가지 기준으로 1~5점 척도로 채점하세요.

## 평가 기준
1. **정확성** (Correctness): 법률적 사실이 정답과 일치하는가
2. **완전성** (Completeness): 정답의 핵심 내용을 빠짐없이 다루는가
3. **충실도** (Faithfulness): 제공된 문서에 근거한 답변인가 (환각 여부)

## 입력
**질문**: {question}
**정답**: {reference}
**모델 답변**: {prediction}

## 출력 (반드시 아래 JSON 형식으로만 답하세요)
{{"correctness": <1-5>, "completeness": <1-5>, "faithfulness": <1-5>, "reason": "<한 줄 평가 근거>"}}"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
    )
    text = response.choices[0].message.content.strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            result = {"correctness": 0, "completeness": 0, "faithfulness": 0, "reason": "parse_error"}
    return result


def run_llm_judge(
    test_data: list[dict], predictions: list[str]
) -> list[dict]:
    """Run LLM-as-a-Judge on all test samples via OpenRouter."""
    import os
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    judge_results = []
    for sample, pred in tqdm(
        zip(test_data, predictions), total=len(predictions), desc="LLM Judge"
    ):
        result = llm_judge_score(
            question=sample["input"],
            reference=sample["output"],
            prediction=pred,
            client=client,
        )
        judge_results.append(result)
    return judge_results


def evaluate_trial(
    trial_name: str, test_data: list[dict], prompt_format: str, use_llm_judge: bool,
) -> dict:
    """Run full evaluation for a single trial."""
    model_path = MERGED_DIR / trial_name

    print(f"\n{'=' * 50}")
    print(f"Evaluating: {trial_name} ({model_path})")
    print(f"  Prompt format: {prompt_format}")
    print(f"{'=' * 50}")

    # ------ Load model ------
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model(model_path)
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # ------ Generate predictions ------
    print("\n[2/4] Generating predictions...")
    predictions = []
    references = []
    for sample in tqdm(test_data, desc=trial_name):
        prompt = build_prompt(sample, prompt_format)
        pred = generate_answer(model, tokenizer, prompt)
        predictions.append(pred)
        references.append(sample["output"])

    # Free GPU memory before metrics
    del model
    torch.cuda.empty_cache()

    # ------ Compute metrics ------
    print("\n[3/4] Computing metrics...")

    print("  Computing ROUGE-L...")
    rouge_scores = compute_rouge(predictions, references)

    print("  Computing BERTScore...")
    bert_scores = compute_bertscore(predictions, references)

    judge_scores = None
    judge_results = None
    if use_llm_judge:
        print("  Running LLM-as-a-Judge...")
        judge_results = run_llm_judge(test_data, predictions)
        judge_scores = {
            "correctness": np.mean([r["correctness"] for r in judge_results]),
            "completeness": np.mean([r["completeness"] for r in judge_results]),
            "faithfulness": np.mean([r["faithfulness"] for r in judge_results]),
        }

    # ------ Save per-trial results ------
    print("\n[4/4] Saving results...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "trial": trial_name,
        "model": str(model_path),
        "prompt_format": prompt_format,
        "test_samples": len(test_data),
        "rouge_l": rouge_scores,
        "bertscore": bert_scores,
    }
    if judge_scores:
        output["llm_judge"] = judge_scores

    results_file = RESULTS_DIR / f"eval_{trial_name}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    details_file = RESULTS_DIR / f"eval_{trial_name}_details.jsonl"
    with open(details_file, "w", encoding="utf-8") as f:
        for i, (sample, pred) in enumerate(zip(test_data, predictions)):
            detail = {
                "id": i,
                "question": sample["input"],
                "reference": sample["output"],
                "prediction": pred,
            }
            if judge_results:
                detail["judge"] = judge_results[i]
            f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    print(f"  Results: {results_file}")
    print(f"  Details: {details_file}")

    return output


def print_results(results: list[dict]):
    """Print results summary, with comparison table if multiple trials."""
    if len(results) == 1:
        r = results[0]
        print(f"\n{'=' * 50}")
        print(f"Results: {r['trial']}")
        print(f"{'=' * 50}")
        print(f"\n  ROUGE-L F1:    {r['rouge_l']['rouge_l_f1']:.4f}")
        print(f"  BERTScore F1:  {r['bertscore']['bertscore_f1']:.4f}")
        if "llm_judge" in r:
            j = r["llm_judge"]
            print(f"  Judge Avg:     {(j['correctness'] + j['completeness'] + j['faithfulness']) / 3:.2f}")
        return

    # Comparison table
    col_width = max(len(r["trial"]) for r in results) + 2
    col_width = max(col_width, 15)
    table_width = 25 + col_width * len(results)

    print(f"\n{'=' * table_width}")
    print("Model Comparison")
    print(f"{'=' * table_width}")

    header = f"{'Metric':<25}"
    for r in results:
        header += f"{r['trial']:>{col_width}}"
    print(header)
    print("-" * table_width)

    rows = [
        ("ROUGE-L Precision", "rouge_l", "rouge_l_precision"),
        ("ROUGE-L Recall", "rouge_l", "rouge_l_recall"),
        ("ROUGE-L F1", "rouge_l", "rouge_l_f1"),
        ("BERTScore Precision", "bertscore", "bertscore_precision"),
        ("BERTScore Recall", "bertscore", "bertscore_recall"),
        ("BERTScore F1", "bertscore", "bertscore_f1"),
    ]

    for label, group, key in rows:
        row = f"{label:<25}"
        values = [r[group][key] for r in results]
        best = max(values)
        for v in values:
            marker = " *" if len(results) > 1 and v == best else "  "
            row += f"{v:>{col_width - 2}.4f}{marker}"
        print(row)

    if all("llm_judge" in r for r in results):
        print()
        for metric in ["correctness", "completeness", "faithfulness"]:
            label = f"Judge {metric.title()}"
            row = f"{label:<25}"
            values = [r["llm_judge"][metric] for r in results]
            best = max(values)
            for v in values:
                marker = " *" if v == best else "  "
                row += f"{v:>{col_width - 2}.2f}{marker}"
            print(row)

    print(f"\n  (* = best)")


def main():
    available = discover_trials()

    parser = argparse.ArgumentParser(description="Evaluate Legal QA models")
    parser.add_argument("--trial", nargs="+", default=None,
                        help=f"Trial name(s) to evaluate (default: all detected). Available: {available}")
    parser.add_argument("--prompt-format", default="chatml", choices=["chatml", "alpaca"],
                        help="Prompt format (default: chatml)")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Enable LLM-as-a-Judge (requires OPENROUTER_API_KEY)")
    args = parser.parse_args()

    # Determine which trials to run
    trials = args.trial if args.trial else available
    if not trials:
        print("No trial models found under models/merged/trial*/")
        return

    # Validate paths
    for t in trials:
        path = MERGED_DIR / t
        if not path.exists():
            parser.error(f"Model not found: {path}")

    print("=" * 50)
    print("Legal QA Model Evaluation")
    print(f"  Trials: {', '.join(trials)}")
    print(f"  Prompt: {args.prompt_format}")
    print("=" * 50)

    # ------ Load test data ------
    print("\nLoading test data...")
    test_data = []
    with open(TEST_DATA, "r", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"  Test samples: {len(test_data)}")

    # ------ Evaluate each trial ------
    all_results = []
    for trial_name in trials:
        result = evaluate_trial(trial_name, test_data, args.prompt_format, args.llm_judge)
        all_results.append(result)

    # ------ Print comparison ------
    print_results(all_results)
    print("\nDone!")


if __name__ == "__main__":
    main()
