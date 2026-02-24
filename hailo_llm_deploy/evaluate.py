"""Model evaluation with ROUGE-L, BERTScore, and optional LLM-as-a-Judge."""

import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from tqdm import tqdm
from unsloth import FastLanguageModel

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate fine-tuned LLM models with multiple metrics.

    Encapsulates model, tokenizer, and evaluation state to eliminate
    parameter threading across functions.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.llm_client = None
        self.test_data: list[dict] = []

    def load_test_data(self):
        """Load test dataset from JSONL file."""
        test_path = self.config.data.test_path
        if not test_path or not Path(test_path).exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        self.test_data = []
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                self.test_data.append(json.loads(line))
        logger.info("Loaded %d test samples", len(self.test_data))

    def discover_trials(self, merged_dir: Path) -> list[str]:
        """Auto-detect trial directories under merged_dir/trial*."""
        if not merged_dir.exists():
            return []
        trials = sorted(
            [d.name for d in merged_dir.iterdir()
             if d.is_dir() and d.name.startswith("trial")],
            key=lambda x: int(re.search(r"\d+", x).group())
            if re.search(r"\d+", x) else 0,
        )
        return trials

    def load_model(self, model_path: Path):
        """Load a fine-tuned model for inference."""
        logger.info("Loading model: %s", model_path)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=self.config.model.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        logger.info("GPU memory: %.2f GB", torch.cuda.memory_allocated() / 1024**3)

    def build_prompt(self, sample: dict) -> str:
        """Build prompt from a test sample using the configured format."""
        fmt = self.config.evaluate.prompt_format
        if fmt == "chatml":
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

    def generate_answer(self, prompt: str) -> str:
        """Generate a single answer from the loaded model."""
        ec = self.config.evaluate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=ec.max_new_tokens,
                temperature=ec.temperature,
                top_p=ec.top_p,
                repetition_penalty=ec.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @staticmethod
    def compute_rouge(predictions: list[str], references: list[str]) -> dict:
        """Compute ROUGE-L scores."""
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
        return {
            "rouge_l_precision": np.mean([s["rougeL"].precision for s in scores]),
            "rouge_l_recall": np.mean([s["rougeL"].recall for s in scores]),
            "rouge_l_f1": np.mean([s["rougeL"].fmeasure for s in scores]),
        }

    @staticmethod
    def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
        """Compute BERTScore using multilingual model."""
        P, R, F1 = bert_score_fn(predictions, references, lang="ko", verbose=True)
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }

    def _get_llm_client(self):
        """Lazily initialize the LLM client for judge scoring."""
        if self.llm_client is None:
            import os
            from openai import OpenAI
            self.llm_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        return self.llm_client

    def llm_judge_score(self, question: str, reference: str, prediction: str) -> dict:
        """Use LLM as a judge to evaluate prediction quality."""
        client = self._get_llm_client()
        prompt = (
            "You are an expert evaluator for a QA model.\n"
            "Score the model's answer on a 1-5 scale for:\n"
            "1. **Correctness**: factual accuracy\n"
            "2. **Completeness**: coverage of key points\n"
            "3. **Faithfulness**: grounded in provided context\n\n"
            f"**Question**: {question}\n"
            f"**Reference**: {reference}\n"
            f"**Model Answer**: {prediction}\n\n"
            'Output JSON only: {"correctness": <1-5>, "completeness": <1-5>, '
            '"faithfulness": <1-5>, "reason": "<one-line justification>"}'
        )
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        text = response.choices[0].message.content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"correctness": 0, "completeness": 0, "faithfulness": 0, "reason": "parse_error"}

    def run_llm_judge(self, predictions: list[str]) -> list[dict]:
        """Run LLM-as-a-Judge on all test samples."""
        results = []
        for sample, pred in tqdm(
            zip(self.test_data, predictions), total=len(predictions), desc="LLM Judge"
        ):
            result = self.llm_judge_score(
                question=sample["input"],
                reference=sample["output"],
                prediction=pred,
            )
            results.append(result)
        return results

    def evaluate_trial(self, trial_name: str, model_path: Path, results_dir: Path) -> dict:
        """Run full evaluation for a single trial."""
        logger.info("Evaluating: %s", trial_name)

        self.load_model(model_path)

        predictions = []
        references = []
        for sample in tqdm(self.test_data, desc=trial_name):
            prompt = self.build_prompt(sample)
            pred = self.generate_answer(prompt)
            predictions.append(pred)
            references.append(sample["output"])

        del self.model
        self.model = None
        torch.cuda.empty_cache()

        logger.info("Computing ROUGE-L...")
        rouge_scores = self.compute_rouge(predictions, references)

        logger.info("Computing BERTScore...")
        bert_scores = self.compute_bertscore(predictions, references)

        judge_scores = None
        judge_results = None
        if self.config.evaluate.llm_judge:
            logger.info("Running LLM-as-a-Judge...")
            judge_results = self.run_llm_judge(predictions)
            judge_scores = {
                "correctness": np.mean([r["correctness"] for r in judge_results]),
                "completeness": np.mean([r["completeness"] for r in judge_results]),
                "faithfulness": np.mean([r["faithfulness"] for r in judge_results]),
            }

        results_dir.mkdir(parents=True, exist_ok=True)
        output = {
            "trial": trial_name,
            "model": str(model_path),
            "prompt_format": self.config.evaluate.prompt_format,
            "test_samples": len(self.test_data),
            "rouge_l": rouge_scores,
            "bertscore": bert_scores,
        }
        if judge_scores:
            output["llm_judge"] = judge_scores

        results_file = results_dir / f"eval_{trial_name}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        details_file = results_dir / f"eval_{trial_name}_details.jsonl"
        with open(details_file, "w", encoding="utf-8") as f:
            for i, (sample, pred) in enumerate(zip(self.test_data, predictions)):
                detail = {
                    "id": i,
                    "question": sample["input"],
                    "reference": sample["output"],
                    "prediction": pred,
                }
                if judge_results:
                    detail["judge"] = judge_results[i]
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

        logger.info("Results saved: %s", results_file)
        return output

    @staticmethod
    def print_results(results: list[dict]):
        """Print comparison table for evaluation results."""
        if len(results) == 1:
            r = results[0]
            print(f"\nResults: {r['trial']}")
            print(f"  ROUGE-L F1:    {r['rouge_l']['rouge_l_f1']:.4f}")
            print(f"  BERTScore F1:  {r['bertscore']['bertscore_f1']:.4f}")
            if "llm_judge" in r:
                j = r["llm_judge"]
                avg = (j["correctness"] + j["completeness"] + j["faithfulness"]) / 3
                print(f"  Judge Avg:     {avg:.2f}")
            return

        col_width = max(len(r["trial"]) for r in results) + 2
        col_width = max(col_width, 15)

        header = f"{'Metric':<25}"
        for r in results:
            header += f"{r['trial']:>{col_width}}"
        print(header)
        print("-" * (25 + col_width * len(results)))

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

        print("\n  (* = best)")
