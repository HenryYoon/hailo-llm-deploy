# hailo-llm-deploy

HuggingFace LLM fine-tuning and Hailo NPU edge deployment pipeline.

> Refactored from [legal-chatbot](https://github.com/HenryYoon/legal-chatbot) — Korean legal QA with RAFT fine-tuning.

## Project Structure

```
hailo_llm_deploy/          # Generic pipeline package
├── cli.py                 # Typer CLI (hailo-llm-deploy)
├── config.py              # Pydantic config (YAML)
├── finetune.py            # FineTuner — LoRA fine-tuning
├── export.py              # ModelExporter — ONNX export
└── evaluate.py            # Evaluator — ROUGE-L, BERTScore, LLM Judge

examples/korean-legal/     # Domain-specific example
├── construct.py           # 5-step RAFT pipeline orchestrator
├── sampler.py             # Sampler — stratified sampling
├── extractor.py           # ReferenceExtractor — legal citation extraction
├── collector.py           # LawApiCollector — law API data collection
├── chunker.py             # DocumentChunker — document chunking
├── raft_builder.py        # RaftBuilder — RAFT dataset assembly
└── convert_formal.py      # FormalConverter — style conversion via LLM

configs/                   # YAML configuration files
├── default.yaml
└── examples/
    └── korean_legal.yaml
```

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Fine-tune
hailo-llm-deploy finetune --config configs/examples/korean_legal.yaml

# Export to ONNX
hailo-llm-deploy export --model models/merged/trial2.1 --output models/onnx/trial2.1

# Evaluate
hailo-llm-deploy evaluate --config configs/examples/korean_legal.yaml
```

## TODO

- [X] Curate data and reform response format
- [X] Construct RAFT data (statutes and judgment)
- [X] Train LoRA adapter (Qwen2.5-1.5B-Instruct)
- [X] Refactor codebase (class-based, domain/generic separation)
- [ ] Combine LoRA adapter with pre-compiled HAR and compile with DFC
- [ ] CLI wrapping (Phase 2)
- [ ] Hailo pipeline integration (Phase 3)

## Evaluation Results

Metric | Trial 1 | Trial 2 | Trial 2-1|
-----------|:-------:|:-------:|:-------:|
Rouge-L Precision|0.3498|0.3154|**0.4854**|
Rouge-L Recall|0.0970|0.1532|**0.4192**|
Rouge-L F1|0.1302|0.1726|**0.4105**|
BERTScore Precision|0.6958|0.7016|**0.7652**|
BERTScore Recall|0.6501|0.6674|**0.7673**|
BERTScore F1|0.6701|0.6830|**0.7655**|
LLM-as-a-Judge Correctness|2.64|2.36|**3.10**|
LLM-as-a-Judge Completeness|2.44|2.22|**2.87**|
LLM-as-a-Judge Faithfulness|2.69|2.31|**3.27**|

## Data Reference

- [jihye-moon/klac_legal_aid_counseling](https://huggingface.co/datasets/jihye-moon/klac_legal_aid_counseling)

## License

Apache-2.0
