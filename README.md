# hailo-llm-deploy

HuggingFace LLM fine-tuning and Hailo NPU edge deployment pipeline.

> Refactored from legal-chatbot — Korean legal QA with RAFT fine-tuning.

## Project Structure

```
hailo_llm_deploy/          # Generic pipeline package
├── cli.py                 # Typer CLI (hailo-llm-deploy)
├── config.py              # Pydantic config (YAML)
├── finetune.py            # FineTuner — LoRA fine-tuning
├── export.py              # ModelExporter — ONNX export
├── compile.py             # HailoCompiler — HAR + LoRA → HEF (blocked)
└── evaluate.py            # Evaluator — ROUGE-L, BERTScore, LLM Judge

examples/korean-legal/     # Domain-specific example
├── construct.py           # 5-step RAFT pipeline orchestrator
├── sampler.py             # Sampler — stratified sampling
├── extractor.py           # ReferenceExtractor — legal citation extraction
├── collector.py           # LawApiCollector — law API data collection
├── chunker.py             # DocumentChunker — document chunking
├── raft_builder.py        # RaftBuilder — RAFT dataset assembly
└── convert_formal.py      # FormalConverter — style conversion via LLM

tests/                     # pytest test suite (69 tests)
docs/                      # Documentation
└── hailo-lora-guide.md    # Hailo LoRA compilation guide

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

# Compile for Hailo NPU (blocked — requires HAR files not yet publicly available)
hailo-llm-deploy compile --config configs/examples/korean_legal.yaml --force
```

## TODO

- [X] Curate data and reform response format
- [X] Construct RAFT data (statutes and judgment)
- [X] Train LoRA adapter (Qwen2.5-1.5B-Instruct)
- [X] Refactor codebase (class-based, domain/generic separation)
- [X] Add compile pipeline code (HAR + LoRA → HEF)
- [X] Add test suite (69 tests)
- [ ] Obtain Hailo GenAI HAR files (blocked — not publicly available as of 2026-02)
- [ ] Compile LoRA adapter with HAR and deploy HEF to Hailo-10H

## Known Limitations

### Hailo NPU Compilation (Blocked)

The compile pipeline (`hailo-llm-deploy compile`) requires pre-optimized HAR files
from the Hailo GenAI Model Zoo. As of 2026-02, Hailo distributes only HEF (compiled
binary), not HAR (intermediate representation). LoRA adapter attachment requires HAR.

**Status**: Waiting for Hailo to release GenAI HAR files or LoRA compilation tools.
See [docs/hailo-lora-guide.md](docs/hailo-lora-guide.md) for details.

## Trial Details
- Trial 1: Train model with full dataset in 1 epoch.
- Trial 2: Train model with selected dataset in 1 epoch.
- Trial 3: Train model with selected dataset in 10 epochs.

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
