# legal-chatbot
LoRA fine-tuning of Qwen2.5-3B for Korean legal QA using Unsloth with 4-bit quantized training.

## Progress
- Completed model training and export to ONNX.
- Failed to compile ONNX to HAR using Hailo Dataflow Compiler (DFC), a prerequisite step before compiling HAR to HEF.

## TODO
- [X] Curate data and reform response format
- [X] Construct RAFT data (statutes and judgment)
- [X] Train LoRA adapter in Qwen2-1.5b-instruct
- [ ] Combine LoRA adapter and pre-compiled model file (HAR) and compile with DFC

## Result

* The score may not be related to output quality.

Metric | Trial 1 | Trial 2
-----------|-------|-------|
Rouge-L Precision|0.2060|**0.2776**
Rouge-L Recall|0.1310|**0.2598**
Rouge-L F1|0.2060|**0.2776**
BERTScore Precision|0.6487|**0.6979**
BERTScore Recall|0.6516|**0.6934**
BERTScore F1|0.6484|**0.6946**


## Data Reference

- [jihye-moon/klac_legal_aid_counseling](https://huggingface.co/datasets/jihye-moon/klac_legal_aid_counseling)

