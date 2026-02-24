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

