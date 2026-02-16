# legal-chatbot
LoRA fine-tuning of Qwen2.5-3B for Korean legal QA using Unsloth with 4-bit quantized training.

## Progress
- Completed model training and export to ONNX.
- Failed to compile ONNX to HAR using Hailo Dataflow Compiler (DFC), a prerequisite step before compiling HAR to HEF.

## TODO
- [ ] Curate data and reform response format
- [ ] Construct RAFT data (statutes and judgment)
- [ ] Train LoRA adapter in Qwen2-1.5b-instruct
- [ ] Combine LoRA adapter and pre-compiled model file (HAR) and compile with DFC

## Result Trial1
- Good quality but overly verbose responses — need to reduce output length and formalize it in the next training iteration.

<img src="docs/result_trial1.png" width="80%">


