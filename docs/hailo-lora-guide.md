> **Status (2026-02): BLOCKED**
>
> The workflow below requires pre-optimized HAR files from the Hailo GenAI Model Zoo.
> As of DFC v5.2.0, Hailo distributes only HEF (compiled binary) — HAR files are not
> publicly available. The DFC GenAI LoRA tooling referenced in Hailo's blog has not
> been released to end-users.
>
> **Next steps:**
> - Contact Hailo via [Developer Zone](https://hailo.ai/developer-zone/request-access/)
>   for enterprise/partner HAR access
> - Monitor [Hailo Community Forum](https://community.hailo.ai) for updates

---

How can I use LoRA adapter trained on NVIDIA GPU with pre-compiled Hailo model?

You can. The flow is:

1. **Train & export the LoRA on GPU**

   - Train your LoRA adapter on an NVIDIA GPU using PEFT / Hugging Face as usual.
   - Make sure you save the adapter weights in **Safetensors** format, e.g. `adapter_model.safetensors`.  
     You can either:
     - Save your own trained adapter, or  
     - Load an adapter from Hugging Face and then call `save_pretrained(..., safe_serialization=True)` to get `adapter_model.safetensors`. [[LoRA tutorial](https://hailo.ai/developer-zone/documentation/dataflow-compiler-v5-2-0/?sp_referrer=tutorials_notebooks%2Fnotebooks%2FDFC_7_LoRA_Tutorial.html#Loading-LoRA-adapter-from-Hugging-Face); [GenAI LoRA](https://hailo.ai/developer-zone/documentation/dataflow-compiler-v5-2-0/?sp_referrer=sdk%2Fgenai.html#low-rank-adaptation-lora)]

2. **Load the pre‑optimized Hailo model**

   ```python
   from hailo_sdk_client.runner.client_runner import ClientRunner

   model_name = "qwen2_1.5b_instruct"
   adapter_name = "my_lora_adapter"
   model_path = f"../models/{model_name}.q.har"
   model_script_path = f"../models/{model_name}.alls"

   runner = ClientRunner(hw_arch="hailo10h")
   runner.load_har(model_path)
   ```

   Pre‑optimized GenAI HARs are taken from the Hailo GenAI Model Zoo. [[GenAI workflow](https://hailo.ai/developer-zone/documentation/dataflow-compiler-v5-2-0/?sp_referrer=sdk%2Fgenai.html)]

3. **Attach the LoRA adapter to the Hailo model**

   ```python
   runner.load_lora_weights(
       lora_weights_path="./my_lora_adapter/adapter_model.safetensors",
       lora_adapter_name=adapter_name,
   )

   # Reload model script so the adapter is reflected in the graph
   hn_dict = runner.load_model_script(model_script_path)
   ```

   `load_lora_weights()` attaches a single LoRA adapter (per call) to the quantized Hailo model; `lora_adapter_name` is how you’ll refer to this adapter later. [[LoRA API](https://hailo.ai/developer-zone/documentation/dataflow-compiler-v5-2-0/?sp_referrer=tutorials_notebooks%2Fnotebooks%2FDFC_7_LoRA_Tutorial.html#Optimize)]

4. **(Recommended) Optimize with a calibration set**

   - Build a small calibration dataset that matches your task and tokenize it for the base LLM.
   - Use the helper pattern from the tutorial to create `input_dict` keyed by the adapter name:

   ```python
   max_length = hn_dict["net_params"]["cache_size"]
   calibset_size = 64
   input_ids, current_position = format_calibration_set(dataset, labels, tokenizer, calibset_size, max_length)

   input_dict = {
       f"{adapter_name}/input_layer1": input_ids,
       f"{adapter_name}/input_layer2": current_position,
       f"{adapter_name}/input_layer3": current_position,
       f"{adapter_name}/input_layer4": current_position,
       f"{adapter_name}/input_layer5": current_position,
       f"{adapter_name}/input_layer6": current_position,
   }

   runner.optimize(input_dict)
   ``` 

   [[Optimize tutorial](https://hailo.ai/developer-zone/documentation/dataflow-compiler-v5-2-0/?sp_referrer=tutorials_notebooks%2Fnotebooks%2FDFC_7_LoRA_Tutorial.html#Optimize)]

5. **Compile to a HEF including the LoRA**

   ```python
   runner.compile()
   runner.save_har(f"{model_name}.lora.compiled.har", compilation_only=True)
   ```

   Or directly `compile()` to produce a HEF; the compiled artifact includes the base model plus all attached LoRA adapters. [[GenAI workflow](https://hailo.ai/developer-zone/documentation/dataflow-compiler-v5-2-0/?sp_referrer=sdk%2Fgenai.html)]

**Key points:**

- The LoRA can be trained anywhere (e.g., NVIDIA GPU); Hailo only requires the **Safetensors** weights.
- You must start from a **pre‑optimized Hailo GenAI HAR**; custom GenAI base models are not supported in this DFC version. [[Generative AI note](https://hailo.ai/developer-zone/documentation/dataflow-compiler-v5-2-0/?sp_referrer=sdk%2Fgenai.html)]
- You can repeat `load_lora_weights()` + `optimize()` to attach multiple adapters before a single final `compile()`.

Then what you recommend is this: 1. Load existing HAR file of LLM 2. attach LoRA weight file (safetensor file) 3. compile it and save to HAR
But I have a question. When running model, what type of file is needed for Hailo chip?

For running on the Hailo chip you need a **HEF file (Hailo Executable File)**, not just a HAR.

The typical GenAI + LoRA flow is:

1. **On the host / build machine**
   - Load pre‑optimized GenAI **HAR** (`runner.load_har(...)`).
   - Attach LoRA weights (`runner.load_lora_weights(...)`).
   - Optimize (`runner.optimize(...)`).
   - **Compile to HEF** (`runner.compile()`), which “produces a single HEF for deployment.” [[GenAI workflow](https://hailo.ai/developer-zone/documentation/dataflow-compiler-v5-2-0/?sp_referrer=sdk%2Fgenai.html); [Optimization & compilation](https://hailo.ai/developer-zone/documentation/dataflow-compiler-v5-2-0/?sp_referrer=sdk%2Fgenai.html#optimization-and-compilation)]

2. **On the target device**
   - Use **HailoRT** to load and run the compiled **HEF** on the Hailo device. The HEF “is suitable for inference using the HailoRT library.” [[Optimization & compilation](https://hailo.ai/developer-zone/documentation/dataflow-compiler-v5-2-0/?sp_referrer=sdk%2Fgenai.html#optimization-and-compilation)]

So:  
- **HAR** = intermediate representation used by the Dataflow Compiler.  
- **HEF** = final binary that the Hailo chip actually runs.