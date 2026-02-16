from optimum.exporters.onnx import main_export
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "../../models/merged/stage1_16bit"
save_dir = "../../models/onnx/legal_ai"

if __name__ == "__main__":
    # Export with explicit float16 dtype (not bfloat16)
    # Load model first
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use float16 instead of bfloat16
        trust_remote_code=True,
        device_map="cpu"  # Keep on CPU for export
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"Exporting to ONNX with float16 dtype...")
    main_export(
        model_name_or_path=model_id,
        output=save_dir,
        task="text-generation",
        trust_remote_code=True,
    )