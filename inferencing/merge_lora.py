import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, required=True)
parser.add_argument("--lora_checkpoint", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

print(f"[INFO] Loading base model {args.base_model}")
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    device_map="auto",
    torch_dtype=torch.float16,
)

# print(f"[INFO] Loading LoRA checkpoint {args.lora_checkpoint}")
# model = PeftModel.from_pretrained(model, args.lora_checkpoint)

# print("[INFO] Merging LoRA weights into base model...")
# model = model.merge_and_unload()

# print(f"[INFO] Saving merged model to {args.output_dir}")
model.save_pretrained(args.output_dir)
tokenizer = AutoTokenizer.from_pretrained(args.base_model)
tokenizer.save_pretrained(args.output_dir)

# print("[INFO] Merge complete.")
