from transformers import AutoModelForCausalLM, AutoTokenizer
import os, random, torch, json
import numpy as np
from peft import PeftModel

# ------------------------------
# Fixing the seeds
# ------------------------------
rseed = 123
os.environ['PYTHONHASHSEED'] = str(rseed)
torch.manual_seed(rseed)
torch.cuda.manual_seed(rseed)
torch.cuda.manual_seed_all(rseed)
np.random.seed(rseed)
random.seed(rseed)

# ------------------------------
# Dataset Loading (JSONL)
# ------------------------------
dataset_path = "/media/data_dump/aarya220007/data_test/fc_dataset/test.jsonl"

prompts = []
references = []

with open(dataset_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3:  # only first 3 lines
            break
        item = json.loads(line)
        prompts.append((item["prompt"], item["qid"]))
        references.append(item["completion"])

print(f"Loaded {len(prompts)} examples from dataset.")

# ------------------------------
# Load Model from Checkpoint
# ------------------------------
checkpoint_path = "/path/to/your/checkpoint"  # <--- specify your checkpoint
base_model_name = "meta-llama/Llama-3.1-8B-Instruct"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model without quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float32,  # ensure quantization is OFF
)

# Load checkpoint (LoRA / PEFT)
print("Loading fine-tuned checkpoint...")
model = PeftModel.from_pretrained(model, checkpoint_path)

# Merge LoRA weights and unload wrapper
print("Merging LoRA weights and unloading PEFT wrapper...")
model = model.merge_and_unload()
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# ------------------------------
# Inference
# ------------------------------
generated_texts = []

for idx, (prompt, qid) in enumerate(prompts):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract only generated part after input
    gen_only_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_answer = tokenizer.decode(gen_only_ids, skip_special_tokens=True).strip()
    generated_texts.append(generated_answer)

    print(f"\n[DEBUG] QID: {qid}")
    print(f"[DEBUG] Prompt: {prompt}")
    print(f"[DEBUG] Generated Completion: {generated_answer}")
    print(f"[DEBUG] Reference Completion: {references[idx]}")

print("\nInference completed for first 3 examples.")
