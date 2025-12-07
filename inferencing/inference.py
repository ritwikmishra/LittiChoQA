import os, random, torch, json, argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
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
# Argument Parser
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--test_file", type=str, required=True, help="Path to test.jsonl")
parser.add_argument("--context", type=str, required=True, help="Context string to prepend")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to fine-tuned checkpoint")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size (examples per file)")
parser.add_argument("--output_dir", type=str, default="outputs", help="Where to save .txt outputs")
parser.add_argument("--base_model", type=str, required=True, help="Base model name")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print(f"\n[INFO] Running inference")
print(f"Base model: {args.base_model}")
print(f"Checkpoint: {args.checkpoint}")
print(f"Test file: {args.test_file}")
print(f"Context: {args.context}")
print(f"Saving to: {args.output_dir}\n")

# ------------------------------
# Dataset Loading (JSONL)
# ------------------------------
prompts, references, qids = [], [], []

with open(args.test_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        prompts.append(item['prompt'])
        qids.append(item["qid"])
        references.append(item["completion"])

print(f"[INFO] Loaded {len(prompts)} examples from dataset.")

# ------------------------------
# Load Model from Checkpoint
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.base_model)

# Load base model without quantization
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    device_map="auto",
    torch_dtype=torch.float32,  # quantization OFF
)

# Load checkpoint (LoRA / PEFT)
print("[INFO] Loading fine-tuned checkpoint...")
model = PeftModel.from_pretrained(model, args.checkpoint)

# Merge LoRA weights and unload wrapper
print("[INFO] Merging LoRA weights and unloading PEFT wrapper...")
model = model.merge_and_unload()
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# ------------------------------
# Resume Support (skip completed batches)
# ------------------------------
completed_batches = len([f for f in os.listdir(args.output_dir) if f.endswith(".txt")])
print(f"[INFO] Found {completed_batches} completed batches, skipping those...")

# ------------------------------
# Inference Loop
# ------------------------------
for batch_idx in range(completed_batches, (len(prompts) + args.batch_size - 1) // args.batch_size):
    start = batch_idx * args.batch_size
    end = min((batch_idx + 1) * args.batch_size, len(prompts))

    batch_prompts = prompts[start:end]
    batch_qids = qids[start:end]
    batch_refs = references[start:end]

    output_path = os.path.join(args.output_dir, f"batch_{batch_idx}.txt")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for prompt, qid, ref in zip(batch_prompts, batch_qids, batch_refs):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            gen_only_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_answer = tokenizer.decode(gen_only_ids, skip_special_tokens=True).strip()

            # Write structured result
            out_f.write(f"QID: {qid}\n")
            out_f.write(f"Prompt: {prompt}\n")
            out_f.write(f"Reference Answer: {ref}\n")
            out_f.write(f"Generated Answer: {generated_answer}\n")
            out_f.write("-----\n")

    print(f"[INFO] Saved batch {batch_idx} to {output_path}")

print("\n[INFO] Inference completed for all examples.")
