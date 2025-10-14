import os, json, argparse, shutil
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--test_file", type=str, required=True)
parser.add_argument("--context", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--tensor_parallel_size", type=int, default=1)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print(f"\n[INFO] Running inference with vLLM")
print(f"Checkpoint: {args.checkpoint}")
print(f"Test file: {args.test_file}")
print(f"Context: {args.context}")
print(f"Saving to: {args.output_dir}\n")

# Load dataset
prompts, references, qids = [], [], []
with open(args.test_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        prompts.append(item['prompt'])
        qids.append(item["qid"])
        references.append(item["completion"])

print(f"[INFO] Loaded {len(prompts)} examples from dataset.")

# Load model with vLLM
llm = LLM(
    model=args.checkpoint,
    tensor_parallel_size=args.tensor_parallel_size,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=300,
)

# ------------------------------
# Resume Support (skip completed batches)
# ------------------------------
completed_batches = len([f for f in os.listdir(args.output_dir) if f.endswith(".txt")])
print(f"[INFO] Found {completed_batches} completed batches, skipping those...")

# Inference loop
total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
for batch_idx in range(completed_batches, total_batches):
    start = batch_idx * args.batch_size
    end = min((batch_idx + 1) * args.batch_size, len(prompts))

    batch_prompts = prompts[start:end]
    batch_qids = qids[start:end]
    batch_refs = references[start:end]

    outputs = llm.generate(batch_prompts, sampling_params)

    output_path = os.path.join(args.output_dir, f"batch_{batch_idx}.txt")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for qid, prompt, ref, out in zip(batch_qids, batch_prompts, batch_refs, outputs):
            generated_answer = out.outputs[0].text.strip()
            out_f.write(f"QID: {qid}\n")
            out_f.write(f"Prompt: {prompt}\n")
            out_f.write(f"Reference Answer: {ref}\n")
            out_f.write(f"Generated Answer: {generated_answer}\n")
            out_f.write("-----\n")

    print(f"[INFO] Saved batch {batch_idx} to {output_path}")

print("\n[INFO] Inference completed for all examples.")
