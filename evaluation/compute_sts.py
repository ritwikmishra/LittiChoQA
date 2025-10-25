#!/usr/bin/env python3
import os
import json
import argparse
from bert_score import BERTScorer
from sts_utils import *   # must provide: sts(gen, ref, scorer, use_model, labse_preproc, labse_enc, laser_enc)
import tensorflow_hub as hub
from laser_encoders import LaserEncoderPipeline
import tensorflow_text  # required by TF-Hub text models
import gc
from tqdm import tqdm
import re

# ------------------------------
# Arguments
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True, help='Model name')
parser.add_argument('--context_type', type=str, required=True, help='Context type (e.g., long, short_l6)')
parser.add_argument('--split_index', type=int, required=True, help='Which split to process (0-based)')
parser.add_argument('--total_splits', type=int, default=4, help='Total number of splits')
args = parser.parse_args()

# ------------------------------
# Paths
# ------------------------------
BATCH_DIR = f"../inferencing/outputs/{args.model_name}/{args.context_type}"
OUTPUT_DIR = f"sts_evaluation/{args.model_name}/{args.context_type}"

if not os.path.exists(BATCH_DIR):
    raise FileNotFoundError(f"Batch directory not found: {BATCH_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Split Handling
# ------------------------------
batch_files = sorted([
    os.path.join(BATCH_DIR, f)
    for f in os.listdir(BATCH_DIR)
    if f.startswith("batch_") and f.endswith(".txt")
], key = lambda d:int(re.search(r"batch_\d+",d)[0].split("_")[1]))
total_files = len(batch_files)

split_size = (total_files + args.total_splits - 1) // args.total_splits  # ceil division
start_index = args.split_index * split_size
end_index = min(start_index + split_size, total_files)

split_batch_files = batch_files[start_index:end_index]
print(f"\nProcessing split {args.split_index+1}/{args.total_splits}: {len(split_batch_files)} batch files ({start_index}–{end_index-1})")

# ------------------------------
# Load Models
# ------------------------------
print("\nLoading STS models...")
with torch.device("cuda"): 
    scorer = BERTScorer(model_type='bert-base-multilingual-cased', device='cpu')

with tf.device('/CPU:0'): # '/GPU:0' '/CPU:0'
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    # https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2
    # https://tfhub.dev/google/universal-sentence-encoder/4
    use_model = hub.load(module_url)
    labse_preprocessor = hub.KerasLayer("https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/cmlm-multilingual-preprocess/2")
    labse_encoder = hub.KerasLayer("https://www.kaggle.com/models/google/labse/TensorFlow2/labse/2")
with tf.device('/GPU:0'):
    laser_encoder = LaserEncoderPipeline(laser="laser2")

# ------------------------------
# Evaluation Loop
# ------------------------------
for batch_file in tqdm(split_batch_files, desc=f"Evaluating {args.model_name}/{args.context_type}"):
    batch_name = os.path.basename(batch_file).replace(".txt", "")
    output_file = os.path.join(OUTPUT_DIR, f"{batch_name}.json")

    # Skip already processed files
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"Skipping {batch_name}, already processed.")
        continue

    # --------------------------
    # Load and parse batch file
    # --------------------------
    references, generated_texts, qids = [], [], []
    with open(batch_file, "r", encoding="utf-8") as f:
        content = f.read()

    qid_blocks = content.split("QID:")[1:]
    for block in qid_blocks:
        qid_end = block.find("Prompt:")
        qid = block[:qid_end].strip() if qid_end != -1 else "unknown"

        if "Reference Answer:" in block and "Generated Answer:" in block:
            ref_start = block.index("Reference Answer:") + len("Reference Answer:")
            gen_start = block.index("Generated Answer:")
            ref_answer = block[ref_start:gen_start].strip()

            gen_start = block.index("Generated Answer:") + len("Generated Answer:")
            if "-----" in block[gen_start:]:
                gen_end = block.index("-----", gen_start)
                gen_answer = block[gen_start:gen_end].strip()
            else:
                gen_answer = block[gen_start:].strip()

            references.append(ref_answer)
            generated_texts.append(gen_answer)
            qids.append(qid)

    # --------------------------
    # Compute STS
    # --------------------------
    sts_scores = []
    for qid, gen, ref in zip(qids, generated_texts, references):
        bert_score, use_score, labse_score, laser_score = sts(
            gen, ref, scorer, use_model, labse_preprocessor, labse_encoder, laser_encoder
        )
        sts_scores.append({
            "qid": qid,
            "bert_score": bert_score,
            "use_score": use_score,
            "labse_score": labse_score,
            "laser_score": laser_score
        })

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sts_scores, f, indent=4, ensure_ascii=False)

# ------------------------------
# Verification & Cleanup
# ------------------------------
print("\nVerifying completeness...")
missing_files = []
for batch_file in split_batch_files:
    batch_name = os.path.basename(batch_file).replace(".txt", "")
    output_file = os.path.join(OUTPUT_DIR, f"{batch_name}.json")
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        missing_files.append(batch_name)

if missing_files:
    print(f" Warning: {len(missing_files)} batch outputs missing for {args.model_name}/{args.context_type}:")
    for name in missing_files:
        print(f"   - {name}")
else:
    print(" All batch outputs are present and processed.")

# Cleanup
del scorer, use_model, labse_preprocessor, labse_encoder, laser_encoder
gc.collect()
print(f"\nFinished processing split {args.split_index+1}/{args.total_splits} for {args.model_name}/{args.context_type}")
