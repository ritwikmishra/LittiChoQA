#!/usr/bin/env python3
import os

# Prefer CPU for parallel splits; comment these two lines if you WANT GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""      # hide GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"     # reduce TF verbosity

import json
import argparse
from bert_score import BERTScorer
from sts_utils import *   # must provide: sts(gen, ref, scorer, use_model, labse_preproc, labse_enc, laser_enc)
import tensorflow_hub as hub
from laser_encoders import LaserEncoderPipeline
import tensorflow_text  # required by some TF-Hub text models, even if unused directly
import gc

# ------------------------------
# Arguments
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--batch_file', type=str, required=True, help='Path to batch file')
parser.add_argument('--output_file', type=str, required=True, help='Path to save results (json)')
args = parser.parse_args()

# ------------------------------
# Dataset Loading
# ------------------------------
references = []
generated_texts = []
qids = []

with open(args.batch_file, "r", encoding="utf-8") as f:
    content = f.read()

# Expect blocks delimited by "QID:" ... "Prompt:" ... "Reference Answer:" ... "Generated Answer:" ... "-----"
qid_blocks = content.split("QID:")[1:]

for block in qid_blocks:
    ref_answer = None
    gen_answer = None

    # Extract QID
    qid_end = block.find("Prompt:")
    qid = block[:qid_end].strip() if qid_end != -1 else "unknown"

    # Extract reference
    if "Reference Answer:" in block and "Generated Answer:" in block:
        ref_start = block.index("Reference Answer:") + len("Reference Answer:")
        gen_start = block.index("Generated Answer:")
        ref_answer = block[ref_start:gen_start].strip()

    # Extract generated
    if "Generated Answer:" in block:
        gen_start = block.index("Generated Answer:") + len("Generated Answer:")
        if "-----" in block[gen_start:]:
            gen_end = block.index("-----", gen_start)
            gen_answer = block[gen_start:gen_end].strip()
        else:
            gen_answer = block[gen_start:].strip()

    if ref_answer and gen_answer:
        references.append(ref_answer)
        generated_texts.append(gen_answer)
        qids.append(qid)

# ------------------------------
# STS Evaluation
# ------------------------------
print("\nLoading STS models...")
scorer = BERTScorer(model_type='bert-base-multilingual-cased', device='cpu')

# Adjust these paths to your local TF-Hub model dirs or URLs
use_model = hub.load("hub_models/universal-sentence-encoder")
labse_preprocessor = hub.KerasLayer("hub_models/labse-preprocessor")
labse_encoder = hub.KerasLayer("hub_models/labse-encoder")

# LASER pipeline (ensure its resources are installed/configured)
laser_encoder = LaserEncoderPipeline(laser="laser2")

print("\nComputing STS scores...")
sts_scores = []
total = len(references)

for i, (qid, gen, ref) in enumerate(zip(qids, generated_texts, references), start=1):
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

# ------------------------------
# Save Results
# ------------------------------
out_dir = os.path.dirname(args.output_file)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(sts_scores, f, indent=4, ensure_ascii=False)

# ------------------------------
# Cleanup
# ------------------------------
del scorer, use_model, labse_preprocessor, labse_encoder, laser_encoder
gc.collect()
print(f"\nDone. Wrote {len(sts_scores)} records to {args.output_file}")
#the sts virtual env path: /media/data_dump/Ritwik/envs/sts
