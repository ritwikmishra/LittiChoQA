import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"   # tokenizer only
DATA_PATH = "/media/data_dump/aarya220007/combined_all_response.json"
OUT_INTERMEDIATE = "/media/data_dump/aarya220007/data/lang_dataset.jsonl"
OUT_GRAPH_DIR = "/media/data_dump/aarya220007/graphs"
os.makedirs(OUT_GRAPH_DIR, exist_ok=True)

# ----------------------------
# Load tokenizer
# ----------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ----------------------------
# Load already processed QIDs
# ----------------------------
seen_qids = set()
if os.path.exists(OUT_INTERMEDIATE):
    print("Loading existing processed QIDs...")
    with open(OUT_INTERMEDIATE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading existing data"):
            row = json.loads(line)
            seen_qids.add(row["qid"])
    print(f"Already processed {len(seen_qids)} QIDs")

# ----------------------------
# Process dataset incrementally
# ----------------------------
print("Processing dataset and tokenizing...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

total_non_factoid = sum(len(entry["qas"].get("non-factoid", [])) for entry in dataset.values())
print(f"Total non-factoid QA samples: {total_non_factoid}")


with open(OUT_INTERMEDIATE, "a", encoding="utf-8") as outf:
    for key, entry in tqdm(dataset.items(), desc="Tokenizing entries"):
        story = entry.get("story", "")
        lang = entry.get("lang", "unknown")

        for qatype in ["non-factoid"]:
            for qa in entry["qas"].get(qatype, []):
                qid = qa.get("id", "")
                if qid in seen_qids:
                    continue  # skip already processed

                question = qa.get("question", "")

                prompt = (
                    "Read the story and answer the question.\n"
                    f"## Story:\n{story}\n"
                    f"## Question:\n{question}\n"
                    f"## Answer:\n"
                )

                tokens = tokenizer(prompt, truncation=False, padding=False)
                tokenized_len = len(tokens["input_ids"])

                row = {
                    "qid": qid,
                    "prompt_length": len(prompt),
                    "tokenized_length": tokenized_len,
                    "lang": lang
                }

                outf.write(json.dumps(row, ensure_ascii=False) + "\n")
                outf.flush()  # write immediately
                seen_qids.add(qid)

print(f"Processed examples → saved to {OUT_INTERMEDIATE}")

# ----------------------------
# Generate graphs from existing JSONL
# ----------------------------
print("Generating graphs...")
processed = []
with open(OUT_INTERMEDIATE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading intermediate data"):
        processed.append(json.loads(line))

thresholds = list(range(1000, 20001, 1000))

for thresh in tqdm(thresholds, desc="Generating graphs"):
    lang_counts = defaultdict(int)
    for row in processed:
        if row["tokenized_length"] <= thresh:
            lang_counts[row["lang"]] += 1

    if not lang_counts:
        continue

    langs = list(lang_counts.keys())
    counts = [lang_counts[l] for l in langs]

    plt.figure(figsize=(12, 7))
    plt.bar(langs, counts)
    plt.xlabel("Languages")
    plt.ylabel("Number of examples")
    plt.title(f"Distribution of languages (tokenized length ≤ {thresh})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = os.path.join(OUT_GRAPH_DIR, f"lang_dist_leq_{thresh}.png")
    plt.savefig(out_path)
    plt.close()

    print(f"Saved: {out_path}")
