import os
import json
import re
from tqdm import tqdm

def shorten_context(input_jsonl: str, lc_dir: str, output_jsonl: str) -> None:
    assert os.path.exists(input_jsonl), f"Input file not found: {input_jsonl}"
    assert os.path.isdir(lc_dir), f"LC directory not found: {lc_dir}"

    entries = []
    with open(input_jsonl, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc=f"Reading {os.path.basename(input_jsonl)}", unit="line"):
            entries.append(json.loads(line.strip()))

    assert len(entries) > 0, f"No entries found in {input_jsonl}"

    processed_entries = []

    for entry in tqdm(entries, desc=f"Processing {os.path.basename(input_jsonl)}", unit="entry"):
        qid = entry.get("qid")
        prompt = entry.get("prompt")
        answer = entry.get("completion")

        question_match = re.search(r"## Question: \n (.*?) ## Answer:", prompt, re.DOTALL)

        if not question_match or not answer:
            print(f"Question/answer not matched for qid: {qid}")
            continue

        question = question_match.group(1).strip()

        sc_file = os.path.join(lc_dir, f"{qid}.json")
        if not os.path.exists(sc_file):
            print(f"Shorten context file not found for qid: {qid}")
            continue

        with open(sc_file, 'r', encoding='utf-8') as sf:
            sc_data = json.load(sf)

        sc = sc_data["qid"]
        assert isinstance(sc, str) and len(sc) > 0, f"Invalid shorten context for qid: {qid}"
        
        new_prompt = (
            "Read the story and answer the question. "
            f"## Story: \n {sc} "
            f"## Question: \n {question} "
            f"## Answer: \n "
        )
        completion = (
            f"{answer}"
        )

        processed_entries.append({"qid": qid, "prompt": new_prompt, "completion": completion})

    processed_entries.sort(key=lambda x: len(x["prompt"]))
    with open(output_jsonl, 'w', encoding='utf-8') as outfile:
        for entry in tqdm(processed_entries, desc=f"Writing {os.path.basename(output_jsonl)}", unit="entry"):
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")


lc_dir = "/media/data_dump/aarya220007/lc"
input_base = "/media/data_dump/aarya220007/data/fc_dataset"
output_base = "/media/data_dump/aarya220007/data/sc_l6v2_dataset"
# input_base = "/media/data_dump/aarya220007/data_test/fc_dataset"
# output_base = "/media/data_dump/aarya220007/data_test/sc_l6v2_dataset"

os.makedirs(output_base, exist_ok=True)

datasets = ["train", "test", "val"]
for ds in datasets:
    input_file = os.path.join(input_base, f"{ds}.jsonl")
    output_file = os.path.join(output_base, f"{ds}.jsonl")
    shorten_context(input_file, lc_dir, output_file)
