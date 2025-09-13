import os
import json
import re
from tqdm import tqdm

def shorten_context(input_jsonl: str, munfquad_dir: str, output_jsonl: str) -> None:
    assert os.path.exists(input_jsonl), f"Input file not found: {input_jsonl}"
    assert os.path.isdir(munfquad_dir), f"Munfquad directory not found: {munfquad_dir}"

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

        story_match = re.search(r"## Story: \n (.*?) ## Question:", prompt, re.DOTALL)
        question_match = re.search(r"## Question: \n (.*?) ## Answer:", prompt, re.DOTALL)

        if not story_match or not question_match or not answer:
            print(f"Story/question/answer not matched for qid: {qid}")
            continue

        story = story_match.group(1)
        question = question_match.group(1).strip()
        story_lines = story.split('\n')

        score_file = os.path.join(munfquad_dir, f"{qid}.json")
        if not os.path.exists(score_file):
            print(f"Score file not found for qid: {qid}")
            continue

        with open(score_file, 'r', encoding='utf-8') as sf:
            scores_data = json.load(sf)

        scores = scores_data["qid"]
        assert isinstance(scores, list), f"Scores must be a list for {qid}"
        assert len(scores) == len(story_lines), f"Mismatch between scores and story lines for {qid}, scores len: {len(scores)}, story lines len: {len(story_lines)}"

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        top_indices.sort()
        short_story = "\n".join([story_lines[i] for i in top_indices])

        new_prompt = (
            "Read the story and answer the question. "
            f"## Story: \n {short_story} "
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


munfquad_dir = "/media/data_dump/aarya220007/munfquad"
input_base = "/media/data_dump/aarya220007/data/fc_dataset"
output_base = "/media/data_dump/aarya220007/data/sc_l6_dataset"
# input_base = "/media/data_dump/aarya220007/data_test/fc_dataset"
# output_base = "/media/data_dump/aarya220007/data_test/sc_l6_dataset"

os.makedirs(output_base, exist_ok=True)

datasets = ["train", "test", "val"]
for ds in datasets:
    input_file = os.path.join(input_base, f"{ds}.jsonl")
    output_file = os.path.join(output_base, f"{ds}.jsonl")
    shorten_context(input_file, munfquad_dir, output_file)
