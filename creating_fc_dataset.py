import json
import os
import random
from tqdm import tqdm

def prepare_ft_dataset(input_path: str, output_dir: str) -> None:
    assert os.path.exists(input_path), f"Input file not found: {input_path}"

    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    assert isinstance(dataset, dict), "Dataset must be a dictionary"

    entries = []
    total_qas = sum(len(story_data["qas"]["non-factoid"]) for story_data in dataset.values())
    assert total_qas > 0, "No non-factoid QAs found in dataset"

    qa_counter = 100  # added

    with tqdm(total=total_qas, desc="Processing non-factoid entries", unit="qa") as pbar:
        for story_key, story_data in dataset.items():
            for qa in story_data["qas"]["non-factoid"]:
                if qa_counter == 0:
                    break
                qid = qa.get("id", "")
                question = qa.get("question", "")
                answer = qa.get("answer", "")

                story = story_data.get('story', "")

                assert qid, "Missing QID in QA pair"

                prompt = (
                    "Read the story and answer the question. "
                    f"## Story: \n {story} "
                    f"## Question: \n {question} "
                    f"## Answer: \n "
                )

                completion = f"{answer}"

                entries.append({"qid": qid, "prompt": prompt, "completion": completion})
                qa_counter -= 1  # added
                pbar.update(1)

    random.shuffle(entries)

    # Train/Val/Test split ratios
    train_split = 0.7
    val_split = 0.1
    test_split = 0.2

    n_total = len(entries)
    assert n_total > 0, "No entries to split"

    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)
    n_test = n_total - n_train - n_val

    splits = {
        "train": entries[:n_train],
        "val": entries[n_train:n_train + n_val],
        "test": entries[n_train + n_val:]
    }

    # ✅ Sort each split by prompt length
    for split_name, split_entries in splits.items():
        split_entries.sort(key=lambda x: len(x["prompt"])) 

        output_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with tqdm(total=len(split_entries), desc=f"Writing {split_name} entries", unit="qa") as pbar:
            with open(output_path, "w", encoding="utf-8") as f:
                for entry in split_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    pbar.update(1)

        print(f"{split_name.capitalize()} JSONL created at: {output_path} ({len(split_entries)} entries)")

    print(f"Total non-factoid entries: {len(entries)})")


input_path = "/media/data_dump/aarya220007/finetuning_proj/combined_all_response.json"
# output_dir = "/media/data_dump/aarya220007/data/fc_dataset"
output_dir = "/media/data_dump/aarya220007/data_test/fc_dataset"
prepare_ft_dataset(input_path, output_dir)
