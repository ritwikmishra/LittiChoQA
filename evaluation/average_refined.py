import os
import json
import numpy as np

def compute_average_scores(model_folders):
    """
    For each model folder, computes average metrics from JSON files in refine_aggregated/
    and writes a single averaged JSON file per input.
    """

    base_dir = "rouge_evaluation"
    input_dir_name = "refined"

    file_map = {
        "refine_long.json": "average_long.json",
        "refine_short_l6.json": "average_short_l6.json",
        "refine_short_l6v2.json": "average_short_l6v2.json"
    }

    for model_path in model_folders:
        model_dir = os.path.join(base_dir, model_path)
        input_dir = os.path.join(model_dir, input_dir_name)

        if not os.path.isdir(input_dir):
            print(f"⚠️ Missing folder: {input_dir}")
            continue

        print(f"\n📂 Processing model: {model_dir}")

        for in_file, out_file in file_map.items():
            fpath = os.path.join(input_dir, in_file)
            if not os.path.exists(fpath):
                print(f"  ⚠️ Missing file: {fpath}")
                continue

            with open(fpath, "r") as f:
                data = json.load(f)

            if not data:
                print(f"  ⚠️ Empty file: {fpath}")
                continue

            # Collect numeric fields dynamically
            numeric_fields = {}
            for item in data:
                for key, value in item.items():
                    if key == "qid":
                        continue
                    if isinstance(value, (int, float)):
                        numeric_fields.setdefault(key, []).append(value)

            # Compute averages
            averages = {
                f"average_{key}": float(np.mean(values))
                for key, values in numeric_fields.items()
            }

            # Write output
            output_path = os.path.join(input_dir, out_file)
            with open(output_path, "w") as out_f:
                json.dump(averages, out_f, indent=4)

            print(f"  💾 Created: {output_path}")

if __name__ == "__main__":
    # ✅ Specify your model folders here
    MODELS = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "krutrim-ai-labs/Krutrim-2-instruct",
        "sarvamai/sarvam-1",
        "CohereLabs/aya-23-8B",
        "sarvamai/OpenHathi-7B-Hi-v0.1-Base",
        "Qwen/Qwen2.5-7B-Instruct"
    ]

    compute_average_scores(MODELS)
