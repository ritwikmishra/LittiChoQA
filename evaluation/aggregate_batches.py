import os
import json
from glob import glob
import re


def aggregate_json_files(model_folders, base_dir="sts_evaluation"):
    """
    Aggregates all batch_*.json files inside each model's subfolders:
    - long
    - short_l6
    - short_l6v2
    Creates aggregated_long.json, aggregated_short_l6.json, aggregated_short_l6v2.json
    inside the model's root folder.
    """

    for model_rel_path in model_folders:
        model_path = os.path.join(base_dir, model_rel_path)
        if not os.path.isdir(model_path):
            print(f"❌ Skipping invalid model folder: {model_path}")
            continue

        print(f"\n🔹 Processing model: {model_rel_path}")

        subfolders = ["long", "short_l6", "short_l6v2"]

        for sub in subfolders:
            sub_path = os.path.join(model_path, sub)
            if not os.path.isdir(sub_path):
                print(f"  ⚠️ Skipping missing folder: {sub_path}")
                continue

            # Find and sort batch_*.json files numerically
            batch_files = sorted(
                glob(os.path.join(sub_path, "batch_*.json")),
                key=lambda x: int(re.search(r'batch_(\d+)\.json', os.path.basename(x)).group(1))
                if re.search(r'batch_(\d+)\.json', os.path.basename(x)) else 0
            )

            if not batch_files:
                print(f"  ⚠️ No batch files found in: {sub_path}")
                continue

            aggregated_data = []
            for bf in batch_files:
                try:
                    with open(bf, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            aggregated_data.extend(data)
                        else:
                            print(f"  ⚠️ Skipped non-list file: {bf}")
                except Exception as e:
                    print(f"  ❌ Error reading {bf}: {e}")

            # Write aggregated data
            output_dir = os.path.join(model_path, "aggregated_batches")
            os.makedirs(output_dir, exist_ok=True)

            # Write aggregated data
            output_file = os.path.join(output_dir, f"aggregated_{sub}.json")
            with open(output_file, "w") as out_f:
                json.dump(aggregated_data, out_f, indent=4)

            print(f"  ✅ Created: {output_file} ({len(aggregated_data)} entries)")


if __name__ == "__main__":
    # ✅ Specify relative model paths (relative to "base_sts_evaluation")
    MODELS = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "krutrim-ai-labs/Krutrim-2-instruct",
        "sarvamai/sarvam-1",
        "CohereLabs/aya-23-8B",
        "sarvamai/OpenHathi-7B-Hi-v0.1-Base",
        "Qwen/Qwen2.5-7B-Instruct"
    ]

    aggregate_json_files(MODELS)
