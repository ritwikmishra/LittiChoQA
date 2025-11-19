import os
import json

def refine_common_qids(model_folders):
    """
    For each model folder:
    - Loads three JSON files from refine_aggregated/
      (aggregated_long_refined.json, aggregated_short_l6_refined.json, aggregated_short_l6v2_refined.json)
    - Finds qids common across all three
    - Writes them into refined/ folder as:
        refine_long.json
        refine_short_l6.json
        refine_short_l6v2.json
    """
    
    base_dir = "base_sts_evaluation"
    input_dir_name = "refine_aggregated"
    output_dir_name = "refined"

    file_map = {
        "aggregated_long_refined.json": "refine_long.json",
        "aggregated_short_l6_refined.json": "refine_short_l6.json",
        "aggregated_short_l6v2_refined.json": "refine_short_l6v2.json"
    }

    for model_path in model_folders:
        model_dir = os.path.join(base_dir, model_path)
        input_dir = os.path.join(model_dir, input_dir_name)
        output_dir = os.path.join(model_dir, output_dir_name)

        if not os.path.isdir(input_dir):
            print(f"⚠️ Missing folder: {input_dir}")
            continue

        os.makedirs(output_dir, exist_ok=True)
        print(f"\n📂 Processing model: {model_dir}")

        data_map = {}
        qid_sets = []

        # Load all three refined input files
        for fname in file_map.keys():
            fpath = os.path.join(input_dir, fname)
            if not os.path.exists(fpath):
                print(f"  ⚠️ Missing file: {fpath}")
                break

            with open(fpath, "r") as f:
                data = json.load(f)
                data_map[fname] = data
                qid_sets.append({item["qid"] for item in data})

        # Skip if any file missing
        if len(qid_sets) != len(file_map):
            print("  ⚠️ Skipping (some files missing)")
            continue

        # Find common qids among all three files
        common_qids = set.intersection(*qid_sets)
        print(f"  ✅ Common QIDs found: {len(common_qids)}")

        # Write refined files with only common qids
        for in_file, out_file in file_map.items():
            filtered_data = [item for item in data_map[in_file] if item["qid"] in common_qids]
            out_path = os.path.join(output_dir, out_file)
            with open(out_path, "w") as out_f:
                json.dump(filtered_data, out_f, indent=4)
            print(f"  💾 Created: {out_path} ({len(filtered_data)} entries)")

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

    refine_common_qids(MODELS)
