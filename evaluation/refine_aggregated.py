import os
import json
from typing import Dict, List, Tuple, Set

# ---------------------------
# Configuration (edit paths)
# ---------------------------
BASE_DIR = "sts_evaluation"   # base directory where model folders live
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "krutrim-ai-labs/Krutrim-2-instruct",
    "sarvamai/sarvam-1",
    "CohereLabs/aya-23-8B",
    "sarvamai/OpenHathi-7B-Hi-v0.1-Base",
    "Qwen/Qwen2.5-7B-Instruct"
]

# aggregated file patterns (input -> output suffix)
FILE_TYPES = {
    "long": ("aggregated_batches/aggregated_long.json", "refine_aggregated/aggregated_long_refined.json"),
    "short_l6": ("aggregated_batches/aggregated_short_l6.json", "refine_aggregated/aggregated_short_l6_refined.json"),
    "short_l6v2": ("aggregated_batches/aggregated_short_l6v2.json", "refine_aggregated/aggregated_short_l6v2_refined.json"),
}
# ---------------------------


def load_aggregated_file(full_path: str) -> Tuple[List[Dict], Dict[str, Dict], Set[str]]:
    """
    Load aggregated_{type}.json file and return:
      - original_list (as loaded)
      - qid_to_entry mapping (qid -> full dict)
      - set of qids
    If file is missing or invalid, returns ([], {}, set()).
    """
    if not os.path.exists(full_path):
        return [], {}, set()

    try:
        with open(full_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ❌ Error reading '{full_path}': {e}")
        return [], {}, set()

    if not isinstance(data, list):
        print(f"  ⚠️ File '{full_path}' does not contain a JSON list; skipping.")
        return [], {}, set()

    qid_to_entry = {}
    qids = set()
    # keep entries as-is, but build mapping
    for item in data:
        if not isinstance(item, dict):
            continue
        qid = item.get("qid")
        if not qid:
            continue
        qid_to_entry[qid] = item
        qids.add(qid)

    return data, qid_to_entry, qids


def refine_for_all_models(models: List[str], base_dir: str = BASE_DIR):
    """
    For each FILE_TYPES entry:
      - find common qids across all models that have the input file
      - for each model that had the input file, write the refined output
        containing only entries whose qid is in the common set
    """
    for type_key, (in_fname, out_fname) in FILE_TYPES.items():
        print(f"\n=== Processing '{in_fname}' -> '{out_fname}' for all models ===")

        per_model_original_list: Dict[str, List[Dict]] = {}
        per_model_qid_to_entry: Dict[str, Dict[str, Dict]] = {}
        per_model_qid_sets: Dict[str, Set[str]] = {}

        # Load all models that have the file
        for model_rel in models:
            model_path = os.path.join(base_dir, model_rel)
            if not os.path.isdir(model_path):
                print(f"❌ Skipping invalid model directory: {model_path}")
                continue

            in_path = os.path.join(model_path, in_fname)
            original_list, qid_to_entry, qids = load_aggregated_file(in_path)
            if not qids:
                print(f"  ⚠️ Model '{model_rel}' does not have valid '{in_fname}' (skipping for this file).")
                continue

            per_model_original_list[model_rel] = original_list
            per_model_qid_to_entry[model_rel] = qid_to_entry
            per_model_qid_sets[model_rel] = qids
            print(f"  ✔ Loaded '{in_path}' ({len(qids)} qids)")

        if not per_model_qid_sets:
            print(f"⚠️ No models provided a valid '{in_fname}'. Skipping this type.")
            continue

        # Compute intersection of qids across all models that had the file
        sets_list = list(per_model_qid_sets.values())
        common_qids = set.intersection(*sets_list)
        print(f"\n  ✅ Found {len(common_qids)} qids common across {len(sets_list)} models for '{in_fname}'.")

        # For each model that had the input file, create refined output (preserve ordering)
        for model_rel, original_list in per_model_original_list.items():
            model_path = os.path.join(base_dir, model_rel)
            output_dir = os.path.join(model_path, "refine_aggregated")
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(model_path, out_fname)

            # Preserve original ordering: iterate original_list and keep entries whose qid in common_qids
            refined_list = []
            for entry in original_list:
                if not isinstance(entry, dict):
                    continue
                qid = entry.get("qid")
                if qid and qid in common_qids:
                    refined_list.append(entry)

            # Write refined file
            try:
                with open(out_path, "w") as f:
                    json.dump(refined_list, f, indent=4)
                print(f"  ✅ Wrote '{out_path}' ({len(refined_list)} entries).")
            except Exception as e:
                print(f"  ❌ Failed to write '{out_path}': {e}")


if __name__ == "__main__":
    refine_for_all_models(MODELS, base_dir=BASE_DIR)
