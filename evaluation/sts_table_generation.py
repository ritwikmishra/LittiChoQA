import os
import json
import pandas as pd

# ------------------- Configuration -------------------
BASE_DIR = "rouge_evaluation"

MODELS = [
    "sarvamai/sarvam-1",
    "CohereLabs/aya-23-8B",
    "sarvamai/OpenHathi-7B-Hi-v0.1-Base",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "krutrim-ai-labs/Krutrim-2-instruct"
]

# Map JSON filenames to readable labels
FILE_MAP = {
    "average_long.json": "Long",
    "average_short_l6.json": "Short_l6",
    "average_short_l6v2.json": "Short_l6v2"
}

# Metrics to include
METRICS = [
    "average_R1",
    "average_R2",
    "average_R3",
    "average_RL"
]

# Output Markdown file
OUTPUT_FILE = os.path.join(BASE_DIR, "rouge_absolute.md")
# ------------------------------------------------------


def collect_results():
    """Collects all metric results for each model and file."""
    all_results = []

    for model in MODELS:
        model_name = model.split("/")[-1]
        for file_name, file_label in FILE_MAP.items():
            file_path = os.path.join(BASE_DIR, model, "refine_aggregated", file_name)

            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Add model and file info
                data["Model"] = model_name
                data["File"] = file_label
                all_results.append(data)
            else:
                print(f"⚠️ Missing file: {file_path}")

    return pd.DataFrame(all_results)


def generate_metric_tables(df):
    """Generates separate Markdown tables per metric."""
    markdown_content = "# 🧾 Rouge Evaluation Results Summary\n\n"
    markdown_content += "This report compares all models across the rouge evaluation metrics.\n"
    markdown_content += "---\n"

    for metric in METRICS:
        markdown_content += f"\n## 📊 {metric}\n\n"
        pivot_df = df.pivot(index="Model", columns="File", values=metric)
        markdown_content += pivot_df.to_markdown() + "\n\n"

    return markdown_content


def main():
    df = collect_results()
    if df.empty:
        print("❌ No data found. Check your paths or JSON files.")
        return

    markdown = generate_metric_tables(df)

    # Save markdown to file
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(markdown)

    print(f"✅ Markdown summary saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
