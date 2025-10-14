import os
import re
import json
import pandas as pd
from collections import defaultdict, Counter
import ast
# ----------------------------
# CONFIGURATION
# ----------------------------
LANGUAGES = ["angika", "assamese", "awadhi", "bagheli", "bhojpuri", "braj",  "bundeli", "dogri", "hindi", "konkani", "magahi", "maithili", "nepali", "pali", "telugu", "urdu" ]

ANSWERS_DIR = "excels/short/answers"
RANKINGS_DIR = "llm_outputs_short"
OUTPUT_FILE = "model_rank_summary_short.xlsx"

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

# def parse_answer_blocks(filepath):
#     # for long
#     """Parse the answers file into blocks keyed by (workbook, sheet)."""
#     with open(filepath, "r", encoding="utf-8") as f:
#         content = f.read()

#     # Split on workbook sections
#     blocks = re.split(r"=== Workbook:\s*", content)
    
#     parsed = {}

#     for block in blocks:
#         if not block.strip():
#             continue
#         # Extract workbook and sheet names
#         header_match = re.match(r"([^|]+)\s*\|\s*Sheet:\s*([^\n=]+)", block)
#         if not header_match:
#             continue
#         workbook, sheet = header_match.groups()
#         key = (workbook.strip(), sheet.strip())
#         # Extract Answer_X : model_name pairs
#         answers = dict(re.findall(r"(Answer_\d+)\s*:\s*(.+)", block))
#         parsed[key] = {a.strip(): m.strip() for a, m in answers.items()}
#     return parsed

def parse_answer_blocks(filepath):
    # for short
    """Parse the answers file into blocks keyed by (workbook, sheet)."""
    parsed = {}

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract workbook name (assume first line starts with 'Workbook:')
    workbook_match = re.search(r"Workbook:\s*(.+)", content)
    if not workbook_match:
        raise ValueError("Workbook name not found in the file.")
    workbook = workbook_match.group(1).strip()

    # Split content by sheet blocks using the separator
    sheet_blocks = re.split(r"-{10,}", content)

    for block in sheet_blocks:
        block = block.strip()
        if not block:
            continue

        # Extract sheet name
        sheet_match = re.search(r"Sheet:\s*(.+)", block)
        if not sheet_match:
            continue
        sheet = sheet_match.group(1).strip()

        # Extract all Answer_X: model_name pairs
        answers = dict(re.findall(r"(Answer_\d+)\s*:\s*(.+)", block))
        answers = {a.strip(): m.strip() for a, m in answers.items()}

        # Store keyed by (workbook, sheet)
        parsed[(workbook, sheet)] = answers

    return parsed


def parse_ranking_blocks(filepath):
    """Parse the ranking file into blocks keyed by (workbook, sheet)."""
    parsed = {}

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split blocks by the separator
    blocks = content.split("="*80)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract workbook and sheet
        header_match = re.match(r"=== Workbook:\s*([^|]+)\s*\|\s*Sheet:\s*([^\n=]+) ===", block)
        if not header_match:
            continue
        workbook, sheet = header_match.groups()
        key = (workbook.strip(), sheet.strip())

        # Extract JSON content after the header
        json_start = block.find("{")
        if json_start == -1:
            continue
        json_text = block[json_start:]

        # Fix numeric keys in Rankings to be valid JSON
        json_text = re.sub(r'(\s*)(\d+)\s*:', r'\1"\2":', json_text)

        data = json.loads(json_text)
        rankings = data.get("Rankings", {})
        parsed[key] = rankings

    return parsed


def aggregate_model_rank_counts(answer_blocks, ranking_blocks):
    """Combine the answers + rankings and count how many times each model got each rank."""
    rank_counts = defaultdict(Counter)
    num_stories = 0

    for key in ranking_blocks:
        if key not in answer_blocks:
            continue
        num_stories += 1
        answer_map = answer_blocks[key]
        rankings = ranking_blocks[key]

        for rank, ans_id in rankings.items():
            model = answer_map.get(ans_id)
            if model:
                rank_counts[int(rank)][model] += 1
    return rank_counts, num_stories


def get_top_models(rank_counts, num_stories):
    """Get the model with highest count for each rank (formatted string)."""
    result = {}
    for rank in range(1, 7):
        counts = rank_counts.get(rank, {})
        if not counts:
            result[rank] = "_"
            continue
        top_model, top_count = max(counts.items(), key=lambda x: x[1])
        result[rank] = f"{top_model}({top_count}/{num_stories})"
    return result


# ----------------------------
# MAIN LOGIC
# ----------------------------

summary_data = []

for lang in LANGUAGES:
    answers_path = os.path.join(ANSWERS_DIR, f"{lang}.txt")
    rankings_path = os.path.join(RANKINGS_DIR, f"{lang}.txt")

    if not os.path.exists(answers_path) or not os.path.exists(rankings_path):
        print(f"⚠️ Missing files for {lang}. Filling with underscores.")
        summary_data.append({
            "Language": lang, **{f"Rank {i}": "_" for i in range(1, 7)}
        })
        continue

    answer_blocks = parse_answer_blocks(answers_path)
    ranking_blocks = parse_ranking_blocks(rankings_path)

    if not answer_blocks or not ranking_blocks:
        print(f"⚠️ Incomplete data for {lang}. Filling with underscores.")
        summary_data.append({
            "Language": lang, **{f"Rank {i}": "_" for i in range(1, 7)}
        })
        continue

    rank_counts, num_stories = aggregate_model_rank_counts(answer_blocks, ranking_blocks)
    top_models = get_top_models(rank_counts, num_stories)

    row = {"Language": lang}
    for i in range(1, 7):
        row[f"Rank {i}"] = top_models.get(i, "_")
    summary_data.append(row)

# ----------------------------
# CREATE EXCEL TABLE
# ----------------------------

df = pd.DataFrame(summary_data)
df.to_excel(OUTPUT_FILE, index=False)
print(f"✅ Excel file created: {OUTPUT_FILE}")
