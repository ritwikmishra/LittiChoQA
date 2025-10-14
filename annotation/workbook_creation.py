import json
import os
import pandas as pd
import random

# -------------------------------
# CONFIGURATION
# -------------------------------
LANG_DATASET_PATH = "../finetuning/data/lang_dataset.jsonl"
TEST_DATASET_PATH = "../finetuning/data/fc_dataset/test.jsonl"
OUTPUT_JSONL_PATH = "intersection_output.jsonl"
OUTPUT_EXCEL_DIR = "excels/long"
MODEL_OUTPUTS_DIR = "../inferencing/outputs"

MODEL_CONFIGS = {
    "meta-llama/Llama-3.1-8B-Instruct": 2,
    "CohereLabs/aya-23-8B": 2,
    "sarvamai/sarvam-1": 2,
    "sarvamai/OpenHathi-7B-Hi-v0.1-Base": 2,
    "krutrim-ai-labs/Krutrim-2-instruct": 2,
    "Qwen2.5-7B-Instruct": 2
}

TARGET_WORD_COUNT = 500
WORD_MARGIN = 200

MIN_WORD_LIMITS = {
    "urdu": 1,
    "hindi": 0,
    "angika": 49,
    "bhojpuri": 46,
    "awadhi": 96,
    "maithili": 304,
    "telugu": 4424,
    "konkani": 1100,
    "bundeli": 120,
    "nepali": 285,
    "braj": 258,
    "dogri": 180,
    "pali": 77,
    "dzongkha": 235,
    "magahi": 228,
    "bagheli": 428,
    "assamese": 771
}

# -------------------------------
# CACHES & HELPERS
# -------------------------------
_BATCH_FILE_CACHE = {}
used_stories_long = set()  # <---- NEW: store all long-context stories here


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def extract_story_and_question(prompt):
    story_part, question_part = "", ""
    if "## Story:" in prompt and "## Question:" in prompt:
        parts = prompt.split("## Story:")
        rest = parts[1].split("## Question:")
        story_part = rest[0].strip()
        question_part = rest[1].split("## Answer:")[0].strip() if "## Answer:" in rest[1] else rest[1].strip()
    return story_part, question_part


def split_blocks(content, batch_size):
    if batch_size == 1:
        return [content.strip()]
    sep = "-----\nQID:"
    if sep in content:
        sep_index = content.find(sep)
        first_block = content[:sep_index + len(sep)].strip()
        second_block = "QID:" + content[sep_index + len(sep):].strip()
        return [first_block, second_block]
    return [content.strip()]


def _read_batch_file_once(batch_file):
    if batch_file in _BATCH_FILE_CACHE:
        return _BATCH_FILE_CACHE[batch_file]
    if not os.path.exists(batch_file):
        _BATCH_FILE_CACHE[batch_file] = None
        return None
    try:
        with open(batch_file, "r", encoding="utf-8") as f:
            content = f.read()
        _BATCH_FILE_CACHE[batch_file] = content
        return content
    except Exception as e:
        print(f"[ERROR] Could not read {batch_file}: {e}")
        _BATCH_FILE_CACHE[batch_file] = None
        return None


def load_model_answers(model_name, context_or_rownum, row_number_or_qid, qid=None):
    """Unified loader for both long and short contexts."""
    if isinstance(context_or_rownum, str):
        context = context_or_rownum
        row_number = row_number_or_qid
        batch_size = MODEL_CONFIGS.get(model_name, 2)
        model_dir = os.path.join(MODEL_OUTPUTS_DIR, model_name)
        batch_id = row_number // batch_size
        batch_file = os.path.join(model_dir, context, f"batch_{batch_id}.txt")
    else:
        context = "long"
        row_number = context_or_rownum
        qid = row_number_or_qid
        batch_size = MODEL_CONFIGS.get(model_name, 2)
        model_dir = os.path.join(MODEL_OUTPUTS_DIR, model_name)
        batch_id = row_number // batch_size
        batch_file = os.path.join(model_dir, context, f"batch_{batch_id}.txt")

    content = _read_batch_file_once(batch_file)
    if not content:
        return None

    try:
        blocks = split_blocks(content, MODEL_CONFIGS.get(model_name, 2))
        for block in blocks:
            if qid in block and "Generated Answer:" in block:
                gen_answer = block.split("Generated Answer:")[1].strip()
                if "-----" in gen_answer:
                    gen_answer = gen_answer.split("-----")[0].strip()
                return gen_answer
        return None
    except Exception as e:
        print(f"Error parsing {batch_file}: {e}")
        return None


def has_all_model_answers(row, row_number):
    for model_name in MODEL_CONFIGS.keys():
        if not load_model_answers(model_name, row_number, row["qid"]):
            return False
    return True


def model_has_any_answer_for_lang_long(model_name, lang_df, global_df):
    for _, row in lang_df.iterrows():
        row_number_list = global_df.index[global_df["qid"] == row["qid"]].tolist()
        if not row_number_list:
            continue
        row_number = row_number_list[0]
        if load_model_answers(model_name, row_number, row["qid"]):
            return True
    return False


def model_has_any_answer_for_lang_short(model_name, context, lang_df, context_test_data):
    for _, row in lang_df.iterrows():
        row_number_list = [i for i, r in enumerate(context_test_data) if r["qid"] == row["qid"]]
        if not row_number_list:
            continue
        row_number = row_number_list[0]
        if load_model_answers(model_name, context, row_number, row["qid"]):
            return True
    return False


# -------------------------------
# LOAD OR BUILD INTERSECTION
# -------------------------------
if os.path.exists(OUTPUT_JSONL_PATH):
    with open(OUTPUT_JSONL_PATH, "r", encoding="utf-8") as f:
        intersection_rows = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"[INFO] Loaded existing intersection: {len(intersection_rows)} rows.")
else:
    lang_data = load_jsonl(LANG_DATASET_PATH)
    test_data = load_jsonl(TEST_DATASET_PATH)
    lang_dict = {entry["qid"]: entry for entry in lang_data}
    intersection_rows = []
    for test_entry in test_data:
        qid = test_entry["qid"]
        if qid not in lang_dict:
            continue
        lang_entry = lang_dict[qid]
        lang = lang_entry["lang"]
        story, question = extract_story_and_question(test_entry["prompt"])
        intersection_rows.append({
            "qid": qid,
            "story": story,
            "question": question,
            "answer": test_entry["completion"],
            "no_of_words": len(story.split()),
            "lang": lang
        })
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for row in intersection_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[INFO] Created intersection file with {len(intersection_rows)} rows.")


# ------------------------------- # MAIN: per-language processing for long context # ------------------------------- 
df = pd.DataFrame(intersection_rows)
os.makedirs(OUTPUT_EXCEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_EXCEL_DIR + '/answers', exist_ok=True)
used_stories_long = set()
for lang in df["lang"].unique():
    lang_df = df[df["lang"] == lang].reset_index(drop=True)
    min_limit = MIN_WORD_LIMITS.get(lang, TARGET_WORD_COUNT)
    print(f"\nProcessing language: '{lang}' (min_limit={min_limit})")

    # Determine selection range:
    if lang_df.empty:
        print(f"⚠️ No data rows for language '{lang}' — skipping.")
        continue

    if lang_df["no_of_words"].min() < TARGET_WORD_COUNT:
        lower, upper = TARGET_WORD_COUNT - WORD_MARGIN, TARGET_WORD_COUNT + WORD_MARGIN
    else:
        # all stories are > 500 words — choose around the min_limit
        lower, upper = min_limit, min_limit + WORD_MARGIN

    filtered_df = lang_df[(lang_df["no_of_words"] >= lower) & (lang_df["no_of_words"] <= upper)].reset_index(drop=True)
    print(f" - Candidate rows in range [{lower}, {upper}]: {len(filtered_df)}")

    # Check whether any model never produced any answer for this language (across all rows of this language)
    missing_models = []
    for model_name in MODEL_CONFIGS.keys():
        has_any = model_has_any_answer_for_lang_long(model_name, lang_df, df)
        print(f" Model {model_name} → has_any_answer_in_lang? {'YES' if has_any else 'NO'}")
        if not has_any:
            missing_models.append(model_name)

    excel_path = os.path.join(OUTPUT_EXCEL_DIR, f"{lang}_stories.xlsx")

    if missing_models:
        # Create a blank workbook (Info sheet) as requested when ANY model has zero outputs for this language
        print(f"⚠️ Some models have ZERO outputs for '{lang}'. Creating blank workbook with info: {missing_models}")
        info_rows = [
            {"Field": "Note", "Value": "One or more models produced NO answers for this language. Workbook intentionally left without story sheets."},
            {"Field": "Missing models", "Value": ", ".join(missing_models)}
        ]
        info_df = pd.DataFrame(info_rows)
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            info_df.to_excel(writer, sheet_name="Info", index=False)
        print(f"Blank workbook written: {excel_path}")
        # move to next language
        continue

    # Now find story-question pairs where ALL models produced answers for that pair
    valid_rows = []
    seen_stories = set()
    for idx, row in filtered_df.iterrows():
        # find row_number in the global df (intersection order) — required by load_model_answers
        row_number_list = df.index[df["qid"] == row["qid"]].tolist()
        if not row_number_list:
            continue
        row_number = row_number_list[0]
        story_text = row["story"].strip()
        if story_text in seen_stories:
            continue
        if has_all_model_answers(row, row_number):
            valid_rows.append((row, row_number))
        else:
            # skip this pair (at least one model missing)
            pass
        # Stop early if we already have 3 (no need to check more)
        seen_stories.add(story_text)
        if len(valid_rows) >= 3:
            break

    if len(valid_rows) == 0:
        print(f"⚠️ No valid data found for language: {lang}. Skipping workbook creation.")
        continue

    # Create the workbook and 3 sheets
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for i, (row, row_number) in enumerate(valid_rows[:3], start=1):
            # collect answers (reference + all models)
            used_stories_long.add(row["story"].strip())
            answers_dict = {"Reference answer": row["answer"]}
            for model_name in MODEL_CONFIGS.keys():
                answers_dict[model_name] = load_model_answers(model_name, row_number, row["qid"])

            # Shuffle answers before placing in sheet (keeps annotator blind)
            model_items = list(answers_dict.items())
            random.shuffle(model_items)

            sheet_data = {
                "Story": row["story"],
                "Question": row["question"],
            }
            for idx_ans, (model_name, ans) in enumerate(model_items, start=1):
                sheet_data[f"Answer_{idx_ans}"] = ans

            sheet_data["Instruction"] = (
                "Please read the story and question carefully. Then, review all the given answers and "
                "rank them in order of correctness, starting from the most appropriate (Rank 1) to the least appropriate."
            )

            story_df = pd.DataFrame({
                "Field": list(sheet_data.keys()),
                "Value": list(sheet_data.values())
            })
            story_df.to_excel(writer, sheet_name=f"Story_{i}", index=False)

            answers_log_path = os.path.join(OUTPUT_EXCEL_DIR, f"answers/{lang}.txt")
            with open(answers_log_path, "a", encoding="utf-8") as log_f:
                log_f.write(f"\n=== Workbook: {os.path.basename(excel_path)} | Sheet: Story_{i} ===\n")
                for idx_ans, (model_name, ans) in enumerate(model_items, start=1):
                    log_f.write(f"Answer_{idx_ans} : {model_name} \n")
                log_f.write("\n")

    print(f"✅ Created workbook for '{lang}' with 3 story–question sheets → {excel_path}")

print("\nDone.")

# -------------------------------
# SHORT CONTEXT PROCESSING
# -------------------------------
OUTPUT_EXCEL_DIR = "excels/short"
df = pd.DataFrame(intersection_rows)
os.makedirs(OUTPUT_EXCEL_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_EXCEL_DIR, 'answers'), exist_ok=True)

TEST_DATA_PATHS = {
    "short_l6": "../finetuning/data/sc_l6_dataset/test.jsonl",
    "short_l6v2": "../finetuning/data/sc_l6v2_dataset/test.jsonl"
}
test_datasets = {context: load_jsonl(path) for context, path in TEST_DATA_PATHS.items()}

for lang in df["lang"].unique():
    lang_df = df[df["lang"] == lang].reset_index(drop=True)
    min_limit = MIN_WORD_LIMITS.get(lang, TARGET_WORD_COUNT)
    print(f"\nProcessing language: '{lang}'  (min_limit={min_limit})")
    if lang_df.empty:
        print(f"⚠️ No data rows for language '{lang}' — skipping.")
        continue

    # determine word limits
    if lang_df["no_of_words"].min() < TARGET_WORD_COUNT:
        lower, upper = TARGET_WORD_COUNT - WORD_MARGIN, TARGET_WORD_COUNT + WORD_MARGIN
    else:
        lower, upper = min_limit, min_limit + WORD_MARGIN
    filtered_df = lang_df[(lang_df["no_of_words"] >= lower) & (lang_df["no_of_words"] <= upper)].reset_index(drop=True)
    print(f" - Candidate rows in range [{lower}, {upper}]: {len(filtered_df)}")

    # Check if any model is completely missing outputs for this language in either context.
    missing_models = []
    for context in ["short_l6", "short_l6v2"]:
        context_test_data = test_datasets[context]
        for model_name in MODEL_CONFIGS.keys():
            has_any = model_has_any_answer_for_lang_short(model_name, context, lang_df, context_test_data)
            print(f"   Model {model_name} → has_any_answer_in_lang? {'YES' if has_any else 'NO'} (context={context})")
            if not has_any and model_name not in missing_models:
                missing_models.append(model_name)

    excel_path = os.path.join(OUTPUT_EXCEL_DIR, f"{lang}_stories.xlsx")
    txt_path = os.path.join(OUTPUT_EXCEL_DIR, f"answers/{lang}.txt")
    txt_lines = [f"Workbook: {os.path.basename(excel_path)}\n"]

    if missing_models:
        # If any model has NO answers at all for the language in either context, we cannot produce
        # a workbook where every model has answers in both contexts for the selected pairs.
        info_df = pd.DataFrame([
            {"Field": "Note", "Value": "One or more models produced NO answers for this language (in one or both contexts)."},
            {"Field": "Missing models", "Value": ", ".join(missing_models)}
        ])
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            info_df.to_excel(writer, sheet_name="Info", index=False)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("No sheets created due to missing model outputs.\n")
        print(f"⚠️ Blank workbook written for '{lang}' → {excel_path}")
        continue

    # Find all qids that exist in both contexts' test files and meet the filtered_df constraints,
    # and for which EVERY model has an answer in BOTH contexts.
    context_test_data_1 = test_datasets["short_l6"]
    context_test_data_2 = test_datasets["short_l6v2"]

    both_context_valid = []  # will store tuples (row, rownum_in_l6, rownum_in_l6v2)

    for _, row in filtered_df.iterrows():
        qid = row["qid"]

        # find positions in both test files
        row_num_1 = [i for i, r in enumerate(context_test_data_1) if r["qid"] == qid]
        row_num_2 = [i for i, r in enumerate(context_test_data_2) if r["qid"] == qid]
        if not row_num_1 or not row_num_2:
            continue
        r1, r2 = row_num_1[0], row_num_2[0]

        # require all models to have answers in both contexts for this qid
        all_models_ok = True
        for model_name in MODEL_CONFIGS.keys():
            ans1 = load_model_answers(model_name, "short_l6", r1, qid)
            ans2 = load_model_answers(model_name, "short_l6v2", r2, qid)
            if not ans1 or not ans2:
                all_models_ok = False
                break

        if all_models_ok:
            both_context_valid.append((row, r1, r2))

    # Deduplicate by story text and pick up to 3 distinct stories
    seen_stories = set()
    selected_rows = []  # (row, r1, r2) for up to 3 distinct stories
    for row, r1, r2 in both_context_valid:
        story_text = row["story"].strip()
        if story_text in seen_stories or story_text in used_stories_long:
            continue
        seen_stories.add(story_text)
        selected_rows.append((row, r1, r2))
        if len(selected_rows) >= 3:
            break

    # Write workbook: first write all short_l6 sheets (Story_1..3_short_l6),
    # then write matching short_l6v2 sheets (Story_1..3_short_l6v2).
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        if not selected_rows:
            info_df = pd.DataFrame([{"Note": "No valid distinct stories found where every model has answers in both contexts."}])
            info_df.to_excel(writer, sheet_name="Info", index=False)
            print(f"⚠️ No valid pairs found for '{lang}'. Wrote Info sheet.")
        else:
            # --- First: short_l6 sheets ---
            for i, (row, r1, r2) in enumerate(selected_rows, start=1):
                # answers for short_l6
                answers_dict = {
                    model_name: load_model_answers(model_name, "short_l6", r1, row["qid"])
                    for model_name in MODEL_CONFIGS.keys()
                }
                model_items = list(answers_dict.items())
                random.shuffle(model_items)
                sheet_name = f"Story_{i}_short_l6"

                txt_lines.append(f"\nSheet: {sheet_name}")
                for idx_ans, (model_name, _) in enumerate(model_items, start=1):
                    txt_lines.append(f"Answer_{idx_ans}: {model_name}")
                txt_lines.append("-" * 40)

                sheet_data = {
                    "Story": row["story"],
                    "Question": row["question"],
                    "Reference Answer": row["answer"],
                }
                for idx_ans, (model_name, ans) in enumerate(model_items, start=1):
                    sheet_data[f"Answer_{idx_ans}"] = ans
                sheet_data["Instruction"] = (
                    "Carefully read the story and question, review all answers, and rank them based on similarity to the reference answer."
                )

                story_df = pd.DataFrame({"Field": list(sheet_data.keys()), "Value": list(sheet_data.values())})
                story_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # --- Second: short_l6v2 sheets (matching order) ---
            for i, (row, r1, r2) in enumerate(selected_rows, start=1):
                # answers for short_l6v2
                answers_dict = {
                    model_name: load_model_answers(model_name, "short_l6v2", r2, row["qid"])
                    for model_name in MODEL_CONFIGS.keys()
                }
                model_items = list(answers_dict.items())
                random.shuffle(model_items)
                sheet_name = f"Story_{i}_short_l6v2"

                txt_lines.append(f"\nSheet: {sheet_name}")
                for idx_ans, (model_name, _) in enumerate(model_items, start=1):
                    txt_lines.append(f"Answer_{idx_ans}: {model_name}")
                txt_lines.append("-" * 40)

                sheet_data = {
                    "Story": row["story"],
                    "Question": row["question"],
                    "Reference Answer": row["answer"],
                }
                for idx_ans, (model_name, ans) in enumerate(model_items, start=1):
                    sheet_data[f"Answer_{idx_ans}"] = ans
                sheet_data["Instruction"] = (
                    "Carefully read the story and question, review all answers, and rank them based on similarity to the reference answer."
                )

                story_df = pd.DataFrame({"Field": list(sheet_data.keys()), "Value": list(sheet_data.values())})
                story_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Write the txt log
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    # Print summary of sheets created
    sheets_count = len(selected_rows) * 2  # two contexts per selected story
    print(f"✅ Created workbook for '{lang}' → {excel_path} ({sheets_count} sheets: {len(selected_rows)} stories × 2 contexts)")
    print(f"    Log: {txt_path}")
