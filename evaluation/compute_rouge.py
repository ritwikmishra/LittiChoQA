import os
import argparse
from indictrans import Transliterator
from rouge_score import rouge_scorer
import json

# ------------------------------
# Arguments
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--batch_file', type=str, required=True, help='Path to batch file')
parser.add_argument('--output_file', type=str, required=True, help='Path to save results (json)')
args = parser.parse_args()

# ------------------------------
# Unicode-based script detection
# ------------------------------
def detect_script(text: str) -> str:
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F or (0xA8E0 <= code <= 0xA8FF):
            return 'deva'
        if 0x0980 <= code <= 0x09FF:
            return 'beng'
        if (0x0600 <= code <= 0x06FF) or (0x0750 <= code <= 0x077F) or (0x08A0 <= code <= 0x08FF):
            return 'arab'
        if 0x0041 <= code <= 0x005A or 0x0061 <= code <= 0x007A:
            return 'latin'
    return 'unknown'

transliterators = {
    'deva': Transliterator(source='hin', target='eng', build_lookup=True),
    'beng': Transliterator(source='asm', target='eng', build_lookup=True),
    'arab': Transliterator(source='urd', target='eng', build_lookup=True)
}

def transliterate_text(text: str) -> str:
    script = detect_script(text)
    if script in transliterators:
        try:
            return transliterators[script].transform(text)
        except Exception:
            pass
    return text

# ------------------------------
# Load reference and generated answers
# ------------------------------
results = []

with open(args.batch_file, 'r', encoding='utf-8') as f:
    content = f.read()

qid_blocks = content.split("QID:")[1:]
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

for block in qid_blocks:
    # Extract QID
    qid_end = block.find("Prompt:")
    qid = block[:qid_end].strip() if qid_end != -1 else "unknown"

    # Extract Reference Answer
    ref_answer = None
    if "Reference Answer:" in block and "Generated Answer:" in block:
        ref_start = block.index("Reference Answer:") + len("Reference Answer:")
        gen_start = block.index("Generated Answer:")
        ref_answer = block[ref_start:gen_start].strip()
    
    # Extract Generated Answer
    gen_answer = None
    if "Generated Answer:" in block and "-----" in block:
        gen_start = block.index("Generated Answer:") + len("Generated Answer:")
        gen_end = block.index("-----", gen_start)
        gen_answer = block[gen_start:gen_end].strip()

    if ref_answer and gen_answer:
        # Transliterate
        ref_translit = transliterate_text(ref_answer)
        gen_translit = transliterate_text(gen_answer)

        # ROUGE
        rouge_scores = scorer.score(ref_translit, gen_translit)

        # Save only QID and ROUGE scores
        results.append({
            "qid": qid,
            "rouge": rouge_scores
        })

# Save results as JSON
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)
