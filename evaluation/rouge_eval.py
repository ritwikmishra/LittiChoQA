import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer

# ------------------------------
# Dataset Loading
# ------------------------------
dataset_path = "combined_all_response.json"
prompts, references, questions = [], [], []

with open(dataset_path, "r", encoding="utf-8") as f:
    data_iter = json.JSONDecoder().raw_decode(f.read())
    data = json.loads(json.dumps(data_iter[0]))  
    keys = list(data.keys())[:1]  

    for key in keys:
        story_text = data[key]["story"]
        qa_list = data[key]["qas"]["non-factoid"][:1]  
        for qa in qa_list:
            question, reference = qa["question"], qa["answer"]

            prompt = (
                "Answer the following question based on the given story.\n\n"
                f"Story:\n{story_text}\n\n"
                f"##Question:\n{question}\n\n"
                "##Answer:"
            )
            prompts.append(prompt)
            references.append(reference)
            questions.append(question)

print(f"Loaded {len(prompts)} Q/A pairs from first story.")

# ------------------------------
# Load Model
# ------------------------------
model_name = "meta-llama/Llama-3.1-8B-Instruct"
cache_dir = f"cache/{model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    dtype=torch.float16,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------
# Inference
# ------------------------------
device = next(model.parameters()).device  
generated_texts = []
for idx, prompt in enumerate(prompts):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_only_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_answer = tokenizer.decode(gen_only_ids, skip_special_tokens=True).strip()
    generated_texts.append(generated_answer)

# ------------------------------
# ROUGE Evaluation
# ------------------------------
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=False)
rouge_results = [scorer.score(gen, ref) for gen, ref in zip(generated_texts, references)]

avg_scores = {key: sum([r[key].fmeasure for r in rouge_results]) / len(rouge_results)
              for key in ['rouge1', 'rouge2', 'rouge3', 'rougeL']}

print("\n--- ROUGE Scores (F1 average) ---")
for k, v in avg_scores.items():
    print(f"{k}: {v:.4f}")
