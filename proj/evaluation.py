import os
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
import tensorflow_hub as hub
from bert_score import BERTScorer
from sts_utils import sts, LaserEncoderPipeline 

# ------------------------------
# Dataset Loading
# ------------------------------
dataset_path = "/media/data_dump/aarya220007/combined_all_response.json"
prompts = []
references = []
questions = []  # keep for debug/printing

with open(dataset_path, "r", encoding="utf-8") as f:
    data_iter = json.JSONDecoder().raw_decode(f.read())
    data = json.loads(json.dumps(data_iter[0]))  
    keys = list(data.keys())[:1]  

    for key in keys:
        story_text = data[key]["story"]
        qa_list = data[key]["qas"]["non-factoid"][:1]  
        for qa in qa_list:
            question = qa["question"]
            reference = qa["answer"]

            # instruction-style prompt
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
# model_name = "google/gemma-3-4b-it"

cache_dir = f"/media/data_dump/aarya220007/cache/{model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    device_map=None
)
model.to("cuda:0")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------
# Inference
# ------------------------------
generated_texts = []

for idx, prompt in enumerate(prompts):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract only generated part after input
    gen_only_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_answer = tokenizer.decode(gen_only_ids, skip_special_tokens=True).strip()
    generated_texts.append(generated_answer)

    if (idx + 1) % 5 == 0:
        print(f"Processed {idx + 1}/{len(prompts)} examples...")


# ------------------------------
# ROUGE Evaluation
# ------------------------------
rouge = evaluate.load("rouge")
results = rouge.compute(predictions=generated_texts, references=references)

print("\n--- ROUGE Scores ---")
for key, value in results.items():
    print(f"{key}: {value:.4f}")


# ------------------------------
# STS Evaluation
# ------------------------------
print("\nLoading STS models...")

# BERTScorer
scorer = BERTScorer(model_type='bert-base-multilingual-cased', device='cpu')

# Universal Sentence Encoder
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# LaBSE
labse_preprocessor = hub.KerasLayer(
    "https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/cmlm-multilingual-preprocess/2"
)
labse_encoder = hub.KerasLayer(
    "https://www.kaggle.com/models/google/labse/TensorFlow2/labse/2"
)

# LASER
laser_encoder = LaserEncoderPipeline(laser="laser2")

semantic_scores = []

for gen, ref in tqdm(zip(generated_texts, references), total=len(references), desc="Computing STS"):
    score_dict = sts(gen, ref, scorer, use_model, labse_preprocessor, labse_encoder, laser_encoder)
    semantic_scores.append(score_dict['avg'])

avg_semantic_score = sum(semantic_scores) / len(semantic_scores)
print(f"\nAverage Semantic Score (4-model avg): {avg_semantic_score:.4f}")

# ------------------------------
# Sample Outputs
# ------------------------------
print("\n--- Sample Predictions ---")
for i in range(min(3, len(prompts))):
    print(f"\nPrompt: {prompts[i][:200]}...")
    print(f"Generated: {generated_texts[i]}")
    print(f"Reference: {references[i]}")
    print("------")
