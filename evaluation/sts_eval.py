import os
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bert_score import BERTScorer
from sts_utils import *
import tensorflow_hub as hub
from laser_encoders import LaserEncoderPipeline

# ------------------------------
# Dataset Loading
# ------------------------------
dataset_path = "/media/data_dump/aarya220007/finetuning_proj/combined_all_response.json"
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
cache_dir = f"/media/data_dump/aarya220007/finetuning_proj/cache/{model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    device_map=None
).to("cuda:0")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------
# Inference
# ------------------------------
generated_texts = []
for prompt in prompts:
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
    gen_only_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_answer = tokenizer.decode(gen_only_ids, skip_special_tokens=True).strip()
    generated_texts.append(generated_answer)

# ------------------------------
# STS Evaluation
# ------------------------------
print("\nLoading STS models...")
scorer = BERTScorer(model_type='bert-base-multilingual-cased', device='cpu')
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
labse_preprocessor = hub.KerasLayer(
    "https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/cmlm-multilingual-preprocess/2"
)
labse_encoder = hub.KerasLayer(
    "https://www.kaggle.com/models/google/labse/TensorFlow2/labse/2"
)
laser_encoder = LaserEncoderPipeline(laser="laser2")

semantic_scores = []
for gen, ref in tqdm(zip(generated_texts, references), total=len(references), desc="Computing STS"):
    score_dict = sts(gen, ref, scorer, use_model, labse_preprocessor, labse_encoder, laser_encoder)
    semantic_scores.append(score_dict['avg'])

avg_semantic_score = sum(semantic_scores) / len(semantic_scores)
print(f"\nAverage Semantic Score (4-model avg): {avg_semantic_score:.4f}")
