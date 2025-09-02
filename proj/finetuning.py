# finetune.py
import os
import argparse
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from transformers import AutoTokenizer  
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from torch.utils.data import SequentialSampler

# --------------------------
# Arguments
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, required=True, help="Unique ID for logging/checkpoints")
parser.add_argument("--model", type=str, help="Model name or path")
parser.add_argument("--train", type=str, help="Training dataset file (JSON)")
parser.add_argument("--val", type=str, help="Validation dataset file (JSON)")
parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from last checkpoint if available")
parser.add_argument("--context", type=str, choices=["short", "long"], default="short", help="Training context type")
args = parser.parse_args()

# --------------------------
# Print arguments
# --------------------------
print("===== Run Arguments =====")
print(json.dumps(vars(args), indent=2))
print("=========================\n")

# --------------------------a
# Load datasets
# --------------------------
train_dataset = load_dataset("json", data_files=args.train, split="train")
val_dataset = load_dataset("json", data_files=args.val, split="train")
# --------------------------
# Tokenizer and Model
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(
    args.model, 
    cache_dir = f"/media/data_dump/aarya220007/cache/{args.model}"
)
tokenizer.pad_token = tokenizer.eos_token


nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,  
   bnb_4bit_quant_type="nf4",  
   bnb_4bit_use_double_quant=True,  
   bnb_4bit_compute_dtype= torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    args.model, 
    device_map="auto",  
    quantization_config=nf4_config, 
    cache_dir = f"/media/data_dump/aarya220007/cache/{args.model}",
    trust_remote_code=True,
    attn_implementation="eager",
)

peft_params = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_params)


finetuned_output_dir = f"/media/data_dump/aarya220007/finetuned_models/{args.model}"

# --------------------------
# Trainer setup
# --------------------------
sft_config = SFTConfig(
    output_dir=finetuned_output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=3e-4,
    num_train_epochs=2,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=3,
    fp16=True,
    completion_only_loss=True,
    max_length=max_len,
)

class OrderedSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, dataset=None):
        return SequentialSampler(self.train_dataset)

trainer = OrderedSFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=sft_config,
    peft_config=peft_params
)

# ================== DEBUG BATCH SIZE ==================
dataloader = trainer.get_train_dataloader()
batch = next(iter(dataloader))

print("\n===== DEBUG: Batch Info =====")
print("input_ids shape:", batch["input_ids"].shape)  # (batch_size, seq_len)
print("Batch size (num samples):", batch["input_ids"].size(0))
print("Sequence length per sample:", batch["input_ids"].size(1))
print("========================================\n")

# --------------------------
# Train and Save
# --------------------------
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
trainer.save_model(finetuned_output_dir)
