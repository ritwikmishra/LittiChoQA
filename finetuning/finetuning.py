import os
import argparse
import json
import shutil
import torch
from torch.utils.data import SequentialSampler
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model


# --------------------------
# Arguments
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, required=True, help="Unique ID for logging/checkpoints")
parser.add_argument("--model", required=True, type=str, help="Model name or path")
parser.add_argument("--train", required=True, type=str, help="Training dataset file (JSON)")
parser.add_argument("--val", required=True, type=str, help="Validation dataset file (JSON)")
parser.add_argument("--context", required=True, type=str, help="Training context type")
parser.add_argument("--max_len", type=int, required=True, help="Max len required for training")
args = parser.parse_args()

print("===== Run Arguments =====")
print(json.dumps(vars(args), indent=2))
print("=========================\n")

# --------------------------
# Load lang_dataset mapping
# --------------------------
lang_dataset_path = "data/lang_dataset.jsonl"
qid_to_token_len = {}

with open(lang_dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        qid_to_token_len[row["qid"]] = row["tokenized_length"]

print(f"Loaded tokenized lengths for {len(qid_to_token_len)} QIDs")


# --------------------------
# Function to filter dataset
# --------------------------
def filter_by_max_len(example):
    qid = example.get("qid")
    token_len = qid_to_token_len.get(qid, None)
    if token_len is None:
        return False
    return token_len <= args.max_len


# --------------------------
# Load and filter datasets
# --------------------------
train_dataset = load_dataset("json", data_files=args.train, split="train")
print(f"Original train dataset size: {len(train_dataset)}")
train_dataset = train_dataset.filter(filter_by_max_len)
print(f"Trimmed train dataset size: {len(train_dataset)}")

val_dataset = load_dataset("json", data_files=args.val, split="train")
print(f"Original val dataset size: {len(val_dataset)}")
val_dataset = val_dataset.filter(filter_by_max_len)
print(f"Trimmed val dataset size: {len(val_dataset)}")


# --------------------------
# Tokenizer
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    cache_dir=f"cache/{args.model}",
    use_fast=True,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id


# --------------------------
# Tokenize function
# --------------------------
def tokenize_and_build_labels(batch):
    prompts = batch["prompt"]
    completions = batch["completion"]
    full_texts = [p + c for p, c in zip(prompts, completions)]

    tokenized_prompts = tokenizer(prompts, truncation=False, add_special_tokens=False)
    tokenized_full = tokenizer(
        full_texts,
        truncation=True,
        max_length=args.max_len,
        padding="max_length",
        return_attention_mask=True
    )

    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]

    labels = []
    for i, ids in enumerate(input_ids):
        prompt_len = len(tokenized_prompts["input_ids"][i])
        if prompt_len >= args.max_len:
            lbl = [-100] * args.max_len
        else:
            lbl = ids.copy()
            for j in range(prompt_len):
                lbl[j] = -100
        labels.append(lbl)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# --------------------------
# Tokenize function
# --------------------------
def tokenize_and_build_labels(batch):
    prompts = batch["prompt"]
    completions = batch["completion"]
    full_texts = [p + c for p, c in zip(prompts, completions)]

    tokenized_prompts = tokenizer(prompts, truncation=False, add_special_tokens=False)
    tokenized_full = tokenizer(
        full_texts,
        truncation=True,
        max_length=args.max_len,
        padding="max_length",
        return_attention_mask=True
    )

    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]

    labels = []
    for i, ids in enumerate(input_ids):
        prompt_len = len(tokenized_prompts["input_ids"][i])
        if prompt_len >= args.max_len:
            lbl = [-100] * args.max_len
        else:
            lbl = ids.copy()
            for j in range(prompt_len):
                lbl[j] = -100
        labels.append(lbl)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }



# Map datasets
tokenized_train = train_dataset.map(
    tokenize_and_build_labels,
    batched=True,
    remove_columns=train_dataset.column_names
)
tokenized_val = val_dataset.map(
    tokenize_and_build_labels,
    batched=True,
    remove_columns=val_dataset.column_names
)

tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# --------------------------
# Filter fully masked
# --------------------------
def has_valid_labels(example):
    return any(l != -100 for l in example["labels"])

print("Filtering fully-masked samples...")
tokenized_train = tokenized_train.filter(has_valid_labels)
tokenized_val = tokenized_val.filter(has_valid_labels)

print(f"Final train dataset size: {len(tokenized_train)}")
print(f"Final val dataset size: {len(tokenized_val)}")


# --------------------------
# Model
# --------------------------
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    device_map="auto",
    quantization_config=nf4_config,
    cache_dir=f"cache/{args.model}",
    trust_remote_code=True,
)

if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id


# --------------------------
# PEFT LoRA
# --------------------------
def detect_lora_targets(model):
    module_names = [name for name, _ in model.named_modules()]
    lower_names = [n.lower() for n in module_names]

    if any("q_proj" in n for n in lower_names):
        return ["q_proj", "v_proj"]
    elif any("c_attn" in n for n in lower_names):
        return ["c_attn", "c_proj"]  # mgpt uses this
    else:
        raise ValueError(f"Could not detect attention projection layer names. Sample names: {module_names[:20]}")

target_modules = detect_lora_targets(model)

peft_params = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_params)
model.print_trainable_parameters()


# --------------------------
# Callback
# --------------------------
class PrintBatchLengthCallback(TrainerCallback):
    def on_train_batch_begin(self, args, state, control, **kwargs):
        batch = kwargs["inputs"]
        if "input_ids" in batch:
            input_ids = batch["input_ids"]
            print(
                f"[CALLBACK][Step {state.global_step}] "
                f"Batch size: {input_ids.size(0)}, Seq length: {input_ids.size(1)}",
                flush=True
            )
        return control


# --------------------------
# Output dirs
# --------------------------
finetuned_base_dir = "finetuned_models"
model_dir = os.path.join(finetuned_base_dir, args.model)
runs_dir = os.path.join(model_dir, "runs")
os.makedirs(runs_dir, exist_ok=True)

# Run-specific directory
finetuned_output_dir = os.path.join(runs_dir, args.run_id)
os.makedirs(finetuned_output_dir, exist_ok=True)

# Run-specific best checkpoint path
best_ckpt_dir = os.path.join(finetuned_output_dir, "best_checkpoint")


# --------------------------
# Trainer setup
# --------------------------
sft_config = SFTConfig(
    output_dir=finetuned_output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=1e-6,
    num_train_epochs=1,
    eval_strategy="steps",
    logging_strategy="steps",
    save_strategy="steps",
    eval_steps=200,
    save_steps=200,
    logging_steps=50,
    save_total_limit=3,
    bf16=True,
    completion_only_loss=True,
    max_length=args.max_len,
    report_to="none",
    load_best_model_at_end=True,
    max_grad_norm=1.0,
)


class OrderedSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, dataset=None):
        return SequentialSampler(self.train_dataset)


trainer = OrderedSFTTrainer(
    model=model,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    args=sft_config,
    peft_config=peft_params,
    callbacks=[PrintBatchLengthCallback()]
)


# --------------------------
# Train and Save
# --------------------------
try:
    # Resume from this run's best checkpoint if exists
    if os.path.exists(best_ckpt_dir):
        resume_path = best_ckpt_dir
        print(f"Resuming training from this run's best checkpoint: {resume_path}")
    else:
        resume_path = None
        print("No run-specific best checkpoint found. Starting training from scratch.")

    trainer.train(resume_from_checkpoint=resume_path)

    trainer.model.save_pretrained(finetuned_output_dir)
    tokenizer.save_pretrained(finetuned_output_dir)
    print(f"Model for this run saved to {finetuned_output_dir}")

    if trainer.state.best_model_checkpoint:
        if os.path.exists(best_ckpt_dir):
            shutil.rmtree(best_ckpt_dir)
        shutil.copytree(trainer.state.best_model_checkpoint, best_ckpt_dir)
        print(f"Best checkpoint for this run saved at: {best_ckpt_dir}")

except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("\n===== CUDA OOM DETECTED =====")
        print("Error:", str(e))
        print("========================================\n")
    else:
        raise
