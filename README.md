# LittiChoQA

## Installation Requirements

This repository uses **multiple virtual environments**, each corresponding to a specific module or script.  
All requirments files are stored in the `envs/` directory.

Below is the mapping between project components and their respective `requirements.txt` files:

| Component / Script                                      | Requirements File             |
|---------------------------------------------------------|-------------------------------|
| Finetuning & Inference (`finetuning/`, `inferencing/`)  | `envs/finetune_inference.txt` |
| ROUGE Evaluation (`evaluation/compute_rouge.py`)        | `envs/rouge_eval.txt`         |
| STS Evaluation (`evaluation/compute_sts.py`)            | `envs/sts_eval.txt`           |
| LLM Annotation (`annotation/llm_annotator.py`)          | `envs/llm_annotator.txt`      |
| HuggingFace Upload (`push_to_hub.py`)                   | `envs/push_to_hub.txt`        |

To set up an environment locally, run:

```bash
python -m venv <virtual_env_name>
source <virtual_env_name>/bin/activate
pip install -r <requirements_file>.txt
````

---

## Dataset

The dataset used in this project is hosted on Hugging Face:

🔗 **Dataset URL:** [https://huggingface.co/datasets/ritwikm/LittiChoQA](https://huggingface.co/datasets/ritwikm/LittiChoQA)

The list of internet sources used during dataset construction can be found in `sources.md`.

---

## Data Processing

Scripts for dataset preprocessing are located in the `dataset_creation/` directory.
These scripts generate multiple dataset formats depending on the required context length:

| Script                        | Purpose                                       |
| ----------------------------- | --------------------------------------------- |
| `creating_fc_dataset.py`      | Generates dataset for **long-context** format |
| `creating_sc_l6_dataset.py`   | Generates dataset for **short L6** context    |
| `creating_sc_l6v2_dataset.py` | Generates dataset for **short L6v2** context  |

---

## Running Finetuning

Finetuning a model can be done in two ways:

### **1. Finetuning a Single Model**

Use the script located at:

```
finetuning/finetuning.py
```

This script accepts the following arguments:

| Argument    | Description                                   |
| ----------- | --------------------------------------------- |
| `--run_id`  | Unique identifier for logging and checkpoints |
| `--model`   | Base model name or path                       |
| `--train`   | Path to the training dataset (JSON)           |
| `--val`     | Path to the validation dataset (JSON)         |
| `--context` | Context type                                  |
| `--max_len` | Maximum sequence length for training          |

Run the script as:

```bash
python finetuning/finetuning.py --run_id <id> --model <model_name> \
--train <train_file.json> --val <val_file.json> --context <context_type> \
--max_len <length>
```

### **2. Finetuning All Models Together**

A helper script is available at:

```
finetuning/run_all.sh
```

Run it as:

```bash
bash finetuning/run_all.sh
```

---

PEFT adapters of the best performing model (Krutrim) are available here: [https://huggingface.co/ritwikm/Finetuned_krutrim_short_l6](https://huggingface.co/ritwikm/Finetuned_krutrim_short_l6)

## Running Inference

Inference is performed using **VLLM**. There are two workflows:

### **1. Inference for a Single Model**

#### **Step 1: Merge LoRA Weights**

Run:

```bash
python merge_lora.py \
  --base_model <base_model_name> \
  --lora_checkpoint <checkpoint_path> \
  --output_dir <merged_output_dir>
```

Arguments for `merge_lora.py`:

| Argument            | Description                        |
| ------------------- | ---------------------------------- |
| `--base_model`      | Name of the base model             |
| `--lora_checkpoint` | Path to LoRA checkpoint            |
| `--output_dir`      | Directory to save the merged model |

#### **Step 2: Run VLLM Inference**

Run:

```bash
python inferencing/inference.py \
  --test_file <test.jsonl> \
  --context <context_type> \
  --checkpoint <merged_model_path> \
  --batch_size <batch_size> \
  --output_dir <output_directory> \
  --base_model <base_model_name>
```

Arguments for `inference.py`:

| Argument       | Description                                            |
| -------------- | ------------------------------------------------------ |
| `--test_file`  | Path to test JSONL file                                |
| `--context`    | Context type                                           |
| `--checkpoint` | Path to the fine-tuned/merged checkpoint               |
| `--batch_size` | Batch size                                             |
| `--output_dir` | Output directory for predictions                       |
| `--base_model` | Base model name                                        |

### **2. Inference for All Model–Context Pairs**

Use the batch script:

```
run_vllm.sh
```

To execute:

```bash
bash run_vllm.sh
```

---

## Running Evaluation

Evaluation scripts are available in the `evaluation/` directory:

| Metric | Script                    |
| ------ | ------------------------- |
| ROUGE  | `evaluation/run_rouge.sh` |
| STS    | `evaluation/run_sts.sh`   |

Run them via:

```bash
bash evaluation/run_rouge.sh
bash evaluation/run_sts.sh
```

---

## Model Selection Criteria

Models were selected from Hugging Face based on the following filters:

* Task: **Text Generation**
* Frameworks: **PyTorch**, **Transformers**
* Languages: **Hindi**, **Tamil**, **Urdu**, **Telugu**
* Parameter Count: **> 1B**
* Architecture: **CausalLM**
* Full list of models is documented in `mt_llms.md`.
