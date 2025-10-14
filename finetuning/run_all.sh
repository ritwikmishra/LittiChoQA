#!/bin/bash

# List of dataset triplets (train, val, context)
datasets=(
  "data/fc_dataset/train.jsonl data/fc_dataset/val.jsonl long"
  "data/sc_l6_dataset/train.jsonl data/sc_l6_dataset/val.jsonl short_l6"
  "data/sc_l6v2_dataset/train.jsonl data/sc_l6v2_dataset/val.jsonl short_l6v2"
)

models=(
  # "1shoomun/qwen2.5-14b-desi"
  # "meta-llama/Llama-3.1-8B-Instruct"
  # "CohereLabs/aya-23-8B"
  # "sarvamai/sarvam-1"
  # "ai-forever/mGPT-13B"
  # "ai-forever/mGPT"
  # "sarvamai/OpenHathi-7B-Hi-v0.1-Base"
  # "krutrim-ai-labs/Krutrim-2-instruct"
  Qwen/Qwen2.5-7B-Instruct
)

mkdir -p logs

for model in "${models[@]}"; do
  run_id=1   

  for dataset in "${datasets[@]}"; do
    set -- $dataset  
    train=$1
    val=$2
    context=$3

    echo "Running Run ID: $run_id | Model: $model | Dataset: $train, $val | Context: $context | Max Sequence Len: 1250"
    python finetuning.py \
      --run_id "$run_id" \
      --train "$train" \
      --val "$val" \
      --context "$context" \
      --max_len 1250 \
      --model "$model" \
      >> "logs/${model//\//_}_run${run_id}.log"

    run_id=$((run_id + 1))
  done
done
