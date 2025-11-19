#!/bin/bash

BASE_OUTPUT="base_sts_evaluation"
MODELS=("sarvamai/sarvam-1" "CohereLabs/aya-23-8B" "sarvamai/OpenHathi-7B-Hi-v0.1-Base" "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "krutrim-ai-labs/Krutrim-2-instruct")

CONTEXTS=("long" "short_l6" "short_l6v2")

TOTAL_SPLITS=1

gpu_order=(2)


for model in "${MODELS[@]}"; do
    for context in "${CONTEXTS[@]}"; do
        echo "🚀 Starting evaluation for Model: $model | Context: $context"

        for split_index in $(seq 0 $((TOTAL_SPLITS - 1))); do
            gpu_id=${gpu_order[$split_index]}
            echo "Running split $split_index on GPU $gpu_id for $model / $context ..."
            
            CUDA_VISIBLE_DEVICES=$gpu_id python3 compute_sts.py \
                --model_name "$model" \
                --context_type "$context" \
                --split_index "$split_index" \
                --total_splits "$TOTAL_SPLITS" &
        done

        # Wait for all splits to finish before moving to next context
        wait
        echo "✅ Completed all splits for $model / $context"
    done
done

echo "🎯 All model/context evaluations completed!"
