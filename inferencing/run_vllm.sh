#!/bin/bash

# Paths
BASE_DIR="../finetuning/finetuned_models"
OUTPUT_DIR="base_outputs"
LOG_DIR="base_logs"
MERGED_DIR="base_models"

# Create directories if they don't exist
mkdir -p "$LOG_DIR" "$MERGED_DIR"

# Test datasets
DATA_LONG="../finetuning/data/fc_dataset/test.jsonl"
DATA_SHORT_L6="../finetuning/data/sc_l6_dataset/test.jsonl"
DATA_SHORT_L6V2="../finetuning/data/sc_l6v2_dataset/test.jsonl"

# Contexts
CONTEXT_LONG="long"
CONTEXT_SHORT_L6="short_l6"
CONTEXT_SHORT_L6V2="short_l6v2"

# Hardcoded models
models=(    
  "meta-llama/Llama-3.1-8B-Instruct"
  "krutrim-ai-labs/Krutrim-2-instruct"   
  "sarvamai/sarvam-1"
  "CohereLabs/aya-23-8B"  
  "sarvamai/OpenHathi-7B-Hi-v0.1-Base"
  "Qwen/Qwen2.5-7B-Instruct"    
  "1shoomun/qwen2.5-14b-desi"   #error: Engine core initialization failed
  "ai-forever/mGPT-13B"  #giving empty generated output
  "ai-forever/mGPT"  #giving empty generated output  
)

# Loop over models
for MODEL_NAME in "${models[@]}"; do
    echo ">>> Running inferences for model: $MODEL_NAME"
    MODEL_DIR="$BASE_DIR/$MODEL_NAME/runs"

    # Log file for this model
    MODEL_LOG="$LOG_DIR/$(echo "$MODEL_NAME" | tr '/' '_').log"

    # --- Merge LoRA checkpoint into base model ---
    MERGED_PATH="$MERGED_DIR/$(echo "$MODEL_NAME" | tr '/' '_')"
    mkdir -p "$MERGED_PATH"

    # echo "[INFO] Merging LoRA for $MODEL_NAME..." >> "$MODEL_LOG"
    python merge_lora.py \
        --base_model "$MODEL_NAME" \
        --lora_checkpoint "$MODEL_DIR/1/best_checkpoint" \
        --output_dir "$MERGED_PATH" >> "$MODEL_LOG" 

    # --- Run inference with vLLM for all contexts ---

    echo "[INFO] Running vLLM inference (short_l6)..." >> "$MODEL_LOG"
    python vllm_inference.py \
        --test_file "$DATA_SHORT_L6" \
        --context "$CONTEXT_SHORT_L6" \
        --checkpoint "$MERGED_PATH" \
        --batch_size 2 \
        --output_dir "$OUTPUT_DIR/$MODEL_NAME/short_l6" \
        --tensor_parallel_size 2 >> "$MODEL_LOG" 

    echo "[INFO] Running vLLM inference (short_l6v2)..." >> "$MODEL_LOG"
    python vllm_inference.py \
        --test_file "$DATA_SHORT_L6V2" \
        --context "$CONTEXT_SHORT_L6V2" \
        --checkpoint "$MERGED_PATH" \
        --batch_size 2 \
        --output_dir "$OUTPUT_DIR/$MODEL_NAME/short_l6v2" \
        --tensor_parallel_size 2 >> "$MODEL_LOG" 

    echo "[INFO] Running vLLM inference (long)..." >> "$MODEL_LOG"
    python vllm_inference.py \
        --test_file "$DATA_LONG" \
        --context "$CONTEXT_LONG" \
        --checkpoint "$MERGED_PATH" \
        --batch_size 2 \
        --output_dir "$OUTPUT_DIR/$MODEL_NAME/long" \
        --tensor_parallel_size 2 >> "$MODEL_LOG" 
    echo "Logs for $MODEL_NAME saved in $MODEL_LOG"

    # --- Delete merged model to save space ---
    echo "[INFO] Deleting merged model for $MODEL_NAME to save space..." >> "$MODEL_LOG"
    rm -rf "$MERGED_PATH"
    echo "[INFO] Deleted $MERGED_PATH" >> "$MODEL_LOG"

done
