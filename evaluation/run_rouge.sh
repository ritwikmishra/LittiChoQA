#!/bin/bash

BASE_OUTPUT="rouge_evaluation"
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "krutrim-ai-labs/Krutrim-2-instruct"
    "sarvamai/sarvam-1"
    "CohereLabs/aya-23-8B"
    "sarvamai/OpenHathi-7B-Hi-v0.1-Base"
)
CONTEXTS=("long" "short_l6" "short_l6v2")

TOTAL_SPLITS=2   # total number of splits (can change globally)
THIS_SPLIT=$1    # 0-based split index passed as argument

if [ -z "$THIS_SPLIT" ]; then
    echo "❌ Please provide split index (0 to $((TOTAL_SPLITS-1)))."
    exit 1
fi

for model in "${MODELS[@]}"; do
    for context in "${CONTEXTS[@]}"; do
        BATCH_DIR="../inferencing/outputs/${model}/${context}"
        OUTPUT_DIR="${BASE_OUTPUT}/${model}/${context}"

        # Create output directory if needed
        mkdir -p "$OUTPUT_DIR"

        if [ ! -d "$BATCH_DIR" ]; then
            echo "⚠️ Directory $BATCH_DIR does not exist. Skipping..."
            continue
        fi

        # Get sorted list of batch files
        mapfile -t batch_files < <(ls "$BATCH_DIR"/batch_*.txt 2>/dev/null | sort -V)
        total_files=${#batch_files[@]}

        if [ "$total_files" -eq 0 ]; then
            echo "⚠️ No batch files found in $BATCH_DIR. Skipping..."
            continue
        fi

        # Compute split range
        split_size=$(( (total_files + TOTAL_SPLITS - 1) / TOTAL_SPLITS ))  # ceil division
        start_index=$(( THIS_SPLIT * split_size ))
        end_index=$(( start_index + split_size - 1 ))
        if [ $end_index -ge $total_files ]; then
            end_index=$(( total_files - 1 ))
        fi

        echo "📘 Model: $model | Context: $context"
        echo "🧩 Split $THIS_SPLIT: Processing batch files $start_index to $end_index (of $total_files total)"

        # Loop through assigned batch files
        for i in $(seq $start_index $end_index); do
            batch_file="${batch_files[$i]}"
            [ -z "$batch_file" ] && continue

            batch_name=$(basename "$batch_file" .txt)
            OUTPUT_FILE="${OUTPUT_DIR}/${batch_name}.json"

            # Skip if output file already exists
            if [ -f "$OUTPUT_FILE" ]; then
                echo "✅ Already processed: $OUTPUT_FILE"
                continue
            fi

            python3 compute_rouge.py \
                --batch_file "$batch_file" \
                --output_file "$OUTPUT_FILE"
        done
        echo "✅ Completed model: $model / context: $context / split: $THIS_SPLIT"
    done
done
