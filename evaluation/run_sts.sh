#!/bin/bash

BASE_OUTPUT="sts_evaluation"
MODELS=("meta-llama/Llama-3.1-8B-Instruct" "krutrim-ai-labs/Krutrim-2-instruct" "sarvamai/sarvam-1" "CohereLabs/aya-23-8B" "sarvamai/OpenHathi-7B-Hi-v0.1-Base" "Qwen2.5-7B-Instruct")
CONTEXTS=("long" "short_l6" "short_l6v2")

TOTAL_SPLITS=4   # Total number of parallel jobs
THIS_SPLIT=$1    # Split index passed as first argument (0-based)

for model in "${MODELS[@]}"; do
    for context in "${CONTEXTS[@]}"; do
        BATCH_DIR="../inferencing/outputs/${model}/${context}"
        OUTPUT_DIR="${BASE_OUTPUT}/${model}/${context}"

        # Create output directory if it doesn't exist
        mkdir -p "$OUTPUT_DIR"

        if [ ! -d "$BATCH_DIR" ]; then
            echo "Directory $BATCH_DIR does not exist. Skipping..."
            continue
        fi

        # Get sorted list of batch files
        batch_files=($(ls "$BATCH_DIR"/batch_*.txt | sort))

        # Total number of batch files
        total_files=${#batch_files[@]}

        # Compute start and end indices for this split
        split_size=$(( (total_files + TOTAL_SPLITS - 1) / TOTAL_SPLITS ))  # ceil division
        start_index=$(( THIS_SPLIT * split_size ))
        end_index=$(( start_index + split_size - 1 ))
        if [ $end_index -ge $total_files ]; then
            end_index=$(( total_files - 1 ))
        fi

        echo "Split $THIS_SPLIT: processing files $start_index to $end_index out of $total_files"

        # Process only batch files in this split
        for i in $(seq $start_index $end_index); do
            batch_file="${batch_files[$i]}"
            batch_name=$(basename "$batch_file" .txt)
            OUTPUT_FILE="${OUTPUT_DIR}/${batch_name}.json"

            echo "Processing $batch_file -> $OUTPUT_FILE"
            python3 compute_sts.py \
                --batch_file "$batch_file" \
                --output_file "$OUTPUT_FILE"
        done
    done
done
