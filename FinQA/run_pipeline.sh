#!/bin/bash

# Simple FinQA Pipeline Runner
# Usage: sh ./run_pipeline.sh [results_suffix] [cpu_usage]

# ===== ARGUMENT PROCESSING =====
VERSION="v$(date +%m%d)"
INPUT_QA="./data/qa_dict.json"
INPUT_QA_LEVELS="./data/qa_dict_levels.json"
CPU_USAGE=${2:-0.75}

# Auto-increment suffix if not provided
if [ -z "$1" ]; then
    # Create directory first if it doesn't exist
    RESULTS_DIR="./results/${VERSION}"
    mkdir -p "$RESULTS_DIR"
    
    echo "Checking for existing files in: $RESULTS_DIR"
    
    # Find highest existing suffix number
    HIGHEST_NUM=0
    FOUND_FILES=false
    
    # Check if any results files exist
    for file in "$RESULTS_DIR"/results_thoughts_v*.json; do
        if [ -f "$file" ]; then
            FOUND_FILES=true
            # Extract number from filename like results_thoughts_v2.json -> 2
            NUM=$(basename "$file" | sed 's/results_thoughts_v\([0-9]*\)\.json/\1/')
            if [ "$NUM" -gt "$HIGHEST_NUM" ] 2>/dev/null; then
                HIGHEST_NUM=$NUM
            fi
        fi
    done
    
    # Increment by 1
    NEXT_NUM=$((HIGHEST_NUM + 1))
    RESULTS_SUFFIX="v${NEXT_NUM}"
    
    if [ "$FOUND_FILES" = true ]; then
        echo "Auto-detected next suffix: $RESULTS_SUFFIX (previous highest: v${HIGHEST_NUM})"
    else
        echo "No existing files found. Starting with: $RESULTS_SUFFIX"
    fi
else
    RESULTS_SUFFIX="$1"
    echo "Using provided suffix: $RESULTS_SUFFIX"
fi

echo "CPU usage: ${CPU_USAGE} ($(echo "$CPU_USAGE * 100" | bc -l | cut -d. -f1)%)"

# ===== CONFIGURATION =====
RESULTS_DIR="./results/${VERSION}"
SCORES_DIR="./scores/${VERSION}"
RESULTS_FILE="${RESULTS_DIR}/results_thoughts_${RESULTS_SUFFIX}.json"
SCORES_FILE="${SCORES_DIR}/results_with_score_${RESULTS_SUFFIX}.json"

# ===== PIPELINE EXECUTION =====
echo "=== FinQA Pipeline Runner ==="
echo "Version: $VERSION"
echo "Results suffix: $RESULTS_SUFFIX"
echo "Input QA: $INPUT_QA"
echo "Results file: $RESULTS_FILE"
echo "Scores file: $SCORES_FILE"
echo

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$SCORES_DIR"

# Step 1: Run question processing
echo "Step 1: Running question processing..."
python mcp_client_with_thought.py --input "$INPUT_QA" --output "$RESULTS_FILE" --cpu-usage "$CPU_USAGE"

if [ $? -ne 0 ]; then
    echo "❌ Question processing failed!"
    exit 1
fi

echo "✅ Question processing completed"
echo

# Step 2: Run scoring
echo "Step 2: Running scoring..."
python score.py --input-qa "$INPUT_QA_LEVELS" --input-results "$RESULTS_FILE" --output "$SCORES_FILE" --cpu-usage "$CPU_USAGE"

if [ $? -ne 0 ]; then
    echo "❌ Scoring failed!"
    exit 1
fi

echo "✅ Scoring completed"
echo

echo "🎉 Pipeline completed successfully!"
echo "Results: $RESULTS_FILE"
echo "Scores: $SCORES_FILE" 