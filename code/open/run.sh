#!/bin/bash

# ── config ────────────────────────────────────────────────────────────────────
INPUT_PATH="shzyk/DiagnosisArena"
OUTPUT_ROOT="./results"

MODEL_NAME="gpt-4o"
MODEL_API_KEY="YOUR_MODEL_API_KEY"
MODEL_BASE_URL="YOUR_MODEL_BASE_URL"

JUDGE_MODEL="gpt-4o"
JUDGE_API_KEY="YOUR_JUDGE_API_KEY"
JUDGE_BASE_URL="YOUR_JUDGE_BASE_URL"

FOLK_NUMS=16
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Step 1: Inference ==="
python "$SCRIPT_DIR/inference.py" \
    --input_path "$INPUT_PATH" \
    --output_root "$OUTPUT_ROOT" \
    --model_name "$MODEL_NAME" \
    --api_key "$MODEL_API_KEY" \
    --base_url "$MODEL_BASE_URL" \
    --folk_nums "$FOLK_NUMS"

echo "=== Step 2: Evaluation ==="
python "$SCRIPT_DIR/evaluation.py" \
    --input_path "$OUTPUT_ROOT/${MODEL_NAME}_answer.jsonl" \
    --judge_model "$JUDGE_MODEL" \
    --api_key "$JUDGE_API_KEY" \
    --base_url "$JUDGE_BASE_URL" \
    --folk_nums "$FOLK_NUMS"

echo "=== Step 3: Metric ==="
python "$SCRIPT_DIR/metric.py" \
    --model_name "$MODEL_NAME" \
    --metric_path "$OUTPUT_ROOT/${MODEL_NAME}_answer_${JUDGE_MODEL}_evaled.jsonl"
