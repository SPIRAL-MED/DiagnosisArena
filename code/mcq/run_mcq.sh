#!/bin/bash

# ── config ────────────────────────────────────────────────────────────────────
INPUT_PATH="shzyk/DiagnosisArena"
OUTPUT_ROOT="./results"

MODEL_NAME="gpt-4o"
API_KEY="YOUR_API_KEY"
BASE_URL="YOUR_BASE_URL"

FOLK_NUMS=16
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Step 1: Inference (MCQ) ==="
python "$SCRIPT_DIR/inference_mcq.py" \
    --input_path "$INPUT_PATH" \
    --output_root "$OUTPUT_ROOT" \
    --model_name "$MODEL_NAME" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --folk_nums "$FOLK_NUMS"

echo "=== Step 2: Metric (MCQ) ==="
python "$SCRIPT_DIR/metric_mcq.py" \
    --model_name "$MODEL_NAME" \
    --metric_path "$OUTPUT_ROOT/${MODEL_NAME}_answer.jsonl"
