#!/bin/bash

# ── config ────────────────────────────────────────────────────────────────────
INPUT_PATH="shzyk/DiagnosisArena"
OUTPUT_ROOT="/nas/zhuyakun/projects/202605_diagnosisarena_leagerboard/data/results_now"

MODEL_NAME="DeepSeek-V4-Flash"
MODEL_API_KEY="sk-1234"
MODEL_BASE_URL="http://127.0.0.1:4000"

JUDGE_MODEL="DeepSeek-V4-Flash"
JUDGE_API_KEY="sk-1234"
JUDGE_BASE_URL="http://127.0.0.1:4000"

FOLK_NUMS=32
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
