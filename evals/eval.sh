#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LMMS_DIR="$ROOT_DIR/modules/lmms-eval"

# Eval parameters
TASKS="fgvqa"  # for a subset, set TASKS="fgvqa_subset" (e.g. fgvqa_cub, fgvqa_inquire etc.)
MODEL_DIR="qwen2_5_vl"
MODEL_ARGS="pretrained=glab-caltech/TWIN-Qwen2.5-VL-3B"
BATCH_SIZE=1
OUTPUT_DIR="$ROOT_DIR/evals/eval_results" 

cd "$LMMS_DIR"

uv run python -m lmms_eval \
  --tasks "$TASKS" \
  --model "$MODEL_DIR" \
  --model_args "$MODEL_ARGS" \
  --batch_size "$BATCH_SIZE" \
  --output_path "$OUTPUT_DIR" \
  --log_samples