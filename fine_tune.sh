#!/usr/bin/env bash
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: ./fine_tune.sh [data_dir] [model_name]"
    echo "Example: ./fine_tune.sh ./my_data my_model"
    exit 1
fi

DATA_DIR="$1"
MODEL_NAME="${2:-fine_tuned_model}"

echo "========================================"
echo "Starting Fine-tuning"
echo "========================================"
echo "Data dir  : $DATA_DIR"
echo "Model name: $MODEL_NAME"
echo "========================================"

python fine_tune.py --data_dir "$DATA_DIR" --model_name "$MODEL_NAME"
