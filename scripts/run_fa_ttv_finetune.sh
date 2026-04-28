#!/usr/bin/env bash
set -euo pipefail

# End-to-end Persian fine-tuning pipeline for HierSpeech++ TTV-v1
# Usage:
#   bash scripts/run_fa_ttv_finetune.sh \
#     --wav_dir /path/to/wav16k \
#     --text_dir /path/to/text \
#     --exp_name ttv_fa_v1 \
#     --cleaner transliteration_cleaners

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WAV_DIR=""
TEXT_DIR=""
EXP_NAME="ttv_fa_v1"
CLEANER="transliteration_cleaners"
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wav_dir) WAV_DIR="$2"; shift 2 ;;
    --text_dir) TEXT_DIR="$2"; shift 2 ;;
    --exp_name) EXP_NAME="$2"; shift 2 ;;
    --cleaner) CLEANER="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$WAV_DIR" || -z "$TEXT_DIR" ]]; then
  echo "ERROR: --wav_dir and --text_dir are required"
  exit 1
fi

DATA_ROOT="${REPO_ROOT}/datasets/fa"
F0_DIR="${DATA_ROOT}/f0"
W2V_DIR="${DATA_ROOT}/w2v"
TOKEN_DIR="${DATA_ROOT}/token"
FILELIST_DIR="${REPO_ROOT}/filelists/fa"
CFG_PATH="${REPO_ROOT}/ttv_v1/config_fa.auto.json"

mkdir -p "$F0_DIR" "$W2V_DIR" "$TOKEN_DIR" "$FILELIST_DIR"

echo "[1/6] Extracting W2V"
python "${REPO_ROOT}/ttv_v1/preprocessing/extract_w2v_generic.py" \
  --input_wav_dir "$WAV_DIR" \
  --output_w2v_dir "$W2V_DIR" \
  --device "$DEVICE"

echo "[2/6] Extracting F0"
python "${REPO_ROOT}/ttv_v1/preprocessing/extract_f0_generic.py" \
  --input_wav_dir "$WAV_DIR" \
  --output_f0_dir "$F0_DIR"

echo "[3/6] Extracting text tokens"
python "${REPO_ROOT}/ttv_v1/preprocessing/extract_token_generic.py" \
  --input_text_dir "$TEXT_DIR" \
  --output_token_dir "$TOKEN_DIR" \
  --cleaner "$CLEANER"

echo "[4/6] Building filelists"
python "${REPO_ROOT}/ttv_v1/preprocessing/prepare_filelist_generic.py" \
  --wav_dir "$WAV_DIR" \
  --f0_dir "$F0_DIR" \
  --token_dir "$TOKEN_DIR" \
  --w2v_dir "$W2V_DIR" \
  --output_dir "$FILELIST_DIR"

echo "[5/6] Creating config at $CFG_PATH"
python - <<PY
import json
from pathlib import Path
repo = Path(r"$REPO_ROOT")
with open(repo / "ttv_v1/config_fa_template.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)
cfg["data"]["train_filelist_path"] = "filelists/fa/train_wav.txt"
cfg["data"]["test_filelist_path"] = "filelists/fa/train_wav.txt"
with open(Path(r"$CFG_PATH"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
print("Saved", Path(r"$CFG_PATH"))
PY

echo "[6/6] Training"
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES=0 python train_ttv_v1.py -c "$CFG_PATH" -m "$EXP_NAME"
