#!/usr/bin/env bash
set -euo pipefail

# Continue TTV training from the latest checkpoint with new Persian data.
# Usage:
#   bash scripts/continue_fa_ttv_from_checkpoint.sh \
#     --exp_name ttv_fa_v1 \
#     --stage_id stage02 \
#     --new_wav_dir /data/fa_stage02/wav16k \
#     --new_text_dir /data/fa_stage02/text

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_NAME=""
STAGE_ID=""
NEW_WAV_DIR=""
NEW_TEXT_DIR=""
CLEANER="transliteration_cleaners"
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp_name) EXP_NAME="$2"; shift 2 ;;
    --stage_id) STAGE_ID="$2"; shift 2 ;;
    --new_wav_dir) NEW_WAV_DIR="$2"; shift 2 ;;
    --new_text_dir) NEW_TEXT_DIR="$2"; shift 2 ;;
    --cleaner) CLEANER="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$EXP_NAME" || -z "$STAGE_ID" || -z "$NEW_WAV_DIR" || -z "$NEW_TEXT_DIR" ]]; then
  echo "ERROR: --exp_name --stage_id --new_wav_dir --new_text_dir are required"
  exit 1
fi

STAGE_ROOT="${REPO_ROOT}/datasets/fa/stages/${STAGE_ID}"
STAGE_F0_DIR="${STAGE_ROOT}/f0"
STAGE_W2V_DIR="${STAGE_ROOT}/w2v"
STAGE_TOKEN_DIR="${STAGE_ROOT}/token"
STAGE_FILELIST_DIR="${REPO_ROOT}/filelists/fa/stages/${STAGE_ID}"
CUM_DIR="${REPO_ROOT}/filelists/fa/cumulative"
CFG_PATH="${REPO_ROOT}/ttv_v1/config_fa.continual.json"

mkdir -p "$STAGE_F0_DIR" "$STAGE_W2V_DIR" "$STAGE_TOKEN_DIR" "$STAGE_FILELIST_DIR" "$CUM_DIR"

echo "[1/7] Stage=${STAGE_ID}: extract W2V"
python "${REPO_ROOT}/ttv_v1/preprocessing/extract_w2v_generic.py" \
  --input_wav_dir "$NEW_WAV_DIR" \
  --output_w2v_dir "$STAGE_W2V_DIR" \
  --device "$DEVICE"

echo "[2/7] Stage=${STAGE_ID}: extract F0"
python "${REPO_ROOT}/ttv_v1/preprocessing/extract_f0_generic.py" \
  --input_wav_dir "$NEW_WAV_DIR" \
  --output_f0_dir "$STAGE_F0_DIR"

echo "[3/7] Stage=${STAGE_ID}: extract token"
python "${REPO_ROOT}/ttv_v1/preprocessing/extract_token_generic.py" \
  --input_text_dir "$NEW_TEXT_DIR" \
  --output_token_dir "$STAGE_TOKEN_DIR" \
  --cleaner "$CLEANER"

echo "[4/7] Stage=${STAGE_ID}: build stage filelists"
python "${REPO_ROOT}/ttv_v1/preprocessing/prepare_filelist_generic.py" \
  --wav_dir "$NEW_WAV_DIR" \
  --f0_dir "$STAGE_F0_DIR" \
  --token_dir "$STAGE_TOKEN_DIR" \
  --w2v_dir "$STAGE_W2V_DIR" \
  --output_dir "$STAGE_FILELIST_DIR"

echo "[5/7] Merge stage filelists into cumulative train set"
for name in train_wav.txt train_f0.txt train_token.txt train_w2v.txt; do
  python "${REPO_ROOT}/scripts/merge_filelists.py" \
    --base "${CUM_DIR}/${name}" \
    --new "${STAGE_FILELIST_DIR}/${name}" \
    --out "${CUM_DIR}/${name}"
done

echo "[6/7] Build continual config"
python - <<PY
import json
from pathlib import Path
repo = Path(r"$REPO_ROOT")
with open(repo / "ttv_v1/config_fa_template.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)
cfg["data"]["train_filelist_path"] = "filelists/fa/cumulative/train_wav.txt"
cfg["data"]["test_filelist_path"] = "filelists/fa/cumulative/train_wav.txt"
with open(repo / "ttv_v1/config_fa.continual.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
print("Saved", repo / "ttv_v1/config_fa.continual.json")
PY

echo "[7/7] Continue training from latest checkpoint in logs/${EXP_NAME}"
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES=0 python train_ttv_v1.py -c "$CFG_PATH" -m "$EXP_NAME"
