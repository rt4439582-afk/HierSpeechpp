#!/usr/bin/env bash
set -euo pipefail

# One-command master pipeline for Persian TTV-v1 training.
# It validates data, runs initial/continual training, exports latest checkpoint,
# and optionally runs evaluation if a manifest is provided.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="initial"           # initial | continue
EXP_NAME="ttv_fa_v1"
STAGE_ID=""
WAV_DIR=""
TEXT_DIR=""
CLEANER="transliteration_cleaners"
DEVICE="cuda"
RUN_EVAL="0"
EVAL_MANIFEST=""
WORK_DIR="${REPO_ROOT}/runs/fa_master"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --exp_name) EXP_NAME="$2"; shift 2 ;;
    --stage_id) STAGE_ID="$2"; shift 2 ;;
    --wav_dir) WAV_DIR="$2"; shift 2 ;;
    --text_dir) TEXT_DIR="$2"; shift 2 ;;
    --cleaner) CLEANER="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --run_eval) RUN_EVAL="$2"; shift 2 ;;
    --eval_manifest) EVAL_MANIFEST="$2"; shift 2 ;;
    --work_dir) WORK_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$WAV_DIR" || -z "$TEXT_DIR" ]]; then
  echo "ERROR: --wav_dir and --text_dir are required"
  exit 1
fi

if [[ "$MODE" != "initial" && "$MODE" != "continue" ]]; then
  echo "ERROR: --mode must be initial or continue"
  exit 1
fi

if [[ "$MODE" == "continue" && -z "$STAGE_ID" ]]; then
  echo "ERROR: --stage_id is required in continue mode"
  exit 1
fi

mkdir -p "$WORK_DIR"
VALID_DIR="${WORK_DIR}/validation"
mkdir -p "$VALID_DIR"

echo "[A/6] Validate dataset"
python "${REPO_ROOT}/scripts/validate_fa_dataset.py" \
  --wav_dir "$WAV_DIR" \
  --text_dir "$TEXT_DIR" \
  --report_csv "${VALID_DIR}/dataset_report.csv" \
  --summary_json "${VALID_DIR}/dataset_summary.json"

if [[ "$MODE" == "initial" ]]; then
  echo "[B/6] Run initial fine-tuning pipeline"
  bash "${REPO_ROOT}/scripts/run_fa_ttv_finetune.sh" \
    --wav_dir "$WAV_DIR" \
    --text_dir "$TEXT_DIR" \
    --exp_name "$EXP_NAME" \
    --cleaner "$CLEANER" \
    --device "$DEVICE"
else
  echo "[B/6] Continue from latest checkpoint with new stage data"
  bash "${REPO_ROOT}/scripts/continue_fa_ttv_from_checkpoint.sh" \
    --exp_name "$EXP_NAME" \
    --stage_id "$STAGE_ID" \
    --new_wav_dir "$WAV_DIR" \
    --new_text_dir "$TEXT_DIR" \
    --cleaner "$CLEANER" \
    --device "$DEVICE"
fi

echo "[C/6] Detect latest checkpoint"
LATEST_CKPT=$(python - <<PY
import glob
from pathlib import Path
model_dir = Path(r"$REPO_ROOT") / "logs" / r"$EXP_NAME"
ckpts = sorted(glob.glob(str(model_dir / "G_*.pth")), key=lambda p: int(''.join(ch for ch in Path(p).stem if ch.isdigit())))
print(ckpts[-1] if ckpts else "")
PY
)

if [[ -z "$LATEST_CKPT" ]]; then
  echo "ERROR: no checkpoint found in logs/${EXP_NAME}"
  exit 1
fi

echo "Latest checkpoint: $LATEST_CKPT"

echo "[D/6] Write ready-to-run inference command"
NEXT_STEPS="${WORK_DIR}/next_steps.txt"
cat > "$NEXT_STEPS" <<TXT
# 1) Put your Persian text into a file, e.g. /path/input.txt
# 2) Put your prompt speaker wav, e.g. /path/prompt.wav
# 3) Run inference:
CUDA_VISIBLE_DEVICES=0 python inference.py \\
  --input_prompt /path/prompt.wav \\
  --input_txt /path/input.txt \\
  --output_dir ${WORK_DIR}/tts_outputs \\
  --ckpt logs/hierspeechpp_eng_kor/hierspeechpp_v1.1_ckpt.pth \\
  --ckpt_text2w2v ${LATEST_CKPT} \\
  --noise_scale_vc 0.333 \\
  --noise_scale_ttv 0.333 \\
  --denoise_ratio 0
TXT

if [[ "$RUN_EVAL" == "1" ]]; then
  if [[ -z "$EVAL_MANIFEST" ]]; then
    echo "ERROR: --eval_manifest is required when --run_eval 1"
    exit 1
  fi
  echo "[E/6] Run evaluation"
  python "${REPO_ROOT}/scripts/eval_tts_similarity.py" \
    --manifest_csv "$EVAL_MANIFEST" \
    --output_csv "${WORK_DIR}/eval_results.csv" \
    --output_summary_csv "${WORK_DIR}/eval_summary.csv" \
    --device "$DEVICE"
else
  echo "[E/6] Skip evaluation (--run_eval 0)"
fi

echo "[F/6] Done"
echo "Validation summary: ${VALID_DIR}/dataset_summary.json"
echo "Inference template: $NEXT_STEPS"
if [[ "$RUN_EVAL" == "1" ]]; then
  echo "Evaluation summary: ${WORK_DIR}/eval_summary.csv"
fi
