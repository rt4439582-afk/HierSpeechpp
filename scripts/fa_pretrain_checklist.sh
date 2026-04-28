#!/usr/bin/env bash
set -euo pipefail

# Quick pre-train checklist for Persian dataset quality gates.
# Usage:
#   bash scripts/fa_pretrain_checklist.sh --wav_dir ... --text_dir ... --out_dir ...

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAV_DIR=""
TEXT_DIR=""
OUT_DIR="${REPO_ROOT}/runs/fa_checklist"
MIN_PAIRS=100

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wav_dir) WAV_DIR="$2"; shift 2 ;;
    --text_dir) TEXT_DIR="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --min_pairs) MIN_PAIRS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$WAV_DIR" || -z "$TEXT_DIR" ]]; then
  echo "ERROR: --wav_dir and --text_dir are required"
  exit 1
fi

mkdir -p "$OUT_DIR"
REPORT_CSV="$OUT_DIR/dataset_report.csv"
SUMMARY_JSON="$OUT_DIR/dataset_summary.json"

echo "[1/3] Running structural validation"
python "$REPO_ROOT/scripts/validate_fa_dataset.py" \
  --wav_dir "$WAV_DIR" \
  --text_dir "$TEXT_DIR" \
  --report_csv "$REPORT_CSV" \
  --summary_json "$SUMMARY_JSON"

echo "[2/3] Running quality gates"
python - <<PY
import csv, json, statistics, sys
from pathlib import Path
summary = json.loads(Path(r"$SUMMARY_JSON").read_text(encoding='utf-8'))
rows = list(csv.DictReader(Path(r"$REPORT_CSV").open(encoding='utf-8')))
num_pairs = summary['num_pairs']
text_lens = [int(r['text_len']) for r in rows]
secs = [float(r['duration_sec']) for r in rows]

issues = []
if num_pairs < int($MIN_PAIRS):
    issues.append(f"num_pairs ({num_pairs}) < min_pairs ({int($MIN_PAIRS)})")
if statistics.median(text_lens) < 10:
    issues.append(f"median text length too low: {statistics.median(text_lens):.1f}")
if statistics.median(secs) < 2.0:
    issues.append(f"median duration too short: {statistics.median(secs):.2f}s")

print("Summary:")
print(json.dumps({
    'num_pairs': num_pairs,
    'median_text_len': round(statistics.median(text_lens),2),
    'median_duration_sec': round(statistics.median(secs),2)
}, ensure_ascii=False, indent=2))

if issues:
    print("\nChecklist FAILED:")
    for x in issues:
        print("-", x)
    sys.exit(2)

print("\nChecklist PASSED")
PY

echo "[3/3] Done. Reports:"
echo "- $REPORT_CSV"
echo "- $SUMMARY_JSON"
