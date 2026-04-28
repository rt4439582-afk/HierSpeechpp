#!/usr/bin/env bash
set -euo pipefail

# Create a single, ready-to-use Persian training workspace under one path.
# Usage:
#   bash scripts/setup_fa_workspace.sh --workspace /workspace/HierSpeechpp/fa_workspace

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${REPO_ROOT}/fa_workspace"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$WORKSPACE"/{data,manifests,reports,outputs,scripts}
mkdir -p "$WORKSPACE/data"/{stage01,stage02,stage03}
mkdir -p "$WORKSPACE/data/stage01"/{wav16k,text}
mkdir -p "$WORKSPACE/data/stage02"/{wav16k,text}
mkdir -p "$WORKSPACE/data/stage03"/{wav16k,text}

cat > "$WORKSPACE/scripts/run_stage01.sh" <<SH
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"

bash scripts/run_fa_master_pipeline.sh \\
  --mode initial \\
  --exp_name ttv_fa_v1 \\
  --wav_dir "$WORKSPACE/data/stage01/wav16k" \\
  --text_dir "$WORKSPACE/data/stage01/text" \\
  --cleaner transliteration_cleaners \\
  --device cuda \\
  --work_dir "$WORKSPACE/reports/stage01" \\
  --run_eval 0
SH
chmod +x "$WORKSPACE/scripts/run_stage01.sh"

cat > "$WORKSPACE/scripts/run_stage02.sh" <<SH
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"

bash scripts/run_fa_master_pipeline.sh \\
  --mode continue \\
  --exp_name ttv_fa_v1 \\
  --stage_id stage02 \\
  --wav_dir "$WORKSPACE/data/stage02/wav16k" \\
  --text_dir "$WORKSPACE/data/stage02/text" \\
  --cleaner transliteration_cleaners \\
  --device cuda \\
  --work_dir "$WORKSPACE/reports/stage02" \\
  --run_eval 0
SH
chmod +x "$WORKSPACE/scripts/run_stage02.sh"

cat > "$WORKSPACE/scripts/run_stage03.sh" <<SH
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"

bash scripts/run_fa_master_pipeline.sh \\
  --mode continue \\
  --exp_name ttv_fa_v1 \\
  --stage_id stage03 \\
  --wav_dir "$WORKSPACE/data/stage03/wav16k" \\
  --text_dir "$WORKSPACE/data/stage03/text" \\
  --cleaner transliteration_cleaners \\
  --device cuda \\
  --work_dir "$WORKSPACE/reports/stage03" \\
  --run_eval 0
SH
chmod +x "$WORKSPACE/scripts/run_stage03.sh"

cat > "$WORKSPACE/README_FA_WORKSPACE.md" <<TXT
# Persian Workspace (Single Path)

## 1) Put your data
- stage01 wav: $WORKSPACE/data/stage01/wav16k
- stage01 txt: $WORKSPACE/data/stage01/text

## 2) Run
- Initial training: bash $WORKSPACE/scripts/run_stage01.sh
- Continue stage02: bash $WORKSPACE/scripts/run_stage02.sh
- Continue stage03: bash $WORKSPACE/scripts/run_stage03.sh

## 3) Reports
- stage outputs & validation: $WORKSPACE/reports/stage01 , stage02 , stage03

TXT

echo "Workspace ready: $WORKSPACE"
