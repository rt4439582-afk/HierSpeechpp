#!/usr/bin/env bash
set -euo pipefail

cat <<'TXT'
# 1) Setup workspace
cd /workspace/HierSpeechpp
bash scripts/setup_fa_workspace.sh --workspace /workspace/HierSpeechpp/fa_workspace

# 2) Put data in:
# /workspace/HierSpeechpp/fa_workspace/data/stage01/wav16k
# /workspace/HierSpeechpp/fa_workspace/data/stage01/text

# 3) Train stage01
bash /workspace/HierSpeechpp/fa_workspace/scripts/run_stage01.sh

# 4) Continue stage02
bash /workspace/HierSpeechpp/fa_workspace/scripts/run_stage02.sh

# 5) Continue stage03
bash /workspace/HierSpeechpp/fa_workspace/scripts/run_stage03.sh
TXT
