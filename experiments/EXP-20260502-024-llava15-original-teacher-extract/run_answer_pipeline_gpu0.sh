#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_answer_reextract_retrain_gpu0.sh"
"${SCRIPT_DIR}/run_chartqa_zap_answer_keep02_gpu0.sh"
