#!/usr/bin/env bash
# 复现 exp4_2（论文对应实验，预期数值见 config/experiments/exp4_2.yaml: expected）
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m eval.run_eval --config config/experiments/exp4_2.yaml "$@"
