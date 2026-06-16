#!/usr/bin/env bash
# 复现 exp6_6（论文对应实验，预期数值见 config/experiments/exp6_6.yaml: expected）
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m eval.run_eval --config config/experiments/exp6_6.yaml "$@"
