#!/usr/bin/env bash
# 按论文顺序全量复现。建议分章跑，先确认基线对齐再往后。
set -euo pipefail
for s in run_exp3_*.sh run_exp4_*.sh run_exp5_*.sh run_exp6_*.sh; do
  echo "==== $s ===="; bash "$s" || echo "[FAIL] $s"; done
