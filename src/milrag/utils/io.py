"""utils/io.py — 实验产出落盘（CLAUDE.md §9.4：记录 git hash + config 快照 + 种子）。"""
from __future__ import annotations
import json, subprocess, time
from pathlib import Path
def git_hash() -> str:
    try: return subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
    except Exception: return "unknown"
def save_run(exp_id: str, cfg: dict, results: dict, root="experiments") -> Path:
    out = Path(root)/exp_id/time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
    (out/"config_snapshot.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    (out/"results.json").write_text(json.dumps({"git": git_hash(), **results}, ensure_ascii=False, indent=2))
    return out
