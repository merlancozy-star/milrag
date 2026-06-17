#!/usr/bin/env python3
"""下载并筛选中文维基百科军事类目文章。

1. 下载中文维基百科 dump（如未下载）
2. 用 wikiextractor 抽取文本
3. 按军事类目关键词过滤
4. 写入 data/raw/zhwiki_military/

需要: pip install wikiextractor
Dump 大小: ~2.5GB 压缩 / ~8GB 解压后
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve


WIKI_DUMP_URL = "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"
OUTPUT_DIR = Path("data/raw/zhwiki_military")

# 军事类目关键词（和 DATA_BUILD_GUIDE.md 保持一致）
MILITARY_KEYWORDS = [
    "军事", "武器", "装备", "战争", "军队", "国防", "战略",
    "战役", "战术", "军种", "兵种", "导弹", "舰艇", "战机",
    "坦克", "雷达", "情报", "特种部队", "军事演习", "军事条约",
    "军事历史", "军事技术", "军事人物", "军事基地",
    "作战", "火炮", "核武器", "生化武器", "装甲", "潜艇",
    "航空母舰", "驱逐舰", "护卫舰", "空军", "海军", "陆军",
    "火箭军", "战略支援", "后勤保障", "军事通信", "电子战",
    "网络战", "太空战", "弹药", "枪械", "步兵", "炮兵",
    "工程兵", "防化兵", "通信兵", "侦察", "机动作战",
    "防空", "反导", "两栖作战", "空降作战", "特种作战",
    "军事法学", "军事医学", "军事地理", "军事气象",
    "军事运输", "军事建筑", "军事院校", "军事训练",
    "征兵", "军衔", "勋章", "军服", "军费", "军购",
    "无人机", "巡飞弹", "高超音速", "激光武器", "电磁炮",
]


def download_dump(target: Path, force: bool = False) -> Path:
    """下载维基百科 dump（支持断点续传）。"""
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        print(f"  Dump 已存在: {target}")
        return target

    print(f"  下载: {WIKI_DUMP_URL}")
    print(f"  目标: {target}")

    # 尝试用 aria2c 加速，回退到 urllib
    try:
        subprocess.run([
            "aria2c", "--continue=true", "--max-connection-per-server=4",
            "--dir", str(target.parent), "--out", target.name,
            WIKI_DUMP_URL,
        ], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  aria2c 不可用，使用 Python urlretrieve（较慢）...")
        urlretrieve(WIKI_DUMP_URL, target)

    return target


def extract_with_wikiextractor(dump_path: Path, extract_dir: Path, force: bool = False):
    """用 wikiextractor 抽取文本。"""
    extract_dir.mkdir(parents=True, exist_ok=True)

    existing = list(extract_dir.glob("**/wiki_*"))
    if existing and not force:
        print(f"  已抽取 {len(existing)} 个文件，跳过")
        return

    print(f"  抽取文本到 {extract_dir}...")
    subprocess.run([
        sys.executable, "-m", "wikiextractor.WikiExtractor",
        str(dump_path),
        "--output", str(extract_dir),
        "--bytes", "1M",
        "--json",
        "--quiet",
    ], check=True)


def is_military_article(text: str) -> bool:
    """判断文本是否为军事相关文章。前 500 字命中 ≥2 个关键词。"""
    head = text[:500]
    score = sum(1 for kw in MILITARY_KEYWORDS if kw in head)
    return score >= 2


def filter_military(extract_dir: Path, output_dir: Path):
    """筛选军事文章并输出。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "checkpoint.json"

    done_dirs = set()
    if checkpoint_file.exists():
        done_dirs = set(json.loads(checkpoint_file.read_text()).get("done_dirs", []))

    count = 0
    error_log = []
    subdirs = [d for d in sorted(extract_dir.iterdir()) if d.is_dir()]

    for subdir in subdirs:
        subdir_name = subdir.name
        if subdir_name in done_dirs:
            continue

        for wiki_file in sorted(subdir.glob("*")):
            if wiki_file.suffix:
                continue  # wikiextractor 输出文件无后缀

            for line in wiki_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = doc.get("text", "")
                if not text.strip():
                    continue

                if is_military_article(text):
                    doc_id = doc.get("id", str(count))
                    title = doc.get("title", "")

                    # 写文本
                    txt_path = output_dir / f"zhwiki_{doc_id}.txt"
                    txt_path.write_text(text, encoding="utf-8")

                    # 写元信息
                    meta = {
                        "source_url": f"https://zh.wikipedia.org/wiki/{title}",
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                        "source_category": "military_news",
                        "content_category": _guess_content_category(text),
                        "authority": "mainstream_media",
                        "language": "zh",
                        "desensitized": False,
                        "title": title,
                        "document_id": f"zhwiki_{doc_id}",
                    }
                    meta_path = output_dir / f"zhwiki_{doc_id}.meta.json"
                    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

                    count += 1
                    if count % 1000 == 0:
                        print(f"  已筛选 {count} 篇军事文章...")

        done_dirs.add(subdir_name)
        checkpoint_file.write_text(json.dumps({"done_dirs": list(done_dirs)}, ensure_ascii=False))

        # 达到目标量就停
        if count >= 60000:
            print(f"  已达标 ({count} 篇)，停止筛选")
            break

    # 记录错误
    if error_log:
        err_path = output_dir / "errors.log"
        err_path.write_text("\n".join(error_log), encoding="utf-8")

    return count


def _guess_content_category(text: str) -> str:
    text_lower = text.lower()
    equip = sum(1 for kw in ["战斗机", "驱逐舰", "坦克", "导弹", "雷达", "潜艇", "航母", "口径", "射程"] if kw in text_lower)
    doctrine = sum(1 for kw in ["条令", "作战", "原则", "兵力", "指挥", "战术", "编制"] if kw in text_lower)
    case = sum(1 for kw in ["战役", "战争", "冲突", "演习", "历史", "战例"] if kw in text_lower)
    scores = {"equipment": equip, "doctrine": doctrine, "case": case}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "situation"


def main():
    parser = argparse.ArgumentParser(description="中文维基百科军事文章下载")
    parser.add_argument("--dump-path", default="data/raw/zhwiki_dump.xml.bz2",
                        help="维基 dump 路径")
    parser.add_argument("--extract-dir", default="data/raw/zhwiki_extracted",
                        help="wikiextractor 输出目录")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过下载（已有 dump）")
    parser.add_argument("--force", action="store_true",
                        help="强制重新处理")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅测试连接和解析，不实际写入")
    args = parser.parse_args()

    start = time.time()

    print("中文维基百科军事文章下载")
    print(f"  Dump URL: {WIKI_DUMP_URL}")

    if args.dry_run:
        print("[DRY RUN] 测试 Wikipedia dump URL 可达性...")
        try:
            import urllib.request
            req = urllib.request.Request(WIKI_DUMP_URL, method="HEAD")
            req.add_header("User-Agent", "milrag-corpus-builder/1.0 (academic research)")
            resp = urllib.request.urlopen(req, timeout=30)
            print(f"  ✅ URL 可达 (HTTP {resp.status})")
        except Exception as e:
            print(f"  ❌ URL 不可达: {e}")
        return

    # 1. 下载
    dump_path = Path(args.dump_path)
    if not args.skip_download:
        download_dump(dump_path, force=args.force)
    elif not dump_path.exists():
        print(f"  错误: dump 文件不存在 {dump_path}，请去掉 --skip-download")
        raise SystemExit(1)

    # 2. 抽取
    extract_dir = Path(args.extract_dir)
    extract_with_wikiextractor(dump_path, extract_dir, force=args.force)

    # 3. 筛选
    output_dir = Path(args.output)
    count = filter_military(extract_dir, output_dir)

    elapsed = time.time() - start
    print(f"✅ 完成: {count} 篇军事文章 → {output_dir}")
    print(f"   耗时: {elapsed / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
