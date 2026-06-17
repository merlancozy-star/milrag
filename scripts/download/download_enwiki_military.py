#!/usr/bin/env python3
"""下载并筛选英文维基百科军事类目文章。

流程同 download_zhwiki_military.py，但针对英文维基。
英文 dump 更大 (~20GB)，需要更多磁盘空间和下载时间。

保持英文原文不翻译；文章标记 language: en。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve


WIKI_DUMP_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
OUTPUT_DIR = Path("data/raw/enwiki_military")

# 英文军事关键词
MILITARY_KEYWORDS = [
    "military", "weapon", "equipment", "warfare", "army", "navy", "air force",
    "defense", "defence", "strategy", "tactics", "doctrine", "operation",
    "missile", "aircraft", "submarine", "tank", "radar", "intelligence",
    "special forces", "military exercise", "military history", "battle",
    "brigade", "division", "corps", "battalion", "regiment", "squadron",
    "artillery", "armor", "armoured", "infantry", "cavalry", "airborne",
    "amphibious", "logistics", "command", "reconnaissance", "surveillance",
    "electronic warfare", "cyber warfare", "nuclear weapon", "ballistic",
    "cruise missile", "fighter jet", "bomber", "helicopter", "drone",
    "frigate", "destroyer", "cruiser", "carrier", "amphibious assault",
    "marine corps", "special operations", "counterinsurgency", "peacekeeping",
    "arms control", "disarmament", "mobilization", "demobilization",
    "military police", "military engineering", "military medicine",
    "military communications", "military satellite", "military base",
    "naval base", "air base", "garrison", "fortification",
    "joint publication", "field manual", "operational art",
    "center of gravity", "decisive point", "mission command",
]


def download_dump(target: Path, force: bool = False) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        print(f"  Dump 已存在: {target}")
        return target

    print(f"  下载: {WIKI_DUMP_URL} (约 20GB，需要较长时间)")
    print(f"  目标: {target}")

    try:
        subprocess.run([
            "aria2c", "--continue=true", "--max-connection-per-server=8",
            "--dir", str(target.parent), "--out", target.name,
            WIKI_DUMP_URL,
        ], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  aria2c 不可用，使用 Python urlretrieve（很慢）...")
        urlretrieve(WIKI_DUMP_URL, target)

    return target


def extract_with_wikiextractor(dump_path: Path, extract_dir: Path, force: bool = False):
    extract_dir.mkdir(parents=True, exist_ok=True)

    existing = list(extract_dir.glob("**/wiki_*"))
    if existing and not force:
        print(f"  已抽取 {len(existing)} 个文件，跳过")
        return

    print(f"  抽取文本到 {extract_dir}（需要较长时间）...")
    subprocess.run([
        sys.executable, "-m", "wikiextractor.WikiExtractor",
        str(dump_path),
        "--output", str(extract_dir),
        "--bytes", "5M",
        "--json",
        "--quiet",
    ], check=True)


def is_military_article(text: str) -> bool:
    head = text[:800].lower()
    score = sum(1 for kw in MILITARY_KEYWORDS if kw.lower() in head)
    return score >= 2


def filter_military(extract_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "checkpoint.json"

    done_dirs = set()
    if checkpoint_file.exists():
        done_dirs = set(json.loads(checkpoint_file.read_text()).get("done_dirs", []))

    count = 0
    subdirs = [d for d in sorted(extract_dir.iterdir()) if d.is_dir()]

    for subdir in subdirs:
        subdir_name = subdir.name
        if subdir_name in done_dirs:
            continue

        for wiki_file in sorted(subdir.glob("*")):
            if wiki_file.suffix:
                continue

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

                    txt_path = output_dir / f"enwiki_{doc_id}.txt"
                    txt_path.write_text(text, encoding="utf-8")

                    meta = {
                        "source_url": f"https://en.wikipedia.org/wiki/{title}",
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                        "source_category": "military_news",
                        "content_category": _guess_content_category_en(text),
                        "authority": "mainstream_media",
                        "language": "en",
                        "desensitized": False,
                        "title": title,
                        "document_id": f"enwiki_{doc_id}",
                    }
                    meta_path = output_dir / f"enwiki_{doc_id}.meta.json"
                    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

                    count += 1
                    if count % 1000 == 0:
                        print(f"  已筛选 {count} 篇军事文章...")

        done_dirs.add(subdir_name)
        checkpoint_file.write_text(json.dumps({"done_dirs": list(done_dirs)}, ensure_ascii=False))

        if count >= 50000:
            print(f"  已达标 ({count} 篇)，停止筛选")
            break

    return count


def _guess_content_category_en(text: str) -> str:
    t = text.lower()[:2000]
    equip = sum(1 for kw in ["fighter", "destroyer", "tank", "missile", "radar", "submarine",
                              "carrier", "caliber", "range", "aircraft", "weapon system",
                              "armored vehicle", "artillery"] if kw in t)
    doctrine = sum(1 for kw in ["doctrine", "operations", "command and control",
                                 "field manual", "joint publication", "tactics",
                                 "principles of war"] if kw in t)
    case = sum(1 for kw in ["battle", "campaign", "war", "conflict", "operation",
                             "exercise", "case study", "lessons learned"] if kw in t)
    scores = {"equipment": equip, "doctrine": doctrine, "case": case}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "situation"


def main():
    parser = argparse.ArgumentParser(description="英文维基百科军事文章下载")
    parser.add_argument("--dump-path", default="data/raw/enwiki_dump.xml.bz2",
                        help="维基 dump 路径")
    parser.add_argument("--extract-dir", default="data/raw/enwiki_extracted",
                        help="wikiextractor 输出目录")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start = time.time()

    print("英文维基百科军事文章下载")
    print(f"  Dump URL: {WIKI_DUMP_URL}")
    print(f"  ⚠️ Dump 约 20GB，下载需要较长时间和充足磁盘空间")

    if args.dry_run:
        print("[DRY RUN] 测试 URL 可达性...")
        try:
            import urllib.request
            req = urllib.request.Request(WIKI_DUMP_URL, method="HEAD")
            req.add_header("User-Agent", "milrag-corpus-builder/1.0 (academic research)")
            resp = urllib.request.urlopen(req, timeout=30)
            print(f"  ✅ URL 可达 (HTTP {resp.status})")
        except Exception as e:
            print(f"  ❌ URL 不可达: {e}")
        return

    dump_path = Path(args.dump_path)
    if not args.skip_download:
        download_dump(dump_path, force=args.force)
    elif not dump_path.exists():
        print(f"  错误: dump 文件不存在 {dump_path}")
        raise SystemExit(1)

    extract_dir = Path(args.extract_dir)
    extract_with_wikiextractor(dump_path, extract_dir, force=args.force)

    output_dir = Path(args.output)
    count = filter_military(extract_dir, output_dir)

    elapsed = time.time() - start
    print(f"✅ 完成: {count} 篇军事文章 → {output_dir}")
    print(f"   耗时: {elapsed / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
