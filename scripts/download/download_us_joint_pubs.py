#!/usr/bin/env python3
"""下载 US Joint Publications (JPs)。

来源: Joint Chiefs of Staff (jcs.mil) / Joint Electronic Library
这些是公开批准发布的美军联合出版物，涵盖联合作战理论。

输出: data/raw/us_joint_pubs/（英文原文，不翻译）
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


OUTPUT_DIR = Path("data/raw/us_joint_pubs")
USER_AGENT = "milrag-corpus-builder/1.0 (academic research; contact: see project README)"

# 公开可获取的 Joint Publications
JP_LIST = [
    {"id": "jp_1", "title": "JP 1: Doctrine for the Armed Forces of the United States",
     "url": "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp1_0.pdf"},
    {"id": "jp_2_0", "title": "JP 2-0: Joint Intelligence",
     "url": "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp2_0.pdf"},
    {"id": "jp_3_0", "title": "JP 3-0: Joint Operations",
     "url": "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_0.pdf"},
    {"id": "jp_3_09", "title": "JP 3-09: Joint Fire Support",
     "url": "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_09.pdf"},
    {"id": "jp_3_12", "title": "JP 3-12: Cyberspace Operations",
     "url": "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_12.pdf"},
    {"id": "jp_3_13", "title": "JP 3-13: Information Operations",
     "url": "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_13.pdf"},
    {"id": "jp_3_14", "title": "JP 3-14: Space Operations",
     "url": "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_14.pdf"},
    {"id": "jp_3_31", "title": "JP 3-31: Joint Land Operations",
     "url": "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_31.pdf"},
    {"id": "jp_3_32", "title": "JP 3-32: Joint Maritime Operations",
     "url": "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_32.pdf"},
    {"id": "jp_3_33", "title": "JP 3-33: Joint Task Force Headquarters",
     "url": "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp3_33.pdf"},
    {"id": "jp_4_0", "title": "JP 4-0: Joint Logistics",
     "url": "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp4_0.pdf"},
    {"id": "jp_5_0", "title": "JP 5-0: Joint Planning",
     "url": "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp5_0.pdf"},
    {"id": "jp_6_0", "title": "JP 6-0: Joint Communications System",
     "url": "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp6_0.pdf"},
]

# FAS 备用镜像
FALLBACK_BASE = "https://irp.fas.org/doddir/dod/"


def fetch_pdf(url: str, max_retries: int = 3) -> bytes | None:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    下载失败: {url} — {e}")
                return None
            time.sleep(10)
    return None


def extract_text_pdf(pdf_bytes: bytes) -> str:
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        if len(text.strip()) > 500:
            return text
    except ImportError:
        pass

    try:
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n\n".join(
                (page.extract_text() or "") for page in pdf.pages
            )
        if len(text.strip()) > 500:
            return text
    except ImportError:
        pass

    raise ImportError("需要安装 pymupdf 或 pdfplumber")


def _guess_content_category_en(text: str) -> str:
    t = text[:3000].lower()
    equip = sum(1 for kw in ["weapon", "fire support", "munition", "targeting"] if kw in t)
    doctrine = sum(1 for kw in ["doctrine", "joint operation", "command", "planning",
                                 "strategy", "campaign"] if kw in t)
    case = sum(1 for kw in ["case", "historical", "lesson", "scenario"] if kw in t)
    scores = {"equipment": equip, "doctrine": doctrine, "case": case}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "doctrine"


def main():
    parser = argparse.ArgumentParser(description="US Joint Publications 下载")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "checkpoint.json"

    if checkpoint_file.exists():
        cp = json.loads(checkpoint_file.read_text())
        done_ids = set(cp.get("done_ids", []))
    else:
        done_ids = set()

    print("US Joint Publications 下载")
    print(f"  出版物数: {len(JP_LIST)}")

    if args.dry_run:
        print("\n[DRY RUN] 检查 URL 可达性...")
        for jp in JP_LIST[:5]:
            try:
                req = Request(jp["url"], headers={"User-Agent": USER_AGENT}, method="HEAD")
                resp = urlopen(req, timeout=15)
                print(f"  {'✅' if resp.status == 200 else '❌'} {jp['id']}: {jp['title']}")
            except Exception as e:
                print(f"  ❌ {jp['id']}: {e}")
        return

    total = 0
    errors: list[str] = []

    for jp in JP_LIST:
        jp_id = jp["id"]
        if jp_id in done_ids:
            continue

        print(f"  {jp_id}: {jp['title']}")

        pdf_bytes = fetch_pdf(jp["url"])
        if not pdf_bytes:
            # 尝试 FAS 备用
            fallback_url = f"{FALLBACK_BASE}{jp_id}.pdf"
            print(f"    尝试备用: {fallback_url}")
            pdf_bytes = fetch_pdf(fallback_url)
            if not pdf_bytes:
                errors.append(f"{jp_id}: 所有 URL 均失败")
                continue

        try:
            text = extract_text_pdf(pdf_bytes)
        except ImportError:
            print("    错误: 需要安装 PDF 文本提取库")
            raise SystemExit(1)

        if len(text.strip()) < 500:
            errors.append(f"{jp_id}: PDF 文本过短")
            continue

        txt_path = output_dir / f"{jp_id}.txt"
        txt_path.write_text(text, encoding="utf-8")

        meta = {
            "source_url": jp["url"],
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "source_category": "doctrine",
            "content_category": _guess_content_category_en(text),
            "authority": "doctrine",
            "language": "en",
            "desensitized": True,
            "title": jp["title"],
            "document_id": jp_id,
        }
        meta_path = output_dir / f"{jp_id}.meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        done_ids.add(jp_id)
        total += 1
        time.sleep(3)

    checkpoint_file.write_text(
        json.dumps({"done_ids": list(done_ids)}, ensure_ascii=False)
    )
    if errors:
        (output_dir / "errors.log").write_text("\n".join(errors), encoding="utf-8")

    print(f"✅ 完成: {total} 份 Joint Publication → {output_dir}")


if __name__ == "__main__":
    main()
