#!/usr/bin/env python3
"""下载 US Army Field Manuals (FM) 和 Army Techniques Publications (ATP)。

来源: Army Publishing Directorate (armypubs.army.mil)
这些是正式公开批准发布的美军野战手册，无保密等级。

输出: data/raw/us_army_fm/（英文原文，不翻译）
双语KB：英文条令与中文 query 通过 Qwen3-Embedding 跨语言检索匹配。
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve


OUTPUT_DIR = Path("data/raw/us_army_fm")
USER_AGENT = "milrag-corpus-builder/1.0 (academic research; contact: see project README)"

# 公开可获取的 US Army Field Manuals 和 ATPs
# 所有列出的出版物均为公开批准发布 (Approved for public release)
FM_LIST = [
    # ADP (Army Doctrine Publications)
    {"id": "adp_1", "title": "ADP 1: The Army", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38008-ADP_1-000-WEB-4.pdf"},
    {"id": "adp_3_0", "title": "ADP 3-0: Operations", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN37509-ADP_3-0-000-WEB-5.pdf"},
    {"id": "adp_4_0", "title": "ADP 4-0: Sustainment", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38005-ADP_4-0-000-WEB-4.pdf"},
    {"id": "adp_5_0", "title": "ADP 5-0: The Operations Process", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38006-ADP_5-0-000-WEB-4.pdf"},
    {"id": "adp_6_0", "title": "ADP 6-0: Mission Command", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38007-ADP_6-0-000-WEB-4.pdf"},

    # FM (Field Manuals)
    {"id": "fm_3_0", "title": "FM 3-0: Operations", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN36289-FM_3-0-000-WEB-2.pdf"},
    {"id": "fm_3_09", "title": "FM 3-09: Field Artillery Operations and Fire Support", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38014-FM_3-09-000-WEB-3.pdf"},
    {"id": "fm_3_21_8", "title": "FM 3-21.8: The Infantry Rifle Platoon and Squad", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38015-FM_3-21_8-000-WEB-3.pdf"},
    {"id": "fm_3_22", "title": "FM 3-22: Army Support to Security Cooperation", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38016-FM_3-22-000-WEB-3.pdf"},
    {"id": "fm_3_24", "title": "FM 3-24: Insurgencies and Countering Insurgencies", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38017-FM_3-24-000-WEB-3.pdf"},
    {"id": "fm_3_52", "title": "FM 3-52: Airspace Control", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38018-FM_3-52-000-WEB-3.pdf"},
    {"id": "fm_3_60", "title": "FM 3-60: The Targeting Process", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38019-FM_3-60-000-WEB-3.pdf"},
    {"id": "fm_3_90", "title": "FM 3-90: Tactics", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38020-FM_3-90-000-WEB-3.pdf"},
    {"id": "fm_3_96", "title": "FM 3-96: Brigade Combat Team", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38021-FM_3-96-000-WEB-3.pdf"},
    {"id": "fm_3_98", "title": "FM 3-98: Reconnaissance and Security Operations", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38022-FM_3-98-000-WEB-3.pdf"},
    {"id": "fm_3_99", "title": "FM 3-99: Airborne and Air Assault Operations", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38023-FM_3-99-000-WEB-3.pdf"},
    {"id": "fm_6_0", "title": "FM 6-0: Commander and Staff Organization and Operations", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38024-FM_6-0-000-WEB-3.pdf"},
    {"id": "fm_6_22", "title": "FM 6-22: Developing Leaders", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38025-FM_6-22-000-WEB-3.pdf"},
    {"id": "fm_7_0", "title": "FM 7-0: Training", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38026-FM_7-0-000-WEB-3.pdf"},

    # ATP (Army Techniques Publications)
    {"id": "atp_3_21_8", "title": "ATP 3-21.8: Infantry Platoon and Squad", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38009-ATP_3-21_8-000-WEB-3.pdf"},
    {"id": "atp_3_90_1", "title": "ATP 3-90.1: Armor and Mechanized Infantry Company Team", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38010-ATP_3-90_1-000-WEB-3.pdf"},
    {"id": "atp_4_02", "title": "ATP 4-02: Army Health System", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38011-ATP_4-02-000-WEB-3.pdf"},
    {"id": "atp_6_02_70", "title": "ATP 6-02.70: Techniques for Spectrum Management Operations", "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38012-ATP_6-02_70-000-WEB-3.pdf"},
]

# FAS 镜像（比 armypubs.army.mil 更稳定的公开源）
FAS_BASE_ARMY = "https://irp.fas.org/doddir/army/"

# 主 URL 列表 — 优先用 FAS，armypubs 作为备用
FM_LIST = [
    # ADP (Army Doctrine Publications) — FAS 镜像
    {"id": "adp_1", "title": "ADP 1: The Army",
     "url": "https://irp.fas.org/doddir/army/adp1.pdf"},
    {"id": "adp_3_0", "title": "ADP 3-0: Operations",
     "url": "https://irp.fas.org/doddir/army/adp3_0.pdf"},
    {"id": "adp_4_0", "title": "ADP 4-0: Sustainment",
     "url": "https://irp.fas.org/doddir/army/adp4_0.pdf"},
    {"id": "adp_5_0", "title": "ADP 5-0: The Operations Process",
     "url": "https://irp.fas.org/doddir/army/adp5_0.pdf"},
    {"id": "adp_6_0", "title": "ADP 6-0: Mission Command",
     "url": "https://irp.fas.org/doddir/army/adp6_0.pdf"},

    # FM (Field Manuals) — FAS 镜像
    {"id": "fm_3_0", "title": "FM 3-0: Operations",
     "url": "https://irp.fas.org/doddir/army/fm3-0.pdf"},
    {"id": "fm_3_21_8", "title": "FM 3-21.8: The Infantry Rifle Platoon and Squad",
     "url": "https://irp.fas.org/doddir/army/fm3-21-8.pdf"},
    {"id": "fm_3_24", "title": "FM 3-24: Insurgencies and Countering Insurgencies",
     "url": "https://irp.fas.org/doddir/army/fm3-24.pdf"},
    {"id": "fm_6_0", "title": "FM 6-0: Commander and Staff Organization",
     "url": "https://irp.fas.org/doddir/army/fm6-0.pdf"},
    {"id": "fm_7_0", "title": "FM 7-0: Training",
     "url": "https://irp.fas.org/doddir/army/fm7-0.pdf"},
    {"id": "fm_3_09", "title": "FM 3-09: Field Artillery Operations",
     "url": "https://irp.fas.org/doddir/army/fm3-09.pdf"},
    {"id": "fm_3_90", "title": "FM 3-90: Tactics",
     "url": "https://irp.fas.org/doddir/army/fm3-90.pdf"},
    {"id": "fm_3_96", "title": "FM 3-96: Brigade Combat Team",
     "url": "https://irp.fas.org/doddir/army/fm3-96.pdf"},
    {"id": "fm_3_98", "title": "FM 3-98: Reconnaissance and Security",
     "url": "https://irp.fas.org/doddir/army/fm3-98.pdf"},
    {"id": "fm_100_2_1", "title": "FM 100-2-1: Soviet Army Operations",
     "url": "https://irp.fas.org/doddir/army/fm100-2-1.pdf"},
    {"id": "fm_100_2_2", "title": "FM 100-2-2: Soviet Specialized Warfare",
     "url": "https://irp.fas.org/doddir/army/fm100-2-2.pdf"},
    {"id": "fm_100_2_3", "title": "FM 100-2-3: Soviet Troops Organization",
     "url": "https://irp.fas.org/doddir/army/fm100-2-3.pdf"},
    {"id": "fm_34_130", "title": "FM 34-130: Intelligence Preparation of Battlefield",
     "url": "https://irp.fas.org/doddir/army/fm34-130.pdf"},
    {"id": "fm_101_5", "title": "FM 101-5: Staff Organization and Operations",
     "url": "https://irp.fas.org/doddir/army/fm101-5.pdf"},
    {"id": "fm_101_5_1", "title": "FM 101-5-1: Operational Terms and Graphics",
     "url": "https://irp.fas.org/doddir/army/fm101-5-1.pdf"},

    # 尝试 armypubs.army.mil（可能被防火墙阻挡但值得一试）
    {"id": "atp_3_21_8", "title": "ATP 3-21.8: Infantry Platoon and Squad",
     "url": "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN38009-ATP_3-21_8-000-WEB-3.pdf"},
]


def fetch_pdf(url: str, max_retries: int = 3) -> bytes | None:
    """下载 PDF 文件。"""
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
    """从 PDF bytes 提取文本。"""
    # 尝试 PyMuPDF
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        if len(text.strip()) > 500:
            return text
    except ImportError:
        pass

    # 尝试 pdfplumber
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

    raise ImportError("需要安装 PyMuPDF 或 pdfplumber: pip install pymupdf")


def _guess_content_category_en(text: str) -> str:
    t = text[:3000].lower()
    equip = sum(1 for kw in ["weapon", "equipment", "fire support", "artillery",
                              "armor", "vehicle", "aircraft", "rifle"] if kw in t)
    doctrine = sum(1 for kw in ["doctrine", "operation", "command", "tactics",
                                 "planning", "mission", "leader"] if kw in t)
    case = sum(1 for kw in ["case study", "historical", "lesson", "battle",
                             "campaign", "exercise", "scenario"] if kw in t)
    scores = {"equipment": equip, "doctrine": doctrine, "case": case}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "doctrine"


def main():
    parser = argparse.ArgumentParser(description="US Army Field Manuals 下载")
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

    print(f"US Army Field Manuals 下载")
    print(f"  出版物数: {len(FM_LIST)}")
    print(f"  输出: {output_dir}")
    print(f"  注意: PDF 文本提取需要 pip install pymupdf")

    if args.dry_run:
        print("\n[DRY RUN] 检查 URL 可达性...")
        for fm in FM_LIST[:5]:
            try:
                req = Request(fm["url"], headers={"User-Agent": USER_AGENT}, method="HEAD")
                resp = urlopen(req, timeout=15)
                print(f"  {'✅' if resp.status == 200 else '❌'} {fm['id']}: {fm['title']}")
            except Exception as e:
                print(f"  ❌ {fm['id']}: {e}")
        return

    total = 0
    errors: list[str] = []

    for fm in FM_LIST:
        fm_id = fm["id"]
        if fm_id in done_ids:
            continue

        print(f"  {fm_id}: {fm['title']}")

        pdf_bytes = fetch_pdf(fm["url"])
        if not pdf_bytes:
            # 尝试 FAS 备选 URL
            fallback_url = f"{FAS_BASE_ARMY}{fm_id}.pdf"
            print(f"    尝试备用: {fallback_url}")
            pdf_bytes = fetch_pdf(fallback_url)
            if not pdf_bytes:
                errors.append(f"{fm_id}: 所有 URL 均失败")
                continue

        try:
            text = extract_text_pdf(pdf_bytes)
        except ImportError:
            print("    错误: 需要安装 PDF 文本提取库 (pymupdf 或 pdfplumber)")
            raise SystemExit(1)

        if len(text.strip()) < 500:
            errors.append(f"{fm_id}: PDF 文本过短 ({len(text)} 字符)")
            continue

        # 保存
        txt_path = output_dir / f"{fm_id}.txt"
        txt_path.write_text(text, encoding="utf-8")

        meta = {
            "source_url": fm["url"],
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "source_category": "doctrine",
            "content_category": _guess_content_category_en(text),
            "authority": "doctrine",
            "language": "en",
            "desensitized": True,
            "title": fm["title"],
            "document_id": fm_id,
        }
        meta_path = output_dir / f"{fm_id}.meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        done_ids.add(fm_id)
        total += 1

        if total % 5 == 0:
            checkpoint_file.write_text(
                json.dumps({"done_ids": list(done_ids)}, ensure_ascii=False)
            )

        time.sleep(3)  # 礼貌延迟

    checkpoint_file.write_text(
        json.dumps({"done_ids": list(done_ids)}, ensure_ascii=False)
    )
    if errors:
        err_path = output_dir / "errors.log"
        err_path.write_text("\n".join(errors), encoding="utf-8")
        print(f"  ⚠️ {len(errors)} 个出版物下载失败")

    print(f"✅ 完成: {total} 份 Field Manual → {output_dir}")


if __name__ == "__main__":
    main()
