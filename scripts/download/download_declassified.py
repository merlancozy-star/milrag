#!/usr/bin/env python3
"""下载公开解密军事文档。

来源:
  - Federation of American Scientists (FAS) Intelligence Resource Program (irp.fas.org)
  - CIA CREST 数据库公开部分 (cia.gov/readingroom)
  - US Army Combined Arms Center 解密历史手册

所有文档均为美国政府公开解密的政府出版物，无保密等级。
文档语言：英文（保留原文，不翻译）。

输出: data/raw/declassified/
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


OUTPUT_DIR = Path("data/raw/declassified")
USER_AGENT = "milrag-corpus-builder/1.0 (academic research; contact: see project README)"

# FAS Intelligence Resource Program — 公开解密的 DoD 文档目录
FAS_URLS = [
    # DoD Directives and Instructions
    "https://irp.fas.org/doddir/dod/",
    # Army Regulations (historical)
    "https://irp.fas.org/doddir/army/",
    # 解密情报研究
    "https://irp.fas.org/threat/",
    "https://irp.fas.org/world/",
    "https://irp.fas.org/agency/",
    # 解密评估报告
    "https://irp.fas.org/eprint/",
    "https://irp.fas.org/offdocs/",
]

# CIA CREST (公开部分)
CIA_READING_ROOM = "https://www.cia.gov/readingroom/"

# 已知的公开解密文档 URL（手动精选）
DECLASSIFIED_DOCS = [
    # DoD Dictionary of Military and Associated Terms
    "https://irp.fas.org/doddir/dod/jp1_02.pdf",
    # Understanding Military Staffs
    "https://irp.fas.org/doddir/army/fm101-5.pdf",
    # Operational Terms and Graphics
    "https://irp.fas.org/doddir/army/fm101-5-1.pdf",
    # Intelligence Preparation of the Battlefield
    "https://irp.fas.org/doddir/army/fm34-130.pdf",
    # 苏联军事力量相关解密报告（历史文件）
    "https://irp.fas.org/doddir/army/fm100-2-1.pdf",  # Soviet Army Operations
    "https://irp.fas.org/doddir/army/fm100-2-2.pdf",  # Soviet Army Specialized Warfare
    "https://irp.fas.org/doddir/army/fm100-2-3.pdf",  # Soviet Army Troops Organization
]


def fetch_url(url: str, max_retries: int = 3) -> bytes | None:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    获取失败: {url} — {e}")
                return None
            time.sleep(10)
    return None


def extract_text_pdf(pdf_bytes: bytes) -> str:
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        if len(text.strip()) > 300:
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
        if len(text.strip()) > 300:
            return text
    except ImportError:
        pass

    raise ImportError("需要安装 pymupdf 或 pdfplumber")


def extract_text_html(html_bytes: bytes) -> str:
    html = html_bytes.decode("utf-8", errors="ignore")
    for tag in ["script", "style", "nav", "footer", "header"]:
        html = re.sub(f"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "\n", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def scrape_fas_directory(dir_url: str, output_dir: Path) -> list[dict]:
    """从 FAS 目录页面抓取 PDF 链接列表。"""
    content = fetch_url(dir_url)
    if not content:
        return []

    html = content.decode("utf-8", errors="ignore")
    pdf_links = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html)
    results = []
    for link in pdf_links:
        if not link.startswith("http"):
            from urllib.parse import urljoin
            link = urljoin(dir_url, link)
        filename = link.split("/")[-1].replace(".pdf", "")
        results.append({"id": f"fas_{filename}", "title": filename, "url": link})
    return results


def _guess_category_en(text: str) -> str:
    t = text[:2000].lower()
    equip = sum(1 for kw in ["weapon", "system", "equipment", "technical", "armament"] if kw in t)
    doctrine = sum(1 for kw in ["doctrine", "operation", "strategy", "command",
                                 "intelligence", "planning", "tactics"] if kw in t)
    case = sum(1 for kw in ["case", "history", "lesson", "campaign", "conflict",
                             "assessment", "estimate"] if kw in t)
    scores = {"equipment": equip, "doctrine": doctrine, "case": case}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "case"


def main():
    parser = argparse.ArgumentParser(description="公开解密军事文档下载")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--max-docs", type=int, default=100,
                        help="最多下载文档数")
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

    print("公开解密军事文档下载")

    if args.dry_run:
        print("\n[DRY RUN] 检查 URL 可达性...")
        for url in DECLASSIFIED_DOCS + FAS_URLS[:3]:
            try:
                req = Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
                resp = urlopen(req, timeout=15)
                print(f"  {'✅' if resp.status == 200 else '❌'} {url}")
            except Exception as e:
                print(f"  ❌ {url} — {e}")
        return

    all_docs = []
    # 先从已知列表开始
    for doc in DECLASSIFIED_DOCS:
        filename = doc.split("/")[-1].replace(".pdf", "")
        all_docs.append({"id": f"fas_{filename}", "title": filename, "url": doc})

    # 从 FAS 目录补充
    print("  抓取 FAS 目录...")
    for fas_url in FAS_URLS:
        found = scrape_fas_directory(fas_url, output_dir)
        all_docs.extend(found)
        if len(found) > 0:
            print(f"    {fas_url.split('/')[-2]}: 找到 {len(found)} 个 PDF")
        time.sleep(2)

    # 去重
    seen_urls = set()
    unique_docs = []
    for doc in all_docs:
        if doc["url"] not in seen_urls:
            seen_urls.add(doc["url"])
            unique_docs.append(doc)

    print(f"  共找到 {len(unique_docs)} 个唯一文档")
    if len(unique_docs) > args.max_docs:
        unique_docs = unique_docs[:args.max_docs]
        print(f"  限制为 {args.max_docs} 个")

    total = 0
    errors: list[str] = []

    for doc in unique_docs:
        doc_id = doc["id"]
        if doc_id in done_ids:
            continue

        content = fetch_url(doc["url"])
        if not content:
            errors.append(doc_id)
            continue

        # 尝试 PDF 提取
        try:
            text = extract_text_pdf(content)
        except ImportError:
            print("    错误: 需要安装 PDF 文本提取库")
            raise SystemExit(1)
        except Exception:
            text = ""

        # 如果不是 PDF，尝试 HTML 提取
        if len(text.strip()) < 200:
            text = extract_text_html(content)

        if len(text.strip()) < 200:
            errors.append(f"{doc_id}: 文本过短")
            continue

        txt_path = output_dir / f"{doc_id}.txt"
        txt_path.write_text(text, encoding="utf-8")

        meta = {
            "source_url": doc["url"],
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "source_category": "doctrine",
            "content_category": _guess_category_en(text),
            "authority": "doctrine",
            "language": "en",
            "desensitized": True,
            "title": doc["title"],
            "document_id": doc_id,
        }
        meta_path = output_dir / f"{doc_id}.meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        done_ids.add(doc_id)
        total += 1

        if total % 10 == 0:
            print(f"  已下载 {total} 个文档...")
            checkpoint_file.write_text(
                json.dumps({"done_ids": list(done_ids)}, ensure_ascii=False)
            )

        time.sleep(2)

    checkpoint_file.write_text(
        json.dumps({"done_ids": list(done_ids)}, ensure_ascii=False)
    )
    if errors:
        (output_dir / "errors.log").write_text("\n".join(errors), encoding="utf-8")
        print(f"  ⚠️ {len(errors)} 个文档下载失败")

    print(f"✅ 完成: {total} 个解密文档 → {output_dir}")


if __name__ == "__main__":
    main()
