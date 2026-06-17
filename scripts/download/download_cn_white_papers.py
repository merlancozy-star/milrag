#!/usr/bin/env python3
"""下载中国国防白皮书。

来源:
  - 国防部官网 mod.gov.cn 公开发布的白皮书
  - 国务院新闻办 scio.gov.cn 国防白皮书

白皮书是公开的政府文件，PDF 格式。
需要: pip install pdfplumber 或 PyMuPDF

输出: data/raw/cn_white_papers/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen, urlretrieve


OUTPUT_DIR = Path("data/raw/cn_white_papers")

# 已知的中国国防白皮书 URL 列表
WHITE_PAPER_URLS = [
    # 新时代的中国国防 (2019)
    "http://www.mod.gov.cn/gfbw/gfww/1622779.html",
    # 中国的军事战略 (2015)
    "http://www.mod.gov.cn/gfbw/gfww/1622778.html",
    # 中国武装力量的多样化运用 (2013)
    "http://www.mod.gov.cn/gfbw/gfww/1622777.html",
    # 国防白皮书系列
    "http://www.scio.gov.cn/ztk/dtzt/2019/16899/16904/index.htm",
    # 2000-2018 历次国防白皮书索引
    "http://www.mod.gov.cn/gfbw/index.htm",
]

# 备用：直接 PDF 链接
PDF_URLS = [
    "http://www.scio.gov.cn/zfbps/ndhf/2019n/201907/P020190724569802625026.pdf",
    "http://www.scio.gov.cn/zfbps/ndhf/2015n/201505/P020150526556914326121.pdf",
    "http://www.scio.gov.cn/zfbps/ndhf/2013n/201304/P020130416559172836002.pdf",
]

USER_AGENT = "milrag-corpus-builder/1.0 (academic research; contact: see project README)"

# PDF 提取器
PDF_BACKEND = None


def get_pdf_backend():
    global PDF_BACKEND
    if PDF_BACKEND is not None:
        return PDF_BACKEND

    for backend_name in ("pdfplumber", "fitz"):  # fitz = PyMuPDF
        try:
            __import__(backend_name)
            PDF_BACKEND = backend_name
            return backend_name
        except ImportError:
            continue
    PDF_BACKEND = False
    return False


def extract_text_from_pdf(pdf_path: Path) -> str:
    backend = get_pdf_backend()
    if backend == "pdfplumber":
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return "\n\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    elif backend == "fitz":
        import fitz
        doc = fitz.open(pdf_path)
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    else:
        raise ImportError("需要安装 pdfplumber 或 PyMuPDF: pip install pdfplumber")


def fetch_html(url: str) -> str | None:
    """获取 HTML 页面内容。"""
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"    获取失败: {url} — {e}")
        return None


def extract_text_from_html(html: str) -> str:
    """从 HTML 中提取正文（简单方法）。"""
    # 去掉 script/style
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # 去掉标签
    text = re.sub(r"<[^>]+>", "\n", html)
    # 去掉多余空白
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def download_and_save(document_id: str, title: str, source_url: str,
                      output_dir: Path) -> bool:
    """下载并保存一份文档。"""
    txt_path = output_dir / f"{document_id}.txt"
    meta_path = output_dir / f"{document_id}.meta.json"

    # 判断是 PDF 还是 HTML
    if source_url.lower().endswith(".pdf"):
        try:
            urlretrieve(source_url, txt_path.with_suffix(".pdf"))
            text = extract_text_from_pdf(txt_path.with_suffix(".pdf"))
            txt_path.with_suffix(".pdf").unlink()  # 删除 PDF
        except Exception as e:
            print(f"    PDF 处理失败: {source_url} — {e}")
            return False
    else:
        html = fetch_html(source_url)
        if html is None:
            return False
        text = extract_text_from_html(html)

    if len(text.strip()) < 500:
        print(f"    文本过短 ({len(text)} 字符): {source_url}")
        return False

    # 写文本
    txt_path.write_text(text, encoding="utf-8")

    # 写元信息
    meta = {
        "source_url": source_url,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "source_category": "doctrine",
        "content_category": "doctrine",
        "authority": "official_bulletin",
        "language": "zh",
        "desensitized": True,
        "title": title,
        "document_id": document_id,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return True


def main():
    parser = argparse.ArgumentParser(description="中国国防白皮书下载")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("中国国防白皮书下载")
    print(f"  输出: {output_dir}")

    # 可用的 PDF 后端
    backend = get_pdf_backend()
    if backend:
        print(f"  PDF 后端: {backend}")
    else:
        print("  ⚠️ 无 PDF 后端，将只下载 HTML 页面（pip install pdfplumber）")

    if args.dry_run:
        print("\n[DRY RUN] 检查 URL 可达性...")
        for url in WHITE_PAPER_URLS + PDF_URLS:
            try:
                req = Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
                resp = urlopen(req, timeout=15)
                print(f"  {'✅' if resp.status == 200 else '❌'} {url}")
            except Exception as e:
                print(f"  ❌ {url} — {e}")
        return

    count = 0
    errors: list[str] = []

    # 先尝试 PDF 链接
    for i, url in enumerate(PDF_URLS):
        doc_id = f"cn_wp_pdf_{i:03d}"
        title = url.split("/")[-1].replace(".pdf", "")
        if download_and_save(doc_id, title, url, output_dir):
            count += 1
            print(f"  ✅ {title}")
        else:
            errors.append(url)
        time.sleep(2)

    # 再尝试 HTML 页面
    for i, url in enumerate(WHITE_PAPER_URLS):
        doc_id = f"cn_wp_html_{i:03d}"
        title = f"国防白皮书_{i:03d}"
        if download_and_save(doc_id, title, url, output_dir):
            count += 1
            print(f"  ✅ {title}")
        else:
            errors.append(url)
        time.sleep(2)

    # 错误日志
    if errors:
        err_path = output_dir / "errors.log"
        err_path.write_text("\n".join(errors), encoding="utf-8")
        print(f"  ⚠️ {len(errors)} 个 URL 获取失败，详见 {err_path}")

    print(f"✅ 完成: {count} 份白皮书 → {output_dir}")


if __name__ == "__main__":
    main()
