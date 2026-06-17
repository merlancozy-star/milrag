#!/usr/bin/env python3
"""下载国防部官网 (mod.gov.cn) 公开发布内容。

中华人民共和国民防部官网公开内容：新闻发布、政策文件、军事交流等。
所有内容均为公开发布的政府信息。

输出: data/raw/mod_gov/
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


OUTPUT_DIR = Path("data/raw/mod_gov")
BASE_URL = "http://www.mod.gov.cn/"
USER_AGENT = "milrag-corpus-builder/1.0 (academic research; contact: see project README)"

# mod.gov.cn 主要频道
CHANNELS = [
    "http://www.mod.gov.cn/gfbw/index.htm",       # 国防部发布
    "http://www.mod.gov.cn/topnews/node_48975.htm",  # 头条
    "http://www.mod.gov.cn/jwzx/node_46906.htm",   # 军事资讯
    "http://www.mod.gov.cn/gdnw/node_46913.htm",   # 国防内网
    "http://www.mod.gov.cn/jmsd/node_46914.htm",   # 军民融合
    "http://www.mod.gov.cn/wslt/node_46916.htm",   # 外事论坛
]


def fetch_url(url: str, max_retries: int = 3) -> str | None:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    获取失败: {url} — {e}")
                return None
            time.sleep(5)
    return None


def extract_article_links(html: str, base_url: str) -> list[str]:
    pattern = r'href=["\']([^"\']*content[^"\']*)["\']'
    links = re.findall(pattern, html)
    if not links:
        pattern = r'href=["\']([^"\']*/202[0-9][^"\']*/[0-9]+[^"\']*)["\']'
        links = re.findall(pattern, html)
    full_links = []
    for link in links:
        full = urljoin(base_url, link) if not link.startswith("http") else link
        if BASE_URL in full and full not in full_links:
            full_links.append(full)
    return full_links


def extract_text(html: str) -> str:
    for tag in ["script", "style", "nav", "footer", "header", "aside"]:
        html = re.sub(f"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE)

    for pattern in [
        r'<div[^>]*class=["\'][^"\']*article[^"\']*["\'][^>]*>(.*?)</div>',
        r'<div[^>]*id=["\']content["\'][^>]*>(.*?)</div>',
        r'<article[^>]*>(.*?)</article>',
    ]:
        match = re.search(pattern, html, re.DOTALL)
        if match:
            html = match.group(1)
            break

    text = re.sub(r"<[^>]+>", "\n", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _guess_category(title: str, text: str) -> str:
    combined = (title + text[:500]).lower()
    equip = sum(1 for kw in ["装备", "武器", "导弹", "战舰", "战机"] if kw in combined)
    doctrine = sum(1 for kw in ["条令", "政策", "法规", "制度", "纲领"] if kw in combined)
    case = sum(1 for kw in ["演习", "行动", "任务", "访问", "交流"] if kw in combined)
    scores = {"equipment": equip, "doctrine": doctrine, "case": case}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "situation"


def main():
    parser = argparse.ArgumentParser(description="国防部官网 (mod.gov.cn) 内容下载")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--max-articles", type=int, default=200,
                        help="每个频道最多下载的文章数")
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "checkpoint.json"

    if checkpoint_file.exists():
        cp = json.loads(checkpoint_file.read_text())
        processed_urls = set(cp.get("processed_urls", []))
    else:
        processed_urls = set()

    print("国防部官网 (mod.gov.cn) 内容下载")

    if args.dry_run:
        print("\n[DRY RUN] 检查频道 URL...")
        for url in CHANNELS:
            try:
                req = Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
                resp = urlopen(req, timeout=15)
                print(f"  {'✅' if resp.status == 200 else '❌'} {url}")
            except Exception as e:
                print(f"  ❌ {url} — {e}")
        return

    total = 0
    errors: list[str] = []

    for channel_url in CHANNELS:
        channel_name = channel_url.split("/")[-2] if "/" in channel_url else "unknown"
        print(f"\n  频道: {channel_name}")

        html = fetch_url(channel_url)
        if not html:
            continue

        links = extract_article_links(html, channel_url)
        print(f"    找到 {len(links)} 个链接")

        channel_count = 0
        for article_url in links:
            if article_url in processed_urls:
                continue
            if channel_count >= args.max_articles:
                break

            page_html = fetch_url(article_url)
            if not page_html:
                errors.append(article_url)
                continue

            text = extract_text(page_html)
            if len(text.strip()) < 200:
                continue

            title_match = re.search(r"<title>(.*?)</title>", page_html, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else f"mod_{total:05d}"

            doc_id = f"modgov_{total:05d}"
            txt_path = output_dir / f"{doc_id}.txt"
            txt_path.write_text(text, encoding="utf-8")

            meta = {
                "source_url": article_url,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "source_category": "commentary",
                "content_category": _guess_category(title, text),
                "authority": "official_bulletin",
                "language": "zh",
                "desensitized": True,
                "title": title,
                "document_id": doc_id,
            }
            meta_path = output_dir / f"{doc_id}.meta.json"
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            processed_urls.add(article_url)
            total += 1
            channel_count += 1

            if total % 50 == 0:
                print(f"    已下载 {total} 篇...")
                checkpoint_file.write_text(
                    json.dumps({"processed_urls": list(processed_urls)}, ensure_ascii=False)
                )

            time.sleep(args.delay)

    checkpoint_file.write_text(
        json.dumps({"processed_urls": list(processed_urls)}, ensure_ascii=False)
    )
    if errors:
        (output_dir / "errors.log").write_text("\n".join(errors), encoding="utf-8")

    print(f"\n✅ 完成: {total} 篇文章 → {output_dir}")


if __name__ == "__main__":
    main()
