#!/usr/bin/env python3
"""下载国际防务媒体公开文章。

来源:
  - Defense News (defensenews.com)
  - Breaking Defense (breakingdefense.com)
  - IISS Military Balance 摘要 (iiss.org)

国际防务媒体的公开报道和分析文章，英文原文保留不翻译。

输出: data/raw/international_defense/
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


OUTPUT_DIR = Path("data/raw/international_defense")
USER_AGENT = "milrag-corpus-builder/1.0 (academic research; contact: see project README)"

# 已知的公开防务媒体 RSS 和栏目 URL
RSS_FEEDS = [
    # Defense News
    "https://www.defensenews.com/arcio/rss/",
    # Breaking Defense
    "https://breakingdefense.com/feed/",
    # IISS (公开分析)
    "https://www.iiss.org/publications/",
    # RAND Corporation — 公开军事研究
    "https://www.rand.org/topics/military.html",
    "https://www.rand.org/topics/national-security.html",
    # CSIS 国际安全
    "https://www.csis.org/programs/international-security-program",
    # War on the Rocks
    "https://warontherocks.com/feed/",
]


def fetch_url(url: str, max_retries: int = 3) -> str | None:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=30) as resp:
                content = resp.read()
                encoding = resp.headers.get_content_charset() or "utf-8"
                return content.decode(encoding, errors="ignore")
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    获取失败: {url} — {e}")
                return None
            time.sleep(5)
    return None


def extract_rss_links(xml_content: str) -> list[dict]:
    """从 RSS/Atom feed 中提取文章链接和标题。"""
    items = re.findall(r"<item>(.*?)</item>", xml_content, re.DOTALL)
    if not items:
        items = re.findall(r"<entry>(.*?)</entry>", xml_content, re.DOTALL)

    results = []
    for item in items:
        title_match = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
        link_match = re.search(r"<link[^>]*href=[\"']([^\"']+)[\"']", item)
        if not link_match:
            link_match = re.search(r"<link>(.*?)</link>", item)
        if title_match and link_match:
            results.append({
                "title": re.sub(r"<[^>]+>", "", title_match.group(1)).strip(),
                "url": link_match.group(1).strip(),
            })
    return results


def extract_article_links(html: str, base_url: str) -> list[dict]:
    """从 HTML 页面中提取文章链接。"""
    # 常见文章链接模式
    patterns = [
        r'<a[^>]*href=["\']([^"\']*(?:article|story|analysis|publication|research|blog)[^"\']*)["\'][^>]*>(.*?)</a>',
        r'<a[^>]*href=["\']([^"\']*/202[0-9]/[^"\']*)["\'][^>]*>(.*?)</a>',
    ]
    results = []
    for pattern in patterns:
        matches = re.findall(pattern, html, re.DOTALL)
        for url, title in matches:
            full_url = url if url.startswith("http") else (
                url if url.startswith(base_url) else f"{base_url.rstrip('/')}/{url.lstrip('/')}"
            )
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            if clean_title and len(clean_title) > 5:
                results.append({"title": clean_title, "url": full_url})
    return results


def extract_text(html: str) -> str:
    for tag in ["script", "style", "nav", "footer", "header", "aside", "noscript"]:
        html = re.sub(f"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE)

    for pattern in [
        r'<article[^>]*>(.*?)</article>',
        r'<div[^>]*class=["\'][^"\']*article-body[^"\']*["\'][^>]*>(.*?)</div>',
        r'<div[^>]*class=["\'][^"\']*post-content[^"\']*["\'][^>]*>(.*?)</div>',
        r'<div[^>]*id=["\']content["\'][^>]*>(.*?)</div>',
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


def _guess_category_en(title: str, text: str) -> str:
    combined = (title + " " + text[:500]).lower()
    equip = sum(1 for kw in ["weapon", "missile", "aircraft", "submarine", "tank",
                              "fighter", "drone", "system"] if kw in combined)
    doctrine = sum(1 for kw in ["doctrine", "strategy", "policy", "defense budget",
                                 "military spending", "force structure"] if kw in combined)
    case = sum(1 for kw in ["exercise", "operation", "deployment", "conflict",
                             "war", "crisis", "analysis"] if kw in combined)
    scores = {"equipment": equip, "doctrine": doctrine, "case": case}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "situation"


def main():
    parser = argparse.ArgumentParser(description="国际防务媒体文章下载")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--max-articles", type=int, default=50,
                        help="每个 RSS/频道最多下载的文章数")
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "checkpoint.json"

    if checkpoint_file.exists():
        cp = json.loads(checkpoint_file.read_text())
        done_urls = set(cp.get("done_urls", []))
    else:
        done_urls = set()

    print("国际防务媒体文章下载")
    print(f"  频道数: {len(RSS_FEEDS)}")

    if args.dry_run:
        print("\n[DRY RUN] 检查 RSS URL 可达性...")
        for url in RSS_FEEDS:
            try:
                req = Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
                resp = urlopen(req, timeout=15)
                print(f"  {'✅' if resp.status == 200 else '❌'} {url}")
            except Exception as e:
                print(f"  ❌ {url} — {e}")
        return

    total = 0
    errors: list[str] = []

    for feed_url in RSS_FEEDS:
        source_name = [part for part in feed_url.split("/") if "." in part][0].split(".")[-2] \
            if "/" in feed_url else "unknown"
        print(f"\n  频道: {source_name}")

        content = fetch_url(feed_url)
        if not content:
            continue

        # 判断是 RSS/XML 还是 HTML
        if content.strip().startswith("<?xml") or "<rss" in content[:200] or "<feed" in content[:200]:
            articles = extract_rss_links(content)
        else:
            articles = extract_article_links(content, feed_url)

        print(f"    找到 {len(articles)} 篇文章")

        channel_count = 0
        for article in articles:
            article_url = article["url"]
            if article_url in done_urls:
                continue
            if channel_count >= args.max_articles:
                break

            html = fetch_url(article_url)
            if not html:
                errors.append(article_url)
                continue

            text = extract_text(html)
            if len(text.strip()) < 300:
                continue

            doc_id = f"intl_defense_{total:05d}"
            txt_path = output_dir / f"{doc_id}.txt"
            txt_path.write_text(text, encoding="utf-8")

            meta = {
                "source_url": article_url,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "source_category": "commentary",
                "content_category": _guess_category_en(article["title"], text),
                "authority": "mainstream_media",
                "language": "en",
                "desensitized": True,
                "title": article["title"][:200],
                "document_id": doc_id,
            }
            meta_path = output_dir / f"{doc_id}.meta.json"
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            done_urls.add(article_url)
            total += 1
            channel_count += 1

            if total % 20 == 0:
                print(f"    已下载 {total} 篇...")
                checkpoint_file.write_text(
                    json.dumps({"done_urls": list(done_urls)}, ensure_ascii=False)
                )

            time.sleep(args.delay)

    checkpoint_file.write_text(
        json.dumps({"done_urls": list(done_urls)}, ensure_ascii=False)
    )
    if errors:
        (output_dir / "errors.log").write_text("\n".join(errors), encoding="utf-8")

    print(f"\n✅ 完成: {total} 篇文章 → {output_dir}")


if __name__ == "__main__":
    main()
