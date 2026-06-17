#!/usr/bin/env python3
"""下载专业军事分析网站内容。

来源:
  - 环球网军事频道 (mil.huanqiu.com) — 中文
  - 观察者网军事 (guancha.cn/military) — 中文
  - Jane's Defence 公开摘要 (janes.com) — 英文
  - RAND Corporation 军事研究摘要 — 英文
  - War on the Rocks — 英文

混合中英文源，涵盖专业军事分析视角。

输出: data/raw/professional_military/
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


OUTPUT_DIR = Path("data/raw/professional_military")
USER_AGENT = "milrag-corpus-builder/1.0 (academic research; contact: see project README)"

CHANNELS = [
    # 中文源
    {"url": "https://mil.huanqiu.com/", "name": "huanqiu_mil", "lang": "zh"},
    {"url": "https://www.guancha.cn/military-affairs", "name": "guancha_mil", "lang": "zh"},
    # 英文源
    {"url": "https://www.janes.com/defence-news/", "name": "janes", "lang": "en"},
    {"url": "https://www.rand.org/topics/military-strategy.html", "name": "rand_mil", "lang": "en"},
    {"url": "https://warontherocks.com/", "name": "warontherocks", "lang": "en"},
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


def extract_links(html: str, base_url: str) -> list[dict]:
    matches = re.findall(
        r'<a[^>]*href=["\']([^"\']*(?:article|news|detail|story|a/|content)[^"\']*)["\'][^>]*>(.*?)</a>',
        html, re.DOTALL,
    )
    results = []
    for url, title in matches:
        full_url = url if url.startswith("http") else f"{base_url.rstrip('/')}/{url.lstrip('/')}"
        clean_title = re.sub(r"<[^>]+>", "", title).strip()
        if clean_title and len(clean_title) > 5:
            results.append({"title": clean_title[:200], "url": full_url})
    return results


def extract_text(html: str) -> str:
    for tag in ["script", "style", "nav", "footer", "header", "aside"]:
        html = re.sub(f"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE)

    for pattern in [
        r'<article[^>]*>(.*?)</article>',
        r'<div[^>]*class=["\'][^"\']*article[^"\']*["\'][^>]*>(.*?)</div>',
        r'<div[^>]*class=["\'][^"\']*content[^"\']*["\'][^>]*>(.*?)</div>',
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


def _guess_category(title: str, text: str, lang: str) -> str:
    combined = (title + " " + text[:800]).lower()
    if lang == "zh":
        equip = sum(1 for kw in ["装备", "武器", "导弹", "战舰", "战机", "新型"] if kw in combined)
        doctrine = sum(1 for kw in ["条令", "战略", "政策", "方针", "军事力量"] if kw in combined)
        case = sum(1 for kw in ["演习", "行动", "冲突", "战争", "分析"] if kw in combined)
    else:
        equip = sum(1 for kw in ["weapon", "missile", "aircraft", "submarine", "tank"] if kw in combined)
        doctrine = sum(1 for kw in ["doctrine", "strategy", "policy", "defense", "force"] if kw in combined)
        case = sum(1 for kw in ["exercise", "operation", "conflict", "war", "analysis"] if kw in combined)

    scores = {"equipment": equip, "doctrine": doctrine, "case": case}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "situation"


def main():
    parser = argparse.ArgumentParser(description="专业军事分析网站下载")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--max-articles", type=int, default=100,
                        help="每个频道最多下载的文章数")
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

    print("专业军事分析网站下载")

    if args.dry_run:
        print("\n[DRY RUN] 检查频道 URL...")
        for ch in CHANNELS:
            try:
                req = Request(ch["url"], headers={"User-Agent": USER_AGENT}, method="HEAD")
                resp = urlopen(req, timeout=15)
                print(f"  {'✅' if resp.status == 200 else '❌'} [{ch['lang']}] {ch['name']}: {ch['url']}")
            except Exception as e:
                print(f"  ❌ [{ch['lang']}] {ch['name']}: {e}")
        return

    total = 0
    errors: list[str] = []

    for ch in CHANNELS:
        print(f"\n  [{ch['lang']}] {ch['name']}")

        html = fetch_url(ch["url"])
        if not html:
            continue

        articles = extract_links(html, ch["url"])
        print(f"    找到 {len(articles)} 个链接")

        channel_count = 0
        for article in articles:
            if article["url"] in done_urls:
                continue
            if channel_count >= args.max_articles:
                break

            page = fetch_url(article["url"])
            if not page:
                errors.append(article["url"])
                continue

            text = extract_text(page)
            if len(text.strip()) < 200:
                continue

            doc_id = f"{ch['name']}_{total:05d}"
            txt_path = output_dir / f"{doc_id}.txt"
            txt_path.write_text(text, encoding="utf-8")

            meta = {
                "source_url": article["url"],
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "source_category": "encyclopedia",
                "content_category": _guess_category(article["title"], text, ch["lang"]),
                "authority": "mainstream_media",
                "language": ch["lang"],
                "desensitized": True,
                "title": article["title"],
                "document_id": doc_id,
            }
            meta_path = output_dir / f"{doc_id}.meta.json"
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            done_urls.add(article["url"])
            total += 1
            channel_count += 1

            if total % 30 == 0:
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
