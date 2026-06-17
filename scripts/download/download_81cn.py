#!/usr/bin/env python3
"""下载中国军网 (81.cn) 公开文章。

中国军网是解放军官方新闻门户，所有公开文章可合法访问。
通过公开 RSS 和文章页获取军事新闻文本。

输出: data/raw/81cn/
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen


OUTPUT_DIR = Path("data/raw/81cn")
BASE_URL = "http://www.81.cn/"
USER_AGENT = "milrag-corpus-builder/1.0 (academic research; contact: see project README)"

# 81.cn 主要频道
CHANNELS = [
    "http://www.81.cn/jwzl/node_68133.htm",     # 军事理论
    "http://www.81.cn/jwzl/node_68134.htm",     # 军事历史
    "http://www.81.cn/jwzl/node_68135.htm",     # 军事科技
    "http://www.81.cn/yw/node_68114.htm",       # 要闻
    "http://www.81.cn/bq/node_65877.htm",       # 部队
    "http://www.81.cn/gj/node_65883.htm",       # 国际
    "http://www.81.cn/js/node_68118.htm",       # 军事
]


def fetch_url(url: str, max_retries: int = 3) -> str | None:
    """获取 URL 内容。"""
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
    """从频道页面提取文章链接。"""
    # 匹配 href 属性中的文章 URL
    pattern = r'href=["\']([^"\']*(?:content|a/)\d+[^"\']*)["\']'
    links = re.findall(pattern, html)
    # 也匹配常见的文章路径
    if not links:
        pattern = r'href=["\']([^"\']*/\d{4,}/\d{4,}[^"\']*)["\']'
        links = re.findall(pattern, html)
    # 转绝对 URL
    full_links = []
    for link in links:
        full = urljoin(base_url, link)
        if BASE_URL in full and full not in full_links:
            full_links.append(full)
    return full_links


def extract_article_text(html: str) -> str:
    """从文章页 HTML 提取正文。"""
    # 去掉 script/style/nav/footer
    for tag in ["script", "style", "nav", "footer", "header", "aside"]:
        html = re.sub(f"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # 尝试找正文容器
    article_patterns = [
        r'<div[^>]*class=["\'][^"\']*article[^"\']*["\'][^>]*>(.*?)</div>',
        r'<div[^>]*id=["\']content["\'][^>]*>(.*?)</div>',
        r'<div[^>]*class=["\'][^"\']*content[^"\']*["\'][^>]*>(.*?)</div>',
        r'<article[^>]*>(.*?)</article>',
    ]
    for pattern in article_patterns:
        match = re.search(pattern, html, re.DOTALL)
        if match:
            html = match.group(1)
            break

    # 去掉剩余标签
    text = re.sub(r"<[^>]+>", "\n", html)
    # 清理空白
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _guess_content_category(title: str, text: str) -> str:
    combined = (title + text[:500]).lower()
    equip = sum(1 for kw in ["装备", "武器", "导弹", "战舰", "战机", "坦克"] if kw in combined)
    doctrine = sum(1 for kw in ["条令", "训练", "编制", "指挥", "战术"] if kw in combined)
    case = sum(1 for kw in ["演习", "行动", "任务", "冲突", "救灾"] if kw in combined)
    scores = {"equipment": equip, "doctrine": doctrine, "case": case}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "situation"


def main():
    parser = argparse.ArgumentParser(description="中国军网 (81.cn) 文章下载")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--max-articles", type=int, default=200,
                        help="每个频道最多下载的文章数 (默认: 200)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="请求间隔秒数 (默认: 2.0)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "checkpoint.json"
    errors_file = output_dir / "errors.log"

    # 恢复进度
    if checkpoint_file.exists():
        cp = json.loads(checkpoint_file.read_text())
        processed_urls = set(cp.get("processed_urls", []))
    else:
        processed_urls = set()

    print("中国军网 (81.cn) 文章下载")
    print(f"  频道数: {len(CHANNELS)}")
    print(f"  每频道最多: {args.max_articles} 篇")

    if args.dry_run:
        print("\n[DRY RUN] 检查频道 URL 可达性...")
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

        # 获取频道页面
        channel_html = fetch_url(channel_url)
        if not channel_html:
            continue

        article_links = extract_article_links(channel_html, channel_url)
        print(f"    找到 {len(article_links)} 个文章链接")

        channel_count = 0
        for article_url in article_links:
            if article_url in processed_urls:
                continue
            if channel_count >= args.max_articles:
                break

            html = fetch_url(article_url)
            if not html:
                errors.append(article_url)
                continue

            text = extract_article_text(html)
            if len(text.strip()) < 200:
                continue

            # 提取标题
            title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else f"article_{total:05d}"

            doc_id = f"81cn_{total:05d}"
            txt_path = output_dir / f"{doc_id}.txt"
            txt_path.write_text(text, encoding="utf-8")

            meta = {
                "source_url": article_url,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "source_category": "commentary",
                "content_category": _guess_content_category(title, text),
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

    # 保存最终状态
    checkpoint_file.write_text(
        json.dumps({"processed_urls": list(processed_urls)}, ensure_ascii=False)
    )
    if errors:
        errors_file.write_text("\n".join(errors), encoding="utf-8")

    print(f"\n✅ 完成: {total} 篇文章 → {output_dir}")


if __name__ == "__main__":
    main()
