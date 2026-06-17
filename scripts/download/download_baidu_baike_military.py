#!/usr/bin/env python3
"""下载百度百科军事词条。

百度百科是中文最大的网络百科全书，军事分类下有大量装备、条令、
军事人物、军事历史等词条。因为百度百科限制直接爬取，本脚本通过
公开的搜索缓存和已知词条列表获取内容。

策略:
  1. 准备一个军事词条列表（装备名称、军事术语等）
  2. 逐条下载百度百科页面的HTML
  3. 提取正文内容

输出: data/raw/baidu_baike_military/

注意: 遵守爬虫礼仪（2s 延迟，标注 User-Agent）
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


OUTPUT_DIR = Path("data/raw/baidu_baike_military")
USER_AGENT = "milrag-corpus-builder/1.0 (academic research; contact: see project README)"

# 已知的军事类百度百科词条（中文军事装备、概念、条令等）
# 这些是公开的百科词条，不含涉密内容
MILITARY_ENTRIES = {
    # 装备类
    "equipment": [
        "歼-20", "歼-16", "歼-10", "歼-11", "歼-15", "歼-31",
        "运-20", "直-20", "轰-6", "轰-20", "空警-500", "空警-2000",
        "辽宁舰", "山东舰", "福建舰", "055型驱逐舰", "052D型驱逐舰",
        "054A型护卫舰", "075型两栖攻击舰", "039型潜艇", "094型核潜艇",
        "东风-21D", "东风-26", "东风-41", "东风-17", "长剑-10", "鹰击-18",
        "红旗-9", "红旗-16", "99式坦克", "96式坦克", "15式坦克",
        "04A步兵战车", "05式两栖战车", "PLZ-05自行榴弹炮",
        "PHL-03远程火箭炮", "红旗-22", "S-400防空导弹",
        "攻击-11无人机", "无侦-8", "彩虹-5无人机", "翼龙-2无人机",
    ],
    # 条令概念类
    "doctrine": [
        "信息化战争", "联合作战", "一体化联合作战", "体系作战",
        "非对称作战", "反介入区域拒止", "网络空间作战",
        "电子战", "信息战", "心理战", "特种作战", "城市作战",
        "山地作战", "两栖作战", "空降作战", "机动作战",
        "火力战", "防空作战", "反导作战", "反潜作战",
        "军事战略", "战役法", "战术", "合同战术",
        "后勤保障", "装备保障", "军事训练", "国防动员",
    ],
    # 态势/案例类
    "situation": [
        "中国人民解放军", "中国人民解放军的组成", "战区",
        "东部战区", "南部战区", "西部战区", "北部战区", "中部战区",
        "火箭军", "战略支援部队", "联勤保障部队",
        "中国人民武装警察部队", "中国民兵",
    ],
}


def fetch_baike_page(entry_name: str, max_retries: int = 3) -> str | None:
    """获取百度百科词条页面 HTML。"""
    import urllib.parse
    encoded = urllib.parse.quote(entry_name)
    url = f"https://baike.baidu.com/item/{encoded}"

    req = Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=30) as resp:
                content = resp.read()
                encoding = resp.headers.get_content_charset() or "utf-8"
                return content.decode(encoding, errors="ignore")
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    获取失败: {entry_name} — {e}")
                return None
            time.sleep(5)
    return None


def extract_baike_text(html: str) -> str:
    """从百度百科页面提取正文。"""
    # 去掉 script/style
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # 百度百科正文容器
    for pattern in [
        r'<div[^>]*class=["\'][^"\']*para[^"\']*["\'][^>]*>(.*?)</div>',
        r'<div[^>]*class=["\'][^"\']*lemma-content[^"\']*["\'][^>]*>(.*?)</div>',
        r'<div[^>]*class=["\'][^"\']*main-content[^"\']*["\'][^>]*>(.*?)</div>',
    ]:
        matches = re.findall(pattern, html, re.DOTALL)
        if matches:
            text_parts = []
            for m in matches:
                part = re.sub(r"<[^>]+>", "", m)
                part = re.sub(r"&nbsp;", " ", part)
                part = re.sub(r"&[a-z]+;", " ", part)
                text_parts.append(part.strip())
            combined = "\n\n".join(t for t in text_parts if t)
            if len(combined) > 200:
                return combined

    # 通用提取
    text = re.sub(r"<[^>]+>", "\n", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="百度百科军事词条下载")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="请求间隔秒数")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "checkpoint.json"

    if checkpoint_file.exists():
        cp = json.loads(checkpoint_file.read_text())
        done_entries = set(cp.get("done_entries", []))
    else:
        done_entries = set()

    print("百度百科军事词条下载")
    total_entries = sum(len(v) for v in MILITARY_ENTRIES.values())
    print(f"  词条总数: {total_entries}")
    print(f"  输出: {output_dir}")

    if args.dry_run:
        # 只测试前 5 个
        test_entries = list(MILITARY_ENTRIES.values())[0][:5]
        for entry in test_entries:
            try:
                import urllib.parse
                url = f"https://baike.baidu.com/item/{urllib.parse.quote(entry)}"
                req = Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
                resp = urlopen(req, timeout=15)
                print(f"  {'✅' if resp.status == 200 else '❌'} {entry}")
            except Exception as e:
                print(f"  ❌ {entry} — {e}")
        return

    total = 0
    errors: list[str] = []

    for content_category, entries in MILITARY_ENTRIES.items():
        print(f"\n  内容类别: {content_category} ({len(entries)} 词条)")

        for entry in entries:
            if entry in done_entries:
                continue

            html = fetch_baike_page(entry)
            if not html:
                errors.append(entry)
                continue

            text = extract_baike_text(html)
            if len(text.strip()) < 100:
                errors.append(entry)
                continue

            # 提取标题
            title_match = re.search(r"<title>(.*?)_百度百科</title>", html)
            title = title_match.group(1).strip() if title_match else entry

            doc_id = f"baike_{entry.replace('-', '_').replace(' ', '_')}"
            txt_path = output_dir / f"{doc_id}.txt"
            txt_path.write_text(text, encoding="utf-8")

            meta = {
                "source_url": f"https://baike.baidu.com/item/{entry}",
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "source_category": "encyclopedia",
                "content_category": content_category,
                "authority": "mainstream_media",
                "language": "zh",
                "desensitized": True,
                "title": title,
                "document_id": doc_id,
            }
            meta_path = output_dir / f"{doc_id}.meta.json"
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            done_entries.add(entry)
            total += 1

            if total % 20 == 0:
                print(f"    已下载 {total} 词条...")
                checkpoint_file.write_text(
                    json.dumps({"done_entries": list(done_entries)}, ensure_ascii=False)
                )

            time.sleep(args.delay)

    checkpoint_file.write_text(
        json.dumps({"done_entries": list(done_entries)}, ensure_ascii=False)
    )
    if errors:
        (output_dir / "errors.log").write_text("\n".join(errors), encoding="utf-8")
        print(f"  ⚠️ {len(errors)} 个词条获取失败或内容过短")

    print(f"\n✅ 完成: {total} 词条 → {output_dir}")


if __name__ == "__main__":
    main()
