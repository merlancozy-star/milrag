"""测试：证据融合。"""
from milrag.dynamic.fuse import fuse_evidence, _deduplicate, EvidenceItem, _authority_score


def test_authority_score():
    assert _authority_score("doctrine") < _authority_score("official_bulletin")
    assert _authority_score("official_bulletin") < _authority_score("mainstream_media")
    assert _authority_score("mainstream_media") < _authority_score("general_commentary")
    assert _authority_score("unknown") > _authority_score("doctrine")


def test_deduplicate_exact_content():
    """无嵌入时按内容去重。"""
    items = [
        EvidenceItem(chunk_id="a", content="歼-20 性能参数", authority="doctrine"),
        EvidenceItem(chunk_id="b", content="歼-20 性能参数", authority="mainstream_media"),
        EvidenceItem(chunk_id="c", content="运-20 性能参数", authority="doctrine"),
    ]
    result = _deduplicate(items, threshold=0.85)
    # a 和 b 相同内容，但 a 权威性更高 → 保留 a
    ids = {item.chunk_id for item in result}
    assert "a" in ids
    assert "c" in ids
    # b 应被去重（内容与 a 相同但权威性低）
    assert len(result) == 2


def test_fuse_evidence_basic():
    cfg = {
        "evidence_fusion": {
            "dedup_sim": 0.85,
            "authority_order": ["doctrine", "official_bulletin", "mainstream_media", "general_commentary"],
            "version_sort": "time_desc",
        },
    }
    existing = [{
        "chunk_id": "e1", "content": "歼-20 最大起飞重量 37 吨",
        "authority": "doctrine", "timestamp": "2025-01-01",
    }]
    new = [{
        "chunk_id": "e2", "content": "运-20 最大载重 66 吨",
        "authority": "official_bulletin", "timestamp": "2025-03-01",
    }]
    result = fuse_evidence(existing, new, cfg)
    assert len(result) == 2
    # doctrine 应该在 official_bulletin 前面
    assert result[0]["authority"] == "doctrine"
