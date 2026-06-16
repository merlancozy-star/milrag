"""测试：查询重构策略。"""
from milrag.dynamic.reformulate import QueryReformulator


class MockNER:
    """Mock NER 用于测试 reformulate。"""
    def extract_entities(self, text):
        from milrag.data.ner import Entity
        return [Entity(text="歼-20", normalized="歼-20", label="EQUIP", start=0, end=3)]


def test_context_fusion():
    cfg = {
        "reformulate": {
            "entity_enhance": {"beta": 0.6, "topk_entities": [3, 5]},
            "content_word_focus": {"keep_pos": ["n", "v", "nr"]},
            "context_fusion": {
                "prefix_chars": 10,
                "template": "{q}当前推理：{ctx}",
            },
        },
    }
    r = QueryReformulator(cfg, MockNER())
    result = r.context_fusion("歼-20 性能", "歼-20 是一款第五代隐身战斗机，其最大起飞重量为 37 吨", trigger_pos=20)
    assert "当前推理" in result


def test_content_word_focus_short_query():
    """短问题不应被删词（CLAUDE.md §6 的关键坑）。"""
    cfg = {
        "reformulate": {
            "entity_enhance": {"beta": 0.6, "topk_entities": [3, 5]},
            "content_word_focus": {"keep_pos": ["n", "v", "nr"]},
            "context_fusion": {"prefix_chars": 100, "template": "{q}当前推理：{ctx}"},
        },
    }
    r = QueryReformulator(cfg, MockNER())
    short_q = "歼-20"  # < 8 字
    result = r.content_word_focus(short_q, "")
    # 短问题不应被截断到只剩片段
    assert len(result) >= len(short_q)
