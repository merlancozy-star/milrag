"""测试：RRF 混合检索。"""
from milrag.retrieval.hybrid import rrf_fuse


def test_rrf_empty():
    assert rrf_fuse([], [], k=60) == []


def test_rrf_single_list():
    result = rrf_fuse(["a", "b", "c"], [], k=60)
    assert result == ["a", "b", "c"]


def test_rrf_merge():
    dense = ["a", "b", "c"]
    sparse = ["c", "b", "d"]
    result = rrf_fuse(dense, sparse, k=60)
    # c 在两个列表中均出现 → RRF 分最高
    assert result[0] == "c"
    # 所有文档都出现
    assert set(result) == {"a", "b", "c", "d"}


def test_rrf_order():
    # 验证 RRF 能提升两列表都出现的高质量文档
    dense = ["x", "y", "z"]
    sparse = ["y", "x", "w"]
    result = rrf_fuse(dense, sparse, k=60)
    # x 和 y 在两个列表中排名都靠前 → 应在 w 和 z 之前
    assert result.index("x") < result.index("w")
    assert result.index("y") < result.index("z")
