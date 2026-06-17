#!/usr/bin/env bash
# run_acquisition.sh — 主控脚本：顺序/并行调度全部语料下载
#
# 用法:
#   bash scripts/download/run_acquisition.sh              # 完整采集
#   bash scripts/download/run_acquisition.sh --dry-run    # 仅测试 URL 可达性
#   bash scripts/download/run_acquisition.sh --phase 1    # 只运行 Phase 1
#
# 在联网 GPU 服务器上运行一次。运行后数据在 data/raw/ 下，后续全部离线。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

DRY_RUN=""
PHASE=""
CMNEE_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --cmnee-path)
            CMNEE_PATH="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: bash scripts/download/run_acquisition.sh [--dry-run] [--phase 1|2|3|4|5] [--cmnee-path /path/to/cmnee]"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "milrag 语料采集主控脚本"
echo "============================================"
echo "项目根目录: $PROJECT_ROOT"

# 创建 data/raw 目录
mkdir -p data/raw

if [ -n "$DRY_RUN" ]; then
    echo ""
    echo "模式: DRY RUN — 仅测试 URL 可达性，不实际下载"
    echo ""
fi

# ═══════════════════════════════════════════════════
# Phase 1: CMNEE + 中文官方源
# ═══════════════════════════════════════════════════
run_phase1() {
    echo ""
    echo "--- Phase 1: CMNEE + 中文官方源 ---"
    echo "预估产出: ~155,000 段中文"
    echo ""

    # CMNEE 导入（如有）
    if [ -n "$CMNEE_PATH" ]; then
        echo "[1a] CMNEE 数据集导入..."
        python "$SCRIPT_DIR/import_cmnee.py" --input "$CMNEE_PATH" $DRY_RUN
    else
        echo "[1a] CMNEE: 跳过 (未指定 --cmnee-path)"
    fi

    # 中文官方源并行下载
    echo "[1b] 中文官方源下载（并行）..."

    python "$SCRIPT_DIR/download_cn_white_papers.py" $DRY_RUN &
    PID1=$!

    python "$SCRIPT_DIR/download_cn_doctrine.py" $DRY_RUN &
    PID2=$!

    python "$SCRIPT_DIR/download_81cn.py" --max-articles 200 $DRY_RUN &
    PID3=$!

    python "$SCRIPT_DIR/download_mod_gov.py" --max-articles 200 $DRY_RUN &
    PID4=$!

    python "$SCRIPT_DIR/download_xinhua_military.py" --max-articles 300 $DRY_RUN &
    PID5=$!

    python "$SCRIPT_DIR/download_baidu_baike_military.py" $DRY_RUN &
    PID6=$!

    wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6

    echo "Phase 1 完成"
}

# ═══════════════════════════════════════════════════
# Phase 2: 维基百科
# ═══════════════════════════════════════════════════
run_phase2() {
    echo ""
    echo "--- Phase 2: 维基百科下载 ---"
    echo "预估产出: ~90,000 段（中英混合）"
    echo ""

    echo "[2a] 中文维基百科军事类..."
    python "$SCRIPT_DIR/download_zhwiki_military.py" $DRY_RUN &
    PID1=$!

    echo "[2b] 英文维基百科军事类..."
    python "$SCRIPT_DIR/download_enwiki_military.py" $DRY_RUN &
    PID2=$!

    wait $PID1 $PID2

    echo "Phase 2 完成"
}

# ═══════════════════════════════════════════════════
# Phase 3: 美军条令 + 国际源
# ═══════════════════════════════════════════════════
run_phase3() {
    echo ""
    echo "--- Phase 3: 美军条令 + 国际源 ---"
    echo "预估产出: ~108,000 段（英中混合）"
    echo ""

    echo "[3a] 美军条令 + 解密文档（并行）..."
    python "$SCRIPT_DIR/download_us_army_fm.py" $DRY_RUN &
    PID1=$!

    python "$SCRIPT_DIR/download_us_joint_pubs.py" $DRY_RUN &
    PID2=$!

    python "$SCRIPT_DIR/download_declassified.py" --max-docs 100 $DRY_RUN &
    PID3=$!

    wait $PID1 $PID2 $PID3

    echo "[3b] 国际源 + 专业分析（并行）..."
    python "$SCRIPT_DIR/download_international_defense.py" --max-articles 50 $DRY_RUN &
    PID4=$!

    python "$SCRIPT_DIR/download_professional_military.py" --max-articles 100 $DRY_RUN &
    PID5=$!

    python "$SCRIPT_DIR/download_academic_military.py" $DRY_RUN &
    PID6=$!

    wait $PID4 $PID5 $PID6

    echo "Phase 3 完成"
}

# ═══════════════════════════════════════════════════
# Phase 4: 合成生成 + 合作数据
# ═══════════════════════════════════════════════════
run_phase4() {
    echo ""
    echo "--- Phase 4: 合成生成 + 合作数据 ---"
    echo "预估产出: 补齐至 315,000 段"
    echo ""

    # 先导入合作数据
    echo "[4a] 合作单位脱密数据..."
    python "$SCRIPT_DIR/import_partner_data.py" $DRY_RUN

    # 计算缺口
    echo "[4b] 计算段落缺口..."
    python "$SCRIPT_DIR/validate_corpus.py" --estimate-chunks

    # 合成生成（按缺口量）
    echo "[4c] 合成知识库段落生成..."
    if [ -z "$DRY_RUN" ]; then
        echo "  请根据缺口量运行:"
        echo "  python $SCRIPT_DIR/generate_synthetic_kb.py --count <缺口量/4> --use-vllm"
        echo "  (缺口量 = (314759 - 当前预估段落数) / 4)"
    else
        python "$SCRIPT_DIR/generate_synthetic_kb.py" --count 10 $DRY_RUN
    fi

    echo "Phase 4 完成"
}

# ═══════════════════════════════════════════════════
# Phase 5: 质量校验
# ═══════════════════════════════════════════════════
run_phase5() {
    echo ""
    echo "--- Phase 5: 质量校验 ---"

    echo "[5a] 生成语料清单..."
    python "$SCRIPT_DIR/corpus_manifest.py"

    echo "[5b] 质量校验..."
    python "$SCRIPT_DIR/validate_corpus.py" --estimate-chunks

    echo "Phase 5 完成"
}

# ═══════════════════════════════════════════════════
# 主调度
# ═══════════════════════════════════════════════════

case "${PHASE:-all}" in
    1)
        run_phase1
        ;;
    2)
        run_phase2
        ;;
    3)
        run_phase3
        ;;
    4)
        run_phase4
        ;;
    5)
        run_phase5
        ;;
    all)
        run_phase1
        run_phase2
        run_phase3
        run_phase4
        run_phase5
        ;;
    *)
        echo "未知阶段: $PHASE"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "采集完成！"
echo ""
echo "数据目录: data/raw/"
echo "下一步:"
echo "  1. python scripts/download/validate_corpus.py --estimate-chunks"
echo "  2. 如段落数不足 315,000，运行:"
echo "     python scripts/download/generate_synthetic_kb.py --count NEEDED --use-vllm"
echo "  3. 清洗分块: python scripts/run_clean.py && python scripts/run_chunk.py"
echo "  4. 构建索引: python scripts/run_index.py"
echo "============================================"
