# run_acquisition.ps1 — Windows PowerShell 语料采集主控脚本
#
# 用法:
#   .\scripts\download\run_acquisition.ps1                  # 完整采集
#   .\scripts\download\run_acquisition.ps1 -DryRun           # 仅测试 URL
#   .\scripts\download\run_acquisition.ps1 -Phase 1          # 只运行 Phase 1
#   .\scripts\download\run_acquisition.ps1 -LocalOnly        # 仅本地可完成的（跳过GPU步骤）
#
# 策略：本地完成所有下载和清洗 → 打包上传到 GPU 服务器

param(
    [switch]$DryRun,
    [int]$Phase = 0,
    [string]$CmneePath = "",
    [switch]$LocalOnly,        # 仅本地任务，跳过需要 GPU 的步骤
    [switch]$SkipWikipedia     # 跳过维基百科大文件下载（在服务器上下载更快）
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)

Set-Location $ProjectRoot

# 确保 data/raw 目录存在
New-Item -ItemType Directory -Force -Path "data/raw" | Out-Null

$DryRunFlag = if ($DryRun) { "--dry-run" } else { "" }

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "milrag 语料采集主控脚本 (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "项目根目录: $ProjectRoot"

if ($DryRun) {
    Write-Host ""
    Write-Host "模式: DRY RUN — 仅测试 URL 可达性，不实际下载" -ForegroundColor Yellow
}

if ($LocalOnly) {
    Write-Host "模式: LOCAL ONLY — 跳过 GPU 依赖步骤" -ForegroundColor Yellow
}

if ($SkipWikipedia) {
    Write-Host "模式: 跳过维基百科大文件下载" -ForegroundColor Yellow
}

# ═══════════════════════════════════════════════════
# Phase 1: CMNEE + 中文官方源（全本地可完成）
# ═══════════════════════════════════════════════════
function Run-Phase1 {
    Write-Host ""
    Write-Host "--- Phase 1: CMNEE + 中文官方源 ---" -ForegroundColor Green
    Write-Host "网络需求: 小量 HTTP 请求（~100MB 文本）"
    Write-Host ""

    # CMNEE 导入（如有）
    if ($CmneePath) {
        Write-Host "[1a] CMNEE 数据集导入..."
        python "$ScriptDir\import_cmnee.py" --input $CmneePath $DryRunFlag
    } else {
        Write-Host "[1a] CMNEE: 跳过 (未指定 -CmneePath)"
    }

    # 中文官方源（并行 Job）
    Write-Host "[1b] 中文官方源下载（并行）..."

    $jobs = @()

    $jobs += Start-Job -Name "cn_wp" -ScriptBlock {
        param($dir, $flag)
        python "$dir\download_cn_white_papers.py" $flag
    } -ArgumentList $ScriptDir, $DryRunFlag

    $jobs += Start-Job -Name "cn_doc" -ScriptBlock {
        param($dir, $flag)
        python "$dir\download_cn_doctrine.py" $flag
    } -ArgumentList $ScriptDir, $DryRunFlag

    if (-not $SkipWikipedia) {
        $jobs += Start-Job -Name "81cn" -ScriptBlock {
            param($dir, $flag)
            python "$dir\download_81cn.py" --max-articles 200 $flag
        } -ArgumentList $ScriptDir, $DryRunFlag

        $jobs += Start-Job -Name "mod" -ScriptBlock {
            param($dir, $flag)
            python "$dir\download_mod_gov.py" --max-articles 200 $flag
        } -ArgumentList $ScriptDir, $DryRunFlag

        $jobs += Start-Job -Name "xinhua" -ScriptBlock {
            param($dir, $flag)
            python "$dir\download_xinhua_military.py" --max-articles 300 $flag
        } -ArgumentList $ScriptDir, $DryRunFlag

        $jobs += Start-Job -Name "baike" -ScriptBlock {
            param($dir, $flag)
            python "$dir\download_baidu_baike_military.py" $flag
        } -ArgumentList $ScriptDir, $DryRunFlag
    }

    $jobs | Wait-Job | Out-Null
    $jobs | Receive-Job
    $jobs | Remove-Job

    Write-Host "Phase 1 完成" -ForegroundColor Green
}

# ═══════════════════════════════════════════════════
# Phase 2: 维基百科（大文件，建议在服务器上下载）
# ═══════════════════════════════════════════════════
function Run-Phase2 {
    Write-Host ""
    Write-Host "--- Phase 2: 维基百科下载 ---" -ForegroundColor Green

    if ($SkipWikipedia) {
        Write-Host "跳过 (SkipWikipedia)" -ForegroundColor Yellow
        Write-Host "建议在 GPU 服务器上运行:"
        Write-Host "  python scripts/download/download_zhwiki_military.py"
        Write-Host "  python scripts/download/download_enwiki_military.py"
        return
    }

    Write-Host "⚠️ 需要下载 ~2.5GB (中文) + ~20GB (英文) 数据" -ForegroundColor Yellow
    Write-Host "如果本地带宽有限，建议 Ctrl+C 并在服务器上运行此阶段" -ForegroundColor Yellow
    Write-Host ""

    $jobs = @()

    $jobs += Start-Job -Name "zhwiki" -ScriptBlock {
        param($dir, $flag)
        python "$dir\download_zhwiki_military.py" $flag
    } -ArgumentList $ScriptDir, $DryRunFlag

    if (-not $DryRun) {
        Write-Host "[2b] 英文维基百科 (约 20GB，可单独跳过)..."
    }
    $jobs += Start-Job -Name "enwiki" -ScriptBlock {
        param($dir, $flag)
        python "$dir\download_enwiki_military.py" $flag
    } -ArgumentList $ScriptDir, $DryRunFlag

    $jobs | Wait-Job | Out-Null
    $jobs | Receive-Job
    $jobs | Remove-Job

    Write-Host "Phase 2 完成" -ForegroundColor Green
}

# ═══════════════════════════════════════════════════
# Phase 3: 美军条令 + 国际源（本地可完成）
# ═══════════════════════════════════════════════════
function Run-Phase3 {
    Write-Host ""
    Write-Host "--- Phase 3: 美军条令 + 国际源 ---" -ForegroundColor Green
    Write-Host "网络需求: PDF 下载 + HTTP 请求（~1GB）"
    Write-Host "需要: pip install pymupdf (PDF 文本提取)"
    Write-Host ""

    Write-Host "[3a] 美军条令 + 解密文档..."

    $jobs = @()

    $jobs += Start-Job -Name "us_fm" -ScriptBlock {
        param($dir, $flag)
        python "$dir\download_us_army_fm.py" $flag
    } -ArgumentList $ScriptDir, $DryRunFlag

    $jobs += Start-Job -Name "us_jp" -ScriptBlock {
        param($dir, $flag)
        python "$dir\download_us_joint_pubs.py" $flag
    } -ArgumentList $ScriptDir, $DryRunFlag

    $jobs += Start-Job -Name "declass" -ScriptBlock {
        param($dir, $flag)
        python "$dir\download_declassified.py" --max-docs 100 $flag
    } -ArgumentList $ScriptDir, $DryRunFlag

    $jobs | Wait-Job | Out-Null
    $jobs | Receive-Job
    $jobs | Remove-Job

    Write-Host "[3b] 国际源 + 专业分析..."

    $jobs2 = @()

    $jobs2 += Start-Job -Name "intl" -ScriptBlock {
        param($dir, $flag)
        python "$dir\download_international_defense.py" --max-articles 50 $flag
    } -ArgumentList $ScriptDir, $DryRunFlag

    $jobs2 += Start-Job -Name "prof" -ScriptBlock {
        param($dir, $flag)
        python "$dir\download_professional_military.py" --max-articles 100 $flag
    } -ArgumentList $ScriptDir, $DryRunFlag

    $jobs2 += Start-Job -Name "acad" -ScriptBlock {
        param($dir, $flag)
        python "$dir\download_academic_military.py" $flag
    } -ArgumentList $ScriptDir, $DryRunFlag

    $jobs2 | Wait-Job | Out-Null
    $jobs2 | Receive-Job
    $jobs2 | Remove-Job

    Write-Host "Phase 3 完成" -ForegroundColor Green
}

# ═══════════════════════════════════════════════════
# Phase 4: 合成生成 + 合作数据（部分需要 GPU）
# ═══════════════════════════════════════════════════
function Run-Phase4 {
    Write-Host ""
    Write-Host "--- Phase 4: 合成生成 + 合作数据 ---" -ForegroundColor Green

    # 合作数据（本地）
    Write-Host "[4a] 合作单位脱密数据..."
    python "$ScriptDir\import_partner_data.py" $DryRunFlag

    # 计算缺口
    Write-Host "[4b] 计算段落缺口..."
    python "$ScriptDir\validate_corpus.py" --estimate-chunks

    # 合成生成（需要 GPU）
    if ($LocalOnly) {
        Write-Host "[4c] 合成生成: 跳过 (LOCAL ONLY)" -ForegroundColor Yellow
        Write-Host "  请在 GPU 服务器上运行:"
        Write-Host "  python scripts/download/generate_synthetic_kb.py --count NEEDED --use-vllm"
    } elseif (-not $DryRun) {
        Write-Host "[4c] 合成生成: 需要 GPU，请在服务器上运行:" -ForegroundColor Yellow
        Write-Host "  python scripts/download/generate_synthetic_kb.py --count NEEDED --use-vllm"
        Write-Host "  (根据 validate_corpus.py 输出的缺口计算 NEEDED 值)"
    } else {
        python "$ScriptDir\generate_synthetic_kb.py" --count 10 --template-only $DryRunFlag
    }

    Write-Host "Phase 4 完成" -ForegroundColor Green
}

# ═══════════════════════════════════════════════════
# Phase 5: 质量校验（全本地可完成）
# ═══════════════════════════════════════════════════
function Run-Phase5 {
    Write-Host ""
    Write-Host "--- Phase 5: 质量校验 ---" -ForegroundColor Green

    Write-Host "[5a] 生成语料清单..."
    python "$ScriptDir\corpus_manifest.py"

    Write-Host "[5b] 质量校验..."
    python "$ScriptDir\validate_corpus.py" --estimate-chunks

    Write-Host ""
    Write-Host "Phase 5 完成" -ForegroundColor Green
}

# ═══════════════════════════════════════════════════
# Phase 6: 打包上传准备（本地）
# ═══════════════════════════════════════════════════
function Run-Phase6 {
    Write-Host ""
    Write-Host "--- Phase 6: 打包上传准备 ---" -ForegroundColor Green

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $archiveName = "milrag_data_$timestamp.tar.gz"

    Write-Host "打包 data/raw/ 目录..."
    Write-Host "  注意: data/raw/sensitive/ 不在打包范围内"

    # 估算大小
    $rawSize = (Get-ChildItem -Path "data/raw" -Recurse -File |
        Where-Object { $_.FullName -notmatch "sensitive|wikipedia.*dump|\.bz2$" } |
        Measure-Object -Property Length -Sum).Sum / 1MB

    Write-Host "  预估打包大小: $([math]::Round($rawSize, 0)) MB"
    Write-Host ""
    Write-Host "  打包命令 (在 Git Bash 中运行):"
    Write-Host "  tar -czf $archiveName --exclude='data/raw/sensitive' --exclude='*dump*' --exclude='*.bz2' data/raw/ data/kb/ data/qa/"
    Write-Host ""
    Write-Host "  上传到 GPU 服务器后运行:"
    Write-Host "  tar -xzf $archiveName"
    Write-Host "  pip install -r requirements.txt"
    Write-Host "  python scripts/download/generate_synthetic_kb.py --count NEEDED --use-vllm"
    Write-Host "  bash scripts/download/run_acquisition.sh --phase 5"
    Write-Host ""
}

# ═══════════════════════════════════════════════════
# 主调度
# ═══════════════════════════════════════════════════

switch ($Phase) {
    1 { Run-Phase1 }
    2 { Run-Phase2 }
    3 { Run-Phase3 }
    4 { Run-Phase4 }
    5 { Run-Phase5 }
    6 { Run-Phase6 }
    default {
        # 全部运行（本地策略：跳过维基大文件 + GPU 步骤）
        Run-Phase1
        Run-Phase2
        Run-Phase3
        Run-Phase4
        Run-Phase5
        Run-Phase6
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "本地采集完成！" -ForegroundColor Cyan
Write-Host ""
Write-Host "本地已完成的任务:"
Write-Host "  ✅ 中文官方源下载 (Phase 1)"
Write-Host "  ✅ 美军条令 + 国际源 (Phase 3)"
Write-Host "  ✅ 质量校验 + 语料清单 (Phase 5)"
Write-Host ""
Write-Host "需要在 GPU 服务器上完成的任务:"
Write-Host "  🖥️ 维基百科大文件下载 (Phase 2) — 如本地跳过"
Write-Host "  🖥️ 合成段落生成 (Phase 4) — 需要 Qwen3-8B"
Write-Host "  🖥️ QA 数据集构造 — 需要 Qwen3-32B"
Write-Host "  🖥️ 运行全部 31 个实验 — 需要 GPU"
Write-Host "============================================" -ForegroundColor Cyan
