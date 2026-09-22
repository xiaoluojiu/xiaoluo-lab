# 小洛实验室 项目初始化（Prompt 235）
# 用法：在项目根目录 xiaoluo-lab/ 下执行
#   pwsh scripts/init.ps1
# 行为：
#   1. 创建 data/ models/ 等必要目录
#   2. 初始化 SQLite 数据库（如不存在）
#   3. 运行 Alembic 迁移到最新版本
#   4. 打印初始化摘要

$ErrorActionPreference = "Stop"
$ROOT = Resolve-Path "$PSScriptRoot/.."
$BACKEND = Join-Path $ROOT "backend"
$DATA = Join-Path $BACKEND "data"
$MODELS = Join-Path $BACKEND "models"

function Write-Step([string]$m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Ok([string]$m)   { Write-Host "    OK  $m" -ForegroundColor Green }

Write-Step "小洛实验室 初始化"

# ---------- 1. 必要目录 ----------
Write-Step "创建必要目录"
foreach ($d in @($DATA, $MODELS)) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
        Write-Ok "创建 $d"
    } else {
        Write-Ok "已存在 $d"
    }
}

# ---------- 2. Python venv ----------
Write-Step "检查 Python 虚拟环境"
$venv = Join-Path $BACKEND ".venv"
if (-not (Test-Path $venv)) {
    Write-Host "    创建 .venv..."
    & python -m venv $venv
}
$py = Join-Path $venv "Scripts/python.exe"
Write-Ok "Python: $venv"

# ---------- 3. 安装依赖 ----------
Write-Step "安装后端依赖"
$env:PYTHONDONTWRITEBYTECODE = "1"
& $py -m pip install --quiet --upgrade pip
& $py -m pip install --quiet -e "$BACKEND[dev]"
Write-Ok "依赖已安装"

# ---------- 4. 数据库初始化 ----------
Write-Step "初始化数据库"
$dbPath = Join-Path $DATA "xiaoluo.db"
if (Test-Path $dbPath) {
    Write-Ok "数据库已存在：$dbPath"
} else {
    Write-Host "    首次创建：$dbPath"
}

# ---------- 5. 运行迁移 ----------
Write-Step "执行 Alembic 迁移"
Push-Location $BACKEND
& $py -m alembic upgrade head
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Write-Host "    迁移失败" -ForegroundColor Red
    exit 1
}
Pop-Location
Write-Ok "数据库已迁移到最新版本"

# ---------- 6. 摘要 ----------
Write-Step "初始化完成"
Write-Host "    项目根:   $ROOT"
Write-Host "    后端:     $BACKEND"
Write-Host "    数据目录: $DATA"
Write-Host "    模型目录: $MODELS"
Write-Host "    数据库:   $dbPath"
Write-Host ""
Write-Host "    下一步：" -ForegroundColor Yellow
Write-Host "      开发启动:  pwsh scripts/dev_start.ps1"
Write-Host "      运行测试:  cd backend; .\.venv\Scripts\python.exe -m pytest"
Write-Host "      运行 Demo: pwsh scripts/demo.ps1"
