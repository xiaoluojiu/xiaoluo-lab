# 小洛实验室 开发环境一键启动（Prompt 234）
# 用法：在项目根目录 xiaoluo-lab/ 下执行
#   pwsh scripts/dev_start.ps1
# 行为：
#   1. 检查 Python / Node 版本
#   2. 后端：创建 .venv、安装依赖、运行迁移、启动 uvicorn（后台）
#   3. 前端：npm install、启动 vite dev server（后台）
#   4. 打印访问地址，Ctrl+C 退出时自动清理子进程

$ErrorActionPreference = "Stop"
$ROOT = Resolve-Path "$PSScriptRoot/.."
$BACKEND = Join-Path $ROOT "backend"
$FRONTEND = Join-Path $ROOT "frontend"

function Write-Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host "    OK  $msg" -ForegroundColor Green }
function Write-Err([string]$msg)  { Write-Host "    ERR $msg" -ForegroundColor Red }

# ---------- 1. 依赖检查 ----------
Write-Step "检查依赖"

# Python >= 3.11
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Err "未找到 python，请先安装 Python 3.11+"; exit 1
}
$pyver = (& python --version) 2>&1
Write-Ok "Python: $pyver"
$pyverMatch = [regex]::Match($pyver, "Python (\d+)\.(\d+)")
if ($pyverMatch.Success -and ([int]$pyverMatch.Groups[1].Value -lt 3 -or ([int]$pyverMatch.Groups[1].Value -eq 3 -and [int]$pyverMatch.Groups[2].Value -lt 11))) {
    Write-Err "需要 Python 3.11+"; exit 1
}

# Node >= 18
$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) {
    Write-Err "未找到 node，请先安装 Node.js 18+"; exit 1
}
$nodever = (& node --version) 2>&1
Write-Ok "Node: $nodever"

# ---------- 2. 后端 ----------
Write-Step "准备后端（$BACKEND）"

if (-not (Test-Path (Join-Path $BACKEND ".venv"))) {
    Write-Host "    创建 .venv..."
    & python -m venv "$BACKEND/.venv"
}

$env:PYTHONDONTWRITEBYTECODE = "1"
$py = Join-Path $BACKEND ".venv/Scripts/python.exe"
Write-Host "    安装依赖（dev）..."
& $py -m pip install --quiet --upgrade pip
& $py -m pip install --quiet -e "$BACKEND[dev]"

# 数据目录 + 迁移
$dataDir = Join-Path $BACKEND "data"
if (-not (Test-Path $dataDir)) { New-Item -ItemType Directory -Path $dataDir | Out-Null }

Write-Host "    执行 alembic 迁移..."
Push-Location $BACKEND
& $py -m alembic upgrade head
if ($LASTEXITCODE -ne 0) { Write-Err "迁移失败"; Pop-Location; exit 1 }
Pop-Location

# 端口占用检查：残留的 uvicorn 会抢占 8000，导致本次启动的后端绑定失败；
# 而残留进程往往是用系统 Python 启动的（缺少项目依赖），只 bind 端口却无法真正服务，
# 前端表现就是所有接口一直 Pending，直到 axios 超时。这里统一先释放再启动。
$port = 8000
$holder = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
if ($holder) {
    $pids = @($holder | Select-Object -ExpandProperty OwningProcess -Unique)
    Write-Host "    端口 $port 已被占用（PID: $($pids -join ', ')），尝试释放..."
    foreach ($pid2 in $pids) {
        $proc = Get-Process -Id $pid2 -ErrorAction SilentlyContinue
        if ($proc -and $proc.ProcessName -match '^python') {
            Stop-Process -Id $pid2 -Force -ErrorAction SilentlyContinue
            Write-Ok "    已终止残留后端进程 PID=$pid2"
        } else {
            Write-Err "    端口 $port 被非 Python 进程占用（PID=$pid2, $($proc.ProcessName)），请手动释放后重试"
            exit 1
        }
    }
    Start-Sleep -Seconds 2
}

Write-Host "    启动 uvicorn（后台）..."
$backendJob = Start-Job -Name "xiaoluo-backend" -ScriptBlock {
    param($py, $BACKEND)
    Set-Location $BACKEND
    & $py -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
} -ArgumentList $py, $BACKEND

# 轻量 health polling：最多 30 次 × 1s，访问 /api/v1/health 成功后继续；超时则清理进程
# 统一使用 127.0.0.1：与 frontend/vite.config.ts 的 proxy target 保持一致，
# 避免 Windows 上 localhost 被解析到 ::1 而 uvicorn 只监听 IPv4 导致误判"后端未就绪"。
$healthUrl = "http://127.0.0.1:8000/api/v1/health"
$tries = 0
$maxTries = 30
$ready = $false
while ($tries -lt $maxTries) {
    try {
        $h = Invoke-WebRequest $healthUrl -UseBasicParsing -TimeoutSec 2
        if ($h.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
    $tries++
    Start-Sleep -Seconds 1
}
if (-not $ready) {
    Write-Err "后端未在 ${maxTries}s 内就绪（$healthUrl 不通），清理进程后退出"
    Stop-Job $backendJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob -ErrorAction SilentlyContinue
    exit 1
}
Write-Ok "后端已启动：$healthUrl（等待 ${tries}s）"

# ---------- 3. 前端 ----------
Write-Step "准备前端（$FRONTEND）"

Push-Location $FRONTEND
if (-not (Test-Path "node_modules")) {
    Write-Host "    npm install..."
    & npm install --no-audit --no-fund
}
Write-Host "    启动 vite（后台）..."
$frontendJob = Start-Job -Name "xiaoluo-frontend" -ScriptBlock {
    param($FRONTEND)
    Set-Location $FRONTEND
    & npm run dev
} -ArgumentList $FRONTEND
Pop-Location

# 前端 health polling：vite 起来后 5173 端口可连通即可（同样用 127.0.0.1 保持一致）
$feUrl = "http://127.0.0.1:5173/"
$feTries = 0
$feMax = 30
$feReady = $false
while ($feTries -lt $feMax) {
    try {
        $f = Invoke-WebRequest $feUrl -UseBasicParsing -TimeoutSec 2
        if ($f.StatusCode -eq 200) { $feReady = $true; break }
    } catch { }
    $feTries++
    Start-Sleep -Seconds 1
}
if (-not $feReady) {
    Write-Err "前端未在 ${feMax}s 内就绪（$feUrl 不通），清理进程后退出"
    Stop-Job $backendJob -ErrorAction SilentlyContinue
    Stop-Job $frontendJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob -ErrorAction SilentlyContinue
    Remove-Job $frontendJob -ErrorAction SilentlyContinue
    exit 1
}
Write-Ok "前端已启动：$feUrl（等待 ${feTries}s）"

# ---------- 4. 等待退出 ----------
Write-Step "开发环境已就绪"
Write-Host "    后端 API:  http://localhost:8000/api/v1/health"
Write-Host "    前端 UI:   http://localhost:5173"
Write-Host "    按 Ctrl+C 停止所有服务..." -ForegroundColor Yellow

try {
    while ($true) {
        $be = Receive-Job $backendJob -ErrorAction SilentlyContinue
        $fe = Receive-Job $frontendJob -ErrorAction SilentlyContinue
        if ($be) { Write-Host "[backend] $be" -ForegroundColor DarkGray }
        if ($fe) { Write-Host "[frontend] $fe" -ForegroundColor DarkGray }
        Start-Sleep -Milliseconds 500
        if ($backendJob.State -eq "Completed" -or $frontendJob.State -eq "Completed") {
            Write-Err "子进程退出，停止全部"; break
        }
    }
} finally {
    Write-Host "`n==> 清理子进程..." -ForegroundColor Cyan
    Stop-Job $backendJob -ErrorAction SilentlyContinue
    Stop-Job $frontendJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob -ErrorAction SilentlyContinue
    Remove-Job $frontendJob -ErrorAction SilentlyContinue
}
