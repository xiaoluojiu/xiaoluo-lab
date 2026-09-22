# 小洛实验室 Demo 脚本（T0 整改后版本）
# 用法：在项目根目录 xiaoluo-lab/ 下执行
#   powershell -File scripts/demo.ps1      # Windows PowerShell 5.1
#   pwsh scripts/demo.ps1                  # PowerShell 7+
# 完整流程（使用真实 /api/v1 端点）：
#   1. 生成 Demo 数据（CSV：users / orders / events）
#   2. 通过 /files/upload + /datasets 上传并创建数据集（自动创建 v1）
#   3. /datasets/{id}/profile 数据画像
#   4. /merge/mapping + /merge/keys + /merge/preview + /merge/execute
#      把 orders 合并到 users（left join on user_id ↔ userId）
#   5. /datasets/{id}/eda/descriptive + /distribution + /correlation + /outlier
#   6. /ml/train 训练 logistic_regression（classification）
#   7. /reports/generate 生成 Demo 报告
#
# 任何一步失败都会以非零退出码终止并打印 ERR，不会输出“成功”。

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$ROOT = Resolve-Path "$PSScriptRoot/.."
$BACKEND = Join-Path $ROOT "backend"
$DATA = Join-Path $BACKEND "data/demo"

function Write-Step([string]$m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Ok([string]$m) { Write-Host "    OK  $m" -ForegroundColor Green }
function Write-Err([string]$m) { Write-Host "    ERR $m" -ForegroundColor Red }

# 以 UTF-8 字节发送 JSON，并按 UTF-8 解码响应。
# 兼容 Windows PowerShell 5.1：默认会把请求中文转成 ?、把响应按 Latin-1 解码导致乱码。
function Invoke-PostJson([string]$uri, $obj, [int]$timeoutSec = 60) {
    $json = $obj | ConvertTo-Json -Depth 8
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    $resp = Invoke-WebRequest -Method Post -Uri $uri -Body $bytes -ContentType "application/json; charset=utf-8" -TimeoutSec $timeoutSec -UseBasicParsing
    $raw = [System.Text.Encoding]::UTF8.GetString($resp.RawContentStream.ToArray())
    return ($raw | ConvertFrom-Json)
}

# ---------- 前置 ----------
$py = Join-Path $BACKEND ".venv/Scripts/python.exe"
if (-not (Test-Path $py)) {
    $py = Join-Path $BACKEND ".venv/bin/python"
}
if (-not (Test-Path $py)) {
    Write-Err "未找到后端 venv，请先运行 scripts/init.ps1"
    exit 1
}
$env:PYTHONDONTWRITEBYTECODE = "1"

# ---------- 1. 生成 Demo 数据 ----------
Write-Step "1. 生成 Demo 数据"
Push-Location $BACKEND
& $py -m scripts.demo_data --out $DATA --formats csv
$genExit = $LASTEXITCODE
Pop-Location
if ($genExit -ne 0) {
    Write-Err "Demo 数据生成失败"
    exit 1
}
Write-Ok "数据已生成到 $DATA"

# ---------- 2. 通过 API 导入数据集 ----------
Write-Step "2. 通过 API 导入数据集"

$api = "http://localhost:8000/api/v1"
$healthOk = $false
try {
    $h = Invoke-WebRequest "$api/health" -UseBasicParsing -TimeoutSec 3
    if ($h.StatusCode -eq 200) { $healthOk = $true }
} catch { }
if (-not $healthOk) {
    Write-Err "后端未启动（$api/health 不通），请先运行 scripts/dev_start.ps1"
    exit 1
}

function Import-Dataset([string]$name, [string]$file, [string]$description) {
    # 上传文件（multipart/form-data）。用系统自带 curl.exe，兼容 Windows PowerShell 5.1（无 -Form）与 7+。
    $uploadJson = & curl.exe -sS -f -X POST "$api/files/upload" -F "file=@$file"
    if ($LASTEXITCODE -ne 0) { throw "文件上传失败: $file" }
    $fileId = ($uploadJson | ConvertFrom-Json).data.id

    # 创建数据集（绑 source_file_id 自动加载初始版本 v1）
    $dsResp = Invoke-PostJson "$api/datasets" @{ name = $name; description = $description; source_file_id = $fileId } 30
    return $dsResp.data.id
}

$usersId = Import-Dataset "demo_users" (Join-Path $DATA "users.csv") "Demo 用户表"
$ordersId = Import-Dataset "demo_orders" (Join-Path $DATA "orders.csv") "Demo 订单表"
$eventsId = Import-Dataset "demo_events" (Join-Path $DATA "events.csv") "Demo 活动表"
Write-Ok "users#$usersId orders#$ordersId events#$eventsId"

# ---------- 3. Profile ----------
Write-Step "3. 数据画像"
foreach ($ds in @($usersId, $ordersId, $eventsId)) {
    $p = Invoke-RestMethod "$api/datasets/$ds/profile" -TimeoutSec 30
    $rows = $p.data.row_count
    $cols = $p.data.column_count
    Write-Host "    dataset#$ds profile: rows=$rows cols=$cols"
}

# ---------- 4. Merge（users ← orders，left join on user_id ↔ userId）----------
Write-Step "4. Merge：users ← orders（left join on user_id ↔ userId）"

# 4.1 字段映射建议
$mappings = Invoke-PostJson "$api/merge/mapping" @{ left_dataset_id = $usersId; right_dataset_id = $ordersId } 30
$mapCount = @($mappings.data).Count
$hits = @($mappings.data | Where-Object { $_.source_column -eq "user_id" -and $_.target_column -eq "userId" })
Write-Host "    mapping 候选数: $mapCount; user_id↔userId 命中: $($hits.Count)"

# 4.2 Key 分析
$keys = Invoke-PostJson "$api/merge/keys" @{ left_dataset_id = $usersId; right_dataset_id = $ordersId; left_key = "user_id"; right_key = "userId" } 30
$card = $keys.data.cardinality
$cov = $keys.data.left_key_coverage_in_right
Write-Host "    cardinality: $card; left_coverage_in_right: $cov"

# 4.3 Preview（不创建版本）
$plan = @{
    left_dataset_id  = $usersId
    right_dataset_id = $ordersId
    plan = @{
        keys = @(@{ left = "user_id"; right = "userId" })
        join_type = "left"
    }
}
$preview = Invoke-PostJson "$api/merge/preview" $plan 60
$il = $preview.data.input_rows_left
$ir = $preview.data.input_rows_right
$or = $preview.data.output_rows
$oc = $preview.data.output_columns
Write-Host "    preview: 输入 $il/$ir 行 → 输出 $or 行 × $oc 列"
Write-Host "    preview 左版本: v$($preview.data.left_version) / 右版本: v$($preview.data.right_version)"

# 4.4 Execute（T0-M4：必须显式指定版本，使用 preview 返回的版本）
$exec = @{
    left_dataset_id  = $usersId
    right_dataset_id = $ordersId
    left_version     = $preview.data.left_version
    right_version    = $preview.data.right_version
    plan = @{
        keys = @(@{ left = "user_id"; right = "userId" })
        join_type = "left"
    }
}
$merge = Invoke-PostJson "$api/merge/execute" $exec 60
$mver = $merge.data.version.version
$mrows = $merge.data.version.row_count
$mcols = $merge.data.version.column_count
Write-Host "    merge 完成：新版本 v$mver（$mrows 行 × $mcols 列）"
$matched = $merge.data.report.matched_rows
$unmatched = $merge.data.report.unmatched_rows_left
Write-Host "    matched=$matched unmatched_left=$unmatched"
Write-Ok "users ← orders 合并完成（新版本 v$mver）"

# ---------- 5. EDA ----------
Write-Step "5. 探索性数据分析（EDA）"
$desc = Invoke-RestMethod "$api/datasets/$eventsId/eda/descriptive" -TimeoutSec 30
$descCols = @($desc.data.columns).Count
Write-Host "    describe: rows=$($desc.data.row_count) cols=$descCols"

$dist = Invoke-RestMethod "$api/datasets/$eventsId/eda/distribution?column=value" -TimeoutSec 30
Write-Host "    distribution[value]: type=$($dist.data.type)"

$corr = Invoke-RestMethod "$api/datasets/$eventsId/eda/correlation" -TimeoutSec 30
$corrCols = ($corr.data.matrix.PSObject.Properties.Name) -join ", "
Write-Host "    correlation 列: $corrCols"

$outlier = Invoke-RestMethod "$api/datasets/$eventsId/eda/outlier" -TimeoutSec 30
$outlierTotal = 0
foreach ($c in @($outlier.data.columns)) { $outlierTotal += [int]$c.outlier_count }
Write-Host "    outlier 总异常数: $outlierTotal（$($outlier.data.method)）"
Write-Ok "EDA 完成"

# ---------- 6. ML 训练 ----------
Write-Step "6. 机器学习训练（logistic_regression / classification）"
$train = Invoke-PostJson "$api/ml/train" @{
    dataset_id       = $eventsId
    task             = "classification"
    model            = "logistic_regression"
    target_column    = "event_type"
    excluded_columns = @("event_id")
    seed             = 42
} 120
if ($train.success) {
    $run = $train.data.run
    if ($run.status -eq "success") {
        $acc = $run.metrics.accuracy
        $f1 = $run.metrics.f1
        Write-Host "    训练成功: run_status=$($run.status) accuracy=$acc f1=$f1"
    } else {
        Write-Err "训练失败: $($run.error)"
        exit 1
    }
} else {
    Write-Err "API 失败: $($train.error)"
    exit 1
}
Write-Ok "ML 训练完成"

# ---------- 7. 报告 ----------
Write-Step "7. 生成 Demo 报告"
$report = Invoke-PostJson "$api/reports/generate" @{ dataset_id = $eventsId; title = "Demo 报告" } 60
$sections = @($report.data.sections)
Write-Host "    报告标题: $($report.data.title); 章节数: $($sections.Count)"
Write-Ok "报告生成完成"

# ---------- 摘要 ----------
Write-Step "Demo 完成"
Write-Host "    数据目录:   $DATA"
Write-Host "    数据集 ID:   users#$usersId  orders#$ordersId  events#$eventsId"
Write-Host "    合并后版本:  users v$mver"
Write-Host "    下一步：在浏览器打开 http://localhost:5173 查看可视化" -ForegroundColor Yellow
