#!/usr/bin/env bash
# 小洛实验室 Demo 脚本（T0 整改后版本）
# 用法：在项目根目录 xiaoluo-lab/ 下执行
#   bash scripts/demo.sh
# 完整流程（使用真实 /api/v1 端点）：
#   1. 生成 Demo 数据（CSV：users / orders / events）
#   2. 通过 /files/upload + /datasets 上传并创建数据集（自动创建 v1）
#   3. /datasets/{id}/profile 数据画像
#   4. /merge/mapping + /merge/keys + /merge/preview + /merge/execute
#      把 orders 合并到 users（left join on user_id ↔ userId）
#   5. /datasets/{id}/eda/descriptive + /distribution + /correlation + /outlier
#   6. /ml/train 训练 logistic_regression（classification）
#   7. /reports/generate 生成 Demo 报告
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
BACKEND="$ROOT/backend"
DATA="$BACKEND/data/demo"

c_ok="\033[32m"
c_err="\033[31m"
c_step="\033[36m"
c_yel="\033[33m"
c_reset="\033[0m"

step() { printf "\n${c_step}==> %s${c_reset}\n" "$1"; }
ok()   { printf "    ${c_ok}OK  %s${c_reset}\n" "$1"; }
err()  { printf "    ${c_err}ERR %s${c_reset}\n" "$1" >&2; }

# ---------- 前置 ----------
PY="$BACKEND/.venv/bin/python"
[ -f "$PY" ] || PY="$BACKEND/.venv/Scripts/python.exe"
if [ ! -f "$PY" ]; then
    err "未找到后端 venv，请先运行 scripts/init.sh"
    exit 1
fi
export PYTHONDONTWRITEBYTECODE=1

# ---------- 1. 生成 Demo 数据 ----------
step "1. 生成 Demo 数据"
( cd "$BACKEND" && "$PY" -m scripts.demo_data --out "$DATA" --formats csv )
ok "数据已生成到 $DATA"

# ---------- 2. 通过 API 导入数据集 ----------
step "2. 通过 API 导入数据集"

API="http://localhost:8000/api/v1"
if ! curl -sf "$API/health" >/dev/null 2>&1; then
    err "后端未启动（$API/health 不通），请先运行 scripts/dev_start.sh"
    exit 1
fi

# 上传文件 → 创建数据集（绑 source_file_id 自动加载初始版本 v1）
create_dataset() {
    local name="$1" file="$2" desc="$3"
    local file_id
    file_id=$(curl -sf -X POST "$API/files/upload" \
        -F "file=@$file" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['id'])")
    local ds_id
    ds_id=$(curl -sf -X POST "$API/datasets" \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"$name\",\"description\":\"$desc\",\"source_file_id\":$file_id}" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['id'])")
    echo "$ds_id"
}

USERS_ID=$(create_dataset "demo_users" "$DATA/users.csv" "Demo 用户表")
ORDERS_ID=$(create_dataset "demo_orders" "$DATA/orders.csv" "Demo 订单表")
EVENTS_ID=$(create_dataset "demo_events" "$DATA/events.csv" "Demo 活动表")
ok "users#$USERS_ID orders#$ORDERS_ID events#$EVENTS_ID"

# ---------- 3. Profile ----------
step "3. 数据画像"
for ds in "$USERS_ID" "$ORDERS_ID" "$EVENTS_ID"; do
    curl -sf "$API/datasets/$ds/profile" | python3 -c "
import sys, json
d = json.load(sys.stdin)['data']
print(f'    dataset#$ds profile: rows={d.get(\"row_count\")} cols={d.get(\"column_count\")}')
"
done

# ---------- 4. Merge（users ← orders，left join on user_id ↔ userId）----------
step "4. Merge：users ← orders（left join on user_id ↔ userId）"

# 4.1 字段映射建议
curl -sf -X POST "$API/merge/mapping" \
    -H "Content-Type: application/json" \
    -d "{\"left_dataset_id\":$USERS_ID,\"right_dataset_id\":$ORDERS_ID}" \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)['data']
# 找出 user_id ↔ userId 的候选
hits = [c for c in d if c.get('source_column') == 'user_id' and c.get('target_column') == 'userId']
print(f'    mapping 候选数: {len(d)}; user_id↔userId 命中: {len(hits)}')
"

# 4.2 Key 分析
curl -sf -X POST "$API/merge/keys" \
    -H "Content-Type: application/json" \
    -d "{\"left_dataset_id\":$USERS_ID,\"right_dataset_id\":$ORDERS_ID,\"left_key\":\"user_id\",\"right_key\":\"userId\"}" \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)['data']
print(f'    cardinality: {d[\"cardinality\"]}; left_coverage_in_right: {d[\"left_key_coverage_in_right\"]}')
"

# 4.3 Preview（不创建版本）
PLAN_JSON=$(python3 -c "
import json
print(json.dumps({
    'left_dataset_id': $USERS_ID,
    'right_dataset_id': $ORDERS_ID,
    'plan': {
        'keys': [{'left': 'user_id', 'right': 'userId'}],
        'join_type': 'left',
    },
}))
")
curl -sf -X POST "$API/merge/preview" \
    -H "Content-Type: application/json" \
    -d "$PLAN_JSON" | python3 -c "
import sys, json
d = json.load(sys.stdin)['data']
print(f'    preview: 输入 {d[\"input_rows_left\"]}/{d[\"input_rows_right\"]} 行 → 输出 {d[\"output_rows\"]} 行 × {d[\"output_columns\"]} 列')
print(f'    preview 左版本: v{d[\"left_version\"]} / 右版本: v{d[\"right_version\"]}')
"

# 4.4 Execute（T0-M4：必须显式指定版本，使用 preview 返回的 v1）
EXEC_JSON=$(python3 -c "
import json
print(json.dumps({
    'left_dataset_id': $USERS_ID,
    'right_dataset_id': $ORDERS_ID,
    'left_version': 1,
    'right_version': 1,
    'plan': {
        'keys': [{'left': 'user_id', 'right': 'userId'}],
        'join_type': 'left',
    },
}))
")
curl -sf -X POST "$API/merge/execute" \
    -H "Content-Type: application/json" \
    -d "$EXEC_JSON" | python3 -c "
import sys, json
d = json.load(sys.stdin)['data']
print(f'    merge 完成：新版本 v{d[\"version\"][\"version\"]}（{d[\"version\"][\"row_count\"]} 行 × {d[\"version\"][\"column_count\"]} 列）')
print(f'    matched={d[\"report\"][\"matched_rows\"]} unmatched_left={d[\"report\"][\"unmatched_rows_left\"]}')
"
ok "users ← orders 合并完成（新版本 v2）"

# ---------- 5. EDA ----------
step "5. 探索性数据分析（EDA）"
DESC=$(curl -sf "$API/datasets/$EVENTS_ID/eda/descriptive")
echo "$DESC" | python3 -c "
import sys, json
d = json.load(sys.stdin)['data']
print(f'    describe: rows={d.get(\"row_count\")} cols={len(d.get(\"columns\", []))}')
"
DIST=$(curl -sf "$API/datasets/$EVENTS_ID/eda/distribution?column=value")
echo "$DIST" | python3 -c "
import sys, json
d = json.load(sys.stdin)['data']
print(f'    distribution[value]: type={d.get(\"type\")}')
"
CORR=$(curl -sf "$API/datasets/$EVENTS_ID/eda/correlation")
echo "$CORR" | python3 -c "
import sys, json
d = json.load(sys.stdin)['data']
print(f'    correlation 列: {list(d.get(\"matrix\", {}).keys())}')
"
OUTLIER=$(curl -sf "$API/datasets/$EVENTS_ID/eda/outlier")
echo "$OUTLIER" | python3 -c "
import sys, json
d = json.load(sys.stdin)['data']
total = sum(int(c.get('outlier_count', 0)) for c in d.get('columns', []))
print(f'    outlier 总异常数: {total}（{d.get(\"method\")}）')
"
ok "EDA 完成"

# ---------- 6. ML 训练 ----------
step "6. 机器学习训练（logistic_regression / classification）"
TRAIN_JSON=$(python3 -c "
import json
print(json.dumps({
    'dataset_id': $EVENTS_ID,
    'task': 'classification',
    'model': 'logistic_regression',
    'target_column': 'event_type',
    'excluded_columns': ['event_id'],
    'seed': 42,
}))
")
curl -sf -X POST "$API/ml/train" \
    -H "Content-Type: application/json" \
    -d "$TRAIN_JSON" | python3 -c "
import sys, json
r = json.load(sys.stdin)
if r.get('success'):
    run = r['data']['run']
    if run['status'] == 'success':
        m = run['metrics']
        print(f'    训练成功: accuracy={m.get(\"accuracy\"):.3f} f1={m.get(\"f1\"):.3f}')
    else:
        print(f'    训练失败: {run.get(\"error\")}')
        sys.exit(1)
else:
    print(f'    API 失败: {r.get(\"error\")}')
    sys.exit(1)
"
ok "ML 训练完成"

# ---------- 7. 报告 ----------
step "7. 生成 Demo 报告"
curl -sf -X POST "$API/reports/generate" \
    -H "Content-Type: application/json" \
    -d "{\"dataset_id\":$EVENTS_ID,\"title\":\"Demo 报告\"}" | python3 -c "
import sys, json
d = json.load(sys.stdin)['data']
print(f'    报告标题: {d.get(\"title\")}; 章节数: {len(d.get(\"sections\", []))}')
"
ok "报告生成完成"

# ---------- 摘要 ----------
step "Demo 完成"
echo "    数据目录:   $DATA"
echo "    数据集 ID:   users#$USERS_ID  orders#$ORDERS_ID  events#$EVENTS_ID"
echo "    合并后版本:  users v2"
printf "    ${c_yel}下一步：在浏览器打开 http://localhost:5173 查看可视化${c_reset}\n"
