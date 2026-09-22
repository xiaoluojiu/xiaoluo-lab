#!/usr/bin/env bash
# 小洛实验室 开发环境一键启动（Prompt 234）
# 用法：在项目根目录 xiaoluo-lab/ 下执行
#   bash scripts/dev_start.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

c_ok="\033[32m"
c_err="\033[31m"
c_step="\033[36m"
c_reset="\033[0m"

step() { printf "\n${c_step}==> %s${c_reset}\n" "$1"; }
ok()   { printf "    OK  %s\n" "$1"; }
err()  { printf "    ${c_err}ERR %s${c_reset}\n" "$1" >&2; }

cleanup() {
    step "清理子进程"
    [ -n "${BACKEND_PID:-}" ] && kill "$BACKEND_PID" 2>/dev/null || true
    [ -n "${FRONTEND_PID:-}" ] && kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---------- 1. 依赖检查 ----------
step "检查依赖"

if ! command -v python3 >/dev/null 2>&1; then
    err "未找到 python3"; exit 1
fi
PYVER=$(python3 --version 2>&1)
ok "Python: $PYVER"

if ! command -v node >/dev/null 2>&1; then
    err "未找到 node"; exit 1
fi
NODEVER=$(node --version 2>&1)
ok "Node: $NODEVER"

# ---------- 2. 后端 ----------
step "准备后端（$BACKEND）"

if [ ! -d "$BACKEND/.venv" ]; then
    echo "    创建 .venv..."
    python3 -m venv "$BACKEND/.venv"
fi

PY="$BACKEND/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1

echo "    安装依赖（dev）..."
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -e "$BACKEND[dev]"

mkdir -p "$BACKEND/data"

echo "    执行 alembic 迁移..."
( cd "$BACKEND" && "$PY" -m alembic upgrade head )

# 端口占用检查：残留 uvicorn 会抢占 8000，使本次启动的后端绑定失败；
# 残留进程常由缺少项目依赖的解释器启动，只 bind 端口却无法服务，
# 前端表现即所有接口一直 Pending 直到超时。这里先释放再启动。
if lsof -ti tcp:8000 >/dev/null 2>&1; then
    echo "    端口 8000 已被占用，尝试释放..."
    lsof -ti tcp:8000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

echo "    启动 uvicorn（后台）..."
( cd "$BACKEND" && "$PY" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 ) &
BACKEND_PID=$!

# 轻量 health polling：最多 30 次 × 1s，访问 /api/v1/health 成功后继续；超时则报错并清理进程
# 统一使用 127.0.0.1，与 frontend/vite.config.ts 的 proxy target 保持一致
health_url="http://127.0.0.1:8000/api/v1/health"
tries=0
max_tries=30
while [ "$tries" -lt "$max_tries" ]; do
    if curl -sf "$health_url" >/dev/null 2>&1; then
        ok "后端已启动：$health_url（等待 ${tries}s）"
        break
    fi
    tries=$((tries + 1))
    sleep 1
done
if [ "$tries" -ge "$max_tries" ]; then
    err "后端未在 ${max_tries}s 内就绪（$health_url 不通），清理进程后退出"
    cleanup
    exit 1
fi

# ---------- 3. 前端 ----------
step "准备前端（$FRONTEND）"

cd "$FRONTEND"
if [ ! -d node_modules ]; then
    echo "    npm install..."
    npm install --no-audit --no-fund
fi
echo "    启动 vite（后台）..."
npm run dev &
FRONTEND_PID=$!

# 前端 health polling：vite 起来后 5173 端口可连通即可
fe_url="http://127.0.0.1:5173/"
fe_tries=0
fe_max=30
while [ "$fe_tries" -lt "$fe_max" ]; do
    if curl -sf "$fe_url" >/dev/null 2>&1; then
        ok "前端已启动：$fe_url（等待 ${fe_tries}s）"
        break
    fi
    fe_tries=$((fe_tries + 1))
    sleep 1
done
if [ "$fe_tries" -ge "$fe_max" ]; then
    err "前端未在 ${fe_max}s 内就绪（$fe_url 不通），清理进程后退出"
    cleanup
    exit 1
fi

# ---------- 4. 等待退出 ----------
step "开发环境已就绪"
echo "    后端 API:  http://localhost:8000/api/v1/health"
echo "    前端 UI:   http://localhost:5173"
printf "    ${c_step}按 Ctrl+C 停止所有服务...${c_reset}\n"

wait
