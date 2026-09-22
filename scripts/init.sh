#!/usr/bin/env bash
# 小洛实验室 项目初始化（Prompt 235）
# 用法：在项目根目录 xiaoluo-lab/ 下执行
#   bash scripts/init.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
BACKEND="$ROOT/backend"
DATA="$BACKEND/data"
MODELS="$BACKEND/models"

c_ok="\033[32m"
c_step="\033[36m"
c_reset="\033[0m"

step() { printf "\n${c_step}==> %s${c_reset}\n" "$1"; }
ok()   { printf "    ${c_ok}OK  %s${c_reset}\n" "$1"; }

step "小洛实验室 初始化"

# ---------- 1. 必要目录 ----------
step "创建必要目录"
for d in "$DATA" "$MODELS"; do
    if [ ! -d "$d" ]; then
        mkdir -p "$d"
        ok "创建 $d"
    else
        ok "已存在 $d"
    fi
done

# ---------- 2. Python venv ----------
step "检查 Python 虚拟环境"
VENV="$BACKEND/.venv"
if [ ! -d "$VENV" ]; then
    echo "    创建 .venv..."
    python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"
ok "Python: $VENV"

# ---------- 3. 安装依赖 ----------
step "安装后端依赖"
export PYTHONDONTWRITEBYTECODE=1
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -e "$BACKEND[dev]"
ok "依赖已安装"

# ---------- 4. 数据库 ----------
step "初始化数据库"
DB="$DATA/xiaoluo.db"
if [ -f "$DB" ]; then
    ok "数据库已存在：$DB"
else
    echo "    首次创建：$DB"
fi

# ---------- 5. 迁移 ----------
step "执行 Alembic 迁移"
( cd "$BACKEND" && "$PY" -m alembic upgrade head )
ok "数据库已迁移到最新版本"

# ---------- 6. 摘要 ----------
step "初始化完成"
echo "    项目根:   $ROOT"
echo "    后端:     $BACKEND"
echo "    数据目录: $DATA"
echo "    模型目录: $MODELS"
echo "    数据库:   $DB"
echo ""
printf "    ${c_step}下一步：${c_reset}\n"
echo "      开发启动:  bash scripts/dev_start.sh"
echo "      运行测试:  cd backend && .venv/bin/python -m pytest"
echo "      运行 Demo: bash scripts/demo.sh"
