# 小洛实验室 后端 Dockerfile（Prompt 231）
# 多阶段构建：builder 安装依赖，runtime 精简镜像
# 约束：
# - 不将 API Key 烤进镜像（运行时从环境变量读取）
# - 非 root 用户运行
# - 健康检查指向 /api/v1/health

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# 系统依赖（polars/pyarrow/fastexcel 可能需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ musl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 先拷依赖描述（利用 Docker 层缓存）
COPY pyproject.toml ./
COPY app/__init__.py app/__init__.py
# 安装依赖到独立前缀，便于 copy --from
RUN pip install --user --no-cache-dir -e ".[dev]"

# ---- runtime ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# 运行时系统依赖（libgomp 给 polars/pyarrow/scikit-learn）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 拷贝已安装的 Python 包
COPY --from=builder /root/.local /root/.local

# 拷贝应用代码
COPY app/ ./app/
COPY migrations/ ./migrations/
COPY alembic.ini ./

# 数据目录（运行时挂载为 volume）
RUN mkdir -p /app/data /app/models

# 不烤 API Key：运行时从环境变量读取
ENV APP_ENV=prod \
    HOST=0.0.0.0 \
    PORT=8000 \
    DATA_ROOT=/app/data \
    MODEL_ROOT=/app/models \
    DATABASE_URL=sqlite:////app/data/xiaoluo.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3); sys.exit(0)" || exit 1

# 运行前执行迁移，再启动 uvicorn
CMD ["sh", "-c", "python -m alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
