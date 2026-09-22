# 小洛实验室 前端 Dockerfile（Prompt 232）
# 多阶段构建：node builder → nginx 静态服务
# builder：安装依赖、TypeScript 编译、Vite 构建
# runtime：nginx serve dist/，反向代理 /api → backend:8000

# ---- builder ----
FROM node:20-alpine AS builder

WORKDIR /build

# 先拷依赖描述（利用层缓存）
COPY package.json package-lock.json* ./

RUN npm ci --no-audit --no-fund

# 拷源代码
COPY tsconfig.json tsconfig.node.json vite.config.ts index.html ./
COPY src/ ./src/

# 构建时指定后端 API 地址（运行时可被 nginx 反代覆盖）
ENV VITE_API_BASE_URL=/api/v1

RUN npm run build

# ---- runtime ----
FROM nginx:alpine AS runtime

# 替换默认配置：前端静态 + /api 反代到 backend
RUN cat > /etc/nginx/conf.d/default.conf <<'EOF'
server {
    listen 80;
    server_name _;

    # 前端静态资源
    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # API 反向代理到 backend 容器
    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        # SSE 长连接支持
        proxy_buffering off;
        proxy_read_timeout 300s;
    }
}
EOF

COPY --from=builder /build/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget --quiet --spider http://127.0.0.1:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
