"""HTTP 请求中间件集合。

顺序由 ``main.py`` 统一编排（Starlette 里**后添加者在外层**），目标顺序为
``RequestContext -> CORS -> RateLimit -> GZip -> app``。
"""

from __future__ import annotations

import gzip
import time
import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import AppException
from app.core.logging import get_logger, set_request_id

logger = get_logger("app.middleware")

REQUEST_ID_HEADER = "X-Request-ID"

# 被限流保护的端点前缀。
# 选的标准是「会消耗钱 / 会占满 CPU」而不是「HTTP 写操作」——普通表单保存
# 被限流只会让人困惑，而一次 /agent/sessions/*/messages 是真的会产生 LLM 账单。
RATE_LIMITED_PREFIXES: tuple[str, ...] = (
    "/api/v1/agent/sessions",
    "/api/v1/reports/generate",
    "/api/v1/reports/export",
    "/api/v1/experiments",
    "/api/v1/workflow",
    "/api/v1/connectors",
    "/api/v1/learning",
)

# 不参与压缩的响应类型。SSE 是逐 token 推送，压缩会把实时性变成"缓冲到
# 一个 gzip 块才吐"，必须排除；二进制类型压缩既无收益又费 CPU。
_GZIP_EXCLUDED_TYPES = (
    "text/event-stream",
    "image/",
    "video/",
    "audio/",
    "application/pdf",
    "application/octet-stream",
    "application/zip",
)


class RequestContextMiddleware:
    """纯 ASGI 中间件：请求 ID、统一异常与请求耗时日志。

    不使用 Starlette BaseHTTPMiddleware，避免额外的 task/stream 包装，
    对文件上传、流式响应和本地开发环境更加稳定。
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        request_id = headers.get(REQUEST_ID_HEADER.lower().encode())
        rid = request_id.decode("latin-1") if request_id else uuid.uuid4().hex
        set_request_id(rid)

        request = Request(scope, receive=receive)
        request.state.request_id = rid
        start = time.perf_counter()
        status_code = 500

        async def send_with_context(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
                response_headers = list(message.get("headers", []))
                response_headers.append((REQUEST_ID_HEADER.lower().encode(), rid.encode()))
                message = {**message, "headers": response_headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_context)
        except AppException as exc:
            # 业务异常也要留痕：4xx 记 warning（含堆栈便于定位规则问题），
            # 5xx 记 error。过去这里完全不记日志，线上只有 access log 的状态码，
            # 无法回溯是哪段业务逻辑抛出的。
            if exc.http_status >= 500:
                logger.error(
                    "business error | path=%s code=%s status=%s",
                    request.url.path,
                    exc.code,
                    exc.http_status,
                    exc_info=True,
                )
            else:
                logger.warning(
                    "business error | path=%s code=%s status=%s message=%s",
                    request.url.path,
                    exc.code,
                    exc.http_status,
                    exc.message,
                )
            response = self._error_response(request, exc.http_status, exc.to_dict())
            status_code = response.status_code
            await response(scope, receive, send_with_context)
        except Exception:
            logger.exception("unhandled error | path=%s", request.url.path)
            response = self._error_response(
                request,
                500,
                {
                    "code": "INTERNAL_ERROR",
                    "message": "Internal Server Error",
                    "details": None,
                },
            )
            status_code = response.status_code
            await response(scope, receive, send_with_context)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "%s %s -> %s (%.1f ms)",
                scope.get("method", "-"),
                scope.get("path", "-"),
                status_code,
                elapsed_ms,
            )

    @staticmethod
    def _error_response(
        request: Request,
        status_code: int,
        body: dict[str, Any],
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={
                **body,
                "request_id": getattr(request.state, "request_id", None),
            },
        )


def _header_value(headers: list[tuple[bytes, bytes]], name: str) -> str:
    lookup = name.encode("latin-1").lower()
    for key, value in headers:
        if key.lower() == lookup:
            return value.decode("latin-1")
    return ""


class GZipMiddleware:
    """纯 ASGI 的 gzip 压缩中间件。

    不直接用 Starlette 的 ``GZipMiddleware``，原因有两个：

    1. 它会对 SSE（``text/event-stream``）也做流式压缩，把「逐 token 推送」
       变成「攒够一个压缩块再吐」，前端表现为卡顿一大段然后一次性刷屏；
    2. 它对没有 ``Content-Length`` 的响应会无界缓冲。

    这里的策略因此非常保守：**只对明确知道长度、且明显值得压缩的响应生效**。
    """

    def __init__(self, app: Any, *, minimum_size: int = 1024, compresslevel: int = 5) -> None:
        self.app = app
        self.minimum_size = max(int(minimum_size), 0)
        self.compresslevel = compresslevel

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        accept = _header_value(list(scope.get("headers", [])), "accept-encoding").lower()
        if "gzip" not in accept:
            await self.app(scope, receive, send)
            return

        start_message: dict[str, Any] | None = None
        chunks: list[bytes] = []
        expected = -1
        compress = False

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal start_message, chunks, expected, compress
            if message["type"] == "http.response.start":
                start_message = message
                headers = list(message.get("headers", []))
                content_type = _header_value(headers, "content-type").lower()
                length_raw = _header_value(headers, "content-length")
                already = _header_value(headers, "content-encoding")
                # 只有「已知长度 + 够大 + 不在排除名单 + 未被压缩过」才压。
                compress = bool(length_raw.isdigit()) and not already
                if compress:
                    expected = int(length_raw)
                    compress = expected >= self.minimum_size and not any(
                        content_type.startswith(bad) for bad in _GZIP_EXCLUDED_TYPES
                    )
                return
            if not compress:
                if start_message is not None and message["type"] == "http.response.body":
                    await send(start_message)
                    start_message = None
                await send(message)
                return

            if message["type"] == "http.response.body":
                chunks.append(bytes(message.get("body", b"")))
                if message.get("more_body"):
                    return
                await self._emit(start_message, chunks, expected, send)
                chunks = []

        await self.app(scope, receive, send_wrapper)
        # 上游提前断开（more_body 没走到 False）时原样放行，避免吞掉半个响应。
        if compress and start_message is not None and chunks:
            await self._emit(start_message, chunks, expected, send)

    async def _emit(
        self,
        start: dict[str, Any] | None,
        chunks: list[bytes],
        expected: int,
        send: Any,
    ) -> None:
        if start is None:
            return
        body = b"".join(chunks)
        if len(body) != expected or not body:
            await send(start)
            for index, chunk in enumerate(chunks):
                await send({
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": index < len(chunks) - 1,
                })
            return
        compressed = gzip.compress(body, compresslevel=self.compresslevel)
        headers = [
            (key, value)
            for key, value in start.get("headers", [])
            if key.lower() not in (b"content-length", b"content-encoding")
        ]
        headers.append((b"content-encoding", b"gzip"))
        headers.append((b"vary", b"Accept-Encoding"))
        headers.append((b"content-length", str(len(compressed)).encode("latin-1")))
        await send({**start, "headers": headers})
        await send({"type": "http.response.body", "body": compressed})


class RateLimitMiddleware:
    """轻量固定窗口限流（进程内存，无外部依赖）。

    刻意只保护 :data:`RATE_LIMITED_PREFIXES` 列出的昂贵端点：
    本项目是单租户本地平台，给所有 GET 加限流只会拖慢正常浏览。
    多实例部署时本限流是**每实例独立计数**，真正的边界防护应放在反向代理层。
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self._hits: dict[tuple[str, str], list[float]] = {}

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", ""))
        # 只读请求不限：浏览列表/刷新页面不该被限流波及；真正贵的是 POST 触发的
        # 训练、导入、报告生成与 LLM 调用。
        if method in ("GET", "HEAD", "OPTIONS") or not self._protected(path):
            await self.app(scope, receive, send)
            return

        from app.core.config import settings

        if not settings.RATE_LIMIT_ENABLED:
            await self.app(scope, receive, send)
            return

        window = max(int(settings.RATE_LIMIT_WINDOW_SECONDS), 1)
        limit = max(int(settings.RATE_LIMIT_MAX_REQUESTS), 1)
        client = self._client_id(scope, bool(settings.RATE_LIMIT_TRUST_X_FORWARDED_FOR))

        now = time.monotonic()
        key = (client, path)
        recent = [ts for ts in self._hits.get(key, ()) if now - ts < window]
        allowed = len(recent) < limit
        if allowed:
            recent.append(now)
        self._hits[key] = recent
        self._evict(now, window)

        if not allowed:
            retry_after = max(int(window - (now - recent[0])), 1)
            response = JSONResponse(
                status_code=429,
                content={
                    "code": "RATE_LIMITED",
                    "message": f"请求过于频繁，请在 {retry_after} 秒后重试",
                    "details": {"path": path, "limit": limit, "window_seconds": window},
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Window": str(window),
                },
            )
            logger.warning(
                "rate limited | path=%s client=%s limit=%s/%ss",
                path, client, limit, window,
            )
            await response(scope, receive, send)
            return

        remaining = limit - len(recent)

        async def send_with_limit(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-ratelimit-limit", str(limit).encode("latin-1")))
                headers.append((b"x-ratelimit-remaining", str(max(remaining, 0)).encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_limit)

    @staticmethod
    def _protected(path: str) -> bool:
        return any(path.startswith(prefix) for prefix in RATE_LIMITED_PREFIXES)

    @staticmethod
    def _client_id(scope: dict[str, Any], trust_proxy: bool) -> str:
        if trust_proxy:
            forwarded = _header_value(list(scope.get("headers", [])), "x-forwarded-for")
            if forwarded:
                return forwarded.split(",")[0].strip() or "unknown"
        client = scope.get("client")
        return str(client[0]) if client else "unknown"

    def _evict(self, now: float, window: int) -> None:
        """清理过期计数，避免长期运行后字典无限增长。"""
        if len(self._hits) <= 512:
            return
        self._hits = {
            key: [ts for ts in stamps if now - ts < window]
            for key, stamps in self._hits.items()
        }
        self._hits = {key: stamps for key, stamps in self._hits.items() if stamps}
