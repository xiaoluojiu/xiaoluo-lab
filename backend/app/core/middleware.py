"""HTTP 请求上下文 ASGI 中间件。"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import AppException
from app.core.logging import get_logger, set_request_id

logger = get_logger("app.middleware")

REQUEST_ID_HEADER = "X-Request-ID"


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
