"""Prompt 004 测试：健康检查与中间件行为。"""

from __future__ import annotations


def test_health_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["app"] == "xiaoluo-lab"
    assert body["env"] in {"dev", "test", "prod"}
    assert "version" in body


def test_request_id_generated(client):
    resp = client.get("/api/v1/health")
    assert resp.headers.get("x-request-id")


def test_request_id_passthrough(client):
    resp = client.get("/api/v1/health", headers={"X-Request-ID": "test-rid-123"})
    assert resp.headers.get("x-request-id") == "test-rid-123"


def test_unhandled_exception_returns_structured_json(client):
    """统一异常处理：未知异常 -> 500 结构化 JSON。"""
    from app.main import app
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/api/v1/_boom")
    async def boom() -> None:
        raise RuntimeError("boom")

    app.include_router(router)
    try:
        resp = client.get("/api/v1/_boom")
        assert resp.status_code == 500
        body = resp.json()
        assert body["code"] == "INTERNAL_ERROR"
        assert "request_id" in body
    finally:
        app.router.routes = [
            r for r in app.router.routes if getattr(r, "path", None) != "/api/v1/_boom"
        ]
