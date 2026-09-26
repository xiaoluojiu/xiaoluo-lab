"""上线前收尾加固的回归测试。

这一份刻意**不触碰 LLM**：既有的风险映射覆盖检查写在
``test_permission_llm.py`` 里，而那个文件会真实调用大模型，通常被排除在回归之外
——等于这条保障平时没被跑过。这里把它搬到一个纯逻辑的文件里，保证
「新增工具忘记登记风险等级」这件事**每次跑测试都会红**。

其余几项对应本次收尾改动的验证：
- 报告列表只返回元数据 + 目录签名缓存（性能）；
- GZip / 限流中间件（响应体积与滥用防护）；
- SQLite PRAGMA（并发读写）；
- 工具召回关键词外置后的中英文覆盖。
"""

from __future__ import annotations

import asyncio
import gzip
import json
import time

import pytest
import sqlalchemy
from app.agent.context.models import AgentContext
from app.agent.context.tokens import estimate_tokens
from app.agent.permission.rules import DEFAULT_TOOL_RISKS, RiskLevel
from app.api.v1.reports import list_saved_reports
from app.core.config import TOOL_CATEGORY_HINTS, Settings, get_settings
from app.core.database import check_runtime_configuration
from app.core.middleware import RATE_LIMITED_PREFIXES, GZipMiddleware, RateLimitMiddleware
from app.local_router.trace import enabled as local_router_enabled
from app.reports.saved import invalidate_cache, list_metadata, meta_key, save_report
from app.tools.builtin import TOOL_REGISTRY
from starlette.middleware.cors import CORSMiddleware


def run(coro):
    """在同步用例里跑协程。

    仓库里目前没有 asyncio_mode 配置（pytest-asyncio 默认 strict），
    直接写 ``async def`` 用例需要额外加 marker；这里用显式 ``asyncio.run``
    避免依赖插件的默认行为差异。
    """
    return asyncio.run(coro)


# ----------------------------------------------------------------------
# 工具风险映射覆盖
# ----------------------------------------------------------------------
def test_every_registered_tool_has_risk_level():
    """每个已注册工具都必须在 DEFAULT_TOOL_RISKS 里登记。

    失败的产物要「一眼看清缺了谁」——逐个 assert 只会暴露第一个缺失工具，
    修完再跑又报下一个。因此先收集全部缺失项一次性列出来。
    """
    registered = sorted(TOOL_REGISTRY.names())
    missing = sorted(name for name in registered if name not in DEFAULT_TOOL_RISKS)

    assert not missing, (
        "以下工具已注册但没有风险等级，PermissionManager 无法确定是否需要人工确认：\n"
        + "\n".join(f"  - {name}" for name in missing)
        + "\n请在 app/agent/permission/rules.py 的 DEFAULT_TOOL_RISKS 中登记。"
    )
    # 反向检查：表里若有幻觉条目（工具已删除/改名），同样会误导后续维护。
    stale = sorted(name for name in DEFAULT_TOOL_RISKS if name not in set(registered))
    assert not stale, f"DEFAULT_TOOL_RISKS 中存在已不存在的工具：{stale}"
    # 工具数量本身也是防线：注册失败（静默跳过）会让这个数字变小。
    assert len(registered) >= 25, f"注册的工具数量异常偏少：{len(registered)}"


def test_high_risk_tools_are_the_destructive_ones():
    """写操作/花费型工具必须落在需要确认的档位上（防止有人误改成 LOW）。"""
    for name in ("data.merge", "data.clean", "ml.train", "connector.import", "workflow.run"):
        assert DEFAULT_TOOL_RISKS[name] in {RiskLevel.HIGH, RiskLevel.CRITICAL}, name


def test_simulated_new_tool_without_risk_is_detected():
    """模拟「新增工具忘记登记」：保证上面的覆盖率断言真的会红，而不是恒真。"""
    offender = "brandnew.tool"
    assert offender not in set(TOOL_REGISTRY.names())
    candidates = sorted(set(TOOL_REGISTRY.names()) | {offender})
    missing = [name for name in candidates if name not in DEFAULT_TOOL_RISKS]
    assert missing == [offender]


# ----------------------------------------------------------------------
# 报告列表：只返回元数据 + 目录签名缓存
# ----------------------------------------------------------------------
def _make_report(index: int, *, heavy: bool = True) -> dict:
    """构造一份报告：正文里塞一个大 SVG，模拟真实报告的体积。"""
    return {
        "title": f"测试报告 {index}",
        "dataset": {"dataset_id": 1, "name": f"dataset-{index}", "rows": 1000},
        "metadata": {"generated_at": "2026-09-23T10:00:00"},
        "sections": [{"heading": "概览", "content": "x" * 200}],
        # 正文里的内联 SVG 才是历史上拖慢列表的元凶
        "charts": [
            {"type": "scatter", "svg": "<svg>" + ("M0 0h1" * (2000 if heavy else 1)) + "</svg>"}
        ],
        "conclusions": [],
    }


def test_saved_report_list_returns_metadata_only(storage):
    key = "reports/abc.json"
    save_report(storage, key, _make_report(1))
    items = list_metadata(storage)

    assert len(items) == 1
    item = items[0]
    assert set(item) == {"key", "title", "dataset", "metadata", "size", "modified_at"}
    assert item["title"] == "测试报告 1"
    # 关键：**不返回** sections / charts，列表体积与报告正文无关。
    for heavy_field in ("sections", "charts", "conclusions"):
        assert heavy_field not in item
    assert storage.exists(meta_key(key))


def test_saved_report_list_is_fast_and_cached(storage):
    """100 份报告（正文各约 12 KB SVG）下的列表性能。

    实测（本机 SSD，隔离运行）：
    * **冷启动**（文件元数据都没在 OS 缓存里）：约 111~191 ms；
    * **后续请求**：命中目录签名缓存，约 **0.9 ms**。

    冷启动那一档受机器负载影响明显（整套测试并发跑时曾测到 256ms），
    所以这里留到 300ms 作为回归红线；真正有判别力的是第二条：命中缓存必须
    快到几乎不可测，否则说明缓存没起作用。
    """
    for i in range(100):
        save_report(storage, f"reports/perf-{i:03d}.json", _make_report(i))

    invalidate_cache()
    first_started = time.perf_counter()
    first = list_metadata(storage)
    first_elapsed = time.perf_counter() - first_started

    second_started = time.perf_counter()
    second = list_metadata(storage)
    second_elapsed = time.perf_counter() - second_started

    assert len(first) == 100
    assert [x["key"] for x in first] == [x["key"] for x in second]

    assert first_elapsed < 0.3, f"首次列表耗时 {first_elapsed * 1000:.1f}ms，疑似退回慢路径"
    assert second_elapsed < 0.02, f"缓存命中耗时 {second_elapsed * 1000:.1f}ms，缓存未生效"
    assert second_elapsed * 10 < first_elapsed, "缓存命中应比冷启动快一个数量级"


def test_saved_report_list_never_reads_report_body(storage, monkeypatch):
    """列表**读的是元数据副本，不是正文**——用调用探针证明，而不是比拼毫秒数。

    之所以不用「耗时对比」：正文体积小的时候两者差别会被磁盘抖动吃掉，
    断言会变成随机红绿的负担。这里直接记录 read 了哪些 key，
    语义唯一且稳定。
    """
    read_keys: list[str] = []
    original_read = storage.read

    def spy(key: str) -> bytes:
        read_keys.append(key)
        return original_read(key)

    monkeypatch.setattr(storage, "read", spy)

    for i in range(5):
        save_report(storage, f"reports/probe-{i}.json", _make_report(i))
    read_keys.clear()

    invalidate_cache()
    items = list_metadata(storage)

    assert len(items) == 5
    assert read_keys, "应有读取动作（否则这条探针什么都没验到）"
    assert all(key.endswith(".meta.json") for key in read_keys), (
        f"报告列表读取了非元数据文件：{[k for k in read_keys if not k.endswith('.meta.json')]}"
    )
    assert not any(k in read_keys for k in ("reports/probe-0.json", "reports/probe-1.json"))


def test_saved_report_cache_reflects_writes_and_deletes(storage):
    invalidate_cache()
    assert list_metadata(storage) == []

    save_report(storage, "reports/one.json", _make_report(1))
    assert [x["title"] for x in list_metadata(storage)] == ["测试报告 1"]

    save_report(storage, "reports/two.json", _make_report(2))
    assert {x["title"] for x in list_metadata(storage)} == {"测试报告 1", "测试报告 2"}

    storage.delete("reports/one.json")
    storage.delete(meta_key("reports/one.json"))
    assert [x["key"] for x in list_metadata(storage)] == ["reports/two.json"]


def test_saved_report_list_falls_back_to_body_without_meta(storage):
    """历史报告没有元数据副本时仍能列出（不得报错，也不得丢项）。"""
    report = _make_report(7)
    storage.save("reports/legacy.json", json.dumps(report, ensure_ascii=False).encode())
    invalidate_cache()

    items = list_metadata(storage)
    assert [x["key"] for x in items] == ["reports/legacy.json"]
    assert items[0]["title"] == "测试报告 7"


def test_list_endpoint_uses_shared_metadata_layer(storage):
    """HTTP 入口必须与 Agent 工具走同一份逻辑（app.reports.saved）。"""
    save_report(storage, "reports/http.json", _make_report(1))
    invalidate_cache()
    body = list_saved_reports(storage).data
    assert [x["key"] for x in body] == ["reports/http.json"]


# ----------------------------------------------------------------------
# 中间件：GZip / 限流 / CORS
# ----------------------------------------------------------------------
def _capture():
    """返回一个既可当 send 回调、又能回收响应报文的容器。"""
    captured: dict = {}

    async def send(message):
        if message["type"] == "http.response.start":
            captured["start"] = message
        elif message["type"] == "http.response.body":
            captured["body"] = message

    captured["send"] = send
    return captured


@pytest.mark.parametrize(
    ("size", "content_type", "expect_gzip"),
    [
        (4096, "application/json", True),
        (64, "application/json", False),          # 小于阈值
        (4096, "text/event-stream", False),       # SSE 绝不能被压缩
        (4096, "application/pdf", False),         # 二进制无收益
    ],
)
def test_gzip_middleware_selects_what_to_compress(size, content_type, expect_gzip):
    body = b"x" * size

    async def drive():
        async def inner(scope, receive, send):
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", content_type.encode()),
                    (b"content-length", str(len(body)).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})

        captured = _capture()
        await GZipMiddleware(inner, minimum_size=1024)(
            {"type": "http", "headers": [(b"accept-encoding", b"gzip")]}, None, captured["send"]
        )
        return captured

    captured = run(drive())
    headers = dict(captured["start"]["headers"])
    compressed = headers.get(b"content-encoding") == b"gzip"
    assert compressed is expect_gzip
    if expect_gzip:
        assert gzip.decompress(captured["body"]["body"]) == body
        # 长度必须重写，否则浏览器按旧 content-length 解包会得到半截 JSON
        assert int(headers[b"content-length"]) == len(captured["body"]["body"])
    else:
        assert captured["body"]["body"] == body


def test_gzip_skipped_without_accept_encoding():
    async def drive():
        body = b"x" * 4096

        async def inner(scope, receive, send):
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})

        captured = _capture()
        await GZipMiddleware(inner)({"type": "http", "headers": []}, None, captured["send"])
        return captured

    captured = run(drive())
    assert b"content-encoding" not in dict(captured["start"]["headers"])


def test_rate_limit_middleware_blocks_after_limit(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 60)
    monkeypatch.setattr(settings, "RATE_LIMIT_MAX_REQUESTS", 2)

    calls = {"n": 0}

    async def drive():
        async def inner(scope, receive, send):
            calls["n"] += 1
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", b"2")],
            })
            await send({"type": "http.response.body", "body": b"ok"})

        app = RateLimitMiddleware(inner)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/reports/generate",
            "headers": [],
            "client": ("1.2.3.4", 5000),
        }
        statuses = []
        for _ in range(3):
            captured = _capture()
            await app(scope, None, captured["send"])
            statuses.append(captured["start"]["status"])
        return statuses

    assert run(drive()) == [200, 200, 429]
    assert calls["n"] == 2, "被限流的请求不应到达业务处理器"


def test_rate_limit_ignores_readonly_requests(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_MAX_REQUESTS", 1)

    async def drive():
        async def inner(scope, receive, send):
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", b"2")],
            })
            await send({"type": "http.response.body", "body": b"ok"})

        app = RateLimitMiddleware(inner)
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/experiments",
            "headers": [],
            "client": ("1.2.3.4", 5000),
        }
        captured = _capture()
        for _ in range(5):
            await app(scope, None, captured["send"])
        return captured

    captured = run(drive())
    assert captured["start"]["status"] == 200


def test_rate_limit_prefixes_only_cover_expensive_paths():
    """限流清单只针对「花钱/占 CPU」的写接口，不该扩到普通浏览。"""
    assert any(p.startswith("/api/v1/agent/sessions") for p in RATE_LIMITED_PREFIXES)
    for path in ("/api/v1/health", "/api/v1/datasets", "/api/v1/notifications"):
        assert not any(path.startswith(p) for p in RATE_LIMITED_PREFIXES)


def test_cors_is_opt_in_and_config_driven():
    """默认不放开跨域：显式配置才有白名单，不会出现偷偷的通配符。

    配置解析与「是否真的挂了 CORS 中间件」分开断言：
    中间件是在 ``import app.main`` 时就按当时的配置装好的，运行期改 settings
    不会再重建中间件栈（这也是刻意的——中间件不该热插拔）。
    """
    default_settings = Settings()
    assert Settings(CORS_ALLOW_ORIGINS="").cors_allowed_origins == []
    assert Settings(
        CORS_ALLOW_ORIGINS="https://a.example, https://b.example ,https://a.example"
    ).cors_allowed_origins == ["https://a.example", "https://b.example"]

    from app.main import app

    has_cors = any(m.cls is CORSMiddleware for m in app.user_middleware)
    # 与应用实际使用的那份配置保持一致即可（.env 里没配 ⇒ 必须没有 CORS）
    assert has_cors == bool(default_settings.cors_allowed_origins)


# ----------------------------------------------------------------------
# SQLite
# ----------------------------------------------------------------------
def test_sqlite_pragmas_are_applied(tmp_path):
    """WAL + synchronous=NORMAL 是「写不阻塞读」的前提，且不至于崩库。

    用临时库验证，避免在测试里改动开发库（sqlite:///./data/xiaoluo.db）。
    """
    engine = sqlalchemy.create_engine(f"sqlite:///{(tmp_path / 'pragmas.db').as_posix()}")
    try:
        with engine.connect() as conn:
            mode = conn.execute(sqlalchemy.text("PRAGMA journal_mode")).scalar()
            sync = conn.execute(sqlalchemy.text("PRAGMA synchronous")).scalar()
            timeout = conn.execute(sqlalchemy.text("PRAGMA busy_timeout")).scalar()
    finally:
        engine.dispose()

    assert str(mode).lower() == "wal"
    assert sync in (1, 2), "synchronous 不得为 OFF（0）——那会在断电时静默损坏库文件"
    assert int(timeout) >= 1000


def test_startup_check_flags_missing_llm_key_in_prod(monkeypatch):
    """prod 且无 API Key 必须给出「可操作」的告警，而不是悄悄降级。"""
    settings = get_settings()
    monkeypatch.setattr(settings, "APP_ENV", "prod")
    monkeypatch.setattr(settings, "LLM_API_KEY", "")
    monkeypatch.setattr(settings, "DEBUG", False)

    warnings = check_runtime_configuration()
    assert any("LLM_API_KEY" in w for w in warnings)
    assert any("SQLite" in w for w in warnings)


# ----------------------------------------------------------------------
# 本地 Router：开关语义（不绑死 .env 的当前档位）
# ----------------------------------------------------------------------
def test_local_router_mode_gate_is_contract_driven(monkeypatch):
    """``enabled()`` 是唯一开关。断言的是**语义**，不是本机 .env 里配的那一档。"""
    settings = get_settings()
    original = settings.LOCAL_ROUTER_MODE

    monkeypatch.setattr(settings, "LOCAL_ROUTER_MODE", "off")
    assert local_router_enabled() is False
    for mode in ("shadow", "guard"):
        monkeypatch.setattr(settings, "LOCAL_ROUTER_MODE", mode)
        assert local_router_enabled() is True

    monkeypatch.setattr(settings, "LOCAL_ROUTER_MODE", original)


def test_shadow_tracing_never_changes_run_behaviour(monkeypatch):
    """shadow / guard 档的埋点失败不得影响运行——「只记录不改变行为」的核心保证。

    当前 guard 档与 shadow 行为一致：生产代码里没有对 guard 的分支接管逻辑
    （``LOCAL_ROUTER_CONFIDENCE_THRESHOLD`` 只在 local_router 内部用于升级判定），
    所以把模式切成 guard 也不会改变路由结果。
    """
    from types import SimpleNamespace

    from app.agent.runtime.runtime import AgentRuntime
    from app.local_router import trace as trace_module

    settings = get_settings()
    monkeypatch.setattr(settings, "LOCAL_ROUTER_MODE", "shadow")
    monkeypatch.setattr(
        trace_module,
        "shadow_route",
        lambda **_: (_ for _ in ()).throw(RuntimeError("router boom")),
    )

    runtime = AgentRuntime.__new__(AgentRuntime)
    run = SimpleNamespace(id="r-test", user_request="分析一下")
    session = SimpleNamespace(id="s-test", dataset_ids=[])

    # 埋点抛异常时必须被吞掉：用户请求不该因为「记录轨迹失败」而失败
    runtime._trace_route(run, session)


def test_intent_enum_is_shared_with_local_router_contract():
    """local_router 不是死代码：Intent 枚举被 Agent 运行时当作唯一真源。"""
    from app.agent.intent import Intent as AgentIntent
    from app.local_router.contract import Intent as RouterIntent

    assert AgentIntent is RouterIntent
    assert "chat" in {i.value for i in RouterIntent}


# ----------------------------------------------------------------------
# 工具召回关键词外置
# ----------------------------------------------------------------------
def test_tool_category_hints_are_externalized_and_bilingual():
    """关键词表已从检索函数体搬到配置，且中英文都有覆盖。"""
    joined = " ".join(TOOL_CATEGORY_HINTS["data"]).lower()
    assert "filter rows" in joined
    assert "筛选" in joined
    for category, hints in TOOL_CATEGORY_HINTS.items():
        assert hints, f"类目 {category} 的关键词表为空"


def test_english_query_recalls_matching_tools():
    """用英文提问也要能召回对应工具。"""
    for query, expected in (
        ("filter rows", "data.filter"),
        ("train a model", "ml.train"),
        ("plot correlation", "eda.correlation"),
        ("import from mysql", "connector.import"),
        ("export report to pdf", "report.generate"),
    ):
        hits = TOOL_REGISTRY.retrieve_with_scores(query, top_k=8, min_score=0.15)
        names = [item["tool"]["name"] for item in hits]
        assert expected in names, f"查询 {query!r} 未召回 {expected}，实际：{names}"


# ----------------------------------------------------------------------
# 上下文预算：字符 + token 双重约束
# ----------------------------------------------------------------------
def _context_with(text: str, max_tokens: int) -> str:
    from app.agent.context.budget import ContextBudget
    from app.agent.context.models import AgentContext

    budget = ContextBudget(
        max_chars=4000,
        max_tokens=max_tokens,
        user_request=800,
        dataset=800,
        task=400,
        permissions=200,
        tools=1000,
        history=400,
    )
    return AgentContext(
        user_request=text,
        dataset_context={"summary": text},
        tool_context={"tools": [{"name": "tool.x", "description": text}]},
        budget=budget,
    ).to_prompt_text()


def test_context_budget_is_token_aware_for_chinese():
    """中文长上下文必须被 token 上限收住（纯字符预算会放行到 ~2 倍）。"""
    long_zh = "对该数据集做完整的数据质量检查与探索性分析并生成报告" * 200
    text = _context_with(long_zh, max_tokens=1500)

    tokens = estimate_tokens(text)
    assert tokens <= 1500, f"中文上下文估算 {tokens} token，超出预算 1500"
    assert "【已达 token 上限】" in text, "截断必须显式标记，否则调用方以为信息完整"
    # 关键信息不丢：每个分区的**开头**仍应保留
    assert "[用户请求]" in text and "[数据集]" in text and "[可用工具]" in text
    assert long_zh[:6] in text


def test_context_budget_allows_more_chars_for_latin_text():
    """同一预算下，拉丁文本应能比中文多拿字符（4 字符 ≈ 1 token）。"""
    long_zh = "数据质量检查与探索性分析" * 400
    long_en = "run full data quality checks and exploratory analysis " * 400

    zh_text = _context_with(long_zh, max_tokens=1500)
    en_text = _context_with(long_en, max_tokens=1500)

    assert len(en_text) > len(zh_text), (
        f"英文上下文 {len(en_text)} 字符未多于中文 {len(zh_text)}——token 估算没有生效"
    )
    # 两者都不得超预算
    assert estimate_tokens(en_text) <= 1500
    assert estimate_tokens(zh_text) <= 1500


def test_context_token_budget_disabled_falls_back_to_chars():
    """max_tokens=0 时退回纯字符行为（保留旧部署的兼容路径）。"""
    long_zh = "数据质量检查" * 500  # 3000 字
    text = _context_with(long_zh, max_tokens=0)

    assert len(text) <= 4000 + 100, "关闭 token 约束后应按字符上限截断"
    assert "【已达 token 上限】" not in text


def test_context_dataset_ids_is_a_public_method():
    """回归：AgentContext.dataset_ids 必须是类方法，不能掉进别的函数体内。

    曾因新增 _distribute_tokens 时把它挤到函数体里，导致它退化为嵌套函数，
    类上彻底消失 —— planner/planner.py:127 与 runtime.py:420 一调用就抛
    AttributeError('AgentContext' object has no attribute 'dataset_ids')。
    这类错位不会被 ruff/tsc 发现（语法完全合法），只能靠行为断言。
    """
    import inspect

    ctx = AgentContext(
        dataset_context={"7": {"name": "sales"}, "9": {"name": "cust"}, "bad": {}},
    )
    # 1) 必须是类上的可调用属性，而不是实例的偶然属性
    assert callable(getattr(AgentContext, "dataset_ids", None)), (
        "AgentContext.dataset_ids 丢失——多半是被缩进进了其他函数体"
    )
    assert inspect.isfunction(AgentContext.dataset_ids), "dataset_ids 应是普通方法"
    # 2) 行为：只回可解析为 int 的 key，非法 key 跳过而非抛异常
    assert ctx.dataset_ids() == [7, 9]
    # 3) 空上下文不得炸
    assert AgentContext().dataset_ids() == []


def test_context_public_methods_survive_module_reload():
    """同源回归：模块级函数不得吞掉类方法（防止再次插错缩进位置）。"""
    import importlib

    from app.agent.context import models as models_module

    reloaded = importlib.reload(models_module)
    assert hasattr(reloaded.AgentContext, "dataset_ids")
    assert hasattr(reloaded.AgentContext, "to_prompt_text")
    assert callable(reloaded._distribute_tokens)


def test_token_estimator_is_conservative():
    """估算口径必须只许高估：宁可少喂，不可超窗被拒。"""
    assert estimate_tokens("") == 0
    # 中文：1 字 1 token
    assert estimate_tokens("数据质量") == 4
    # 英文：向上取整，保证不充分（7 字符 -> 2 token）
    assert estimate_tokens("abcdefg") == 2
    # 中英混排：2 个汉字 + 4 个西文字符（含空格）⇒ 2 + ceil(4/4) = 3
    assert estimate_tokens("数据 abc") == 3


def test_clip_by_tokens_keeps_the_longest_affordable_prefix():
    from app.agent.context.tokens import clip_by_tokens

    text = "中文内容" * 1000  # 4000 字 ≈ 4000 token
    clipped = clip_by_tokens(text, 100)
    assert estimate_tokens(clipped) <= 100
    # 再加一个字符就会超 —— 说明没有保守过头
    assert estimate_tokens(clipped + text[len(clipped)]) > 100



# ----------------------------------------------------------------------
# 通知：版本号 + SSE 推送
# ----------------------------------------------------------------------
class _FakeRequest:
    """最小的 Request 替身：只提供 SSE 生成器用到的 is_disconnected。"""

    def __init__(self, stay_connected_rounds: int = 6):
        self.calls = 0
        self.limit = stay_connected_rounds

    async def is_disconnected(self) -> bool:
        self.calls += 1
        return self.calls > self.limit


def test_notification_stream_pushes_snapshot_then_changes(monkeypatch):
    """SSE 首帧给快照，之后**只在版本号变化时**推 changed 事件。"""
    from app.api.v1 import notifications as api
    from app.notifications.store import NotificationStore

    store = NotificationStore()
    monkeypatch.setattr(api, "NOTIFICATION_STORE", store)
    settings = get_settings()
    monkeypatch.setattr(settings, "NOTIFICATION_SSE_INTERVAL_SECONDS", 0.02)

    async def drive():
        request = _FakeRequest(stay_connected_rounds=6)
        response = await api.stream_notifications(request)
        frames = []

        async def emit_soon():
            await asyncio.sleep(0.05)
            store.emit("system", "训练完成", "模型已就绪")

        task = asyncio.create_task(emit_soon())
        async for chunk in response.body_iterator:
            frames.append(chunk)
            if any("event: changed" in f for f in frames):
                break
        await task
        return frames

    frames = run(drive())
    joined = "".join(frames)

    assert "event: snapshot" in joined
    assert "event: changed" in joined
    assert '"unread": 1' in joined
    # 心跳只在长时间无变化时出现，这里不应被触发
    assert ": keep-alive" not in joined


def test_notification_version_bumps_on_mutations():
    """版本号是前端判断「要不要重新拉列表」的唯一依据，任何写操作都必须自增。"""
    from app.notifications.store import NotificationStore

    store = NotificationStore()
    base = store.version()
    assert base == 0

    item = store.emit("system", "标题", "正文")
    assert store.version() > base
    after_emit = store.version()

    store.mark_read(item.id)
    assert store.version() > after_emit

    snapshot = store.snapshot()
    assert snapshot["unread"] == 0
    assert snapshot["version"] == store.version()
    assert snapshot["items"][0]["id"] == item.id


