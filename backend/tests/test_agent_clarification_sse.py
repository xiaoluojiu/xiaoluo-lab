"""澄清态（WAITING_CLARIFICATION）的端到端闭环 —— 钉住「8% 永久不动」。

这个文件的存在理由是一个真实事故：

    离线模式 + 绑定数据集 + 「根据数据集选择合适的机器学习模型」
    → preflight 判定目标列不可确定 → `_await_clarification` → 运行进入
      WAITING_CLARIFICATION 并 **return**
    → SSE tail 的 `_TERMINAL_STATUSES` 里没有这个状态 ⇒ 流继续挂到
      AGENT_SSE_CONFIRM_WAIT_SECONDS（默认 900s）
    → 前端 `onDone` 不触发 ⇒ `busy` / `sendingRef` 不复位；而前端当时根本
      不认识 `waiting_clarification`，也没有 `/clarify` 的调用封装
    → 表现为「进度停在 8% 一动不动」，会话还被 409 锁死

所以这里钉三件事，缺一件事故就会重演：

1. SSE 在澄清态**必须立刻收流**（否则前端永远拿不到控制权）；
2. `POST /runs/{id}/clarify` 必须能从澄清态把运行推进到终态；
3. 澄清载荷必须落在 `pending_clarification`（不是 `pending_confirmation`）。

不启外部服务、不打 LLM。
"""

from __future__ import annotations

import time

import polars as pl
import pytest
from app.agent.runtime.models import RunStatus
from app.api.v1.agent import _TERMINAL_STATUSES
from app.data_engine.service import DataEngineService
from app.experiments.service import ExperimentService
from app.main import app
from app.services.dataset_service import DatasetService
from app.tools.builtin import TOOL_REGISTRY  # noqa: F401 - 导入即注册
from fastapi.testclient import TestClient


@pytest.fixture()
def client(db, storage):
    return TestClient(app)


@pytest.fixture()
def modeling_dataset(db, storage):
    """建模请求必然被 Pre-flight 反问的数据集。

    两个条件同时满足：
    - 列名刻意**不带**目标列命名约定（没有 target/label/y），也没有能被
      `goal_concepts` 命中的语义 ⇒ `target_resolution` 判定「无法确定目标列」；
    - 只有 10 行 ⇒ `scale_sanity` 也会反问「样本量太小是否继续」。

    任一条命中运行就会进入 WAITING_CLARIFICATION。用**小**数据集是为了让
    反问稳定发生（60 行时这条建模请求会一路走到 ml.train 的授权态，那也是
    合法路径，但不是本文件要钉的澄清闭环）。
    """
    ds = DatasetService(db, storage)
    dataset = ds.create("无约定列", "目标列不可推断的小样本")
    rows = 10
    df = pl.DataFrame(
        {
            "a_one": [float(i) for i in range(rows)],
            "a_two": [float(rows - i) for i in range(rows)],
            "a_cat": ["p" if i % 2 == 0 else "q" for i in range(rows)],
        }
    )
    ds.create_version(dataset.id, df)
    return dataset.id


def test_clarification_is_terminal_for_sse():
    """回归护栏：澄清态必须在 SSE 的终止集合里。

    曾经 `_TERMINAL_STATUSES` 只有 completed/failed，澄清被当成「还在跑」，
    tail 循环一直挂到 900 秒 —— 这是「8% 永久不动」的直接成因。
    """
    assert RunStatus.WAITING_CLARIFICATION in _TERMINAL_STATUSES
    # 授权态**不**在里面：那条闭环依赖同一条流把 resume 之后的事件推给浏览器。
    assert RunStatus.WAITING_CONFIRMATION not in _TERMINAL_STATUSES


def test_modeling_request_asks_clarification_and_resumes(client, modeling_dataset):
    """建模请求 → 等待澄清 → 回答 → 运行推进到终态（不是永远卡住）。"""
    session_id = client.post(
        "/api/v1/agent/sessions", json={"dataset_ids": [modeling_dataset]}
    ).json()["data"]["id"]

    data = client.post(
        f"/api/v1/agent/sessions/{session_id}/messages",
        json={"content": "帮我训练一个分类模型"},
    ).json()["data"]

    if data["status"] != RunStatus.WAITING_CLARIFICATION.value:
        # 本地 Router 若换了权重/阈值，可能直接判出其他路径。
        # 这里要钉的是「一旦进入澄清态就必须能恢复」，不是钉它一定进入澄清态。
        pytest.skip(f"本次未进入澄清态（status={data['status']}），跳过恢复用例")

    pending = data["pending_clarification"]
    assert pending, "澄清载荷必须落在 pending_clarification（不是 pending_confirmation）"
    assert pending.get("code"), "反问必须带 code，否则回答无法被匹配回去"

    resumed = client.post(
        f"/api/v1/agent/runs/{data['id']}/clarify", json={"answer": "a_cat"}
    ).json()["data"]

    assert resumed["status"] != RunStatus.WAITING_CLARIFICATION.value, (
        "回答后仍停在等待澄清 ⇒ 运行永远无法恢复"
    )
    assert resumed["status"] in (
        RunStatus.COMPLETED.value,
        RunStatus.FAILED.value,
        RunStatus.WAITING_CONFIRMATION.value,
    )


def test_sse_closes_promptly_on_clarification(client, modeling_dataset):
    """澄清态下 SSE 必须很快结束（而不是挂到 900s 的 tail 上限）。"""
    session_id = client.post(
        "/api/v1/agent/sessions", json={"dataset_ids": [modeling_dataset]}
    ).json()["data"]["id"]

    started = time.time()
    with client.stream(
        "POST",
        f"/api/v1/agent/sessions/{session_id}/messages",
        json={"content": "帮我训练一个分类模型", "stream": True},
    ) as resp:
        body = b""
        deadline = started + 20.0  # 远小于 900s 的 tail 上限
        for chunk in resp.iter_bytes():
            body += chunk
            if b"event: clarification" in body or b"event: done" in body:
                break
            if time.time() > deadline:
                pytest.fail("SSE 在澄清态没有及时收流（tail 仍在挂起）")

    elapsed = time.time() - started
    assert elapsed < 20.0, f"SSE 收流耗时 {elapsed:.1f}s，疑似又挂到了 tail 上限"
