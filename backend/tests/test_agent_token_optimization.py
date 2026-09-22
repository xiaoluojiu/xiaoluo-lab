from app.agent.context.budget import ContextBudget
from app.agent.context.cache import DatasetContextCache
from app.agent.context.models import AgentContext
from app.agent.llm.capabilities import LLMCapabilities
from app.agent.llm.mock import MockLLM
from app.agent.llm.usage import LLMUsage


def test_context_budget_scales_sections_to_total_limit() -> None:
    context = AgentContext(user_request="u" * 1000, dataset_context={"1": {"profile": "d" * 2000}}, task_context={"task": "t" * 1000}, permission_context={"role": "analyst"}, tool_context={"tools": [{"name": "dataset.inspect", "description": "x" * 2000}]}, conversation_history=[{"role": "user", "content": "h" * 1000}])
    rendered = context.to_prompt_text(budget=ContextBudget(max_chars=1200, user_request=800, dataset=800, task=400, permissions=200, tools=800, history=400, history_messages=1))
    assert len(rendered) <= 1500
    assert "[用户请求]" in rendered
    assert "[数据集]" in rendered


def test_dataset_context_cache_is_version_scoped() -> None:
    cache = DatasetContextCache(max_items=2)
    cache.set(1, 1, {"version": 1})
    assert cache.get(1, 1) == {"version": 1}
    assert cache.get(1, 2) is None


def test_dataset_context_cache_is_bounded() -> None:
    cache = DatasetContextCache(max_items=2)
    cache.set(1, 1, {"v": 1})
    cache.set(2, 1, {"v": 2})
    cache.set(3, 1, {"v": 3})
    assert cache.get(1, 1) is None
    assert cache.get(3, 1) == {"v": 3}


def test_mock_llm_counts_calls_for_budget() -> None:
    llm = MockLLM(structured_responses=[])
    assert llm.usage_snapshot()["calls"] == 0


def test_capabilities_require_explicit_support() -> None:
    caps = LLMCapabilities(tool_calling=None, structured_output=True)
    assert caps.supports("structured_output") is True
    assert caps.supports("tool_calling") is False


def test_usage_normalizes_provider_fields() -> None:
    usage = LLMUsage.from_raw({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}, provider="test", model="m")
    assert usage.input_tokens == 10
    assert usage.output_tokens == 4
    assert usage.total_tokens == 14
    assert usage.estimated is False
