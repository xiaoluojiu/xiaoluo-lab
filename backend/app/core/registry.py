"""标准化内核 · 统一注册表协议。

改造前平台里有 **三份各自实现的注册表**，语义相同、报错不同、能力不同：

===========================  ==============================  ====================
注册表                        位置                              冲突/缺失报错
===========================  ==============================  ====================
``ToolRegistry``             ``app/tools/registry.py``       ``AppException``
                                                             (TOOL_ALREADY_REGISTERED /
                                                             TOOL_NOT_FOUND)
``ModelRegistry``            ``app/ml_engine/registry.py``   ``MLEngineException``
``OPERATION_REGISTRY``       ``app/data_engine/service.py``  裸 ``dict`` + 旁路
                                                            ``OPERATION_METADATA``
===========================  ==============================  ====================

于是「新增一个特征工程操作」这种事没有统一套路可循 —— 每个作者都按上一个
注册表照抄一遍，报错口径与元数据结构各写各的（这正是 Data Engine 里
``OPERATION_REGISTRY`` 与 ``OPERATION_METADATA`` 分成两个结构的原因）。

本模块把「键值注册表」这件事标准化为一份泛型实现：

* 键唯一性、缺失报错、遍历、快照由基类统一提供；
* 报错类型通过两个钩子（:meth:`_conflict_error` / :meth:`_missing_error`）定制，
  各层保留自己的异常体系，不强制改调用方的 except；
* 元数据不再是旁路字典，而是注册项自己的能力（由注册项自己实现 ``describe()``）。

新增一张注册表时只做两件事：继承 :class:`Registry`，实现两个错误钩子。
"""

from __future__ import annotations

from typing import Any, Generic, Iterator, TypeVar

from app.core.exceptions import AppException

__all__ = [
    "Registry",
    "RegistryConflictError",
    "RegistryNotFoundError",
]

T = TypeVar("T")


class RegistryConflictError(AppException):
    """重复注册。"""

    http_status = 409
    default_code = "REGISTRY_CONFLICT"
    default_message = "Registry key already registered"


class RegistryNotFoundError(AppException):
    """注册项不存在。"""

    http_status = 404
    default_code = "REGISTRY_NOT_FOUND"
    default_message = "Registry key not found"


class Registry(Generic[T]):
    """键值注册表基类。

    子类约定
    --------
    * 覆写 :meth:`_conflict_error` / :meth:`_missing_error` 以复用本层既有异常；
    * 需要额外能力（如工具的语义检索）时**新增方法**，不要重写 ``register``；
    * 注册表实例保持进程级单例（与现有 ``TOOL_REGISTRY`` / ``MODEL_REGISTRY`` 一致）。
    """

    def __init__(self, *, label: str = "item") -> None:
        self._items: dict[str, T] = {}
        self._label = label

    # ---- 错误钩子（子类定制） -------------------------------------------
    def _conflict_error(self, key: str) -> AppException:
        return RegistryConflictError(
            f"{self._label} {key!r} 已注册",
            details={"key": key, "available": self.keys()},
        )

    def _missing_error(self, key: str) -> AppException:
        return RegistryNotFoundError(
            f"{self._label} {key!r} 未注册",
            details={"key": key, "available": self.keys()},
        )

    # ---- 标准操作 -------------------------------------------------------
    def register(self, key: str, item: T, *, replace: bool = False) -> T:
        """注册；``replace=True`` 时允许覆盖（仅用于测试与热更新）。"""
        if not key:
            raise RegistryConflictError(f"{self._label} 必须声明名称")
        if key in self._items and not replace:
            raise self._conflict_error(key)
        self._items[key] = item
        return item

    def unregister(self, key: str) -> T:
        if key not in self._items:
            raise self._missing_error(key)
        return self._items.pop(key)

    def get(self, key: str) -> T:
        item = self._items.get(key)
        if item is None:
            raise self._missing_error(key)
        return item

    def try_get(self, key: str) -> T | None:
        """取不到返回 None（用于可选能力探测，避免到处 try/except）。"""
        return self._items.get(key)

    def keys(self) -> list[str]:
        """已注册键（排序，保证输出稳定、可做文档与测试基线）。"""
        return sorted(self._items)

    def values(self) -> list[T]:
        return [self._items[k] for k in self.keys()]

    def items(self) -> list[tuple[str, T]]:
        return [(k, self._items[k]) for k in self.keys()]

    def clear(self) -> None:
        """清空（仅供测试隔离使用）。"""
        self._items.clear()

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def describe_all(self) -> list[dict[str, Any]]:
        """统一自描述出口：注册项实现 ``describe()`` 时用它，否则退化成键名。

        这一条是为了消除 Data Engine 那种「注册表 + 旁路元数据字典」的双份结构：
        元数据属于注册项自己，注册表只负责汇总。
        """
        out: list[dict[str, Any]] = []
        for key in self.keys():
            item = self._items[key]
            describe = getattr(item, "describe", None)
            if callable(describe):
                try:
                    payload = describe()
                except Exception:  # noqa: BLE001 - 单个注册项自描述失败不能拖垮清单
                    payload = {"name": key}
                if isinstance(payload, dict):
                    payload.setdefault("name", key)
                    out.append(payload)
                    continue
            out.append({"name": key})
        return out
