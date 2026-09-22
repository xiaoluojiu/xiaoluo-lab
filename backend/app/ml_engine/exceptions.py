"""ML Engine 异常。

继承统一异常体系 AppException，便于 API 中间件统一处理；
本模块自身不依赖 FastAPI。
"""

from __future__ import annotations

from app.core.exceptions import AppException


class MLEngineException(AppException):
    """ML Engine 基础异常。"""

    http_status = 400
    default_code = "ML_ENGINE_ERROR"
    default_message = "ML engine error"
