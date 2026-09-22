"""Data Engine 异常。"""

from __future__ import annotations

from app.core.exceptions import AppException


class DataEngineException(AppException):
    """Data Engine 基础异常。"""

    http_status = 400
    default_code = "DATA_ENGINE_ERROR"
    default_message = "Data engine error"


class UnsupportedFormat(DataEngineException):
    default_code = "UNSUPPORTED_FORMAT"
    default_message = "Unsupported file format"


class LoadError(DataEngineException):
    default_code = "LOAD_ERROR"
    default_message = "Failed to load data"


class SchemaError(DataEngineException):
    default_code = "SCHEMA_ERROR"
    default_message = "Schema error"


class DataQualityError(DataEngineException):
    default_code = "DATA_QUALITY_ERROR"
    default_message = "Data quality check failed"


class TransformError(DataEngineException):
    default_code = "TRANSFORM_ERROR"
    default_message = "Transform failed"


class MergeError(DataEngineException):
    default_code = "MERGE_ERROR"
    default_message = "Merge failed"
