"""
统一异常处理模块

定义项目中所有自定义异常类，提供一致的错误处理机制。
"""
from typing import Any, Dict, Optional
from enum import Enum


class ErrorCode(str, Enum):
    """错误码枚举"""

    # 配置错误 (1000-1099)
    CONFIG_INVALID = "CONFIG001"
    CONFIG_MISSING = "CONFIG002"
    CONFIG_VALIDATION = "CONFIG003"

    # 数据库错误 (2000-2099)
    DATABASE_INIT = "DATABASE001"
    DATABASE_INDEX = "DATABASE002"
    DATABASE_SEARCH = "DATABASE003"
    DATABASE_INSERT = "DATABASE004"
    DATABASE_DELETE = "DATABASE005"
    DATABASE_UPDATE = "DATABASE006"

    # 向量错误 (3000-3099)
    VECTOR_EMBEDDING = "VECTOR001"
    VECTOR_DIMENSION = "VECTOR002"
    VECTOR_EMPTY = "VECTOR003"

    # 检索错误 (4000-4099)
    SEARCH_QUERY = "SEARCH001"
    SEARCH_TIMEOUT = "SEARCH002"
    SEARCH_NO_RESULTS = "SEARCH003"

    # 验证错误 (5000-5099)
    VALIDATION_FAILED = "VALIDATE001"
    VALIDATION_PARAM = "VALIDATE002"
    VALIDATION_BUSINESSTYPE = "VALIDATE003"

    # MCP 错误 (6000-6099)
    MCP_CONNECTION = "MCP001"
    MCP_TIMEOUT = "MCP002"
    MCP_PROTOCOL = "MCP003"
    MCP_UNAVAILABLE = "MCP004"


class BaseError(Exception):
    """基础异常类

    所有自定义异常的基类，提供统一的错误信息格式。
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """初始化异常

        Args:
            message: 错误消息
            code: 错误码
            details: 额外详细信息
            cause: 原始异常（用于异常链）
        """
        self.message = message
        self.code = code
        self.details = details or {}
        self.cause = cause

        # 构建完整的错误消息
        full_message = f"[{code.value}] {message}"
        if details:
            details_str = ", ".join(f"{k}={v}" for k, v in details.items())
            full_message += f" ({details_str})"

        super().__init__(full_message)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            包含错误信息的字典
        """
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code.value}, message={self.message})"


# ==================== 配置错误 ====================


class ConfigError(BaseError):
    """配置错误基类"""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.CONFIG_INVALID,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, code, details, cause)


class ConfigValidationError(ConfigError):
    """配置验证错误

    当配置参数不符合要求时抛出。
    """

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        # 合并 details 字典
        merged_details = details or {}

        # 如果指定了 parameter 或 value，添加到 details 中
        if parameter:
            merged_details["parameter"] = parameter
        if value is not None:
            merged_details["value"] = str(value)

        super().__init__(
            message=message,
            code=ErrorCode.CONFIG_VALIDATION,
            details=merged_details if merged_details else None,
            cause=cause,
        )


class InvalidBusinesstypeError(ConfigError):
    """无效的业务类型错误

    当业务类型名称不符合规范时抛出。
    """

    def __init__(
        self,
        businesstype: str,
        reason: str = "Must be 1-50 alphanumeric characters, underscores, or hyphens only",
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Invalid businesstype: '{businesstype}'. {reason}",
            code=ErrorCode.VALIDATION_BUSINESSTYPE,
            details={"businesstype": businesstype},
            cause=cause,
        )


# ==================== 数据库错误 ====================


class DatabaseError(BaseError):
    """数据库错误基类"""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.DATABASE_INIT,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, code, details, cause)


class DatabaseInitError(DatabaseError):
    """数据库初始化错误

    当数据库初始化失败时抛出。
    """

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        details = {"path": path} if path else None
        super().__init__(
            message=message,
            code=ErrorCode.DATABASE_INIT,
            details=details,
            cause=cause,
        )


class DatabaseIndexError(DatabaseError):
    """索引错误

    当索引操作失败时抛出。
    """

    def __init__(
        self,
        message: str,
        index_type: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        details = {"index_type": index_type} if index_type else None
        super().__init__(
            message=message,
            code=ErrorCode.DATABASE_INDEX,
            details=details,
            cause=cause,
        )


class DatabaseSearchError(DatabaseError):
    """搜索错误

    当数据库搜索失败时抛出。
    """

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        details = {}
        if query:
            details["query"] = query[:100]  # 限制长度
        if top_k is not None:
            details["top_k"] = top_k

        super().__init__(
            message=message,
            code=ErrorCode.DATABASE_SEARCH,
            details=details if details else None,
            cause=cause,
        )


# ==================== 向量错误 ====================


class VectorError(BaseError):
    """向量错误基类"""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.VECTOR_EMBEDDING,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, code, details, cause)


class EmbeddingError(VectorError):
    """嵌入向量生成错误

    当生成嵌入向量失败时抛出。
    """

    def __init__(
        self,
        message: str,
        text_length: Optional[int] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        details = {"text_length": text_length} if text_length is not None else None
        super().__init__(
            message=message,
            code=ErrorCode.VECTOR_EMBEDDING,
            details=details,
            cause=cause,
        )


class DimensionMismatchError(VectorError):
    """向量维度不匹配错误

    当向量维度与预期不符时抛出。
    """

    def __init__(
        self,
        expected: int,
        actual: int,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Vector dimension mismatch: expected {expected}, got {actual}",
            code=ErrorCode.VECTOR_DIMENSION,
            details={"expected": expected, "actual": actual},
            cause=cause,
        )


# ==================== 验证错误 ====================


class ValidationError(BaseError):
    """验证错误基类"""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.VALIDATION_FAILED,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, code, details, cause)


class ParameterValidationError(ValidationError):
    """参数验证错误

    当函数参数验证失败时抛出。
    """

    def __init__(
        self,
        parameter_name: str,
        value: Any,
        reason: str,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Parameter '{parameter_name}' validation failed: {reason}",
            code=ErrorCode.VALIDATION_PARAM,
            details={"parameter": parameter_name, "value": str(value)},
            cause=cause,
        )


# ==================== 检索错误 ====================


class SearchError(BaseError):
    """检索错误基类"""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.SEARCH_QUERY,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, code, details, cause)


class QueryValidationError(SearchError):
    """查询验证错误"""

    def __init__(
        self,
        query: str,
        reason: str,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Query validation failed: {reason}",
            code=ErrorCode.SEARCH_QUERY,
            details={"query": query[:100]},
            cause=cause,
        )


# ==================== MCP 错误 ====================


class MCPError(BaseError):
    """MCP 错误基类"""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.MCP_CONNECTION,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, code, details, cause)


class MCPConnectionError(MCPError):
    """MCP 连接错误"""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        details = {"url": url} if url else None
        super().__init__(
            message=message,
            code=ErrorCode.MCP_CONNECTION,
            details=details,
            cause=cause,
        )


class MCPTimeoutError(MCPError):
    """MCP 超时错误"""

    def __init__(
        self,
        message: str,
        timeout: Optional[float] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        details = {"timeout": timeout} if timeout is not None else None
        super().__init__(
            message=message,
            code=ErrorCode.MCP_TIMEOUT,
            details=details,
            cause=cause,
        )
