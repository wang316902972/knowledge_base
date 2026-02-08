"""
统一异常处理模块单元测试
"""
import pytest
from exceptions import (
    BaseError,
    ConfigError,
    ConfigValidationError,
    InvalidBusinesstypeError,
    DatabaseError,
    DatabaseInitError,
    DatabaseSearchError,
    VectorError,
    EmbeddingError,
    DimensionMismatchError,
    ValidationError,
    ParameterValidationError,
    SearchError,
    MCPError,
    MCPConnectionError,
    MCPTimeoutError,
    ErrorCode,
)


class TestBaseError:
    """测试基础异常类"""

    def test_base_error_creation(self):
        """测试创建基础异常"""
        error = BaseError(
            message="Test error",
            code=ErrorCode.CONFIG_INVALID,
            details={"key": "value"}
        )
        
        assert error.message == "Test error"
        assert error.code == ErrorCode.CONFIG_INVALID
        assert error.details == {"key": "value"}
        assert ErrorCode.CONFIG_INVALID in str(error)

    def test_base_error_to_dict(self):
        """测试转换为字典"""
        error = BaseError(
            message="Test error",
            code=ErrorCode.CONFIG_INVALID,
            details={"key": "value"}
        )
        
        result = error.to_dict()
        
        assert result["code"] == ErrorCode.CONFIG_INVALID.value
        assert result["message"] == "Test error"
        assert result["details"] == {"key": "value"}

    def test_base_error_with_cause(self):
        """测试带原因的异常"""
        original_error = ValueError("Original error")
        error = BaseError(
            message="Wrapper error",
            code=ErrorCode.CONFIG_INVALID,
            cause=original_error
        )
        
        assert error.cause == original_error


class TestConfigErrors:
    """测试配置异常"""

    def test_invalid_businesstype_error(self):
        """测试无效业务类型异常"""
        error = InvalidBusinesstypeError(
            businesstype="invalid@name",
            reason="Contains invalid character"
        )
        
        assert error.code == ErrorCode.VALIDATION_BUSINESSTYPE
        assert "invalid@name" in error.message
        assert error.details["businesstype"] == "invalid@name"

    def test_config_validation_error(self):
        """测试配置验证异常"""
        error = ConfigValidationError(
            message="Validation failed",
            parameter="TEST_PARAM",
            value="invalid"
        )
        
        assert error.code == ErrorCode.CONFIG_VALIDATION
        assert error.details["parameter"] == "TEST_PARAM"


class TestDatabaseErrors:
    """测试数据库异常"""

    def test_database_init_error(self):
        """测试数据库初始化异常"""
        error = DatabaseInitError(
            message="Failed to initialize",
            path="/path/to/db"
        )
        
        assert error.code == ErrorCode.DATABASE_INIT
        assert error.details["path"] == "/path/to/db"

    def test_database_search_error(self):
        """测试数据库搜索异常"""
        error = DatabaseSearchError(
            message="Search failed",
            query="test query",
            top_k=5
        )
        
        assert error.code == ErrorCode.DATABASE_SEARCH
        assert error.details["query"] == "test query"
        assert error.details["top_k"] == 5


class TestVectorErrors:
    """测试向量异常"""

    def test_dimension_mismatch_error(self):
        """测试维度不匹配异常"""
        error = DimensionMismatchError(
            expected=384,
            actual=512
        )
        
        assert error.code == ErrorCode.VECTOR_DIMENSION
        assert "384" in error.message
        assert "512" in error.message
        assert error.details["expected"] == 384
        assert error.details["actual"] == 512

    def test_embedding_error(self):
        """测试嵌入向量生成异常"""
        error = EmbeddingError(
            message="Failed to generate embedding",
            text_length=10000
        )
        
        assert error.code == ErrorCode.VECTOR_EMBEDDING
        assert error.details["text_length"] == 10000


class TestMCErrors:
    """测试 MCP 异常"""

    def test_mcp_connection_error(self):
        """测试 MCP 连接异常"""
        error = MCPConnectionError(
            message="Connection refused",
            url="http://localhost:8003"
        )
        
        assert error.code == ErrorCode.MCP_CONNECTION
        assert error.details["url"] == "http://localhost:8003"

    def test_mcp_timeout_error(self):
        """测试 MCP 超时异常"""
        error = MCPTimeoutError(
            message="Request timeout",
            timeout=30.0
        )
        
        assert error.code == ErrorCode.MCP_TIMEOUT
        assert error.details["timeout"] == 30.0


class TestErrorCode:
    """测试错误码枚举"""

    def test_error_code_values(self):
        """测试错误码值"""
        assert ErrorCode.CONFIG_INVALID == "CONFIG001"
        assert ErrorCode.DATABASE_INIT == "DATABASE001"
        assert ErrorCode.VECTOR_EMBEDDING == "VECTOR001"
        assert ErrorCode.SEARCH_QUERY == "SEARCH001"
        assert ErrorCode.MCP_CONNECTION == "MCP001"

    def test_error_code_ranges(self):
        """测试错误码范围分配"""
        # 配置错误: 1000-1099
        assert ErrorCode.CONFIG_INVALID.value.startswith("CONFIG")
        
        # 数据库错误: 2000-2099
        assert ErrorCode.DATABASE_INIT.value.startswith("DATABASE")
        
        # 向量错误: 3000-3099
        assert ErrorCode.VECTOR_EMBEDDING.value.startswith("VECTOR")
        
        # 检索错误: 4000-4099
        assert ErrorCode.SEARCH_QUERY.value.startswith("SEARCH")
        
        # MCP 错误: 6000-6099
        assert ErrorCode.MCP_CONNECTION.value.startswith("MCP")


class TestErrorChaining:
    """测试异常链"""

    def test_error_with_cause_preserved(self):
        """测试异常原因被保留"""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise DatabaseError(
                    message="Wrapped error",
                    code=ErrorCode.DATABASE_INIT,
                    cause=e
                )
        except DatabaseError as e:
            assert e.cause is not None
            assert isinstance(e.cause, ValueError)
            assert str(e.cause) == "Original error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
