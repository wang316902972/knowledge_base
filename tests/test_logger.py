"""
统一日志管理模块单元测试
"""
import pytest
import logging
import tempfile
import os
from pathlib import Path
from logger import (
    setup_logger,
    get_logger,
    LoggerContext,
    LogLevel,
    LogFormatter,
    configure_global_logging,
)


class TestSetupLogger:
    """测试日志设置"""

    def test_setup_logger_basic(self):
        """测试基础日志设置"""
        logger = setup_logger("test_basic")
        
        assert logger.name == "test_basic"
        assert logger.handlers  # 应该有处理器
        assert logger.level >= 0

    def test_setup_logger_with_file(self):
        """测试带文件输出的日志设置"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            
            logger = setup_logger(
                "test_file",
                log_file=log_file,
                use_console=False
            )
            
            # 写入日志
            logger.info("Test message")
            
            # 验证文件创建
            assert os.path.exists(log_file)
            
            # 清理
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    def test_setup_logger_level(self):
        """测试日志级别设置"""
        logger_debug = setup_logger("test_debug", level=logging.DEBUG)
        logger_info = setup_logger("test_info", level=logging.INFO)
        
        assert logger_debug.level == logging.DEBUG
        assert logger_info.level == logging.INFO

    def test_setup_logger_idempotent(self):
        """测试重复设置不会重复添加处理器"""
        logger1 = setup_logger("test_idempotent")
        handlers_count_1 = len(logger1.handlers)
        
        logger2 = setup_logger("test_idempotent")
        handlers_count_2 = len(logger2.handlers)
        
        # 如果已经有处理器，应该直接返回
        assert handlers_count_2 == handlers_count_1


class TestGetLogger:
    """测试获取日志记录器"""

    def test_get_logger_existing(self):
        """测试获取已存在的日志记录器"""
        logger1 = setup_logger("test_existing")
        logger2 = get_logger("test_existing")
        
        # 应该返回同一个实例
        assert logger1 is logger2

    def test_get_logger_new(self):
        """测试获取新的日志记录器"""
        logger = get_logger("test_new_logger")
        
        assert logger.name == "test_new_logger"
        assert logger.handlers  # 应该自动添加处理器


class TestLoggerContext:
    """测试日志上下文管理器"""

    def test_logger_context_temporary_level(self):
        """测试临时修改日志级别"""
        logger = setup_logger("test_context", level=logging.WARNING)
        original_level = logger.level
        
        with LoggerContext(logger, level=logging.DEBUG):
            assert logger.level == logging.DEBUG
            logger.debug("This should be logged")
        
        # 恢复原级别
        assert logger.level == original_level

    def test_logger_context_preserve_level(self):
        """测试不修改级别时保持原级别"""
        logger = setup_logger("test_preserve", level=logging.INFO)
        original_level = logger.level
        
        with LoggerContext(logger, level=None):
            assert logger.level == original_level


class TestLogFormatter:
    """测试日志格式化器"""

    def test_log_formatter_with_color(self):
        """测试彩色格式化器"""
        formatter = LogFormatter(use_color=True)
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        assert "Test message" in formatted
        assert "INFO" in formatted

    def test_log_formatter_without_color(self):
        """测试非彩色格式化器"""
        formatter = LogFormatter(use_color=False)
        
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        assert "Error message" in formatted
        assert "ERROR" in formatted


class TestGlobalLogging:
    """测试全局日志配置"""

    def test_configure_global_logging(self):
        """测试全局日志配置"""
        configure_global_logging(
            level=logging.DEBUG,
            use_console=True,
            use_color=True
        )
        
        # 配置应该被保存
        # 实际测试需要验证全局配置变量的值

    def test_multiple_loggers_use_global_config(self):
        """测试多个日志记录器使用全局配置"""
        configure_global_logging(level=logging.WARNING)
        
        logger1 = setup_logger("global_test_1")
        logger2 = setup_logger("global_test_2")
        
        # 两个日志记录器应该都能正常工作
        logger1.warning("Test 1")
        logger2.warning("Test 2")


class TestLoggerFunctionality:
    """测试日志功能"""

    def test_logger_levels(self):
        """测试不同日志级别"""
        logger = setup_logger("test_levels", level=logging.DEBUG)
        
        # 不应该抛出异常
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_logger_with_exception(self):
        """测试记录异常信息"""
        logger = setup_logger("test_exception")
        
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            # 不应该抛出异常
            logger.error("Error occurred", exc_info=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
