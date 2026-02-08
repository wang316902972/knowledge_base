"""
统一日志模块

提供项目中所有模块的日志配置和管理。
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class LogLevel:
    """日志级别常量"""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogFormatter(logging.Formatter):
    """自定义日志格式化器

    提供更详细的日志格式，包括时间戳、级别、文件名、行号等。
    """

    # 颜色代码（用于终端输出）
    COLORS = {
        logging.DEBUG: "\033[36m",  # 青色
        logging.INFO: "\033[32m",  # 绿色
        logging.WARNING: "\033[33m",  # 黄色
        logging.ERROR: "\033[31m",  # 红色
        logging.CRITICAL: "\033[35m",  # 紫色
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True, include_caller: bool = True) -> None:
        """初始化格式化器

        Args:
            use_color: 是否在终端中使用颜色
            include_caller: 是否包含调用者信息（文件名和行号）
        """
        self.use_color = use_color and sys.stderr.isatty()
        self.include_caller = include_caller

        # 构建格式字符串
        if include_caller:
            fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
        else:
            fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

        super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录

        Args:
            record: 日志记录对象

        Returns:
            格式化后的日志字符串
        """
        # 调用父类方法获取基本格式
        result = super().format(record)

        # 添加颜色（如果启用且是终端）
        if self.use_color:
            level_color = self.COLORS.get(record.levelno, "")
            result = f"{level_color}{result}{self.RESET}"

        return result


def setup_logger(
    name: str,
    level: int = LogLevel.INFO,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    use_console: bool = True,
    use_color: bool = True,
    include_caller: bool = True,
) -> logging.Logger:
    """设置并返回一个配置好的日志记录器

    Args:
        name: 日志记录器名称（通常使用 __name__）
        level: 日志级别
        log_file: 日志文件路径（如果指定，则输出到文件）
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的备份文件数量
        use_console: 是否输出到控制台
        use_color: 控制台输出是否使用颜色
        include_caller: 是否包含调用者信息（文件名和行号）

    Returns:
        配置好的日志记录器

    Example:
        >>> logger = setup_logger(__name__, level=logging.DEBUG)
        >>> logger.info("This is an info message")
        >>> logger.error("This is an error message")
    """
    # 创建日志记录器
    logger = logging.getLogger(name)

    # 如果已经配置过处理器，直接返回
    if logger.handlers:
        return logger

    # 设置日志级别
    logger.setLevel(level)

    # 清除任何现有的处理器
    logger.handlers.clear()

    # 创建格式化器
    formatter = LogFormatter(use_color=use_color, include_caller=include_caller)

    # 添加控制台处理器
    if use_console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 添加文件处理器（如果指定）
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用滚动文件处理器
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        # 文件输出不使用颜色
        file_formatter = LogFormatter(use_color=False, include_caller=include_caller)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # 防止日志传播到父日志记录器
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """获取一个已配置的日志记录器

    这是一个便捷函数，用于获取已经通过 setup_logger 配置过的日志记录器。
    如果日志记录器不存在，则返回一个默认配置的日志记录器。

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
    """
    logger = logging.getLogger(name)

    # 如果日志记录器还没有处理器，添加一个默认的控制台处理器
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(LogFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


class LoggerContext:
    """日志上下文管理器

    用于临时修改日志级别或在特定上下文中捕获日志。

    Example:
        >>> with LoggerContext(logger, level=logging.DEBUG):
        ...     logger.debug("This will be logged")
        >>> logger.debug("This won't be logged")
    """

    def __init__(self, logger: logging.Logger, level: Optional[int] = None) -> None:
        """初始化上下文管理器

        Args:
            logger: 日志记录器
            level: 临时设置的日志级别（None 表示不修改）
        """
        self.logger = logger
        self.level = level
        self.original_level: Optional[int] = None

    def __enter__(self) -> logging.Logger:
        """进入上下文

        Returns:
            日志记录器
        """
        if self.level is not None:
            self.original_level = self.logger.level
            self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """退出上下文，恢复原始日志级别"""
        if self.original_level is not None:
            self.logger.setLevel(self.original_level)


# 全局日志配置
_global_config = {
    "level": logging.INFO,
    "log_file": None,
    "max_bytes": 10 * 1024 * 1024,
    "backup_count": 5,
    "use_console": True,
    "use_color": True,
    "include_caller": True,
}


def configure_global_logging(
    level: int = LogLevel.INFO,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    use_console: bool = True,
    use_color: bool = True,
    include_caller: bool = True,
) -> None:
    """配置全局日志设置

    这个函数会影响所有后续创建的日志记录器。

    Args:
        level: 默认日志级别
        log_file: 默认日志文件路径
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的备份文件数量
        use_console: 是否默认输出到控制台
        use_color: 控制台输出是否默认使用颜色
        include_caller: 是否默认包含调用者信息
    """
    _global_config.update(
        {
            "level": level,
            "log_file": log_file,
            "max_bytes": max_bytes,
            "backup_count": backup_count,
            "use_console": use_console,
            "use_color": use_color,
            "include_caller": include_caller,
        }
    )


# 便捷的日志记录器获取函数
def create_logger(name: str) -> logging.Logger:
    """创建一个新的日志记录器（使用全局配置）

    Args:
        name: 日志记录器名称

    Returns:
        配置好的日志记录器
    """
    return setup_logger(name, **_global_config)
