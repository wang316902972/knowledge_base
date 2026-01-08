#!/usr/bin/env python3
"""
MCP工具包 - 提供高级连接池管理和工具调用功能
特性:
- 连接池管理
- 自动重试机制
- 熔断器模式
- 工具调用代理
- 批量操作
- 性能监控
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import functools

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"  # 正常运行
    OPEN = "open"  # 熔断打开，拒绝请求
    HALF_OPEN = "half_open"  # 半开，尝试恢复


@dataclass
class PoolConfig:
    """连接池配置"""
    min_size: int = 2  # 最小连接数
    max_size: int = 10  # 最大连接数
    idle_timeout: float = 300.0  # 空闲连接超时(秒)
    max_lifetime: float = 3600.0  # 连接最大生命周期(秒)
    acquisition_timeout: float = 10.0  # 获取连接超时(秒)


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3  # 最大重试次数
    base_delay: float = 1.0  # 基础延迟(秒)
    max_delay: float = 10.0  # 最大延迟(秒)
    exponential_base: float = 2.0  # 指数退避基数
    jitter: bool = True  # 是否添加随机抖动


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5  # 失败阈值
    success_threshold: int = 2  # 成功阈值(半开状态)
    timeout: float = 60.0  # 熔断超时(秒)
    window_size: int = 100  # 滑动窗口大小


@dataclass
class Metrics:
    """性能指标"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    last_call_time: Optional[datetime] = None
    avg_call_duration: float = 0.0

    def record_call(self, duration: float, success: bool):
        """记录调用"""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        self.total_duration += duration
        self.last_call_time = datetime.now()
        if self.total_calls > 0:
            self.avg_call_duration = self.total_duration / self.total_calls

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls if self.total_calls > 0 else 0,
            "avg_call_duration": self.avg_call_duration,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 0,
            "last_call_time": self.last_call_time.isoformat()
            if self.last_call_time else None
        }


class CircuitBreaker:
    """熔断器实现"""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """通过熔断器调用函数"""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                # 检查是否可以尝试恢复
                if (self.last_failure_time and
                    datetime.now() - self.last_failure_time >
                        timedelta(seconds=self.config.timeout)):
                    self.state = CircuitState.HALF_OPEN
                    logger.info("熔断器进入半开状态")
                else:
                    raise Exception("熔断器打开，拒绝请求")

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e

    async def _on_success(self):
        """处理成功"""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info("熔断器恢复到关闭状态")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)

    async def _on_failure(self):
        """处理失败"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
                logger.warning("熔断器从半开状态切换到打开状态")
            elif (self.state == CircuitState.CLOSED and
                  self.failure_count >= self.config.failure_threshold):
                self.state = CircuitState.OPEN
                logger.warning(f"熔断器打开(失败次数: {self.failure_count})")

    def get_state(self) -> Dict[str, Any]:
        """获取熔断器状态"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat()
            if self.last_failure_time else None
        }


class RetryHandler:
    """重试处理器"""

    def __init__(self, config: RetryConfig):
        self.config = config

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """带重试的函数调用"""
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"调用失败(尝试 {attempt + 1}/{self.config.max_attempts}): {e}"
                    )
                    logger.info(f"等待 {delay:.2f} 秒后重试...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"所有重试均失败 ({self.config.max_attempts} 次): {e}"
                    )

        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """计算延迟时间"""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )

        if self.config.jitter:
            import random
            delay *= (0.5 + random.random())

        return delay


class EnhancedMCPConnectionPool:
    """增强的MCP连接池"""

    def __init__(self,
                 mcp_url: str,
                 pool_config: Optional[PoolConfig] = None,
                 retry_config: Optional[RetryConfig] = None,
                 circuit_config: Optional[CircuitBreakerConfig] = None):
        """
        初始化连接池

        Args:
            mcp_url: MCP服务地址
            pool_config: 连接池配置
            retry_config: 重试配置
            circuit_config: 熔断器配置
        """
        self.mcp_url = mcp_url
        self.pool_config = pool_config or PoolConfig()
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_config or CircuitBreakerConfig()

        self._pool: List[Any] = []
        self._pool_lock = asyncio.Lock()
        self._metrics = Metrics()

        # 初始化重试处理器和熔断器
        self.retry_handler = RetryHandler(self.retry_config)
        self.circuit_breaker = CircuitBreaker(self.circuit_config)

        logger.info(f"MCP连接池初始化: {mcp_url}")
        logger.info(f"池配置: min={self.pool_config.min_size}, "
                   f"max={self.pool_config.max_size}")

    async def _create_client(self) -> Any:
        """创建新的MCP客户端"""
        try:
            # 动态导入MCP客户端
            try:
                from mcp_search_client import MCPSearchClient as MCPClient
                from mcp_search_client import create_mcp_search_client
                client = create_mcp_search_client(
                    search_url=self.mcp_url,
                    client_name=f"mcp-toolkit-{len(self._pool)}"
                )
            except ImportError:
                from mcp_client import MCPClient, create_mcp_client
                client = create_mcp_client(
                    mcp_url=self.mcp_url,
                    client_name=f"mcp-toolkit-{len(self._pool)}"
                )

            # 异步初始化
            if hasattr(client, 'initialize_async'):
                await client.initialize_async()

            logger.info(f"创建新MCP客户端: {len(self._pool) + 1}")
            return client

        except Exception as e:
            logger.error(f"创建MCP客户端失败: {e}")
            raise

    async def get_client(self) -> Any:
        """从连接池获取客户端"""
        async with self._pool_lock:
            # 清理过期连接
            await self._cleanup_idle_connections()

            # 如果池为空或未达到最小大小,创建新连接
            if len(self._pool) < self.pool_config.min_size:
                client = await self._create_client()
                self._pool.append(client)
                return client

            # 返回可用连接
            if self._pool:
                return self._pool[0]

            # 创建临时连接
            logger.warning("连接池为空,创建临时连接")
            return await self._create_client()

    async def _cleanup_idle_connections(self):
        """清理空闲连接"""
        if not self._pool:
            return

        # 保留最小连接数
        while len(self._pool) > self.pool_config.max_size:
            client = self._pool.pop()
            if hasattr(client, 'close_async'):
                await client.close_async()

    async def close_all(self):
        """关闭所有连接"""
        async with self._pool_lock:
            for client in self._pool:
                try:
                    if hasattr(client, 'close_async'):
                        await client.close_async()
                except Exception as e:
                    logger.error(f"关闭客户端失败: {e}")
            self._pool.clear()
            logger.info("所有MCP连接已关闭")

    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self._metrics.to_dict()


class MCPToolkit:
    """MCP工具包 - 高级工具调用接口"""

    def __init__(self,
                 mcp_url: str,
                 pool_config: Optional[PoolConfig] = None,
                 retry_config: Optional[RetryConfig] = None,
                 circuit_config: Optional[CircuitBreakerConfig] = None,
                 enable_cache: bool = True,
                 cache_ttl: int = 3600):
        """
        初始化MCP工具包

        Args:
            mcp_url: MCP服务地址
            pool_config: 连接池配置
            retry_config: 重试配置
            circuit_config: 熔断器配置
            enable_cache: 是否启用缓存
            cache_ttl: 缓存生存时间(秒)
        """
        self.mcp_url = mcp_url
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl

        # 初始化连接池
        self.pool = EnhancedMCPConnectionPool(
            mcp_url=mcp_url,
            pool_config=pool_config,
            retry_config=retry_config,
            circuit_config=circuit_config
        )

        # 工具缓存
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_time: Optional[datetime] = None

        logger.info(f"MCP工具包初始化完成: {mcp_url}")

    async def call_tool(self,
                       tool_name: str,
                       arguments: Dict[str, Any],
                       timeout: float = 30.0) -> Dict[str, Any]:
        """
        调用MCP工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            timeout: 超时时间(秒)

        Returns:
            工具执行结果
        """
        start_time = time.time()
        success = False

        try:
            # 通过熔断器和重试机制调用
            result = await self.pool.circuit_breaker.call(
                self._execute_tool_with_retry,
                tool_name,
                arguments,
                timeout
            )

            success = True
            duration = time.time() - start_time
            self.pool._metrics.record_call(duration, success)

            return result

        except Exception as e:
            duration = time.time() - start_time
            self.pool._metrics.record_call(duration, success)
            logger.error(f"工具调用失败: {tool_name}, 错误: {e}")
            return {
                "error": str(e),
                "tool_name": tool_name,
                "arguments": arguments
            }

    async def _execute_tool_with_retry(self,
                                      tool_name: str,
                                      arguments: Dict[str, Any],
                                      timeout: float) -> Dict[str, Any]:
        """带重试的工具执行"""

        async def execute() -> Dict[str, Any]:
            client = await self.pool.get_client()

            # 带超时的执行
            result = await asyncio.wait_for(
                self._execute_tool(client, tool_name, arguments),
                timeout=timeout
            )

            return result

        # 使用重试机制
        return await self.pool.retry_handler.call(execute)

    async def _execute_tool(self,
                           client: Any,
                           tool_name: str,
                           arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具调用"""
        try:
            # 尝试使用MCP工具调用接口
            if hasattr(client, 'call_tool'):
                result = await client.call_tool(tool_name, arguments)
                return result
            elif hasattr(client, 'search_async'):
                # 兼容旧的搜索接口
                if tool_name == "search_knowledge":
                    query = arguments.get("query", "")
                    top_k = arguments.get("top_k", 5)
                    result = await client.search_async(query, top_k)
                    return result
                else:
                    raise ValueError(f"不支持的工具: {tool_name}")
            else:
                raise ValueError("客户端不支持工具调用")

        except Exception as e:
            logger.error(f"执行工具失败: {tool_name}, 错误: {e}")
            raise

    async def list_tools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        获取可用工具列表

        Args:
            force_refresh: 是否强制刷新

        Returns:
            工具列表
        """
        # 检查缓存
        if (not force_refresh and
            self._tools_cache is not None and
            self._cache_time is not None and
            datetime.now() - self._cache_time < timedelta(seconds=self.cache_ttl)):
            self.pool._metrics.cache_hits += 1
            return self._tools_cache

        self.pool._metrics.cache_misses += 1

        try:
            client = await self.pool.get_client()

            # 获取工具列表
            if hasattr(client, 'list_tools'):
                tools = await client.list_tools()
            else:
                # 默认工具列表
                tools = [
                    {
                        "name": "search_knowledge",
                        "description": "在向量数据库中搜索相关知识",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "top_k": {"type": "integer", "default": 5}
                            }
                        }
                    }
                ]

            self._tools_cache = tools
            self._cache_time = datetime.now()
            return tools

        except Exception as e:
            logger.error(f"获取工具列表失败: {e}")
            return []

    async def batch_call_tools(self,
                             calls: List[Dict[str, Any]],
                             max_concurrency: int = 5,
                             timeout: float = 30.0) -> List[Dict[str, Any]]:
        """
        批量调用工具

        Args:
            calls: 调用列表,每个元素包含 tool_name 和 arguments
            max_concurrency: 最大并发数
            timeout: 单个调用超时时间

        Returns:
            结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def call_with_limit(call_info: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self.call_tool(
                    tool_name=call_info["tool_name"],
                    arguments=call_info["arguments"],
                    timeout=timeout
                )

        tasks = [call_with_limit(call) for call in calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append({
                    "error": str(result),
                    "tool_name": calls[i]["tool_name"]
                })
            else:
                formatted_results.append(result)

        return formatted_results

    async def search(self,
                    query: str,
                    top_k: int = 5,
                    use_optimization: bool = True,
                    timeout: float = 10.0) -> Dict[str, Any]:
        """
        便捷的搜索方法

        Args:
            query: 搜索查询
            top_k: 返回结果数量
            use_optimization: 是否使用优化
            timeout: 超时时间

        Returns:
            搜索结果
        """
        return await self.call_tool(
            tool_name="search_knowledge",
            arguments={
                "query": query,
                "top_k": top_k,
                "use_optimization": use_optimization
            },
            timeout=timeout
        )

    async def get_stats(self) -> Dict[str, Any]:
        """获取工具包统计信息"""
        return {
            "pool_metrics": self.pool.get_metrics(),
            "circuit_breaker": self.pool.circuit_breaker.get_state(),
            "cache_info": {
                "enabled": self.enable_cache,
                "ttl": self.cache_ttl,
                "tools_cached": self._tools_cache is not None,
                "cache_time": self._cache_time.isoformat()
                if self._cache_time else None
            }
        }

    async def close(self):
        """关闭工具包"""
        await self.pool.close_all()
        self._tools_cache = None
        self._cache_time = None
        logger.info("MCP工具包已关闭")

    def __del__(self):
        """析构函数"""
        if hasattr(self, 'pool'):
            # 尝试清理资源
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.pool.close_all())
            except Exception:
                pass


# 全局工具包实例
_global_toolkit: Optional[MCPToolkit] = None


def get_toolkit(mcp_url: Optional[str] = None,
                pool_config: Optional[PoolConfig] = None,
                retry_config: Optional[RetryConfig] = None,
                circuit_config: Optional[CircuitBreakerConfig] = None) -> MCPToolkit:
    """
    获取全局MCP工具包实例

    Args:
        mcp_url: MCP服务地址
        pool_config: 连接池配置
        retry_config: 重试配置
        circuit_config: 熔断器配置

    Returns:
        MCPToolkit实例
    """
    global _global_toolkit

    if _global_toolkit is None:
        import os

        mcp_url = mcp_url or os.getenv(
            "KNOWLEDGE_BASE_SEARCH_URL",
            os.getenv(
                "MCP_URL",
                "http://192.168.244.189:8003"
            )
        )

        logger.info(f"初始化全局MCP工具包: {mcp_url}")
        _global_toolkit = MCPToolkit(
            mcp_url=mcp_url,
            pool_config=pool_config,
            retry_config=retry_config,
            circuit_config=circuit_config
        )

    return _global_toolkit


async def close_global_toolkit():
    """关闭全局工具包"""
    global _global_toolkit
    if _global_toolkit:
        await _global_toolkit.close()
        _global_toolkit = None
        logger.info("全局MCP工具包已关闭")


# 便捷装饰器
def with_toolkit(mcp_url: Optional[str] = None):
    """
    工具包装饰器 - 自动注入MCP工具包

    Usage:
        @with_toolkit(mcp_url="http://localhost:8003")
        async def my_function(toolkit: MCPToolkit, query: str):
            result = await toolkit.search(query)
            return result
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            toolkit = get_toolkit(mcp_url)
            return await func(toolkit=toolkit, *args, **kwargs)
        return wrapper
    return decorator


# 示例使用
async def main():
    """示例: 使用MCP工具包"""

    # 创建工具包实例
    toolkit = get_toolkit(
        mcp_url="http://192.168.244.189:8003",
        pool_config=PoolConfig(min_size=2, max_size=5),
        retry_config=RetryConfig(max_attempts=3),
        circuit_config=CircuitBreakerConfig(failure_threshold=5)
    )

    try:
        # 1. 获取可用工具列表
        print("\n=== 获取工具列表 ===")
        tools = await toolkit.list_tools()
        for tool in tools:
            print(f"- {tool['name']}: {tool['description']}")

        # 2. 单次搜索
        print("\n=== 单次搜索 ===")
        result = await toolkit.search(
            query="人工智能",
            top_k=3,
            timeout=10.0
        )
        print(f"搜索结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

        # 3. 批量调用
        print("\n=== 批量调用 ===")
        calls = [
            {"tool_name": "search_knowledge",
             "arguments": {"query": "机器学习", "top_k": 2}},
            {"tool_name": "search_knowledge",
             "arguments": {"query": "深度学习", "top_k": 2}},
            {"tool_name": "search_knowledge",
             "arguments": {"query": "神经网络", "top_k": 2}}
        ]
        results = await toolkit.batch_call_tools(calls, max_concurrency=3)
        for i, result in enumerate(results):
            print(f"结果 {i+1}: {json.dumps(result, ensure_ascii=False, indent=2)}")

        # 4. 获取统计信息
        print("\n=== 统计信息 ===")
        stats = await toolkit.get_stats()
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    finally:
        # 清理资源
        await toolkit.close()


if __name__ == "__main__":
    asyncio.run(main())
