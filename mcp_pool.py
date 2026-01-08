#!/usr/bin/env python3
"""
MCP连接池和缓存管理
提供高性能的MCP客户端复用、查询缓存和批量查询功能
"""

import json
import asyncio
import hashlib
import logging
from typing import Dict, Any, Optional, List
from functools import lru_cache
from datetime import datetime, timedelta

# 优先使用新的搜索客户端，如果不存在则使用旧的MCP客户端
try:
    from mcp_search_client import MCPSearchClient as MCPClient
    logger = logging.getLogger(__name__)
    logger.info("✅ 使用新的MCP搜索客户端")

    # 创建适配器函数，将 mcp_url 参数映射到 search_url
    def create_mcp_client(mcp_url: str, client_name: str = "mcp-pool-client", **kwargs) -> MCPClient:
        """
        创建MCP客户端的适配器函数

        Args:
            mcp_url: 搜索服务地址
            client_name: 客户端名称
            **kwargs: 其他参数

        Returns:
            MCPClient: 客户端实例
        """
        from mcp_search_client import create_mcp_search_client
        return create_mcp_search_client(
            search_url=mcp_url,
            client_name=client_name,
            **kwargs
        )

except ImportError:
    from mcp_client import MCPClient, create_mcp_client
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ 降级使用旧的MCP客户端")


class MCPConnectionPool:
    """MCP连接池管理类"""

    def __init__(self,
                 mcp_url: str,
                 pool_size: int = 3,
                 client_name: str = "mcp-pool-client"):
        """
        初始化连接池

        Args:
            mcp_url: MCP服务地址
            pool_size: 连接池大小
            client_name: 客户端名称
        """
        self.mcp_url = mcp_url
        self.pool_size = pool_size
        self.client_name = client_name
        self._pool: List[MCPClient] = []
        self._current_index = 0
        self._lock = asyncio.Lock()

    async def get_client(self) -> MCPClient:
        """
        从连接池获取客户端

        Returns:
            MCPClient: 可用的客户端实例
        """
        async with self._lock:
            # 如果池未满,创建新客户端
            if len(self._pool) < self.pool_size:
                client = create_mcp_client(
                    mcp_url=self.mcp_url,
                    client_name=f"{self.client_name}-{len(self._pool)}"
                )
                # 异步初始化
                if await client.initialize_async():
                    self._pool.append(client)
                    logger.info(f"创建新MCP客户端,当前池大小: {len(self._pool)}")
                    return client
                else:
                    logger.error("MCP客户端初始化失败")

            # 使用轮询策略分配客户端
            if self._pool:
                client = self._pool[self._current_index % len(self._pool)]
                self._current_index += 1
                return client

            # 降级:创建临时客户端
            logger.warning("连接池为空,创建临时客户端")
            client = create_mcp_client(mcp_url=self.mcp_url)
            await client.initialize_async()
            return client

    async def close_all(self):
        """关闭所有连接"""
        async with self._lock:
            for client in self._pool:
                try:
                    await client.close_async()
                except Exception as e:
                    logger.error(f"关闭客户端失败: {e}")
            self._pool.clear()
            logger.info("所有MCP连接已关闭")


class MCPCache:
    """MCP查询缓存管理类"""

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        """
        初始化缓存

        Args:
            ttl_seconds: 缓存过期时间(秒)
            max_size: 最大缓存条目数
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_time: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    def _generate_key(self, question: str, top_k: int) -> str:
        """生成缓存键"""
        content = f"{question}:{top_k}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    async def get(self, question: str, top_k: int) -> Optional[Dict[str, Any]]:
        """
        从缓存获取结果

        Args:
            question: 搜索问题
            top_k: 返回结果数量

        Returns:
            缓存的结果,如果不存在或已过期返回None
        """
        cache_key = self._generate_key(question, top_k)

        async with self._lock:
            if cache_key not in self._cache:
                return None

            # 检查是否过期
            cache_time = self._access_time.get(cache_key)
            if cache_time and datetime.now() - cache_time > timedelta(seconds=self.ttl_seconds):
                # 缓存过期,删除
                del self._cache[cache_key]
                del self._access_time[cache_key]
                logger.info(f"缓存过期并删除: {question[:50]}...")
                return None

            # 更新访问时间
            self._access_time[cache_key] = datetime.now()
            logger.info(f"缓存命中: {question[:50]}...")
            return self._cache[cache_key]

    async def set(self, question: str, top_k: int, result: Dict[str, Any]):
        """
        设置缓存

        Args:
            question: 搜索问题
            top_k: 返回结果数量
            result: 查询结果
        """
        cache_key = self._generate_key(question, top_k)

        async with self._lock:
            # LRU淘汰策略
            if len(self._cache) >= self.max_size:
                # 找到最久未使用的条目
                oldest_key = min(self._access_time.keys(),
                                key=lambda k: self._access_time[k])
                del self._cache[oldest_key]
                del self._access_time[oldest_key]
                logger.info(f"缓存已满,淘汰最久未使用条目")

            self._cache[cache_key] = result
            self._access_time[cache_key] = datetime.now()
            logger.info(f"缓存已设置: {question[:50]}...")

    async def clear(self):
        """清空缓存"""
        async with self._lock:
            self._cache.clear()
            self._access_time.clear()
            logger.info("缓存已清空")

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }


class MCPQueryOptimizer:
    """MCP查询优化器 - 提供批量查询和智能优化"""

    def __init__(self,
                 mcp_url: str,
                 pool_size: int = 3,
                 cache_ttl: int = 3600,
                 cache_max_size: int = 1000):
        """
        初始化查询优化器

        Args:
            mcp_url: MCP服务地址
            pool_size: 连接池大小
            cache_ttl: 缓存过期时间(秒)
            cache_max_size: 最大缓存条目数
        """
        self.mcp_url = mcp_url
        self.pool = MCPConnectionPool(mcp_url, pool_size)
        self.cache = MCPCache(cache_ttl, cache_max_size)

    async def search_single(self,
                           question: str,
                           top_k: int = 5,
                           use_cache: bool = True,
                           timeout: float = 5.0) -> str:
        """
        单次查询(带缓存和超时控制)

        Args:
            question: 搜索问题
            top_k: 返回结果数量
            use_cache: 是否使用缓存
            timeout: 超时时间(秒)

        Returns:
            str: JSON格式的搜索结果
        """
        # 尝试从缓存获取
        if use_cache:
            cached_result = await self.cache.get(question, top_k)
            if cached_result is not None:
                return json.dumps(cached_result, ensure_ascii=False)

        try:
            # 使用连接池获取客户端
            client = await self.pool.get_client()

            # 带超时的查询
            result = await asyncio.wait_for(
                client.search_async(question, top_k),
                timeout=timeout
            )

            # 缓存结果
            if use_cache and isinstance(result, dict):
                await self.cache.set(question, top_k, result)

            logger.info(f"查询成功: {question[:50]}...")
            return json.dumps(result, ensure_ascii=False)

        except asyncio.TimeoutError:
            logger.error(f"查询超时({timeout}秒): {question[:50]}...")
            return json.dumps({
                "status": "timeout",
                "message": f"MCP服务响应超时({timeout}秒)",
                "fallback_mode": "使用经验估算",
                "results": []
            }, ensure_ascii=False)

        except Exception as e:
            logger.error(f"查询异常: {e}, 问题: {question[:50]}...")
            return json.dumps({
                "status": "error",
                "message": str(e),
                "fallback_mode": "使用经验估算",
                "results": []
            }, ensure_ascii=False)

    async def search_batch(self,
                          questions: List[str],
                          top_k: int = 5,
                          use_cache: bool = True,
                          timeout: float = 5.0,
                          max_concurrency: int = 5) -> List[str]:
        """
        批量并发查询

        Args:
            questions: 问题列表
            top_k: 返回结果数量
            use_cache: 是否使用缓存
            timeout: 单个查询超时时间(秒)
            max_concurrency: 最大并发数

        Returns:
            List[str]: JSON格式的搜索结果列表
        """
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(max_concurrency)

        async def search_with_semaphore(question: str) -> str:
            async with semaphore:
                return await self.search_single(
                    question=question,
                    top_k=top_k,
                    use_cache=use_cache,
                    timeout=timeout
                )

        # 并发执行所有查询
        tasks = [search_with_semaphore(q) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"批量查询第{i+1}项失败: {result}")
                formatted_results.append(json.dumps({
                    "status": "error",
                    "message": str(result),
                    "fallback_mode": "使用经验估算",
                    "results": []
                }, ensure_ascii=False))
            else:
                formatted_results.append(result)

        logger.info(f"批量查询完成: {len(questions)}个问题")
        return formatted_results

    async def close(self):
        """关闭优化器"""
        await self.pool.close_all()
        await self.cache.clear()


# 全局优化器实例
_global_optimizer: Optional[MCPQueryOptimizer] = None


def get_optimizer(mcp_url: Optional[str] = None) -> MCPQueryOptimizer:
    """
    获取全局查询优化器实例

    Args:
        mcp_url: MCP服务地址,默认从环境变量读取

    Returns:
        MCPQueryOptimizer: 查询优化器实例
    """
    global _global_optimizer

    if _global_optimizer is None:
        import os
        
        # 优先使用新的搜索服务地址
        mcp_url = mcp_url or os.getenv(
            "KNOWLEDGE_BASE_SEARCH_URL",
            os.getenv(
                "MCP_URL",
                "http://192.168.244.189:8003/search"  # 新的默认地址
            )
        )

        logger.info(f"📡 知识库搜索服务地址: {mcp_url}")
        logger.info(f"💡 提示: 如果连接失败，请检查:")
        logger.info(f"   1. 搜索服务是否正在运行")
        logger.info(f"   2. 网络连接是否正常")
        logger.info(f"   3. URL配置是否正确")
        logger.info(f"   4. 可通过 export KNOWLEDGE_BASE_SEARCH_URL=your_url 设置自定义地址")

        _global_optimizer = MCPQueryOptimizer(
            mcp_url=mcp_url,
            pool_size=3,  # 连接池大小
            cache_ttl=3600,  # 缓存1小时
            cache_max_size=1000  # 最多缓存1000条
        )
        logger.info("✅ 全局MCP查询优化器已初始化")

    return _global_optimizer


async def close_global_optimizer():
    """关闭全局优化器"""
    global _global_optimizer
    if _global_optimizer:
        await _global_optimizer.close()
        _global_optimizer = None
        logger.info("全局MCP查询优化器已关闭")
