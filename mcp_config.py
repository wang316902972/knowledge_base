#!/usr/bin/env python3
"""
MCP工具包配置示例
展示如何配置和使用MCP工具包的各种功能
"""

import asyncio
import logging
from mcp_toolkit import (
    MCPToolkit,
    PoolConfig,
    RetryConfig,
    CircuitBreakerConfig,
    get_toolkit,
    with_toolkit
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 配置示例
def create_configs():
    """创建各种配置示例"""

    # 1. 基础配置 - 适合低负载场景
    basic_pool = PoolConfig(
        min_size=2,
        max_size=5,
        idle_timeout=300.0,
        max_lifetime=3600.0
    )

    basic_retry = RetryConfig(
        max_attempts=2,
        base_delay=1.0,
        max_delay=5.0
    )

    basic_circuit = CircuitBreakerConfig(
        failure_threshold=10,
        success_threshold=2,
        timeout=60.0
    )

    # 2. 高性能配置 - 适合高负载场景
    high_perf_pool = PoolConfig(
        min_size=5,
        max_size=20,
        idle_timeout=600.0,
        max_lifetime=7200.0
    )

    high_perf_retry = RetryConfig(
        max_attempts=5,
        base_delay=0.5,
        max_delay=30.0,
        exponential_base=2.0
    )

    high_perf_circuit = CircuitBreakerConfig(
        failure_threshold=20,
        success_threshold=3,
        timeout=30.0
    )

    # 3. 安全配置 - 适合生产环境
    safe_pool = PoolConfig(
        min_size=3,
        max_size=10,
        idle_timeout=300.0,
        max_lifetime=3600.0
    )

    safe_retry = RetryConfig(
        max_attempts=3,
        base_delay=2.0,
        max_delay=10.0,
        jitter=True
    )

    safe_circuit = CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout=60.0
    )

    return {
        "basic": (basic_pool, basic_retry, basic_circuit),
        "high_perf": (high_perf_pool, high_perf_retry, high_perf_circuit),
        "safe": (safe_pool, safe_retry, safe_circuit)
    }


# 使用示例
async def example_basic_usage():
    """示例1: 基础使用"""
    print("\n" + "="*60)
    print("示例1: 基础使用")
    print("="*60)

    # 创建工具包
    toolkit = MCPToolkit(
        mcp_url="http://192.168.244.189:8003"
    )

    try:
        # 搜索
        result = await toolkit.search("AI技术", top_k=3)
        print(f"搜索结果: {result}")
    finally:
        await toolkit.close()


async def example_custom_config():
    """示例2: 自定义配置"""
    print("\n" + "="*60)
    print("示例2: 自定义配置")
    print("="*60)

    configs = create_configs()
    pool_cfg, retry_cfg, circuit_cfg = configs["safe"]

    toolkit = MCPToolkit(
        mcp_url="http://192.168.244.189:8003",
        pool_config=pool_cfg,
        retry_config=retry_cfg,
        circuit_config=circuit_cfg
    )

    try:
        # 搜索
        result = await toolkit.search("机器学习", top_k=5)
        print(f"搜索结果: {result}")

        # 查看统计
        stats = await toolkit.get_stats()
        print(f"\n统计信息:")
        print(f"- 总调用次数: {stats['pool_metrics']['total_calls']}")
        print(f"- 成功率: {stats['pool_metrics']['success_rate']:.2%}")
        print(f"- 平均响应时间: {stats['pool_metrics']['avg_call_duration']:.3f}秒")
        print(f"- 熔断器状态: {stats['circuit_breaker']['state']}")

    finally:
        await toolkit.close()


async def example_tool_list():
    """示例3: 获取和使用工具列表"""
    print("\n" + "="*60)
    print("示例3: 工具列表和使用")
    print("="*60)

    toolkit = get_toolkit()

    try:
        # 获取工具列表
        tools = await toolkit.list_tools()
        print(f"\n可用工具 ({len(tools)} 个):")
        for tool in tools:
            print(f"\n📦 {tool['name']}")
            print(f"   描述: {tool['description']}")

        # 使用工具调用
        result = await toolkit.call_tool(
            tool_name="search_knowledge",
            arguments={
                "query": "深度学习",
                "top_k": 3
            }
        )
        print(f"\n调用结果: {result}")

    finally:
        await toolkit.close()


async def example_batch_operations():
    """示例4: 批量操作"""
    print("\n" + "="*60)
    print("示例4: 批量操作")
    print("="*60)

    toolkit = get_toolkit()

    try:
        # 批量搜索
        queries = [
            "人工智能",
            "机器学习",
            "深度学习",
            "神经网络",
            "自然语言处理"
        ]

        calls = [
            {
                "tool_name": "search_knowledge",
                "arguments": {"query": q, "top_k": 2}
            }
            for q in queries
        ]

        print(f"\n批量搜索 {len(calls)} 个查询...")
        results = await toolkit.batch_call_tools(
            calls=calls,
            max_concurrency=3
        )

        for i, (query, result) in enumerate(zip(queries, results)):
            print(f"\n查询 {i+1}: {query}")
            if "error" in result:
                print(f"  ❌ 错误: {result['error']}")
            else:
                print(f"  ✅ 成功")

    finally:
        await toolkit.close()


@with_toolkit(mcp_url="http://192.168.244.189:8003")
async def example_decorator_usage(toolkit: MCPToolkit, query: str):
    """示例5: 使用装饰器"""
    print("\n" + "="*60)
    print("示例5: 装饰器使用")
    print("="*60)

    result = await toolkit.search(query, top_k=3)
    print(f"搜索结果: {result}")
    return result


async def example_error_handling():
    """示例6: 错误处理和重试"""
    print("\n" + "="*60)
    print("示例6: 错误处理和重试")
    print("="*60)

    # 配置激进的重试策略
    retry_cfg = RetryConfig(
        max_attempts=5,
        base_delay=1.0,
        max_delay=10.0,
        exponential_base=2.0,
        jitter=True
    )

    circuit_cfg = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=30.0
    )

    toolkit = MCPToolkit(
        mcp_url="http://192.168.244.189:8003",
        retry_config=retry_cfg,
        circuit_config=circuit_cfg
    )

    try:
        # 尝试调用不存在的工具
        result = await toolkit.call_tool(
            tool_name="invalid_tool",
            arguments={},
            timeout=5.0
        )

        if "error" in result:
            print(f"捕获到错误: {result['error']}")

        # 查看熔断器状态
        stats = await toolkit.get_stats()
        print(f"\n熔断器状态: {stats['circuit_breaker']['state']}")
        print(f"失败次数: {stats['circuit_breaker']['failure_count']}")

    finally:
        await toolkit.close()


async def example_performance_monitoring():
    """示例7: 性能监控"""
    print("\n" + "="*60)
    print("示例7: 性能监控")
    print("="*60)

    toolkit = get_toolkit()

    try:
        # 执行多次查询
        queries = ["AI"] * 10

        for i, query in enumerate(queries, 1):
            result = await toolkit.search(
                f"{query} 查询 {i}",
                top_k=1
            )
            print(f"查询 {i}: 完成")

        # 获取性能指标
        stats = await toolkit.get_stats()
        metrics = stats['pool_metrics']

        print(f"\n性能指标:")
        print(f"- 总调用次数: {metrics['total_calls']}")
        print(f"- 成功次数: {metrics['successful_calls']}")
        print(f"- 失败次数: {metrics['failed_calls']}")
        print(f"- 成功率: {metrics['success_rate']:.2%}")
        print(f"- 平均响应时间: {metrics['avg_call_duration']:.3f}秒")
        print(f"- 缓存命中次数: {metrics['cache_hits']}")
        print(f"- 缓存命中率: {metrics['cache_hit_rate']:.2%}")

    finally:
        await toolkit.close()


async def example_concurrent_requests():
    """示例8: 并发请求"""
    print("\n" + "="*60)
    print("示例8: 并发请求")
    print("="*60)

    toolkit = get_toolkit()

    try:
        # 创建多个并发任务
        tasks = [
            toolkit.search(f"并发查询 {i}", top_k=1)
            for i in range(20)
        ]

        print(f"执行 {len(tasks)} 个并发查询...")
        start_time = asyncio.get_event_loop().time()

        results = await asyncio.gather(*tasks)

        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time

        print(f"\n完成!")
        print(f"- 总耗时: {duration:.2f}秒")
        print(f"- 平均每个查询: {duration/len(tasks):.3f}秒")
        print(f"- 吞吐量: {len(tasks)/duration:.2f} 查询/秒")

    finally:
        await toolkit.close()


# 主函数
async def main():
    """运行所有示例"""
    examples = [
        ("基础使用", example_basic_usage),
        ("自定义配置", example_custom_config),
        ("工具列表", example_tool_list),
        ("批量操作", example_batch_operations),
        ("装饰器使用", lambda: example_decorator_usage("装饰器示例")),
        ("错误处理", example_error_handling),
        ("性能监控", example_performance_monitoring),
        ("并发请求", example_concurrent_requests),
    ]

    print("\n" + "="*60)
    print("MCP工具包使用示例")
    print("="*60)

    for name, example_func in examples:
        try:
            await example_func()
            print(f"\n✅ {name} - 完成")
        except Exception as e:
            print(f"\n❌ {name} - 失败: {e}")
            logger.exception(f"示例 {name} 执行失败")

        # 等待一段时间再执行下一个示例
        await asyncio.sleep(1)

    print("\n" + "="*60)
    print("所有示例执行完毕")
    print("="*60)


if __name__ == "__main__":
    # 运行所有示例
    asyncio.run(main())

    # 或者运行单个示例
    # asyncio.run(example_basic_usage())
    # asyncio.run(example_custom_config())
    # asyncio.run(example_tool_list())
    # asyncio.run(example_batch_operations())
    # asyncio.run(example_decorator_usage("测试查询"))
    # asyncio.run(example_error_handling())
    # asyncio.run(example_performance_monitoring())
    # asyncio.run(example_concurrent_requests())
