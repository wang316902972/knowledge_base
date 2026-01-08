#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Client Utility
基于 test_mcp_with_session.py 封装的 MCP 客户端工具
"""

import json
import time
import asyncio
import logging
from typing import Dict, Any, Optional, Union
import requests

# httpx 是可选依赖
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP 客户端类，提供与 MCP 服务通信的功能"""

    def __init__(self,
                 mcp_url: str = "http://192.168.244.189:5678/mcp/d73b2d78-7565-4e75-9943-665d5b4bdb18",
                 client_name: str = "mcp-client",
                 client_version: str = "1.0.0"):
        """
        初始化 MCP 客户端

        Args:
            mcp_url: MCP 服务地址
            client_name: 客户端名称
            client_version: 客户端版本
        """
        self.mcp_url = mcp_url
        self.client_name = client_name
        self.client_version = client_version
        self.session = None
        self.session_id = None
        self.headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json, text/event-stream; charset=utf-8",
            "Accept-Charset": "utf-8"
        }
        self.is_initialized = False

    def initialize(self) -> bool:
        """
        初始化 MCP 连接（同步版本）

        Returns:
            bool: 初始化是否成功
        """
        try:
            if not self.session:
                self.session = requests.Session()

            # 1. 初始化 MCP 连接
            init_data = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {
                        "name": self.client_name,
                        "version": self.client_version
                    }
                },
                "id": 1
            }

            response = self.session.post(self.mcp_url, json=init_data, headers=self.headers)
            logger.info(f"MCP初始化状态: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"MCP初始化失败: {response.text}")
                return False

            # 解析响应
            self._parse_init_response(response)

            # 2. 发送 initialized 通知
            self._send_initialized_notification()

            # 等待服务器处理
            time.sleep(0.5)

            self.is_initialized = True
            logger.info("MCP客户端初始化成功")
            return True

        except Exception as e:
            logger.error(f"MCP初始化异常: {e}")
            return False

    async def initialize_async(self) -> bool:
        """
        初始化 MCP 连接（异步版本）

        Returns:
            bool: 初始化是否成功
        """
        if not HTTPX_AVAILABLE:
            logger.error("httpx 未安装，无法使用异步功能")
            return False

        try:
            if not self.session:
                self.session = httpx.AsyncClient(timeout=30.0)

            # 1. 初始化 MCP 连接
            init_data = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {
                        "name": self.client_name,
                        "version": self.client_version
                    }
                },
                "id": 1
            }

            response = await self.session.post(self.mcp_url, json=init_data, headers=self.headers)
            logger.info(f"MCP初始化状态: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"MCP初始化失败: {response.text}")
                return False

            # 解析响应
            self._parse_init_response(response)

            # 2. 发送 initialized 通知
            await self._send_initialized_notification_async()

            # 等待服务器处理
            await asyncio.sleep(0.3)

            self.is_initialized = True
            logger.info("MCP客户端异步初始化成功")
            return True

        except Exception as e:
            logger.error(f"MCP异步初始化异常: {e}")
            return False

    def _parse_init_response(self, response) -> None:
        """解析初始化响应"""
        try:
            # 提取会话 ID
            session_id = response.headers.get('mcp-session-id')
            if session_id:
                self.session_id = session_id
                self.headers['mcp-session-id'] = session_id
                logger.info(f"获取到会话ID: {session_id}")

            # 检查 SSE 格式响应
            if "event:" in response.text:
                logger.info("响应为SSE格式")
                lines = response.text.split('\n')
                for line in lines:
                    if line.startswith('data: '):
                        try:
                            json_data = line[6:]
                            parsed = json.loads(json_data)
                            if 'result' in parsed and 'protocolVersion' in parsed['result']:
                                logger.info("协议版本匹配")
                                break
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.warning(f"解析初始化响应时出错: {e}")

    def _send_initialized_notification(self) -> None:
        """发送 initialized 通知（同步版本）"""
        try:
            notify_data = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }

            response = self.session.post(self.mcp_url, json=notify_data, headers=self.headers)
            logger.info(f"MCP通知状态: {response.status_code}")

            if response.status_code not in [200, 204]:
                logger.warning(f"通知响应异常: {response.text[:200]}")

        except Exception as e:
            logger.warning(f"发送通知异常（可能不影响后续调用）: {e}")

    async def _send_initialized_notification_async(self) -> None:
        """发送 initialized 通知（异步版本）"""
        try:
            notify_data = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }

            response = await self.session.post(self.mcp_url, json=notify_data, headers=self.headers)
            logger.info(f"MCP异步通知状态: {response.status_code}")

        except Exception as e:
            logger.warning(f"发送异步通知异常（可能不影响后续调用）: {e}")

    def search(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        执行搜索（同步版本）

        Args:
            question: 搜索问题
            top_k: 返回结果数量

        Returns:
            Dict: 搜索结果
        """
        if not self.is_initialized:
            if not self.initialize():
                return {
                    "error": "MCP客户端未初始化",
                    "status": "not_initialized"
                }

        try:
            search_data = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {
                        "originalBody_question": question,
                        "originalBody_top_k": top_k
                    }
                },
                "id": 2
            }

            response = self.session.post(self.mcp_url, json=search_data, headers=self.headers)
            logger.info(f"MCP搜索状态: {response.status_code}")

            if response.status_code == 200:
                return self._parse_search_response(response)
            else:
                return {
                    "error": f"MCP调用失败，状态码: {response.status_code}",
                    "response": response.text[:200]
                }

        except Exception as e:
            logger.error(f"MCP搜索异常: {e}")
            return {
                "error": str(e),
                "status": "exception"
            }

    async def search_async(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        执行搜索（异步版本）

        Args:
            question: 搜索问题
            top_k: 返回结果数量

        Returns:
            Dict: 搜索结果
        """
        if not HTTPX_AVAILABLE:
            return {
                "error": "httpx 未安装，无法使用异步功能",
                "status": "httpx_not_available"
            }

        if not self.is_initialized:
            if not await self.initialize_async():
                return {
                    "error": "MCP客户端未初始化",
                    "status": "not_initialized"
                }

        try:
            search_data = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {
                        "originalBody_question": question,
                        "originalBody_top_k": top_k
                    }
                },
                "id": 2
            }

            response = await self.session.post(self.mcp_url, json=search_data, headers=self.headers)
            logger.info(f"MCP异步搜索状态: {response.status_code}")

            if response.status_code == 200:
                return self._parse_search_response(response)
            else:
                return {
                    "error": f"MCP调用失败，状态码: {response.status_code}",
                    "response": response.text[:200]
                }

        except Exception as e:
            logger.error(f"MCP异步搜索异常: {e}")
            return {
                "error": str(e),
                "status": "exception"
            }

    def _parse_search_response(self, response) -> Dict[str, Any]:
        """解析搜索响应"""
        try:
            response_text = response.text

            # 尝试从 SSE 格式提取 JSON
            if "data: " in response_text:
                lines = response_text.strip().split('\n')
                for line in lines:
                    if line.startswith('data: '):
                        try:
                            json_data = json.loads(line[6:])
                            if "result" in json_data:
                                return json_data["result"]
                        except json.JSONDecodeError:
                            continue

            # 尝试直接解析 JSON
            try:
                result = response.json()
                if "result" in result:
                    return result["result"]
            except json.JSONDecodeError:
                pass

            return {
                "error": "无法解析MCP响应",
                "raw_response": response_text[:200]
            }

        except Exception as e:
            logger.error(f"解析搜索响应时出错: {e}")
            return {
                "error": f"解析响应时出错: {str(e)}",
                "raw_response": response.text[:200] if hasattr(response, 'text') else str(response)
            }

    def list_tools(self) -> Dict[str, Any]:
        """
        列出可用工具

        Returns:
            Dict: 工具列表
        """
        if not self.is_initialized:
            if not self.initialize():
                return {
                    "error": "MCP客户端未初始化",
                    "status": "not_initialized"
                }

        try:
            list_tools_data = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 3
            }

            response = self.session.post(self.mcp_url, json=list_tools_data, headers=self.headers)
            logger.info(f"工具列表状态: {response.status_code}")

            if response.status_code == 200:
                return self._parse_search_response(response)
            else:
                return {
                    "error": f"获取工具列表失败，状态码: {response.status_code}",
                    "response": response.text[:200]
                }

        except Exception as e:
            logger.error(f"获取工具列表异常: {e}")
            return {
                "error": str(e),
                "status": "exception"
            }

    def close(self):
        """关闭客户端连接"""
        if self.session:
            # 检查是否是 httpx.AsyncClient
            if hasattr(self.session, 'aclose'):
                logger.warning("检测到异步客户端，请使用 close_async() 方法")
                # 尝试创建事件循环来关闭
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        logger.warning("事件循环正在运行，跳过同步关闭")
                    else:
                        loop.run_until_complete(self.session.aclose())
                except Exception as e:
                    logger.warning(f"异步关闭失败: {e}")
            elif hasattr(self.session, 'close'):
                self.session.close()
            self.session = None
            self.is_initialized = False

    async def close_async(self):
        """关闭客户端连接（异步版本）"""
        if self.session:
            if hasattr(self.session, 'aclose'):
                await self.session.aclose()
            elif hasattr(self.session, 'close'):
                if asyncio.iscoroutinefunction(self.session.close):
                    await self.session.close()
                else:
                    self.session.close()
            self.session = None
            self.is_initialized = False


def create_mcp_client(mcp_url: Optional[str] = None,
                     client_name: str = "mcp-client",
                     client_version: str = "1.0.0") -> MCPClient:
    """
    创建 MCP 客户端实例

    Args:
        mcp_url: MCP 服务地址
        client_name: 客户端名称
        client_version: 客户端版本

    Returns:
        MCPClient: MCP 客户端实例
    """
    if mcp_url is None:
        mcp_url = "http://192.168.244.189:5678/mcp/d73b2d78-7565-4e75-9943-665d5b4bdb18"

    return MCPClient(mcp_url=mcp_url, client_name=client_name, client_version=client_version)


# 便捷函数
def mcp_search(question: str, top_k: int = 5, mcp_url: Optional[str] = None) -> str:
    """
    便捷的 MCP 搜索函数

    Args:
        question: 搜索问题
        top_k: 返回结果数量
        mcp_url: MCP 服务地址

    Returns:
        str: JSON 格式的搜索结果
    """
    client = create_mcp_client(mcp_url=mcp_url)
    try:
        result = client.search(question, top_k)
        return json.dumps(result, ensure_ascii=False)
    finally:
        client.close()


async def mcp_search_async(question: str, top_k: int = 5, mcp_url: Optional[str] = None) -> str:
    """
    便捷的异步 MCP 搜索函数

    Args:
        question: 搜索问题
        top_k: 返回结果数量
        mcp_url: MCP 服务地址

    Returns:
        str: JSON 格式的搜索结果
    """
    if not HTTPX_AVAILABLE:
        return json.dumps({
            "error": "httpx 未安装，无法使用异步功能",
            "status": "httpx_not_available"
        }, ensure_ascii=False)

    client = create_mcp_client(mcp_url=mcp_url)
    try:
        result = await client.search_async(question, top_k)
        return json.dumps(result, ensure_ascii=False)
    finally:
        await client.close_async()