"""
FAISS Vector Database MCP Server

基于 Model Context Protocol 的向量搜索服务。
提供语义搜索、文档管理等功能，支持业务类型隔离。

Example:
    >>> python mcp_server.py
    >>> # 通过 stdio 与 MCP 客户端通信
"""
import asyncio
import json
from typing import Any, Sequence, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from pydantic import BaseModel, Field

from config import get_config, Config
from faiss_server_optimized import FaissVectorDB
from logger import setup_logger

# 初始化日志
logger = setup_logger(__name__)

# 初始化配置和向量数据库
config = get_config()
vector_db = None


class SearchRequest(BaseModel):
    """搜索请求模型"""
    query: str = Field(..., description="搜索查询文本")
    top_k: int = Field(default=5, ge=1, le=50, description="返回结果数量")
    use_optimization: bool = Field(default=True, description="是否使用搜索优化")


class AddDocumentRequest(BaseModel):
    """添加文档请求模型"""
    content: str = Field(..., description="文档内容")
    chunk_size: int = Field(default=500, ge=50, le=2000, description="分块大小")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="分块重叠")


class DeleteDocumentRequest(BaseModel):
    """删除文档请求模型"""
    content: str = Field(..., description="要删除的文档内容")
    chunk_size: int = Field(default=500, ge=50, le=2000, description="分块大小")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="分块重叠")


class BatchAddRequest(BaseModel):
    """批量添加请求模型"""
    texts: list[str] = Field(..., description="批量文本列表")


# 创建 MCP Server
mcp_server = Server("faiss-vector-search")


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """列出所有可用的工具

    所有工具都支持可选的 businesstype 参数，用于指定业务类型。
    默认使用环境变量 BUINESSTYPE 或 'default' 作为业务类型。
    """
    default_bt = config.DEFAULT_BUSINESSTYPE

    return [
        Tool(
            name="search_knowledge",
            description=f"""在向量数据库中搜索相关知识。支持语义搜索和优化搜索模式。

默认业务类型: {default_bt}

功能特性:
- 语义向量搜索
- 可配置返回结果数量 (top_k)
- 搜索优化选项（多样性重排序、相关性过滤）
- 支持业务类型隔离的独立索引""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询文本，支持自然语言问题"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的最相关结果数量（1-50）",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "use_optimization": {
                        "type": "boolean",
                        "description": "是否使用优化搜索（包括多样性重排序、相关性过滤等）",
                        "default": True
                    },
                    "businesstype": {
                        "type": "string",
                        "description": "业务类型标识符（可选，默认使用环境变量配置）"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="add_document",
            description=f"""将文档添加到向量数据库。文档会自动分块并生成向量索引。

默认业务类型: {default_bt}

功能特性:
- 自动文本分块（可配置大小和重叠）
- 向量化和索引生成
- 支持业务类型隔离的独立索引
- 自动保存选项（可配置）""",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要添加的文档内容"
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "文本分块大小（字符数）",
                        "default": 500,
                        "minimum": 50,
                        "maximum": 2000
                    },
                    "chunk_overlap": {
                        "type": "integer",
                        "description": "分块之间的重叠字符数",
                        "default": 50,
                        "minimum": 0,
                        "maximum": 500
                    },
                    "businesstype": {
                        "type": "string",
                        "description": "业务类型标识符（可选，默认使用环境变量配置）"
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="delete_document",
            description=f"""从向量数据库中删除指定文档。需要提供与添加时相同的内容和分块参数。

默认业务类型: {default_bt}

功能特性:
- 精确匹配文档删除
- 需要提供与添加时相同的分块参数
- 支持业务类型隔离的独立索引""",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要删除的文档内容"
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "文本分块大小（需与添加时一致）",
                        "default": 500,
                        "minimum": 50,
                        "maximum": 2000
                    },
                    "chunk_overlap": {
                        "type": "integer",
                        "description": "分块重叠（需与添加时一致）",
                        "default": 50,
                        "minimum": 0,
                        "maximum": 500
                    },
                    "businesstype": {
                        "type": "string",
                        "description": "业务类型标识符（可选，默认使用环境变量配置）"
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="update_document",
            description=f"""批量更新知识库内容。旧内容会软删除，新内容会追加为稳定 vector_id，适合百万级索引。

默认业务类型: {default_bt}""",
            inputSchema={
                "type": "object",
                "properties": {
                    "updates": {
                        "type": "array",
                        "description": "更新列表，每项包含 old_text 和 new_text",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": {"type": "string"},
                                "new_text": {"type": "string"}
                            },
                            "required": ["old_text", "new_text"]
                        },
                        "maxItems": 100
                    },
                    "businesstype": {
                        "type": "string",
                        "description": "业务类型标识符（可选，默认使用环境变量配置）"
                    }
                },
                "required": ["updates"]
            }
        ),
        Tool(
            name="compact_index",
            description=f"""压缩索引，物理移除软删除向量。建议 deleted_ratio 超过 0.3 时执行。

默认业务类型: {default_bt}""",
            inputSchema={
                "type": "object",
                "properties": {
                    "businesstype": {
                        "type": "string",
                        "description": "业务类型标识符（可选，默认使用环境变量配置）"
                    }
                }
            }
        ),
        Tool(
            name="get_stats",
            description=f"""获取向量数据库的统计信息，包括向量数量、索引类型、优化状态等。

默认业务类型: {default_bt}

返回信息:
- 向量总数
- 索引类型和配置
- 模型信息
- 优化状态
- 业务类型路径""",
            inputSchema={
                "type": "object",
                "properties": {
                    "businesstype": {
                        "type": "string",
                        "description": "业务类型标识符（可选，默认使用环境变量配置）"
                    }
                }
            }
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """处理工具调用"""
    global vector_db

    # 提取 business type 用于日志记录
    businesstype = arguments.get("businesstype", config.DEFAULT_BUSINESSTYPE)

    # 确保向量数据库已初始化
    if vector_db is None:
        logger.info(f"[{businesstype}] 初始化向量数据库...")
        vector_db = FaissVectorDB(config)
    
    try:
        if name == "search_knowledge":
            # 搜索知识
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            use_optimization = arguments.get("use_optimization", True)

            if not query:
                raise ValueError("query parameter is required")

            # 记录搜索日志
            logger.info(f"[{businesstype}] 🔍 搜索请求 | Query: {query[:100]}{'...' if len(query) > 100 else ''} | top_k: {top_k} | optimization: {use_optimization}")
            
            if vector_db.index.ntotal == 0:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "relevant_chunks": [],
                        "message": "知识库为空，请先添加文档",
                        "total_found": 0
                    }, ensure_ascii=False, indent=2)
                )]
            
            results = vector_db.search(query, top_k, use_optimization, use_enhanced=True)

            # 记录搜索结果日志
            logger.info(f"[{businesstype}] ✅ 搜索完成 | 找到 {len(results)} 个结果")

            # 添加搜索方法信息
            search_method = results[0].get("search_method", "unknown") if results else "unknown"

            # 提取质量指标（如果有）
            quality_metrics = None
            if results and "quality_metrics" in results[0]:
                quality_metrics = results[0]["quality_metrics"]
                logger.info(f"[{businesstype}] 📊 质量指标 | avg_relevance: {quality_metrics.get('avg_relevance_score', 0):.3f} | "
                           f"diversity: {quality_metrics.get('diversity_score', 0):.3f} | "
                           f"coverage: {quality_metrics.get('coverage_ratio', 0):.3f}")

            response = {
                "relevant_chunks": [result["text"] for result in results],
                "detailed_results": results,
                "query": query,
                "total_found": len(results),
                "search_method": search_method,
                "optimization_enabled": use_optimization,
                "enhanced_search_enabled": True
            }

            # 添加质量指标到响应
            if quality_metrics:
                response["quality_metrics"] = quality_metrics
            
            return [TextContent(
                type="text",
                text=json.dumps(response, ensure_ascii=False, indent=2)
            )]
        
        elif name == "add_document":
            # 添加文档
            content = arguments.get("content")
            chunk_size = arguments.get("chunk_size", 500)
            chunk_overlap = arguments.get("chunk_overlap", 50)

            if not content:
                raise ValueError("content parameter is required")

            # 记录添加文档日志
            content_preview = content[:150].replace('\n', ' ') + '...' if len(content) > 150 else content.replace('\n', ' ')
            logger.info(f"[{businesstype}] 📝 添加文档 | Content: {content_preview} | chunk_size: {chunk_size} | chunk_overlap: {chunk_overlap}")

            # 文本分块
            chunks = vector_db._generate_chunks(content, chunk_size, chunk_overlap)
            
            if not chunks:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "文档内容过短，无法生成有效分块"
                    }, ensure_ascii=False)
                )]
            
            # 添加到索引
            ids = vector_db.add_texts(chunks)

            # 记录添加成功日志
            logger.info(f"[{businesstype}] ✅ 文档添加成功 | 新增 {len(chunks)} 个知识块 | 总向量数: {vector_db.index.ntotal}")

            # 自动保存
            if config.AUTO_SAVE:
                vector_db.save()
            
            response = {
                "message": f"文档处理成功，新增 {len(chunks)} 个知识块",
                "total_vectors": vector_db.index.ntotal,
                "chunks_added": len(chunks),
                "chunk_ids": ids[:5] if len(ids) > 5 else ids
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(response, ensure_ascii=False, indent=2)
            )]
        
        elif name == "delete_document":
            # 删除文档
            content = arguments.get("content")
            chunk_size = arguments.get("chunk_size", 500)
            chunk_overlap = arguments.get("chunk_overlap", 50)

            if not content:
                raise ValueError("content parameter is required")

            # 记录删除文档日志
            content_preview = content[:150].replace('\n', ' ') + '...' if len(content) > 150 else content.replace('\n', ' ')
            logger.info(f"[{businesstype}] 🗑️ 删除文档 | Content: {content_preview}")

            # 重新生成相同的分块来精确匹配
            chunks = vector_db._generate_chunks(content, chunk_size, chunk_overlap)
            
            if not chunks:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "文档内容过短，无法生成有效分块"
                    }, ensure_ascii=False)
                )]
            
            deleted_count = vector_db.delete_texts(chunks)
            
            # 自动保存
            if config.AUTO_SAVE and deleted_count > 0:
                vector_db.save()
            
            response = {
                "message": f"成功删除 {deleted_count} 个知识块",
                "total_vectors": vector_db.index.ntotal,
                "deleted_count": deleted_count
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(response, ensure_ascii=False, indent=2)
            )]

        elif name == "update_document":
            updates = arguments.get("updates", [])
            if not updates:
                raise ValueError("updates parameter is required")
            if len(updates) > 100:
                raise ValueError("Maximum 100 updates allowed per batch")

            result = vector_db.update_texts(updates)

            if config.AUTO_SAVE and result["success_count"] > 0:
                vector_db.save()

            response = {
                "message": f"更新完成: 成功 {result['success_count']}, 新增 {result['inserted_count']}, 更新 {result['updated_count']}, 失败 {result['failed_count']}",
                "total_vectors": vector_db.index.ntotal,
                **result,
                "lifecycle_metrics": vector_db.metadata_store.get_metrics() if vector_db.metadata_store else {}
            }

            return [TextContent(
                type="text",
                text=json.dumps(response, ensure_ascii=False, indent=2)
            )]

        elif name == "compact_index":
            response = vector_db.compact_index()

            if config.AUTO_SAVE:
                vector_db.save()

            return [TextContent(
                type="text",
                text=json.dumps(response, ensure_ascii=False, indent=2)
            )]

        elif name == "get_stats":
            # 获取统计信息
            logger.info(f"[{businesstype}] 📊 获取统计信息")
            stats = vector_db.get_stats()
            
            return [TextContent(
                type="text",
                text=json.dumps(stats, ensure_ascii=False, indent=2)
            )]

        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        logger.error(f"[{businesstype}] ❌ 工具执行错误 | Tool: {name} | Error: {e}")
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": str(e),
                "tool": name
            }, ensure_ascii=False)
        )]


async def main():
    """运行 MCP 服务器"""
    logger.info("=" * 60)
    logger.info("🚀 启动 FAISS Vector Database MCP Server...")
    logger.info(f"📋 Business Type: {config.DEFAULT_BUSINESSTYPE}")
    logger.info(f"📊 索引类型: {config.INDEX_TYPE}")
    logger.info(f"🤖 模型: {config.MODEL_NAME}")
    logger.info(f"📁 数据目录: {config.DATA_DIR}")
    logger.info("=" * 60)
    
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
