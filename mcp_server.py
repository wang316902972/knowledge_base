"""
FAISS Vector Database MCP Server
基于 Model Context Protocol 的向量搜索服务
"""
import asyncio
import json
import logging
from typing import Any, Sequence
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
from pydantic import BaseModel, Field, AnyUrl

from config import get_config, Config
from faiss_server_optimized import FaissVectorDB

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    """列出所有可用的工具"""
    return [
        Tool(
            name="search_knowledge",
            description="在向量数据库中搜索相关知识。支持语义搜索和优化搜索模式。",
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
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="add_document",
            description="将文档添加到向量数据库。文档会自动分块并生成向量索引。",
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
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="delete_document",
            description="从向量数据库中删除指定文档。需要提供与添加时相同的内容和分块参数。",
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
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="batch_add_texts",
            description="批量添加多个文本到向量数据库。适用于一次性添加多个独立的文本片段。",
            inputSchema={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "description": "要添加的文本列表",
                        "items": {
                            "type": "string"
                        },
                        "maxItems": 100
                    }
                },
                "required": ["texts"]
            }
        ),
        Tool(
            name="get_stats",
            description="获取向量数据库的统计信息，包括向量数量、索引类型、优化状态等。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="enable_optimization",
            description="启用高级搜索优化功能，包括语义分块、多样性重排序等。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_recommendations",
            description="获取针对当前数据量和索引类型的搜索优化建议。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="save_index",
            description="手动保存当前的向量索引到磁盘。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """处理工具调用"""
    global vector_db
    
    # 确保向量数据库已初始化
    if vector_db is None:
        logger.info("初始化向量数据库...")
        vector_db = FaissVectorDB(config)
    
    try:
        if name == "search_knowledge":
            # 搜索知识
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            use_optimization = arguments.get("use_optimization", True)
            
            if not query:
                raise ValueError("query parameter is required")
            
            if vector_db.index.ntotal == 0:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "relevant_chunks": [],
                        "message": "知识库为空，请先添加文档",
                        "total_found": 0
                    }, ensure_ascii=False, indent=2)
                )]
            
            results = vector_db.search(query, top_k, use_optimization)
            
            # 添加搜索方法信息
            search_method = "optimized" if use_optimization and hasattr(vector_db, 'advanced_search_index') else "traditional"
            
            response = {
                "relevant_chunks": [result["text"] for result in results],
                "detailed_results": results,
                "query": query,
                "total_found": len(results),
                "search_method": search_method,
                "optimization_enabled": use_optimization
            }
            
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
        
        elif name == "batch_add_texts":
            # 批量添加文本
            texts = arguments.get("texts", [])
            
            if not texts:
                raise ValueError("texts parameter is required")
            
            if len(texts) > 100:
                raise ValueError("Maximum 100 texts allowed per batch")
            
            ids = vector_db.add_texts(texts)
            
            # 自动保存
            if config.AUTO_SAVE:
                vector_db.save()
            
            response = {
                "message": f"批量添加成功，新增 {len(texts)} 个文本",
                "total_vectors": vector_db.index.ntotal,
                "added_count": len(ids)
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(response, ensure_ascii=False, indent=2)
            )]
        
        elif name == "get_stats":
            # 获取统计信息
            stats = vector_db.get_stats()
            
            return [TextContent(
                type="text",
                text=json.dumps(stats, ensure_ascii=False, indent=2)
            )]
        
        elif name == "enable_optimization":
            # 启用搜索优化
            success = vector_db.enable_search_optimization()
            
            if success:
                response = {
                    "message": "搜索优化已成功启用",
                    "features": [
                        "语义感知文本分块",
                        "动态搜索参数调整",
                        "多样性重排序(MMR)",
                        "搜索质量评估",
                        "相关性阈值过滤"
                    ]
                }
            else:
                response = {
                    "error": "启用搜索优化失败"
                }
            
            return [TextContent(
                type="text",
                text=json.dumps(response, ensure_ascii=False, indent=2)
            )]
        
        elif name == "get_recommendations":
            # 获取搜索建议
            recommendations = vector_db.get_search_recommendations()
            
            return [TextContent(
                type="text",
                text=json.dumps(recommendations, ensure_ascii=False, indent=2)
            )]
        
        elif name == "save_index":
            # 手动保存
            vector_db.save()
            
            response = {
                "message": "索引保存成功",
                "total_vectors": vector_db.index.ntotal
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(response, ensure_ascii=False, indent=2)
            )]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": str(e),
                "tool": name
            }, ensure_ascii=False)
        )]


async def main():
    """运行 MCP 服务器"""
    logger.info("启动 FAISS Vector Database MCP Server...")
    logger.info(f"业务ID: {config.BUSINESS_ID}")
    logger.info(f"索引类型: {config.INDEX_TYPE}")
    logger.info(f"模型: {config.MODEL_NAME}")
    
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
