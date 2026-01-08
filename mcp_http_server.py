"""
FAISS Vector Database MCP HTTP Server
基于 HTTP 的 Model Context Protocol 向量搜索服务
支持通过 HTTP REST API 访问 MCP 功能
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

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


# MCP 协议请求/响应模型
class MCPTool(BaseModel):
    """MCP 工具定义"""
    name: str
    description: str
    inputSchema: Dict[str, Any]


class MCPToolsListResponse(BaseModel):
    """工具列表响应"""
    tools: List[MCPTool]


class MCPCallToolRequest(BaseModel):
    """调用工具请求"""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class MCPContent(BaseModel):
    """MCP 内容"""
    type: str
    text: str


class MCPCallToolResponse(BaseModel):
    """调用工具响应"""
    content: List[MCPContent]
    isError: bool = False


class MCPInitializeRequest(BaseModel):
    """初始化请求"""
    protocolVersion: str = "2024-11-05"
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    clientInfo: Dict[str, str] = Field(default_factory=dict)


class MCPInitializeResponse(BaseModel):
    """初始化响应"""
    protocolVersion: str = "2024-11-05"
    capabilities: Dict[str, Any]
    serverInfo: Dict[str, str]


# 创建 FastAPI 应用
app = FastAPI(
    title="FAISS Vector Database MCP HTTP Server",
    description="基于 HTTP 的 MCP 向量搜索服务",
    version="2.0.0"
)

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_mcp_tools() -> List[MCPTool]:
    """获取所有 MCP 工具定义"""
    return [
        MCPTool(
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
        MCPTool(
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
        MCPTool(
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
        MCPTool(
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
        MCPTool(
            name="get_stats",
            description="获取向量数据库的统计信息，包括向量数量、索引类型、优化状态等。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        MCPTool(
            name="enable_optimization",
            description="启用高级搜索优化功能，包括语义分块、多样性重排序等。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        MCPTool(
            name="get_recommendations",
            description="获取针对当前数据量和索引类型的搜索优化建议。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        MCPTool(
            name="save_index",
            description="手动保存当前的向量索引到磁盘。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


async def execute_tool(name: str, arguments: Dict[str, Any]) -> MCPCallToolResponse:
    """执行工具调用"""
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
                return MCPCallToolResponse(
                    content=[MCPContent(
                        type="text",
                        text=json.dumps({
                            "relevant_chunks": [],
                            "message": "知识库为空，请先添加文档",
                            "total_found": 0
                        }, ensure_ascii=False, indent=2)
                    )]
                )
            
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
            
            return MCPCallToolResponse(
                content=[MCPContent(
                    type="text",
                    text=json.dumps(response, ensure_ascii=False, indent=2)
                )]
            )
        
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
                return MCPCallToolResponse(
                    content=[MCPContent(
                        type="text",
                        text=json.dumps({
                            "error": "文档内容过短，无法生成有效分块"
                        }, ensure_ascii=False)
                    )],
                    isError=True
                )
            
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
            
            return MCPCallToolResponse(
                content=[MCPContent(
                    type="text",
                    text=json.dumps(response, ensure_ascii=False, indent=2)
                )]
            )
        
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
                return MCPCallToolResponse(
                    content=[MCPContent(
                        type="text",
                        text=json.dumps({
                            "error": "文档内容过短，无法生成有效分块"
                        }, ensure_ascii=False)
                    )],
                    isError=True
                )
            
            deleted_count = vector_db.delete_texts(chunks)
            
            # 自动保存
            if config.AUTO_SAVE and deleted_count > 0:
                vector_db.save()
            
            response = {
                "message": f"成功删除 {deleted_count} 个知识块",
                "total_vectors": vector_db.index.ntotal,
                "deleted_count": deleted_count
            }
            
            return MCPCallToolResponse(
                content=[MCPContent(
                    type="text",
                    text=json.dumps(response, ensure_ascii=False, indent=2)
                )]
            )
        
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
            
            return MCPCallToolResponse(
                content=[MCPContent(
                    type="text",
                    text=json.dumps(response, ensure_ascii=False, indent=2)
                )]
            )
        
        elif name == "get_stats":
            # 获取统计信息
            stats = vector_db.get_stats()
            
            return MCPCallToolResponse(
                content=[MCPContent(
                    type="text",
                    text=json.dumps(stats, ensure_ascii=False, indent=2)
                )]
            )
        
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
            
            return MCPCallToolResponse(
                content=[MCPContent(
                    type="text",
                    text=json.dumps(response, ensure_ascii=False, indent=2)
                )],
                isError=not success
            )
        
        elif name == "get_recommendations":
            # 获取搜索建议
            recommendations = vector_db.get_search_recommendations()
            
            return MCPCallToolResponse(
                content=[MCPContent(
                    type="text",
                    text=json.dumps(recommendations, ensure_ascii=False, indent=2)
                )]
            )
        
        elif name == "save_index":
            # 手动保存
            vector_db.save()
            
            response = {
                "message": "索引保存成功",
                "total_vectors": vector_db.index.ntotal
            }
            
            return MCPCallToolResponse(
                content=[MCPContent(
                    type="text",
                    text=json.dumps(response, ensure_ascii=False, indent=2)
                )]
            )
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return MCPCallToolResponse(
            content=[MCPContent(
                type="text",
                text=json.dumps({
                    "error": str(e),
                    "tool": name
                }, ensure_ascii=False)
            )],
            isError=True
        )


# MCP HTTP API 端点
@app.post("/mcp/initialize", response_model=MCPInitializeResponse)
async def mcp_initialize(request: MCPInitializeRequest):
    """MCP 初始化"""
    logger.info(f"MCP 初始化请求: {request.dict()}")
    
    return MCPInitializeResponse(
        protocolVersion="2024-11-05",
        capabilities={
            "tools": {},
            "resources": {}
        },
        serverInfo={
            "name": "faiss-vector-search",
            "version": "2.0.0"
        }
    )


@app.get("/mcp/tools", response_model=MCPToolsListResponse)
async def mcp_list_tools():
    """列出所有可用工具"""
    tools = get_mcp_tools()
    return MCPToolsListResponse(tools=tools)


@app.post("/mcp/tools/call", response_model=MCPCallToolResponse)
async def mcp_call_tool(request: MCPCallToolRequest):
    """调用工具"""
    logger.info(f"调用工具: {request.name}, 参数: {request.arguments}")
    
    response = await execute_tool(request.name, request.arguments)
    return response


# 兼容性端点 - 支持传统 REST API 风格
@app.post("/search")
async def search_endpoint(query: str, top_k: int = 5, use_optimization: bool = True):
    """搜索端点（REST API 风格）"""
    response = await execute_tool("search_knowledge", {
        "query": query,
        "top_k": top_k,
        "use_optimization": use_optimization
    })
    
    if response.isError:
        raise HTTPException(status_code=500, detail=json.loads(response.content[0].text))
    
    return json.loads(response.content[0].text)


@app.post("/add")
async def add_endpoint(content: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """添加文档端点（REST API 风格）"""
    response = await execute_tool("add_document", {
        "content": content,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    })
    
    if response.isError:
        raise HTTPException(status_code=500, detail=json.loads(response.content[0].text))
    
    return json.loads(response.content[0].text)


@app.get("/stats")
async def stats_endpoint():
    """统计信息端点（REST API 风格）"""
    response = await execute_tool("get_stats", {})
    return json.loads(response.content[0].text)


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy" if vector_db else "initializing",
        "mcp_protocol_version": "2024-11-05",
        "server_version": "2.0.0"
    }


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global vector_db
    logger.info("正在初始化 FAISS Vector Database MCP HTTP Server...")
    logger.info(f"业务ID: {config.BUSINESS_ID}")
    logger.info(f"索引类型: {config.INDEX_TYPE}")
    logger.info(f"模型: {config.MODEL_NAME}")
    vector_db = FaissVectorDB(config)
    logger.info("✅ 向量数据库初始化完成")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时保存"""
    global vector_db
    if vector_db and vector_db.dirty:
        logger.info("正在保存未保存的更改...")
        vector_db.save()
    logger.info("✅ 向量数据库已关闭")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower()
    )
