import faiss
import numpy as np
import json
import logging
import threading
import uuid
import asyncio
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from sentence_transformers import SentenceTransformer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置类
class Config:
    INDEX_FILE = "knowledge_base.index"
    METADATA_FILE = "knowledge_base.json"  # 改用JSON替代pickle
    EMBEDDING_DIM = 384
    MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
    INDEX_TYPE = "FlatIP"  # 可配置的索引类型
    MAX_CHUNK_SIZE = 2000
    MIN_CHUNK_SIZE = 50
    BATCH_SIZE = 32
    AUTO_SAVE = False  # 默认不自动保存

# 线程安全的向量数据库类
class FaissVectorDB:
    def __init__(self, config: Config):
        self.config = config
        self.lock = threading.RLock()  # 重入锁保证线程安全
        self.embedding_model = None
        self.index = None
        self.id_to_chunk: Dict[str, str] = {}
        self.chunk_to_id: Dict[str, str] = {}  # 反向索引用于精确删除
        self.dirty = False  # 标记是否有未保存的更改
        
        self._initialize()
    
    def _initialize(self):
        """初始化模型和索引"""
        try:
            logger.info(f"业务ID: {self.config.BUSINESS_ID}")
            logger.info(f"数据目录: {self.config.DATA_DIR}")
            logger.info(f"索引文件: {self.config.INDEX_FILE}")
            logger.info(f"元数据文件: {self.config.METADATA_FILE}")

            # 确保数据目录存在
            os.makedirs(self.config.DATA_DIR, exist_ok=True)

            logger.info(f"正在加载嵌入模型: {self.config.MODEL_NAME}")
            self.embedding_model = SentenceTransformer(self.config.MODEL_NAME)

            # 动态获取模型维度
            actual_dim = self.embedding_model.get_sentence_embedding_dimension()
            if actual_dim != self.config.EMBEDDING_DIM:
                logger.warning(f"模型实际维度 {actual_dim} 与配置维度 {self.config.EMBEDDING_DIM} 不匹配，使用实际维度")
                self.config.EMBEDDING_DIM = actual_dim

            self._create_or_load_index()
            logger.info(f"业务 {self.config.BUSINESS_ID} 向量数据库初始化完成")

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise
    
    def _create_index(self):
        """根据配置创建FAISS索引"""
        if self.config.INDEX_TYPE == "FlatIP":
            return faiss.IndexFlatIP(self.config.EMBEDDING_DIM)
        elif self.config.INDEX_TYPE == "FlatL2":
            return faiss.IndexFlatL2(self.config.EMBEDDING_DIM)
        elif self.config.INDEX_TYPE == "IVFFlat":
            # IVF索引需要训练
            nlist = 100  # 聚类数量
            quantizer = faiss.IndexFlatL2(self.config.EMBEDDING_DIM)
            index = faiss.IndexIVFFlat(quantizer, self.config.EMBEDDING_DIM, nlist)
            return index
        else:
            logger.warning(f"未知索引类型 {self.config.INDEX_TYPE}，使用默认 FlatIP")
            return faiss.IndexFlatIP(self.config.EMBEDDING_DIM)
    
    def _create_or_load_index(self):
        """创建或加载索引"""
        try:
            self.load()
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError) as e:
            logger.info(f"未找到现有索引或加载失败: {e}，创建新索引...")
            self.index = self._create_index()
            self.id_to_chunk = {}
            self.chunk_to_id = {}
    
    def _generate_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """改进的文本分块逻辑"""
        if chunk_size > self.config.MAX_CHUNK_SIZE:
            chunk_size = self.config.MAX_CHUNK_SIZE
        if chunk_size < self.config.MIN_CHUNK_SIZE:
            chunk_size = self.config.MIN_CHUNK_SIZE
        
        chunks = []
        step = max(1, chunk_size - chunk_overlap)
        
        for i in range(0, len(content), step):
            chunk = content[i:i + chunk_size].strip()
            if len(chunk) >= self.config.MIN_CHUNK_SIZE:
                chunks.append(chunk)
        
        return chunks
    
    def add_texts(self, texts: List[str]) -> List[str]:
        """批量添加文本，返回生成的ID列表"""
        if not texts:
            return []
        
        with self.lock:
            try:
                # 生成嵌入向量
                logger.info(f"正在生成 {len(texts)} 个文本的嵌入向量...")
                embeddings = self.embedding_model.encode(
                    texts, 
                    convert_to_numpy=True, 
                    show_progress_bar=False, 
                    normalize_embeddings=True,
                    batch_size=self.config.BATCH_SIZE
                )
                
                # 生成唯一ID
                ids = [str(uuid.uuid4()) for _ in texts]
                
                # 添加到索引
                self.index.add(embeddings)
                
                # 更新元数据
                start_faiss_id = self.index.ntotal - len(texts)
                for i, (text, unique_id) in enumerate(zip(texts, ids)):
                    faiss_id = start_faiss_id + i
                    self.id_to_chunk[str(faiss_id)] = text
                    self.chunk_to_id[text] = str(faiss_id)
                
                self.dirty = True
                logger.info(f"成功添加 {len(texts)} 个文本块，总向量数: {self.index.ntotal}")
                return ids
                
            except Exception as e:
                logger.error(f"添加文本失败: {e}")
                raise HTTPException(status_code=500, detail=f"添加文本失败: {str(e)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似文本"""
        with self.lock:
            try:
                if self.index.ntotal == 0:
                    return []
                
                # 生成查询向量
                query_embedding = self.embedding_model.encode(
                    [query], 
                    convert_to_numpy=True, 
                    show_progress_bar=False, 
                    normalize_embeddings=True
                )
                
                # 搜索
                distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
                
                results = []
                for dist, idx in zip(distances[0], indices[0]):
                    if idx != -1 and str(idx) in self.id_to_chunk:
                        results.append({
                            "text": self.id_to_chunk[str(idx)],
                            "score": float(dist),
                            "faiss_id": int(idx)
                        })
                
                logger.info(f"搜索完成，返回 {len(results)} 个结果")
                return results
                
            except Exception as e:
                logger.error(f"搜索失败: {e}")
                raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")
    
    def delete_texts(self, texts_to_delete: List[str]) -> int:
        """精确删除指定文本"""
        with self.lock:
            try:
                ids_to_remove = []
                for text in texts_to_delete:
                    if text in self.chunk_to_id:
                        faiss_id = self.chunk_to_id[text]
                        ids_to_remove.append(int(faiss_id))
                
                if not ids_to_remove:
                    logger.info("未找到要删除的文本")
                    return 0
                
                # 从FAISS索引删除
                self.index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
                
                # 从元数据删除
                for faiss_id in ids_to_remove:
                    faiss_id_str = str(faiss_id)
                    if faiss_id_str in self.id_to_chunk:
                        text = self.id_to_chunk[faiss_id_str]
                        del self.id_to_chunk[faiss_id_str]
                        if text in self.chunk_to_id:
                            del self.chunk_to_id[text]
                
                self.dirty = True
                logger.info(f"成功删除 {len(ids_to_remove)} 个文本块")
                return len(ids_to_remove)
                
            except Exception as e:
                logger.error(f"删除文本失败: {e}")
                raise HTTPException(status_code=500, detail=f"删除文本失败: {str(e)}")
    
    def save(self):
        """保存索引和元数据"""
        with self.lock:
            try:
                if not self.dirty:
                    logger.info("没有更改，跳过保存")
                    return
                
                # 原子性保存：先保存到临时文件
                temp_index_file = f"{self.config.INDEX_FILE}.tmp"
                temp_metadata_file = f"{self.config.METADATA_FILE}.tmp"
                
                # 保存FAISS索引
                faiss.write_index(self.index, temp_index_file)
                
                # 保存元数据为JSON（替代pickle）
                metadata = {
                    "id_to_chunk": self.id_to_chunk,
                    "chunk_to_id": self.chunk_to_id,
                    "embedding_dim": self.config.EMBEDDING_DIM,
                    "total_vectors": self.index.ntotal
                }
                
                with open(temp_metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                # 原子性重命名
                Path(temp_index_file).rename(self.config.INDEX_FILE)
                Path(temp_metadata_file).rename(self.config.METADATA_FILE)
                
                self.dirty = False
                logger.info("索引和元数据保存成功")
                
            except Exception as e:
                logger.error(f"保存失败: {e}")
                # 清理临时文件
                for temp_file in [temp_index_file, temp_metadata_file]:
                    try:
                        Path(temp_file).unlink(missing_ok=True)
                    except:
                        pass
                raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")
    
    def load(self):
        """加载索引和元数据"""
        try:
            # 加载FAISS索引
            self.index = faiss.read_index(self.config.INDEX_FILE)
            
            # 加载元数据（JSON格式）
            with open(self.config.METADATA_FILE, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.id_to_chunk = metadata.get("id_to_chunk", {})
            self.chunk_to_id = metadata.get("chunk_to_id", {})
            
            logger.info(f"成功加载索引，包含 {self.index.ntotal} 个向量")
            
        except FileNotFoundError as e:
            logger.warning(f"索引文件不存在: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"元数据文件格式错误: {e}")
            raise
        except Exception as e:
            logger.error(f"加载失败: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        with self.lock:
            return {
                "total_vectors": self.index.ntotal if self.index else 0,
                "embedding_dim": self.config.EMBEDDING_DIM,
                "index_type": self.config.INDEX_TYPE,
                "model_name": self.config.MODEL_NAME,
                "has_unsaved_changes": self.dirty
            }

# 全局向量数据库实例
config = Config()
vector_db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global vector_db
    try:
        # 启动时初始化
        logger.info("正在初始化向量数据库...")
        vector_db = FaissVectorDB(config)
        yield
    finally:
        # 关闭时保存
        if vector_db and vector_db.dirty:
            logger.info("正在保存未保存的更改...")
            vector_db.save()
        logger.info("向量数据库已关闭")

# 创建FastAPI应用
app = FastAPI(
    title="FAISS向量数据库API",
    description="优化的FAISS向量数据库服务",
    version="2.0.0",
    lifespan=lifespan
)

# 请求模型
class Document(BaseModel):
    content: str = Field(..., description="文档内容", max_length=50000)
    chunk_size: int = Field(default=500, description="分块大小", ge=50, le=2000)
    chunk_overlap: int = Field(default=50, description="分块重叠", ge=0, le=500)
    
    @validator('chunk_overlap')
    def validate_overlap(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v

class Query(BaseModel):
    question: str = Field(..., description="查询问题", max_length=1000)
    top_k: int = Field(default=3, description="返回结果数量", ge=1, le=50)

class BatchTexts(BaseModel):
    texts: List[str] = Field(..., description="批量文本列表", max_items=100)

# API端点
@app.post("/add", summary="添加文档")
async def add_document(doc: Document, background_tasks: BackgroundTasks):
    """添加文档到知识库"""
    try:
        if not vector_db:
            raise HTTPException(status_code=503, detail="向量数据库未初始化")
        
        # 文本分块
        chunks = vector_db._generate_chunks(doc.content, doc.chunk_size, doc.chunk_overlap)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="文档内容过短，无法生成有效分块")
        
        # 添加到索引
        ids = vector_db.add_texts(chunks)
        
        # 后台保存（如果启用自动保存）
        if config.AUTO_SAVE:
            background_tasks.add_task(vector_db.save)
        
        return {
            "message": f"文档处理成功，新增 {len(chunks)} 个知识块",
            "total_vectors": vector_db.index.ntotal,
            "chunk_ids": ids[:5] if len(ids) > 5 else ids  # 只返回前5个ID
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加文档失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/batch_add", summary="批量添加文本")
async def batch_add_texts(batch: BatchTexts, background_tasks: BackgroundTasks):
    """批量添加文本"""
    try:
        if not vector_db:
            raise HTTPException(status_code=503, detail="向量数据库未初始化")
        
        ids = vector_db.add_texts(batch.texts)
        
        if config.AUTO_SAVE:
            background_tasks.add_task(vector_db.save)
        
        return {
            "message": f"批量添加成功，新增 {len(batch.texts)} 个文本",
            "total_vectors": vector_db.index.ntotal,
            "added_count": len(ids)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量添加失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.delete("/delete", summary="删除文档")
async def delete_document(doc: Document, background_tasks: BackgroundTasks):
    """精确删除文档"""
    try:
        if not vector_db:
            raise HTTPException(status_code=503, detail="向量数据库未初始化")
        
        # 重新生成相同的分块来精确匹配
        chunks = vector_db._generate_chunks(doc.content, doc.chunk_size, doc.chunk_overlap)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="文档内容过短，无法生成有效分块")
        
        deleted_count = vector_db.delete_texts(chunks)
        
        if config.AUTO_SAVE and deleted_count > 0:
            background_tasks.add_task(vector_db.save)
        
        return {
            "message": f"成功删除 {deleted_count} 个知识块",
            "total_vectors": vector_db.index.ntotal,
            "deleted_count": deleted_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/search", summary="搜索知识")
async def search_knowledge(query: Query):
    """搜索相关知识"""
    try:
        if not vector_db:
            raise HTTPException(status_code=503, detail="向量数据库未初始化")
        
        if vector_db.index.ntotal == 0:
            return {"relevant_chunks": [], "message": "知识库为空，请先添加文档"}
        
        results = vector_db.search(query.question, query.top_k)
        
        return {
            "relevant_chunks": [result["text"] for result in results],
            "detailed_results": results,
            "query": query.question,
            "total_found": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/save", summary="手动保存")
async def manual_save():
    """手动保存索引和元数据"""
    try:
        if not vector_db:
            raise HTTPException(status_code=503, detail="向量数据库未初始化")
        
        vector_db.save()
        return {"message": "保存成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"保存失败: {e}")
        raise HTTPException(status_code=500, detail="保存失败")

@app.get("/stats", summary="获取统计信息")
async def get_stats():
    """获取数据库统计信息"""
    try:
        if not vector_db:
            raise HTTPException(status_code=503, detail="向量数据库未初始化")
        
        return vector_db.get_stats()
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/health", summary="健康检查")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy" if vector_db else "initializing",
        "timestamp": "2024-01-01T00:00:00Z"  # 实际应用中使用当前时间
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
