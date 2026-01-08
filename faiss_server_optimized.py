import faiss
import numpy as np
import json
import logging
import threading
import uuid
import asyncio
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Sequence, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer
from config import get_config, Config
from search_optimization import AdvancedSearchIndex, QualityMetrics

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
            logger.info(f"环境: {self.config.ENVIRONMENT}")
            logger.info(f"业务ID: {self.config.BUSINESS_ID}")
            logger.info(f"数据目录: {self.config.DATA_DIR}")
            logger.info(f"索引文件: {self.config.INDEX_FILE}")
            logger.info(f"元数据文件: {self.config.METADATA_FILE}")
            logger.info(f"是否使用GPU: {self.config.USE_GPU}")
            logger.info(f"模型名称: {self.config.MODEL_NAME}")

            # 确保数据目录存在
            os.makedirs(self.config.DATA_DIR, exist_ok=True)

            # 优先尝试使用本地缓存的模型
            try:
                logger.info("尝试加载本地缓存的模型...")
                self.embedding_model = SentenceTransformer(
                    self.config.MODEL_NAME,
                    local_files_only=True
                )
                logger.info("成功加载本地缓存的模型")
            except Exception as e:
                logger.warning(f"本地模型加载失败，尝试从网络下载: {e}")
                # 读取 Hugging Face token（如果存在）
                token = None
                token_path = os.path.expanduser("~/.huggingface/token")
                if os.path.exists(token_path):
                    with open(token_path, 'r') as f:
                        token = f.read().strip()
                    logger.info("使用 Hugging Face token 进行认证")

                # 加载模型，如果有 token 则传递
                if token:
                    self.embedding_model = SentenceTransformer(self.config.MODEL_NAME, token=token)
                else:
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
            logger.info(f"创建 FlatIP 索引，维度: {self.config.EMBEDDING_DIM}")
            return faiss.IndexFlatIP(self.config.EMBEDDING_DIM)
        elif self.config.INDEX_TYPE == "FlatL2":
            logger.info(f"创建 FlatL2 索引，维度: {self.config.EMBEDDING_DIM}")
            return faiss.IndexFlatL2(self.config.EMBEDDING_DIM)
        elif self.config.INDEX_TYPE == "IVFFlat":
            # IVF索引需要训练
            logger.info(f"创建 IVFFlat 索引，维度: {self.config.EMBEDDING_DIM}, nlist: {self.config.NLIST}")
            quantizer = faiss.IndexFlatL2(self.config.EMBEDDING_DIM)
            index = faiss.IndexIVFFlat(quantizer, self.config.EMBEDDING_DIM, self.config.NLIST)
            return index
        elif self.config.INDEX_TYPE == "HNSW":
            # HNSW索引
            logger.info(f"创建 HNSW 索引，维度: {self.config.EMBEDDING_DIM}, M: {self.config.M}")
            index = faiss.IndexHNSWFlat(self.config.EMBEDDING_DIM, self.config.M)
            index.hnsw.efConstruction = self.config.EF_CONSTRUCTION
            index.hnsw.efSearch = self.config.EF_SEARCH
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

            # 对于IVF索引，尝试创建初始训练数据
            if self.config.INDEX_TYPE == "IVFFlat":
                self._initialize_ivf_index_if_needed()

    def _initialize_ivf_index_if_needed(self):
        """为IVF索引创建初始训练数据"""
        try:
            # 生成一些示例训练向量
            min_train_size = max(self.config.NLIST * 39, 1000)  # 至少1000个向量用于训练
            logger.info(f"IVF索引需要至少 {min_train_size} 个训练向量...")

            # 生成随机但逼真的训练向量
            sample_texts = [
                "这是一个示例文本，用于训练向量索引。",
                "机器学习是人工智能的一个重要分支。",
                "向量数据库可以高效地进行相似性搜索。",
                "深度学习模型需要大量数据进行训练。",
                "自然语言处理技术越来越成熟。",
                "计算机视觉在医疗领域有广泛应用。",
                "推荐系统使用用户行为数据进行个性化推荐。",
                "数据挖掘可以发现数据中的隐藏模式。",
                "云计算提供了弹性的计算资源。",
                "物联网设备连接着物理世界和数字世界。",
            ] * (min_train_size // 10 + 1)  # 重复以获得足够的文本

            sample_texts = sample_texts[:min_train_size]
            logger.info(f"生成 {len(sample_texts)} 个样本向量用于IVF索引训练...")

            # 生成嵌入向量
            train_embeddings = self.embedding_model.encode(
                sample_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=self.config.BATCH_SIZE
            )

            # 训练索引
            self.index.train(train_embeddings)
            logger.info(f"IVF索引训练完成，使用了 {len(train_embeddings)} 个向量")

        except Exception as e:
            logger.warning(f"IVF索引初始化训练失败: {e}")
            logger.info("切换到FlatIP索引以避免训练问题...")
            # 回退到简单的FlatIP索引
            self.index = faiss.IndexFlatIP(self.config.EMBEDDING_DIM)

    def _generate_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """改进的文本分块逻辑 - 支持传统和语义分块"""
        if chunk_size > self.config.MAX_CHUNK_SIZE:
            chunk_size = self.config.MAX_CHUNK_SIZE
        if chunk_size < self.config.MIN_CHUNK_SIZE:
            chunk_size = self.config.MIN_CHUNK_SIZE

        # 如果配置了语义分块，使用优化的分块器
        if hasattr(self, 'use_semantic_chunking') and self.use_semantic_chunking:
            return self.semantic_chunker.chunk_text(content)

        # 传统固定窗口分块
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

                # 检查IVF索引是否需要训练
                if self.config.INDEX_TYPE == "IVFFlat" and hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    logger.info(f"IVF索引需要训练，使用当前 {len(texts)} 个向量作为训练数据...")
                    # 对于IVF索引，训练数据数量需要至少大于 nlist * 39
                    min_train_size = max(self.config.NLIST * 39, len(texts))

                    if len(texts) >= min_train_size:
                        # 使用当前向量训练索引
                        self.index.train(embeddings)
                        logger.info(f"IVF索引训练完成，使用了 {len(texts)} 个训练向量")
                    else:
                        logger.warning(f"IVF索引训练需要至少 {min_train_size} 个向量，当前只有 {len(texts)} 个")
                        logger.warning("切换到FlatIP索引以避免训练问题...")
                        # 自动切换到FlatIP索引
                        self.index = faiss.IndexFlatIP(self.config.EMBEDDING_DIM)

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
    
    def search(self, query: str, top_k: int = 5, use_optimization: bool = True) -> List[Dict[str, Any]]:
        """搜索相似文本 - 支持优化模式"""
        with self.lock:
            try:
                if self.index.ntotal == 0:
                    return []

                # 检查索引是否已训练（对于IVF索引）
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    logger.warning("IVF索引尚未训练，无法进行搜索")
                    return []

                # 使用优化的搜索方法
                if use_optimization and hasattr(self, 'advanced_search_index'):
                    return self.advanced_search_index.optimized_search(query, top_k)

                # 传统搜索方法
                return self._traditional_search(query, top_k)

            except Exception as e:
                logger.error(f"搜索失败: {e}")
                raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

    def _traditional_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """传统搜索方法"""
        # 生成查询向量
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        # 动态调整搜索参数
        if hasattr(self.index, 'nprobe'):  # IVF索引
            # 根据数据量动态调整nprobe
            optimal_nprobe = min(self.config.NPROBE, self.config.NLIST // 2, 20)
            self.index.nprobe = max(optimal_nprobe, 5)  # 确保至少搜索5个聚类
        elif hasattr(self.index, 'hnsw'):  # HNSW索引
            # 增加搜索深度以提高精度
            self.index.hnsw.efSearch = max(self.config.EF_SEARCH, 100)

        # 搜索，使用更大的候选集以提高精度
        search_k = min(top_k * 4, self.index.ntotal)  # 增加候选数量
        distances, indices = self.index.search(query_embedding, search_k)

        # 构建结果并添加基本的相关性过滤
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and str(idx) in self.id_to_chunk:
                # 处理可能的无效浮点值
                score = float(dist)
                if not (score == float('inf') or score == float('-inf') or score != score):  # 检查 inf, -inf, nan
                    # 基本的文本相关性检查：确保查询词在结果中
                    text = self.id_to_chunk[str(idx)]
                    query_terms = set(query.lower().split())
                    text_lower = text.lower()

                    # 计算简单的文本匹配分数
                    term_matches = sum(1 for term in query_terms if term in text_lower)
                    text_relevance = term_matches / max(len(query_terms), 1)

                    # 只有当有一定文本相关性时才添加结果
                    if text_relevance > 0.1:  # 至少有一些词匹配
                        results.append({
                            "text": text,
                            "score": score,
                            "faiss_id": int(idx),
                            "text_relevance": text_relevance,  # 添加文本相关性分数
                            "search_method": "traditional"
                        })

        # 按向量和文本相关性综合排序
        results.sort(key=lambda x: (x["score"], x["text_relevance"]), reverse=True)

        logger.info(f"传统搜索完成，返回 {len(results)} 个结果")
        return results[:top_k]
    
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
            stats = {
                "total_vectors": self.index.ntotal if self.index else 0,
                "embedding_dim": self.config.EMBEDDING_DIM,
                "index_type": self.config.INDEX_TYPE,
                "model_name": self.config.MODEL_NAME,
                "has_unsaved_changes": self.dirty,
                "search_optimization_enabled": hasattr(self, 'advanced_search_index')
            }

            # 添加索引特定的状态信息
            if self.index:
                if hasattr(self.index, 'is_trained'):
                    stats["is_trained"] = self.index.is_trained
                if hasattr(self.index, 'ntotal'):
                    stats["total_vectors"] = self.index.ntotal
                if hasattr(self.index, 'nlist'):
                    stats["nlist"] = self.index.nlist
                if hasattr(self.index, 'nprobe'):
                    stats["nprobe"] = self.index.nprobe

            # 添加搜索质量监控
            if hasattr(self, 'search_metrics'):
                stats.update(self.search_metrics.get_summary())

            return stats

    def enable_search_optimization(self):
        """启用搜索优化"""
        with self.lock:
            try:
                if not hasattr(self, 'advanced_search_index'):
                    logger.info("正在初始化AdvancedSearchIndex...")
                    logger.info(f"Config type: {type(self.config)}")
                    logger.info(f"Config attributes: {[attr for attr in dir(self.config) if not attr.startswith('_')]}")

                    # 验证必要的配置属性
                    required_attrs = ['MODEL_NAME', 'EMBEDDING_DIM', 'MIN_CHUNK_SIZE', 'MAX_CHUNK_SIZE']
                    for attr in required_attrs:
                        if not hasattr(self.config, attr):
                            logger.error(f"Config missing required attribute: {attr}")
                            return False
                        else:
                            logger.info(f"Config {attr}: {getattr(self.config, attr)}")

                    from search_optimization import AdvancedSearchIndex
                    self.advanced_search_index = AdvancedSearchIndex(self.config)
                    self.advanced_search_index.index = self.index
                    self.advanced_search_index.id_to_chunk = self.id_to_chunk
                    self.advanced_search_index.chunk_to_id = self.chunk_to_id

                    # 启用语义分块
                    from search_optimization import SemanticChunker
                    self.semantic_chunker = SemanticChunker(
                        self.embedding_model,
                        min_chunk_size=self.config.MIN_CHUNK_SIZE,
                        max_chunk_size=self.config.MAX_CHUNK_SIZE
                    )
                    self.use_semantic_chunking = True

                    logger.info("✅ 搜索优化已启用")
                    return True
            except Exception as e:
                import traceback
                logger.error(f"启用搜索优化失败: {e}")
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                return False

    def get_search_recommendations(self) -> Dict[str, Any]:
        """获取搜索优化建议"""
        if hasattr(self, 'advanced_search_index'):
            return self.advanced_search_index.get_search_recommendations()
        else:
            # 基础建议
            total_vectors = self.index.ntotal if self.index else 0
            recommendations = []

            if total_vectors < 1000:
                recommendations.append("数据量较小，建议使用FlatIP索引获得最佳精度")
                recommendations.append("考虑启用语义分块以提升向量质量")
            elif total_vectors < 50000:
                recommendations.append("考虑使用HNSW索引平衡精度和性能")
                if self.config.INDEX_TYPE == "IVFFlat":
                    recommendations.append("建议增加NPROBE参数到15-20以提升精度")
            else:
                recommendations.append("大数据量场景，建议使用IVFFlat或HNSW索引")
                recommendations.append("考虑启用搜索优化以获得更好的结果多样性")

            return {
                "current_vectors": total_vectors,
                "current_index_type": self.config.INDEX_TYPE,
                "optimization_enabled": hasattr(self, 'advanced_search_index'),
                "recommendations": recommendations
            }

    def benchmark_search_quality(self, test_queries: List[str]) -> Dict[str, Any]:
        """搜索质量基准测试"""
        if not test_queries or self.index.ntotal == 0:
            return {"error": "无法进行基准测试：无查询或数据为空"}

        try:
            quality_results = []
            for query in test_queries:
                # 传统搜索
                traditional_results = self.search(query, top_k=5, use_optimization=False)

                # 优化搜索
                if hasattr(self, 'advanced_search_index'):
                    optimized_results = self.search(query, top_k=5, use_optimization=True)
                    opt_avg_score = np.mean([r.get('relevance_score', r.get('score', 0)) for r in optimized_results])
                else:
                    opt_avg_score = 0

                trad_avg_score = np.mean([r.get('score', 0) for r in traditional_results])

                # 安全的浮点数转换函数
                def safe_float(value):
                    try:
                        if value == float('inf') or value == float('-inf') or value != value:
                            return 0.0
                        return float(value)
                    except (ValueError, TypeError):
                        return 0.0

                quality_results.append({
                    "query": query,
                    "traditional_avg_score": safe_float(trad_avg_score),
                    "optimized_avg_score": safe_float(opt_avg_score),
                    "improvement": safe_float(opt_avg_score - trad_avg_score)
                })

            overall_improvement = np.mean([r["improvement"] for r in quality_results])

            return {
                "test_queries_count": len(test_queries),
                "overall_improvement": safe_float(overall_improvement),
                "detailed_results": quality_results,
                "optimization_enabled": hasattr(self, 'advanced_search_index')
            }

        except Exception as e:
            return {"error": f"基准测试失败: {str(e)}"}

# 全局向量数据库实例
config = get_config()
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
    title="FAISS向量数据库API (MCP集成版)",
    description="优化的FAISS向量数据库服务，支持MCP协议和传统REST API",
    version="2.0.0",
    lifespan=lifespan
)

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class Document(BaseModel):
    content: str = Field(..., description="文档内容", max_length=50000)
    chunk_size: int = Field(default=500, description="分块大小", ge=50, le=2000)
    chunk_overlap: int = Field(default=50, description="分块重叠", ge=0, le=500)

    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v, info):
        if info.data.get('chunk_size') and v >= info.data['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v

class Query(BaseModel):
    question: str = Field(..., description="查询问题", max_length=1000)
    top_k: int = Field(default=3, description="返回结果数量", ge=1, le=50)
    use_optimization: bool = Field(default=True, description="是否使用搜索优化")

class BenchmarkQueries(BaseModel):
    queries: List[str] = Field(..., description="测试查询列表", max_length=20)

class BatchTexts(BaseModel):
    texts: List[str] = Field(..., description="批量文本列表", max_length=100)


# ============================================================================
# MCP 协议模型定义 - JSON-RPC 2.0 格式
# ============================================================================

class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 请求"""
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[int, str]] = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 响应"""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[int, str]] = None


class MCPTool(BaseModel):
    """MCP 工具定义"""
    name: str
    description: str
    inputSchema: Dict[str, Any]


class MCPContent(BaseModel):
    """MCP 内容"""
    type: str
    text: str


# ============================================================================
# MCP 工具定义和执行函数
# ============================================================================

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


async def execute_mcp_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """执行 MCP 工具调用 - 返回标准格式"""
    global vector_db
    
    if vector_db is None:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"error": "向量数据库未初始化"}, ensure_ascii=False)
            }],
            "isError": True
        }
    
    try:
        if name == "search_knowledge":
            # 搜索知识
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            use_optimization = arguments.get("use_optimization", True)
            
            if not query:
                raise ValueError("query parameter is required")
            
            if vector_db.index.ntotal == 0:
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "relevant_chunks": [],
                            "message": "知识库为空，请先添加文档",
                            "total_found": 0
                        }, ensure_ascii=False, indent=2)
                    }]
                }
            
            results = vector_db.search(query, top_k, use_optimization)
            search_method = "optimized" if use_optimization and hasattr(vector_db, 'advanced_search_index') else "traditional"
            
            response = {
                "relevant_chunks": [result["text"] for result in results],
                "detailed_results": results,
                "query": query,
                "total_found": len(results),
                "search_method": search_method,
                "optimization_enabled": use_optimization
            }
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, ensure_ascii=False, indent=2)
                }]
            }
        
        elif name == "add_document":
            # 添加文档
            content = arguments.get("content")
            chunk_size = arguments.get("chunk_size", 500)
            chunk_overlap = arguments.get("chunk_overlap", 50)
            
            if not content:
                raise ValueError("content parameter is required")
            
            chunks = vector_db._generate_chunks(content, chunk_size, chunk_overlap)
            
            if not chunks:
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "error": "文档内容过短，无法生成有效分块"
                        }, ensure_ascii=False)
                    }],
                    "isError": True
                }
            
            ids = vector_db.add_texts(chunks)
            
            if config.AUTO_SAVE:
                vector_db.save()
            
            response = {
                "message": f"文档处理成功，新增 {len(chunks)} 个知识块",
                "total_vectors": vector_db.index.ntotal,
                "chunks_added": len(chunks),
                "chunk_ids": ids[:5] if len(ids) > 5 else ids
            }
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, ensure_ascii=False, indent=2)
                }]
            }
        
        elif name == "delete_document":
            # 删除文档
            content = arguments.get("content")
            chunk_size = arguments.get("chunk_size", 500)
            chunk_overlap = arguments.get("chunk_overlap", 50)
            
            if not content:
                raise ValueError("content parameter is required")
            
            chunks = vector_db._generate_chunks(content, chunk_size, chunk_overlap)
            
            if not chunks:
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "error": "文档内容过短，无法生成有效分块"
                        }, ensure_ascii=False)
                    }],
                    "isError": True
                }
            
            deleted_count = vector_db.delete_texts(chunks)
            
            if config.AUTO_SAVE and deleted_count > 0:
                vector_db.save()
            
            response = {
                "message": f"成功删除 {deleted_count} 个知识块",
                "total_vectors": vector_db.index.ntotal,
                "deleted_count": deleted_count
            }
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, ensure_ascii=False, indent=2)
                }]
            }
        
        elif name == "batch_add_texts":
            # 批量添加文本
            texts = arguments.get("texts", [])
            
            if not texts:
                raise ValueError("texts parameter is required")
            
            if len(texts) > 100:
                raise ValueError("Maximum 100 texts allowed per batch")
            
            ids = vector_db.add_texts(texts)
            
            if config.AUTO_SAVE:
                vector_db.save()
            
            response = {
                "message": f"批量添加成功，新增 {len(texts)} 个文本",
                "total_vectors": vector_db.index.ntotal,
                "added_count": len(ids)
            }
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, ensure_ascii=False, indent=2)
                }]
            }
        
        elif name == "get_stats":
            # 获取统计信息
            stats = vector_db.get_stats()
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(stats, ensure_ascii=False, indent=2)
                }]
            }
        
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
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, ensure_ascii=False, indent=2)
                }],
                "isError": not success
            }
        
        elif name == "get_recommendations":
            # 获取搜索建议
            recommendations = vector_db.get_search_recommendations()
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(recommendations, ensure_ascii=False, indent=2)
                }]
            }
        
        elif name == "save_index":
            # 手动保存
            vector_db.save()
            
            response = {
                "message": "索引保存成功",
                "total_vectors": vector_db.index.ntotal
            }
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, ensure_ascii=False, indent=2)
                }]
            }
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "error": str(e),
                    "tool": name
                }, ensure_ascii=False)
            }],
            "isError": True
        }


async def handle_jsonrpc_request(request: JSONRPCRequest) -> JSONRPCResponse:
    """处理 JSON-RPC 请求"""
    try:
        method = request.method
        params = request.params or {}
        
        # 处理 initialize 方法
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "faiss-vector-search",
                    "version": "2.0.0"
                }
            }
            return JSONRPCResponse(jsonrpc="2.0", result=result, id=request.id)
        
        # 处理 tools/list 方法
        elif method == "tools/list":
            tools = get_mcp_tools()
            result = {
                "tools": [tool.dict() for tool in tools]
            }
            return JSONRPCResponse(jsonrpc="2.0", result=result, id=request.id)
        
        # 处理 tools/call 方法
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if not tool_name:
                return JSONRPCResponse(
                    jsonrpc="2.0",
                    error={
                        "code": -32602,
                        "message": "Invalid params: 'name' is required"
                    },
                    id=request.id
                )
            
            result = await execute_mcp_tool(tool_name, arguments)
            return JSONRPCResponse(jsonrpc="2.0", result=result, id=request.id)
        
        # 处理 notifications/initialized 通知
        elif method == "notifications/initialized":
            logger.info("收到客户端 initialized 通知")
            # 通知不需要响应
            return None
        
        else:
            return JSONRPCResponse(
                jsonrpc="2.0",
                error={
                    "code": -32601,
                    "message": f"Method not found: {method}"
                },
                id=request.id
            )
    
    except Exception as e:
        logger.error(f"JSON-RPC 处理错误: {e}")
        return JSONRPCResponse(
            jsonrpc="2.0",
            error={
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            },
            id=request.id
        )


# ============================================================================
# MCP 协议端点 - 标准 JSON-RPC 2.0
# ============================================================================

@app.post("/mcp", tags=["MCP"])
async def mcp_jsonrpc_endpoint(request: JSONRPCRequest):
    """
    MCP 标准 JSON-RPC 2.0 端点
    
    支持的方法:
    - initialize: 初始化 MCP 连接
    - tools/list: 列出所有可用工具
    - tools/call: 调用指定工具
    - notifications/initialized: 客户端初始化完成通知
    """
    logger.info(f"收到 JSON-RPC 请求: method={request.method}, id={request.id}")
    
    response = await handle_jsonrpc_request(request)
    
    # 通知类型的请求不返回响应
    if response is None:
        return {"status": "notification_received"}
    
    return response


# ============================================================================
# 传统 REST API 端点
# ============================================================================

@app.post("/add", summary="添加文档", tags=["传统API"])
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

@app.post("/batch_add", summary="批量添加文本", tags=["传统API"])
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

@app.delete("/delete", summary="删除文档", tags=["传统API"])
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

@app.post("/search", summary="搜索知识", tags=["传统API"])
async def search_knowledge(query: Query):
    """搜索相关知识"""
    try:
        if not vector_db:
            raise HTTPException(status_code=503, detail="向量数据库未初始化")

        if vector_db.index.ntotal == 0:
            return {"relevant_chunks": [], "message": "知识库为空，请先添加文档"}

        results = vector_db.search(query.question, query.top_k, query.use_optimization)

        # 添加搜索方法信息
        search_method = "optimized" if query.use_optimization and hasattr(vector_db, 'advanced_search_index') else "traditional"

        return {
            "relevant_chunks": [result["text"] for result in results],
            "detailed_results": results,
            "query": query.question,
            "total_found": len(results),
            "search_method": search_method,
            "optimization_enabled": query.use_optimization
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/save", summary="手动保存", tags=["传统API"])
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

@app.get("/stats", summary="获取统计信息", tags=["传统API"])
async def get_stats():
    """获取数据库统计信息"""
    try:
        if not vector_db:
            raise HTTPException(status_code=503, detail="向量数据库未初始化")
        
        return vector_db.get_stats()
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/enable_optimization", summary="启用搜索优化", tags=["传统API"])
async def enable_search_optimization():
    """启用搜索优化功能"""
    try:
        if not vector_db:
            raise HTTPException(status_code=503, detail="向量数据库未初始化")

        success = vector_db.enable_search_optimization()
        if success:
            return {
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
            raise HTTPException(status_code=500, detail="启用搜索优化失败")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启用搜索优化失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/search_recommendations", summary="获取搜索优化建议", tags=["传统API"])
async def get_search_recommendations():
    """获取针对当前数据量的搜索优化建议"""
    try:
        if not vector_db:
            raise HTTPException(status_code=503, detail="向量数据库未初始化")

        recommendations = vector_db.get_search_recommendations()
        return recommendations

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取搜索建议失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/benchmark_search_quality", summary="搜索质量基准测试", tags=["传统API"])
async def benchmark_search_quality(benchmark: BenchmarkQueries):
    """对比传统搜索和优化搜索的质量差异"""
    try:
        if not vector_db:
            raise HTTPException(status_code=503, detail="向量数据库未初始化")

        results = vector_db.benchmark_search_quality(benchmark.queries)
        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"基准测试失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/health", summary="健康检查", tags=["系统"])
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy" if vector_db else "initializing",
        "mcp_enabled": True,
        "mcp_protocol": "JSON-RPC 2.0",
        "mcp_protocol_version": "2024-11-05",
        "api_version": "2.0.0",
        "endpoints": {
            "mcp_standard": "POST /mcp (JSON-RPC 2.0)"
        },
        "features": {
            "mcp_protocol": True,
            "rest_api": True,
            "search_optimization": hasattr(vector_db, 'advanced_search_index') if vector_db else False
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower(),
        workers=config.API_WORKERS if config.API_WORKERS > 1 else 1
    )
