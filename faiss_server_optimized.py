"""
FAISS向量数据库服务器 - 优化版本

提供高性能的向量搜索、文档管理和检索增强功能。
支持业务类型隔离、混合检索、查询优化等高级特性。

Example:
    >>> from config import get_config
    >>> from faiss_server_optimized import FaissVectorDB
    >>> config = get_config()
    >>> db = FaissVectorDB(config)
    >>> results = await db.search("查询文本", top_k=5)
"""
import faiss
import hashlib
import json
import os
import math
import re
import threading
import time
import asyncio
import unicodedata
from pathlib import Path
from typing import List, Optional, Dict, Any, Sequence, Union
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer

from config import get_config, Config
from exceptions import (
    DatabaseError,
    DatabaseInitError,
    DatabaseSearchError,
    EmbeddingError,
    DimensionMismatchError,
    SearchError,
    ValidationError,
)
from logger import setup_logger
from metadata_store import SQLiteMetadataStore
from retrieval_enhancement import (
    EnhancedQuery,
    QualityMetrics as EnhancedQualityMetrics,
    RetrievalEnhancementCoordinator,
    SearchResult,
)
from search_optimization import AdvancedSearchIndex, QualityMetrics

# 初始化日志
logger = setup_logger(__name__)

# 设置 Hugging Face 镜像（中国网络优化）
# 优先使用环境变量 HF_ENDPOINT，否则使用默认镜像
if not os.environ.get('HF_ENDPOINT'):
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    logger.info(f"设置 Hugging Face 镜像: {os.environ['HF_ENDPOINT']}")

# 强制设置 HuggingFace Hub 缓存路径（sentence_transformers 依赖此路径）
os.environ['HF_HOME'] = '/root/.cache/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/.cache/huggingface/hub'
logger.info(f"设置 HuggingFace 缓存路径: {os.environ['HF_HOME']}")


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}|[a-zA-Z][a-zA-Z0-9_-]+|\d+")
SENSITIVE_PATH_PATTERN = re.compile(
    r"(?im)^obsidian path:\s*(账号配置|account|credential|secret|password|token|key)(/|\\|$)"
)
MIN_LEXICAL_OVERLAP_FOR_RESULTS = 0.3
SENSITIVE_BUSINESSTYPES = {"account_config"}
ACCESS_KEY_ENV = "ACCOUNT_CONFIG_ACCESS_KEY"
ACCESS_KEY_FILENAME = ".access_key"
SQL_TEMPLATE_SLOT_PATTERN = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")
SQL_TEMPLATE_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
SQL_TEMPLATE_SLOT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SQL_TEMPLATE_STATUSES = {"pending_review", "active", "disabled", "stale"}
SQL_TEMPLATE_SLOT_TYPES = {"string", "uint64", "date", "enum", "integer"}


class IndexMetadataConsistencyError(RuntimeError):
    """Raised when active SQLite chunks are absent from the FAISS index."""


def _search_tokens(text: str) -> List[str]:
    """Tokenize text for lightweight lexical relevance checks."""
    tokens: List[str] = []
    for token in TOKEN_PATTERN.findall(text or ""):
        token = token.lower()
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
            tokens.append(token)
            tokens.extend(token[i:i + 2] for i in range(len(token) - 1))
        else:
            tokens.append(token)
    return tokens


def _lexical_overlap_score(query: str, text: str) -> float:
    query_tokens = {token for token in _search_tokens(query) if len(token) > 1 or "\u4e00" <= token <= "\u9fff"}
    if not query_tokens:
        return 0.0

    text_lower = (text or "").lower()
    matches = sum(1 for token in query_tokens if token in text_lower)
    return matches / len(query_tokens)


def _looks_like_high_entropy_secret(text: str, allow_sensitive_path: bool = False) -> bool:
    """Detect encrypted/base64-like blobs that are poor search results."""
    if not text:
        return False

    if not allow_sensitive_path and SENSITIVE_PATH_PATTERN.search(text):
        return True

    compact_lines = [
        line.strip()
        for line in text.splitlines()
        if len(line.strip()) >= 120
    ]
    if not compact_lines:
        return False

    for line in compact_lines:
        chars = re.sub(r"\s+", "", line)
        if len(chars) < 120:
            continue

        alpha_numeric_symbol = sum(
            1 for char in chars if char.isascii() and (char.isalnum() or char in "+/=_-")
        )
        ascii_ratio = alpha_numeric_symbol / len(chars)
        if ascii_ratio < 0.92:
            continue

        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", line))
        if chinese_count > 0:
            continue

        unique_ratio = len(set(chars)) / len(chars)
        has_secret_markers = any(marker in text.lower() for marker in ("密文", "解密", "secret", "token", "password"))
        if unique_ratio > 0.18 or has_secret_markers:
            return True

    return False


# 线程安全的向量数据库类
class FaissVectorDB:
    def __init__(self, config: Config, businesstype: str = None):
        self.config = config
        self.businesstype = businesstype or config.DEFAULT_BUSINESSTYPE
        self.businesstype = config._validate_businesstype(self.businesstype)
        self.lock = threading.RLock()  # 重入锁保证线程安全
        self.embedding_model = None
        self.index = None
        self.id_to_chunk: Dict[str, str] = {}
        self.chunk_to_id: Dict[str, str] = {}  # 反向索引用于精确删除
        self.dirty = False  # 标记是否有未保存的更改
        self.hybrid_search_metrics = {
            "search_count": 0,
            "fallback_count": 0,
            "total_latency_ms": 0.0,
            "last_latency_ms": 0.0,
            "last_strategy_counts": {},
            "strategy_hit_totals": {"vector": 0, "bm25": 0, "exact": 0},
        }

        # 业务类型特定的路径
        self.index_file = config.get_index_file(self.businesstype)
        self.metadata_file = config.get_metadata_file(self.businesstype)
        self.data_dir = os.path.join(config.DATA_DIR, self.businesstype)
        self.sqlite_metadata_file = os.path.join(
            self.data_dir,
            f"{self.businesstype}_metadata.sqlite",
        )
        self.metadata_store: Optional[SQLiteMetadataStore] = None

        self._initialize()
    
    def _initialize(self):
        """初始化模型和索引"""
        try:
            logger.info(f"环境: {self.config.ENVIRONMENT}")
            logger.info(f"业务类型: {self.businesstype}")
            logger.info(f"数据目录: {self.data_dir}")
            logger.info(f"索引文件: {self.index_file}")
            logger.info(f"元数据文件: {self.metadata_file}")
            logger.info(f"是否使用GPU: {self.config.USE_GPU}")
            logger.info(f"模型名称: {self.config.MODEL_NAME}")

            # 确保数据目录存在
            os.makedirs(self.data_dir, exist_ok=True)
            self.metadata_store = SQLiteMetadataStore(self.sqlite_metadata_file)

            # 优先尝试使用本地缓存的模型
            # 构造模型本地路径
            model_cache_path = '/root/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots'

            try:
                # 查找实际的快照目录
                if os.path.exists(model_cache_path):
                    snapshots = [d for d in os.listdir(model_cache_path) if os.path.isdir(os.path.join(model_cache_path, d))]
                    if snapshots:
                        actual_model_path = os.path.join(model_cache_path, snapshots[0])
                        logger.info(f"找到本地模型路径: {actual_model_path}")
                        self.embedding_model = SentenceTransformer(actual_model_path)
                        logger.info("成功加载本地缓存的模型")
                    else:
                        raise Exception("未找到模型快照目录")
                else:
                    raise Exception(f"模型缓存路径不存在: {model_cache_path}")
            except Exception as e:
                logger.warning(f"本地模型加载失败 ({e})，尝试从网络下载...")
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
            logger.info(f"业务类型 {self.businesstype} 向量数据库初始化完成")

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise
    
    def _create_index(self):
        """根据配置创建FAISS索引"""
        if self.config.INDEX_TYPE == "FlatIP":
            logger.info(f"创建 FlatIP 索引，维度: {self.config.EMBEDDING_DIM}")
            return self._wrap_index_with_id_map(faiss.IndexFlatIP(self.config.EMBEDDING_DIM))
        elif self.config.INDEX_TYPE == "FlatL2":
            logger.info(f"创建 FlatL2 索引，维度: {self.config.EMBEDDING_DIM}")
            return self._wrap_index_with_id_map(faiss.IndexFlatL2(self.config.EMBEDDING_DIM))
        elif self.config.INDEX_TYPE == "IVFFlat":
            # IVF索引需要训练
            logger.info(f"创建 IVFFlat 索引，维度: {self.config.EMBEDDING_DIM}, nlist: {self.config.NLIST}")
            quantizer = faiss.IndexFlatL2(self.config.EMBEDDING_DIM)
            index = faiss.IndexIVFFlat(quantizer, self.config.EMBEDDING_DIM, self.config.NLIST)
            return self._wrap_index_with_id_map(index)
        elif self.config.INDEX_TYPE == "HNSW":
            # HNSW索引
            logger.info(f"创建 HNSW 索引，维度: {self.config.EMBEDDING_DIM}, M: {self.config.M}")
            index = faiss.IndexHNSWFlat(self.config.EMBEDDING_DIM, self.config.M)
            index.hnsw.efConstruction = self.config.EF_CONSTRUCTION
            index.hnsw.efSearch = self.config.EF_SEARCH
            return self._wrap_index_with_id_map(index)
        else:
            logger.warning(f"未知索引类型 {self.config.INDEX_TYPE}，使用默认 FlatIP")
            return self._wrap_index_with_id_map(faiss.IndexFlatIP(self.config.EMBEDDING_DIM))

    def _wrap_index_with_id_map(self, index):
        """Wrap FAISS index with stable external IDs when supported."""
        if hasattr(faiss, "IndexIDMap2"):
            return faiss.IndexIDMap2(index)
        if hasattr(faiss, "IndexIDMap"):
            return faiss.IndexIDMap(index)
        logger.warning("当前 FAISS 版本不支持 IndexIDMap，回退到顺序 ID 模式")
        return index

    def _supports_stable_vector_ids(self) -> bool:
        return hasattr(self.index, "add_with_ids")

    def _sync_active_metadata_cache(self) -> None:
        """Refresh in-memory maps used by search, BM25, and legacy responses."""
        if not self.metadata_store:
            return
        self.id_to_chunk = self.metadata_store.get_active_text_map()
        self.chunk_to_id = self.metadata_store.get_chunk_to_id_map()

    def _record_hybrid_search(
        self,
        started_at: float,
        strategy_counts: Optional[Dict[str, int]] = None,
        fallback: bool = False,
    ) -> None:
        latency_ms = (time.perf_counter() - started_at) * 1000
        metrics = self.hybrid_search_metrics
        metrics["search_count"] += 1
        metrics["fallback_count"] += int(fallback)
        metrics["total_latency_ms"] += latency_ms
        metrics["last_latency_ms"] = latency_ms
        if strategy_counts is not None:
            metrics["last_strategy_counts"] = dict(strategy_counts)
            for strategy, count in strategy_counts.items():
                metrics["strategy_hit_totals"][strategy] = (
                    metrics["strategy_hit_totals"].get(strategy, 0) + int(count)
                )

    def _get_index_vector_ids(self) -> set[int]:
        """Return external vector IDs stored in FAISS."""
        if self.index is None:
            return set()

        total = int(getattr(self.index, "ntotal", 0))
        id_map = getattr(self.index, "id_map", None)
        if id_map is not None:
            try:
                vector_ids = np.asarray(faiss.vector_to_array(id_map), dtype=np.int64)
                if vector_ids.size == total:
                    return {int(vector_id) for vector_id in vector_ids.tolist()}
            except Exception as exc:
                logger.warning(f"读取 FAISS IDMap 失败，按顺序 ID 校验: {exc}")

        return set(range(total))

    def _get_index_consistency(self) -> Dict[str, Any]:
        """Compare active SQLite IDs with the IDs addressable in FAISS."""
        if not self.metadata_store:
            return {
                "consistent": True,
                "checked": False,
                "missing_active_count": 0,
            }

        faiss_ids = self._get_index_vector_ids()
        active_ids = set(self.metadata_store.get_active_vector_ids())
        missing_active_ids = sorted(active_ids - faiss_ids)
        return {
            "consistent": not missing_active_ids,
            "checked": True,
            "faiss_id_count": len(faiss_ids),
            "active_metadata_id_count": len(active_ids),
            "missing_active_count": len(missing_active_ids),
            "missing_active_ids": missing_active_ids[:20],
        }

    def _validate_index_metadata_consistency(self) -> Dict[str, Any]:
        consistency = self._get_index_consistency()
        if not consistency["consistent"]:
            raise IndexMetadataConsistencyError(
                "SQLite 活动元数据包含 FAISS 中不存在的向量 ID: "
                f"count={consistency['missing_active_count']}, "
                f"sample={consistency['missing_active_ids']}"
            )
        return consistency

    def _add_embeddings_with_ids(self, embeddings, vector_ids: List[int]) -> None:
        id_array = np.array(vector_ids, dtype=np.int64)
        if self._supports_stable_vector_ids():
            self.index.add_with_ids(embeddings, id_array)
            return
        self.index.add(embeddings)

    def _is_active_vector_id(self, vector_id: int) -> bool:
        if self.metadata_store:
            return self.metadata_store.is_active(vector_id)
        return str(vector_id) in self.id_to_chunk

    def _get_text_for_vector_id(self, vector_id: int) -> Optional[str]:
        if self.metadata_store:
            return self.metadata_store.get_text(vector_id)
        return self.id_to_chunk.get(str(vector_id))

    def _is_searchable_text(self, text: Optional[str]) -> bool:
        allow_sensitive_path = self.businesstype == "account_config"
        return bool(text and not _looks_like_high_entropy_secret(text, allow_sensitive_path=allow_sensitive_path))

    def _index_metric_kind(self) -> str:
        configured = str(getattr(self.config, "INDEX_TYPE", "FlatIP")).lower()
        if configured in {"flatl2", "ivfflat", "hnsw"}:
            return "distance"
        return "similarity"

    def _normalize_vector_relevance(self, raw_score: float) -> float:
        if not math.isfinite(raw_score):
            return 0.0

        if self._index_metric_kind() == "similarity":
            return max(0.0, min(1.0, raw_score))

        # Embeddings are normalized. For squared L2 distance, cosine ~= 1 - d/2.
        return max(0.0, min(1.0, 1.0 - (raw_score / 2.0)))
    
    def _create_or_load_index(self):
        """创建或加载索引"""
        try:
            self.load()
        except FileNotFoundError as e:
            active_ids = (
                self.metadata_store.get_active_vector_ids()
                if self.metadata_store
                else []
            )
            if active_ids:
                raise IndexMetadataConsistencyError(
                    "FAISS 索引文件缺失，但 SQLite 仍有活动元数据；"
                    "拒绝创建空索引以避免静默丢失"
                ) from e

            logger.info(f"未找到现有索引: {e}，创建新索引...")
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
            self.index = self._wrap_index_with_id_map(faiss.IndexFlatIP(self.config.EMBEDDING_DIM))

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
                        self.index = self._wrap_index_with_id_map(faiss.IndexFlatIP(self.config.EMBEDDING_DIM))

                # 先由 SQLite 分配稳定 vector_id，再写入 FAISS。
                vector_ids = self.metadata_store.add_chunks(texts) if self.metadata_store else []
                if not vector_ids:
                    start_faiss_id = self.index.ntotal
                    vector_ids = [start_faiss_id + i for i in range(len(texts))]

                try:
                    self._add_embeddings_with_ids(embeddings, vector_ids)
                except Exception:
                    if self.metadata_store and vector_ids:
                        self.metadata_store.soft_delete_vector_ids(vector_ids)
                        self._sync_active_metadata_cache()
                    raise
                self._sync_active_metadata_cache()

                self.dirty = True
                logger.info(f"成功添加 {len(texts)} 个文本块，总向量数: {self.index.ntotal}")
                return [str(vector_id) for vector_id in vector_ids]

            except Exception as e:
                logger.error(f"添加文本失败: {e}")
                raise HTTPException(status_code=500, detail=f"添加文本失败: {str(e)}")
    
    def search(self, query: str, top_k: int = 5, use_optimization: bool = True,
             use_enhanced: bool = True) -> List[Dict[str, Any]]:
        """
        搜索相似文本 - 支持优化模式和增强检索

        Args:
            query: Search query text
            top_k: Number of results to return
            use_optimization: Use advanced search optimization
            use_enhanced: Use new enhanced retrieval with adaptive thresholding

        Returns:
            List of search results
        """
        with self.lock:
            try:
                if self.index.ntotal == 0:
                    return []

                # 检查索引是否已训练（对于IVF索引）
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    logger.warning("IVF索引尚未训练，无法进行搜索")
                    return []

                # 优先使用增强检索（如果启用）
                if use_enhanced and self.config.ENABLE_ADAPTIVE_THRESHOLD:
                    return self.enhanced_search(query, top_k)

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
        # 软删除会留下 inactive 向量，候选集需要适度放大后再过滤。
        search_k = min(max(top_k * 8, top_k), self.index.ntotal)
        distances, indices = self.index.search(query_embedding, search_k)

        # 构建结果并添加基本的相关性过滤
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and self._is_active_vector_id(int(idx)):
                # 处理可能的无效浮点值
                score = float(dist)
                if not (score == float('inf') or score == float('-inf') or score != score):  # 检查 inf, -inf, nan
                    # 基本的文本相关性检查：确保查询词在结果中
                    text = self._get_text_for_vector_id(int(idx))
                    if not self._is_searchable_text(text):
                        continue
                    text_relevance = _lexical_overlap_score(query, text)

                    # 只有当有一定文本相关性时才添加结果
                    if text_relevance >= MIN_LEXICAL_OVERLAP_FOR_RESULTS:
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

    def enhanced_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search with vector, BM25, exact match, and weighted RRF fusion."""
        started_at = time.perf_counter()
        try:
            if not hasattr(self, 'enhancement_coordinator'):
                self.enhancement_coordinator = RetrievalEnhancementCoordinator(
                    self.config,
                    self.embedding_model,
                    self.businesstype,
                    documents_provider=(
                        self.metadata_store.get_active_text_map
                        if self.metadata_store
                        else None
                    ),
                    revision_provider=(
                        self.metadata_store.get_chunks_revision
                        if self.metadata_store
                        else None
                    ),
                )

            coordinator = self.enhancement_coordinator
            enhanced_query = coordinator.enhance_query(query)
            hybrid_enabled = bool(self.config.ENABLE_HYBRID_RETRIEVAL)
            candidate_k = max(top_k * 3, top_k)

            vector_by_id: Dict[int, SearchResult] = {}
            for result_dict in self._execute_vector_search(query, candidate_k):
                faiss_id = int(result_dict['faiss_id'])
                vector_by_id[faiss_id] = SearchResult(
                    text=result_dict['text'],
                    similarity_score=result_dict.get('similarity_score', result_dict.get('score', 0.0)),
                    relevance_score=result_dict.get('relevance_score', result_dict.get('score', 0.0)),
                    faiss_id=faiss_id,
                    match_type='vector',
                    strategies_used=['vector']
                )

            if (self.config.ENABLE_QUERY_EXPANSION and
                enhanced_query.expanded_variants and
                enhanced_query.metadata.get('has_domain_terms', False)):
                for expanded_query in enhanced_query.expanded_variants[:3]:
                    expanded_results = self._execute_vector_search(expanded_query, top_k)
                    for result_dict in expanded_results:
                        faiss_id = int(result_dict['faiss_id'])
                        if faiss_id in vector_by_id:
                            if 'vector_expanded' not in vector_by_id[faiss_id].strategies_used:
                                vector_by_id[faiss_id].strategies_used.append('vector_expanded')
                            vector_by_id[faiss_id].relevance_score = max(
                                vector_by_id[faiss_id].relevance_score,
                                result_dict.get('relevance_score', result_dict.get('score', 0.0)),
                            )
                        else:
                            vector_by_id[faiss_id] = SearchResult(
                                text=result_dict['text'],
                                similarity_score=result_dict.get('similarity_score', result_dict.get('score', 0.0)),
                                relevance_score=result_dict.get('relevance_score', result_dict.get('score', 0.0)),
                                faiss_id=faiss_id,
                                match_type='vector',
                                strategies_used=['vector_expanded']
                            )

            avg_relevance = (
                float(np.mean([r.relevance_score for r in vector_by_id.values()]))
                if vector_by_id
                else 0.0
            )
            lexical_hit_count = sum(
                1 for result in vector_by_id.values()
                if _lexical_overlap_score(query, result.text) > 0
            )
            run_exact = bool(
                self.config.ENABLE_EXACT_MATCH_FALLBACK
                and (
                    hybrid_enabled
                    or len(vector_by_id) < top_k
                    or avg_relevance < 0.3
                )
            )
            exact_results = []
            if run_exact:
                exact_results = [
                    SearchResult(
                        text=item['text'],
                        similarity_score=item['score'],
                        relevance_score=item['score'],
                        faiss_id=int(item['faiss_id']),
                        match_type='exact',
                        strategies_used=['exact'],
                    )
                    for item in self._execute_exact_match_search(query, candidate_k)
                ]

            run_bm25 = bool(
                self.config.ENABLE_BM25_FALLBACK
                and coordinator.is_bm25_available()
                and (
                    hybrid_enabled
                    or len(vector_by_id) < top_k
                    or lexical_hit_count < min(top_k, 3)
                )
            )
            bm25_results = coordinator.search_bm25(query, candidate_k) if run_bm25 else []
            bm25_results = [
                result
                for result in bm25_results
                if result.faiss_id >= 0 and self._is_searchable_text(result.text)
            ]

            strategy_results = {
                'vector': list(vector_by_id.values()),
                'bm25': bm25_results,
                'exact': exact_results,
            }
            results_list = coordinator.result_fusion.fuse(
                strategy_results,
                candidate_k,
            )

            if not results_list:
                logger.info("Enhanced search found no results")
                self._record_hybrid_search(
                    started_at,
                    {
                        name: len(results)
                        for name, results in strategy_results.items()
                    },
                )
                return []

            lexical_scores = {
                id(result): _lexical_overlap_score(query, result.text)
                for result in results_list
            }
            has_lexical_candidates = any(score > 0 for score in lexical_scores.values())

            relevance_scores = [r.relevance_score for r in results_list]
            threshold, adjustments = coordinator.calculate_adaptive_threshold(
                query, relevance_scores, len(results_list)
            )

            logger.info(f"Adaptive threshold: {threshold:.3f} (base: {self.config.BASE_RELEVANCE_THRESHOLD})")
            if adjustments:
                logger.debug(f"Threshold adjustments: {adjustments}")

            filtered_results = [r for r in results_list if r.relevance_score >= threshold]

            # Keep lexical hits for technical/document queries. Pure vector hits with
            # zero lexical overlap are often encrypted blobs or unrelated notes in a
            # small mixed-purpose knowledge base.
            if has_lexical_candidates:
                filtered_results = [
                    r for r in filtered_results
                    if (
                        lexical_scores.get(id(r), 0.0) >= MIN_LEXICAL_OVERLAP_FOR_RESULTS
                        or "exact" in r.strategies_used
                        or "bm25" in r.strategies_used
                    )
                ]

            # If still too few results, relax threshold
            if len(filtered_results) < min(top_k, 3) and not has_lexical_candidates:
                relaxed_threshold = max(self.config.MIN_RELEVANCE_THRESHOLD, threshold * 0.5)
                filtered_results = [r for r in results_list if r.relevance_score >= relaxed_threshold]
                logger.info(f"Relaxed threshold to {relaxed_threshold:.3f}, got {len(filtered_results)} results")

            # If still no results, return all sorted by relevance
            if len(filtered_results) == 0 and not has_lexical_candidates:
                filtered_results = sorted(results_list, key=lambda x: x.relevance_score, reverse=True)
                logger.warning("No results passed threshold, returning all results sorted by relevance")

            filtered_results.sort(
                key=lambda x: (
                    x.composite_score,
                    lexical_scores.get(id(x), 0.0),
                    x.relevance_score,
                ),
                reverse=True,
            )

            # Step 6: Calculate quality metrics
            quality_metrics = coordinator.calculate_quality_metrics(
                filtered_results[:top_k], query
            )

            # Step 7: Format results for return
            formatted_results = []
            for i, result in enumerate(filtered_results[:top_k]):
                result_dict = {
                    "text": result.text,
                    "similarity_score": float(result.similarity_score),
                    "relevance_score": float(result.relevance_score),
                    "faiss_id": int(result.faiss_id),
                    "match_type": result.match_type,
                    "strategies_used": result.strategies_used,
                    "lexical_score": float(lexical_scores.get(id(result), 0.0)),
                    "rrf_score": float(result.rrf_score),
                    "composite_score": float(result.composite_score),
                    "rank": i + 1,
                    "search_method": "hybrid" if hybrid_enabled else "enhanced",
                }

                # Add quality metrics to first result only
                if i == 0:
                    result_dict["quality_metrics"] = {
                        "avg_relevance_score": float(quality_metrics.avg_relevance_score),
                        "diversity_score": float(quality_metrics.diversity_score),
                        "coverage_ratio": float(quality_metrics.coverage_ratio),
                        "precision_at_k": float(quality_metrics.precision_at_k),
                        "total_results": int(quality_metrics.total_results)
                    }

                formatted_results.append(result_dict)

            self._record_hybrid_search(
                started_at,
                {
                    name: len(results)
                    for name, results in strategy_results.items()
                },
            )
            logger.info(
                f"Hybrid search complete: {len(formatted_results)} results "
                f"(threshold: {threshold:.3f})"
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Enhanced search failed: {e}, falling back to traditional search")
            self._record_hybrid_search(started_at, fallback=True)
            return self._traditional_search(query, top_k)

    def _execute_vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Execute vector search and return results with relevance scores"""
        if self.index.ntotal == 0:
            return []

        # Check if IVF index is trained
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.warning("IVF index not trained, cannot search")
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        # Adjust search parameters for different index types
        if hasattr(self.index, 'nprobe'):  # IVF index
            optimal_nprobe = min(self.config.NPROBE, self.config.NLIST // 2, 20)
            self.index.nprobe = max(optimal_nprobe, 5)
        elif hasattr(self.index, 'hnsw'):  # HNSW index
            self.index.hnsw.efSearch = max(self.config.EF_SEARCH, 100)

        # Search with candidate multiplier
        search_k = min(top_k * self.config.SEARCH_CANDIDATE_MULTIPLIER, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, search_k)

        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and self._is_active_vector_id(int(idx)):
                score = float(dist)
                if not (score == float('inf') or score == float('-inf') or score != score):
                    text = self._get_text_for_vector_id(int(idx))
                    if not self._is_searchable_text(text):
                        continue

                    relevance_score = self._normalize_vector_relevance(score)
                    lexical_score = _lexical_overlap_score(query, text)

                    results.append({
                        "text": text,
                        "score": score,
                        "similarity_score": score,
                        "relevance_score": relevance_score,
                        "lexical_score": lexical_score,
                        "faiss_id": int(idx)
                    })

        return results

    def _execute_exact_match_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Execute indexed full-text equality matching on SQLite metadata."""
        normalized_query = query.strip()
        if not normalized_query:
            return []

        if self.metadata_store:
            vector_ids = self.metadata_store.get_active_ids_for_text(normalized_query)
        else:
            vector_id = self.chunk_to_id.get(normalized_query)
            vector_ids = [int(vector_id)] if vector_id is not None else []

        return [
            {
                "text": normalized_query,
                "score": 1.0,
                "similarity_score": 1.0,
                "relevance_score": 1.0,
                "faiss_id": int(vector_id),
            }
            for vector_id in vector_ids[:top_k]
            if self._is_searchable_text(normalized_query)
        ]
    
    def delete_texts(self, texts_to_delete: List[str]) -> int:
        """精确删除指定文本。

        百万级知识库采用软删除：FAISS 向量保留到下一次 compaction，
        查询阶段通过 SQLite active 状态过滤。
        """
        with self.lock:
            try:
                if not self.metadata_store:
                    ids_to_remove = []
                    for text in texts_to_delete:
                        if text in self.chunk_to_id:
                            faiss_id = self.chunk_to_id[text]
                            ids_to_remove.append(int(faiss_id))

                    if not ids_to_remove:
                        logger.info("未找到要删除的文本")
                        return 0

                    self.index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
                    for faiss_id in ids_to_remove:
                        faiss_id_str = str(faiss_id)
                        if faiss_id_str in self.id_to_chunk:
                            text = self.id_to_chunk[faiss_id_str]
                            del self.id_to_chunk[faiss_id_str]
                            if text in self.chunk_to_id:
                                del self.chunk_to_id[text]

                    self.dirty = True
                    return len(ids_to_remove)

                deleted_ids = self.metadata_store.soft_delete_texts(texts_to_delete)
                if not deleted_ids:
                    logger.info("未找到要删除的文本")
                    return 0

                self._sync_active_metadata_cache()
                self.dirty = True
                logger.info(f"成功软删除 {len(deleted_ids)} 个文本块")
                return len(deleted_ids)
                
            except Exception as e:
                logger.error(f"删除文本失败: {e}")
                raise HTTPException(status_code=500, detail=f"删除文本失败: {str(e)}")

    def update_texts(self, updates: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        批量更新文本（upsert 语义）

        Args:
            updates: 更新列表，每项包含 {
                "old_text": "原始文本内容",
                "new_text": "新文本内容"
            }

        Returns:
            {
                "success_count": int,      # 成功更新数量
                "failed_count": int,       # 失败数量
                "inserted_count": int,     # 新增数量（old_text 不存在）
                "updated_count": int,      # 更新数量（找到并替换）
                "errors": List[Dict]       # 失败详情 [{index, old_text, error}]
            }
        """
        with self.lock:
            try:
                errors = []
                old_ids_to_deactivate = []
                all_new_texts = []
                update_indices = []  # 跟踪哪些是更新，哪些是新增

                logger.info(f"开始处理 {len(updates)} 个更新请求...")

                # Step 1: 查找阶段 - 分类为"更新"或"新增"
                for i, update in enumerate(updates):
                    try:
                        old_text = update.get("old_text", "").strip()
                        new_text = update.get("new_text", "").strip()

                        # 验证输入
                        if not old_text or not new_text:
                            errors.append({
                                "index": i,
                                "old_text": old_text[:100] + "..." if len(old_text) > 100 else old_text,
                                "error": "文本内容为空"
                            })
                            continue

                        # 跳过相同文本
                        if old_text == new_text:
                            logger.debug(f"索引 {i}: 旧文本和新文本相同，跳过")
                            continue

                        # 查找旧文本。SQLite store 会返回所有 active exact-match chunk。
                        active_ids = (
                            self.metadata_store.get_active_ids_for_text(old_text)
                            if self.metadata_store
                            else ([int(self.chunk_to_id[old_text])] if old_text in self.chunk_to_id else [])
                        )
                        if active_ids:
                            # 找到了，标记为更新
                            old_ids_to_deactivate.extend(active_ids)
                            all_new_texts.append(new_text)
                            update_indices.append({"type": "update", "index": i})
                            logger.debug(f"索引 {i}: 找到旧文本 (vector_ids: {active_ids})，将更新")
                        else:
                            # 未找到，标记为新增
                            all_new_texts.append(new_text)
                            update_indices.append({"type": "insert", "index": i})
                            logger.debug(f"索引 {i}: 未找到旧文本，将新增")

                    except Exception as e:
                        errors.append({
                            "index": i,
                            "old_text": update.get("old_text", "")[:100],
                            "error": str(e)
                        })
                        logger.error(f"处理索引 {i} 时出错: {e}")

                if not all_new_texts:
                    logger.info("没有需要处理的新文本")
                    return {
                        "success_count": 0,
                        "failed_count": len(errors),
                        "inserted_count": 0,
                        "updated_count": 0,
                        "errors": errors
                    }

                # Step 2: 批量生成嵌入向量。先准备新向量，成功后再停用旧块。
                logger.info(f"为 {len(all_new_texts)} 个新文本生成嵌入向量...")
                embeddings = self.embedding_model.encode(
                    all_new_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=self.config.BATCH_SIZE
                )

                # Step 3: 批量添加新向量
                logger.info(f"添加 {len(embeddings)} 个新向量到索引...")
                vector_ids = self.metadata_store.add_chunks(all_new_texts) if self.metadata_store else []
                if not vector_ids:
                    start_faiss_id = self.index.ntotal
                    vector_ids = [start_faiss_id + i for i in range(len(all_new_texts))]
                    for new_text, vector_id in zip(all_new_texts, vector_ids):
                        self.id_to_chunk[str(vector_id)] = new_text
                        self.chunk_to_id[new_text] = str(vector_id)

                try:
                    self._add_embeddings_with_ids(embeddings, vector_ids)
                except Exception:
                    if self.metadata_store and vector_ids:
                        self.metadata_store.soft_delete_vector_ids(vector_ids)
                        self._sync_active_metadata_cache()
                    raise

                # Step 4: 新向量已写入后再软删除旧向量，避免失败时知识丢失。
                if old_ids_to_deactivate and self.metadata_store:
                    logger.info(f"软删除 {len(old_ids_to_deactivate)} 个旧向量...")
                    self.metadata_store.soft_delete_vector_ids(old_ids_to_deactivate)
                elif old_ids_to_deactivate:
                    logger.info(f"删除 {len(old_ids_to_deactivate)} 个旧向量...")
                    self.index.remove_ids(np.array(old_ids_to_deactivate, dtype=np.int64))

                self._sync_active_metadata_cache()

                self.dirty = True

                # 统计结果
                inserted_count = sum(1 for u in update_indices if u["type"] == "insert")
                updated_count = sum(1 for u in update_indices if u["type"] == "update")
                success_count = inserted_count + updated_count

                logger.info(f"更新完成: 成功 {success_count}, 新增 {inserted_count}, 更新 {updated_count}, 失败 {len(errors)}")

                return {
                    "success_count": success_count,
                    "failed_count": len(errors),
                    "inserted_count": inserted_count,
                    "updated_count": updated_count,
                    "errors": errors
                }

            except Exception as e:
                logger.error(f"批量更新失败: {e}")
                raise HTTPException(status_code=500, detail=f"批量更新失败: {str(e)}")

    def _validate_sql_template_record(
        self,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        required_fields = (
            "external_id",
            "dataset_id",
            "template_id",
            "intent_key",
            "canonical_template",
            "search_text",
            "required_slots",
            "sql_template",
            "schema_fingerprint",
            "template_version",
        )
        missing = [field for field in required_fields if not record.get(field)]
        if missing:
            raise ValueError(
                "Missing SQL template fields: " + ", ".join(sorted(missing))
            )

        normalized = dict(record)
        for field in (
            "external_id",
            "dataset_id",
            "template_id",
            "intent_key",
            "canonical_template",
            "search_text",
            "sql_template",
            "schema_fingerprint",
        ):
            normalized[field] = str(normalized[field]).strip()

        for field in ("external_id", "dataset_id", "template_id", "intent_key"):
            if not SQL_TEMPLATE_NAME_PATTERN.fullmatch(normalized[field]):
                raise ValueError(f"Invalid SQL template {field}: {normalized[field]}")

        normalized["template_version"] = int(normalized["template_version"])
        if normalized["template_version"] <= 0:
            raise ValueError("template_version must be positive")

        status = str(normalized.get("status", "pending_review")).strip()
        if status not in SQL_TEMPLATE_STATUSES:
            raise ValueError(f"Invalid SQL template status: {status}")
        normalized["status"] = status
        normalized["source"] = str(normalized.get("source", "manual")).strip()

        required_slots = normalized["required_slots"]
        if not isinstance(required_slots, dict) or not required_slots:
            raise ValueError("required_slots must be a non-empty object")
        for slot_name, slot_type in required_slots.items():
            if not SQL_TEMPLATE_SLOT_NAME_PATTERN.fullmatch(str(slot_name)):
                raise ValueError(f"Invalid SQL template slot name: {slot_name}")
            if str(slot_type) not in SQL_TEMPLATE_SLOT_TYPES:
                raise ValueError(
                    f"Unsupported SQL template slot type: {slot_type}"
                )
        normalized["required_slots"] = {
            str(name): str(slot_type)
            for name, slot_type in required_slots.items()
        }

        sql = normalized["sql_template"]
        if not sql.upper().startswith(("SELECT", "WITH")):
            raise ValueError("sql_template must start with SELECT or WITH")
        if ";" in sql.rstrip(";"):
            raise ValueError("sql_template must contain a single statement")
        placeholders = set(SQL_TEMPLATE_SLOT_PATTERN.findall(sql))
        declared_slots = set(normalized["required_slots"])
        if placeholders != declared_slots:
            raise ValueError(
                "sql_template placeholders must exactly match required_slots"
            )
        return normalized

    def upsert_sql_template(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Upsert a SQL template while embedding only its search_text."""
        with self.lock:
            if not self.metadata_store:
                raise ValueError("SQLite metadata store is required")

            normalized = self._validate_sql_template_record(record)
            external_id = normalized["external_id"]
            existing = self.metadata_store.get_sql_template(external_id)
            old_vector_id = existing.get("vector_id") if existing else None
            vector_id = old_vector_id
            added_vector_id = None

            needs_index = normalized["status"] == "active"
            search_text_changed = bool(
                existing and existing.get("search_text") != normalized["search_text"]
            )
            old_vector_active = bool(
                old_vector_id is not None
                and self.metadata_store.is_active(int(old_vector_id))
            )

            if needs_index and (search_text_changed or not old_vector_active):
                embeddings = self.embedding_model.encode(
                    [normalized["search_text"]],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=self.config.BATCH_SIZE,
                )
                vector_ids = self.metadata_store.add_chunks(
                    [normalized["search_text"]],
                    source_document=f"sql-template:{external_id}",
                )
                added_vector_id = int(vector_ids[0])
                try:
                    self._add_embeddings_with_ids(embeddings, [added_vector_id])
                except Exception:
                    self.metadata_store.soft_delete_vector_ids([added_vector_id])
                    self._sync_active_metadata_cache()
                    raise
                vector_id = added_vector_id
            elif not needs_index:
                vector_id = None

            try:
                stored = self.metadata_store.upsert_sql_template(
                    normalized,
                    vector_id=vector_id,
                )
            except Exception:
                if added_vector_id is not None:
                    self.metadata_store.soft_delete_vector_ids([added_vector_id])
                    self._sync_active_metadata_cache()
                raise

            if old_vector_id is not None and old_vector_id != vector_id:
                self.metadata_store.soft_delete_vector_ids([int(old_vector_id)])

            self._sync_active_metadata_cache()
            if old_vector_id != vector_id:
                self.dirty = True
            return {
                "action": "updated" if existing else "inserted",
                "template": stored,
            }

    def search_sql_templates(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 10,
        intent_key: Optional[str] = None,
        canonical_template: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search active templates using exact template keys and hybrid retrieval."""
        with self.lock:
            if not self.metadata_store:
                return []
            dataset_id = str(dataset_id).strip()
            if not dataset_id:
                raise ValueError("dataset_id is required")
            limit = max(1, min(int(top_k), 50))
            matches: Dict[str, Dict[str, Any]] = {}

            if intent_key and canonical_template:
                exact_records = self.metadata_store.find_sql_templates(
                    dataset_id,
                    intent_key=str(intent_key),
                    canonical_template=str(canonical_template),
                    statuses=("active",),
                    limit=limit,
                )
                for record in exact_records:
                    matches[record["external_id"]] = {
                        **record,
                        "match_type": "template_exact",
                        "similarity_score": 1.0,
                        "relevance_score": 1.0,
                        "lexical_score": 1.0,
                        "rerank_score": 1.0,
                        "strategies_used": ["template_key"],
                    }

            query = str(query or "").strip()
            if query and len(matches) < limit and self.index.ntotal > 0:
                candidates = self.search(
                    query,
                    max(limit * 3, limit),
                    use_optimization=False,
                    use_enhanced=True,
                )
                by_vector = self.metadata_store.get_sql_templates_by_vector_ids(
                    [candidate.get("faiss_id", -1) for candidate in candidates],
                    dataset_id,
                    statuses=("active",),
                )
                for candidate in candidates:
                    vector_id = int(candidate.get("faiss_id", -1))
                    record = by_vector.get(vector_id)
                    if record is None or record["external_id"] in matches:
                        continue
                    relevance = float(candidate.get("relevance_score", 0.0))
                    lexical = float(candidate.get("lexical_score", 0.0))
                    rerank_score = (0.7 * relevance) + (0.3 * lexical)
                    matches[record["external_id"]] = {
                        **record,
                        "match_type": "template_semantic",
                        "similarity_score": float(
                            candidate.get("similarity_score", 0.0)
                        ),
                        "relevance_score": relevance,
                        "lexical_score": lexical,
                        "rerank_score": rerank_score,
                        "strategies_used": list(
                            candidate.get("strategies_used", ["vector"])
                        ),
                    }

            ordered = sorted(
                matches.values(),
                key=lambda item: (
                    item["match_type"] == "template_exact",
                    float(item.get("rerank_score", 0.0)),
                    int(item.get("template_version", 0)),
                ),
                reverse=True,
            )
            return ordered[:limit]

    def list_sql_templates(
        self,
        dataset_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """List templates and their lifecycle counters for administration."""
        with self.lock:
            if not self.metadata_store:
                return []
            return self.metadata_store.list_sql_templates(
                dataset_id=str(dataset_id).strip() if dataset_id else None,
                status=str(status).strip() if status else None,
                limit=limit,
            )

    def set_sql_template_status(
        self,
        external_id: str,
        status: str,
    ) -> Optional[Dict[str, Any]]:
        with self.lock:
            if not self.metadata_store:
                return None
            existing = self.metadata_store.get_sql_template(external_id)
            if existing is None:
                return None
            return self.upsert_sql_template({**existing, "status": status})["template"]

    def delete_sql_template(
        self,
        external_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Permanently remove a SQL template and retire its search vector."""
        with self.lock:
            if not self.metadata_store:
                return None
            deleted = self.metadata_store.delete_sql_template(external_id)
            if deleted is None:
                return None
            vector_id = deleted.get("vector_id")
            if vector_id is not None:
                self.metadata_store.soft_delete_vector_ids([int(vector_id)])
                self._sync_active_metadata_cache()
                self.dirty = True
            return deleted

    def record_sql_template_outcome(
        self,
        external_id: str,
        outcome: str,
    ) -> Optional[Dict[str, Any]]:
        with self.lock:
            if not self.metadata_store:
                return None
            return self.metadata_store.record_sql_template_outcome(
                external_id,
                outcome,
            )

    def get_sql_template_metrics(self) -> Dict[str, Any]:
        if not self.metadata_store:
            return {"total_templates": 0, "by_status": {}}
        return self.metadata_store.get_sql_template_metrics()

    def migrate_sql_templates(
        self,
        records: Sequence[Dict[str, Any]],
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Validate or apply an explicit batch of reviewed SQL templates."""
        if len(records) > 100:
            raise ValueError("Maximum 100 SQL templates allowed per batch")
        results = []
        valid_records = []
        for index, record in enumerate(records):
            try:
                normalized = self._validate_sql_template_record(record)
                valid_records.append(normalized)
                results.append(
                    {
                        "index": index,
                        "external_id": normalized["external_id"],
                        "valid": True,
                        "applied": False,
                    }
                )
            except Exception as error:
                results.append(
                    {
                        "index": index,
                        "external_id": str(record.get("external_id", "")),
                        "valid": False,
                        "applied": False,
                        "error": str(error),
                    }
                )

        if not dry_run:
            valid_by_id = {
                record["external_id"]: record for record in valid_records
            }
            for result in results:
                if not result["valid"]:
                    continue
                applied = self.upsert_sql_template(
                    valid_by_id[result["external_id"]]
                )
                result["applied"] = True
                result["action"] = applied["action"]

        return {
            "dry_run": bool(dry_run),
            "total_count": len(records),
            "valid_count": sum(1 for result in results if result["valid"]),
            "invalid_count": sum(1 for result in results if not result["valid"]),
            "applied_count": sum(1 for result in results if result["applied"]),
            "results": results,
        }

    def audit_sql_documents(self, detail_limit: int = 100) -> Dict[str, Any]:
        """Inspect legacy Question/SQL chunks without modifying the index."""
        if not self.metadata_store:
            return {
                "total_chunks": 0,
                "parseable_count": 0,
                "invalid_count": 0,
                "duplicate_question_count": 0,
                "details": [],
            }

        details = []
        seen_questions: Dict[str, int] = {}
        parseable_count = 0
        invalid_count = 0
        duplicate_count = 0
        question_pattern = re.compile(
            r"(?is)^\s*Question:\s*(.*?)\s*SQL:\s*(.+?)\s*$"
        )

        for chunk in self.metadata_store.get_active_chunks():
            text = str(chunk["text"])
            match = question_pattern.match(text)
            status = "parseable"
            reason = ""
            question = ""
            sql = ""
            if match is None:
                status = "invalid"
                reason = "missing_question_or_sql_marker"
            else:
                question = match.group(1).strip()
                sql = match.group(2).strip()
                if not question:
                    status = "invalid"
                    reason = "empty_question"
                elif not sql.upper().startswith(("SELECT", "WITH")):
                    status = "invalid"
                    reason = "unsupported_or_incomplete_sql"
                elif len(text) >= self.config.MAX_CHUNK_SIZE:
                    status = "invalid"
                    reason = "chunk_at_size_limit"

            canonical_question = ""
            if status == "parseable":
                parseable_count += 1
                canonical_question = "".join(
                    character.lower()
                    for character in unicodedata.normalize("NFKC", question)
                    if not character.isspace()
                    and not unicodedata.category(character).startswith("P")
                )
                seen_questions[canonical_question] = (
                    seen_questions.get(canonical_question, 0) + 1
                )
                if seen_questions[canonical_question] > 1:
                    duplicate_count += 1
            else:
                invalid_count += 1

            if len(details) < max(1, min(int(detail_limit), 500)):
                details.append(
                    {
                        "vector_id": int(chunk["vector_id"]),
                        "status": status,
                        "reason": reason,
                        "question": question,
                        "canonical_question_hash": (
                            hashlib.sha256(
                                canonical_question.encode("utf-8")
                            ).hexdigest()
                            if canonical_question
                            else ""
                        ),
                        "sql_hash": (
                            hashlib.sha256(sql.encode("utf-8")).hexdigest()
                            if sql
                            else ""
                        ),
                    }
                )

        duplicate_groups = sum(
            1 for count in seen_questions.values() if count > 1
        )
        return {
            "total_chunks": len(self.metadata_store.get_active_vector_ids()),
            "parseable_count": parseable_count,
            "invalid_count": invalid_count,
            "duplicate_question_count": duplicate_count,
            "duplicate_question_groups": duplicate_groups,
            "details": details,
        }

    def save(self):
        """保存索引和元数据"""
        with self.lock:
            temp_index_file = f"{self.index_file}.tmp"
            temp_metadata_file = f"{self.metadata_file}.tmp"
            try:
                if not self.dirty:
                    logger.info("没有更改，跳过保存")
                    return

                # 确保数据目录存在
                os.makedirs(self.data_dir, exist_ok=True)
                logger.debug(f"确保数据目录存在: {self.data_dir}")

                # 原子性保存：先保存到临时文件
                # 保存FAISS索引
                faiss.write_index(self.index, temp_index_file)
                with open(temp_index_file, "rb") as index_file:
                    os.fsync(index_file.fileno())

                self._sync_active_metadata_cache()
                self._validate_index_metadata_consistency()
                if self.metadata_store:
                    self.metadata_store.checkpoint()

                # 保存元数据为JSON（兼容 BM25 和旧部署；主元数据在 SQLite）
                metadata = {
                    "id_to_chunk": self.id_to_chunk,
                    "chunk_to_id": self.chunk_to_id,
                    "embedding_dim": self.config.EMBEDDING_DIM,
                    "total_vectors": self.index.ntotal,
                    "metadata_backend": "sqlite",
                    "sqlite_metadata_file": self.sqlite_metadata_file,
                    "lifecycle_metrics": self.metadata_store.get_metrics() if self.metadata_store else {}
                }

                with open(temp_metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                # 原子性重命名
                os.replace(temp_index_file, self.index_file)
                os.replace(temp_metadata_file, self.metadata_file)
                directory_fd = os.open(self.data_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)

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
            self.index = faiss.read_index(self.index_file)

            # JSON 仅用于旧版本迁移和兼容导出；SQLite 是运行时事实源。
            metadata: Dict[str, Any] = {}
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except FileNotFoundError:
                if not self.metadata_store or not self.metadata_store.get_all_vector_ids():
                    if self.index.ntotal > 0:
                        raise IndexMetadataConsistencyError(
                            "FAISS 索引存在，但 JSON 与 SQLite 元数据均不可用"
                        )
            except json.JSONDecodeError:
                if not self.metadata_store or not self.metadata_store.get_all_vector_ids():
                    raise
                logger.warning("兼容 JSON 元数据损坏，继续使用 SQLite 事实源")

            self.id_to_chunk = metadata.get("id_to_chunk", {})
            self.chunk_to_id = metadata.get("chunk_to_id", {})

            # 旧 JSON 元数据迁移到 SQLite。IndexIDMap 新索引会直接以 SQLite 为准。
            if self.metadata_store and self.id_to_chunk:
                legacy_chunks = [
                    (int(vector_id), text)
                    for vector_id, text in self.id_to_chunk.items()
                    if str(vector_id).lstrip("-").isdigit()
                ]
                self.metadata_store.import_active_chunks(legacy_chunks)
            self._sync_active_metadata_cache()
            consistency = self._validate_index_metadata_consistency()

            logger.info(
                f"成功加载索引，包含 {self.index.ntotal} 个向量，"
                f"活动元数据 {consistency['active_metadata_id_count']} 条"
            )

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
                "search_optimization_enabled": hasattr(self, 'advanced_search_index'),
                "metadata_backend": "sqlite" if self.metadata_store else "json",
                "auto_save": bool(self.config.AUTO_SAVE),
                "index_consistency": self._get_index_consistency(),
            }

            hybrid_metrics = dict(self.hybrid_search_metrics)
            search_count = int(hybrid_metrics["search_count"])
            hybrid_metrics["average_latency_ms"] = (
                hybrid_metrics["total_latency_ms"] / search_count
                if search_count
                else 0.0
            )
            stats["hybrid_retrieval"] = hybrid_metrics
            if hasattr(self, "enhancement_coordinator"):
                stats["bm25"] = self.enhancement_coordinator.get_bm25_stats()

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

            if self.metadata_store:
                stats.update(self.metadata_store.get_metrics())

            return stats

    def compact_index(self) -> Dict[str, Any]:
        """Rebuild FAISS index from active SQLite metadata.

        Compaction removes inactive vectors physically. It requires FAISS to
        reconstruct vectors from the current index, which is available for Flat
        and many IDMap-backed indexes. If the active index type cannot
        reconstruct vectors, return a clear error instead of corrupting data.
        """
        with self.lock:
            if not self.metadata_store:
                raise HTTPException(status_code=500, detail="SQLite 元数据未初始化，无法压缩索引")

            active_chunks = self.metadata_store.get_active_chunks()
            old_total = self.index.ntotal if self.index else 0
            if not active_chunks:
                self.index = self._create_index()
                self.metadata_store.replace_with_active_chunks([])
                self._sync_active_metadata_cache()
                self.dirty = True
                return {
                    "message": "索引已压缩为空",
                    "old_total_vectors": old_total,
                    "new_total_vectors": 0,
                    "active_chunks": 0,
                    "deleted_chunks_removed": old_total,
                }

            vectors = []
            new_chunks = []
            for chunk in active_chunks:
                old_vector_id = int(chunk["vector_id"])
                try:
                    vector = self.index.reconstruct(old_vector_id)
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "当前 FAISS 索引不支持 reconstruct，无法在线 compaction；"
                            "请从源文档重新 embedding 后全量重建。"
                        ),
                    ) from e
                vectors.append(vector)
                new_chunks.append((old_vector_id, chunk["text"]))

            compacted_index = self._create_index()
            embeddings = np.asarray(vectors, dtype=np.float32)
            self.index = compacted_index
            self._add_embeddings_with_ids(embeddings, [vector_id for vector_id, _ in new_chunks])
            self.metadata_store.replace_with_active_chunks(new_chunks)
            self._sync_active_metadata_cache()
            self.dirty = True

            return {
                "message": "索引压缩完成",
                "old_total_vectors": old_total,
                "new_total_vectors": self.index.ntotal,
                "active_chunks": len(new_chunks),
                "deleted_chunks_removed": max(old_total - len(new_chunks), 0),
                "lifecycle_metrics": self.metadata_store.get_metrics(),
            }

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


class VectorDBManager:
    """多实例数据库管理器，支持缓存和生命周期管理"""

    def __init__(self, config: Config, max_instances: int = 10):
        self.config = config
        self.max_instances = max_instances
        self.instances: Dict[str, FaissVectorDB] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()
        logger.info(f"VectorDBManager 初始化完成，max_instances={max_instances}")

    def get_instance(self, businesstype: str = None) -> FaissVectorDB:
        """获取或创建指定业务类型的数据库实例"""
        businesstype = businesstype or self.config.DEFAULT_BUSINESSTYPE
        businesstype = self.config._validate_businesstype(businesstype)

        with self.lock:
            # 更新访问时间
            self.access_times[businesstype] = time.time()

            # 如果实例已存在，直接返回
            if businesstype in self.instances:
                logger.debug(f"使用缓存的实例: '{businesstype}'")
                return self.instances[businesstype]

            # 检查是否达到容量上限
            if len(self.instances) >= self.max_instances:
                self._evict_lru()

            # 创建新实例
            logger.info(f"为业务类型 '{businesstype}' 创建新数据库实例")
            instance = FaissVectorDB(self.config, businesstype)
            self.instances[businesstype] = instance
            return instance

    def _evict_lru(self):
        """淘汰最久未使用的实例"""
        if not self.access_times:
            return

        lru_businesstype = min(self.access_times.items(), key=lambda x: x[1])[0]
        logger.info(f"淘汰 LRU 实例: '{lru_businesstype}'")

        # 保存被淘汰的实例
        instance = self.instances.pop(lru_businesstype, None)
        if instance and instance.dirty:
            logger.info(f"淘汰前保存未保存的实例: {lru_businesstype}")
            try:
                instance.save()
            except Exception as e:
                logger.error(f"淘汰过程中保存失败: {e}")

        self.access_times.pop(lru_businesstype, None)

    def save_all(self):
        """保存所有实例"""
        with self.lock:
            for businesstype, instance in self.instances.items():
                if instance.dirty:
                    logger.info(f"保存实例: '{businesstype}'")
                    try:
                        instance.save()
                    except Exception as e:
                        logger.error(f"保存实例 '{businesstype}' 失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        with self.lock:
            return {
                "total_instances": len(self.instances),
                "max_instances": self.max_instances,
                "active_business_types": list(self.instances.keys()),
                "instances_with_unsaved_changes": sum(
                    1 for inst in self.instances.values() if inst.dirty
                )
            }


# 全局向量数据库实例
config = get_config()
vector_db_manager = None


def _is_sensitive_businesstype(businesstype: Optional[str]) -> bool:
    resolved = config._validate_businesstype(businesstype or config.DEFAULT_BUSINESSTYPE)
    return resolved in SENSITIVE_BUSINESSTYPES


def _get_sensitive_access_key(businesstype: str) -> Optional[str]:
    env_key = os.getenv(ACCESS_KEY_ENV)
    if env_key:
        return env_key.strip()

    key_file = Path(config.DATA_DIR) / businesstype / ACCESS_KEY_FILENAME
    try:
        if key_file.exists():
            return key_file.read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.error(f"读取敏感知识库访问密钥失败: {e}")

    return None


def _validate_sensitive_access(businesstype: Optional[str], access_key: Optional[str]) -> Optional[Dict[str, Any]]:
    resolved = config._validate_businesstype(businesstype or config.DEFAULT_BUSINESSTYPE)
    if resolved not in SENSITIVE_BUSINESSTYPES:
        return None

    expected_key = _get_sensitive_access_key(resolved)
    if expected_key and access_key == expected_key:
        return None

    return {
        "relevant_chunks": [],
        "detailed_results": [],
        "total_found": 0,
        "access_granted": False,
        "message": "账号配置知识库包含敏感信息，请提供正确的 access_key 后重试",
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global vector_db_manager
    try:
        # 启动时初始化管理器
        logger.info("正在初始化向量数据库管理器...")
        max_instances = int(os.getenv("MAX_DB_INSTANCES", "10"))
        vector_db_manager = VectorDBManager(config, max_instances)
        logger.info(f"向量数据库管理器初始化完成 (max_instances={max_instances})")
        yield
    finally:
        # 关闭时保存所有实例
        if vector_db_manager:
            logger.info("正在保存所有数据库实例...")
            vector_db_manager.save_all()
            logger.info("所有数据库实例已关闭")

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
    businesstype: Optional[str] = Field(None, description="业务类型标识符", pattern="^[a-zA-Z0-9_-]{1,50}$")
    chunk_size: int = Field(default=500, description="分块大小", ge=50, le=100000)
    chunk_overlap: int = Field(default=50, description="分块重叠", ge=0, le=10000)

    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v, info):
        if info.data.get('chunk_size') and v >= info.data['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v

class Query(BaseModel):
    question: str = Field(..., description="查询问题", max_length=1000)
    businesstype: Optional[str] = Field(None, description="业务类型标识符", pattern="^[a-zA-Z0-9_-]{1,50}$")
    top_k: int = Field(default=3, description="返回结果数量", ge=1, le=50)
    use_optimization: bool = Field(default=True, description="是否使用搜索优化")
    access_key: Optional[str] = Field(None, description="敏感知识库访问密钥", max_length=200)

class BenchmarkQueries(BaseModel):
    queries: List[str] = Field(..., description="测试查询列表", max_length=20)

class BatchTexts(BaseModel):
    texts: List[str] = Field(..., description="批量文本列表", max_length=100)
    businesstype: Optional[str] = Field(None, description="业务类型标识符", pattern="^[a-zA-Z0-9_-]{1,50}$")


class TextUpdate(BaseModel):
    """单个文本更新请求"""
    old_text: str = Field(..., description="原始文本内容", min_length=1, max_length=50000)
    new_text: str = Field(..., description="新文本内容", min_length=1, max_length=50000)


class UpdateRequest(BaseModel):
    """批量更新文档请求"""
    updates: List[TextUpdate] = Field(..., description="更新列表", min_length=1, max_length=100)
    businesstype: Optional[str] = Field(None, description="业务类型标识符", pattern="^[a-zA-Z0-9_-]{1,50}$")


class CompactRequest(BaseModel):
    """索引压缩请求"""
    businesstype: Optional[str] = Field(None, description="业务类型标识符", pattern="^[a-zA-Z0-9_-]{1,50}$")


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
            name="search",
            description="在向量数据库中搜索相关知识。支持语义搜索。",
            inputSchema={
                "type": "object",
                "properties": {
                    "businesstype": {
                        "type": "string",
                        "description": "业务类型标识符（可选，默认使用环境变量配置）"
                    },
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
                    "access_key": {
                        "type": "string",
                        "description": "敏感知识库访问密钥（查询 account_config 时必填）"
                    }
                },
                "required": ["query"]
            }
        ),
        MCPTool(
            name="add",
            description="将文档添加到向量数据库。文档会自动分块并生成向量索引。",
            inputSchema={
                "type": "object",
                "properties": {
                    "businesstype": {
                        "type": "string",
                        "description": "业务类型标识符（可选，默认使用环境变量配置）"
                    },
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
            name="delete",
            description="从向量数据库中删除指定文档。需要提供与添加时相同的内容和分块参数。",
            inputSchema={
                "type": "object",
                "properties": {
                    "businesstype": {
                        "type": "string",
                        "description": "业务类型标识符（可选，默认使用环境变量配置）"
                    },
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
            name="update",
            description="批量更新知识块。使用软删除旧块并追加新向量，适合百万级索引。",
            inputSchema={
                "type": "object",
                "properties": {
                    "businesstype": {
                        "type": "string",
                        "description": "业务类型标识符（可选，默认使用环境变量配置）"
                    },
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
                    }
                },
                "required": ["updates"]
            }
        ),
        MCPTool(
            name="template_upsert",
            description=(
                "按 external_id 幂等写入结构化 SQL 模板。"
                "仅 search_text 进入向量索引，SQL 和槽位保存在 SQLite payload。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "businesstype": {"type": "string"},
                    "record": {"type": "object"},
                },
                "required": ["record"],
            },
        ),
        MCPTool(
            name="template_search",
            description=(
                "按 dataset、模板 key 和混合检索候选搜索 active SQL 模板，"
                "返回结构化 payload、分数和策略。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "businesstype": {"type": "string"},
                    "dataset_id": {"type": "string"},
                    "query": {"type": "string", "default": ""},
                    "intent_key": {"type": "string"},
                    "canonical_template": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["dataset_id"],
            },
        ),
        MCPTool(
            name="template_list",
            description=(
                "列出 SQL 模板及其状态、复用次数、最近使用时间和结构化内容。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "businesstype": {"type": "string"},
                    "dataset_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": sorted(SQL_TEMPLATE_STATUSES),
                    },
                    "limit": {
                        "type": "integer",
                        "default": 200,
                        "minimum": 1,
                        "maximum": 500,
                    },
                },
            },
        ),
        MCPTool(
            name="template_status",
            description="启用、停用或标记待复核/失效 SQL 模板。",
            inputSchema={
                "type": "object",
                "properties": {
                    "businesstype": {"type": "string"},
                    "external_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": sorted(SQL_TEMPLATE_STATUSES),
                    },
                },
                "required": ["external_id", "status"],
            },
        ),
        MCPTool(
            name="template_delete",
            description="按 external_id 永久删除 SQL 模板及其检索向量。",
            inputSchema={
                "type": "object",
                "properties": {
                    "businesstype": {"type": "string"},
                    "external_id": {"type": "string"},
                },
                "required": ["external_id"],
            },
        ),
        MCPTool(
            name="template_outcome",
            description="记录 SQL 模板复用、shadow、校验失败或成功证据。",
            inputSchema={
                "type": "object",
                "properties": {
                    "businesstype": {"type": "string"},
                    "external_id": {"type": "string"},
                    "outcome": {
                        "type": "string",
                        "enum": [
                            "reuse",
                            "shadow_match",
                            "validation_failure",
                            "success",
                        ],
                    },
                },
                "required": ["external_id", "outcome"],
            },
        ),
        MCPTool(
            name="template_stats",
            description="获取 SQL 模板状态与复用效果统计。",
            inputSchema={
                "type": "object",
                "properties": {"businesstype": {"type": "string"}},
            },
        ),
        MCPTool(
            name="template_migrate",
            description=(
                "批量校验或迁移已审核的结构化 SQL 模板。"
                "默认 dry_run=true，不写入索引。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "businesstype": {"type": "string"},
                    "records": {
                        "type": "array",
                        "items": {"type": "object"},
                        "maxItems": 100,
                    },
                    "dry_run": {"type": "boolean", "default": True},
                },
                "required": ["records"],
            },
        ),
        MCPTool(
            name="sql_document_audit",
            description=(
                "只读审计现有 Question/SQL 文档的完整性和规范化问题重复情况。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "businesstype": {"type": "string"},
                    "detail_limit": {
                        "type": "integer",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 500,
                    },
                },
            },
        ),
        MCPTool(
            name="compact",
            description="压缩索引，物理移除软删除向量。建议 deleted_ratio 超过 0.3 时执行。",
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
        MCPTool(
            name="stats",
            description="获取向量数据库的统计信息，包括向量数量、索引类型等。",
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


async def execute_mcp_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """执行 MCP 工具调用 - 返回标准格式"""
    global vector_db_manager

    if vector_db_manager is None:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"error": "向量数据库管理器未初始化"}, ensure_ascii=False)
            }],
            "isError": True
        }

    # 从参数中提取 businesstype
    businesstype = arguments.get("businesstype") or None

    try:
        # 获取适当的数据库实例
        vector_db = vector_db_manager.get_instance(businesstype)

        if name == "search":
            # 搜索知识
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            access_key = arguments.get("access_key")

            if not query:
                raise ValueError("query parameter is required")

            access_error = _validate_sensitive_access(businesstype, access_key)
            if access_error is not None:
                access_error["query"] = query
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(access_error, ensure_ascii=False, indent=2)
                    }]
                }

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

            results = vector_db.search(query, top_k, use_optimization=False)

            response = {
                "relevant_chunks": [result["text"] for result in results],
                "detailed_results": results,
                "query": query,
                "total_found": len(results),
                "search_method": (
                    results[0].get("search_method", "enhanced")
                    if results
                    else "enhanced"
                ),
            }

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, ensure_ascii=False, indent=2)
                }]
            }

        elif name == "add":
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
                "chunks_added": len(chunks)
            }

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, ensure_ascii=False, indent=2)
                }]
            }

        elif name == "delete":
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

        elif name == "update":
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

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, ensure_ascii=False, indent=2)
                }]
            }

        elif name == "template_upsert":
            record = arguments.get("record")
            if not isinstance(record, dict):
                raise ValueError("record parameter is required")
            response = vector_db.upsert_sql_template(record)
            if config.AUTO_SAVE and vector_db.dirty:
                vector_db.save()
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, ensure_ascii=False, indent=2),
                }]
            }

        elif name == "template_search":
            dataset_id = arguments.get("dataset_id")
            if not dataset_id:
                raise ValueError("dataset_id parameter is required")
            matches = vector_db.search_sql_templates(
                dataset_id=str(dataset_id),
                query=str(arguments.get("query", "")),
                top_k=int(arguments.get("top_k", 10)),
                intent_key=arguments.get("intent_key"),
                canonical_template=arguments.get("canonical_template"),
            )
            scores = [float(match.get("rerank_score", 0.0)) for match in matches]
            top1_score = scores[0] if scores else 0.0
            top2_score = scores[1] if len(scores) > 1 else 0.0
            response = {
                "templates": matches,
                "total_found": len(matches),
                "top1_score": top1_score,
                "top2_score": top2_score,
                "margin": top1_score - top2_score,
            }
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, ensure_ascii=False, indent=2),
                }]
            }

        elif name == "template_list":
            templates = vector_db.list_sql_templates(
                dataset_id=arguments.get("dataset_id"),
                status=arguments.get("status"),
                limit=int(arguments.get("limit", 200)),
            )
            response = {
                "templates": templates,
                "total_found": len(templates),
            }
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, ensure_ascii=False, indent=2),
                }]
            }

        elif name == "template_status":
            external_id = str(arguments.get("external_id", "")).strip()
            status = str(arguments.get("status", "")).strip()
            if not external_id or not status:
                raise ValueError("external_id and status are required")
            template = vector_db.set_sql_template_status(external_id, status)
            if template is None:
                raise ValueError(f"SQL template not found: {external_id}")
            if config.AUTO_SAVE and vector_db.dirty:
                vector_db.save()
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(
                        {"template": template},
                        ensure_ascii=False,
                        indent=2,
                    ),
                }]
            }

        elif name == "template_delete":
            external_id = str(arguments.get("external_id", "")).strip()
            if not external_id:
                raise ValueError("external_id is required")
            template = vector_db.delete_sql_template(external_id)
            if template is None:
                raise ValueError(f"SQL template not found: {external_id}")
            if config.AUTO_SAVE and vector_db.dirty:
                vector_db.save()
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(
                        {
                            "external_id": external_id,
                            "status": "deleted",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                }]
            }

        elif name == "template_outcome":
            external_id = str(arguments.get("external_id", "")).strip()
            outcome = str(arguments.get("outcome", "")).strip()
            if not external_id or not outcome:
                raise ValueError("external_id and outcome are required")
            template = vector_db.record_sql_template_outcome(
                external_id,
                outcome,
            )
            if template is None:
                raise ValueError(f"SQL template not found: {external_id}")
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(
                        {"template": template},
                        ensure_ascii=False,
                        indent=2,
                    ),
                }]
            }

        elif name == "template_stats":
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(
                        vector_db.get_sql_template_metrics(),
                        ensure_ascii=False,
                        indent=2,
                    ),
                }]
            }

        elif name == "template_migrate":
            records = arguments.get("records")
            if not isinstance(records, list):
                raise ValueError("records parameter is required")
            response = vector_db.migrate_sql_templates(
                records,
                dry_run=bool(arguments.get("dry_run", True)),
            )
            if config.AUTO_SAVE and response["applied_count"] > 0:
                vector_db.save()
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, ensure_ascii=False, indent=2),
                }]
            }

        elif name == "sql_document_audit":
            response = vector_db.audit_sql_documents(
                int(arguments.get("detail_limit", 100))
            )
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, ensure_ascii=False, indent=2),
                }]
            }

        elif name == "compact":
            response = vector_db.compact_index()
            if config.AUTO_SAVE:
                vector_db.save()

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, ensure_ascii=False, indent=2)
                }]
            }

        elif name == "stats":
            # 获取统计信息
            stats = vector_db.get_stats()

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(stats, ensure_ascii=False, indent=2)
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
@app.post("/v1/mcp", tags=["MCP"])  # 支持 /v1/mcp 路径
async def mcp_jsonrpc_endpoint(request: JSONRPCRequest):
    """
    MCP 标准 JSON-RPC 2.0 端点

    支持的路径:
    - POST /mcp (标准路径)
    - POST /v1/mcp (版本化路径)

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
        if not vector_db_manager:
            raise HTTPException(status_code=503, detail="向量数据库管理器未初始化")

        # 获取适当的数据库实例
        vector_db = vector_db_manager.get_instance(doc.businesstype)

        # 文本分块
        chunks = vector_db._generate_chunks(doc.content, doc.chunk_size, doc.chunk_overlap)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="文档内容过短，无法生成有效分块")
        
        # 添加到索引
        ids = vector_db.add_texts(chunks)
        
        # 后台保存（如果启用自动保存）
        if config.AUTO_SAVE:
            vector_db.save()
        
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

@app.delete("/delete", summary="删除文档", tags=["传统API"])
async def delete_document(doc: Document, background_tasks: BackgroundTasks):
    """精确删除文档"""
    try:
        if not vector_db_manager:
            raise HTTPException(status_code=503, detail="向量数据库管理器未初始化")

        # 获取适当的数据库实例
        vector_db = vector_db_manager.get_instance(doc.businesstype)
        
        # 重新生成相同的分块来精确匹配
        chunks = vector_db._generate_chunks(doc.content, doc.chunk_size, doc.chunk_overlap)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="文档内容过短，无法生成有效分块")
        
        deleted_count = vector_db.delete_texts(chunks)
        
        if config.AUTO_SAVE and deleted_count > 0:
            vector_db.save()
        
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

@app.put("/update", summary="更新文档", tags=["传统API"])
async def update_documents(request: UpdateRequest, background_tasks: BackgroundTasks):
    """
    批量更新文档（upsert 语义）

    - 如果 old_text 存在，则更新为新文本
    - 如果 old_text 不存在，则插入新文本
    - 支持批量操作，最多100条
    - 返回详细的统计信息
    """
    try:
        if not vector_db_manager:
            raise HTTPException(status_code=503, detail="向量数据库管理器未初始化")

        # 获取适当的数据库实例
        vector_db = vector_db_manager.get_instance(request.businesstype)

        # 准备更新数据
        updates = [
            {"old_text": u.old_text, "new_text": u.new_text}
            for u in request.updates
        ]

        # 执行更新
        result = vector_db.update_texts(updates)

        # 后台保存（如果启用自动保存）
        if config.AUTO_SAVE and result["success_count"] > 0:
            vector_db.save()

        return {
            "message": f"更新完成: 成功 {result['success_count']}, 新增 {result['inserted_count']}, 更新 {result['updated_count']}, 失败 {result['failed_count']}",
            "total_vectors": vector_db.index.ntotal,
            **result,
            "lifecycle_metrics": vector_db.metadata_store.get_metrics() if vector_db.metadata_store else {}
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@app.post("/compact", summary="压缩索引", tags=["传统API"])
async def compact_index(request: CompactRequest, background_tasks: BackgroundTasks):
    """压缩索引，物理移除软删除向量"""
    try:
        if not vector_db_manager:
            raise HTTPException(status_code=503, detail="向量数据库管理器未初始化")

        vector_db = vector_db_manager.get_instance(request.businesstype)
        result = vector_db.compact_index()

        if config.AUTO_SAVE:
            vector_db.save()

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"压缩索引失败: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@app.post("/search", summary="搜索知识", tags=["传统API"])
async def search_knowledge(query: Query):
    """搜索相关知识"""
    try:
        if not vector_db_manager:
            raise HTTPException(status_code=503, detail="向量数据库管理器未初始化")

        access_error = _validate_sensitive_access(query.businesstype, query.access_key)
        if access_error is not None:
            access_error.update({
                "query": query.question,
                "search_method": "access_control",
                "optimization_enabled": query.use_optimization,
            })
            return access_error

        # 获取适当的数据库实例
        vector_db = vector_db_manager.get_instance(query.businesstype)

        if vector_db.index.ntotal == 0:
            return {"relevant_chunks": [], "message": "知识库为空，请先添加文档"}

        results = vector_db.search(query.question, query.top_k, query.use_optimization)

        # 添加搜索方法信息
        search_method = (
            results[0].get("search_method")
            if results
            else (
                "optimized"
                if query.use_optimization and hasattr(vector_db, 'advanced_search_index')
                else "traditional"
            )
        )

        return {
            "relevant_chunks": [result["text"] for result in results],
            "detailed_results": results,
            "query": query.question,
            "total_found": len(results),
            "access_granted": True,
            "search_method": search_method,
            "optimization_enabled": query.use_optimization
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/stats", summary="获取统计信息", tags=["传统API"])
async def get_stats(businesstype: Optional[str] = None):
    """获取数据库统计信息"""
    try:
        if not vector_db_manager:
            raise HTTPException(status_code=503, detail="向量数据库管理器未初始化")

        if businesstype:
            # 获取特定实例的统计信息
            vector_db = vector_db_manager.get_instance(businesstype)
            return {
                "businesstype": businesstype,
                **vector_db.get_stats()
            }
        else:
            # 获取管理器的统计信息
            return vector_db_manager.get_stats()

    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/health", summary="健康检查", tags=["系统"])
async def health_check():
    """健康检查端点"""
    manager_stats = vector_db_manager.get_stats() if vector_db_manager else {}
    return {
        "status": "healthy" if vector_db_manager else "initializing",
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
            "multi_business_type": True,
            "manager_stats": manager_stats
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
