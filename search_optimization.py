#!/usr/bin/env python3
"""
搜索精度优化模块
提供多种策略来提升FAISS向量搜索的精度
"""

import faiss
import numpy as np
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import re
from collections import namedtuple

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """搜索结果封装"""
    text: str
    score: float
    faiss_id: int
    relevance_score: float = 0.0
    diversity_rank: int = 0

@dataclass
class QualityMetrics:
    """搜索质量指标"""
    avg_relevance_score: float
    diversity_score: float
    coverage_ratio: float
    precision_at_k: float

class SemanticChunker:
    """语义感知的文本分块器"""

    def __init__(self, model: SentenceTransformer,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 800,
                 similarity_threshold: float = 0.85):
        self.model = model
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold

    def _split_sentences(self, text: str) -> List[str]:
        """句子分割"""
        # 支持中英文的句子分割
        sentence_endings = re.compile(r'[.!?。！？]+[\s\n]+')
        sentences = sentence_endings.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _semantic_chunk(self, sentences: List[str]) -> List[str]:
        """基于语义相似度的分块"""
        if len(sentences) <= 1:
            return sentences

        chunks = []
        current_chunk = []
        current_length = 0

        # 获取句子向量
        sentence_vectors = self.model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)

            # 如果当前块为空，直接添加
            if not current_chunk:
                current_chunk.append(sentence)
                current_length = sentence_length
                continue

            # 检查长度限制
            if current_length + sentence_length > self.max_chunk_size:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
                continue

            # 计算语义相似度
            if i > 0:
                similarity = np.dot(sentence_vectors[i], sentence_vectors[i-1])

                # 如果相似度过低，开始新的分块
                if similarity < self.similarity_threshold and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                    continue

            current_chunk.append(sentence)
            current_length += sentence_length

        # 添加最后一个分块
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(' '.join(current_chunk))

        return chunks

    def chunk_text(self, text: str) -> List[str]:
        """智能文本分块"""
        sentences = self._split_sentences(text)

        if len(sentences) <= 3:  # 短文本不进行分块
            return [text] if len(text) >= self.min_chunk_size else []

        return self._semantic_chunk(sentences)

class SearchQualityOptimizer:
    """搜索质量优化器"""

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def calculate_relevance_scores(self, query: str, results: List[SearchResult]) -> List[float]:
        """计算相关性得分"""
        if not results:
            return []

        # 获取查询向量
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        # 获取结果文本向量
        result_texts = [result.text for result in results]
        result_vectors = self.model.encode(
            result_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 计算余弦相似度
        relevance_scores = []
        for i, result_vector in enumerate(result_vectors):
            similarity = np.dot(query_vector, result_vector)
            relevance_scores.append(similarity)

        return relevance_scores

    def calculate_diversity_score(self, results: List[SearchResult]) -> float:
        """计算结果多样性得分"""
        if len(results) <= 1:
            return 0.0

        texts = [result.text for result in results]
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 计算两两之间的平均距离
        total_distance = 0
        count = 0

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                distance = 1 - np.dot(vectors[i], vectors[j])  # 转换为距离
                total_distance += distance
                count += 1

        return total_distance / count if count > 0 else 0.0

    def rerank_by_diversity(self, results: List[SearchResult],
                          diversity_weight: float = 0.3) -> List[SearchResult]:
        """基于多样性重新排序"""
        if len(results) <= 1:
            return results

        # 计算每对结果之间的距离
        texts = [result.text for result in results]
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 建立faiss_id到结果索引的映射
        faiss_id_to_result_idx = {result.faiss_id: i for i, result in enumerate(results)}

        # 使用最大边际相关性(MMR)算法
        reranked = []
        remaining_indices = list(range(len(results)))

        # 选择最好的第一个结果
        first_idx = np.argmax([result.score for result in results])
        reranked.append(results[first_idx])
        remaining_indices.remove(first_idx)

        while remaining_indices:
            best_score = float('-inf')
            best_idx = -1

            for idx in remaining_indices:
                # 原始相关性得分
                relevance = results[idx].score

                # 与已选择结果的最大相似度
                max_similarity = 0
                for selected in reranked:
                    # 使用正确的结果索引，而不是faiss_id
                    selected_idx = faiss_id_to_result_idx[selected.faiss_id]
                    similarity = np.dot(vectors[idx], vectors[selected_idx])
                    max_similarity = max(max_similarity, similarity)

                # MMR得分
                mmr_score = (1 - diversity_weight) * relevance - diversity_weight * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            reranked.append(results[best_idx])
            remaining_indices.remove(best_idx)

        return reranked

class AdvancedSearchIndex:
    """高级搜索索引类"""

    def __init__(self, config):
        self.config = config
        self.embedding_model = SentenceTransformer(config.MODEL_NAME)
        self.semantic_chunker = SemanticChunker(
            self.embedding_model,
            min_chunk_size=config.MIN_CHUNK_SIZE,
            max_chunk_size=config.MAX_CHUNK_SIZE
        )
        self.quality_optimizer = SearchQualityOptimizer(self.embedding_model)
        self.index = None
        self.id_to_chunk = {}
        self.chunk_to_id = {}

        # 搜索配置
        self.search_config = {
            'ivf_nprobe': min(config.NPROBE, config.NLIST // 2),  # 动态调整nprobe
            'hnsw_ef_search': max(config.EF_SEARCH, 100),  # 增加搜索精度
            'diversity_weight': 0.1,  # 多样性权重（降低以保持相关性优先）
            'relevance_threshold': 0.1  # 相关性阈值（降低以避免过度过滤）
        }

    def _create_optimized_index(self):
        """创建优化的索引"""
        if self.config.INDEX_TYPE == "IVFFlat":
            # 优化的IVF索引
            nlist = self.config.NLIST
            quantizer = faiss.IndexFlatIP(self.config.EMBEDDING_DIM)
            index = faiss.IndexIVFFlat(quantizer, self.config.EMBEDDING_DIM, nlist)

            # 设置优化的训练参数
            index.nlist = nlist
            return index

        elif self.config.INDEX_TYPE == "HNSW":
            # 优化的HNSW索引
            M = self.config.M
            ef_construction = self.config.EF_CONSTRUCTION
            index = faiss.IndexHNSWFlat(self.config.EMBEDDING_DIM, M)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = self.search_config['hnsw_ef_search']

            return index

        else:
            # 回退到精确搜索
            return faiss.IndexFlatIP(self.config.EMBEDDING_DIM)

    def optimized_search(self, query: str, top_k: int = 10,
                        use_reranking: bool = True) -> List[Dict[str, Any]]:
        """优化的搜索方法"""
        if not self.index or self.index.ntotal == 0:
            return []

        # 生成查询向量
        query_vector = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 动态调整搜索参数
        search_k = min(top_k * 3, self.index.ntotal)  # 搜索更多候选

        # 设置索引特定的搜索参数
        if hasattr(self.index, 'nprobe'):  # IVF索引
            self.index.nprobe = self.search_config['ivf_nprobe']
        elif hasattr(self.index, 'hnsw'):  # HNSW索引
            self.index.hnsw.efSearch = self.search_config['hnsw_ef_search']

        # 执行搜索
        distances, indices = self.index.search(query_vector, search_k)

        # 构建结果对象
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and str(idx) in self.id_to_chunk:
                result = SearchResult(
                    text=self.id_to_chunk[str(idx)],
                    score=float(dist),
                    faiss_id=int(idx)
                )
                results.append(result)

        # 计算相关性得分
        relevance_scores = self.quality_optimizer.calculate_relevance_scores(query, results)
        for i, score in enumerate(relevance_scores):
            results[i].relevance_score = score

        # 自适应阈值：如果高质量结果不够，放宽阈值
        threshold = self.search_config['relevance_threshold']
        filtered_results = [r for r in results if r.relevance_score >= threshold]

        # 如果过滤后结果太少，放宽阈值
        if len(filtered_results) < top_k:
            # 使用更低的阈值
            relaxed_threshold = max(0.05, threshold * 0.5)
            filtered_results = [r for r in results if r.relevance_score >= relaxed_threshold]

        # 如果结果仍然太少，使用所有结果按相关性排序
        if len(filtered_results) < top_k:
            # 按相关性分数排序所有结果
            all_results_sorted = sorted(results, key=lambda x: x.relevance_score, reverse=True)
            filtered_results = all_results_sorted[:max(top_k, len(all_results_sorted))]

        # 重新排序（仅在结果充足时进行多样性重排）
        if use_reranking and len(filtered_results) > top_k // 2:
            filtered_results = self.quality_optimizer.rerank_by_diversity(
                filtered_results,
                self.search_config['diversity_weight']
            )

        # 返回前top_k个结果
        final_results = filtered_results[:top_k]

        # 计算质量指标
        quality_metrics = self._calculate_quality_metrics(query, final_results)

        # 转换为字典格式
        return [
            {
                "text": result.text,
                "similarity_score": float(result.score),
                "relevance_score": float(result.relevance_score),
                "faiss_id": int(result.faiss_id),
                "quality_metrics": quality_metrics.__dict__ if i == 0 else None
            }
            for i, result in enumerate(final_results)
        ]

    def _calculate_quality_metrics(self, query: str, results: List[SearchResult]) -> QualityMetrics:
        """计算搜索质量指标"""
        if not results:
            return QualityMetrics(0.0, 0.0, 0.0, 0.0)

        # 平均相关性得分
        relevance_scores = [r.relevance_score for r in results]
        avg_relevance = np.mean(relevance_scores)

        # 多样性得分
        diversity_score = self.quality_optimizer.calculate_diversity_score(results)

        # 覆盖率（高分结果比例）
        high_score_count = sum(1 for r in results if r.relevance_score > 0.7)
        coverage_ratio = high_score_count / len(results)

        # Precision@k（假设阈值为0.6）
        precision_at_k = sum(1 for r in results if r.relevance_score > 0.6) / len(results)

        return QualityMetrics(
            avg_relevance_score=float(avg_relevance),
            diversity_score=float(diversity_score),
            coverage_ratio=float(coverage_ratio),
            precision_at_k=float(precision_at_k)
        )

    def get_search_recommendations(self) -> Dict[str, Any]:
        """获取搜索优化建议"""
        total_vectors = self.index.ntotal if self.index else 0

        recommendations = {
            "current_config": self.search_config.copy(),
            "recommendations": []
        }

        # 基于数据量的建议
        if total_vectors < 1000:
            recommendations["recommendations"].append(
                "数据量较小，建议使用FlatIP索引获得最佳精度"
            )
            recommendations["suggested_index_type"] = "FlatIP"
        elif total_vectors < 100000:
            recommendations["recommendations"].append(
                "数据量中等，建议使用HNSW索引平衡精度和性能"
            )
            recommendations["suggested_index_type"] = "HNSW"
            recommendations["suggested_hnsw_params"] = {"M": 32, "efConstruction": 200}
        else:
            recommendations["recommendations"].append(
                "数据量较大，建议使用IVFFlat索引"
            )
            recommendations["suggested_index_type"] = "IVFFlat"
            recommendations["suggested_ivf_params"] = {"nlist": min(1000, total_vectors // 10)}

        # 搜索参数建议
        if hasattr(self.index, 'nprobe'):
            optimal_nprobe = min(self.config.NLIST // 4, 20)
            recommendations["recommendations"].append(
                f"建议将IVF nprobe设置为 {optimal_nprobe} 以提升搜索精度"
            )

        if hasattr(self.index, 'hnsw'):
            recommendations["recommendations"].append(
                "建议将HNSW efSearch设置为100-200以提升搜索精度"
            )

        return recommendations

# 使用示例和配置
class OptimizedConfig:
    """优化后的配置示例"""
    INDEX_TYPE = "HNSW"  # 更好的精度性能平衡
    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIM = 384
    NLIST = 100
    NPROBE = 20  # 增加探测数量
    M = 32  # 增加连接数
    EF_CONSTRUCTION = 200
    EF_SEARCH = 100  # 增加搜索深度
    MIN_CHUNK_SIZE = 100
    MAX_CHUNK_SIZE = 800

if __name__ == "__main__":
    # 示例使用
    config = OptimizedConfig()
    optimizer = AdvancedSearchIndex(config)

    print("✅ 搜索精度优化模块已加载")
    print("📊 主要优化特性:")
    print("  - 语义感知文本分块")
    print("  - 动态搜索参数调整")
    print("  - 多样性重排序(MMR)")
    print("  - 搜索质量评估")
    print("  - 智能索引选择建议")