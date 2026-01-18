#!/usr/bin/env python3
"""
Retrieval Enhancement Module
Improves vector search accuracy through hybrid retrieval, adaptive thresholding, and query enhancement
"""

import json
import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Import BM25 search strategy
try:
    from bm25_search import BM25SearchStrategy, BM25SearchResult
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25SearchStrategy = None
    BM25SearchResult = None

logger = logging.getLogger(__name__)


@dataclass
class EnhancedQuery:
    """Enhanced query object with expansion and metadata"""
    original: str
    normalized: str
    expanded_variants: List[str] = field(default_factory=list)
    domain_terms: List[str] = field(default_factory=list)
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {
                'created_at': datetime.now().isoformat(),
                'language': self._detect_language(),
                'length': len(self.original),
                'complexity': 'medium'
            }

    def _detect_language(self) -> str:
        """Detect query language (Chinese, English, or Mixed)"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', self.original))
        english_chars = len(re.findall(r'[a-zA-Z]', self.original))

        if chinese_chars > 0 and english_chars > 0:
            return 'mixed'
        elif chinese_chars > 0:
            return 'zh'
        else:
            return 'en'


@dataclass
class SearchResult:
    """Enhanced search result"""
    text: str
    similarity_score: float  # FAISS distance or BM25 score
    relevance_score: float  # Cosine similarity or normalized score
    rrf_score: float = 0.0  # Reciprocal Rank Fusion score
    composite_score: float = 0.0  # Final ranking score
    faiss_id: int = -1  # -1 for non-vector results (BM25, exact match)
    chunk_id: str = ""  # Chunk ID for BM25 results
    match_type: str = 'vector'  # vector, bm25, exact, hybrid
    strategies_used: List[str] = field(default_factory=list)
    domain_boost: float = 0.0
    rank: int = 0

    @classmethod
    def from_bm25(cls, bm25_result: 'BM25SearchResult') -> 'SearchResult':
        """Create SearchResult from BM25SearchResult"""
        return cls(
            text=bm25_result.text,
            similarity_score=bm25_result.score,
            relevance_score=min(bm25_result.score / 10.0, 1.0),  # Normalize to 0-1
            faiss_id=-1,
            chunk_id=bm25_result.chunk_id,
            match_type='bm25',
            strategies_used=['bm25']
        )


@dataclass
class QualityMetrics:
    """Search quality metrics"""
    avg_relevance_score: float = 0.0
    diversity_score: float = 0.0
    coverage_ratio: float = 0.0
    precision_at_k: float = 0.0
    total_results: int = 0
    strategies_used: List[str] = field(default_factory=list)
    threshold_used: float = 0.1
    threshold_adjustments: List[Dict] = field(default_factory=list)


class DomainTermExpander:
    """Expands queries with domain-specific terminology"""

    def __init__(self, dict_path: str = "config/domain_terms.json"):
        self.dict_path = dict_path
        self.terms_dict = self._load_dictionary()
        logger.info(f"Loaded {len(self.terms_dict)} domain term mappings")

    def _load_dictionary(self) -> Dict[str, List[str]]:
        """Load domain term dictionary from JSON file"""
        try:
            with open(self.dict_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Domain term dict not found at {self.dict_path}, using empty dict")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse domain term dict: {e}")
            return {}

    def expand(self, query: str, max_variants: int = 5) -> List[str]:
        """
        Generate expanded query variants

        Args:
            query: Original query text
            max_variants: Maximum number of expanded variants to generate

        Returns:
            List of expanded query variants
        """
        expansions = [query]

        for term, synonyms in self.terms_dict.items():
            if term in query:
                for synonym in synonyms[:2]:  # Use top 2 synonyms per term
                    expanded = query.replace(term, synonym)
                    if expanded not in expansions:
                        expansions.append(expanded)
                        if len(expansions) >= max_variants:
                            break

            if len(expansions) >= max_variants:
                break

        return expansions

    def detect_domain_terms(self, query: str) -> List[str]:
        """Detect domain-specific terms in query"""
        detected = []
        for term in self.terms_dict.keys():
            if term in query:
                detected.append(term)
        return detected


class QueryNormalizer:
    """Normalizes query text for better matching"""

    @staticmethod
    def normalize(query: str) -> str:
        """Apply normalization rules"""
        # Preserve Chinese characters, process English
        # Lowercase English
        query = query.lower()

        # Remove special characters except Chinese, alphanumeric, spaces
        query = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', query)

        # Normalize whitespace
        query = ' '.join(query.split())

        return query.strip()


class AdaptiveThresholdCalculator:
    """Calculates adaptive relevance thresholds based on query and results"""

    def __init__(self, config):
        self.config = config
        self.base_threshold = float(getattr(config, 'BASE_RELEVANCE_THRESHOLD', 0.1))
        self.min_threshold = float(getattr(config, 'MIN_RELEVANCE_THRESHOLD', 0.05))
        self.max_threshold = float(getattr(config, 'MAX_RELEVANCE_THRESHOLD', 0.3))

    def calculate(self, query: str, relevance_scores: List[float],
                  result_count: int, has_domain_terms: bool = False) -> Tuple[float, List[Dict]]:
        """
        Calculate adaptive threshold based on multiple factors

        Args:
            query: Search query text
            relevance_scores: List of relevance scores from search results
            result_count: Number of results found
            has_domain_terms: Whether query contains domain-specific terms

        Returns:
            Tuple of (calculated_threshold, list_of_adjustments)
        """
        adjustments = []

        # Factor 1: Score distribution
        if relevance_scores:
            avg_score = float(np.mean(relevance_scores))
            std_score = float(np.std(relevance_scores))
            distribution_factor = max(0, avg_score - std_score)
            adjustments.append({
                "factor": "score_distribution",
                "avg_score": avg_score,
                "std_score": std_score,
                "factor_value": distribution_factor
            })
        else:
            distribution_factor = self.base_threshold

        # Factor 2: Domain term detection
        if has_domain_terms:
            domain_factor = 0.7  # 30% reduction for domain queries
            adjustments.append({
                "factor": "domain_terms",
                "reduction": "30%",
                "factor_value": domain_factor
            })
        else:
            domain_factor = 1.0

        # Factor 3: Result count
        if result_count == 0:
            count_factor = 0.5  # Significant reduction when no results
            adjustments.append({
                "factor": "result_count",
                "count": 0,
                "action": "significant_reduction",
                "factor_value": count_factor
            })
        elif result_count < 5:
            count_factor = 0.7  # Moderate reduction for few results
            adjustments.append({
                "factor": "result_count",
                "count": result_count,
                "action": "moderate_reduction",
                "factor_value": count_factor
            })
        else:
            count_factor = 1.0

        # Calculate final threshold
        threshold = distribution_factor * domain_factor * count_factor

        # Clamp to valid range
        final_threshold = max(self.min_threshold, min(threshold, self.max_threshold))

        # Log the adjustment
        if final_threshold != self.base_threshold:
            adjustments.append({
                "factor": "final_threshold",
                "from": self.base_threshold,
                "to": final_threshold,
                "reason": self._get_adjustment_reason(adjustments)
            })

        return final_threshold, adjustments

    def _get_adjustment_reason(self, adjustments: List[Dict]) -> str:
        """Generate human-readable reason for threshold adjustment"""
        reasons = []

        for adj in adjustments:
            factor = adj.get("factor", "")
            if factor == "domain_terms":
                reasons.append("domain_terms_detected")
            elif factor == "result_count":
                action = adj.get("action", "")
                if action == "significant_reduction":
                    reasons.append("no_results_found")
                elif action == "moderate_reduction":
                    reasons.append("low_result_count")
            elif factor == "score_distribution":
                if adj.get("avg_score", 0) < 0.5:
                    reasons.append("low_score_distribution")

        return ", ".join(reasons) if reasons else "adaptive_adjustment"


class ResultFusion:
    """Fuses results from multiple retrieval strategies"""

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        self.rrf_k = 60  # RRF constant

    def fuse(self, results_dict: Dict[str, List[SearchResult]],
             top_k: int) -> List[SearchResult]:
        """
        Fuse results from multiple strategies using Reciprocal Rank Fusion

        Args:
            results_dict: Dict mapping strategy names to result lists
            top_k: Number of final results to return

        Returns:
            Fused and ranked list of search results
        """
        # Calculate RRF scores using unique keys (faiss_id or chunk_id)
        rrf_scores = {}

        for strategy, results in results_dict.items():
            weight = self.weights.get(strategy, 1.0)
            for rank, result in enumerate(results, 1):
                rrf_score = weight / (self.rrf_k + rank)
                # Use faiss_id for vector results, chunk_id for BM25
                result_key = self._get_result_key(result)

                if result_key in rrf_scores:
                    rrf_scores[result_key] += rrf_score
                    # Track which strategies found this result
                    for existing_result in self._find_results_by_key(results_dict.values(), result_key):
                        if strategy not in existing_result.strategies_used:
                            existing_result.strategies_used.append(strategy)
                else:
                    rrf_scores[result_key] = rrf_score
                    result.rrf_score = rrf_score
                    result.strategies_used.append(strategy)

        # Sort by RRF score
        sorted_keys = sorted(rrf_scores.keys(),
                           key=lambda x: rrf_scores[x],
                           reverse=True)

        # Collect unique results
        all_results = []
        seen_keys = set()

        for strategy_results in results_dict.values():
            for result in strategy_results:
                result_key = self._get_result_key(result)
                if result_key not in seen_keys:
                    all_results.append(result)
                    seen_keys.add(result_key)

        # Build final results list
        final_results = []
        for result_key in sorted_keys[:top_k]:
            for result in all_results:
                if self._get_result_key(result) == result_key:
                    # Update RRF score and composite score
                    result.rrf_score = rrf_scores[result_key]
                    result.composite_score = self._calculate_composite_score(result)
                    result.match_type = 'hybrid' if len(result.strategies_used) > 1 else result.strategies_used[0]
                    final_results.append(result)
                    break

        # Assign ranks
        for rank, result in enumerate(final_results, 1):
            result.rank = rank

        return final_results

    def _get_result_key(self, result: SearchResult) -> str:
        """Generate unique key for result (handles both FAISS and BM25)"""
        if result.faiss_id >= 0:
            return f"faiss_{result.faiss_id}"
        else:
            return f"chunk_{result.chunk_id}"

    def _find_results_by_key(self, all_results: List[List[SearchResult]], result_key: str) -> List[SearchResult]:
        """Find all results with given result key"""
        found = []
        for results in all_results:
            for result in results:
                if self._get_result_key(result) == result_key:
                    found.append(result)
        return found

    def _find_results_by_id(self, all_results: List[List[SearchResult]], faiss_id: int) -> List[SearchResult]:
        """Find all results with given faiss_id (legacy method)"""
        found = []
        for results in all_results:
            for result in results:
                if result.faiss_id == faiss_id:
                    found.append(result)
        return found

    def _calculate_composite_score(self, result: SearchResult) -> float:
        """Calculate composite score from multiple signals"""
        # Weighted combination of relevance and RRF scores
        relevance_weight = 0.7
        rrf_weight = 0.3

        composite = (relevance_weight * result.relevance_score) + \
                   (rrf_weight * (result.rrf_score * 10))  # Scale RRF to similar range

        # Apply domain boost
        if result.domain_boost > 0:
            composite *= (1.0 + result.domain_boost)

        return composite


class QualityMetricsCalculator:
    """Calculates quality metrics for search results"""

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def calculate(self, results: List[SearchResult], query: str) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        if not results:
            return QualityMetrics()

        # Extract relevance scores
        relevance_scores = [r.relevance_score for r in results]

        # Average relevance
        avg_relevance = float(np.mean(relevance_scores)) if relevance_scores else 0.0

        # Diversity score (average pairwise distance)
        diversity_score = self._calculate_diversity(results)

        # Coverage ratio (high-quality results proportion)
        high_quality_count = sum(1 for r in results if r.relevance_score > 0.7)
        coverage_ratio = high_quality_count / len(results) if results else 0.0

        # Precision@k (results with relevance > 0.6)
        precision_count = sum(1 for r in results if r.relevance_score > 0.6)
        precision_at_k = precision_count / len(results) if results else 0.0

        return QualityMetrics(
            avg_relevance_score=avg_relevance,
            diversity_score=diversity_score,
            coverage_ratio=coverage_ratio,
            precision_at_k=precision_at_k,
            total_results=len(results),
            strategies_used=list(set(s for r in results for s in r.strategies_used))
        )

    def _calculate_diversity(self, results: List[SearchResult]) -> float:
        """Calculate diversity score based on result embeddings"""
        if len(results) <= 1:
            return 0.0

        texts = [r.text[:200] for r in results]  # Use first 200 chars
        try:
            vectors = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # Calculate average pairwise distance
            total_distance = 0.0
            count = 0

            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    distance = 1 - np.dot(vectors[i], vectors[j])
                    total_distance += distance
                    count += 1

            return total_distance / count if count > 0 else 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate diversity: {e}")
            return 0.0


# Main enhancement coordinator
class RetrievalEnhancementCoordinator:
    """Coordinates all retrieval enhancement components"""

    def __init__(self, config, model: SentenceTransformer, businesstype: str = "default"):
        self.config = config
        self.model = model
        self.businesstype = businesstype

        # Initialize components
        self.domain_expander = DomainTermExpander()
        self.normalizer = QueryNormalizer()
        self.threshold_calculator = AdaptiveThresholdCalculator(config)
        self.result_fusion = ResultFusion({
            'vector': config.VECTOR_SEARCH_WEIGHT if hasattr(config, 'VECTOR_SEARCH_WEIGHT') else 0.7,
            'bm25': config.BM25_SEARCH_WEIGHT if hasattr(config, 'BM25_SEARCH_WEIGHT') else 0.2,
            'exact': config.EXACT_MATCH_WEIGHT if hasattr(config, 'EXACT_MATCH_WEIGHT') else 0.1
        })
        self.metrics_calculator = QualityMetricsCalculator(model)

        # Initialize BM25 search strategy if available
        self.bm25_strategy = None
        enable_bm25 = getattr(config, 'ENABLE_BM25_FALLBACK', True)
        if BM25_AVAILABLE and enable_bm25:
            metadata_file = config.get_metadata_file(businesstype)
            self.bm25_strategy = BM25SearchStrategy(str(metadata_file), businesstype)
            if self.bm25_strategy.is_available():
                logger.info(f"[{businesstype}] BM25 search strategy initialized")
            else:
                logger.warning(f"[{businesstype}] BM25 search strategy not available")
                self.bm25_strategy = None

        logger.info(f"[{businesstype}] Retrieval Enhancement Coordinator initialized")

    def enhance_query(self, query: str) -> EnhancedQuery:
        """Enhance query with expansion and normalization"""
        # Normalize
        normalized = self.normalizer.normalize(query)

        # Detect domain terms
        domain_terms = self.domain_expander.detect_domain_terms(query)

        # Generate expansions
        expanded_variants = self.domain_expander.expand(query, max_variants=5)

        enhanced = EnhancedQuery(
            original=query,
            normalized=normalized,
            expanded_variants=expanded_variants,
            domain_terms=domain_terms
        )

        # Update metadata
        enhanced.metadata['has_domain_terms'] = len(domain_terms) > 0
        enhanced.metadata['complexity'] = self._assess_complexity(query)

        return enhanced

    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        length = len(query)
        has_domain_terms = len(self.domain_expander.detect_domain_terms(query)) > 0

        if length < 10 and not has_domain_terms:
            return 'simple'
        elif length > 50 or has_domain_terms:
            return 'complex'
        else:
            return 'medium'

    def calculate_adaptive_threshold(self, query: str, relevance_scores: List[float],
                                     result_count: int) -> Tuple[float, List[Dict]]:
        """Calculate adaptive relevance threshold"""
        has_domain_terms = len(self.domain_expander.detect_domain_terms(query)) > 0
        return self.threshold_calculator.calculate(query, relevance_scores,
                                                    result_count, has_domain_terms)

    def calculate_quality_metrics(self, results: List[SearchResult], query: str) -> QualityMetrics:
        """Calculate quality metrics for search results"""
        return self.metrics_calculator.calculate(results, query)

    def is_bm25_available(self) -> bool:
        """Check if BM25 search is available"""
        return self.bm25_strategy is not None and self.bm25_strategy.is_available()

    def search_bm25(self, query: str, top_k: int = 5, min_score: float = 0.1) -> List[SearchResult]:
        """
        Perform BM25 keyword search

        Args:
            query: Search query text
            top_k: Maximum number of results to return
            min_score: Minimum BM25 score threshold

        Returns:
            List of SearchResult objects from BM25 search
        """
        if not self.is_bm25_available():
            logger.warning(f"[{self.businesstype}] BM25 search not available")
            return []

        try:
            bm25_results = self.bm25_strategy.search(query, top_k, min_score)
            # Convert BM25SearchResult to SearchResult
            search_results = [SearchResult.from_bm25(r) for r in bm25_results]
            logger.info(f"[{self.businesstype}] BM25 search returned {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"[{self.businesstype}] BM25 search failed: {e}")
            return []

    def get_bm25_stats(self) -> Dict[str, Any]:
        """Get BM25 search statistics"""
        if not self.is_bm25_available():
            return {"available": False}
        return self.bm25_strategy.get_stats()
