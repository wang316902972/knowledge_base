# Design Document: Retrieval Accuracy Improvements

## Overview
This document provides the technical design for improving retrieval accuracy in the FAISS vector database search system through hybrid retrieval, adaptive thresholding, and query enhancement.

## Architecture

### Current Architecture

```
Query → Embedding Model → Vector Search → Relevance Filter → Results
                    ↓
                FAISS Index
```

**Limitations**:
- Single retrieval strategy (vector-only)
- Fixed relevance threshold (0.1)
- No fallback for failed searches
- Limited domain-specific term handling

### Proposed Architecture

```
Query → Query Enhancement → Multi-Strategy Retrieval → Result Fusion → Quality Filter → Results
             ↓                        ↓                        ↓
    - Expansion                  - Vector Search          - Score Merging
    - Normalization              - Keyword Search (BM25)  - Deduplication
    - Augmentation               - Exact Field Match      - Re-ranking
                                                          ↓
                                                    Quality Metrics
```

## Component Design

### 1. Query Enhancement Module

**Purpose**: Improve query quality before retrieval

**Components**:

#### 1.1 Domain Term Expander
```python
class DomainTermExpander:
    """Expands queries with domain-specific terminology"""

    TERM_DICTIONARY = {
        "反作弊": ["anti-cheat", "反外挂", "防作弊", "安全检测"],
        "归因": ["attribution", "溯源", "分析", "追踪"],
        "预制件": ["prefab", "组件", "模块", "plugin"],
        # ... more mappings
    }

    def expand(self, query: str) -> List[str]:
        """Generate expanded queries"""
        expansions = [query]
        for term, synonyms in self.TERM_DICTIONARY.items():
            if term in query:
                for synonym in synonyms:
                    expansions.append(query.replace(term, synonym))
        return expansions
```

#### 1.2 Query Normalizer
```python
class QueryNormalizer:
    """Normalizes query text for better matching"""

    def normalize(self, query: str) -> str:
        """Apply normalization rules"""
        # Lowercase
        query = query.lower()
        # Remove special chars
        query = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', query)
        # Normalize whitespace
        query = ' '.join(query.split())
        return query
```

### 2. Multi-Strategy Retrieval

**Purpose**: Execute multiple retrieval strategies in parallel

#### 2.1 Vector Search Strategy (Primary)
```python
class VectorSearchStrategy:
    """Semantic vector search with adaptive parameters"""

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        # Generate embedding
        query_vector = self.model.encode(query)

        # Adaptive nprobe based on index size
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self._calculate_adaptive_nprobe()

        # Search with candidate multiplier
        candidates = top_k * self.search_config['candidate_multiplier']
        distances, indices = self.index.search(query_vector, candidates)

        # Calculate relevance scores
        results = self._build_results(distances, indices)
        relevance_scores = self._calculate_relevance(query, results)

        # Adaptive threshold
        threshold = self._calculate_adaptive_threshold(relevance_scores)

        return [r for r in results if r.score >= threshold]
```

#### 2.2 BM25 Keyword Search Strategy (Fallback)
```python
class BM25SearchStrategy:
    """BM25 keyword-based search as fallback"""

    def __init__(self, documents: List[str]):
        self.bm25 = BM25Okapi([self._tokenize(doc) for doc in documents])
        self.documents = documents

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(
                text=self.documents[idx],
                score=float(scores[idx]),
                faiss_id=idx
            )
            for idx in top_indices
            if scores[idx] > 0
        ]
```

#### 2.3 Exact Field Match Strategy (Fallback)
```python
class ExactFieldMatchStrategy:
    """Exact matching on critical fields"""

    def __init__(self, metadata: Dict):
        self.metadata = metadata
        self.fields = ['id', 'name', 'tags', 'description']

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        results = []
        query_lower = query.lower()

        for doc_id, doc in self.metadata.items():
            # Check each field
            for field in self.fields:
                if field in doc:
                    value = str(doc[field]).lower()
                    if query_lower in value:
                        score = 1.0 if query_lower == value else 0.8
                        results.append(SearchResult(
                            text=json.dumps(doc, ensure_ascii=False),
                            score=score,
                            faiss_id=int(doc_id)
                        ))
                        break

        # Sort by score and return top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
```

### 3. Result Fusion Module

**Purpose**: Combine results from multiple strategies

```python
class ResultFusion:
    """Fuses results from multiple retrieval strategies"""

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights  # {'vector': 0.7, 'bm25': 0.2, 'exact': 0.1}

    def fuse(self, results_dict: Dict[str, List[SearchResult]],
             top_k: int) -> List[SearchResult]:
        """
        Fuses results using reciprocal rank fusion (RRF)

        Args:
            results_dict: {'vector': [...], 'bm25': [...], 'exact': [...]}
            top_k: final number of results
        """
        # Calculate RRF scores
        rrf_scores = {}

        for strategy, results in results_dict.items():
            weight = self.weights.get(strategy, 0)
            for rank, result in enumerate(results, 1):
                rrf_score = weight / (rank + 60)  # k=60 for RRF
                faiss_id = result.faiss_id

                if faiss_id in rrf_scores:
                    rrf_scores[faiss_id] += rrf_score
                else:
                    rrf_scores[faiss_id] = rrf_score

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(),
                          key=lambda x: rrf_scores[x],
                          reverse=True)

        # Reconstruct results
        all_results = []
        seen_ids = set()
        for strategy_results in results_dict.values():
            for result in strategy_results:
                if result.faiss_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result.faiss_id)

        # Create final results with RRF scores
        final_results = []
        for faiss_id in sorted_ids[:top_k]:
            for result in all_results:
                if result.faiss_id == faiss_id:
                    result.rrf_score = rrf_scores[faiss_id]
                    final_results.append(result)
                    break

        return final_results
```

### 4. Adaptive Threshold Module

```python
class AdaptiveThresholdCalculator:
    """Calculates adaptive relevance thresholds"""

    def __init__(self, config):
        self.config = config
        self.base_threshold = 0.1
        self.min_threshold = 0.05
        self.max_threshold = 0.3

    def calculate(self, query: str, relevance_scores: List[float],
                  result_count: int) -> float:
        """Calculate adaptive threshold based on multiple factors"""

        # Factor 1: Score distribution
        if relevance_scores:
            avg_score = np.mean(relevance_scores)
            std_score = np.std(relevance_scores)
            distribution_factor = max(0, avg_score - std_score)
        else:
            distribution_factor = self.base_threshold

        # Factor 2: Domain term detection
        domain_factor = 0.7 if self._has_domain_terms(query) else 1.0

        # Factor 3: Result count
        if result_count == 0:
            count_factor = 0.5  # Lower threshold when no results
        elif result_count < 5:
            count_factor = 0.7
        else:
            count_factor = 1.0

        # Calculate final threshold
        threshold = distribution_factor * domain_factor * count_factor

        # Clamp to valid range
        return max(self.min_threshold,
                  min(threshold, self.max_threshold))
```

## Data Flow

### Search Flow

```
1. User Query
   ↓
2. Query Enhancement
   - Normalize
   - Expand (generate 3-5 variants)
   ↓
3. Multi-Strategy Retrieval (parallel)
   ├─ Vector Search (expanded queries)
   ├─ BM25 Search (original query)
   └─ Exact Field Match (original query)
   ↓
4. Result Fusion
   - RRF combination
   - Deduplication
   - Re-ranking
   ↓
5. Quality Filter
   - Adaptive threshold
   - Relevance check
   ↓
6. Return Results + Quality Metrics
```

## Configuration

### New Configuration Parameters

```python
class SearchConfig:
    # Query enhancement
    ENABLE_QUERY_EXPANSION: bool = True
    MAX_EXPANDED_QUERIES: int = 5
    DOMAIN_TERM_DICT_PATH: str = "config/domain_terms.json"

    # Retrieval strategies
    ENABLE_VECTOR_SEARCH: bool = True
    ENABLE_BM25_FALLBACK: bool = True
    ENABLE_EXACT_MATCH_FALLBACK: bool = True

    # Fusion weights
    VECTOR_SEARCH_WEIGHT: float = 0.7
    BM25_SEARCH_WEIGHT: float = 0.2
    EXACT_MATCH_WEIGHT: float = 0.1

    # Thresholding
    BASE_RELEVANCE_THRESHOLD: float = 0.1
    MIN_RELEVANCE_THRESHOLD: float = 0.05
    MAX_RELEVANCE_THRESHOLD: float = 0.3
    ADAPTIVE_THRESHOLD_ENABLED: bool = True

    # Performance
    MAX_CANDIDATES: int = 100  # candidates before fusion
    CANDIDATE_MULTIPLIER: int = 3  # top_k * multiplier
```

## Performance Considerations

### Latency Budget
- Vector search: 50-100ms
- BM25 search: 20-50ms
- Exact match: 10-20ms
- Fusion: 10-20ms
- **Total target: <200ms**

### Optimization Strategies
1. **Lazy BM25 indexing**: Build index on first use
2. **Query caching**: Cache embeddings and expanded queries
3. **Parallel execution**: Run strategies in parallel threads
4. **Result limiting**: Limit candidates before expensive operations

### Memory Impact
- BM25 index: ~10-20MB per 100K documents
- Query cache: ~5MB for 1000 cached queries
- Domain term dict: ~1MB
- **Total overhead: ~16-26MB**

## Testing Strategy

### Unit Tests
- Query expansion correctness
- BM25 search accuracy
- Result fusion logic
- Adaptive threshold calculation
- RRF scoring

### Integration Tests
- End-to-end search workflows
- MCP tool integration
- Multi-language queries
- Domain-specific queries

### Performance Tests
- Search latency benchmarks
- Memory usage profiling
- Concurrent query handling
- Cache effectiveness

### Validation Tests
- Historical failed queries
- Known-to-exist content retrieval
- Result quality assessment
- User acceptance testing

## Migration Path

### Phase 1: Foundation (Week 1)
- Implement query enhancement
- Add BM25 fallback
- Basic result fusion

### Phase 2: Enhancement (Week 2)
- Implement adaptive thresholding
- Add exact field match
- Enhanced fusion logic

### Phase 3: Optimization (Week 3)
- Query caching
- Performance tuning
- Quality metrics

### Phase 4: Validation (Week 4)
- Comprehensive testing
- User acceptance
- Documentation

## Rollback Strategy

If issues occur:
1. Feature flag can disable new functionality
2. Fallback to original vector-only search
3. Configuration reverts to previous settings
4. No data migration required

## Monitoring and Observability

### Key Metrics
- Retrieval rate (success finding known content)
- Average search latency
- Fallback activation rate
- Cache hit rate
- Result relevance scores
- User satisfaction (if feedback available)

### Logging
- All search queries (anonymized)
- Strategy selection decisions
- Threshold adjustments
- Fallback activations
- Performance metrics

### Alerts
- Retrieval rate <90%
- Search latency >500ms
- Fallback rate >20%
- Error rate >1%

## Future Enhancements

1. **Embedding Model Fine-tuning**: Domain-specific adaptation
2. **Learning to Rank**: ML-based result ranking
3. **User Feedback Integration**: Continuous improvement
4. **Query Intent Detection**: Different strategies for different intents
5. **Distributed Search**: Horizontal scaling for large datasets
