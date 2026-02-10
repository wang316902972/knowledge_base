# Spec Delta: Result Processing Capability

## Capability Overview
**Capability ID**: result-processing
**Change Type**: ADDED
**Related Changes**: vector-search, query-processing, mcp-integration

---

## ADDED Requirements

### Requirement: Multi-Strategy Result Fusion

The system MUST combine and fuse results from multiple retrieval strategies using Reciprocal Rank Fusion (RRF).

#### Scenario: Fuse vector and BM25 results
**Given** vector search returns 8 results and BM25 search returns 12 results
**When** the result fusion module processes both result sets
**Then** the system SHOULD apply RRF algorithm with configured weights (vector: 0.7, BM25: 0.2)
**And** the system SHOULD calculate combined RRF scores for each unique result
**And** the system SHOULD return top-k results sorted by RRF score
**And** duplicate results SHOULD appear only once with combined score

#### Scenario: Fuse three strategy results
**Given** vector (10 results), BM25 (15 results), and exact match (3 results)
**When** all three strategies return results
**Then** the system SHOULD fuse all three result sets with weights (0.7, 0.2, 0.1)
**And** the system SHOULD apply RRF with k=60 constant
**And** the system SHOULD return unique results ranked by combined RRF score
**And** the system SHOULD include metadata indicating which strategies contributed

#### Scenario: Deduplicate overlapping results
**Given** multiple strategies return the same document (faiss_id=19)
**When** fusing results
**Then** the system SHOULD identify duplicates by faiss_id
**And** the system SHOULD combine scores from all strategies
**And** the system SHOULD return the document once with the combined score
**And** the system SHOULD track which strategies found the document

### Requirement: Result Re-ranking

The system MUST re-rank fused results to optimize for relevance and diversity.

#### Scenario: Diversity-based re-ranking
**Given** 10 fused results with high relevance but low diversity
**When** the re-ranking module processes results
**Then** the system SHOULD apply Maximal Marginal Relevance (MMR) algorithm
**And** the system SHOULD balance relevance (70%) and diversity (30%)
**And** the system SHOULD re-order results to include diverse content
**And** the final results SHOULD maintain ≥80% of original relevance

#### Scenario: Domain-specific result boosting
**Given** results including documents with exact domain term matches
**When** re-ranking with domain awareness
**Then** the system SHOULD boost results with exact domain term matches by 10%
**And** the system SHOULD boost results with multiple domain term matches by 20%
**And** the boosting SHOULD be applied after RRF fusion
**And** the system SHOULD log boosted results for monitoring

### Requirement: Quality Metrics Calculation

The system MUST calculate comprehensive quality metrics for search results.

#### Scenario: Calculate result quality metrics
**Given** a set of search results with relevance scores
**When** quality metrics are calculated
**Then** the system SHOULD compute:
- Average relevance score (mean of all relevance scores)
- Diversity score (average pairwise distance between result vectors)
- Coverage ratio (percentage of results with relevance >0.7)
- Precision@k (percentage of results with relevance >0.6)
**And** metrics SHOULD be included in the first result's metadata
**And** metrics SHOULD be used for threshold adaptation

#### Scenario: Track search strategy effectiveness
**Given** a search that used multiple strategies
**When** the search completes
**Then** the system SHOULD track:
- Number of results from each strategy
- Average relevance per strategy
- Strategy contribution to final results
- Fallback activation count
**And** the system SHOULD log strategy effectiveness metrics
**And** metrics SHOULD be available for monitoring dashboards

---

## MODIFIED Requirements

### Requirement: Result Filtering

The system MUST apply adaptive filtering based on result quality, query characteristics, and adaptive thresholds, replacing the previous fixed threshold filtering.

**Previous Behavior**: The system filtered results based on a fixed relevance threshold of 0.1.

**New Behavior**: The system applies adaptive filtering based on result quality, query characteristics, and adaptive thresholds.

#### Scenario: Apply adaptive relevance filtering
**Given** search results with relevance scores [0.85, 0.62, 0.38, 0.12, 0.08, 0.05]
**When** adaptive filtering is applied with threshold 0.1
**Then** the system SHOULD return results with scores ≥0.1
**And** if fewer than 3 results pass the threshold, the system SHOULD lower it to 0.05
**And** the system SHOULD return the top 3 results after threshold adjustment
**And** the system SHOULD log threshold adjustments

#### Scenario: Result quality-based filtering
**Given** fused results with varying quality scores
**When** quality-based filtering is applied
**Then** the system SHOULD calculate quality metrics for all results
**And** the system SHOULD apply stricter filtering if quality metrics are high (avg relevance >0.7)
**And** the system SHOULD apply relaxed filtering if quality metrics are low (avg relevance <0.3)
**And** the filtering threshold SHOULD be bounded between 0.05 and 0.3

### Requirement: Result Ranking

The system MUST rank results using composite scores from multiple signals, replacing the previous single-signal ranking.

**Previous Behavior**: The system ranked results solely by FAISS distance scores.

**New Behavior**: The system ranks results using composite scores from multiple signals.

#### Scenario: Composite score ranking
**Given** search results with multiple score signals:
- FAISS similarity_score: 0.85
- Cosine relevance_score: 0.82
- RRF score: 0.015
- Domain match boost: +10%
**When** results are ranked
**Then** the system SHOULD calculate composite_score = weighted_sum(all_signals)
**And** the system SHOULD sort results by composite_score in descending order
**And** the system SHOULD include all individual scores in result metadata
**And** the composite_score SHOULD be the primary sorting key

#### Scenario: Preserve result metadata
**Given** search results from multiple strategies
**When** results are processed and ranked
**Then** each result SHOULD include:
- text (document content)
- similarity_score (FAISS distance)
- relevance_score (cosine similarity)
- rrf_score (reciprocal rank fusion)
- composite_score (final ranking score)
- faiss_id (document identifier)
- match_type (vector, bm25, exact)
- strategies_used (list of strategies that found this result)
- domain_boost (applied boost factor, if any)
**And** all metadata SHOULD be transparent to the caller

---

## DELETED Requirements

None. All existing requirements are preserved or enhanced.

---

## Implementation Notes

### Configuration Changes

New configuration parameters added to `ResultConfig`:
```python
# Result fusion
ENABLE_RESULT_FUSION: bool = True
RRF_K_CONSTANT: int = 60  # RRF constant
VECTOR_SEARCH_WEIGHT: float = 0.7
BM25_SEARCH_WEIGHT: float = 0.2
EXACT_MATCH_WEIGHT: float = 0.1

# Re-ranking
ENABLE_DIVERSION_RERANKING: bool = True
DIVERSITY_WEIGHT: float = 0.3
DOMAIN_BOOST_FACTOR: float = 0.1  # 10% boost
MULTI_DOMAIN_BOOST_FACTOR: float = 0.2  # 20% boost

# Filtering
ENABLE_ADAPTIVE_FILTERING: bool = True
MIN_RELEVANCE_THRESHOLD: float = 0.05
BASE_RELEVANCE_THRESHOLD: float = 0.1
MAX_RELEVANCE_THRESHOLD: float = 0.3
MIN_RESULTS_THRESHOLD: int = 3  # minimum results to return
```

### Data Structures

**Enhanced Result Object**:
```python
@dataclass
class EnhancedSearchResult:
    text: str
    similarity_score: float  # FAISS distance
    relevance_score: float  # Cosine similarity
    rrf_score: float  # Reciprocal rank fusion
    composite_score: float  # Final ranking score
    faiss_id: int
    match_type: str  # 'vector', 'bm25', 'exact', 'hybrid'
    strategies_used: List[str]
    domain_boost: float  # Applied boost (0.0 to 1.0)
    rank: int  # Final position

@dataclass
class QualityMetrics:
    avg_relevance_score: float
    diversity_score: float
    coverage_ratio: float
    precision_at_k: float
    total_results: int
    strategies_used: List[str]
    threshold_used: float
    threshold_adjustments: List[Dict]
```

### Fusion Algorithm

**Reciprocal Rank Fusion (RRF)**:
```python
def reciprocal_rank_fusion(
    results_dict: Dict[str, List[SearchResult]],
    weights: Dict[str, float],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Combine rankings using RRF algorithm

    Args:
        results_dict: {'strategy': [results]}
        weights: {'strategy': weight}
        k: RRF constant (default 60)

    Returns:
        List of (faiss_id, rrf_score) tuples
    """
    rrf_scores = {}

    for strategy, results in results_dict.items():
        weight = weights.get(strategy, 1.0)
        for rank, result in enumerate(results, 1):
            faiss_id = result.faiss_id
            rrf_score = weight / (k + rank)

            if faiss_id in rrf_scores:
                rrf_scores[faiss_id] += rrf_score
            else:
                rrf_scores[faiss_id] = rrf_score

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
```

### Re-ranking Algorithm

**Maximal Marginal Relevance (MMR)**:
```python
def maximal_marginal_relevance(
    results: List[EnhancedSearchResult],
    query_embedding: np.ndarray,
    lambda_: float = 0.7  # relevance weight
) -> List[EnhancedSearchResult]:
    """
    Re-rank results using MMR for diversity

    Args:
        results: Initial ranked results
        query_embedding: Query vector
        lambda_: Balance between relevance and diversity

    Returns:
        Re-ranked results with diversity consideration
    """
    if len(results) <= 1:
        return results

    selected = []
    remaining = results.copy()

    # Select first result (highest relevance)
    selected.append(remaining.pop(0))

    while remaining and len(selected) < len(results):
        best_score = float('-inf')
        best_idx = -1

        for i, result in enumerate(remaining):
            # Relevance to query
            relevance = result.relevance_score

            # Maximum similarity to already selected
            max_similarity = 0
            for selected_result in selected:
                similarity = cosine_similarity(
                    result.embedding,
                    selected_result.embedding
                )
                max_similarity = max(max_similarity, similarity)

            # MMR score
            mmr_score = (lambda_ * relevance) - ((1 - lambda_) * max_similarity)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected
```

### API Changes

**Result Processing API** (enhanced):
```python
def fuse_results(
    results_dict: Dict[str, List[SearchResult]],
    weights: Dict[str, float]
) -> List[EnhancedSearchResult]:
    """Fuse results from multiple strategies"""

def rerank_results(
    results: List[EnhancedSearchResult],
    query: EnhancedQuery,
    diversity_weight: float = 0.3
) -> List[EnhancedSearchResult]:
    """Re-rank results for diversity"""

def calculate_quality_metrics(
    results: List[EnhancedSearchResult],
    query: EnhancedQuery
) -> QualityMetrics:
    """Calculate comprehensive quality metrics"""

def apply_adaptive_filtering(
    results: List[EnhancedSearchResult],
    metrics: QualityMetrics
) -> List[EnhancedSearchResult]:
    """Apply adaptive relevance filtering"""
```

### Performance Impact

- **Fusion Latency**: +10-20ms for RRF calculation
- **Re-ranking Latency**: +20-30ms for MMR algorithm
- **Total Overhead**: +30-50ms for result processing
- **Memory**: +2-5MB for intermediate result structures
- **Net Impact**: Acceptable within 200ms total budget

### Testing Requirements

1. **Unit Tests**:
   - Test RRF fusion correctness
   - Test MMR re-ranking logic
   - Test adaptive filtering behavior
   - Test quality metrics calculation

2. **Integration Tests**:
   - Test end-to-end result processing
   - Test multi-strategy fusion
   - Test re-ranking effectiveness

3. **Performance Tests**:
   - Benchmark fusion latency
   - Profile memory usage
   - Test with large result sets (100+ results)

4. **Quality Tests**:
   - Validate diversity improvement
   - Measure precision/recall impact
   - Test with real user queries

---

## Related Capabilities

- **vector-search**: Provides initial results from multiple strategies
- **query-processing**: Supplies query embeddings for re-ranking
- **mcp-integration**: Returns enhanced results to MCP tools

---

## Migration Notes

### Backward Compatibility

- Existing result processing remains functional
- Enhanced fields are additive
- Feature flags can disable fusion/re-ranking
- Result object maintains backward-compatible structure

### Rollback Plan

If issues occur:
1. Set `ENABLE_RESULT_FUSION=False`
2. Set `ENABLE_DIVERSION_RERANKING=False`
3. System reverts to simple filtering and ranking
4. No data migration required

### Deployment Strategy

1. **Phase 1**: Deploy with fusion disabled
2. **Phase 2**: Enable fusion for testing
3. **Phase 3**: Enable re-ranking gradually
4. **Phase 4**: Monitor quality metrics and adjust parameters
5. **Phase 5**: Full production deployment

### Monitoring Requirements

Key metrics to track:
- Fusion effectiveness (unique result count)
- Re-ranking impact (diversity score improvement)
- Quality metrics (average relevance, precision)
- Latency breakdown (fusion, re-ranking, filtering)
- User satisfaction (if feedback available)
