# Spec Delta: Vector Search Capability

## Capability Overview
**Capability ID**: vector-search
**Change Type**: ADDED
**Related Changes**: query-processing, result-processing, mcp-integration

---

## ADDED Requirements

### Requirement: Hybrid Retrieval Strategy

The vector search system MUST support hybrid retrieval combining multiple search strategies with automatic fallback mechanisms.

#### Scenario: Primary vector search succeeds
**Given** a user query for "反作弊归因分析"
**When** the query is processed through the vector search engine
**Then** the system SHOULD return results from the primary vector search with relevance scores ≥0.1
**And** the results SHOULD include the anti-cheat prefab documentation if it exists in the database

#### Scenario: Vector search fails, BM25 fallback activates
**Given** a user query that fails to return results from vector search
**When** the adaptive threshold determines results are insufficient
**Then** the system SHOULD automatically activate BM25 keyword search as a fallback
**And** the system SHOULD return keyword-based results sorted by BM25 score
**And** the system SHOULD log the fallback activation for monitoring

#### Scenario: Both vector and BM25 fail, exact match activates
**Given** a user query for exact field values (e.g., id="anti-cheat-attribution-analyzer")
**When** both vector and BM25 searches return insufficient results
**Then** the system SHOULD perform exact field matching on critical fields (id, name, tags, description)
**And** the system SHOULD return documents with field scores ≥0.8
**And** the system SHOULD indicate the match type in the result metadata

### Requirement: Adaptive Relevance Thresholding

The system MUST calculate and apply adaptive relevance thresholds based on query characteristics and result quality.

#### Scenario: Domain-specific query with low threshold
**Given** a user query containing domain-specific terms (e.g., "反作弊归因分析预制件")
**When** the domain term detector identifies technical terminology
**Then** the system SHOULD apply a 30% lower relevance threshold
**And** the system SHOULD log the threshold adjustment with reasoning

#### Scenario: No results found, threshold relaxation
**Given** a user query that returns zero results with the base threshold
**When** the result quality check indicates insufficient results
**Then** the system SHOULD progressively lower the threshold (0.1 → 0.05 → 0.025)
**And** the system SHOULD return results if any are found with the relaxed threshold
**And** the system SHOULD indicate the threshold relaxation in result metadata

#### Scenario: High-quality results, maintain strict threshold
**Given** a user query that returns ≥5 results with relevance scores ≥0.7
**When** the result quality check indicates strong matches
**Then** the system SHOULD maintain or increase the threshold to filter low-quality results
**And** the system SHOULD return only the highest quality results

---

## MODIFIED Requirements

### Requirement: Vector Search Execution

The system MUST perform vector search with adaptive parameters as part of a multi-strategy retrieval approach, replacing the previous single-strategy fixed-parameter search.

**Previous Behavior**: The system performed a single vector search with fixed parameters and returned results above a fixed threshold of 0.1.

**New Behavior**: The system performs vector search with adaptive parameters as part of a multi-strategy retrieval approach.

#### Scenario: Execute vector search with adaptive nprobe
**Given** a user query and a FAISS IVF index with 100K vectors
**When** the vector search strategy is executed
**Then** the system SHOULD calculate adaptive nprobe based on index size (nlist/4 to nlist/2)
**And** the system SHOULD search for `top_k * 3` candidates to allow for quality filtering
**And** the system SHOULD return results with both FAISS distance and relevance scores

#### Scenario: Calculate relevance scores for results
**Given** a set of candidate results from vector search
**When** the results are being processed
**Then** the system SHOULD calculate cosine similarity between query and each result text
**And** the system SHOULD assign the cosine similarity as the relevance_score for each result
**And** the system SHOULD maintain the original FAISS distance as similarity_score

### Requirement: Search Quality Metrics

The system MUST calculate and return comprehensive quality metrics for search results, enhancing the previous basic similarity scoring.

**Previous Behavior**: The system returned basic similarity scores without quality assessment.

**New Behavior**: The system returns comprehensive quality metrics for search results.

#### Scenario: Generate quality metrics for search results
**Given** a search query that returns 5 results
**When** the search completes successfully
**Then** the system SHOULD calculate and return:
- Average relevance score across all results
- Diversity score (average pairwise distance between results)
- Coverage ratio (percentage of results with relevance >0.7)
- Precision@k (percentage of results with relevance >0.6)
**And** the metrics SHOULD be included in the first result's metadata

---

## DELETED Requirements

None. All existing requirements are preserved or enhanced.

---

## Implementation Notes

### Configuration Changes

New configuration parameters added to `SearchConfig`:
```python
# Hybrid retrieval
ENABLE_BM25_FALLBACK: bool = True
ENABLE_EXACT_MATCH_FALLBACK: bool = True
VECTOR_SEARCH_WEIGHT: float = 0.7
BM25_SEARCH_WEIGHT: float = 0.2
EXACT_MATCH_WEIGHT: float = 0.1

# Adaptive thresholding
BASE_RELEVANCE_THRESHOLD: float = 0.1
MIN_RELEVANCE_THRESHOLD: float = 0.05
MAX_RELEVANCE_THRESHOLD: float = 0.3
ADAPTIVE_THRESHOLD_ENABLED: bool = True

# Candidate generation
CANDIDATE_MULTIPLIER: int = 3
MAX_CANDIDATES: int = 100
```

### API Changes

**Search Response Format** (enhanced):
```python
{
    "results": [
        {
            "text": "document text",
            "similarity_score": 0.85,  # FAISS distance
            "relevance_score": 0.82,  # Cosine similarity
            "faiss_id": 123,
            "match_type": "vector|bm25|exact",  # NEW
            "threshold_used": 0.08,  # NEW
            "quality_metrics": {...}  # ENHANCED
        }
    ],
    "search_strategy": "hybrid",  # NEW
    "strategies_used": ["vector", "bm25"],  # NEW
    "total_candidates": 15,  # NEW
    "threshold_adjustments": [  # NEW
        {"from": 0.1, "to": 0.07, "reason": "domain_terms"}
    ]
}
```

### Dependencies

**New Dependencies**:
- `rank-bm25`: BM25 algorithm implementation
- Existing FAISS, sentence-transformers (no changes)

**Internal Dependencies**:
- Query enhancement module (must execute before search)
- Result fusion module (combines multi-strategy results)
- Adaptive threshold calculator (used during filtering)

### Performance Impact

- **Latency**: +50-100ms for additional search strategies
- **Memory**: +10-20MB for BM25 index
- **Throughput**: May decrease by ~20% due to parallel strategy execution
- **Mitigation**: Query caching, lazy BM25 indexing, parallel execution

### Testing Requirements

1. **Unit Tests**:
   - Test adaptive threshold calculation
   - Test multi-strategy execution
   - Test result fusion logic

2. **Integration Tests**:
   - Test end-to-end hybrid search
   - Test fallback activation
   - Test quality metrics calculation

3. **Performance Tests**:
   - Benchmark search latency
   - Memory usage profiling
   - Concurrent query handling

4. **Validation Tests**:
   - Test retrieval of known-to-exist content
   - Test domain-specific queries
   - Test multi-language queries

---

## Related Capabilities

- **query-processing**: Provides query enhancement and expansion
- **result-processing**: Handles result fusion and filtering
- **mcp-integration**: Exposes enhanced search through MCP tools

---

## Migration Notes

### Backward Compatibility

- Existing search API calls remain functional
- New fields are additive, not breaking
- Feature flag can disable new functionality if needed
- Default behavior unchanged unless explicitly enabled

### Rollback Plan

If issues occur:
1. Set `ENABLE_BM25_FALLBACK=False` and `ENABLE_EXACT_MATCH_FALLBACK=False`
2. Set `ADAPTIVE_THRESHOLD_ENABLED=False`
3. System reverts to original vector-only search
4. No data migration required

### Deployment Strategy

1. **Phase 1**: Deploy with feature flags disabled
2. **Phase 2**: Enable for testing environment
3. **Phase 3**: Gradual rollout with monitoring (10% → 50% → 100%)
4. **Phase 4**: Full production deployment
