# Proposal: Improve Vector Search Retrieval Accuracy

## Metadata
- **Change ID**: improve-retrieval-accuracy
- **Status**: Proposed
- **Created**: 2025-01-18
- **Author**: AI Assistant (Claude)

## Problem Statement

Users are experiencing poor retrieval accuracy when searching for domain-specific technical content in the FAISS vector database. Specifically:

- **Symptom**: Search queries fail to retrieve known existing content
- **Example**: Query "反作弊归因分析" (anti-cheat attribution analysis) fails to retrieve the documented prefab, even though the data exists in chunk 19 of `gtplanner_prefabs_knowledge_base.json`
- **Root Cause**: Multiple factors contributing to semantic retrieval gaps:
  1. **Relevance Threshold Too Aggressive**: Current `relevance_threshold=0.1` filters out valid results
  2. **Embedding Model Limitations**: `paraphrase-multilingual-MiniLM-L12-v2` trained on general corpus may not capture domain-specific technical terminology well
  3. **Query-Content Semantic Gap**: Natural language queries don't align with technical documentation chunking
  4. **Single-Stage Retrieval**: No fallback mechanisms when primary search fails

## Impact Analysis

**User Impact**:
- Users cannot find relevant technical documentation
- MCP tools return empty results for valid queries
- Loss of trust in the search system

**Business Impact**:
- Reduced utility of the vector database for domain-specific knowledge
- Increased support burden
- Potential abandonment of the system

**Technical Impact**:
- Current search optimization features (diversity reranking, relevance filtering) may be **over-filtering** valid results
- Need for more adaptive retrieval strategies

## Proposed Solution

Implement a multi-layered retrieval accuracy improvement strategy:

### 1. Hybrid Retrieval with Fallback
- **Primary**: Semantic vector search (current approach)
- **Fallback**: Keyword-based BM25 retrieval when semantic search fails
- **Tertiary**: Exact string matching for critical fields (id, name, tags)

### 2. Adaptive Relevance Thresholding
- Dynamic threshold adjustment based on result quality
- Lower threshold when domain-specific terms detected
- Result count-aware thresholding

### 3. Query Expansion and Rewriting
- Automatic expansion of technical terms
- Synonym injection for domain-specific vocabulary
- Multi-lingual query augmentation

### 4. Enhanced Embedding Strategy
- Domain-adaptive fine-tuning of embedding model (optional)
- Hybrid embeddings: general + domain-specific models
- Embedding caching for repeated queries

### 5. Retrieval Quality Feedback Loop
- Track failed queries and results
- User feedback mechanism for relevance
- Continuous improvement of retrieval strategy

## Success Criteria

1. **Retrieval Rate**: ≥95% of known-to-exist content should be retrievable
2. **Relevance Precision**: ≥80% of top-3 results should be relevant
3. **Fallback Activation**: Fallback mechanisms should activate in <10% of queries
4. **Performance**: Search latency should remain <200ms for top-5 results
5. **User Satisfaction**: ≥90% of searches should return useful results

## Alternatives Considered

### Alternative 1: Lower Relevance Threshold Globally
- **Pros**: Simple implementation
- **Cons**: Increases false positives, reduces overall result quality
- **Decision**: Rejected in favor of adaptive thresholding

### Alternative 2: Switch to Different Embedding Model
- **Pros**: Potentially better domain coverage
- **Cons**: Requires re-indexing all data, model evaluation overhead, larger model size
- **Decision**: Keep as optional enhancement, not primary solution

### Alternative 3: Pure Keyword Search
- **Pros**: Guaranteed retrieval for exact matches
- **Cons**: No semantic understanding, poor user experience for natural language queries
- **Decision**: Use as fallback only

## Dependencies

- **Required**: None (standalone improvement)
- **Optional**:
  - User feedback collection system (future enhancement)
  - A/B testing framework (for validation)
  - Embedding model fine-tuning pipeline (future work)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Increased search latency | Medium | Medium | Implement caching, lazy fallback activation |
| More false positives | Low | High | Maintain relevance scoring for ranking |
| Complex query logic | Medium | Low | Comprehensive testing, clear documentation |
| Breaking existing behavior | Low | High | Feature flag, gradual rollout |

## Implementation Scope

**In Scope**:
- Hybrid retrieval with fallback mechanisms
- Adaptive relevance thresholding
- Query expansion for domain-specific terms
- Enhanced search API with quality metrics
- Comprehensive testing and validation

**Out of Scope**:
- Embedding model retraining/fine-tuning (future phase)
- User feedback UI (future enhancement)
- Distributed search scaling (future architecture)

## Rollout Plan

1. **Phase 1**: Core improvements (adaptive thresholding, query expansion)
2. **Phase 2**: Hybrid retrieval with fallback
3. **Phase 3**: Quality metrics and monitoring
4. **Phase 4**: Validation and tuning with real queries

## Open Questions

1. Should we implement a configurable search strategy per business type?
2. What's the acceptable increase in search latency for improved accuracy?
3. Should failed queries be logged for analysis?
4. Do we need a UI for tuning search parameters?

## Related Changes

- None (this is a standalone improvement)

## Approval Status

- [ ] Technical Review
- [ ] Product Review
- [ ] Security Review
- [ ] Final Approval
