# Spec Delta: MCP Integration Capability

## Capability Overview
**Capability ID**: mcp-integration
**Change Type**: ADDED
**Related Changes**: vector-search, query-processing, result-processing

---

## ADDED Requirements

### Requirement: Enhanced Search Tool Parameters

The MCP search tools MUST support new parameters for controlling retrieval strategy and output format.

#### Scenario: Specify retrieval strategy
**Given** a user calling the `search_knowledge` MCP tool
**When** the user provides `retrieval_strategy` parameter
**Then** the system SHOULD use the specified strategy ("vector", "hybrid", "bm25", "exact")
**And** if not specified, the system SHOULD auto-select the best strategy
**And** the system SHOULD include the selected strategy in the response metadata

#### Scenario: Enable/disable search optimization
**Given** a user calling `search_knowledge` with `use_optimization=true`
**When** the search is executed
**Then** the system SHOULD enable query expansion, adaptive thresholding, and result re-ranking
**And** when `use_optimization=false`, the system SHOULD use basic vector search only
**And** the system SHOULD respect the optimization flag in all cases

#### Scenario: Request quality metrics
**Given** a user calling `search_knowledge` with `include_metrics=true`
**When** the search completes
**Then** the system SHOULD return comprehensive quality metrics
**And** the metrics SHOULD include: avg_relevance, diversity, coverage, precision
**And** the metrics SHOULD be included in the tool response metadata

### Requirement: Enhanced Tool Response Format

The MCP tools MUST return enhanced result metadata to provide transparency into search behavior.

#### Scenario: Return enhanced search results
**Given** a successful search execution via MCP tool
**When** results are returned
**Then** each result SHOULD include:
- text (document content)
- similarity_score (FAISS distance)
- relevance_score (cosine similarity)
- match_type (vector/bm25/exact/hybrid)
- strategies_used (list of strategies that found this result)
**And** the response SHOULD include search_metadata with:
- search_strategy_used
- total_candidates_processed
- threshold_used
- threshold_adjustments (if any)
- quality_metrics (if requested)

#### Scenario: Transparent fallback reporting
**Given** a search that activated fallback mechanisms
**When** results are returned via MCP tool
**Then** the response metadata SHOULD include:
- activated_strategies (list of all strategies used)
- fallback_reason (why fallback was activated)
- strategy_contribution (results per strategy)
**And** the user SHOULD be able to see which search methods were successful

---

## MODIFIED Requirements

### Requirement: search_knowledge Tool

The MCP search_knowledge tool MUST perform hybrid retrieval with enhancement options and return comprehensive result metadata, replacing the previous basic vector search tool.

**Previous Behavior**: The tool performed basic vector search and returned results with similarity scores.

**New Behavior**: The tool performs hybrid retrieval with enhancement options and returns comprehensive result metadata.

#### Scenario: Execute optimized search
**Given** a user calls `search_knowledge` with:
  - query: "反作弊归因分析预制件"
  - businesstype: "gtplanner_prefabs"
  - top_k: 5
  - use_optimization: true
**When** the tool is executed
**Then** the system SHOULD:
1. Enhance the query with domain term expansion
2. Execute hybrid retrieval (vector + BM25 + exact match)
3. Fuse results using RRF
4. Apply adaptive thresholding
5. Re-rank for diversity
6. Return top-5 results with enhanced metadata
**And** the results SHOULD include the anti-cheat prefab documentation if it exists

#### Scenario: Execute basic search (backward compatible)
**Given** a user calls `search_knowledge` with:
  - query: "test query"
  - businesstype: "default"
  - top_k: 3
  - use_optimization: false (or omitted)
**When** the tool is executed
**Then** the system SHOULD perform basic vector search without enhancements
**And** the system SHOULD return results in the original format
**And** the system SHOULD maintain backward compatibility with existing MCP clients

### Requirement: add_document Tool

The MCP add_document tool MUST maintain existing behavior while supporting enhanced indexing for future improvements.

**Previous Behavior**: The tool added documents to the vector index with basic chunking.

**New Behavior**: The tool maintains existing behavior but supports enhanced indexing for future improvements.

#### Scenario: Add document with semantic chunking
**Given** a user calls `add_document` with a long document
**When** semantic chunking is enabled (future enhancement)
**Then** the system SHOULD use semantic-aware chunking instead of fixed-size chunks
**And** the system SHOULD preserve chunk boundaries at semantic boundaries
**And** the system SHOULD store chunk metadata for retrieval

#### Scenario: Backward compatible document addition
**Given** a user calls `add_document` without enhanced parameters
**When** the document is added
**Then** the system SHOULD use existing chunking behavior
**And** the system SHOULD maintain backward compatibility
**And** no changes to existing functionality

---

## DELETED Requirements

None. All existing requirements are preserved or enhanced.

---

## Implementation Notes

### MCP Tool Schema Updates

**Enhanced search_knowledge Tool**:
```python
Tool(
    name="search_knowledge",
    description="""在向量数据库中搜索相关知识。支持语义搜索和混合检索。

默认业务类型: {default_bt}

功能特性:
- 语义向量搜索
- 混合检索策略（向量 + 关键词 + 精确匹配）
- 可配置返回结果数量 (top_k)
- 搜索优化选项（查询扩展、自适应阈值、多样性重排序）
- 业务类型隔离的独立索引
- 全面的质量指标和元数据""",
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
            "businesstype": {
                "type": "string",
                "description": "业务类型标识符（可选，默认使用环境变量）",
                "default": None
            },
            "use_optimization": {
                "type": "boolean",
                "description": "是否启用搜索优化（查询扩展、自适应阈值、重排序）",
                "default": True
            },
            "retrieval_strategy": {
                "type": "string",
                "enum": ["auto", "vector", "hybrid", "bm25", "exact"],
                "description": "检索策略：auto（自动选择）, vector（仅向量）, hybrid（混合）, bm25（关键词）, exact（精确匹配）",
                "default": "auto"
            },
            "include_metrics": {
                "type": "boolean",
                "description": "是否在响应中包含质量指标",
                "default": False
            }
        },
        "required": ["query"]
    }
)
```

### Response Format Enhancements

**Enhanced Search Response**:
```json
{
    "results": [
        {
            "text": "文档内容...",
            "similarity_score": 0.85,
            "relevance_score": 0.82,
            "match_type": "hybrid",
            "strategies_used": ["vector", "bm25"],
            "faiss_id": 19
        }
    ],
    "search_metadata": {
        "query": "反作弊归因分析预制件",
        "search_strategy_used": "hybrid",
        "total_candidates": 15,
        "threshold_used": 0.08,
        "threshold_adjustments": [
            {
                "from": 0.1,
                "to": 0.08,
                "reason": "domain_terms_detected"
            }
        ],
        "activated_strategies": ["vector", "bm25", "exact_match"],
        "fallback_reason": null,
        "strategy_contribution": {
            "vector": 8,
            "bm25": 5,
            "exact_match": 2
        }
    },
    "quality_metrics": {
        "avg_relevance_score": 0.75,
        "diversity_score": 0.35,
        "coverage_ratio": 0.80,
        "precision_at_k": 0.85
    }
}
```

### Backward Compatibility

**Legacy Response Format** (when `use_optimization=false`):
```json
{
    "results": [
        {
            "text": "文档内容...",
            "similarity_score": 0.85,
            "faiss_id": 19
        }
    ]
}
```

### Configuration Changes

New MCP-specific configuration:
```python
class MCPToolConfig:
    # Tool behavior
    DEFAULT_OPTIMIZATION_ENABLED: bool = True
    DEFAULT_RETRIEVAL_STRATEGY: str = "auto"
    DEFAULT_INCLUDE_METRICS: bool = False

    # Response format
    ENABLE_ENHANCED_RESPONSES: bool = True
    INCLUDE_DEBUG_INFO: bool = False  # For troubleshooting

    # Rate limiting
    MAX_SEARCH_PER_MINUTE: int = 100
    MAX_TOP_K: int = 50
```

### Error Handling

**Enhanced Error Messages**:
```python
class MCPSearchError(Exception):
    """Base class for MCP search errors"""

class QueryTooLongError(MCPSearchError):
    """Query exceeds maximum length"""

class InvalidBusinessTypeError(MCPSearchError):
    """Business type validation failed"""

class SearchStrategyError(MCPSearchError):
    """Specified strategy not available"""

class NoResultsError(MCPSearchError):
    """No results found with any strategy"""
```

### Performance Considerations

- **Tool Latency**: +30-50ms for enhanced search with optimization
- **Response Size**: +20-30% for enhanced metadata
- **Memory**: +5MB for tool-level caching
- **Monitoring**: Track tool usage patterns and performance

### Testing Requirements

1. **Unit Tests**:
   - Test tool parameter validation
   - Test response format generation
   - Test error handling

2. **Integration Tests**:
   - Test MCP tool execution
   - Test multi-strategy search via MCP
   - Test backward compatibility

3. **End-to-End Tests**:
   - Test real MCP client scenarios
   - Test with Claude AI assistant
   - Test with different business types

4. **Performance Tests**:
   - Benchmark tool execution latency
   - Test with concurrent tool calls
   - Profile memory usage

---

## Related Capabilities

- **vector-search**: Provides search functionality exposed by MCP tools
- **query-processing**: Enhances queries before MCP tool execution
- **result-processing**: Formats results for MCP tool responses

---

## Migration Notes

### Backward Compatibility

- Existing MCP tool calls remain functional
- New parameters are optional with sensible defaults
- Enhanced response format is backward compatible
- Legacy clients can ignore new fields

### Client Migration

**No action required** for existing MCP clients:
- Old calls continue to work
- New features are opt-in via parameters
- Response format extends, doesn't replace

**Recommended updates** for MCP clients:
1. Set `use_optimization=true` for better accuracy
2. Set `include_metrics=true` for quality insights
3. Handle new metadata fields in responses
4. Update error handling for new error types

### Rollback Plan

If issues occur:
1. Set `DEFAULT_OPTIMIZATION_ENABLED=False`
2. Set `ENABLE_ENHANCED_RESPONSES=False`
3. MCP tools revert to original behavior
4. No client changes required

### Deployment Strategy

1. **Phase 1**: Deploy MCP server with enhanced tools (disabled by default)
2. **Phase 2**: Enable for testing environment
3. **Phase 3**: Gradual rollout with monitoring (25% → 50% → 100%)
4. **Phase 4**: Make optimization the default
5. **Phase 5: Full production deployment

### Monitoring Requirements

Track MCP-specific metrics:
- Tool call frequency and patterns
- Parameter usage (optimization, strategy)
- Error rates by error type
- Response times by strategy
- User satisfaction (if feedback available)
- Feature adoption rates

---

## Documentation Updates

Required documentation updates:

1. **MCP Tool Reference**:
   - Update tool descriptions
   - Document new parameters
   - Provide usage examples

2. **Integration Guide**:
   - Explain enhanced search
   - Show response format examples
   - Document best practices

3. **Migration Guide**:
   - Backward compatibility notes
   - Recommended client updates
   - Troubleshooting tips

4. **API Documentation**:
   - Auto-generated from tool schemas
   - Include response format examples
   - Document error conditions
