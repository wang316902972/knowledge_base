# Spec Delta: Query Processing Capability

## Capability Overview
**Capability ID**: query-processing
**Change Type**: ADDED
**Related Changes**: vector-search, result-processing

---

## ADDED Requirements

### Requirement: Domain Term Expansion

The system MUST expand user queries with domain-specific terminology to improve retrieval accuracy.

#### Scenario: Expand query with domain synonyms
**Given** a user query containing "反作弊" (anti-cheat)
**When** the query enhancement module processes the query
**Then** the system SHOULD generate expanded queries including "anti-cheat", "反外挂", "防作弊", "安全检测"
**And** the system SHOULD execute vector search with all expanded query variants
**And** the system SHOULD merge results from all query variants

#### Scenario: Multi-lingual query expansion
**Given** a user query in Chinese "预制件"
**When** the query enhancement module processes the query
**Then** the system SHOULD generate English expansions "prefab", "component", "module", "plugin"
**And** the system SHOULD maintain the original Chinese query
**And** the system SHOULD search with both Chinese and English variants

#### Scenario: Technical term normalization
**Given** a user query with inconsistent terminology "AI辅助 反作弊"
**When** the query normalizer processes the query
**Then** the system SHOULD normalize to "ai 辅助 反作弊"
**And** the system SHOULD remove extra whitespace
**And** the system SHOULD lowercase English terms
**And** the system SHOULD preserve Chinese characters

### Requirement: Query Caching

The system MUST cache query embeddings and expanded queries to improve performance.

#### Scenario: Cache query embeddings
**Given** a user query that has been searched before
**When** the query is received
**Then** the system SHOULD check the query cache for existing embeddings
**And** if found, the system SHOULD reuse the cached embedding
**And** the system SHOULD skip re-encoding the query
**And** the system SHOULD update the cache access timestamp

#### Scenario: Cache expanded queries
**Given** a user query "反作弊分析"
**When** the query expansion module generates 5 expanded variants
**Then** the system SHOULD cache all expanded variants with the original query
**And** subsequent identical queries SHOULD reuse cached expansions
**And** the system SHOULD implement TTL-based cache invalidation (default 1 hour)

#### Scenario: Cache hit rate monitoring
**Given** the query caching system is operational
**When** search operations are performed
**Then** the system SHOULD track cache hits and misses
**And** the system SHOULD calculate cache hit rate
**And** the system SHOULD log cache performance metrics
**And** the target cache hit rate SHOULD be ≥30%

---

## MODIFIED Requirements

### Requirement: Query Preprocessing

The system MUST perform comprehensive query enhancement including expansion, normalization, and augmentation, replacing the previous basic text cleaning.

**Previous Behavior**: The system performed basic text cleaning on queries.

**New Behavior**: The system performs comprehensive query enhancement including expansion, normalization, and augmentation.

#### Scenario: Enhanced query preprocessing pipeline
**Given** a raw user query
**When** the query preprocessing pipeline executes
**Then** the system SHOULD perform the following steps in order:
1. Normalize whitespace and special characters
2. Detect and mark domain-specific terms
3. Generate expanded query variants (max 5)
4. Encode all query variants to embeddings
5. Cache embeddings and expansions
**And** the system SHOULD return enhanced query object with all variants

#### Scenario: Query metadata extraction
**Given** an enhanced query object
**When** the query is passed to search strategies
**Then** the query object SHOULD contain:
- Original query text
- Normalized query text
- Expanded query variants (list)
- Detected domain terms (list)
- Query embedding (cached or computed)
- Query metadata (language, length, complexity)

### Requirement: Query Validation

The system MUST perform comprehensive validation including semantic analysis and intent detection, enhancing the previous basic validation.

**Previous Behavior**: Basic validation of query length and format.

**New Behavior**: Comprehensive validation including semantic analysis and intent detection.

#### Scenario: Validate query quality
**Given** a user query
**When** query validation is performed
**Then** the system SHOULD check:
- Query length (min 2, max 500 characters)
- Character encoding (valid UTF-8)
- Language detection (Chinese, English, or mixed)
- Malicious content detection (SQL injection, XSS attempts)
**And** the system SHOULD reject invalid queries with appropriate error messages
**And** the system SHOULD log rejected queries for security monitoring

#### Scenario: Detect query intent
**Given** a user query
**When** intent detection is performed
**Then** the system SHOULD classify the query intent:
- Exact match (id search, exact phrase)
- Semantic search (natural language)
- Keyword search (specific terms)
- Fuzzy search (typo tolerance)
**And** the system SHOULD pass intent information to search strategies
**And** the system SHOULD adjust search parameters based on detected intent

---

## DELETED Requirements

None. All existing requirements are preserved or enhanced.

---

## Implementation Notes

### Configuration Changes

New configuration parameters added to `QueryConfig`:
```python
# Query enhancement
ENABLE_QUERY_EXPANSION: bool = True
MAX_EXPANDED_QUERIES: int = 5
DOMAIN_TERM_DICT_PATH: str = "config/domain_terms.json"

# Query caching
ENABLE_QUERY_CACHE: bool = True
QUERY_CACHE_SIZE: int = 1000
QUERY_CACHE_TTL: int = 3600  # seconds

# Query validation
MIN_QUERY_LENGTH: int = 2
MAX_QUERY_LENGTH: int = 500
ENABLE_INTENT_DETECTION: bool = True
```

### Data Structures

**Enhanced Query Object**:
```python
@dataclass
class EnhancedQuery:
    original: str
    normalized: str
    expanded_variants: List[str]
    domain_terms: List[str]
    embeddings: Dict[str, np.ndarray]  # variant -> embedding
    metadata: QueryMetadata
    cache_key: str

@dataclass
class QueryMetadata:
    language: str  # 'zh', 'en', 'mixed'
    length: int
    complexity: str  # 'simple', 'medium', 'complex'
    intent: str  # 'exact', 'semantic', 'keyword', 'fuzzy'
    has_domain_terms: bool
    created_at: datetime
```

### Domain Term Dictionary Format

```json
{
  "反作弊": ["anti-cheat", "反外挂", "防作弊", "安全检测"],
  "归因": ["attribution", "溯源", "分析", "追踪"],
  "预制件": ["prefab", "组件", "模块", "plugin"],
  "AI诊断": ["ai diagnosis", "ai诊断", "智能诊断", "自动分析"],
  "关联分析": ["correlation analysis", "关联查询", "横向关联", "网络分析"]
}
```

### API Changes

**Query Enhancement API** (new):
```python
def enhance_query(query: str) -> EnhancedQuery:
    """Enhance a query with expansion and normalization"""

def expand_query(query: str, max_variants: int = 5) -> List[str]:
    """Generate expanded query variants"""

def normalize_query(query: str) -> str:
    """Normalize query text"""

def detect_domain_terms(query: str) -> List[str]:
    """Detect domain-specific terms in query"""

def cache_query(query: EnhancedQuery) -> None:
    """Cache enhanced query and embeddings"""

def get_cached_query(cache_key: str) -> Optional[EnhancedQuery]:
    """Retrieve cached query if available"""
```

### Performance Impact

- **Query Processing Latency**: +20-30ms for expansion and normalization
- **Cache Benefit**: -40-50ms for cached queries
- **Memory**: +5MB for query cache (1000 queries)
- **Net Impact**: Neutral to positive with caching

### Testing Requirements

1. **Unit Tests**:
   - Test query expansion correctness
   - Test query normalization rules
   - Test domain term detection
   - Test query caching behavior

2. **Integration Tests**:
   - Test enhanced query object creation
   - Test cache hit/miss scenarios
   - Test intent detection accuracy

3. **Performance Tests**:
   - Benchmark query processing latency
   - Measure cache effectiveness
   - Profile memory usage

4. **Validation Tests**:
   - Test query validation rules
   - Test malicious query detection
   - Test intent classification accuracy

---

## Related Capabilities

- **vector-search**: Consumes enhanced queries for search
- **result-processing**: Uses query metadata for result filtering

---

## Migration Notes

### Backward Compatibility

- Existing query processing remains functional
- Query enhancement is additive
- Caching is transparent to callers
- Feature flags can disable new features

### Rollback Plan

If issues occur:
1. Set `ENABLE_QUERY_EXPANSION=False`
2. Set `ENABLE_QUERY_CACHE=False`
3. System reverts to basic query processing
4. Clear query cache if needed

### Deployment Strategy

1. **Phase 1**: Deploy query processing enhancements
2. **Phase 2**: Enable for testing environment
3. **Phase 3**: Monitor cache hit rates and query quality
4. **Phase 4**: Full production deployment
