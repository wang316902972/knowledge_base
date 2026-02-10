# Project Context

## Purpose

FAISS-GPU is a high-performance vector database service optimized for semantic search and knowledge retrieval. Built on top of Facebook AI Similarity Search (FAISS) with GPU acceleration, it provides:

- **Semantic Vector Search**: Fast similarity search using sentence transformer embeddings
- **MCP Integration**: Native Model Context Protocol support for AI assistant integration
- **Multi-Instance Architecture**: Business-type-based data isolation for multi-tenant scenarios
- **Production-Ready**: Security hardening, performance optimization, and comprehensive monitoring

The project solves critical issues from earlier versions including pickle security vulnerabilities, concurrency problems, and lack of proper data isolation.

## Tech Stack

### Core Technologies
- **Python 3.10+**: Primary development language
- **FastAPI 0.104+**: Modern async web framework for REST API
- **FAISS-GPU 1.12+**: High-performance vector similarity search with GPU acceleration
- **sentence-transformers 2.2+**: Multilingual text embeddings (MiniLM-L12-v2)
- **MCP 0.9+**: Model Context Protocol for AI assistant integration
- **Pydantic 2.0+**: Data validation and settings management
- **NumPy 1.24+**: Numerical computing backbone

### Infrastructure
- **Uvicorn**: ASGI server for production deployment
- **Docker**: Containerization support
- **GPU Support**: CUDA 12.x for FAISS GPU acceleration

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting (23.0+)
- **isort**: Import sorting (5.12+)
- **flake8**: Linting (6.0+)
- **httpx**: Async HTTP client for testing

## Project Conventions

### Code Style

**Formatting Rules**:
- Use `black` for code formatting (strict mode, 88 character line length)
- Use `isort` for import organization
- Follow PEP 8 naming conventions with these specifics:
  - Classes: `PascalCase` (e.g., `FaissVectorDB`, `Config`)
  - Functions/Methods: `snake_case` (e.g., `get_index_file`, `search_knowledge`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_BUSINESSTYPE`, `MAX_TOP_K`)
  - Private methods: `_leading_underscore` (e.g., `_validate_businesstype`)

**Type Hints**:
- All functions must have type annotations using Python 3.10+ syntax
- Use `list[str]` instead of `List[str]` (no typing module imports)
- Return types are mandatory for all public methods

**Documentation**:
- Chinese language for user-facing documentation (README, API docs)
- English language for code comments and technical documentation
- Docstrings for all public classes and methods using Google style

### Architecture Patterns

**Multi-Instance Architecture**:
- Business-type-based data isolation (`businesstype` parameter)
- Each business type gets separate index and metadata files
- Dynamic path generation: `data/{businesstype}/{businesstype}_knowledge_base.*`

**Core Components**:
1. **Config Class**: Centralized configuration with environment variable support
2. **FaissVectorDB Class**: Thread-safe vector database core with reentrant locks
3. **MCP Server**: Model Context Protocol integration for AI tools
4. **Search Optimization**: Diversity re-ranking and relevance filtering

**Design Principles**:
- **Security First**: JSON serialization only (no pickle), input validation, path sanitization
- **Thread Safety**: Reentrant locks for all shared state operations
- **Configuration as Code**: Environment-based configuration with validation
- **Graceful Degradation**: Proper error handling and recovery mechanisms

**Data Flow**:
```
Document Input → Text Chunking → Embedding Generation → FAISS Index → Metadata Storage (JSON)
                                                            ↓
Query Input → Embedding Generation → Similarity Search → Result Ranking → Return Results
```

**Storage Format**:
- FAISS Index: Binary format (.index) for vector data
- Metadata: JSON format (.json) for human-readable text chunks
- Atomic file operations to prevent corruption

### Testing Strategy

**Current State**: Test framework setup but no test files yet (needs implementation)

**Testing Requirements**:
- **Unit Tests**: All core logic with ≥80% coverage requirement
  - Config validation and path generation
  - Vector database operations (add, delete, search)
  - Thread safety under concurrent load
  - Input validation and error handling

- **Integration Tests**: API endpoints using httpx
  - MCP tool interactions
  - FastAPI endpoint responses
  - End-to-end workflows (add → search → delete)

- **Security Tests**: Input validation and sanitization
  - Path injection attempts
  - Invalid business types
  - Boundary condition testing

- **Performance Tests**: Load testing for concurrent operations
  - Benchmark search latency
  - Throughput under load
  - Memory usage profiling

**Test Organization**:
```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # API and MCP integration tests
├── security/       # Security and validation tests
└── performance/    # Benchmark and load tests
```

### Git Workflow

**Branching Strategy**:
- `main`: Production-ready code, protected branch
- `feature/ai/*`: Feature development branches (current pattern)
- Example: `feature/ai/business-type-storage`

**Commit Message Conventions**:
Follow Conventional Commits format:
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `docs`: Documentation changes
- `test`: Test additions/changes
- `chore`: Build/process changes

**Scopes**:
- `mcp`: MCP server changes
- `storage`: Data storage architecture
- `api`: FastAPI endpoints
- `config`: Configuration changes
- `search`: Search optimization

**Recent Examples**:
- `refactor(storage): migrate to business-type-based multi-instance architecture`
- `feat(mcp): 实现 MCP 标准协议和工具包生态`
- `feat(mcp): 集成 Model Context Protocol 支持`

**Workflow**:
1. Create feature branch from `main`: `git checkout -b feature/ai/your-feature`
2. Implement changes with commit messages following convention
3. Use OpenSpec for significant changes (see openspec/AGENTS.md)
4. Create pull request for code review
5. Merge to `main` after approval

## Domain Context

**Vector Database**: Specialized database for efficient similarity search in high-dimensional vector spaces. FAISS (Facebook AI Similarity Search) provides state-of-the-art performance for:
- Approximate nearest neighbor search
- GPU-accelerated computations
- Multiple index types (Flat, IVF, HNSW) for different use cases

**Semantic Search**: Search based on meaning rather than keyword matching. Uses sentence transformers to convert text to dense vector embeddings where semantic similarity translates to geometric proximity.

**MCP (Model Context Protocol)**: Open protocol for integrating AI assistants with tools and data sources. This project implements MCP server specification to provide vector search capabilities to Claude and other AI assistants.

**Business Type Isolation**: Multi-tenancy pattern where each "business type" (tenant, project, or domain) gets completely isolated data storage. Prevents cross-contamination and enables per-configuration tuning.

**Index Types**:
- **FlatIP/FlatL2**: Exact search, suitable for <100K vectors
- **IVFFlat**: Approximate search, requires training, suitable for 100K-10M vectors
- **HNSW**: Graph-based approximate search, fastest for >10M vectors

**Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2` produces 384-dimensional vectors optimized for semantic similarity across multiple languages.

## Important Constraints

### Technical Constraints
- **GPU Dependency**: Requires NVIDIA GPU with CUDA 12.x for faiss-gpu (falls back to CPU if unavailable)
- **Model Size**: Embedding model ~470MB, downloaded on first use
- **Memory Usage**: Index size grows linearly with vector count (~1.5KB per 384-dim vector + metadata)
- **File System**: Requires local file system storage (not network/S3 mounts) for FAISS index files

### Security Constraints
- **No Pickle**: Absolutely no pickle serialization due to RCE vulnerability (JSON only)
- **Path Validation**: All business types must pass strict validation (alphanumeric, underscore, hyphen only, max 50 chars)
- **Input Bounds**: All user inputs have enforced minimum/maximum values
- **Thread Safety**: All state mutations protected by reentrant locks

### Performance Constraints
- **Concurrent Writers**: Single writer at a time due to index lock (multiple readers OK)
- **Batch Size**: Maximum 100 texts per batch add operation
- **Search Latency**: Target <100ms for top-5 search on <1M vectors
- **Memory**: Recommended minimum 8GB RAM for 1M vectors

### Operational Constraints
- **Manual Save**: Auto-save disabled by default (requires explicit save or configure AUTO_SAVE=true)
- **No Index Merging**: Cannot merge separate indices without rebuilding
- **Atomic Operations**: Index and metadata must be saved together to maintain consistency

## External Dependencies

### AI/ML Models
- **sentence-transformers model**: `paraphrase-multilingual-MiniLM-L12-v2` from HuggingFace
  - Auto-downloaded on first startup
  - Cached locally in `~/.cache/torch/sentence_transformers/`
  - Requires internet connection for initial download

### Python Packages (via PyPI)
- **FAISS**: Meta Research distribution (faiss-gpu-cu12 package)
- **sentence-transformers**: HuggingFace ecosystem
- **FastAPI/Uvicorn**: FastAPI ecosystem
- **MCP SDK**: Model Context Protocol Python SDK

### Infrastructure Services
- **None**: Currently standalone, no external database or message queue dependencies

### Optional Production Integrations
- **Docker**: For containerization (Dockerfile present)
- **Prometheus**: For metrics collection (prometheus-client installed)
- **Load Balancer**: For multi-instance deployment (Nginx, HAProxy, etc.)

### API Consumer Context
- **MCP Clients**: AI assistants (Claude, etc.) via stdio or HTTP transport
- **REST Clients**: Direct HTTP API consumers (FastAPI endpoints)
- **Integration Points**: Expected to be integrated with:
  - RAG (Retrieval-Augmented Generation) pipelines
  - Knowledge management systems
  - Semantic search applications
  - Multi-tenant AI services
