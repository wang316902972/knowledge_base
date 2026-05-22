# Million Chunk Updatable Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrofit the FAISS knowledge base for ~1M chunks with stable vector IDs, SQLite-backed metadata, soft deletes, and compaction.

**Architecture:** FAISS remains the vector search engine. SQLite becomes the source of truth for chunk metadata and lifecycle state. Updates mark old chunks inactive, append new vectors with stable integer IDs via IDMap/add_with_ids when possible, filter inactive search results, and rebuild active vectors during compaction.

**Tech Stack:** Python 3.12, FastAPI, FAISS Python bindings, SQLite stdlib, Pydantic, existing MCP servers.

---

### Task 1: Metadata Store

**Files:**
- Create: `metadata_store.py`
- Test: `tests/test_metadata_store.py`

- [ ] Write tests for adding chunks, soft deleting by text, listing active IDs, and metrics.
- [ ] Run metadata tests and verify they fail because `metadata_store` does not exist.
- [ ] Implement `SQLiteMetadataStore` using stdlib `sqlite3` with WAL mode, integer vector IDs, and `active/deleted` state.
- [ ] Run metadata tests and verify they pass.

### Task 2: FAISS Core Integration

**Files:**
- Modify: `faiss_server_optimized.py`
- Test: `tests/test_update.py`

- [ ] Add tests proving `add_texts`, `delete_texts`, and `update_texts` use stable IDs and leave deleted chunks out of search results.
- [ ] Run focused update tests and verify they fail on current behavior.
- [ ] Wire `FaissVectorDB` to `SQLiteMetadataStore` while keeping legacy JSON metadata load/save compatible.
- [ ] Add active-result filtering and stable `add_with_ids` fallback behavior.
- [ ] Add `compact_index()` to rebuild the FAISS index from active metadata and vectors when reconstruction is supported.
- [ ] Run focused update tests and verify they pass.

### Task 3: API and MCP Surface

**Files:**
- Modify: `faiss_server_optimized.py`
- Modify: `mcp_server.py`
- Modify: `mcp_http_server.py`

- [ ] Expose update and compaction through REST and MCP tool lists.
- [ ] Add response metadata showing active/deleted counts and compaction need.
- [ ] Run syntax/import checks.

### Task 4: Verification

**Files:**
- All changed files

- [ ] Run available tests.
- [ ] Run `python3 -m py_compile` on changed Python files.
- [ ] Summarize dependency blockers if FAISS/pytest are unavailable locally.
