from types import SimpleNamespace

import pytest

from bm25_search import BM25SearchResult, BM25SearchStrategy
from faiss_server_optimized import (
    FaissVectorDB,
    IndexMetadataConsistencyError,
)
from metadata_store import SQLiteMetadataStore
from retrieval_enhancement import ResultFusion, SearchResult


def test_bm25_refreshes_from_sqlite_revision(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite")
    vector_ids = store.add_chunks(
        [
            "GitNexus provides repository graph search",
            "FAISS provides vector similarity search",
            "SQLite stores active metadata state",
        ]
    )
    strategy = BM25SearchStrategy(
        None,
        documents_provider=store.get_active_text_map,
        revision_provider=store.get_chunks_revision,
    )

    assert strategy.get_stats()["document_count"] == 3
    assert strategy.get_stats()["metadata_source"] == "provider"

    store.soft_delete_vector_ids([vector_ids[0]])
    strategy.search("GitNexus", top_k=5, min_score=-100.0)

    assert strategy.get_stats()["document_count"] == 2
    assert str(vector_ids[0]) not in strategy.chunk_ids


def test_weighted_rrf_deduplicates_strategies_by_stable_id():
    fusion = ResultFusion({"vector": 0.7, "bm25": 0.2, "exact": 0.1})
    results = fusion.fuse(
        {
            "vector": [
                SearchResult("same", 0.8, 0.8, faiss_id=7, strategies_used=["vector"])
            ],
            "bm25": [
                SearchResult("same", 2.0, 0.2, faiss_id=7, chunk_id="7", strategies_used=["bm25"])
            ],
            "exact": [
                SearchResult("same", 1.0, 1.0, faiss_id=7, strategies_used=["exact"])
            ],
        },
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].faiss_id == 7
    assert results[0].match_type == "hybrid"
    assert results[0].strategies_used == ["vector", "bm25", "exact"]
    assert results[0].rrf_score == pytest.approx(1.0 / 61.0)


def test_bm25_result_uses_numeric_chunk_id_as_faiss_id():
    result = SearchResult.from_bm25(
        BM25SearchResult(chunk_id="42", text="answer", score=3.0)
    )

    assert result.faiss_id == 42


def test_rrf_keeps_vector_score_when_bm25_score_has_larger_scale():
    fusion = ResultFusion({"vector": 0.7, "bm25": 0.2, "exact": 0.1})
    results = fusion.fuse(
        {
            "vector": [
                SearchResult("same", 0.4, 0.8, faiss_id=7, strategies_used=["vector"])
            ],
            "bm25": [
                SearchResult("same", 12.0, 1.0, faiss_id=7, chunk_id="7", strategies_used=["bm25"])
            ],
        },
        top_k=1,
    )

    assert results[0].similarity_score == 0.4
    assert results[0].relevance_score == 0.8


def test_consistency_gate_rejects_missing_active_faiss_id(tmp_path):
    db = FaissVectorDB.__new__(FaissVectorDB)
    db.metadata_store = SQLiteMetadataStore(tmp_path / "metadata.sqlite")
    db.metadata_store.add_chunks_with_ids([(2, "missing")])
    db.index = SimpleNamespace(ntotal=2)

    with pytest.raises(IndexMetadataConsistencyError):
        db._validate_index_metadata_consistency()
