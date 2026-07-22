import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import faiss_server_optimized as server
from faiss_server_optimized import FaissVectorDB
from metadata_store import SQLiteMetadataStore


@pytest.fixture
def template_db(tmp_path):
    db = FaissVectorDB.__new__(FaissVectorDB)
    db.lock = threading.RLock()
    db.metadata_store = SQLiteMetadataStore(tmp_path / "metadata.sqlite")
    db.index = MagicMock()
    db.index.ntotal = 0
    db.embedding_model = MagicMock()
    db.embedding_model.encode.return_value = np.ones((1, 384), dtype=np.float32)
    db.config = SimpleNamespace(BATCH_SIZE=32, MAX_CHUNK_SIZE=5000)
    db.id_to_chunk = {}
    db.chunk_to_id = {}
    db.dirty = False
    yield db
    db.metadata_store.close()


def booth_template(**overrides):
    record = {
        "external_id": "risk:booth:v1",
        "dataset_id": "risk",
        "template_id": "booth",
        "intent_key": "booth_trade_records",
        "canonical_template": "角色 {role_id} {date} 摆摊交易记录",
        "search_text": "摆摊交易记录 角色 日期 明细",
        "required_slots": {"role_id": "uint64", "date": "date"},
        "sql_template": (
            "SELECT role_id FROM booth_log " "WHERE role_id = :role_id AND dt = :date"
        ),
        "schema_fingerprint": "schema-v1",
        "template_version": 1,
        "status": "active",
        "source": "reviewed_execution",
    }
    return {**record, **overrides}


def test_upsert_sql_template_indexes_only_search_text(template_db):
    result = template_db.upsert_sql_template(booth_template())

    assert result["action"] == "inserted"
    stored = result["template"]
    assert stored["vector_id"] is not None
    assert template_db.metadata_store.get_text(stored["vector_id"]) == (
        "摆摊交易记录 角色 日期 明细"
    )
    assert "SELECT" not in template_db.chunk_to_id
    template_db.index.add_with_ids.assert_called_once()


def test_upsert_sql_template_replaces_changed_search_vector(template_db):
    first = template_db.upsert_sql_template(booth_template())["template"]
    template_db.index.add_with_ids.reset_mock()

    second = template_db.upsert_sql_template(
        booth_template(search_text="角色摆摊交易查询 日期 区服")
    )["template"]

    assert second["vector_id"] != first["vector_id"]
    assert not template_db.metadata_store.is_active(first["vector_id"])
    assert template_db.metadata_store.is_active(second["vector_id"])
    template_db.index.add_with_ids.assert_called_once()


def test_search_sql_templates_combines_exact_and_semantic(template_db):
    stored = template_db.upsert_sql_template(booth_template())["template"]
    template_db.index.ntotal = 1
    template_db.search = MagicMock(
        return_value=[
            {
                "faiss_id": stored["vector_id"],
                "similarity_score": 0.9,
                "relevance_score": 0.8,
                "lexical_score": 0.6,
                "strategies_used": ["vector", "bm25"],
            }
        ]
    )

    exact = template_db.search_sql_templates(
        "risk",
        "查询角色摆摊交易",
        intent_key="booth_trade_records",
        canonical_template="角色 {role_id} {date} 摆摊交易记录",
    )
    assert len(exact) == 1
    assert exact[0]["match_type"] == "template_exact"
    assert exact[0]["sql_template"].startswith("SELECT")

    semantic = template_db.search_sql_templates(
        "risk",
        "查询角色摆摊交易",
    )
    assert len(semantic) == 1
    assert semantic[0]["match_type"] == "template_semantic"
    assert semantic[0]["rerank_score"] == pytest.approx(0.74)


def test_template_status_removes_and_restores_search_vector(template_db):
    active = template_db.upsert_sql_template(booth_template())["template"]

    disabled = template_db.set_sql_template_status(
        active["external_id"],
        "disabled",
    )
    assert disabled["status"] == "disabled"
    assert disabled["vector_id"] is None
    assert not template_db.metadata_store.is_active(active["vector_id"])

    restored = template_db.set_sql_template_status(
        active["external_id"],
        "active",
    )
    assert restored["status"] == "active"
    assert restored["vector_id"] is not None


def test_delete_template_removes_record_and_search_vector(template_db):
    active = template_db.upsert_sql_template(booth_template())["template"]

    deleted = template_db.delete_sql_template(active["external_id"])

    assert deleted["external_id"] == active["external_id"]
    assert template_db.metadata_store.get_sql_template(active["external_id"]) is None
    assert not template_db.metadata_store.is_active(active["vector_id"])
    assert template_db.list_sql_templates() == []
    assert template_db.search_sql_templates("risk", "摆摊交易记录") == []


def test_template_validation_rejects_undeclared_placeholders(template_db):
    with pytest.raises(ValueError, match="placeholders"):
        template_db.upsert_sql_template(
            booth_template(
                sql_template=(
                    "SELECT role_id FROM booth_log "
                    "WHERE role_id = :role_id AND dt = :missing_date"
                )
            )
        )


def test_template_migration_defaults_to_validation_only(template_db):
    result = template_db.migrate_sql_templates([booth_template()])

    assert result["dry_run"] is True
    assert result["valid_count"] == 1
    assert result["applied_count"] == 0
    assert template_db.metadata_store.get_sql_template("risk:booth:v1") is None


def test_sql_document_audit_reports_duplicates_and_invalid_chunks(template_db):
    template_db.metadata_store.add_chunks(
        [
            "Question: 查询摆摊交易\nSQL:\nSELECT 1 FROM booth_log",
            "Question: 查询摆摊交易。\nSQL:\nSELECT 2 FROM booth_log",
            "没有结构化标记的普通文档",
        ]
    )

    audit = template_db.audit_sql_documents()

    assert audit["total_chunks"] == 3
    assert audit["parseable_count"] == 2
    assert audit["invalid_count"] == 1
    assert audit["duplicate_question_groups"] == 1


@pytest.mark.asyncio
async def test_template_mcp_tools_are_listed_and_executable(
    template_db,
    monkeypatch,
):
    monkeypatch.setattr(
        server,
        "vector_db_manager",
        SimpleNamespace(get_instance=lambda _businesstype: template_db),
    )
    monkeypatch.setattr(server.config, "AUTO_SAVE", False)
    tool_names = {tool.name for tool in server.get_mcp_tools()}
    assert {
        "template_upsert",
        "template_search",
        "template_list",
        "template_status",
        "template_delete",
        "template_outcome",
        "template_stats",
        "template_migrate",
        "sql_document_audit",
    }.issubset(tool_names)

    upsert_result = await server.execute_mcp_tool(
        "template_upsert",
        {"businesstype": "test", "record": booth_template()},
    )
    upsert_payload = json.loads(upsert_result["content"][0]["text"])
    assert upsert_payload["action"] == "inserted"

    list_result = await server.execute_mcp_tool(
        "template_list",
        {"businesstype": "test"},
    )
    list_payload = json.loads(list_result["content"][0]["text"])
    assert list_payload["total_found"] == 1
    assert list_payload["templates"][0]["template_id"] == "booth"
    assert list_payload["templates"][0]["reuse_count"] == 0

    search_result = await server.execute_mcp_tool(
        "template_search",
        {
            "businesstype": "test",
            "dataset_id": "risk",
            "query": "",
            "intent_key": "booth_trade_records",
            "canonical_template": "角色 {role_id} {date} 摆摊交易记录",
        },
    )
    search_payload = json.loads(search_result["content"][0]["text"])
    assert search_payload["total_found"] == 1
    assert search_payload["margin"] == 1.0

    delete_result = await server.execute_mcp_tool(
        "template_delete",
        {
            "businesstype": "test",
            "external_id": "risk:booth:v1",
        },
    )
    delete_payload = json.loads(delete_result["content"][0]["text"])
    assert delete_payload == {
        "external_id": "risk:booth:v1",
        "status": "deleted",
    }
    assert template_db.list_sql_templates() == []
