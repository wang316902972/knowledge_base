import tempfile
import unittest
from pathlib import Path

from metadata_store import SQLiteMetadataStore


class SQLiteMetadataStoreTests(unittest.TestCase):
    def test_add_chunks_allocates_stable_vector_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteMetadataStore(Path(tmpdir) / "metadata.sqlite")

            ids = store.add_chunks(["alpha", "beta"], source_document="doc-1")

            self.assertEqual(ids, [1, 2])
            self.assertEqual(store.get_text(1), "alpha")
            self.assertEqual(store.get_text(2), "beta")
            self.assertEqual(store.get_active_vector_ids(), [1, 2])

    def test_soft_delete_by_text_keeps_ids_inactive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteMetadataStore(Path(tmpdir) / "metadata.sqlite")
            store.add_chunks(["alpha", "beta", "alpha"], source_document="doc-1")

            deleted = store.soft_delete_texts(["alpha"])

            self.assertEqual(deleted, [1, 3])
            self.assertFalse(store.is_active(1))
            self.assertTrue(store.is_active(2))
            self.assertFalse(store.is_active(3))
            self.assertEqual(store.get_active_vector_ids(), [2])

    def test_compaction_metrics_report_deleted_ratio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteMetadataStore(Path(tmpdir) / "metadata.sqlite")
            store.add_chunks(["alpha", "beta", "gamma", "delta"])
            store.soft_delete_texts(["alpha", "beta"])

            metrics = store.get_metrics()

            self.assertEqual(metrics["total_chunks"], 4)
            self.assertEqual(metrics["active_chunks"], 2)
            self.assertEqual(metrics["deleted_chunks"], 2)
            self.assertEqual(metrics["deleted_ratio"], 0.5)
            self.assertTrue(metrics["needs_compaction"])

    def test_sql_template_upsert_separates_search_text_and_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteMetadataStore(Path(temp_dir) / "metadata.sqlite")
            record = {
                "external_id": "risk:booth:v1",
                "dataset_id": "risk",
                "template_id": "booth",
                "intent_key": "booth_trade_records",
                "canonical_template": "角色 {role_id} {date} 摆摊交易记录",
                "search_text": "摆摊交易记录 角色 日期",
                "required_slots": {"role_id": "uint64", "date": "date"},
                "sql_template": (
                    "SELECT role_id FROM booth_log "
                    "WHERE role_id = :role_id AND dt = :date"
                ),
                "schema_fingerprint": "schema-v1",
                "template_version": 1,
                "status": "active",
                "source": "reviewed_execution",
            }

            inserted = store.upsert_sql_template(record, vector_id=7)
            self.assertEqual(inserted["vector_id"], 7)
            self.assertEqual(inserted["required_slots"]["date"], "date")

            updated_record = {
                **record,
                "sql_template": record["sql_template"] + " LIMIT 100",
                "success_count": 3,
            }
            updated = store.upsert_sql_template(updated_record, vector_id=7)
            self.assertEqual(updated["success_count"], 3)
            self.assertIn("LIMIT 100", updated["sql_template"])

            matches = store.find_sql_templates(
                "risk",
                intent_key="booth_trade_records",
                canonical_template=record["canonical_template"],
            )
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["external_id"], "risk:booth:v1")

            by_vector = store.get_sql_templates_by_vector_ids([7], "risk")
            self.assertEqual(by_vector[7]["search_text"], record["search_text"])
            self.assertNotIn(record["sql_template"], record["search_text"])

            metrics = store.get_sql_template_metrics()
            self.assertEqual(metrics["total_templates"], 1)
            self.assertEqual(metrics["by_status"], {"active": 1})
            store.close()

    def test_sql_template_lifecycle_tracks_outcomes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteMetadataStore(Path(temp_dir) / "metadata.sqlite")
            record = {
                "external_id": "risk:booth:v1",
                "dataset_id": "risk",
                "template_id": "booth",
                "intent_key": "booth_trade_records",
                "canonical_template": "角色 {role_id} {date} 摆摊交易记录",
                "search_text": "摆摊交易记录 角色 日期",
                "required_slots": {"role_id": "uint64", "date": "date"},
                "sql_template": "SELECT 1 WHERE role_id = :role_id",
                "schema_fingerprint": "schema-v1",
                "template_version": 1,
                "status": "active",
                "source": "manual",
            }
            store.upsert_sql_template(record, vector_id=9)

            tracked = store.record_sql_template_outcome(
                "risk:booth:v1",
                "shadow_match",
            )
            self.assertEqual(tracked["shadow_match_count"], 1)

            updated = store.upsert_sql_template(
                {**record, "search_text": "摆摊交易记录 角色 日期 明细"},
                vector_id=10,
            )
            self.assertEqual(updated["shadow_match_count"], 1)

            disabled = store.set_sql_template_status(
                "risk:booth:v1",
                "disabled",
                vector_id=None,
            )
            self.assertEqual(disabled["status"], "disabled")
            self.assertIsNone(disabled["vector_id"])
            self.assertEqual(store.find_sql_templates("risk"), [])
            store.close()

    def test_delete_sql_template_removes_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteMetadataStore(Path(temp_dir) / "metadata.sqlite")
            record = {
                "external_id": "risk:booth:v1",
                "dataset_id": "risk",
                "template_id": "booth",
                "intent_key": "booth_trade_records",
                "canonical_template": "角色 {role_id} 摆摊交易记录",
                "search_text": "摆摊交易记录",
                "required_slots": {"role_id": "uint64"},
                "sql_template": "SELECT 1 WHERE role_id = :role_id",
                "schema_fingerprint": "schema-v1",
                "template_version": 1,
                "status": "active",
            }
            store.upsert_sql_template(record, vector_id=11)

            deleted = store.delete_sql_template("risk:booth:v1")

            self.assertEqual(deleted["external_id"], "risk:booth:v1")
            self.assertIsNone(store.get_sql_template("risk:booth:v1"))
            self.assertIsNone(store.delete_sql_template("risk:booth:v1"))
            store.close()

    def test_list_sql_templates_returns_lifecycle_details(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteMetadataStore(Path(temp_dir) / "metadata.sqlite")
            base = {
                "dataset_id": "risk",
                "intent_key": "trade_records",
                "canonical_template": "角色 {role_id} 交易记录",
                "search_text": "角色交易记录",
                "required_slots": {"role_id": "uint64"},
                "sql_template": "SELECT 1 WHERE role_id = :role_id",
                "schema_fingerprint": "schema-v1",
                "template_version": 1,
                "status": "active",
                "source": "manual",
            }
            store.upsert_sql_template(
                {**base, "external_id": "risk:first:v1", "template_id": "first"},
                vector_id=11,
            )
            store.upsert_sql_template(
                {**base, "external_id": "risk:second:v1", "template_id": "second"},
                vector_id=12,
            )
            store.record_sql_template_outcome("risk:second:v1", "reuse")

            templates = store.list_sql_templates()

            self.assertEqual(
                [template["external_id"] for template in templates],
                ["risk:second:v1", "risk:first:v1"],
            )
            self.assertEqual(templates[0]["reuse_count"], 1)
            self.assertIsNotNone(templates[0]["last_used_at"])
            store.close()


if __name__ == "__main__":
    unittest.main()
