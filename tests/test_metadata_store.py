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


if __name__ == "__main__":
    unittest.main()
