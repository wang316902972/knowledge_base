"""
FAISS 向量数据库更新功能单元测试
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from faiss_server_optimized import FaissVectorDB
from config import Config


@pytest.fixture
def mock_config():
    """Mock 配置对象"""
    config = Mock(spec=Config)
    config.EMBEDDING_DIM = 384
    config.BATCH_SIZE = 32
    config.DEFAULT_BUSINESSTYPE = "test"
    config.DATA_DIR = "/tmp/test_faiss"
    config.INDEX_TYPE = "FlatIP"
    config.MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    config.AUTO_SAVE = False
    config.ENVIRONMENT = "test"

    def validate_businesstype(bt):
        return bt.lower() if bt else "test"

    config._validate_businesstype = validate_businesstype
    config.get_index_file = lambda bt: f"/tmp/test_faiss/{bt}/{bt}_knowledge_base.index"
    config.get_metadata_file = lambda bt: f"/tmp/test_faiss/{bt}/{bt}_knowledge_base.json"

    return config


@pytest.fixture
def mock_db(mock_config):
    """Mock FaissVectorDB 实例"""
    with patch('faiss_server_optimized.SentenceTransformer') as mock_model, \
         patch('faiss_server_optimized.faiss') as mock_faiss:

        # Mock 模型
        mock_embedding_model = MagicMock()
        mock_model.return_value = mock_embedding_model
        mock_embedding_model.encode.return_value = np.random.rand(2, 384).astype(np.float32)

        # Mock FAISS 索引
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_index.is_trained = True
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.read_index.return_value = mock_index

        # 创建实例
        db = FaissVectorDB(mock_config, businesstype="test")
        db.index = mock_index
        db.embedding_model = mock_embedding_model
        db.id_to_chunk = {}
        db.chunk_to_id = {}
        db.dirty = False

        yield db


class TestUpdateTexts:
    """测试 update_texts 方法"""

    def test_update_single_existing_text(self, mock_db):
        """测试更新已存在的文本"""
        # 先添加一个文本
        mock_db.id_to_chunk = {"0": "原始内容"}
        mock_db.chunk_to_id = {"原始内容": "0"}
        mock_db.index.ntotal = 1

        # Mock 嵌入生成
        mock_db.embedding_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)

        # 执行更新
        result = mock_db.update_texts([
            {"old_text": "原始内容", "new_text": "更新后的内容"}
        ])

        # 验证结果
        assert result["success_count"] == 1
        assert result["updated_count"] == 1
        assert result["inserted_count"] == 0
        assert result["failed_count"] == 0
        assert len(result["errors"]) == 0

        # 验证旧文本被删除，新文本被添加
        assert "原始内容" not in mock_db.chunk_to_id
        assert "更新后的内容" in mock_db.chunk_to_id

        # 验证 FAISS 操作被调用
        mock_db.index.remove_ids.assert_called_once()
        mock_db.index.add.assert_called_once()

    def test_upsert_non_existing_text(self, mock_db):
        """测试 upsert 语义：不存在的文本则插入"""
        # Mock 嵌入生成
        mock_db.embedding_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)

        # 执行更新（old_text 不存在）
        result = mock_db.update_texts([
            {"old_text": "不存在的文本", "new_text": "新内容"}
        ])

        # 验证结果
        assert result["success_count"] == 1
        assert result["inserted_count"] == 1
        assert result["updated_count"] == 0
        assert result["failed_count"] == 0

        # 验证新文本被添加
        assert "新内容" in mock_db.chunk_to_id

        # 验证只调用了 add，没有调用 remove_ids
        mock_db.index.remove_ids.assert_not_called()
        mock_db.index.add.assert_called_once()

    def test_update_same_text_skips(self, mock_db):
        """测试 old_text == new_text 时跳过"""
        mock_db.id_to_chunk = {"0": "相同文本"}
        mock_db.chunk_to_id = {"相同文本": "0"}
        mock_db.index.ntotal = 1

        # 执行更新
        result = mock_db.update_texts([
            {"old_text": "相同文本", "new_text": "相同文本"}
        ])

        # 验证结果
        assert result["success_count"] == 0
        assert result["inserted_count"] == 0
        assert result["updated_count"] == 0

        # 验证没有调用 FAISS 操作
        mock_db.index.remove_ids.assert_not_called()
        mock_db.index.add.assert_not_called()

    def test_batch_update_mixed_operations(self, mock_db):
        """测试批量更新：混合更新和新增操作"""
        # 设置初始状态
        mock_db.id_to_chunk = {"0": "文本1", "1": "文本2"}
        mock_db.chunk_to_id = {"文本1": "0", "文本2": "1"}
        mock_db.index.ntotal = 2

        # Mock 嵌入生成（3个新文本）
        mock_db.embedding_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)

        # 执行批量更新
        result = mock_db.update_texts([
            {"old_text": "文本1", "new_text": "文本1_更新"},  # 更新
            {"old_text": "不存在的文本", "new_text": "新文本"},  # 新增
            {"old_text": "文本2", "new_text": "文本2_更新"},  # 更新
        ])

        # 验证结果
        assert result["success_count"] == 3
        assert result["updated_count"] == 2
        assert result["inserted_count"] == 1
        assert result["failed_count"] == 0

    def test_empty_text_validation(self, mock_db):
        """测试空文本验证"""
        result = mock_db.update_texts([
            {"old_text": "", "new_text": "新内容"},
            {"old_text": "旧内容", "new_text": ""},
            {"old_text": "   ", "new_text": "新内容2"},
        ])

        # 验证所有项都失败
        assert result["success_count"] == 0
        assert result["failed_count"] == 3
        assert len(result["errors"]) == 3

        # 验证错误信息
        for error in result["errors"]:
            assert "文本内容为空" in error["error"]

    def test_batch_update_partial_failure(self, mock_db):
        """测试批量更新部分失败"""
        mock_db.id_to_chunk = {"0": "存在的文本"}
        mock_db.chunk_to_id = {"存在的文本": "0"}
        mock_db.index.ntotal = 1

        # Mock 嵌入生成（只生成有效的）
        mock_db.embedding_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)

        # 执行更新：一个有效，一个无效
        result = mock_db.update_texts([
            {"old_text": "存在的文本", "new_text": "更新内容"},  # 有效
            {"old_text": "", "new_text": "空文本"},  # 无效
        ])

        # 验证部分成功
        assert result["success_count"] == 1
        assert result["failed_count"] == 1
        assert len(result["errors"]) == 1

        # 验证错误详情
        error = result["errors"][0]
        assert "index" in error
        assert "error" in error

    def test_dirty_flag_set_after_update(self, mock_db):
        """测试更新后 dirty 标志被设置"""
        mock_db.embedding_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)

        # 初始状态
        assert mock_db.dirty == False

        # 执行更新
        mock_db.update_texts([
            {"old_text": "不存在的", "new_text": "新内容"}
        ])

        # 验证 dirty 被设置
        assert mock_db.dirty == True


class TestUpdateAPI:
    """测试更新 API 端点"""

    @pytest.mark.asyncio
    async def test_update_api_endpoint(self):
        """测试 PUT /update 端点"""
        # 这是一个集成测试，在任务6中实现
        pass

    @pytest.mark.asyncio
    async def test_update_api_with_auto_save(self):
        """测试更新 API 并自动保存"""
        # 这是一个集成测试，在任务6中实现
        pass


class TestEdgeCases:
    """测试边缘情况"""

    def test_update_empty_list(self, mock_db):
        """测试空列表更新"""
        result = mock_db.update_texts([])

        assert result["success_count"] == 0
        assert result["failed_count"] == 0
        assert result["inserted_count"] == 0
        assert result["updated_count"] == 0

        # 验证没有调用 FAISS 操作
        mock_db.index.remove_ids.assert_not_called()
        mock_db.index.add.assert_not_called()

    def test_update_duplicate_old_texts(self, mock_db):
        """测试重复的 old_text 出现多次"""
        mock_db.id_to_chunk = {"0": "重复文本"}
        mock_db.chunk_to_id = {"重复文本": "0"}
        mock_db.index.ntotal = 1

        mock_db.embedding_model.encode.return_value = np.random.rand(2, 384).astype(np.float32)

        # 执行更新：相同的 old_text 出现两次
        result = mock_db.update_texts([
            {"old_text": "重复文本", "new_text": "更新1"},
            {"old_text": "重复文本", "new_text": "更新2"},
        ])

        # 两次都应该成功（第一次删除，第二次当作新增）
        assert result["success_count"] == 2
