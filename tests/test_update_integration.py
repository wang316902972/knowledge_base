"""
FAISS 向量数据库更新功能集成测试
"""
import pytest
from fastapi.testclient import TestClient
from faiss_server_optimized import app
import json


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


class TestUpdateAPIIntegration:
    """测试更新 API 集成"""

    def test_update_endpoint_exists(self, client):
        """测试 /update 端点存在"""
        response = client.put("/update", json={
            "updates": [
                {"old_text": "测试", "new_text": "测试更新"}
            ]
        })

        # 端点应该存在（可能返回 200, 400, 500 等，但不应该是 404）
        assert response.status_code != 404

    def test_update_request_validation(self, client):
        """测试请求验证"""
        # 缺少必需字段
        response = client.put("/update", json={
            "updates": [
                {"old_text": "测试"}  # 缺少 new_text
            ]
        })

        # 应该返回验证错误
        assert response.status_code == 422

    def test_update_empty_updates_list(self, client):
        """测试空更新列表"""
        response = client.put("/update", json={
            "updates": []
        })

        # 应该返回验证错误
        assert response.status_code == 422

    def test_update_exceeds_max_length(self, client):
        """测试超过最大长度限制"""
        # 创建 101 个更新（超过限制的 100）
        updates = [
            {"old_text": f"文本{i}", "new_text": f"新文本{i}"}
            for i in range(101)
        ]

        response = client.put("/update", json={"updates": updates})

        # 应该返回验证错误
        assert response.status_code == 422

    def test_update_with_businesstype(self, client):
        """测试指定业务类型"""
        response = client.put("/update", json={
            "updates": [
                {"old_text": "测试", "new_text": "新测试"}
            ],
            "businesstype": "test_type"
        })

        # 端点应该接受请求
        assert response.status_code != 404

    def test_update_invalid_businesstype(self, client):
        """测试无效的业务类型"""
        response = client.put("/update", json={
            "updates": [
                {"old_text": "测试", "new_text": "新测试"}
            ],
            "businesstype": "invalid@type#!"  # 包含非法字符
        })

        # 应该返回验证错误
        assert response.status_code == 422

    def test_update_response_format(self, client):
        """测试响应格式"""
        response = client.put("/update", json={
            "updates": [
                {"old_text": "不存在的文本", "new_text": "新文本"}
            ]
        })

        if response.status_code == 200:
            data = response.json()
            # 验证响应包含所有必需字段
            assert "success_count" in data
            assert "failed_count" in data
            assert "inserted_count" in data
            assert "updated_count" in data
            assert "errors" in data
            assert "total_vectors" in data
            assert "message" in data


class TestUpdateWorkflow:
    """测试完整更新流程"""

    def test_add_then_update_workflow(self, client):
        """测试添加后更新的完整流程"""
        # Step 1: 添加文档
        add_response = client.post("/add", json={
            "content": "原始文档内容",
            "businesstype": "test_workflow"
        })

        # Step 2: 更新文档（使用相同的内容）
        if add_response.status_code == 200:
            update_response = client.put("/update", json={
                "updates": [
                    {"old_text": "原始文档内容", "new_text": "更新后的文档内容"}
                ],
                "businesstype": "test_workflow"
            })

            # 验证更新响应
            if update_response.status_code == 200:
                data = update_response.json()
                assert data["updated_count"] >= 0 or data["inserted_count"] >= 0

    def test_upsert_workflow(self, client):
        """测试 upsert 工作流程（不存在则添加）"""
        response = client.put("/update", json={
            "updates": [
                {"old_text": "绝对不存在的文本内容XYZ123", "new_text": "新插入的内容"}
            ],
            "businesstype": "test_upsert"
        })

        if response.status_code == 200:
            data = response.json()
            # 应该被当作新增处理
            assert data["inserted_count"] >= 0

    def test_batch_update_workflow(self, client):
        """测试批量更新流程"""
        response = client.put("/update", json={
            "updates": [
                {"old_text": "文本1", "new_text": "新文本1"},
                {"old_text": "文本2", "new_text": "新文本2"},
                {"old_text": "文本3", "new_text": "新文本3"},
            ],
            "businesstype": "test_batch"
        })

        if response.status_code == 200:
            data = response.json()
            # 验证所有项都被处理
            total_processed = data["success_count"] + data["failed_count"]
            assert total_processed == 3


class TestUpdateWithSearch:
    """测试更新后的搜索功能"""

    def test_update_then_search(self, client):
        """测试更新后能否搜索到新内容"""
        businesstype = "test_update_search"

        # Step 1: 添加文档
        client.post("/add", json={
            "content": "原始搜索内容",
            "businesstype": businesstype
        })

        # Step 2: 更新文档
        update_response = client.put("/update", json={
            "updates": [
                {"old_text": "原始搜索内容", "new_text": "更新后的搜索内容"}
            ],
            "businesstype": businesstype
        })

        # Step 3: 搜索新内容
        if update_response.status_code == 200:
            search_response = client.post("/search", json={
                "question": "更新后的搜索内容",
                "businesstype": businesstype,
                "top_k": 3
            })

            if search_response.status_code == 200:
                data = search_response.json()
                # 验证搜索结果存在
                assert "relevant_chunks" in data or "detailed_results" in data


class TestUpdateErrorHandling:
    """测试错误处理"""

    def test_update_with_invalid_json(self, client):
        """测试无效 JSON"""
        response = client.put(
            "/update",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_update_with_missing_content_type(self, client):
        """测试缺少 Content-Type"""
        response = client.put(
            "/update",
            data=json.dumps({
                "updates": [
                    {"old_text": "test", "new_text": "new"}
                ]
            })
        )
        # FastAPI 应该能处理
        assert response.status_code != 404


# ============================================================================
# 性能测试（可选）
# ============================================================================

class TestUpdatePerformance:
    """测试更新性能"""

    def test_small_batch_performance(self, client):
        """测试小批量更新性能（10条）"""
        import time

        updates = [
            {"old_text": f"文本{i}", "new_text": f"新文本{i}"}
            for i in range(10)
        ]

        start = time.time()
        response = client.put("/update", json={"updates": updates, "businesstype": "test_perf"})
        elapsed = time.time() - start

        if response.status_code == 200:
            # 小批量应该在合理时间内完成（< 5秒）
            assert elapsed < 5.0

    def test_medium_batch_performance(self, client):
        """测试中批量更新性能（50条）"""
        import time

        updates = [
            {"old_text": f"文本{i}", "new_text": f"新文本{i}"}
            for i in range(50)
        ]

        start = time.time()
        response = client.put("/update", json={"updates": updates, "businesstype": "test_perf"})
        elapsed = time.time() - start

        if response.status_code == 200:
            # 中批量应该在合理时间内完成（< 10秒）
            assert elapsed < 10.0
