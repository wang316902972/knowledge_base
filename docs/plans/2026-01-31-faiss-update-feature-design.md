# FAISS 更新功能设计文档

**日期**: 2026-01-31
**作者**: Claude AI
**状态**: 已批准

## 概述

为 FAISS 向量数据库添加文档更新功能，支持 upsert 语义（更新或插入）。

## 功能需求

### 核心功能
- 通过原始文本内容定位数据
- 批量更新文档
- upsert 语义：未找到时静默添加
- 删除旧向量，只保留新向量
- 部分失败容忍：成功的更新，失败的记录日志
- 返回详细统计信息

## 架构设计

### 核心方法：`update_texts()`

```python
def update_texts(self, updates: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    批量更新文本（upsert 语义）

    Args:
        updates: 更新列表，每项包含 {
            "old_text": "原始文本内容",
            "new_text": "新文本内容"
        }

    Returns:
        {
            "success_count": int,      # 成功更新数量
            "failed_count": int,       # 失败数量
            "inserted_count": int,     # 新增数量（old_text 不存在）
            "updated_count": int,      # 更新数量
            "errors": List[Dict]       # 失败详情 [{old_text, error}]
        }
    """
```

### 实现流程

1. **查找阶段**：遍历 `chunk_to_id` 映射，分类为"更新"或"新增"
2. **批量删除**：使用 `index.remove_ids()` 删除所有待更新向量
3. **批量嵌入**：一次性生成所有新文本的嵌入向量
4. **批量添加**：添加新向量到索引
5. **更新元数据**：更新 `id_to_chunk` 和 `chunk_to_id`

### 并发安全

整个操作在 `self.lock`（重入锁）保护下执行。

## API 设计

### REST 端点

```
PUT /update
```

**请求体**：
```json
{
  "updates": [
    {"old_text": "原文1", "new_text": "新文1"},
    {"old_text": "原文2", "new_text": "新文2"}
  ],
  "businesstype": "default"  // 可选
}
```

**响应**：
```json
{
  "success_count": 5,
  "failed_count": 1,
  "inserted_count": 2,
  "updated_count": 3,
  "errors": [
    {
      "index": 1,
      "old_text": "原文预览...",
      "error": "错误详情"
    }
  ]
}
```

### Pydantic 模型

```python
class TextUpdate(BaseModel):
    old_text: str = Field(..., min_length=1, max_length=50000)
    new_text: str = Field(..., min_length=1, max_length=50000)

class UpdateRequest(BaseModel):
    updates: List[TextUpdate] = Field(..., max_length=100)
    businesstype: str = Field(default="default")
```

## 错误处理

### 边缘情况

| 场景 | 处理方式 |
|------|---------|
| `old_text` 不存在 | 当作新增插入 |
| `old_text == new_text` | 跳过，不做任何操作 |
| `old_text` 重复出现 | 使用第一个匹配的ID |
| 嵌入向量生成失败 | 记录错误，跳过该项 |
| 索引删除失败 | 记录错误，继续添加新向量 |
| 元数据更新失败 | 回滚已删除的向量，抛出异常 |

### 部分失败容忍

- 成功的更新继续执行
- 失败的项目记录到 `errors` 列表
- 返回详细的统计信息

## 性能优化

### 批量操作

- 批量生成嵌入向量
- 一次性删除所有旧向量
- 一次性添加所有新向量

### 性能预期

| 操作数 | 预期时间 |
|--------|---------|
| 10条   | ~150ms  |
| 50条   | ~600ms  |
| 100条  | ~1000ms |

## 实现计划

### Phase 1：核心功能（P0）

1. 实现 `FaissVectorDB.update_texts()` 方法
2. 添加 Pydantic 模型（`TextUpdate`, `UpdateRequest`）
3. 实现 API 端点 `PUT /update`

### Phase 2：错误处理增强（P1）

4. 完善错误捕获和日志
5. 添加详细的返回值统计
6. 边缘情况处理

### Phase 3：测试（P1）

7. 单元测试
8. 集成测试
9. 性能基准测试

### Phase 4：性能优化（P2）

10. 大批量分块处理
11. 性能监控和日志

## 测试策略

### 单元测试

- `test_update_single_existing()`: 更新已存在的文本
- `test_upsert_non_existing()`: upsert语义测试
- `test_batch_update_partial_failure()`: 批量更新部分失败

### 集成测试

- `test_update_api_endpoint()`: API端点测试

## 验收标准

- [ ] 支持通过原始文本内容更新数据
- [ ] 支持批量更新（最多100条）
- [ ] 未找到时自动插入（upsert语义）
- [ ] 部分失败容忍，返回详细统计
- [ ] 线程安全
- [ ] 单元测试覆盖率 > 80%
- [ ] 集成测试通过
