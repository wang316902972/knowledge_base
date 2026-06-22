from faiss_server_optimized import (
    _validate_sensitive_access,
    _lexical_overlap_score,
    _looks_like_high_entropy_secret,
)


def test_detects_obsidian_account_config_as_unsearchable():
    text = (
        "Source: Obsidian\n"
        "Obsidian Path: 账号配置/上游DB账号信息.md\n"
        "Title: 上游DB账号信息\n\n"
        "普通说明文本"
    )

    assert _looks_like_high_entropy_secret(text) is True


def test_detects_long_encrypted_blob_as_unsearchable():
    encrypted = (
        "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789+/="
        "QmFzZTY0TG9va2luZ0Jsb2JXaXRoTG90c09mQ2hhcmFjdGVycw=="
        "Z2F0ZXdheU1jcFRleHQyU3FsU2VjcmV0UGF5bG9hZA=="
    )
    text = f"## 加密内容\n\n{encrypted}\n\n## 解密方法\n请使用 crypto-encrypt skill 解密"

    assert _looks_like_high_entropy_secret(text) is True


def test_plain_technical_note_is_searchable():
    text = (
        "Source: Obsidian\n"
        "Obsidian Path: 方案设计/Vanna Text2SQL MCP路由替换方案设计.md\n"
        "Title: Vanna Text2SQL MCP路由替换方案设计\n\n"
        "Text2SQL 通过 MCP 查询 StarRocks Doris 元数据和血缘。"
    )

    assert _looks_like_high_entropy_secret(text) is False


def test_lexical_overlap_scores_query_tokens():
    score = _lexical_overlap_score(
        "Text2SQL MCP 18081",
        "vanna-text2sql 服务使用 MCP，端口是 18081。",
    )

    assert score > 0.5


def test_account_config_requires_access_key(monkeypatch):
    monkeypatch.setenv("ACCOUNT_CONFIG_ACCESS_KEY", "expected-key")

    access_error = _validate_sensitive_access("account_config", None)

    assert access_error is not None
    assert access_error["access_granted"] is False
    assert access_error["total_found"] == 0


def test_account_config_rejects_wrong_access_key(monkeypatch):
    monkeypatch.setenv("ACCOUNT_CONFIG_ACCESS_KEY", "expected-key")

    access_error = _validate_sensitive_access("account_config", "wrong-key")

    assert access_error is not None
    assert access_error["access_granted"] is False
    assert access_error["relevant_chunks"] == []


def test_account_config_accepts_correct_access_key(monkeypatch):
    monkeypatch.setenv("ACCOUNT_CONFIG_ACCESS_KEY", "expected-key")

    assert _validate_sensitive_access("account_config", "expected-key") is None


def test_default_businesstype_does_not_require_access_key(monkeypatch):
    monkeypatch.setenv("ACCOUNT_CONFIG_ACCESS_KEY", "expected-key")

    assert _validate_sensitive_access("default", None) is None
