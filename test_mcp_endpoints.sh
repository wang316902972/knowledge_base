#!/bin/bash
# MCP 端点测试脚本

echo "==================================="
echo "MCP 端点测试"
echo "==================================="
echo ""

BASE_URL="http://localhost:8003"
# 如果用户使用的是 8004 端口，可以取消注释下面的行
# BASE_URL="http://localhost:8004"

echo "测试 1: 健康检查"
echo "GET /health"
curl -s "${BASE_URL}/health" | python3 -m json.tool 2>/dev/null || echo "❌ 健康检查失败"
echo ""
echo ""

echo "测试 2: MCP 初始化 (标准路径 /mcp)"
echo "POST /mcp"
curl -s -X POST "${BASE_URL}/mcp" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "initialize",
    "id": 1
  }' | python3 -m json.tool 2>/dev/null || echo "❌ /mcp 端点失败"
echo ""
echo ""

echo "测试 3: MCP 初始化 (版本化路径 /v1/mcp)"
echo "POST /v1/mcp"
curl -s -X POST "${BASE_URL}/v1/mcp" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "initialize",
    "id": 2
  }' | python3 -m json.tool 2>/dev/null || echo "❌ /v1/mcp 端点失败"
echo ""
echo ""

echo "测试 4: 列出可用工具 (版本化路径)"
echo "POST /v1/mcp - tools/list"
curl -s -X POST "${BASE_URL}/v1/mcp" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 3
  }' | python3 -m json.tool 2>/dev/null || echo "❌ tools/list 失败"
echo ""
echo ""

echo "测试 5: 调用搜索工具 (指定业务类型 sd)"
echo "POST /v1/mcp - tools/call (search_knowledge with businesstype=sd)"
curl -s -X POST "${BASE_URL}/v1/mcp" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "search_knowledge",
      "arguments": {
        "query": "测试",
        "businesstype": "sd",
        "top_k": 3
      }
    },
    "id": 4
  }' | python3 -m json.tool 2>/dev/null || echo "❌ search_knowledge 失败"
echo ""
echo ""

echo "测试 6: 添加文档到 sd 业务类型"
echo "POST /v1/mcp - tools/call (add_document with businesstype=sd)"
curl -s -X POST "${BASE_URL}/v1/mcp" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "add_document",
      "arguments": {
        "content": "SD模型相关文档",
        "businesstype": "sd"
      }
    },
    "id": 5
  }' | python3 -m json.tool 2>/dev/null || echo "❌ add_document 失败"
echo ""
echo ""

echo "测试 7: 添加文档到 warning 业务类型"
echo "POST /v1/mcp - tools/call (add_document with businesstype=warning)"
curl -s -X POST "${BASE_URL}/v1/mcp" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "add_document",
      "arguments": {
        "content": "Warning系统告警相关文档",
        "businesstype": "warning"
      }
    },
    "id": 6
  }' | python3 -m json.tool 2>/dev/null || echo "❌ add_document 失败"
echo ""
echo ""

echo "测试 8: 在 sd 中搜索（应该找到）"
echo "POST /v1/mcp - tools/call (search in sd)"
curl -s -X POST "${BASE_URL}/v1/mcp" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "search_knowledge",
      "arguments": {
        "query": "SD模型",
        "businesstype": "sd",
        "top_k": 3
      }
    },
    "id": 7
  }' | python3 -m json.tool 2>/dev/null || echo "❌ search 失败"
echo ""
echo ""

echo "测试 9: 在 warning 中搜索 SD（应该为空）"
echo "POST /v1/mcp - tools/call (search in warning for SD)"
curl -s -X POST "${BASE_URL}/v1/mcp" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "search_knowledge",
      "arguments": {
        "query": "SD",
        "businesstype": "warning",
        "top_k": 3
      }
    },
    "id": 8
  }' | python3 -m json.tool 2>/dev/null || echo "❌ search 失败"
echo ""
echo ""

echo "==================================="
echo "测试完成"
echo "==================================="
