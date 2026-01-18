#!/bin/bash

# 测试 MCP 工具列表
echo "=== 1. 测试 tools/list ==="
curl -s -X POST 'http://192.168.136.224:8003/v1/mcp' \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 1
  }' | jq '.'

echo -e "\n=== 2. 测试 stats 工具 ==="
curl -s -X POST 'http://192.168.136.224:8003/v1/mcp' \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "stats",
      "arguments": {}
    },
    "id": 2
  }' | jq '.'

echo -e "\n=== 3. 测试 add 工具 ==="
curl -s -X POST 'http://192.168.136.224:8003/v1/mcp' \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "add",
      "arguments": {
        "content": "这是一个测试文档，用于验证 MCP 工具是否正常工作。"
      }
    },
    "id": 3
  }' | jq '.'

echo -e "\n=== 4. 测试 search 工具 ==="
sleep 2  # 等待索引完成
curl -s -X POST 'http://192.168.136.224:8003/v1/mcp' \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "search",
      "arguments": {
        "query": "测试文档",
        "top_k": 3
      }
    },
    "id": 4
  }' | jq '.'

echo -e "\n=== 5. 测试 delete 工具 ==="
curl -s -X POST 'http://192.168.136.224:8003/v1/mcp' \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "delete",
      "arguments": {
        "content": "这是一个测试文档，用于验证 MCP 工具是否正常工作。"
      }
    },
    "id": 5
  }' | jq '.'

echo -e "\n=== 测试完成 ==="
