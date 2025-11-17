#!/bin/bash

# 启动脚本 - 解决环境变量传递问题

set -e

echo "启动FAISS向量数据库服务..."
echo "配置信息："
echo "  BUSINESS_ID: ${BUSINESS_ID:-default}"
echo "  API_PORT: ${API_PORT:-8001}"
echo "  ENVIRONMENT: ${ENVIRONMENT:-development}"
echo ""

# 确保端口是有效的整数
PORT=${API_PORT:-8001}
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "错误: 无效的端口号: $PORT"
    exit 1
fi

echo "正在启动服务在端口 $PORT..."

# 启动uvicorn服务
exec uvicorn faiss_server_optimized:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --reload