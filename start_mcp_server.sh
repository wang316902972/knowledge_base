#!/bin/bash
# FAISS Vector Database MCP Server 启动脚本

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}FAISS Vector Database MCP Server${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}错误: 未找到 python3${NC}"
    exit 1
fi

# 检查依赖
echo -e "${GREEN}检查依赖...${NC}"
if ! python3 -c "import mcp" 2>/dev/null; then
    echo -e "${YELLOW}警告: mcp 包未安装，正在安装依赖...${NC}"
    pip install -r requirements.txt
fi

echo ""
echo "请选择启动模式："
echo ""
echo "  1) HTTP 服务器模式 (推荐)"
echo "     - 支持 HTTP REST API 访问"
echo "     - 地址: http://localhost:8001"
echo "     - 适合任何编程语言客户端"
echo ""
echo "  2) STDIO 模式"
echo "     - 支持 Claude Desktop 等客户端"
echo "     - 通过标准输入/输出通信"
echo ""
echo "  3) 测试 HTTP 客户端"
echo "     - 运行 HTTP 客户端测试脚本"
echo ""
echo "  4) 测试 STDIO 客户端"
echo "     - 运行 STDIO 客户端测试脚本"
echo ""
echo -n "请输入选项 (1-4): "
read choice

case $choice in
    1)
        echo -e "${GREEN}启动 HTTP 服务器模式...${NC}"
        echo -e "${BLUE}服务器地址: http://localhost:8001${NC}"
        echo -e "${BLUE}API 文档: http://localhost:8001/docs${NC}"
        echo ""
        python3 mcp_http_server.py
        ;;
    2)
        echo -e "${GREEN}启动 STDIO 模式...${NC}"
        echo -e "${YELLOW}注意: 此模式用于与支持 MCP stdio 的客户端通信${NC}"
        echo ""
        python3 mcp_server.py
        ;;
    3)
        echo -e "${GREEN}运行 HTTP 客户端测试...${NC}"
        echo -e "${YELLOW}确保 HTTP 服务器正在运行: python3 mcp_http_server.py${NC}"
        echo ""
        sleep 2
        python3 test_mcp_http_client.py
        ;;
    4)
        echo -e "${GREEN}运行 STDIO 客户端测试...${NC}"
        echo ""
        python3 test_mcp_client.py
        ;;
    *)
        echo -e "${YELLOW}无效选项${NC}"
        exit 1
        ;;
esac
