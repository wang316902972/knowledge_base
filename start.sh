#!/bin/bash

# FAISS向量数据库快速启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 FAISS向量数据库启动脚本${NC}"
echo "=================================="

# 检查Python是否安装
check_python() {
    if command -v python3.11 &> /dev/null; then
        echo -e "${GREEN}✅ Python 3.11 已安装${NC}"
        python3.11 --version
    elif command -v python3 &> /dev/null; then
        echo -e "${GREEN}✅ Python3 已安装${NC}"
        python3 --version
    else
        echo -e "${RED}❌ Python3 未找到，请先安装Python3${NC}"
        exit 1
    fi
}

# 检查pip是否安装
check_pip() {
    if command -v pip3.11 &> /dev/null; then
        echo -e "${GREEN}✅ pip3.11 已安装${NC}"
    elif command -v pip3 &> /dev/null; then
        echo -e "${GREEN}✅ pip3 已安装${NC}"
    else
        echo -e "${RED}❌ pip3 未找到，请先安装pip3${NC}"
        exit 1
    fi
}

# 安装依赖
install_dependencies() {
    echo -e "${YELLOW}📦 安装Python依赖...${NC}"
    if [ -f "requirements.txt" ]; then
        if command -v pip3.11 &> /dev/null; then
            pip3.11 install -r requirements.txt
        else
            pip3 install -r requirements.txt
        fi
        echo -e "${GREEN}✅ 依赖安装完成${NC}"
    else
        echo -e "${RED}❌ requirements.txt 文件未找到${NC}"
        exit 1
    fi
}

# 验证配置
validate_config() {
    echo -e "${YELLOW}🔧 验证配置...${NC}"
    if command -v python3.11 &> /dev/null; then
        python3.11 config.py &> /dev/null
    else
        python3 config.py &> /dev/null
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 配置验证通过${NC}"
    else
        echo -e "${RED}❌ 配置验证失败${NC}"
        if command -v python3.11 &> /dev/null; then
            python3.11 config.py
        else
            python3 config.py
        fi
        exit 1
    fi
}

# 启动服务
start_service() {
    echo -e "${YELLOW}🎯 启动FAISS向量数据库服务...${NC}"

    # 设置默认环境变量
    export ENVIRONMENT=${ENVIRONMENT:-development}
    export LOG_LEVEL=${LOG_LEVEL:-INFO}
    export AUTO_SAVE=${AUTO_SAVE:-false}
    export BUSINESS_ID=${BUSINESS_ID:-default}

    # 确定端口
    local port=${API_PORT:-8001}
    if [ "$BUSINESS_ID" != "default" ]; then
        case $BUSINESS_ID in
            "ecommerce") port=8002 ;;
            "medical") port=8003 ;;
            "finance") port=8004 ;;
            "document") port=8005 ;;
        esac
    fi

    echo "业务ID: $BUSINESS_ID"
    echo "服务将在 http://localhost:$port 上启动"
    echo "API文档: http://localhost:$port/docs"
    echo "健康检查: http://localhost:$port/health"
    echo ""
    echo -e "${BLUE}按 Ctrl+C 停止服务${NC}"
    echo ""

    # 启动服务
    if [ "$1" = "dev" ]; then
        echo -e "${YELLOW}🔧 开发模式启动${NC}"
        python3.11 faiss_server_optimized.py
    else
        echo -e "${YELLOW}🚀 生产模式启动${NC}"
        python3.11 -m uvicorn faiss_server_optimized:app --host 0.0.0.0 --port $port --reload
    fi
}

# Docker启动
start_docker() {
    echo -e "${YELLOW}🐳 使用Docker启动服务...${NC}"

    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker 未安装${NC}"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose 未安装${NC}"
        exit 1
    fi

    # 创建数据目录
    mkdir -p data logs

    # 根据参数选择compose文件
    local compose_file="docker-compose.yml"
    if [ "$1" = "multi" ]; then
        compose_file="docker-compose-multi-business.yml"
        echo "启动多业务实例..."
    elif [ "$1" = "dev" ]; then
        compose_file="docker-compose-dev.yml"
        echo "启动开发环境实例..."
    fi

    echo "使用 compose 文件: $compose_file"
    echo "构建并启动容器..."
    docker-compose -f $compose_file up --build
}

# 运行测试
run_test() {
    echo -e "${YELLOW}🧪 运行API测试...${NC}"
    if [ "$1" = "simple" ]; then
        if command -v python3.11 &> /dev/null; then
            python3.11 test_api.py simple
        else
            python3 test_api.py simple
        fi
    else
        if command -v python3.11 &> /dev/null; then
            python3.11 test_api.py
        else
            python3 test_api.py
        fi
    fi
}

# 显示帮助信息
show_help() {
    echo "FAISS向量数据库启动脚本 - 多业务版本"
    echo ""
    echo "用法:"
    echo "  ./start.sh [选项]"
    echo ""
    echo "选项:"
    echo "  help          显示此帮助信息"
    echo "  install       仅安装依赖"
    echo "  check         仅检查环境和配置"
    echo "  dev           开发模式启动（使用python直接运行）"
    echo "  prod          生产模式启动（使用uvicorn）"
    echo "  docker        使用Docker启动"
    echo "  docker-multi  使用Docker启动多业务实例"
    echo "  docker-dev    使用Docker启动开发环境实例"
    echo "  test          运行完整测试"
    echo "  test-simple   运行简单测试"
    echo ""
    echo "环境变量:"
    echo "  BUSINESS_ID   业务标识符 (default/ecommerce/medical/finance/document)"
    echo "  ENVIRONMENT   运行环境 (development/production/test)"
    echo "  LOG_LEVEL     日志级别 (DEBUG/INFO/WARNING/ERROR)"
    echo "  AUTO_SAVE     是否自动保存 (true/false)"
    echo "  FAISS_INDEX_TYPE  索引类型 (FlatIP/FlatL2/IVFFlat)"
    echo "  API_PORT      API端口号 (自动根据BUSINESS_ID分配)"
    echo "  FAISS_DATA_DIR    数据目录路径"
    echo ""
    echo "业务端口分配:"
    echo "  default     -> 8001"
    echo "  ecommerce   -> 8002"
    echo "  medical     -> 8003"
    echo "  finance     -> 8004"
    echo "  document    -> 8005"
    echo ""
    echo "示例:"
    echo "  ./start.sh dev                           # 开发模式启动默认业务"
    echo "  BUSINESS_ID=ecommerce ./start.sh dev     # 启动电商业务"
    echo "  ./start.sh docker-multi                  # Docker启动所有业务实例"
    echo "  BUSINESS_ID=medical ./start.sh prod      # 生产模式启动医疗业务"
    echo "  ./start.sh docker-dev                    # Docker启动开发环境实例"
}

# 清理函数
cleanup() {
    echo -e "\n${YELLOW}🧹 正在清理...${NC}"
    # 这里可以添加清理逻辑
    exit 0
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# 主程序
main() {
    case "${1:-prod}" in
        "help"|"-h"|"--help")
            show_help
            ;;
        "install")
            check_python
            check_pip
            install_dependencies
            ;;
        "check")
            check_python
            check_pip
            validate_config
            echo -e "${GREEN}✅ 环境检查完成${NC}"
            ;;
        "dev")
            check_python
            check_pip
            install_dependencies
            validate_config
            start_service dev
            ;;
        "prod")
            check_python
            check_pip
            install_dependencies
            validate_config
            start_service prod
            ;;
        "docker")
            start_docker
            ;;
        "docker-multi")
            start_docker multi
            ;;
        "docker-dev")
            start_docker dev
            ;;
        "test")
            run_test
            ;;
        "test-simple")
            run_test simple
            ;;
        *)
            echo -e "${RED}❌ 未知选项: $1${NC}"
            show_help
            exit 1
            ;;
    esac
}

# 执行主程序
main "$@"
