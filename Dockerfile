# 使用官方Python镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

ENV http_proxy=http://192.168.244.188:7897
ENV https_proxy=http://192.168.244.188:7897

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY faiss_server_optimized.py .
COPY config.py .

# 创建数据目录和日志目录
RUN mkdir -p /app/data /app/logs

# 设置文件权限
RUN chmod +x faiss_server_optimized.py

# 设置默认环境变量
ENV BUSINESS_ID=default
ENV FAISS_DATA_DIR=/app/data

# 暴露端口
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health', timeout=5)" || exit 1

# 启动命令
CMD ["uvicorn", "faiss_server_optimized:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
