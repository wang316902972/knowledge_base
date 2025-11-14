# FAISS 向量数据库多业务版本

这个版本的FAISS向量数据库支持为不同的业务生成独立的索引文件，解决Docker环境中的多业务隔离问题。

## 功能特性

- 🏢 **多业务隔离**: 每个业务拥有独立的索引文件和配置
- 🐳 **Docker支持**: 支持Docker容器化部署，每个业务实例独立运行
- 🔧 **动态配置**: 基于环境变量自动配置业务参数
- 📊 **独立端口**: 每个业务分配独立端口号
- 💾 **数据隔离**: 不同业务的数据完全分离存储

## 业务配置

### 预定义业务

| 业务ID | 端口 | 索引类型 | 用途 |
|--------|------|----------|------|
| default | 8001 | FlatIP | 默认业务 |
| ecommerce | 8002 | IVFFlat | 电商商品搜索 |
| medical | 8003 | FlatL2 | 医疗文档检索 |
| finance | 8004 | FlatIP | 金融知识库 |
| document | 8005 | IVFFlat | 通用文档检索 |

### 环境变量

```bash
# 业务标识
BUSINESS_ID=ecommerce                    # 必须设置
SERVICE_NAME=faiss-vector-db-ecommerce   # 服务名称

# 索引配置
FAISS_INDEX_TYPE=IVFFlat                 # 索引类型
FAISS_NLIST=100                          # IVF索引参数
FAISS_DATA_DIR=/app/data                 # 数据目录

# 文本处理
MAX_CHUNK_SIZE=1000                      # 最大分块大小
DEFAULT_CHUNK_SIZE=300                   # 默认分块大小
DEFAULT_CHUNK_OVERLAP=50                # 分块重叠

# 其他配置
ENVIRONMENT=production                   # 运行环境
LOG_LEVEL=INFO                          # 日志级别
AUTO_SAVE=true                          # 自动保存
BATCH_SIZE=32                           # 批处理大小
```

## 文件结构

每个业务生成独立的索引文件：

```
data/
├── default_knowledge_base.index      # 默认业务索引
├── default_knowledge_base.json       # 默认业务元数据
├── ecommerce_knowledge_base.index    # 电商业务索引
├── ecommerce_knowledge_base.json     # 电商业务元数据
├── medical_knowledge_base.index      # 医疗业务索引
├── medical_knowledge_base.json       # 医疗业务元数据
└── ...
```

## 使用方法

### 1. 单业务启动

```bash
# 启动默认业务
./start.sh dev

# 启动电商业务
BUSINESS_ID=ecommerce ./start.sh dev

# 启动医疗业务
BUSINESS_ID=medical ./start.sh prod

# 自定义业务
BUSINESS_ID=custom_business ./start.sh prod
```

### 2. Docker部署

#### 单业务部署
```bash
# 默认业务
./start.sh docker

# 自定义业务
docker-compose -f docker-compose.yml up -d
```

#### 多业务部署
```bash
# 启动所有预定义业务
./start.sh docker-multi

# 开发环境多业务
./start.sh docker-dev
```

#### 环境变量配置
```bash
# 为每个业务设置不同的环境变量
export BUSINESS_ID=ecommerce
export FAISS_INDEX_TYPE=IVFFlat
export MAX_CHUNK_SIZE=800

docker-compose -f docker-compose-ecommerce.yml up -d
```

### 3. 自定义业务配置

创建新的docker-compose文件：

```yaml
version: '3.8'

services:
  faiss-custom:
    build: .
    ports:
      - "8010:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - BUSINESS_ID=custom_business
      - SERVICE_NAME=faiss-vector-db-custom
      - FAISS_INDEX_TYPE=FlatIP
      - AUTO_SAVE=true
      - LOG_LEVEL=INFO
      - BATCH_SIZE=64
      - MAX_CHUNK_SIZE=1500
      - DEFAULT_CHUNK_SIZE=400
    restart: unless-stopped
```

## API使用

每个业务实例都有独立的API端点：

```bash
# 电商业务API
curl -X POST http://localhost:8002/add_texts \
  -H "Content-Type: application/json" \
  -d '{"texts": ["商品A描述", "商品B描述"]}'

# 医疗业务API
curl -X POST http://localhost:8003/add_texts \
  -H "Content-Type: application/json" \
  -d '{"texts": ["疾病A症状", "疾病B症状"]}'
```

## 业务定制

### 新增业务类型

1. 修改配置类 `config.py`：
```python
class CustomBusinessConfig(Config):
    BUSINESS_ID = "custom"
    INDEX_TYPE = "FlatIP"
    MAX_CHUNK_SIZE = 1200
    DEFAULT_CHUNK_SIZE = 300
```

2. 添加端口映射到 `start.sh`：
```bash
case $BUSINESS_ID in
    "custom") port=8010 ;;
esac
```

3. 创建对应的Docker Compose配置

### 自定义索引参数

不同业务可以使用不同的FAISS索引类型：

- **FlatIP**: 适用于小规模数据，精确搜索
- **FlatL2**: 适用于欧几里得距离度量
- **IVFFlat**: 适用于中大规模数据，快速搜索
- **HNSW**: 适用于大规模数据，近似最近邻搜索

## 监控和日志

每个业务实例都有独立的日志：

```bash
# 查看不同业务的日志
tail -f logs/faiss-vector-db-ecommerce.log
tail -f logs/faiss-vector-db-medical.log
```

健康检查端点：

```bash
# 检查各业务实例状态
curl http://localhost:8002/health  # 电商业务
curl http://localhost:8003/health  # 医疗业务
curl http://localhost:8004/health  # 金融业务
```

## 故障排除

### 常见问题

1. **端口冲突**: 确保 `BUSINESS_ID` 和端口映射正确
2. **权限问题**: 检查数据目录权限，确保容器有写入权限
3. **内存不足**: 调整 `BATCH_SIZE` 和索引参数
4. **模型加载失败**: 确认模型文件正确，网络连接正常

### 调试命令

```bash
# 检查业务配置
BUSINESS_ID=ecommerce python3.11 -c "from config import Config; c = Config(); print(c.INDEX_FILE)"

# 测试向量数据库初始化
BUSINESS_ID=medical python3.11 -c "from faiss_server_optimized import FaissVectorDB, Config; FaissVectorDB(Config())"

# 查看Docker容器状态
docker-compose ps
docker logs faiss-vector-db-ecommerce
```

## 生产环境建议

1. **资源配置**: 为不同业务配置不同的资源限制
2. **数据备份**: 定期备份各业务的索引文件
3. **监控告警**: 监控各业务实例的健康状态
4. **负载均衡**: 使用Nginx或负载均衡器分发请求
5. **版本管理**: 使用Git管理不同业务的配置

## 性能优化

### 业务特定优化

- **电商业务**: 使用IVFFlat索引，优化商品描述搜索
- **医疗业务**: 使用FlatL2索引，确保医疗文档搜索的准确性
- **金融业务**: 使用FlatIP索引，优化金融知识库的检索速度

### 资源分配

```yaml
# 在docker-compose中为不同业务分配不同资源
services:
  faiss-ecommerce:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  faiss-medical:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

## 示例

完整的电商业务部署示例：

```bash
# 1. 创建电商业务配置
export BUSINESS_ID=ecommerce
export FAISS_INDEX_TYPE=IVFFlat
export FAISS_NLIST=100
export MAX_CHUNK_SIZE=800

# 2. 启动服务
./start.sh docker-multi

# 3. 添加商品数据
curl -X POST http://localhost:8002/add_texts \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "iPhone 14 Pro Max 256GB 深空黑色",
      "Apple MacBook Pro 14英寸 M2 Pro芯片",
      "Sony WH-1000XM4 无线降噪耳机"
    ]
  }'

# 4. 搜索商品
curl -X POST http://localhost:8002/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "苹果手机",
    "top_k": 5
  }'
```

这样就实现了为不同业务生成不同索引文件的需求，每个业务都有独立的配置、存储和API端点。