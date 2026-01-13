"""
FAISS向量数据库配置文件
"""
import os
from typing import Optional

class Config:
    """向量数据库配置类"""

    # 业务标识配置
    DEFAULT_BUSINESSTYPE: str = os.getenv("BUINESSTYPE", "default")  # 业务类型标识符
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "faiss-vector-db")  # 服务名称

    # 文件路径配置 - 支持基于业务类型的动态路径
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.getenv("FAISS_DATA_DIR", os.path.join(SCRIPT_DIR, "data"))  # 数据目录

    def get_index_file(self, businesstype: str = None) -> str:
        """动态生成索引文件路径"""
        if businesstype is None:
            businesstype = self.DEFAULT_BUSINESSTYPE
        businesstype = self._validate_businesstype(businesstype)
        filename = f"{businesstype}_knowledge_base.index"
        businesstype_dir = os.path.join(self.DATA_DIR, businesstype)
        return os.getenv("FAISS_INDEX_FILE", os.path.join(businesstype_dir, filename))

    def get_metadata_file(self, businesstype: str = None) -> str:
        """动态生成元数据文件路径"""
        if businesstype is None:
            businesstype = self.DEFAULT_BUSINESSTYPE
        businesstype = self._validate_businesstype(businesstype)
        filename = f"{businesstype}_knowledge_base.json"
        businesstype_dir = os.path.join(self.DATA_DIR, businesstype)
        return os.getenv("FAISS_METADATA_FILE", os.path.join(businesstype_dir, filename))

    @staticmethod
    def _validate_businesstype(businesstype: str) -> str:
        """验证和清理业务类型名称"""
        import re
        # 只允许字母、数字、下划线和连字符；最长50个字符
        if not re.match(r'^[a-zA-Z0-9_-]{1,50}$', businesstype):
            raise ValueError(f"Invalid businesstype: '{businesstype}'. Must be 1-50 alphanumeric characters, underscores, or hyphens only")
        return businesstype.lower()
    
    # 模型配置
    MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))
    
    # FAISS索引配置
    INDEX_TYPE: str = os.getenv("FAISS_INDEX_TYPE", "FlatIP")  # FlatIP, FlatL2, IVFFlat, HNSW
    NLIST: int = int(os.getenv("FAISS_NLIST", "100"))  # IVF索引的聚类数量
    NPROBE: int = int(os.getenv("FAISS_NPROBE", "10"))  # IVF索引搜索时的探测数量
    M: int = int(os.getenv("FAISS_HNSW_M", "16"))  # HNSW索引的连接数
    EF_CONSTRUCTION: int = int(os.getenv("FAISS_EF_CONSTRUCTION", "200"))  # HNSW构建时的搜索深度
    EF_SEARCH: int = int(os.getenv("FAISS_EF_SEARCH", "50"))  # HNSW搜索时的深度
    
    # 文本处理配置
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "2000"))
    MIN_CHUNK_SIZE: int = int(os.getenv("MIN_CHUNK_SIZE", "50"))
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50"))
    
    # 批处理配置
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    MAX_BATCH_TEXTS: int = int(os.getenv("MAX_BATCH_TEXTS", "100"))
    
    # 搜索配置
    MAX_TOP_K: int = int(os.getenv("MAX_TOP_K", "50"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))

    # 搜索优化配置
    ENABLE_SEARCH_OPTIMIZATION: bool = os.getenv("ENABLE_SEARCH_OPTIMIZATION", "true").lower() == "true"
    SEMANTIC_CHUNKING: bool = os.getenv("SEMANTIC_CHUNKING", "true").lower() == "true"
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))
    DIVERSITY_WEIGHT: float = float(os.getenv("DIVERSITY_WEIGHT", "0.2"))
    SEARCH_CANDIDATE_MULTIPLIER: int = int(os.getenv("SEARCH_CANDIDATE_MULTIPLIER", "3"))  # 搜索候选数量倍数
    
    # 性能配置
    AUTO_SAVE: bool = os.getenv("AUTO_SAVE", "false").lower() == "true"
    SAVE_INTERVAL: int = int(os.getenv("SAVE_INTERVAL", "100"))  # 每N次操作自动保存
    
    # 安全配置
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", "50000"))
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
    
    # API配置
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8001"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # GPU配置（如果需要）
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"
    GPU_ID: int = int(os.getenv("GPU_ID", "0"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production").lower()
    
    @classmethod
    def validate(cls):
        """验证配置参数"""
        errors = []

        if cls.MIN_CHUNK_SIZE >= cls.MAX_CHUNK_SIZE:
            errors.append("MIN_CHUNK_SIZE must be less than MAX_CHUNK_SIZE")

        if cls.DEFAULT_CHUNK_OVERLAP >= cls.DEFAULT_CHUNK_SIZE:
            errors.append("DEFAULT_CHUNK_OVERLAP must be less than DEFAULT_CHUNK_SIZE")

        if cls.INDEX_TYPE not in ["FlatIP", "FlatL2", "IVFFlat", "HNSW"]:
            errors.append(f"Unsupported INDEX_TYPE: {cls.INDEX_TYPE}")

        if cls.INDEX_TYPE == "IVFFlat" and (cls.NLIST <= 0 or cls.NPROBE <= 0):
            errors.append("IVFFlat index requires NLIST > 0 and NPROBE > 0")

        if cls.INDEX_TYPE == "HNSW" and (cls.M <= 0 or cls.EF_CONSTRUCTION <= 0 or cls.EF_SEARCH <= 0):
            errors.append("HNSW index requires M > 0, EF_CONSTRUCTION > 0, and EF_SEARCH > 0")

        if cls.BATCH_SIZE <= 0:
            errors.append("BATCH_SIZE must be positive")

        # 验证默认业务类型
        try:
            cls._validate_businesstype(cls.DEFAULT_BUSINESSTYPE)
        except ValueError as e:
            errors.append(str(e))

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

        return True

# 生产环境配置示例
class ProductionConfig(Config):
    """生产环境配置"""
    INDEX_TYPE = "HNSW"  # 更好的精度性能平衡
    AUTO_SAVE = True
    LOG_LEVEL = "WARNING"
    BATCH_SIZE = 64
    USE_GPU = True
    ENABLE_SEARCH_OPTIMIZATION = True
    SEMANTIC_CHUNKING = True
    RELEVANCE_THRESHOLD = 0.6  # 生产环境更严格
    DIVERSITY_WEIGHT = 0.15
    # HNSW优化参数
    M = 32
    EF_CONSTRUCTION = 200
    EF_SEARCH = 100

# 开发环境配置示例
class DevelopmentConfig(Config):
    """开发环境配置"""
    INDEX_TYPE = "FlatIP"  # 使用简单索引便于调试
    AUTO_SAVE = False
    LOG_LEVEL = "DEBUG"
    BATCH_SIZE = 16
    ENABLE_SEARCH_OPTIMIZATION = True
    SEMANTIC_CHUNKING = True
    RELEVANCE_THRESHOLD = 0.4  # 开发环境更宽松
    DIVERSITY_WEIGHT = 0.3


# 根据环境变量选择配置
def get_config() -> Config:
    """根据环境变量获取配置"""
    env = os.getenv("ENVIRONMENT", "production").lower()
    
    if env == "production":
        return ProductionConfig()
    else:
        return DevelopmentConfig()
