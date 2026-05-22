"""
FAISS向量数据库配置文件

提供完整的配置管理，包括类型注解、验证和统一异常处理。
"""
import os
import re
from pathlib import Path
from typing import Optional, Literal

from exceptions import ConfigValidationError, InvalidBusinesstypeError


class Config:
    """向量数据库配置类

    所有配置参数都通过环境变量设置，并提供合理的默认值。
    配置参数支持类型转换和验证。

    Attributes:
        DEFAULT_BUSINESSTYPE: 默认业务类型标识符
        SERVICE_NAME: 服务名称
        DATA_DIR: 数据目录路径

    Example:
        >>> config = Config()
        >>> config.validate()
        >>> index_file = config.get_index_file("my_business")
    """

    # ==================== 业务标识配置 ====================
    DEFAULT_BUSINESSTYPE: str = os.getenv("BUINESSTYPE", "default")
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "faiss-vector-db")

    # ==================== 文件路径配置 ====================
    SCRIPT_DIR: str = str(Path(__file__).parent)
    DATA_DIR: str = os.getenv("FAISS_DATA_DIR", os.path.join(SCRIPT_DIR, "data"))

    # ==================== 模型配置 ====================
    MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))

    # ==================== FAISS索引配置 ====================
    INDEX_TYPE: Literal["FlatIP", "FlatL2", "IVFFlat", "HNSW"] = os.getenv(
        "FAISS_INDEX_TYPE", "FlatIP"
    )  # type: ignore[assignment]
    NLIST: int = int(os.getenv("FAISS_NLIST", "100"))  # IVF索引的聚类数量
    NPROBE: int = int(os.getenv("FAISS_NPROBE", "10"))  # IVF索引搜索时的探测数量
    M: int = int(os.getenv("FAISS_HNSW_M", "16"))  # HNSW索引的连接数
    EF_CONSTRUCTION: int = int(os.getenv("FAISS_EF_CONSTRUCTION", "200"))  # HNSW构建时的搜索深度
    EF_SEARCH: int = int(os.getenv("FAISS_EF_SEARCH", "100"))  # HNSW搜索时的深度

    # ==================== 文本处理配置 ====================
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "5000"))
    MIN_CHUNK_SIZE: int = int(os.getenv("MIN_CHUNK_SIZE", "100"))
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "2000"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "100"))

    # ==================== 批处理配置 ====================
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    MAX_BATCH_TEXTS: int = int(os.getenv("MAX_BATCH_TEXTS", "100"))

    # ==================== 搜索配置 ====================
    MAX_TOP_K: int = int(os.getenv("MAX_TOP_K", "50"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))

    # ==================== 搜索优化配置 ====================
    ENABLE_SEARCH_OPTIMIZATION: bool = (
        os.getenv("ENABLE_SEARCH_OPTIMIZATION", "true").lower() == "true"
    )
    SEMANTIC_CHUNKING: bool = os.getenv("SEMANTIC_CHUNKING", "true").lower() == "true"
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))
    DIVERSITY_WEIGHT: float = float(os.getenv("DIVERSITY_WEIGHT", "0.2"))
    SEARCH_CANDIDATE_MULTIPLIER: int = int(os.getenv("SEARCH_CANDIDATE_MULTIPLIER", "10"))

    # ==================== 检索增强配置 (Phase 1) ====================
    BASE_RELEVANCE_THRESHOLD: float = float(os.getenv("BASE_RELEVANCE_THRESHOLD", "0.1"))
    MIN_RELEVANCE_THRESHOLD: float = float(os.getenv("MIN_RELEVANCE_THRESHOLD", "0.05"))
    MAX_RELEVANCE_THRESHOLD: float = float(os.getenv("MAX_RELEVANCE_THRESHOLD", "0.3"))
    ENABLE_ADAPTIVE_THRESHOLD: bool = (
        os.getenv("ENABLE_ADAPTIVE_THRESHOLD", "true").lower() == "true"
    )

    # ==================== 查询增强配置 (Phase 2) ====================
    ENABLE_QUERY_EXPANSION: bool = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
    MAX_EXPANDED_QUERIES: int = int(os.getenv("MAX_EXPANDED_QUERIES", "5"))
    DOMAIN_TERM_DICT_PATH: str = os.getenv("DOMAIN_TERM_DICT_PATH", "config/domain_terms.json")

    # ==================== 混合检索配置 (Phase 3) ====================
    ENABLE_HYBRID_RETRIEVAL: bool = os.getenv("ENABLE_HYBRID_RETRIEVAL", "true").lower() == "true"
    VECTOR_SEARCH_WEIGHT: float = float(os.getenv("VECTOR_SEARCH_WEIGHT", "0.7"))
    BM25_SEARCH_WEIGHT: float = float(os.getenv("BM25_SEARCH_WEIGHT", "0.2"))
    EXACT_MATCH_WEIGHT: float = float(os.getenv("EXACT_MATCH_WEIGHT", "0.1"))
    ENABLE_BM25_FALLBACK: bool = os.getenv("ENABLE_BM25_FALLBACK", "true").lower() == "true"
    ENABLE_EXACT_MATCH_FALLBACK: bool = (
        os.getenv("ENABLE_EXACT_MATCH_FALLBACK", "true").lower() == "true"
    )

    # ==================== 性能配置 ====================
    AUTO_SAVE: bool = os.getenv("AUTO_SAVE", "false").lower() == "true"
    SAVE_INTERVAL: int = int(os.getenv("SAVE_INTERVAL", "100"))

    # ==================== 安全配置 ====================
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", "50000"))
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "1000"))

    # ==================== API配置 ====================
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8001"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))

    # ==================== 日志配置 ====================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # ==================== GPU配置 ====================
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"
    GPU_ID: int = int(os.getenv("GPU_ID", "0"))

    # ==================== 环境配置 ====================
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production").lower()

    def __init__(self) -> None:
        """初始化配置"""
        # 确保数据目录存在
        Path(self.DATA_DIR).mkdir(parents=True, exist_ok=True)

    def get_index_file(self, businesstype: Optional[str] = None) -> str:
        """动态生成索引文件路径

        Args:
            businesstype: 业务类型标识符，None 表示使用默认值

        Returns:
            索引文件的完整路径

        Raises:
            InvalidBusinesstypeError: 如果业务类型无效

        Example:
            >>> config = Config()
            >>> index_file = config.get_index_file("my_business")
            >>> print(index_file)
            '/path/to/data/my_business/my_business_knowledge_base.index'
        """
        if businesstype is None:
            businesstype = self.DEFAULT_BUSINESSTYPE

        businesstype = self._validate_businesstype(businesstype)
        filename = f"{businesstype}_knowledge_base.index"
        businesstype_dir = Path(self.DATA_DIR) / businesstype
        return str(
            os.getenv(
                "FAISS_INDEX_FILE",
                str(businesstype_dir / filename),
            )
        )

    def get_metadata_file(self, businesstype: Optional[str] = None) -> str:
        """动态生成元数据文件路径

        Args:
            businesstype: 业务类型标识符，None 表示使用默认值

        Returns:
            元数据文件的完整路径

        Raises:
            InvalidBusinesstypeError: 如果业务类型无效

        Example:
            >>> config = Config()
            >>> metadata_file = config.get_metadata_file("my_business")
            >>> print(metadata_file)
            '/path/to/data/my_business/my_business_knowledge_base.json'
        """
        if businesstype is None:
            businesstype = self.DEFAULT_BUSINESSTYPE

        businesstype = self._validate_businesstype(businesstype)
        filename = f"{businesstype}_knowledge_base.json"
        businesstype_dir = Path(self.DATA_DIR) / businesstype
        return str(
            os.getenv(
                "FAISS_METADATA_FILE",
                str(businesstype_dir / filename),
            )
        )

    @staticmethod
    def _validate_businesstype(businesstype: str) -> str:
        """验证和清理业务类型名称

        Args:
            businesstype: 业务类型名称

        Returns:
            验证并清理后的业务类型名称（小写）

        Raises:
            InvalidBusinesstypeError: 如果业务类型名称不符合规范

        Example:
            >>> Config._validate_businesstype("My-Business_123")
            'my-business_123'
            >>> Config._validate_businesstype("invalid@name")
            Traceback (most recent call last):
                ...
            InvalidBusinesstypeError: [VALIDATE003] Invalid businesstype: 'invalid@name'...
        """
        # 只允许字母、数字、下划线和连字符；最长50个字符
        pattern = r"^[a-zA-Z0-9_-]{1,50}$"
        if not re.match(pattern, businesstype):
            raise InvalidBusinesstypeError(
                businesstype=businesstype,
                reason="Must be 1-50 alphanumeric characters, underscores, or hyphens only",
            )
        return businesstype.lower()

    def validate(self) -> bool:
        """验证配置参数

        检查所有配置参数是否符合要求，确保参数之间的依赖关系正确。

        Returns:
            True 如果验证通过

        Raises:
            ConfigValidationError: 如果配置验证失败

        Example:
            >>> config = Config()
            >>> try:
            ...     config.validate()
            ...     print("Configuration is valid")
            ... except ConfigValidationError as e:
            ...     print(f"Configuration error: {e}")
        """
        errors: list[str] = []

        # 验证分块大小配置
        if self.MIN_CHUNK_SIZE >= self.MAX_CHUNK_SIZE:
            errors.append(
                f"MIN_CHUNK_SIZE ({self.MIN_CHUNK_SIZE}) must be less than MAX_CHUNK_SIZE ({self.MAX_CHUNK_SIZE})"
            )

        if self.DEFAULT_CHUNK_OVERLAP >= self.DEFAULT_CHUNK_SIZE:
            errors.append(
                f"DEFAULT_CHUNK_OVERLAP ({self.DEFAULT_CHUNK_OVERLAP}) must be less than DEFAULT_CHUNK_SIZE ({self.DEFAULT_CHUNK_SIZE})"
            )

        # 验证索引类型配置
        valid_index_types = ["FlatIP", "FlatL2", "IVFFlat", "HNSW"]
        if self.INDEX_TYPE not in valid_index_types:
            errors.append(f"Unsupported INDEX_TYPE: {self.INDEX_TYPE}")

        # IVF 索引特定验证
        if self.INDEX_TYPE == "IVFFlat":
            if self.NLIST <= 0:
                errors.append("IVFFlat index requires NLIST > 0")
            if self.NPROBE <= 0:
                errors.append("IVFFlat index requires NPROBE > 0")
            if self.NPROBE > self.NLIST:
                errors.append(f"NPROBE ({self.NPROBE}) cannot exceed NLIST ({self.NLIST})")

        # HNSW 索引特定验证
        if self.INDEX_TYPE == "HNSW":
            if self.M <= 0:
                errors.append("HNSW index requires M > 0")
            if self.EF_CONSTRUCTION <= 0:
                errors.append("HNSW index requires EF_CONSTRUCTION > 0")
            if self.EF_SEARCH <= 0:
                errors.append("HNSW index requires EF_SEARCH > 0")

        # 验证批处理配置
        if self.BATCH_SIZE <= 0:
            errors.append("BATCH_SIZE must be positive")

        # 验证混合检索权重
        total_weight = self.VECTOR_SEARCH_WEIGHT + self.BM25_SEARCH_WEIGHT + self.EXACT_MATCH_WEIGHT
        if abs(total_weight - 1.0) > 0.01:  # 允许小的浮点误差
            errors.append(
                f"Hybrid retrieval weights must sum to 1.0, got {total_weight:.2f}"
            )

        # 验证默认业务类型
        try:
            self._validate_businesstype(self.DEFAULT_BUSINESSTYPE)
        except InvalidBusinesstypeError as e:
            errors.append(str(e))

        # 如果有错误，抛出异常
        if errors:
            raise ConfigValidationError(
                message="Configuration validation failed",
                details={"errors": errors},
            )

        return True


class ProductionConfig(Config):
    """生产环境配置

    针对生产环境优化的配置，注重性能和稳定性。

    Example:
        >>> config = ProductionConfig()
        >>> config.validate()
    """

    INDEX_TYPE: Literal["FlatIP", "FlatL2", "IVFFlat", "HNSW"] = os.getenv(
        "FAISS_INDEX_TYPE", "HNSW"
    )  # type: ignore[assignment]
    AUTO_SAVE: bool = os.getenv("AUTO_SAVE", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "WARNING")
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "64"))
    USE_GPU: bool = os.getenv("USE_GPU", "true").lower() == "true"
    ENABLE_SEARCH_OPTIMIZATION: bool = True
    SEMANTIC_CHUNKING: bool = True
    RELEVANCE_THRESHOLD: float = 0.6
    DIVERSITY_WEIGHT: float = 0.15

    # HNSW 优化参数
    M: int = 32
    EF_CONSTRUCTION: int = 200
    EF_SEARCH: int = 100


class DevelopmentConfig(Config):
    """开发环境配置

    针对开发环境优化的配置，注重调试和开发效率。

    Example:
        >>> config = DevelopmentConfig()
        >>> config.validate()
    """

    INDEX_TYPE: Literal["FlatIP", "FlatL2", "IVFFlat", "HNSW"] = "FlatIP"  # type: ignore[assignment]
    AUTO_SAVE: bool = False
    LOG_LEVEL: str = "DEBUG"
    BATCH_SIZE: int = 16
    ENABLE_SEARCH_OPTIMIZATION: bool = True
    SEMANTIC_CHUNKING: bool = True
    RELEVANCE_THRESHOLD: float = 0.4
    DIVERSITY_WEIGHT: float = 0.3


def get_config() -> Config:
    """根据环境变量获取配置

    根据环境变量 ENVIRONMENT 自动选择合适的配置类。

    Returns:
        配置实例

    Example:
        >>> config = get_config()
        >>> print(config.ENVIRONMENT)
        'production'
    """
    env = os.getenv("ENVIRONMENT", "production").lower()

    if env == "production":
        return ProductionConfig()
    else:
        return DevelopmentConfig()
