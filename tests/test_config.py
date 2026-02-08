"""
配置模块单元测试
"""
import pytest
import tempfile
import os
from pathlib import Path
from config import Config, ProductionConfig, DevelopmentConfig, get_config
from exceptions import ConfigValidationError, InvalidBusinesstypeError


class TestConfigValidation:
    """测试配置验证"""

    def test_businesstype_validation_valid(self):
        """测试有效的业务类型验证"""
        config = Config()
        
        # 有效的业务类型
        valid_types = [
            "test",
            "my-business",
            "Test_Business_123",
            "a" * 50,  # 最大长度
        ]
        
        for bt in valid_types:
            result = config._validate_businesstype(bt)
            assert result == bt.lower()
            assert result.islower()

    def test_businesstype_validation_invalid(self):
        """测试无效的业务类型验证"""
        config = Config()
        
        # 无效的业务类型
        invalid_types = [
            "invalid@name",  # 包含特殊字符
            "has space",    # 包含空格
            "a" * 51,        # 超过最大长度
            "",              # 空字符串
        ]
        
        for bt in invalid_types:
            with pytest.raises(InvalidBusinesstypeError):
                config._validate_businesstype(bt)

    def test_config_validate_success(self):
        """测试配置验证成功"""
        config = Config()
        
        # 应该验证通过
        assert config.validate() is True

    def test_config_validate_chunk_size_error(self):
        """测试分块大小验证错误"""
        # 创建一个配置并修改参数
        config = Config()
        config.MIN_CHUNK_SIZE = 5000
        config.MAX_CHUNK_SIZE = 2000  # MIN > MAX
        
        with pytest.raises(ConfigValidationError):
            config.validate()

    def test_config_validate_ivf_params(self):
        """测试 IVF 索引参数验证"""
        config = Config()
        config.INDEX_TYPE = "IVFFlat"
        config.NLIST = 100
        config.NPROBE = 150  # NPROBE > NLIST
        
        with pytest.raises(ConfigValidationError):
            config.validate()


class TestConfigPaths:
    """测试配置路径生成"""

    def test_get_index_file_default(self):
        """测试获取默认索引文件路径"""
        config = Config()
        config.DEFAULT_BUSINESSTYPE = "test"
        
        index_file = config.get_index_file()
        
        assert "test" in index_file
        assert index_file.endswith(".index")

    def test_get_index_file_custom(self):
        """测试获取自定义索引文件路径"""
        config = Config()
        config.DEFAULT_BUSINESSTYPE = "default"
        
        index_file = config.get_index_file("my-business")
        
        assert "my-business" in index_file
        assert index_file.endswith(".index")

    def test_get_metadata_file(self):
        """测试获取元数据文件路径"""
        config = Config()
        config.DEFAULT_BUSINESSTYPE = "test"
        
        metadata_file = config.get_metadata_file()
        
        assert "test" in metadata_file
        assert metadata_file.endswith(".json")


class TestConfigEnvironments:
    """测试环境配置"""

    def test_production_config(self):
        """测试生产环境配置"""
        config = ProductionConfig()
        
        assert config.INDEX_TYPE == "HNSW"
        assert config.AUTO_SAVE is True
        assert config.LOG_LEVEL == "WARNING"
        assert config.BATCH_SIZE == 64
        assert config.USE_GPU is True

    def test_development_config(self):
        """测试开发环境配置"""
        config = DevelopmentConfig()
        
        assert config.INDEX_TYPE == "FlatIP"
        assert config.AUTO_SAVE is False
        assert config.LOG_LEVEL == "DEBUG"
        assert config.BATCH_SIZE == 16

    def test_get_config_production(self):
        """测试获取生产环境配置"""
        os.environ["ENVIRONMENT"] = "production"
        config = get_config()
        
        assert isinstance(config, ProductionConfig)

    def test_get_config_development(self):
        """测试获取开发环境配置"""
        os.environ["ENVIRONMENT"] = "development"
        config = get_config()
        
        assert isinstance(config, DevelopmentConfig)


class TestConfigDefaults:
    """测试配置默认值"""

    def test_default_values(self):
        """测试配置默认值"""
        config = Config()
        
        # 测试关键默认值
        assert config.EMBEDDING_DIM == 384
        assert config.INDEX_TYPE == "FlatIP"
        assert config.BATCH_SIZE == 32
        assert config.DEFAULT_TOP_K == 5
        assert config.DEFAULT_CHUNK_SIZE == 2000
        assert config.DEFAULT_CHUNK_OVERLAP == 100

    def test_default_businesstype(self):
        """测试默认业务类型"""
        config = Config()
        
        assert config.DEFAULT_BUSINESSTYPE == "default"

    def test_data_dir_creation(self):
        """测试数据目录创建"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.DATA_DIR = tmpdir
            
            # 初始化时应该创建目录
            config.__init__()
            
            assert os.path.exists(tmpdir)


class TestConfigWeights:
    """测试混合检索权重"""

    def test_weight_sum_validation(self):
        """测试权重和验证"""
        config = Config()
        
        # 权重和应该接近 1.0
        total = (
            config.VECTOR_SEARCH_WEIGHT +
            config.BM25_SEARCH_WEIGHT +
            config.EXACT_MATCH_WEIGHT
        )
        
        assert abs(total - 1.0) < 0.01  # 允许小的浮点误差


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
