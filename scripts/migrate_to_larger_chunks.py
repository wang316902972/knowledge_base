#!/usr/bin/env python3
"""
数据迁移脚本：重新处理现有数据以使用更大的 chunk_size

用于修复数据被截断的问题
"""

import json
import logging
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from faiss_server_optimized import FaissVectorDB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_businesstype(businesstype: str, new_chunk_size: int = 2000,
                         new_chunk_overlap: int = 100, backup: bool = True):
    """
    迁移指定业务类型的数据，使用更大的 chunk_size

    Args:
        businesstype: 业务类型标识符
        new_chunk_size: 新的分块大小（默认 2000）
        new_chunk_overlap: 新的分块重叠（默认 100）
        backup: 是否备份旧数据（默认 True）
    """
    config = get_config()

    # 验证 businesstype
    try:
        businesstype = config._validate_businesstype(businesstype)
    except ValueError as e:
        logger.error(f"Invalid businesstype: {e}")
        return False

    logger.info(f"开始迁移业务类型: {businesstype}")
    logger.info(f"新配置: chunk_size={new_chunk_size}, chunk_overlap={new_chunk_overlap}")

    # 备份旧数据
    if backup:
        index_file = config.get_index_file(businesstype)
        metadata_file = config.get_metadata_file(businesstype)

        backup_suffix = f".bak.{os.getpid()}"

        if os.path.exists(index_file):
            backup_index = index_file + backup_suffix
            import shutil
            shutil.copy2(index_file, backup_index)
            logger.info(f"已备份索引文件: {backup_index}")

        if os.path.exists(metadata_file):
            backup_metadata = metadata_file + backup_suffix
            import shutil
            shutil.copy2(metadata_file, backup_metadata)
            logger.info(f"已备份元数据文件: {backup_metadata}")

    # 读取现有元数据
    metadata_file = config.get_metadata_file(businesstype)
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"读取元数据失败: {e}")
        return False

    # 提取所有原始文本
    id_to_chunk = metadata.get('id_to_chunk', {})
    logger.info(f"找到 {len(id_to_chunk)} 个文档块")

    # 收集所有唯一文本（去除重复）
    unique_texts = list(set(id_to_chunk.values()))
    logger.info(f"去重后有 {len(unique_texts)} 个唯一文档")

    if not unique_texts:
        logger.warning("没有文档需要迁移")
        return True

    # 创建新的数据库实例
    try:
        db = FaissVectorDB(config, businesstype)

        # 清空现有索引
        logger.info("清空现有索引...")
        db.index = db._create_index()
        db.id_to_chunk = {}
        db.chunk_to_id = {}

        # 使用新的 chunk_size 重新添加所有文本
        logger.info(f"使用 chunk_size={new_chunk_size} 重新添加文档...")

        # 重新分块并添加
        all_chunks = []
        for text in unique_texts:
            chunks = db._generate_chunks(text, new_chunk_size, new_chunk_overlap)
            all_chunks.extend(chunks)

        logger.info(f"生成了 {len(all_chunks)} 个新文档块")

        # 批量添加
        if all_chunks:
            db.add_texts(all_chunks)
            logger.info(f"成功添加 {len(all_chunks)} 个文档块")

        # 保存
        logger.info("保存新索引和元数据...")
        db.save()

        logger.info(f"✅ 迁移完成！")
        logger.info(f"   原始文档数: {len(unique_texts)}")
        logger.info(f"   新文档块数: {len(all_chunks)}")
        logger.info(f"   总向量数: {db.index.ntotal}")

        return True

    except Exception as e:
        logger.error(f"迁移失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # 如果失败，恢复备份
        if backup:
            logger.info("尝试恢复备份...")
            try:
                if os.path.exists(index_file + backup_suffix):
                    import shutil
                    shutil.move(index_file + backup_suffix, index_file)
                    logger.info("已恢复索引文件")

                if os.path.exists(metadata_file + backup_suffix):
                    import shutil
                    shutil.move(metadata_file + backup_suffix, metadata_file)
                    logger.info("已恢复元数据文件")
            except Exception as e2:
                logger.error(f"恢复备份失败: {e2}")

        return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='迁移数据到更大的 chunk_size')
    parser.add_argument('businesstype', help='业务类型标识符')
    parser.add_argument('--chunk-size', type=int, default=2000,
                       help='新的分块大小（默认: 2000）')
    parser.add_argument('--chunk-overlap', type=int, default=100,
                       help='新的分块重叠（默认: 100）')
    parser.add_argument('--no-backup', action='store_true',
                       help='不备份旧数据')

    args = parser.parse_args()

    success = migrate_businesstype(
        args.businesstype,
        new_chunk_size=args.chunk_size,
        new_chunk_overlap=args.chunk_overlap,
        backup=not args.no_backup
    )

    if success:
        logger.info("✅ 迁移成功完成")
        sys.exit(0)
    else:
        logger.error("❌ 迁移失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
