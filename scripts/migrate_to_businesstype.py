#!/usr/bin/env python3
"""
迁移脚本: 将扁平的数据结构转换为业务类型子目录结构

旧结构:
data/
├── default_knowledge_base.index
└── default_knowledge_base.json

新结构:
data/
├── default/
│   ├── default_knowledge_base.index
│   └── default_knowledge_base.json
├── sd/
│   ├── sd_knowledge_base.index
│   └── sd_knowledge_base.json
└── warning/
    ├── warning_knowledge_base.index
    └── warning_knowledge_base.json
"""

import os
import shutil
from pathlib import Path
import sys


def migrate_old_data(data_dir: str, dry_run: bool = False):
    """
    迁移旧的扁平结构到新的分层结构

    Args:
        data_dir: 数据目录路径
        dry_run: 是否只是预览而不实际执行迁移

    Returns:
        bool: 迁移是否成功
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"❌ 错误: 数据目录不存在: {data_dir}")
        return False

    if not data_path.is_dir():
        print(f"❌ 错误: 路径不是目录: {data_dir}")
        return False

    # 查找旧格式的文件
    old_index_files = list(data_path.glob("*_knowledge_base.index"))
    old_metadata_files = list(data_path.glob("*_knowledge_base.json"))

    if not old_index_files:
        print("ℹ️  未找到旧格式文件，无需迁移。")
        return True

    print(f"📊 找到 {len(old_index_files)} 个索引文件需要迁移")
    print()

    success_count = 0
    failed_count = 0

    for old_index in old_index_files:
        try:
            # 从文件名提取业务类型ID
            # 例如: "default_knowledge_base.index" -> "default"
            business_type = old_index.stem.replace("_knowledge_base", "")

            # 创建新的子目录
            new_dir = data_path / business_type

            if not dry_run:
                new_dir.mkdir(exist_ok=True)

            # 移动索引文件
            new_index = new_dir / old_index.name

            if new_index.exists():
                print(f"⊘ 跳过（已存在）: {new_index}")
            else:
                if dry_run:
                    print(f"[预览] 将迁移: {old_index} -> {new_index}")
                else:
                    shutil.move(str(old_index), str(new_index))
                    print(f"✓ 已迁移: {old_index} -> {new_index}")
                    success_count += 1

            # 移动元数据文件（如果存在）
            old_metadata = old_index.with_suffix(".json")
            new_metadata = new_dir / old_metadata.name

            if old_metadata.exists():
                if new_metadata.exists():
                    print(f"⊘ 跳过（已存在）: {new_metadata}")
                else:
                    if dry_run:
                        print(f"[预览] 将迁移: {old_metadata} -> {new_metadata}")
                    else:
                        shutil.move(str(old_metadata), str(new_metadata))
                        print(f"✓ 已迁移: {old_metadata} -> {new_metadata}")

        except Exception as e:
            print(f"❌ 迁移失败 {old_index}: {e}")
            failed_count += 1
            return False

    print()
    if failed_count == 0:
        print(f"✅ 迁移完成: {success_count}/{len(old_index_files)} 个文件迁移成功")
        return True
    else:
        print(f"⚠️  迁移完成但存在失败: {success_count} 成功, {failed_count} 失败")
        return False


def verify_migration(data_dir: str):
    """验证迁移结果"""
    data_path = Path(data_dir)

    print("\n📋 验证迁移结果:")
    print("=" * 60)

    # 检查子目录
    subdirs = [d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    print(f"\n子目录数量: {len(subdirs)}")

    for subdir in sorted(subdirs):
        index_files = list(subdir.glob("*.index"))
        metadata_files = list(subdir.glob("*.json"))

        print(f"\n📁 {subdir.name}/")
        print(f"  - 索引文件: {len(index_files)}")
        for idx in index_files:
            print(f"    ✓ {idx.name}")
        print(f"  - 元数据文件: {len(metadata_files)}")
        for meta in metadata_files:
            print(f"    ✓ {meta.name}")

    # 检查是否还有遗留的旧格式文件
    old_index_files = list(data_path.glob("*_knowledge_base.index"))
    old_metadata_files = list(data_path.glob("*_knowledge_base.json"))

    if old_index_files or old_metadata_files:
        print(f"\n⚠️  警告: 仍有旧格式文件未迁移:")
        for f in old_index_files + old_metadata_files:
            print(f"  - {f}")
    else:
        print(f"\n✅ 没有遗留的旧格式文件")


def main():
    """主函数"""
    # 解析命令行参数
    data_dir = "./data"
    dry_run = False

    if len(sys.argv) > 1:
        if sys.argv[1] == "--dry-run" or sys.argv[1] == "-n":
            dry_run = True
        elif len(sys.argv) > 2:
            data_dir = sys.argv[1]
            if sys.argv[2] == "--dry-run" or sys.argv[2] == "-n":
                dry_run = True
        else:
            data_dir = sys.argv[1]

    print("=" * 60)
    print("FAISS 向量数据库迁移脚本")
    print("=" * 60)
    print(f"目标目录: {data_dir}")
    if dry_run:
        print("模式: 预览（不会实际迁移文件）")
    print()

    # 确认是否继续
    if not dry_run:
        response = input("⚠️  此操作将移动文件。继续迁移? (yes/no): ")
        if response.lower() != "yes":
            print("❌ 迁移已取消")
            sys.exit(0)
        print()

    # 执行迁移
    success = migrate_old_data(data_dir, dry_run=dry_run)

    if success:
        if not dry_run:
            print()
            verify_migration(data_dir)

            print("\n📝 后续步骤:")
            print("1. 验证新的目录结构: ls -R data/")
            print("2. 测试应用: docker-compose up")
            print("3. 确认无误后删除备份: rm -rf data.backup.*")
        sys.exit(0)
    else:
        print("\n❌ 迁移失败，请检查上述错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
