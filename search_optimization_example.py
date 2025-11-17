#!/usr/bin/env python3
"""
搜索优化使用示例和测试脚本
演示如何使用优化功能提升搜索精度
"""

import requests
import json
import time
from typing import List, Dict

class SearchOptimizationDemo:
    """搜索优化演示类"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.sample_docs = [
            """机器学习是人工智能的一个重要分支，它使用算法来分析数据，从中学习模式，并做出预测或决策。机器学习的核心思想是让计算机能够从数据中自动学习，而不需要明确编程。主要类型包括监督学习、无监督学习和强化学习。监督学习使用标记的训练数据来学习输入和输出之间的映射关系，常见的算法有线性回归、逻辑回归、决策树和随机森林等。无监督学习则从未标记的数据中发现隐藏的模式和结构，包括聚类分析和主成分分析等技术。机器学习在各个领域都有广泛应用，从垃圾邮件过滤到自动驾驶汽车，从医疗诊断到金融风险评估，都发挥着重要作用。""",

            """深度学习是机器学习的一个子领域，它使用多层神经网络来模拟人脑的学习过程。深度学习的"深度"指的是神经网络具有多个隐藏层，这些层能够逐级提取数据的抽象特征。卷积神经网络（CNN）在图像识别和处理方面表现出色，能够自动学习图像的空间层次结构。循环神经网络（RNN）及其变体如长短期记忆网络（LSTM）特别适合处理序列数据，如自然语言和时间序列。Transformer架构的出现彻底改变了自然语言处理领域，其自注意力机制能够更好地捕捉长距离依赖关系。深度学习在计算机视觉、自然语言处理、语音识别等领域取得了突破性进展。""",

            """自然语言处理（Natural Language Processing, NLP）是人工智能的一个重要分支，专注于计算机与人类语言之间的交互。NLP的目标是让计算机能够理解、解释、生成人类语言。主要任务包括文本分类、情感分析、命名实体识别、关系抽取、机器翻译、问答系统和文本摘要等。传统的NLP方法依赖于语言学规则和统计模型，而现代NLP主要基于深度学习技术。词嵌入技术如Word2Vec和GloVe将词语映射到向量空间，捕捉语义关系。预训练语言模型如BERT、GPT和T5在各种NLP任务上都取得了state-of-the-art的性能。NLP技术广泛应用于搜索引擎、智能客服、内容推荐、机器翻译等领域。""",

            """计算机视觉是人工智能的一个领域，专注于让计算机能够理解和解释视觉信息。计算机视觉的目标是让机器能够像人类一样"看懂"图像和视频。核心任务包括图像分类、目标检测、语义分割、实例分割、人脸识别、姿态估计和图像生成等。传统方法依赖于手工设计的特征提取器，而现代计算机视觉主要基于深度学习，特别是卷积神经网络（CNN）。CNN能够自动学习图像的多层次特征表示，从边缘和纹理到复杂的形状和对象。预训练模型如ResNet、EfficientNet和Vision Transformer在各种视觉任务上都取得了优异性能。计算机视觉技术广泛应用于自动驾驶、医学影像分析、安防监控、工业检测等领域。""",

            """强化学习是一种机器学习方法，智能体通过与环境交互来学习最优策略。在强化学习中，智能体采取行动获得奖励或惩罚，目标是最大化累积奖励。关键概念包括状态、动作、奖励、策略和价值函数。马尔可夫决策过程（MDP）为强化学习提供了数学框架。主要算法包括Q-Learning、深度Q网络（DQN）、策略梯度方法和演员-评论家方法。强化学习在游戏AI（如AlphaGo）、机器人控制、资源调度、金融投资组合管理等领域有重要应用。然而，强化学习也面临着样本效率低、探索与利用的平衡、奖励设计困难等挑战。近年来，深度强化学习结合了深度学习的表示能力，在复杂环境中取得了显著成功。""",

            """数据挖掘是从大量数据中发现有用模式和知识的过程。数据挖掘结合了统计学、机器学习、数据库系统和可视化技术，旨在从数据中提取有价值的模式。主要任务包括分类、回归、聚类、关联规则挖掘、异常检测和时间序列分析等。分类任务预测离散的目标变量，如垃圾邮件检测。回归任务预测连续值，如房价预测。聚类将相似的数据点分组，如客户细分。关联规则发现项之间的关系，如购物篮分析。异常检测识别与正常模式不同的数据点，如欺诈检测。数据挖掘过程包括数据清洗、数据集成、数据选择、数据转换、模式挖掘和模式评估等步骤。常用的工具有Python的scikit-learn、pandas，R语言，以及专门的软件如Weka和KNIME。""",

            """Python是一种高级编程语言，以其简洁的语法、强大的功能和丰富的生态系统而闻名。Python的设计哲学强调代码的可读性和简洁性，使用缩进来定义代码块。Python支持多种编程范式，包括面向对象编程、函数式编程和过程式编程。Python拥有强大的标准库，涵盖了文件操作、网络通信、数据处理、图形界面等各个方面。更重要的是，Python拥有庞大的第三方包生态系统，PyPI（Python Package Index）上有超过300,000个包。在数据科学和机器学习领域，Python是事实上的标准语言，拥有NumPy、Pandas、Matplotlib、Scikit-learn、TensorFlow、PyTorch等强大的库。Python在Web开发、自动化脚本、科学计算、人工智能等领域都有广泛应用。""",

            """TensorFlow是Google开发的开源机器学习框架，专门用于深度学习和大规模数值计算。TensorFlow提供了灵活的编程模型，可以部署到各种平台，从移动设备到分布式计算集群。核心概念包括张量（多维数组）、计算图（表示计算的图形结构）、会话（执行计算图的环境）和变量（存储模型参数）。TensorFlow 2.x引入了Eager Execution，使开发更加直观，同时保持了TensorFlow 1.x的性能优势。Keras作为TensorFlow的高级API，提供了简洁的接口来构建和训练深度学习模型。TensorFlow拥有丰富的预训练模型库（TensorFlow Hub）和部署工具（TensorFlow Lite、TensorFlow.js、TensorFlow Serving）。TensorFlow在工业界和学术界都有广泛应用，支持从研究原型到生产部署的完整机器学习生命周期。""",

            """PyTorch是Facebook（现Meta）人工智能研究团队开发的深度学习框架，以其灵活性和易用性而受到研究人员和开发者的青睐。PyTorch的核心特点是动态计算图（Define-by-Run），允许在运行时修改计算图，这使得调试更加直观，便于实现复杂的模型结构。PyTorch提供了强大的张量操作库，支持GPU加速计算，并具有自动求导功能，简化了梯度计算过程。torch.nn模块提供了构建神经网络所需的组件，如层、激活函数、损失函数和优化器。PyTorch生态系统包括torchvision（计算机视觉）、torchaudio（音频处理）、torchtext（自然语言处理）等专门库。PyTorch在学术界特别受欢迎，是许多研究论文的首选框架。近年来，PyTorch在工业界的应用也在快速增长，许多公司选择PyTorch作为主要的深度学习平台。""",

            """神经网络是深度学习的基础，由多个相互连接的神经元层组成。每个神经元接收输入信号，应用激活函数，并产生输出信号传递给下一层的神经元。基本的神经网络结构包括输入层、隐藏层和输出层。前馈神经网络是最简单的类型，信息单向流动。卷积神经网络（CNN）专门用于处理网格状数据，如图像，使用卷积层自动学习空间特征。循环神经网络（RNN）适合处理序列数据，具有记忆功能来捕捉时间依赖关系。长短期记忆网络（LSTM）和门控循环单元（GRU）解决了传统RNN的梯度消失问题。Transformer架构基于自注意力机制，能够并行处理序列中的所有位置，在自然语言处理任务中取得了巨大成功。神经网络的训练通常使用反向传播算法和随机梯度下降（SGD）及其变种来优化网络参数。"""
        ]

        self.test_queries = [
            "机器学习的主要类型有哪些？",
            "卷积神经网络和循环神经网络的区别是什么？",
            "Python在数据科学和机器学习中的优势",
            "Transformer架构在自然语言处理中的作用",
            "强化学习在游戏AI中的应用案例",
            "数据挖掘的主要任务和技术",
            "TensorFlow和PyTorch的特点对比",
            "计算机视觉的核心任务有哪些？",
            "神经网络训练过程中的反向传播算法",
            "深度学习在医疗诊断中的应用"
        ]

    def add_sample_documents(self):
        """添加示例文档"""
        print("📚 正在添加示例文档...")

        for i, doc in enumerate(self.sample_docs):
            response = requests.post(
                f"{self.base_url}/add",
                json={
                    "content": doc,
                    "chunk_size": 500,
                    "chunk_overlap": 100
                }
            )

            if response.status_code == 200:
                print(f"✅ 文档 {i+1}/10 添加成功")
            else:
                print(f"❌ 文档 {i+1} 添加失败: {response.text}")

        # 手动保存
        save_response = requests.post(f"{self.base_url}/save")
        if save_response.status_code == 200:
            print("💾 文档已保存")

        print()

    def compare_search_methods(self):
        """对比传统搜索和优化搜索"""
        print("🔍 搜索精度对比测试")
        print("=" * 50)

        for query in self.test_queries:
            print(f"\n📝 查询: {query}")

            # 传统搜索
            traditional_response = requests.post(
                f"{self.base_url}/search",
                json={
                    "question": query,
                    "top_k": 5,
                    "use_optimization": False
                }
            )

            # 优化搜索
            optimized_response = requests.post(
                f"{self.base_url}/search",
                json={
                    "question": query,
                    "top_k": 5,
                    "use_optimization": True
                }
            )

            if traditional_response.status_code == 200:
                trad_data = traditional_response.json()
                trad_scores = [r.get('score', 0) for r in trad_data['detailed_results']]
                avg_trad = sum(trad_scores) / len(trad_scores) if trad_scores else 0
                print(f"  📊 传统搜索平均得分: {avg_trad:.4f}")

                if optimized_response.status_code == 200:
                    opt_data = optimized_response.json()
                    opt_scores = [r.get('relevance_score', r.get('score', 0)) for r in opt_data['detailed_results']]
                    avg_opt = sum(opt_scores) / len(opt_scores) if opt_scores else 0
                    improvement = ((avg_opt - avg_trad) / avg_trad * 100) if avg_trad > 0 else 0

                    print(f"  📈 优化搜索平均得分: {avg_opt:.4f}")
                    print(f"  🚀 精度提升: {improvement:+.2f}%")

                    # 显示最佳结果
                    if opt_data['detailed_results']:
                        best_result = opt_data['detailed_results'][0]['text']
                        print(f"  💡 最佳匹配: {best_result[:80]}...")
                else:
                    print(f"  ❌ 优化搜索失败: {optimized_response.text}")
                    print(f"  📈 优化搜索平均得分: 0.0000")
                    print(f"  🚀 精度提升: -100.00%")
            else:
                print(f"  ❌ 传统搜索失败: {traditional_response.text}")
                if optimized_response.status_code == 200:
                    print(f"  📈 优化搜索成功，但无法对比")
                else:
                    print(f"  ❌ 优化搜索也失败: {optimized_response.text}")

        print()

    def run_benchmark(self):
        """运行基准测试"""
        print("📊 运行搜索质量基准测试")
        print("=" * 50)

        benchmark_response = requests.post(
            f"{self.base_url}/benchmark_search_quality",
            json={"queries": self.test_queries}
        )

        if benchmark_response.status_code == 200:
            results = benchmark_response.json()
            print(f"✅ 基准测试完成")

            # 安全地获取各种字段
            if 'test_queries_count' in results:
                print(f"测试查询数量: {results['test_queries_count']}")
            if 'overall_improvement' in results:
                print(f"总体精度提升: {results['overall_improvement']:.4f}")
            if 'optimization_enabled' in results:
                print(f"优化功能状态: {'✅ 已启用' if results['optimization_enabled'] else '❌ 未启用'}")

            # 详细结果
            if 'detailed_results' in results:
                print("\n📋 详细测试结果:")
                for result in results['detailed_results']:
                    query = result.get('query', 'Unknown')[:30] + '...'
                    trad_score = result.get('traditional_avg_score', 0)
                    opt_score = result.get('optimized_avg_score', 0)
                    improvement = result.get('improvement', 0)

                    print(f"  查询: {query}")
                    print(f"  传统得分: {trad_score:.4f}")
                    print(f"  优化得分: {opt_score:.4f}")
                    print(f"  改进幅度: {improvement:+.4f}")
                    print()
            else:
                print("\n📊 基准测试响应结构:")
                print(f"  可用字段: {list(results.keys())}")
        else:
            print(f"❌ 基准测试失败: {benchmark_response.text}")
            print(f"   错误代码: {benchmark_response.status_code}")
            print("   建议检查服务器日志或API实现")

    def get_search_recommendations(self):
        """获取搜索优化建议"""
        print("💡 搜索优化建议")
        print("=" * 30)

        response = requests.get(f"{self.base_url}/search_recommendations")

        if response.status_code == 200:
            recommendations = response.json()

            # 获取系统状态信息
            stats_response = requests.get(f"{self.base_url}/stats")
            optimization_enabled = False
            current_vectors = 0
            current_index_type = "Unknown"

            if stats_response.status_code == 200:
                stats = stats_response.json()
                optimization_enabled = stats.get('search_optimization_enabled', False)
                current_vectors = stats.get('total_vectors', 0)
                current_index_type = stats.get('index_type', 'Unknown')

            print(f"当前向量数量: {current_vectors}")
            print(f"当前索引类型: {current_index_type}")
            print(f"优化功能状态: {'✅ 已启用' if optimization_enabled else '❌ 未启用'}")

            # 显示当前配置
            if 'current_config' in recommendations:
                config = recommendations['current_config']
                print("\n⚙️ 当前配置:")
                if 'ivf_nprobe' in config:
                    print(f"  IVF探测数量: {config['ivf_nprobe']}")
                if 'hnsw_ef_search' in config:
                    print(f"  HNSW搜索参数: {config['hnsw_ef_search']}")
                if 'diversity_weight' in config:
                    print(f"  多样性权重: {config['diversity_weight']}")
                if 'relevance_threshold' in config:
                    print(f"  相关性阈值: {config['relevance_threshold']}")

            # 显示建议索引类型
            if 'suggested_index_type' in recommendations:
                print(f"\n🎯 建议索引类型: {recommendations['suggested_index_type']}")

            print("\n🔧 优化建议:")
            for i, rec in enumerate(recommendations['recommendations'], 1):
                print(f"  {i}. {rec}")
        else:
            print(f"❌ 获取建议失败: {response.text}")

    def get_system_stats(self):
        """获取系统统计信息"""
        print("📈 系统统计信息")
        print("=" * 30)

        response = requests.get(f"{self.base_url}/stats")

        if response.status_code == 200:
            stats = response.json()
            print(f"总向量数量: {stats['total_vectors']}")
            print(f"嵌入维度: {stats['embedding_dim']}")
            print(f"索引类型: {stats['index_type']}")
            print(f"模型名称: {stats['model_name']}")
            print(f"搜索优化: {'✅ 已启用' if stats['search_optimization_enabled'] else '❌ 未启用'}")

            # 索引特定信息
            if 'is_trained' in stats:
                print(f"索引训练状态: {'✅ 已训练' if stats['is_trained'] else '❌ 未训练'}")
            if 'nlist' in stats:
                print(f"IVF聚类数量: {stats['nlist']}")
            if 'nprobe' in stats:
                print(f"IVF探测数量: {stats['nprobe']}")
        else:
            print(f"❌ 获取统计信息失败: {response.text}")

    def enable_optimization(self):
        """启用搜索优化"""
        print("🚀 正在启用搜索优化...")

        response = requests.post(f"{self.base_url}/enable_optimization")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            print("\n🎯 启用的功能:")
            for feature in result['features']:
                print(f"  • {feature}")
        else:
            print(f"❌ 启用优化失败: {response.text}")

    def run_complete_demo(self):
        """运行完整演示"""
        print("🎯 FAISS搜索优化完整演示")
        print("=" * 50)

        try:
            # 1. 添加示例文档
            self.add_sample_documents()

            # 2. 获取初始统计信息
            self.get_system_stats()

            # 3. 获取优化建议
            self.get_search_recommendations()

            # 4. 启用优化功能
            self.enable_optimization()

            # 5. 对比搜索方法
            self.compare_search_methods()

            # 6. 运行基准测试
            self.run_benchmark()

            # 7. 最终统计信息
            print("\n📋 最终系统状态:")
            self.get_system_stats()

            print("\n🎉 演示完成！")

        except requests.exceptions.ConnectionError:
            print("❌ 无法连接到FAISS服务器，请确保服务器正在运行")
        except Exception as e:
            print(f"❌ 演示过程中发生错误: {e}")

    def test_chunk_effectiveness(self):
        """测试分块效果"""
        print("🔧 测试文档分块效果")
        print("=" * 30)

        # 添加一个测试文档并检查分块结果
        test_doc = self.sample_docs[0]  # 使用第一个文档作为测试

        response = requests.post(
            f"{self.base_url}/add",
            json={
                "content": test_doc,
                "chunk_size": 500,
                "chunk_overlap": 100,
                "return_chunks": True  # 假设API支持返回分块信息
            }
        )

        if response.status_code == 200:
            result = response.json()
            if 'chunks' in result:
                print(f"✅ 原始文档长度: {len(test_doc)} 字符")
                print(f"✅ 生成分块数量: {len(result['chunks'])}")
                print("\n📋 分块详情:")
                for i, chunk in enumerate(result['chunks']):
                    print(f"  分块 {i+1}: {len(chunk)} 字符")
                    print(f"  内容预览: {chunk[:100]}...")
                    print()
            else:
                print("✅ 文档添加成功，但未返回分块详情")
                print(f"📊 原始文档长度: {len(test_doc)} 字符")
        else:
            print(f"❌ 测试分块失败: {response.text}")

    def analyze_document_quality(self):
        """分析文档质量"""
        print("📊 分析示例文档质量")
        print("=" * 30)

        for i, doc in enumerate(self.sample_docs):
            char_count = len(doc)
            word_count = len(doc.replace('，', ' ').replace('。', ' ').replace('！', ' ').replace('？', ' ').split())
            sentence_count = doc.count('。') + doc.count('！') + doc.count('？') + doc.count('.')

            print(f"📄 文档 {i+1} ({doc.split('，')[0]}...):")
            print(f"  字符数: {char_count}")
            print(f"  词数: {word_count}")
            print(f"  句子数: {sentence_count}")
            print(f"  预计分块数: {max(1, char_count // 400)}")  # 假设400字符/分块
            print()

        total_chars = sum(len(doc) for doc in self.sample_docs)
        avg_chars = total_chars / len(self.sample_docs)

        print(f"📈 总体统计:")
        print(f"  总文档数: {len(self.sample_docs)}")
        print(f"  总字符数: {total_chars}")
        print(f"  平均文档长度: {avg_chars:.1f} 字符")
        print(f"  预计总分块数: {max(10, total_chars // 400)}")

if __name__ == "__main__":
    print("FAISS搜索优化演示工具")
    print("请确保FAISS服务器正在 http://localhost:8001 运行")
    print()

    demo = SearchOptimizationDemo()

    # 先分析文档质量
    demo.analyze_document_quality()
    print()

    # 运行完整演示
    demo.run_complete_demo()