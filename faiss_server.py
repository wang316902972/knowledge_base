import faiss
import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

# --- 初始化 ---
INDEX_FILE = "knowledge_base.index"
METADATA_FILE = "knowledge_base.pkl"
EMBEDDING_DIM = 384 # 'all-MiniLM-L6-v2' 模型的维度

# 加载嵌入模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 加载或创建 FAISS 索引和元数据
try:
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        id_to_chunk = pickle.load(f)
    print(f"已成功加载现有的索引，包含 {index.ntotal} 个向量。")
except (FileNotFoundError, RuntimeError):
    print("未找到现有索引，正在创建新索引...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    id_to_chunk = {} # 从索引ID映射到原始文本块

app = FastAPI()

# --- API 模型定义 ---
class Document(BaseModel):
    content: str
    chunk_size: int = 500
    chunk_overlap: int = 50

class Query(BaseModel):
    question: str
    top_k: int = 3

# --- API 端点 ---
@app.post("/add")
def add_document(doc: Document):
    """接收文本，分块，向量化，并添加到索引中"""
    # 简单的文本分块逻辑
    chunks = [
        doc.content[i: i + doc.chunk_size]
        for i in range(0, len(doc.content), doc.chunk_size - doc.chunk_overlap)
    ]

    # 生成向量
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    
    # 添加到 FAISS 和元数据
    start_id = index.ntotal
    index.add(embeddings)
    for i, chunk in enumerate(chunks):
        id_to_chunk[start_id + i] = chunk

    # 保存索引和元数据
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(id_to_chunk, f)
        
    return {"message": f"文档处理成功，新增 {len(chunks)} 个知识块。", "total_vectors": index.ntotal}

@app.delete("/delete")
def delete_document(doc: Document):
    """接收文本，找到对应的知识块并从索引中删除"""
    global id_to_chunk, index
    
    # 找出所有属于该文档的块的ID
    ids_to_remove = [
        id for id, chunk in id_to_chunk.items() if chunk in doc.content
    ]

    if not ids_to_remove:
        return {"message": "未找到与该文档内容匹配的知识块。", "total_vectors": index.ntotal}

    # 从 FAISS 索引中删除
    index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
    
    # 从元数据中删除
    for id in ids_to_remove:
        del id_to_chunk[id]

    # 保存更新后的索引和元数据
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(id_to_chunk, f)
        
    return {"message": f"成功删除 {len(ids_to_remove)} 个知识块。", "total_vectors": index.ntotal}

@app.post("/search")
def search_knowledge(query: Query):
    """接收问题，向量化，并在索引中搜索相关知识"""
    if index.ntotal == 0:
        print("知识库为空，请先添加文档。")
        return {"relevant_chunks": ["知识库为空，请先添加文档。"]}
    print(f"正在搜索与问题 '{query.question}' 相关的知识...") 
    query_embedding = model.encode([query.question], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    distances, indices = index.search(query_embedding, query.top_k)

    valid_indices = [i for i in indices[0] if i != -1]
    if not valid_indices:
        print("没有找到相关的知识。")
        return {"relevant_chunks": ["没有找到相关的知识。"]}
    
    # 根据索引ID获取原始文本块
    results = [id_to_chunk.get(i, "") for i in valid_indices]
        
    return {"relevant_chunks": results}
