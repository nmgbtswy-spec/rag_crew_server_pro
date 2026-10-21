import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from mcp.server.fastmcp import FastMCP


OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-embedding:4b")


DISTANCE_THRESHOLD = 0.8
N_RESULTS = 4

# 初始化 MCP Server
mcp = FastMCP("RAG-Knowledge-Server",host=MCP_HOST,port=MCP_PORT)

# 加载向量模型和数据库
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

ollama_ef = OllamaEmbeddingFunction(
    url=OLLAMA_BASE_URL,
    model_name=MODEL_NAME
)

client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_or_create_collection(
    name='plc_knowledge',
    metadata={"hnsw:space": "cosine"},       # 余弦距离向量
    embedding_function=ollama_ef             # 挂载 ollama_ef
)

@mcp.tool()
def search_knowledge(query: str) -> str:
    """去向量库检索最相关的文本"""
    print(f"\n 大模型的搜索词是{query}\n")
    # query_embedding = model.encode(query).tolist()                # 旧代码，自己手动调用模型变向量
    results = collection.query(
        # query_embeddings=[query_embedding],                       # 不能用了，换下面的参数，直接传入旧文本
        query_texts=query,
        n_results=N_RESULTS
    )

    # 用 Zip 配对，过滤距离大于 0.8 的废话，并把结果拼成一段话
    valid_docs = []
    for doc, distance, meta in zip(results['documents'][0], results['distances'][0],results['metadatas'][0]):
        print(f"DEBUG -> 检索到：{doc[:10]}...距离：{distance}")
        if distance <= DISTANCE_THRESHOLD:
            # 从 meta 中抠出页码，如果没找到显示“未知”
            page = meta.get("page", "未知")           # "page" 是键，提取出的 meta 是整个的 metadatas 元数据对应的整包标签，里面包含了filename,page,file_id,status
            # 从 meta 中找到文件名
            file_name = meta.get("filename", "未知")
            # 把页码、文件名和原文拼在一起，告诉大模型这是哪里的内容
            valid_docs.append(f"（来源：《{file_name}》第{page}页）\n{doc}")
    if not valid_docs:
        return "知识库中为找到相关内容。"

    return "\n\n".join(valid_docs)


if __name__ == '__main__':
    # 关键：以 SSE (HTTP) 模式运行，避免 Windows 下 stdio 拉起进程报错
    print(f"启动 RAG MCP Server，监听 http://{MCP_HOST}:{MCP_PORT}/sse")
    mcp.run(transport="sse")