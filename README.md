# Local Agentic RAG

本地私有化的 Agentic RAG 问答系统：**CrewAI + MCP + ChromaDB + Ollama**，全链路可 Docker 化部署，不依赖任何云服务。

## 它是什么

把一份 PDF 技术手册切成片段、向量化后存进本地向量库；用户在前端提问时，Agent 通过 **MCP 协议**调用检索工具，从知识库取回相关段落，再交给本地大模型组织答案，并标注出处页码。

## 架构

```
浏览器 ──▶ web_ui (Streamlit :8501)
              │  连接 MCP（SSE）
              ▼
          mcp_server (FastMCP :8000)
              ├─▶ ChromaDB（本地向量库，挂载 ./chroma_db）
              └─▶ Ollama（embedding: qwen3-embedding:4b）
                      │
          回答经 Agent ──▶ Ollama（chat: qwen3.5:4b）──▶ 流式返回前端
```

## 快速开始

前置条件：

- Docker Desktop（或 Linux 上的 Docker + Compose v2）
- [Ollama](https://ollama.com/) 已在本机运行

```bash
# 1. 拉取需要的两个模型
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:4b

# 2. 启动
docker compose up -d --build

# 3. 打开前端
# http://localhost:8501
```

首次启动 `mcp_server` 需要加载向量库和模型，健康检查通过后再启动前端（约 1 分钟）。

## 目录结构

```
.
├── app.py                  # Streamlit 前端
├── agent_core.py           # CrewAI Agent + MCP 客户端
├── rag_mcp_server.py       # FastMCP 服务端，暴露 search_knowledge 工具
├── ingest_pdf.py           # 灌库脚本：PDF → 切片 → 向量 → ChromaDB
├── Dockerfile
├── docker-compose.yml      # mcp_server + web_ui 两个服务
├── chroma_db/              # 向量库（挂载，不进镜像）
└── src/rag_crew_demo/
```

## 配置

| 变量 | 作用 | 默认 |
|---|---|---|
| `OLLAMA_HOST` | Ollama 地址 | `http://127.0.0.1:11434` |
| `MCP_SERVER_URL` | MCP 服务地址（容器内用服务名） | `http://127.0.0.1:8000/sse` |
| `MCP_HOST` / `MCP_PORT` | MCP 监听地址 | `0.0.0.0` / `8000` |
| `LLM_NAME` | 对话模型 | `qwen3.5:4b` |
| `MODEL_NAME` | Embedding 模型 | `qwen3-embedding:4b` |

容器之间通信用**服务名**（`http://mcp_server:8000/sse`）；访问宿主服务用 `host.docker.internal`。

## 本机开发（不用 Docker）

```bash
uv sync
python ingest_pdf.py                      # 首次需要灌库
python rag_mcp_server.py                  # 终端 1
streamlit run app.py                      # 终端 2
```

## 常见问题

| 现象 | 排查方向 |
|---|---|
| `Couldn't connect to the MCP server after 30 seconds` | `MCP_SERVER_URL` 是否用了服务名而非 `127.0.0.1`；`mcp_server` 是否 healthy |
| `ModuleNotFoundError: litellm` | 镜像依赖缺失，重新 `docker compose build` |
| 回答"知识库中暂无相关信息" | `chroma_db` 是否挂载成功、是否有数据 |
| 容器连不上 Ollama | Ollama 是否在运行；`host.docker.internal` 是否可达 |

## License

仅供学习与内部使用。
