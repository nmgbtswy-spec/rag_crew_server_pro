# 1. 直接用最标准的python镜像，是起点，所有层都叠在它上面。用这个镜像是因为它已经带了 uv 和 Python 3.12，省掉自己装 Python 的麻烦
# bookworm-slim 是 Debian 12 精简版，体积小；代价是没有 curl、没有编译器。所以后面别写需要 gcc 的东西。
# 镜像里自带 pip，且 Python 是"系统级"的 /usr/local/bin/python——这点决定了后面为什么用 --system
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
# 2.设定后续所有指令的工作目录，不存在会自动创建。
# 它的作用是让 COPY、RUN、以及运行时的相对路径都落在 /app。你的代码里 chromadb.PersistentClient(path='./chroma_db') 是相对路径
# 最终就指向 /app/chroma_db——这是后面挂载点必须写 /app/chroma_db 的原因。
WORKDIR /app
# 3. 配置国内清华源，防止下载 nvidia 那些几个 G 的包时卡死
# PIP_INDEX_URL / UV_INDEX_URL：换清华源。注意 pip 和 uv 读的是不同变量名，只设一个另一个照样走官方源，在国内会卡很久甚至超时。
# PYTHONUNBUFFERED=1：Python 默认对非终端输出做缓冲，容器里 print 会攒着不输出，docker compose logs 就是空的。加了它日志实时可见——排查容器问题全靠日志，这个变量不是可选项。
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple\
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple\
    PYTHONUNBUFFERED=1

# 4. 安装 uv
# 这四行是整个文件的核心，逐条拆：
# pip install uv：基础镜像里其实已有 uv，这行是兜底（老版本镜像没有）。代价很小，留着无害。
# COPY pyproject.toml uv.lock ./：只先拷清单，不拷代码。这是分层的关键——依赖清单很少变，代码天天变。这样改代码时，下面那句巨大的 uv pip install 会命中缓存，构建从几分钟变成几秒。
# RUN uv pip install --system -r pyproject.toml：
# --system = 装进镜像的系统 Python，不建虚拟环境。这样运行时 python rag_mcp_server.py 就能直接找到依赖，compose 里不用写 uv run。
# -r pyproject.toml = 按项目声明的依赖装。注意它不读 uv.lock，只认 >= 下限，所以每次构建解析出的版本可能略有不同——这正是"本机能跑、镜像里缺 litellm"的根因。
# RUN uv pip install --system "litellm==1.74.9"：补漏。crewai-tools 的 RAG 模块在被 import 的那一刻就 import litellm，但它的包元数据里没把 litellm 声明成必需依赖，于是按清单装完就是缺的。
# 为什么钉 1.74.9？因为你本机 .venv 里就是这个版本（来自 uv.lock 的真依赖集合）。让镜像对齐本机已验证的组合，比"装最新"可靠得多。
# 为什么单独一行不合并进上一行？合并会改变上一条 RUN 的指令文本，导致整层缓存失效、几个 G 重下。分成两条，只有这一小条会跑。
# COPY . .：把项目（除 .dockerignore 排除的）拷进来。放最后是因为它是"最常变"的层。
RUN pip install uv
# 5. 复制依赖文件
COPY pyproject.toml uv.lock ./
# 6. 核心：用uv pip 直接安装依赖，绝对不检查项目结构
RUN uv pip install --system -r pyproject.toml
# 7. 安装 litellm 依赖
RUN uv pip install --system "litellm==1.74.9"
# 7. 复制所有的代码和文件进去
COPY . .

# 纯文档性质：EXPOSE 不会真的开放端口，也不影响任何行为。真正的端口映射在 compose 的 ports: 里。写它是为了让人知道这个镜像打算用哪些端口。
EXPOSE 8000
EXPOSE 8501