import os
from dotenv import load_dotenv

load_dotenv()

from crewai import LLM

# 两个作用：容器里读 compose 给的值；本机直跑时没有环境变量，回落到 127.0.0.1。一份代码两种环境都能跑，这就是用环境变量而不是硬编码的意义。
# 那句 load_dotenv() 默认 override=False，意思是已存在的真实环境变量优先，.env 只补充缺失的。所以 compose 的 environment 会赢过 .env——但前提是键名对得上。
# 只服务"本机直跑"这个场景（load_dotenv() 会读它）。
# 已加进 .dockerignore，不进镜像。容器里的配置 100% 来自 compose，单一来源，不会有两份配置打架
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/sse")
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_NAME=os.getenv("LLM_NAME")

from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter
from typing import Generator



def ask_agent(question: str) -> Generator[str, None, None]:
    """接受用户问题，调用 MCP 知识库，返回最终的回答文本"""

    # 1. 直接连接网络地址（不同版本的 CrewAI 传参可能不同，先试 url）
    # 注意： 有些版本可能是 server_url="http://127.0.0.1:8000/see"
    server_params = {
        "url": MCP_SERVER_URL,
        "transport": "sse"
    }

    yield "_THINKING_"
    try:
        # 2. 连接 TAG MCP Server
        with MCPServerAdapter(server_params) as mcp_tools:
            print(f"✅ 成功连接 RAG 服务器，发现工具：{[t.name for t in mcp_tools]}")

            local_llm = LLM(
                model=f"ollama_chat/{LLM_NAME}",
                base_url=OLLAMA_BASE_URL
            )

            # 3. 给 Agent 挂载工具
            worker = Agent(
                role="专属技术知识库问答助手",
                goal="根据真实的技术资料准确、严谨的回答用户问题",
                backstory="你是一个专业的技术专家，遇到不懂的问题，必须先用工具检索。",
                tools=mcp_tools,
                llm=local_llm,                   # 流式输出必须加_chat后缀
                verbose=True
            )

            task = Task(
                description=f"""
                用户问题：“{question}。”
                请你务必使用你的搜索工具，提取问题中的核心关键字，然后【仅根据】检索到的资料回答用户的问题。
                如果检索结果中没有相关信息，请直接回答“抱歉，我的本地知识库中暂无相关信息。”
                注意：资料中带有“（来源：《xxx》第x页）”的标记，请在你的回答末尾务必注明信息来源的文件名称和页码，格式如：*（参考：《xxx》第x页）
                如果你被问道关于你身份的问题，请直接根据你的角色设定回答，不需要查库。
                """,
                expected_output="一段准确、专业的文本回答。",
                agent=worker,
            )

            crew = Crew(
                agents=[worker],
                tasks=[task],
                process=Process.sequential,
                stream=True,                    # 开启流式输出
                verbose=True
            )

            #result = crew.kickoff()                # 正常输出
            # print(result)
            # return result.raw

            # 启动流式输出
            streaming_output = crew.kickoff()

            yield "_THINKING_"

            for chunk in streaming_output:
                # 1. 获取文本内容
                content = chunk.content
                # 2. 如果内容是空的，或者发现里面有{"query":这种废话，直接跳过}
                if not content or '{"query":' in content:
                    yield "_TOOL_RUNNING_"
                    continue
                yield content             # yild 生成器，必须用 for 循环才能读取，代码在app.py里

    except Exception as e:
        yield f"\n\n❌ 系统连接失败：{e}"
        # print(f"❌ 连接失败：{e}")
        # return f"❌ 连接失败：{e}"
