# 前端
from cProfile import label

import streamlit as st
from agent_core import ask_agent

# 设置网页标题
st.set_page_config(page_title="本地私有化 AI 知识库",page_icon="🤖",layout="centered")
st.title("🤖 本地私有化 Agentic RAG")
st.caption("基于 CrewAI + MCP + ChromaDB + 本地千问大模型")

# 初始化聊天历史记录
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 渲染历史聊天记录，提取历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接受用户输入
if prompt := st.chat_input("请输入你想查的技术问题：（例如：CrewAI 是什么？）..."):

    # 1. 把用户的问题显示在界面上
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ====================== 旧代码：非流式输出 ======================
    # 2. 呼叫核心大脑，并显示“思考中”的动画
    # with st.chat_message("assistant"):
    #     with st.spinner("Agent 正在查阅本地知识库并思考，本地模型较慢请耐心等待..."):
    #
    #         # 核心逻辑，完全解耦
    #         answer = ask_agent(str(prompt))
    #
    #         # 把回答显示在界面上
    #         st.markdown(answer)
    #         st.session_state.messages.append({"role":"assistant","content":answer})
    # ==============================================================

    # 3. 流式输出
    with st.chat_message("assistant"):
        # 创建一个空的占位符，用来放打字机效果
        message_placeholder = st.empty()
        full_response = ""

        # 获取流式生成器
        stream_generator = ask_agent(str(prompt))           # prompt 提示词，ask_agent()是一个流式生成器，方法的返回值为 Generator

        # 用 Streamlit 的 st.status 作为一个动态状态框
        with st.status("Agent 正在思考，请稍后...",expanded=False) as status:
            # 实时迭代并更新网页
            for chunk in stream_generator:          # 这里 for 循环不断拉取 生成器里的内容
                if chunk == "_THINKING_":
                    # 收到暗号保持转圈不懂，不输出文字
                    status.update(label="Agent 正在查阅知识库...", state="running")
                    continue
                if chunk == "_TOOL_RUNNING_":
                    status.update(label="Agent 正在执行检索或分析工具...", state ="running")
                    continue
                if not full_response:
                    # 如果之前美属楚国，这是第一次真话，改变状态框
                    status.update(label="Agent 正在为您生成回答...", state="running")
                full_response += chunk
                # 每次收到一个字，旧更新一次占位符，加个闪烁的光标
                message_placeholder.markdown(full_response + "✏️")

            # 循环结束，关闭状态框，显示已完成
            status.update(label="回答完成",state="complete", expanded=False)

        message_placeholder.markdown(full_response)
            # 保存历史数据
        st.session_state.messages.append({"role":"assistant","content":full_response})