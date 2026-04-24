import streamlit as st
from rag import RagService
import time
import config_data
from view_history import view_history,list_all_sessions

# 标题
st.title("问答系统")
st.divider()        # 分割符

if 'message' not in st.session_state:
    st.session_state['message'] = [
        {'role': 'assistant', 'content': '你好，我是你的智能助手，请问有什么可以帮助你的吗？'}
    ]

if 'rag' not in st.session_state:
    st.session_state['rag'] = RagService()

for message in st.session_state['message']:
    st.chat_message(message['role']).write(message['content'])

# 在页面最下方提供用户输入栏
prompt=st.chat_input()

if prompt:
    # 输出用户提问
    st.chat_message('user').write(prompt)
    st.session_state['message'].append(
        {'role': 'user', 'content': prompt}
    )

    ai_res_list=[]

    with st.spinner("正在思考..."):
        res_stream=st.session_state['rag'].chain.stream(
            {'input': prompt},
            config_data.session_config
        )
        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message('assistant').write_stream(capture(res_stream, ai_res_list))
        st.session_state['message'].append(
            {'role': 'assistant', 'content': ''.join(ai_res_list)}
        )

# 侧边栏：查看历史记录
with st.sidebar:
    st.header("📜 历史记录管理")

    # 显示当前会话ID
    current_session = config_data.session_config['configurable']['session_id']
    st.info(f"当前会话: `{current_session}`")

    # 按钮1：查看当前会话历史
    if st.button("📖 查看当前会话历史", use_container_width=True):
        history = view_history(current_session)
        if history:
            st.markdown("### 当前会话历史记录")
            for i, msg in enumerate(history, 1):
                if msg['role'] == 'user':
                    st.markdown(f"**{i}. 👤 用户:** {msg['content']}")
                elif msg['role'] == 'assistant':
                    st.markdown(f"**{i}. 🤖 AI:** {msg['content']}")
                st.divider()
        else:
            st.warning("暂无历史记录")

    # 按钮2：查看所有会话列表
    if st.button("📋 查看所有会话", use_container_width=True):
        sessions = list_all_sessions()
        if sessions:
            st.markdown("### 所有会话列表")
            for session in sessions:
                with st.expander(f"💬 {session['session_id']} ({session['message_count']}条消息)"):
                    st.write(f"- 消息数: {session['message_count']}")
                    st.write(f"- 文件大小: {session['file_size']} bytes")

                    # 可选：点击查看该会话详情
                    if st.button(f"查看此会话", key=f"view_{session['session_id']}"):
                        history = view_history(session['session_id'])
                        if history:
                            st.markdown("**对话内容:**")
                            for msg in history[-5:]:  # 只显示最近5条
                                icon = "👤" if msg['role'] == 'user' else "🤖"
                                st.markdown(f"{icon} **{msg['role']}:** {msg['content'][:100]}...")
        else:
            st.warning("暂无任何会话记录")

    # 按钮3：清空当前会话历史
    if st.button("🗑️ 清空当前会话", use_container_width=True, type="secondary"):
        from file_history_store import get_history
        chat_history = get_history(current_session)
        chat_history.clear()
        st.success("已清空当前会话历史")
        st.rerun()
