import json
import os
from langchain_core.messages import messages_from_dict


def view_history(session_id, storage_path="./chat_history"):
    """
    查看指定会话的历史记录

    :param session_id: 会话ID
    :param storage_path: 历史记录存储路径
    :return: 格式化的历史消息列表
    """
    file_path = os.path.join(storage_path, session_id)

    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            messages_data = json.load(f)

        if not messages_data:
            return []

        history_list = []
        for msg_data in messages_data:
            role = msg_data.get('type', 'unknown')
            content = msg_data.get('data', {}).get('content', '')

            role_map = {
                'human': 'user',
                'ai': 'assistant',
                'system': 'system'
            }
            role_display = role_map.get(role, role)

            history_list.append({
                'role': role_display,
                'content': content
            })

        return history_list

    except Exception as e:
        return []


def list_all_sessions(storage_path="./chat_history"):
    """
    列出所有会话

    :param storage_path: 历史记录存储路径
    :return: 会话信息列表
    """
    if not os.path.exists(storage_path):
        return []

    sessions = os.listdir(storage_path)

    session_info = []
    for session in sessions:
        file_path = os.path.join(storage_path, session)
        file_size = os.path.getsize(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
            msg_count = len(messages_data)
        except:
            msg_count = 0

        session_info.append({
            'session_id': session,
            'message_count': msg_count,
            'file_size': file_size
        })

    return session_info


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "list":
            list_all_sessions()

        elif command == "view":
            if len(sys.argv) > 2:
                session_id = sys.argv[2]
                view_history(session_id)
            else:
                print("用法: python view_history.py view <session_id>")
                print("示例: python view_history.py view user_001")

        else:
            print("用法:")
            print("  python view_history.py list              # 列出所有会话")
            print("  python view_history.py view <session_id> # 查看指定会话")
    else:
        view_history("user_001")
        print("\n提示: 可以使用以下命令:")
        print("  python view_history.py list              # 列出所有会话")
        print("  python view_history.py view <session_id> # 查看指定会话")
