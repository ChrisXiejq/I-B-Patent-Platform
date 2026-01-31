import time
import threading

class MemoryStore:
    """
    支持短期记忆（对话历史）和长期记忆（业务数据/事件/用户画像）的本地内存实现。
    可扩展为持久化（如Redis/DB）。
    """
    def __init__(self):
        self.short_term = {}  # {session_id: [(timestamp, user, text)]}
        self.long_term = {}   # {user_id: {key: value}}
        self.lock = threading.Lock()

    def add_short_term(self, session_id, role, text):
        with self.lock:
            self.short_term.setdefault(session_id, []).append((time.time(), role, text))
            # 只保留最近N轮
            self.short_term[session_id] = self.short_term[session_id][-20:]

    def get_short_term(self, session_id):
        with self.lock:
            return self.short_term.get(session_id, [])

    def clear_short_term(self, session_id):
        with self.lock:
            self.short_term.pop(session_id, None)

    def add_long_term(self, user_id, key, value):
        with self.lock:
            self.long_term.setdefault(user_id, {})[key] = value

    def get_long_term(self, user_id, key=None):
        with self.lock:
            if key:
                return self.long_term.get(user_id, {}).get(key)
            return self.long_term.get(user_id, {})

    def clear_long_term(self, user_id):
        with self.lock:
            self.long_term.pop(user_id, None)

# 单例
memory_store = MemoryStore()
