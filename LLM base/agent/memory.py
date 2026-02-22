"""
分层记忆架构：短期缓存 + 长期向量化
- 短期记忆：最近 N 轮对话，内存缓存，快速访问
- 工作记忆：当前任务上下文（可选），可压缩入长期
- 长期记忆：ChromaDB 向量化
  - episodic：对话/事件摘要、业务数据（专利/问卷）摘要
  - semantic：用户画像、知识、偏好
对外提供标准化 API，Agent 在多轮对话中利用历史上下文。
"""
import time
import threading
import chromadb
from sentence_transformers import SentenceTransformer
import os
from typing import Optional, List, Tuple, Any


class MemoryStore:
    """
    分层记忆：短期缓存（内存） + 长期向量化（ChromaDB）
    支持用户多轮对话和业务数据（专利/问卷）的持久化与检索。
    """

    # 长期记忆类型
    LT_EPISODIC = "episodic"  # 对话/事件/业务数据摘要
    LT_SEMANTIC = "semantic"  # 用户画像、知识、偏好

    def __init__(self, persist_dir=None, embedding_model_name=None, short_term_size=20):
        self.short_term = {}  # {session_id: [(timestamp, role, text)]}
        self.working = {}    # {session_id: [(timestamp, key, value)]} 工作记忆
        self.lock = threading.Lock()
        self.short_term_size = short_term_size

        persist_dir = persist_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../chroma_db_multi/user_long_term')
        )
        os.makedirs(persist_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            "long_term_memory",
            metadata={"description": "episodic + semantic 长期记忆"}
        )
        self.embedding_model = SentenceTransformer(
            embedding_model_name or "paraphrase-multilingual-MiniLM-L12-v2"
        )

    # ========== 短期记忆（缓存最近对话） ==========

    def add_short_term(self, session_id: str, role: str, text: str) -> None:
        """写入短期记忆（对话轮次）"""
        with self.lock:
            self.short_term.setdefault(session_id, []).append((time.time(), role, text))
            self.short_term[session_id] = self.short_term[session_id][-self.short_term_size:]

    def get_short_term(self, session_id: str) -> List[Tuple[float, str, str]]:
        """获取短期记忆（对话历史）"""
        with self.lock:
            return self.short_term.get(session_id, []).copy()

    def clear_short_term(self, session_id: str) -> None:
        with self.lock:
            self.short_term.pop(session_id, None)

    # ========== 工作记忆（当前任务上下文） ==========

    def add_working(self, session_id: str, key: str, value: Any) -> None:
        """写入工作记忆（如当前任务计划、工具调用结果）"""
        with self.lock:
            self.working.setdefault(session_id, []).append((time.time(), key, str(value)))
            self.working[session_id] = self.working[session_id][-10:]  # 最多保留 10 条

    def get_working(self, session_id: str) -> List[Tuple[float, str, str]]:
        with self.lock:
            return self.working.get(session_id, []).copy()

    def clear_working(self, session_id: str) -> None:
        with self.lock:
            self.working.pop(session_id, None)

    # ========== 长期记忆（向量化） ==========

    def add_long_term(
        self, user_id: str, key: str, value: Any,
        memory_type: str = LT_SEMANTIC
    ) -> None:
        """
        将长期记忆向量化存入 ChromaDB。
        memory_type: episodic（事件/业务数据）或 semantic（画像/知识）
        """
        doc_id = f"{user_id}:{memory_type}:{key}"
        text = str(value)
        embedding = self.embedding_model.encode([text])[0].tolist()
        with self.lock:
            try:
                self.collection.delete(ids=[doc_id])
            except Exception:
                pass
            self.collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[{"user_id": user_id, "key": key, "type": memory_type}],
                embeddings=[embedding]
            )

    def add_business_data(
        self, user_id: str, data_type: str, data: Any
    ) -> None:
        """
        写入业务数据摘要（专利、问卷等）到长期 episodic 记忆
        """
        self.add_long_term(user_id, f"business_{data_type}", data, memory_type=self.LT_EPISODIC)

    def get_long_term(
        self, user_id: str,
        key: Optional[str] = None,
        query_text: Optional[str] = None,
        top_k: int = 3,
        memory_type: Optional[str] = None
    ):
        """
        - key: 精确查找
        - query_text: 语义检索
        - memory_type: 限定 episodic 或 semantic，None 表示全部
        """
        # ChromaDB 新版 where 语法：需使用 $eq，多条件用 $and
        def _where():
            conds = [{"user_id": {"$eq": user_id}}]
            if memory_type:
                conds.append({"type": {"$eq": memory_type}})
            return {"$and": conds} if len(conds) > 1 else conds[0]

        where = _where()

        if key:
            # 尝试新格式 user_id:type:key 与旧格式 user_id:key
            for doc_id in [
                f"{user_id}:{memory_type or self.LT_SEMANTIC}:{key}",
                f"{user_id}:{self.LT_SEMANTIC}:{key}",
                f"{user_id}:{self.LT_EPISODIC}:{key}",
                f"{user_id}:{key}",
            ]:
                try:
                    results = self.collection.get(ids=[doc_id])
                    if results and results.get("documents"):
                        return results["documents"][0]
                except Exception:
                    continue
            return None

        if query_text:
            embedding = self.embedding_model.encode([query_text])[0].tolist()
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where if where else None
            )
            docs = results.get("documents", [[]])
            return docs[0] if docs else []

        # 返回该用户全部
        results = self.collection.get(where=where)
        return results.get("documents", []) if results else []

    def clear_long_term(self, user_id: str) -> None:
        results = self.collection.get(where={"user_id": {"$eq": user_id}})
        ids = results.get("ids", []) if results else []
        if ids:
            self.collection.delete(ids=ids)

    # ========== 标准化 API：供 Agent 获取上下文 ==========

    def get_context_for_agent(
        self,
        user_id: str,
        session_id: str,
        query: str,
        short_term_limit: int = 10,
        long_term_top_k: int = 3,
    ) -> dict:
        """
        获取 Agent 可用的记忆上下文，返回结构化数据。
        用于拼接到 prompt 中，提升多轮对话一致性和上下文利用。
        """
        history = self.get_short_term(session_id)[-short_term_limit:]
        profile = self.get_long_term(user_id, memory_type=self.LT_SEMANTIC)
        relevant_memories = self.get_long_term(
            user_id, query_text=query, top_k=long_term_top_k, memory_type=self.LT_EPISODIC
        )
        if not relevant_memories and query:
            # 未指定 type 时也做一次检索
            relevant_memories = self.get_long_term(user_id, query_text=query, top_k=long_term_top_k)
        working = self.get_working(session_id)

        return {
            "history": history,
            "profile": profile,
            "relevant_long_term": relevant_memories or [],
            "working": working,
        }

    def format_context_for_prompt(self, context: dict) -> str:
        """将 context 格式化为可拼接到 prompt 的字符串"""
        parts = []
        if context.get("relevant_long_term"):
            parts.append("【长期记忆召回】\n" + "\n".join(f"- {m}" for m in context["relevant_long_term"]))
        if context.get("profile") and isinstance(context["profile"], list):
            if len(context["profile"]) > 0:
                parts.append("【用户画像/知识】\n" + "\n".join(f"- {p}" for p in context["profile"]))
        elif context.get("profile") and isinstance(context["profile"], str):
            parts.append("【用户画像】\n" + context["profile"])
        if context.get("working"):
            parts.append("【当前任务】\n" + "\n".join(f"- {k}: {v}" for _, k, v in context["working"]))
        if not parts:
            return ""
        return "\n\n".join(parts) + "\n\n"


# 单例
memory_store = MemoryStore()
