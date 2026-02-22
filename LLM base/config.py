"""
专利 Agent 平台配置：从环境变量读取 API Keys、后端地址等
"""
import os
from pathlib import Path

# 加载 .env（如存在）
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

# Qwen API（OpenAI 兼容）
QWEN_API_BASE = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "sk-b9dc7ac8811d4a10b9ee1f084005053c")

# Cohere Rerank（可选）
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
USE_COHERE_RERANK = os.getenv("USE_COHERE_RERANK", "false").lower() == "true"

# Spring Boot 业务中台地址（MCP 工具调用）
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8190")

# RAG 向量库路径
RAG_PERSIST_ROOT = os.getenv("RAG_PERSIST_ROOT", str(Path(__file__).parent / "chroma_db_multi"))
