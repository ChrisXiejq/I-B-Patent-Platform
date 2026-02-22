"""
FastAPI 接口：独立启动 LLM/Agent 服务，无需 Java 后端
可用 Postman 等工具直接测试：POST /chat
"""
import os
import sys
from pathlib import Path

# 确保项目根在 path 中
_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# 复用 agent_server 的 Runtime
from agent_server import AgentRuntime

# ========== 启动 Agent Runtime ==========
MCP_SCRIPT = _root / "agent" / "mcp_server.py"
if not MCP_SCRIPT.exists():
    MCP_SCRIPT = "agent/mcp_server.py"

runtime = AgentRuntime(str(MCP_SCRIPT))

# ========== FastAPI ==========
app = FastAPI(
    title="专利成果转化 Agent API",
    description="直接调用 LLM/Agent，无需 Java 服务",
    version="1.0",
)


class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"
    mode: Optional[str] = "cot+react"  # cot | react | cot+react


class ChatResponse(BaseModel):
    answer: str
    query: str


@app.get("/health")
def health():
    """健康检查"""
    return {"status": "ok", "service": "patent-agent"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Agent 对话接口
    - query: 用户问题
    - user_id: 用户 ID，用于多用户记忆区分
    - mode: 推理模式 cot | react | cot+react
    """
    try:
        answer = runtime.process(
            req.query,
            user_id=req.user_id or "default_user",
            mode=req.mode or "cot+react",
            timeout=120.0,
        )
        return ChatResponse(answer=answer, query=req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/simple")
def chat_simple(query: str, user_id: Optional[str] = None, mode: Optional[str] = "cot+react"):
    """
    简化版：Query 参数
    POST /chat/simple?query=xxx&user_id=user1&mode=cot+react
    """
    try:
        answer = runtime.process(
            query,
            user_id=user_id or "default_user",
            mode=mode or "cot+react",
            timeout=120.0,
        )
        return {"answer": answer, "query": query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("AGENT_API_PORT", "8000"))
    print(f"Agent API: http://127.0.0.1:{port}")
    print(f"Postman 测试: POST http://127.0.0.1:{port}/chat")
    print('  Body JSON: {"query": "你好", "user_id": "test_user", "mode": "cot+react"}')
    uvicorn.run(app, host="0.0.0.0", port=port)
