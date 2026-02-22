"""
专利转化 Agent 的 MCP Server：对接 RAG、专利检索、企业兴趣等能力
工具通过 HTTP 调用 Spring Boot 业务中台 REST API
"""
from typing import Any
import httpx
from fastmcp import FastMCP

import sys
from pathlib import Path

# 确保项目根在 path 中
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import BACKEND_BASE_URL, QWEN_API_BASE, QWEN_API_KEY, COHERE_API_KEY, USE_COHERE_RERANK, RAG_PERSIST_ROOT
from rag.rag_chain import adaptive_rag_answer

# Initialize FastMCP server
mcp = FastMCP("patent")


async def _call_backend(path: str, method: str = "POST", params: dict = None, json_data: dict = None) -> dict:
    """调用 Spring Boot 业务中台 REST API"""
    url = f"{BACKEND_BASE_URL.rstrip('/')}{path}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            if method.upper() == "POST":
                resp = await client.post(url, params=params or {}, json=json_data)
            else:
                resp = await client.get(url, params=params or {})
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        except httpx.HTTPError as e:
            return {"error": str(e), "code": getattr(e, "response", None) and getattr(e.response, "status_code", 500)}


@mcp.tool()
async def get_identification(user_id: str) -> str:
    """获取用户身份类型（企业/高校/个人）.

    Args:
        user_id: 用户标识
    """
    # TODO: 对接后端 /agent/tools/identification 或用户服务
    return "enterprise"


@mcp.tool()
async def get_enterprise_interest(patent_no: str) -> str:
    """获取该专利的企业兴趣度（填写问卷的企业数量/热度）.

    Args:
        patent_no: 专利号
    """
    data = await _call_backend("/agent/tools/patent/enterprise", params={"patent_no": patent_no})
    if "error" in data:
        return f"查询失败: {data['error']}"
    if data.get("code") == 1 and "data" in data:
        return f"企业兴趣度: {data['data']}"
    return str(data)


@mcp.tool()
async def get_patent_analysis(patent_no: str) -> str:
    """根据专利号查询专利详情（名称、摘要、链接等）.

    Args:
        patent_no: 专利号
    """
    data = await _call_backend("/agent/tools/patent/search", params={"patent_no": patent_no})
    if "error" in data:
        return f"查询失败: {data['error']}"
    if data.get("code") == 1 and "data" in data:
        p = data["data"]
        parts = []
        if p.get("name"):
            parts.append(f"名称: {p['name']}")
        if p.get("summary"):
            parts.append(f"摘要: {p['summary']}")
        if p.get("link"):
            parts.append(f"链接: {p['link']}")
        return "\n".join(parts) if parts else str(p)
    return str(data)


@mcp.tool()
async def get_rag_patent_info(patent_no: str, query: str = "") -> str:
    """基于 RAG（向量+BM25 多路召回、RRF 融合、可选重排）获取专利相关知识增强回答.

    Args:
        patent_no: 专利号
        query: 用户查询（如不传则用专利号检索）
    """
    q = query.strip() or f"专利 {patent_no} 的相关信息、技术要点、应用场景"
    if not QWEN_API_KEY:
        return "RAG 未配置 QWEN_API_KEY，请在 .env 或环境变量中设置"
    try:
        insights = adaptive_rag_answer(
            q,
            qwen_api_base=QWEN_API_BASE,
            qwen_api_key=QWEN_API_KEY,
            cohere_api_key=COHERE_API_KEY,
            top_k=5,
            rerank_top_n=5,
            use_cohere=USE_COHERE_RERANK,
            persist_root=RAG_PERSIST_ROOT,
        )
        return f"专利 {patent_no} RAG 知识增强回答:\n{insights}"
    except Exception as e:
        return f"RAG 调用异常: {str(e)}"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
