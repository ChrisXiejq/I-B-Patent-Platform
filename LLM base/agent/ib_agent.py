import asyncio
import sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

## OpenAI 的包
from openai import OpenAI
from dotenv import load_dotenv
from agent.memory import memory_store

load_dotenv()  # load environment variables from .env

try:
    from config import QWEN_API_KEY, QWEN_API_BASE
except ImportError:
    import os
    QWEN_API_KEY = os.getenv("QWEN_API_KEY", "sk-b9dc7ac8811d4a10b9ee1f084005053c")
    QWEN_API_BASE = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")


class IBAgent:
    def __init__(self, user_id: str):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.user_id = user_id
        self.session_id = f"session_{user_id}"
        self.mcp_session = None

        # 替换为 Qwen (OpenAI-compatible) client，从 .env/config 读取
        self.llm = OpenAI(
            api_key=QWEN_API_KEY or "put-your-qwen-api-key-here",
            base_url=QWEN_API_BASE
        )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def cot_plan_and_reason(self, query: str, history, profile, available_tools):
        """
        Chain-of-Thought（CoT）推理与规划：
        1. 先让LLM输出思考链和行动计划（如任务分解、工具选择、推理步骤）。
        2. 再按计划逐步执行，每步可调用工具或继续LLM。
        """
        # 1. 让LLM输出思考链和行动计划
        cot_prompt = (
            "你是一个智能Agent，请根据用户问题，结合历史对话和用户画像，"
            "先输出详细的思考链（Chain-of-Thought），再给出行动计划（如需要哪些工具/步骤/推理）。"
            "最后用JSON格式输出：{\"thoughts\":..., \"plan\":...}"
        )
        cot_messages = [
            {"role": "system", "content": cot_prompt},
        ]
        for _, role, text in history:
            r = "assistant" if role == "agent" else role  # API 仅接受 assistant 而非 agent
            if r in ("system", "assistant", "user"):
                cot_messages.append({"role": r, "content": text})
        cot_messages.append({"role": "user", "content": query})
        cot_response = self.llm.chat.completions.create(
            model="qwen-plus",
            messages=cot_messages,
            max_tokens=800
        )
        import json
        try:
            cot_json = json.loads(cot_response.choices[0].message.content)
            thoughts = cot_json.get("thoughts", "")
            plan = cot_json.get("plan", "")
        except Exception:
            thoughts = cot_response.choices[0].message.content
            plan = ""
        # 2. 可选：根据plan自动执行工具/多轮推理（这里只做演示，实际可按plan分步执行）
        return thoughts, plan

    async def react_reasoning(self, query: str, history, profile, available_tools):
        """
        ReAct（Reason+Act）推理：
        1. 让LLM以“思考-行动-观察”格式输出推理和行动。
        2. 自动解析并执行行动（如工具调用），再将观察结果反馈给LLM，循环多轮。
        """
        react_prompt = (
            "你是一个智能Agent，请用如下格式进行推理和行动：\n"
            "Thought: ...\nAction: <工具名>(参数)\nObservation: ...\n"
            "每轮先思考(Thought)，再决定是否调用工具(Action)，然后根据Observation继续推理，直到完成任务。"
        )
        messages = [
            {"role": "system", "content": react_prompt},
        ]
        for _, role, text in history:
            r = "assistant" if role == "agent" else role  # API 仅接受 assistant 而非 agent
            if r in ("system", "assistant", "user"):
                messages.append({"role": r, "content": text})
        messages.append({"role": "user", "content": query})
        max_steps = 3
        for step in range(max_steps):
            llm_response = self.llm.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                tools=available_tools,
                tool_choice="auto",
                max_tokens=800
            )
            content = llm_response.choices[0].message.content
            messages.append({"role": "assistant", "content": content})
            # 简单解析Action
            import re
            action_match = re.search(r"Action: (\w+)\\((.*?)\\)", content)
            if action_match:
                tool_name = action_match.group(1)
                tool_args = action_match.group(2)
                try:
                    import ast
                    tool_args_dict = ast.literal_eval(tool_args) if tool_args else {}
                except Exception:
                    tool_args_dict = {}
                result = await self.session.call_tool(tool_name, tool_args_dict)
                obs = f"Observation: {result.content}"
                messages.append({"role": "user", "content": obs})
            else:
                break  # 没有Action则结束
        # 返回最终推理链
        return '\n'.join([m['content'] for m in messages if m['role'] == 'assistant'])

    async def process_query(self, query: str, mode: str = "cot+react", user_id: str = None) -> str:
        uid = user_id or self.user_id
        session_id = f"session_{uid}"
        # 1. 记录用户输入到短期记忆
        memory_store.add_short_term(session_id, 'user', query)
        # 2. 通过标准化 API 获取分层记忆上下文
        ctx = memory_store.get_context_for_agent(uid, session_id, query, short_term_limit=10, long_term_top_k=3)
        history = ctx["history"]
        profile = ctx["profile"]
        relevant_memories = ctx["relevant_long_term"]
        # 3. 将长期记忆拼接到 prompt
        if relevant_memories:
            history = list(history)  # 副本，避免污染
            history.append((None, "system", f"长期记忆召回：{relevant_memories}"))
        # 注入当前 user_id，供 LLM 调用工具时使用（如 get_identification(user_id)）
        history = list(history)
        history.append((None, "system", f"【会话上下文】当前用户 user_id={uid}，调用需要 user_id 的工具时请使用此值。"))
        # 4. 获取可用工具
        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in response.tools
        ]
        # 5. 串联CoT+ReAct（URL 中 + 会变为空格，需兼容）
        mode = (mode or "").replace(" ", "+").lower()
        if mode == "cot+react":
            thoughts, plan = await self.cot_plan_and_reason(query, history, profile, available_tools)
            # 将plan作为目标，传递给ReAct
            react_intro = f"请根据以下行动计划逐步完成任务：{plan}"
            react_history = history + [(None, "system", react_intro)]
            react_trace = await self.react_reasoning(query, react_history, profile, available_tools)
            memory_store.add_short_term(session_id, 'agent', f"[推理链]\n{thoughts}\n[计划]\n{plan}\n[ReAct]\n{react_trace}")
            return f"[推理链]\n{thoughts}\n[计划]\n{plan}\n[ReAct]\n{react_trace}"
        elif mode == "cot":
            thoughts, plan = await self.cot_plan_and_reason(query, history, profile, available_tools)
            memory_store.add_short_term(session_id, 'agent', f"[推理链]\n{thoughts}\n[计划]\n{plan}")
            return f"[推理链]\n{thoughts}\n[计划]\n{plan}"
        elif mode == "react":
            react_trace = await self.react_reasoning(query, history, profile, available_tools)
            memory_store.add_short_term(session_id, 'agent', f"[ReAct]\n{react_trace}")
            return f"[ReAct]\n{react_trace}"
        else:
            return "未知推理模式"

    def update_profile(self, key, value):
        memory_store.add_long_term(self.user_id, key, value)

    def clear_memory(self):
        memory_store.clear_short_term(self.session_id)
        memory_store.clear_long_term(self.user_id)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = IBAgent(user_id="default_user")
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())