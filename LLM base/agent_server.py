import grpc
from concurrent import futures
import asyncio
import threading

import rag_pb2
import rag_pb2_grpc
from agent.ib_agent import IBAgent


# =========================
# Async Agent Runtime
# =========================
class AgentRuntime:
    """
    独立线程 + 常驻 asyncio event loop
    所有 Agent / MCP 调用都在这里跑
    """
    def __init__(self, mcp_server_script: str):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_loop,
            daemon=True
        )
        self.agent = IBAgent(user_id="default_user")

        self.thread.start()

        # 在 event loop 线程中初始化 MCP 连接
        fut = asyncio.run_coroutine_threadsafe(
            self.agent.connect_to_server(mcp_server_script),
            self.loop
        )
        fut.result()  # 启动阶段允许阻塞

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def process(self, query: str, timeout: float = 60.0) -> str:
        """
        线程安全地提交 coroutine 给 asyncio loop
        """
        fut = asyncio.run_coroutine_threadsafe(
            self.agent.process_query(query),
            self.loop
        )
        return fut.result(timeout=timeout)


# =========================
# gRPC Service
# =========================
class AgentService(rag_pb2_grpc.AgentServiceServicer):

    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime

    def Chat(self, request, context):
        try:
            # answer = self.runtime.process(request.query)
            return rag_pb2.AgentResponse(answer="answer")

        except Exception as e:
            import traceback
            print("Agent exception:")
            traceback.print_exc()

            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Agent internal error")
            return rag_pb2.AgentResponse()



# =========================
# gRPC Server Bootstrap
# =========================
def serve(mcp_server_script: str):
    # 1️⃣ 启动 Agent Runtime（只一次）
    runtime = AgentRuntime(mcp_server_script)

    # 2️⃣ gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10)
    )

    rag_pb2_grpc.add_AgentServiceServicer_to_server(
        AgentService(runtime),
        server
    )

    server.add_insecure_port("[::]:50052")
    server.start()
    print("Agent gRPC server started on port 50052")

    server.wait_for_termination()


if __name__ == "__main__":
    mcp_server_script = "agent/mcp_server.py"
    serve(mcp_server_script)