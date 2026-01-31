import grpc
from concurrent import futures
import rag_pb2
import rag_pb2_grpc
# rag_server.py
from rag.rag_chain import adaptive_rag_answer    

# 废弃，rag作为agent的tool集成在agent/mcp_server.py中
class RagServiceServicer(rag_pb2_grpc.RagServiceServicer):
    def GetRagAnswer(self, request, context):
        # 你可以把API key等放到配置文件或环境变量
        qwen_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        qwen_api_key = "sk-b9dc7ac8811d4a10b9ee1f084005053c"
        cohere_api_key = "your-cohere-api-key"
        answer = adaptive_rag_answer(
            request.user_query, qwen_api_base, qwen_api_key, cohere_api_key
        )
        return rag_pb2.RagResponse(answer=answer)

    def GetPatentInfo(self, request, context):
        # 查询专利信息
        patent_info = "专利信息内容"
        return rag_pb2.PatentInfoResponse(patent_info=patent_info)

    def GetEnterpriseInterest(self, request, context):
        # 查询企业兴趣度
        interest_level = "HIGH"
        return rag_pb2.EnterpriseInterestResponse(interest_level=interest_level)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rag_pb2_grpc.add_RagServiceServicer_to_server(RagServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()