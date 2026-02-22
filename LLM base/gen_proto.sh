#!/bin/bash
# 重新生成 Python gRPC 代码（更新 proto 后需执行）
# 使用: pip install grpcio-tools && ./gen_proto.sh
cd "$(dirname "$0")"
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. rag.proto
echo "Generated rag_pb2.py, rag_pb2_grpc.py"
