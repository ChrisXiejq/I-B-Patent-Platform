
# RAG-Patent-Innovation-Behaviour 项目总览

本项目是一个面向专利成果转化的智能平台，集成了前端可视化、SpringBoot中台服务、Agentic RAG智能模型层，支持专利检索、问答、企业画像、智能推理与多轮对话。

## 目录结构

- `frontend/`  —— 前端 Vue3 + Vite 项目，用户交互与可视化
- `backend/`   —— SpringBoot Java 中台，REST/gRPC服务、业务逻辑、与模型层对接
- `LLM base/`  —— Python 智能模型层，RAG、Agent、gRPC服务、MCP工具等
- `data exec/` —— 数据处理与分析脚本
- `env/`       —— Python依赖环境

## 技术架构

```
用户 <-> 前端(Vue3) <-> 中台(SpringBoot REST/gRPC) <-> 模型层(Python RAG/Agent) <-> 专利/企业/问卷/政策数据
```

### 前端
- Vue3 + Vite + Element Plus
- 多页面：专利检索、企业画像、问卷、新闻、团队等
- 国际化支持（en/zh/de）

### 中台
- SpringBoot，RESTful API + gRPC
- 业务聚合、权限、与Python模型层对接
- 支持Agent工具调用、专利/企业/问卷等多数据源

### 模型层（LLM base）
- Python 3.9+
- RAG（多表征向量检索+BM25+RRF+可选Cohere重排）
- Agent（支持CoT、ReAct推理、MCP工具集成、记忆管理）
- gRPC服务对接中台
- 支持Qwen/OpenAI等大模型

## 快速启动

### 1. 前端
```bash
cd frontend
npm install
npm run dev
# 访问 http://localhost:5173
```

### 2. 中台
```bash
cd backend
mvn spring-boot:run
# 默认端口 8190，REST/gRPC接口见Controller
```

### 3. 模型层（LLM base）
```bash
cd "LLM base"
conda env create -f env/environment.yml
pip install -r env/requirements.txt
# 数据入库
python rag/build_vector_db.py
# 启动MCP Server
python agent/mcp_server.py
# 启动Agent Server
python agent_server.py
```

## 主要功能

- 专利/企业/问卷/政策多源数据融合检索
- Agentic RAG：多表征+BM25混合检索+RRF融合+可选Cohere重排
- 智能Agent：支持CoT/Plan/ReAct推理、工具自动调用、记忆管理
- gRPC接口：中台与模型层高效通信
- 多轮对话、用户画像、企业兴趣度分析
- 可观测性：推理链、行动链、工具调用日志

## 进阶能力

- 支持MCP协议，便于多Agent/多工具协作
- 可扩展多模态（语音/图像）与多租户/多院区部署
- 代码结构清晰，便于二次开发与定制
