# Patent Intelligence Transformation Agent Platform

> Tongji University & Max Planck Institute Collaborative Project

A patent open platform integrating patent research, intelligent query, RAG knowledge enhancement, and Agent multi-turn reasoning. It serves enterprises, universities, and individuals with patent transformation services, addressing challenges such as high patent technology comprehension costs and difficult patent transfer decisions.

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | Vue 3, Vite, Element Plus, Vue Router, Pinia, i18n |
| **Backend** | Spring Boot 3, RESTful API, gRPC, MySQL, Redis, JWT |
| **Model Layer** | Python, LangChain, ChromaDB, Agentic RAG, MCP, FastMCP |
| **LLM** | Qwen (Alibaba Cloud DashScope, OpenAI-compatible API) |

---

## Project Structure

```
I-B-Patent-Platform/
├── frontend/           # Vue 3 frontend
├── backend/            # Spring Boot backend (REST + gRPC)
├── LLM base/           # Python model layer
│   ├── agent/          # Agent, memory, MCP Server
│   ├── rag/            # RAG retrieval chain, vector DB build
│   ├── agent_api.py    # FastAPI (standalone testing)
│   ├── agent_server.py # gRPC service (backend integration)
│   └── config.py       # Configuration
├── .gitignore
└── README.md
```

---

## Prerequisites

- **Node.js** 18+
- **Java** 17+
- **Python** 3.10+
- **MySQL**, **Redis** (for backend)
- **Qwen API Key** (Alibaba Cloud DashScope)

---

## Quick Start

### 1. Configure LLM Environment

Create a `.env` file under `LLM base/`:

```env
# Required: Qwen API Key (Alibaba Cloud DashScope)
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=sk-your-qwen-api-key

# Optional: Cohere semantic rerank
COHERE_API_KEY=
USE_COHERE_RERANK=false

# Spring Boot backend URL (for MCP tool calls)
BACKEND_BASE_URL=http://localhost:8190
```

### 2. Option A: Full Stack (Frontend + Backend + Model Layer)

```bash
# 1. Start backend (configure MySQL, Redis first)
cd backend
mvn spring-boot:run
# Port 8190

# 2. Start model layer gRPC service
cd "LLM base"
pip install -r env/requirements.txt
python agent_server.py
# gRPC port 50052

# 3. Start frontend
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### 3. Option B: LLM/Agent Only (No Java Required)

```bash
cd "LLM base"
pip install -r env/requirements.txt
python agent_api.py
```

After startup:

- **API docs**: http://localhost:8000/docs
- **Health check**: `GET http://localhost:8000/health`
- **Chat API**: `POST http://localhost:8000/chat` or `/chat/simple`

**Postman example**:
```
POST http://localhost:8000/chat/simple?query=hello&user_id=test&mode=react
```
Or JSON body:
```json
{"query": "What is my user identity?", "user_id": "111000", "mode": "cot+react"}
```

---

## Features

- **Patent retrieval**: Vector + BM25 multi-retrieval, RRF fusion, optional Cohere rerank
- **Agent reasoning**: CoT, ReAct, CoT+ReAct modes with tool auto-invocation
- **MCP tools**: `get_identification`, `get_patent_analysis`, `get_enterprise_interest`, `get_rag_patent_info`
- **Hierarchical memory**: Short-term cache + long-term vectorization (episodic/semantic), multi-user support
- **gRPC**: Decoupled backend and Python model layer for independent iteration

---

## Build Vector DB (Optional)

Build ChromaDB before using RAG:

```bash
cd "LLM base"
python rag/build_vector_db.py
```

Output goes to `chroma_db_multi/`. Add to `.gitignore` if the directory is large.

---

## License

This is a collaborative project. Please comply with the relevant agreements when using it.
