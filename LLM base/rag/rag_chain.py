
import os
# LangChain 0.2+ 将 vectorstores/embeddings/retrievers 迁移到 langchain_community
# LangChain 新版本将模块迁移到 langchain_community，兼容多种版本
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.retrievers import BM25Retriever

from langchain_openai import OpenAI
import requests
from collections import defaultdict

def rrf_fusion(results_lists, k=60):
    """
    Reciprocal Rank Fusion (RRF) 融合多路检索结果并去重。
    results_lists: List[List[Document]]
    返回融合去重后的Document列表，按RRF分数降序
    """
    scores = defaultdict(float)
    doc_map = {}
    for result in results_lists:
        for rank, doc in enumerate(result):
            doc_id = getattr(doc, 'page_content', str(doc))[:128]  # 简单用内容hash做唯一标识
            scores[doc_id] += 1.0 / (rank + 60)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
    # 按分数降序
    sorted_docs = sorted(scores.items(), key=lambda x: -x[1])
    return [doc_map[doc_id] for doc_id, _ in sorted_docs]

def cohere_semantic_rerank(query, docs, cohere_api_key, top_n=5, use_cohere=False):
    """
    用Cohere模型对检索结果做语义感知重排序。
    use_cohere: False时直接返回RRF融合结果的前top_n
    """
    if not use_cohere:
        return docs[:top_n]
    url = "https://api.cohere.ai/v1/rerank"
    headers = {"Authorization": f"Bearer {cohere_api_key}", "Content-Type": "application/json"}
    passages = [getattr(doc, 'page_content', str(doc)) for doc in docs]
    data = {"query": query, "documents": passages, "top_n": top_n, "model": "rerank-english-v2.0"}
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    indices = [r['index'] for r in resp.json()['results']]
    return [docs[i] for i in indices]

# 1. 加载Chroma向量库

def load_multi_chroma(persist_root="./chroma_db_multi"):
    """
    加载多表征Chroma索引，返回dict: tag->vectordb
    """
    configs = [
        ("bge-base-zh-v1.5", "BAAI/bge-base-zh-v1.5"),
        ("text2vec-base-chinese", "GanymedeNil/text2vec-base-chinese"),
        ("e5-base", "intfloat/e5-base"),
    ]
    dbs = {}
    for tag, model_name in configs:
        persist_dir = os.path.join(persist_root, tag)
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        dbs[tag] = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return dbs

# 2. 构建 BM25+向量 混合检索器（使用 rrf_fusion，不依赖 EnsembleRetriever）

def build_adaptive_retriever(multi_dbs, docs, query, top_k=5):
    """
    自适应融合 BM25 和多表征向量检索，使用 RRF 融合。
    不依赖 EnsembleRetriever，兼容不同 LangChain 版本。
    """
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = top_k
    retrievers = [bm25_retriever] + [db.as_retriever(search_kwargs={"k": top_k}) for db in multi_dbs.values()]

    class RRFEnsembleRetriever:
        def get_relevant_documents(self, q):
            results_lists = [r.get_relevant_documents(q) if hasattr(r, 'get_relevant_documents') else r.invoke(q) for r in retrievers]
            return rrf_fusion(results_lists, k=60)

    return RRFEnsembleRetriever()

# 3. RAG 检索+生成（不依赖 RetrievalQA，兼容各版本 LangChain）

def _custom_retrieve(multi_dbs, docs, query, top_k, rerank_top_n, cohere_api_key, use_cohere):
    """多路检索：BM25 + 向量，RRF 融合，可选 Cohere 重排"""
    results_lists = []
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = top_k
    results_lists.append(bm25.get_relevant_documents(query))
    for db in multi_dbs.values():
        retr = db.as_retriever(search_kwargs={"k": top_k})
        results_lists.append(retr.get_relevant_documents(query))
    fused = rrf_fusion(results_lists, k=60)
    return cohere_semantic_rerank(query, fused, cohere_api_key, top_n=rerank_top_n, use_cohere=use_cohere)


def build_adaptive_rag_chain(qwen_api_base, qwen_api_key, cohere_api_key, persist_root="./chroma_db_multi", query="", top_k=5, rerank_top_n=5, use_cohere=False):
    """构建 RAG 链（ manual 实现，无 RetrievalQA 依赖）"""
    multi_dbs = load_multi_chroma(persist_root)
    main_db = multi_dbs["bge-base-zh-v1.5"]
    docs = main_db.get()["documents"]
    llm = OpenAI(
        openai_api_base=qwen_api_base,
        openai_api_key=qwen_api_key,
        model_name="qwen-turbo",
        temperature=0.2
    )

    def run_rag(q: str):
        retrieved = _custom_retrieve(multi_dbs, docs, q, top_k, rerank_top_n, cohere_api_key, use_cohere)
        context = "\n\n".join(getattr(d, "page_content", str(d)) for d in retrieved)
        prompt = f"""基于以下参考内容回答问题。如果参考内容中没有相关信息，请基于常识回答。

参考内容：
{context}

问题：{q}

请给出简洁准确的回答："""
        return llm.invoke(prompt), retrieved

    return run_rag


def adaptive_rag_answer(query, qwen_api_base, qwen_api_key, cohere_api_key, top_k=5, rerank_top_n=5,
                       use_cohere=False, persist_root="./chroma_db_multi"):
    run_rag = build_adaptive_rag_chain(
        qwen_api_base, qwen_api_key, cohere_api_key,
        persist_root=persist_root, query=query, top_k=top_k, rerank_top_n=rerank_top_n, use_cohere=use_cohere
    )
    ans, _ = run_rag(query)
    return ans.content if hasattr(ans, "content") else str(ans)


if __name__ == "__main__":
    qwen_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_api_key = "sk-xxx"
    cohere_api_key = "your-cohere-api-key"
    query = "这个专利的潜在应用场景是什么？"
    run_rag = build_adaptive_rag_chain(qwen_api_base, qwen_api_key, cohere_api_key, query=query)
    ans, source_docs = run_rag(query)
    print("答案：", ans.content if hasattr(ans, "content") else ans)
    print("检索片段：", [getattr(d, "page_content", str(d)) for d in source_docs])