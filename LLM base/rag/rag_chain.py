
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import RetrievalQA
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

# 2. 构建BE25混合检索器

def build_adaptive_retriever(multi_dbs, docs, query, top_k=5):
    """
    自适应融合BM25和多表征向量检索。
    可根据query类型动态调整权重或检索策略。
    """
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = top_k
    vector_retrievers = [db.as_retriever(search_kwargs={"k": top_k}) for db in multi_dbs.values()]
    # 这里简单平均融合，可按需自适应调整
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever] + vector_retrievers,
        weights=[0.3] + [0.7/len(vector_retrievers)]*len(vector_retrievers)
    )
    return retriever

# 3. 构建RAG链

def build_adaptive_rag_chain(qwen_api_base, qwen_api_key, cohere_api_key, persist_root="./chroma_db_multi", query="", top_k=5, rerank_top_n=5, use_cohere=False):
    multi_dbs = load_multi_chroma(persist_root)
    main_db = multi_dbs["bge-base-zh-v1.5"]
    docs = main_db.get()["documents"]
    retriever = build_adaptive_retriever(multi_dbs, docs, query, top_k=top_k)
    llm = OpenAI(
        openai_api_base=qwen_api_base,
        openai_api_key=qwen_api_key,
        model_name="qwen-turbo",
        temperature=0.2
    )

    def custom_retrieve(query):
        # 多路检索
        results_lists = []
        # BM25
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = top_k
        results_lists.append(bm25.get_relevant_documents(query))
        # 各向量表征
        for db in multi_dbs.values():
            retr = db.as_retriever(search_kwargs={"k": top_k})
            results_lists.append(retr.get_relevant_documents(query))
        # RRF融合去重
        fused = rrf_fusion(results_lists, k=60)
        # Cohere语义重排
        reranked = cohere_semantic_rerank(query, fused, cohere_api_key, top_n=rerank_top_n, use_cohere=use_cohere)
        return reranked

    class AdaptiveRetriever:
        def get_relevant_documents(self, query):
            return custom_retrieve(query)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=AdaptiveRetriever(),
        return_source_documents=True
    )
    return qa_chain

# rag/rag_chain.py
def adaptive_rag_answer(query, qwen_api_base, qwen_api_key, cohere_api_key, top_k=5, rerank_top_n=5):
    chain = build_adaptive_rag_chain(
        qwen_api_base, qwen_api_key, cohere_api_key, query=query, top_k=top_k, rerank_top_n=rerank_top_n
    )
    result = chain(query)
    return result["result"]


if __name__ == "__main__":
    # Qwen-API兼容OpenAI接口
    qwen_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 你的Qwen OpenAI兼容API地址
    qwen_api_key = "sk-b9dc7ac8811d4a10b9ee1f084005053c"
    cohere_api_key = "your-cohere-api-key"
    query = "这个专利的潜在应用场景是什么？"
    chain = build_adaptive_rag_chain(qwen_api_base, qwen_api_key, cohere_api_key, query=query)
    result = chain(query)
    print("答案：", result["result"])
    print("检索片段：", [doc.page_content for doc in result["source_documents"]])