
import os
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(pdf_dir):
    docs = []
    for fname in tqdm(os.listdir(pdf_dir)):
        if fname.lower().endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(pdf_dir, fname))
            docs.extend(loader.load())
    return docs


def build_multi_representation_indexes(pdf_dir, persist_root="./chroma_db_multi"):
    """
    对同一批文档，分别用多种embedding模型编码，分层存储。
    persist_root: 根目录，每种表征一个子目录
    """
    docs = load_pdfs(pdf_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    # 多表征模型，可按需扩展
    embedding_configs = [
        ("bge-base-zh-v1.5", "BAAI/bge-base-zh-v1.5"),
        ("text2vec-base-chinese", "GanymedeNil/text2vec-base-chinese"),
        ("e5-base", "intfloat/e5-base"),
    ]
    for tag, model_name in embedding_configs:
        print(f"正在处理embedding模型: {model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        persist_dir = os.path.join(persist_root, tag)
        vectordb = Chroma.from_documents(splits, embeddings, persist_directory=persist_dir)
        vectordb.persist()
        print(f"[{tag}] 入库完成，文档数：{len(docs)}，切分块数：{len(splits)}")
        
if __name__ == "__main__":
    # 假设你的PDF都在 ./patent_pdfs 目录
    build_multi_representation_indexes("./patent_pdfs")