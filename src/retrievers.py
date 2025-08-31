from langchain_community.retrievers import BM25Retriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever

def build_bm25(chunks, k=3):
    r = BM25Retriever.from_documents(chunks); r.k = k
    return r

def build_chroma(chunks, k=3, embeddings=None):
    embeddings = embeddings or HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vector_store = Chroma(
        collection_name="resume",
        embedding_function=embeddings,
        persist_directory="../db/chroma_langchain_db",
    )
    vector_store.add_documents(chunks)

    return vector_store.as_retriever(search_kwargs={"k": k})

def build_ensemble(bm25, semantic):
    return EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[0.30, 0.70],
        search_type="rrf",
        c=60
    )