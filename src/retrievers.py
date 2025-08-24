from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document, BaseRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.pydantic_v1 import PrivateAttr
from pydantic import model_validator
from typing import List

def build_bm25(chunks, k=3):
    r = BM25Retriever.from_documents(chunks); r.k = k
    return r

def build_chroma(chunks, k=3, embeddings=None):
    embeddings = embeddings or OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        collection_name="resume",
        embedding_function=embeddings,
        persist_directory="../db/chroma_langchain_db",
    )
    vector_store.add_documents(chunks)

    return vector_store.as_retriever(search_kwargs={"k": k})

class HyDERetriever(BaseRetriever):
    retriever: BaseRetriever
    llm: BaseChatModel
    
    _prompt = PrivateAttr()
    _hyde_chain = PrivateAttr()

    @model_validator(mode="after")
    def _initialize_hyde_chain(self):
        """Initializes the HyDE chain after the object is created."""
        template = """You are an AI assistant. Given a user's query, generate a detailed and comprehensive document that a user might be looking for to answer their question. The document should be rich in context and relevant information.
        Query: {query}
        Hypothetical Document:"""
        
        self._prompt = PromptTemplate.from_template(template)
        self._hyde_chain = (
            {"query": RunnablePassthrough()}
            | self._prompt
            | self.llm
        )
        return self

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            hypo_text = self._hyde_chain.invoke({"query": query}).content
        except Exception:
            hypo_text = query
        return self.retriever.get_relevant_documents(hypo_text)

def build_hyde(retriever, model):
    llm = ChatOllama(model=model, temperature=0)
    return HyDERetriever(retriever=retriever, llm=llm)

def build_ensemble(bm25, semantic, hyde):
    return EnsembleRetriever(
        retrievers=[bm25, semantic, hyde],
        weights=[0.35, 0.45, 0.20],
        search_type="rrf",
        c=60
    )