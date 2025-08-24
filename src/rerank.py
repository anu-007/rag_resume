from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# FlashrankRerank.model_rebuild()

def build_compressed_retriever(base_retriever, top_n=8):
    reranker = FlashrankRerank(top_n=top_n, model="ms-marco-MiniLM-L-12-v2")
    compressor = DocumentCompressorPipeline(transformers=[reranker])
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )