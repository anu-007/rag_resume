from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from .loader import load_local
from .splitter import chunk
from .retrievers import build_bm25, build_chroma, build_ensemble
from .rerank import build_compressed_retriever
from .prompt import ATS_PROMPT

def make_context(docs):
    context = "\n\n".join(doc.page_content for doc in docs)
    return context

def build_pipeline(jd_doc: Document, model="gpt-3.5-turbo", retriever_type="Ensemble", k=12, use_reranker=True, top_n=8):
    resumes = chunk(load_local())

    all_chunks = resumes

    if retriever_type == "Ensemble":
        semantic = build_chroma(all_chunks, k=k)
        bm25 = build_bm25(all_chunks, k=k)
        retriever = build_ensemble(bm25, semantic)
    elif retriever_type == "BM25":
        retriever = build_bm25(all_chunks, k=k)
    elif retriever_type == "Chroma":
        retriever = build_chroma(all_chunks, k=k)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    if use_reranker:
        retriever = build_compressed_retriever(retriever, top_n=top_n)

    llm = ChatOpenAI(model=model, temperature=0)

    def answer_with(prompt):
        return ({"context": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    
    return {
        "retrieve": lambda q: retriever.get_relevant_documents(q),
        "ats": answer_with(ATS_PROMPT),
        "jd_doc": jd_doc,
        # "cover":     answer_with(COVER_LETTER_PROMPT),
        # "email":     answer_with(EMAIL_PROMPT),
        # "dm":        answer_with(DM_PROMPT),
    }

def generate_artifacts(p, question):
    query = p.get("jd_doc", question)
    docs = p["retrieve"](str(query))
    ctx  = make_context(docs)
    return {
        "ats_resume": p["ats"].invoke(ctx),
        # "cover_letter": p["cover"].invoke(ctx),
        # "recruiter_email": p["email"].invoke(ctx),
        # "recruiter_dm": p["dm"].invoke(ctx),
        "docs_used": len(docs),
    }
