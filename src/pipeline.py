from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from .loader import load_local
from .splitter import chunk
from .retrievers import build_bm25, build_chroma, build_ensemble
from .rerank import build_compressed_retriever
from .prompt import rewrite_prompt, score_prompt

def make_context(docs):
    context = "\n\n".join(doc.page_content for doc in docs)
    return context

def build_pipeline(jd_doc: Document, retriever_type="Ensemble", k=12, use_reranker=True, top_n=8):
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
    
    return {
        "retrieve": lambda q: retriever.get_relevant_documents(q),
        "jd_doc": jd_doc
    }

def generate_artifacts(p, model, question="resume"):
    try:
        llm = ChatOpenAI(model=model, temperature=0)
        if question == 'resume':
            query = p.get("jd_doc")
            docs = p["retrieve"](str(query))
            ctx  = make_context(docs)
            rewrite_chain = rewrite_prompt | llm
            resp = rewrite_chain.invoke({"jd": query, "resume": ctx })
            return resp.content
        else:
            query = p.get("jd_doc")
            score_chain = score_prompt | llm
            resume = "\n\n".join(doc.page_content for doc in load_local())
            resp = score_chain.invoke({"jd": query, "resume": resume })
            return resp.content
    except Exception as e:
        print(e)


TODO:
# print and see what is coming from the retriver, and check different overlaps
# update the prompt to only rewrite the matching job point to include ATS keywords
# find a way to add skills to resume which is mentioned in jd
# maintain the formatting
# find a way to pass the improvements to rewrite resume and rewrite it