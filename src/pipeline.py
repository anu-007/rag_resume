from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from .loader import load_local, scrape_jd
from .splitter import chunk
from .retrievers import build_bm25, build_chroma, build_hyde, build_ensemble
from .rerank import build_compressed_retriever
from .prompt import ATS_PROMPT

def make_context(docs):
    blocks = []
    for i, d in enumerate(docs, 1):
        title = d.metadata.get("title") or d.metadata.get("source") or f"doc_{i}"
        page  = d.metadata.get("page")
        head  = f"[S{i}] {title}" + (f" | page {page}" if page is not None else "")
        blocks.append(f"{head}\n{d.page_content}")
    return "\n\n".join(blocks)

def sources_footer(docs):
    lines = ["Sources:"]
    for i, d in enumerate(docs,1):
        title = d.metadata.get("title") or d.metadata.get("source") or f"doc_{i}"
        page  = d.metadata.get("page")
        url   = d.metadata.get("url")
        tail  = [x for x in [d.metadata.get("source"), f"page {page}" if page is not None else None, url] if x]
        lines.append(f"[S{i}] {title} → " + (" | ".join(tail) if tail else title))
    return "\n".join(lines)

def build_pipeline(resume_paths, jd_url, model="gemma3:4b"):
    resumes = chunk(load_local(resume_paths))
    jd_doc  = scrape_jd(jd_url)
    jd_chunks = chunk([jd_doc])

    all_chunks = resumes + jd_chunks

    semantic = build_chroma(all_chunks, k=12)
    bm25 = build_bm25(all_chunks, k=12)
    hyde = build_hyde(semantic, model=model)
    ensemble = build_ensemble(bm25, semantic, hyde)
    retriever = build_compressed_retriever(ensemble, top_n=8)

    llm = ChatOllama(model=model, temperature=0)

    def answer_with(prompt):
        return ({"context": RunnablePassthrough()} | prompt | llm | StrOutputParser())

    return {
        "retrieve": lambda q: retriever.get_relevant_documents(q),
        "ats":       answer_with(ATS_PROMPT),
        # "cover":     answer_with(COVER_LETTER_PROMPT),
        # "email":     answer_with(EMAIL_PROMPT),
        # "dm":        answer_with(DM_PROMPT),
    }

def generate_artifacts(p, question):
    docs = p["retrieve"](question)
    ctx  = make_context(docs) + "\n\n" + sources_footer(docs)
    return {
        "ats_resume": p["ats"].invoke(ctx),
        # "cover_letter": p["cover"].invoke(ctx),
        # "recruiter_email": p["email"].invoke(ctx),
        # "recruiter_dm": p["dm"].invoke(ctx),
        "docs_used": len(docs),
    }
