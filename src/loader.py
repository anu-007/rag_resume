import os, requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_local(paths):
    docs = []
    for p in paths:
        if p.lower().endswith(".pdf"):
            for d in PyPDFLoader(p).load():
                d.metadata.setdefault("source", os.path.basename(p))
                docs.append(d)
        else:
            for d in TextLoader(p, encoding="utf-8").load():
                d.metadata.setdefault("source", os.path.basename(p))
                docs.append(d) 
    return docs

def scrape_jd(url: str) -> Document:
    r = requests.get(url, timeout=20); r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    title_el = soup.find(["h1","h2"])
    parts = [title_el.get_text(" ", strip=True)] if title_el else []
    for sel in ["p","li","div"]:
        for el in soup.find_all(sel):
            txt = el.get_text(" ", strip=True)
            if txt and len(txt) > 40: parts.append(txt)
    return Document(
        page_content="\n".join(parts),
        metadata={"source":"Job Description", "title": parts[0] if parts else "JD", "url": url}
    )