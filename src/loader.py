import os, requests, io
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from PIL import Image
import pytesseract

def load_local(data_dir="data"):
    docs = []
    paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, fname))]
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

def _parse_html(html_content: str) -> Document:
    soup = BeautifulSoup(html_content, "html.parser")
    title_el = soup.find(["h1","h2"])
    parts = [title_el.get_text(" ", strip=True)] if title_el else []
    for sel in ["p","li","div"]:
        for el in soup.find_all(sel):
            txt = el.get_text(" ", strip=True)
            if txt and len(txt) > 40: parts.append(txt)
    return Document(
        page_content="\n".join(parts),
        metadata={"source":"Job Description", "title": parts[0] if parts else "JD"}
    )

def scrape_jd(url: str) -> Document:
    r = requests.get(url, timeout=20); r.raise_for_status()
    doc = _parse_html(r.text)
    doc.metadata["url"] = url
    return doc

def load_image(file_bytes: bytes, file_name: str) -> Document:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)

        return Document(
            page_content=text,
            metadata={"source": file_name, "title": "Job Description"}
        )
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError("Tesseract is not installed. Please install it to use image uploads.")
