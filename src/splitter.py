from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk(docs, size=900, overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n","\n"," ",""]
    )
    return splitter.split_documents(docs)