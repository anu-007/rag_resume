from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk(docs, size=400, overlap=80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n- ","\n\n","\n"," ",""]
    )
    return splitter.split_documents(docs)