from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk(docs, size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap
    )
    return splitter.split_documents(docs)