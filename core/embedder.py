# EMBEDDING AND CHROMADB SETUP IS HERE

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def embedding_and_chromadb(documents, VECTORSTORE_PATH):
    # Creating Embeddings and Storing it in ChromaDB
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embedding, persist_directory=VECTORSTORE_PATH)
    
    return vectorstore