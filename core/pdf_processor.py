# PDF TO CHUNKS FUNCTIONS ARE HERE

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split(data_path):
    # Loading the PDF file
    loader = PyPDFLoader(data_path)
    pages = loader.load()

    # Splitting the text into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = splitter.split_documents(pages)
    
    return documents
