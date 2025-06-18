# RAG PIPELINE IS HERE (QA CHAIN WITH OLLaMa)

from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

def create_qa_chain(vectorstore):
    # Setting up Local LLM with Ollama
    llm = Ollama(model='llama2')
    # Creating the RetrievalQA chain(RAG)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa_chain