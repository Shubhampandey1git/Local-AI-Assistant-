# MAIN GUI LOGIC IS HERE

# Packages
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.pdf_processor import load_and_split
from core.embedder import embedding_and_chromadb
from core.retriever import create_qa_chain
import gradio as gr

# Globals
VECTORSTORE_PATH = "../chroma_db"
qa_chain = None
#data_path = "../data/Shubham_Pandey_Resume.pdf"

# Loading and processing data
def upload_and_process(pdf):
    global qa_chain
    if not pdf:
        return "No PDF uploaded."
    
    document = load_and_split(pdf.name)
    vectorstore = embedding_and_chromadb(document, VECTORSTORE_PATH)
    qa_chain = create_qa_chain(vectorstore)
    
    return "PDF has been processed. You can now ask questions."

# Answer questions
def ask_question(query):
    if not qa_chain:
        return "Please upload a PDF first."
    return qa_chain.run(query)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("*Local PDF AI Assistant.*")
    gr.Markdown("Upload a PDF and ask questions about its content.")
    
    with gr.Row():
        pdf_input = gr.File(label = "Upload PDF: ", file_types = ['.pdf'])
        upload_button = gr.Button("Process PDF")
    upload_output = gr.Textbox(label = "Upload Status")
    
    question = gr.Textbox(label= "Enter Your Question: ")
    answer = gr.Textbox(label= "Answer: ")
    ask_button = gr.Button("ASK")
    
    upload_button.click(fn=upload_and_process, inputs=pdf_input, outputs=upload_output)
    ask_button.click(fn=ask_question, inputs=question, outputs=answer)

# Main function
if __name__ == "__main__":
    demo.launch()