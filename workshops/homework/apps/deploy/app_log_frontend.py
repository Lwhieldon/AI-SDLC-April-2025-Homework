import os
import gradio as gr
import fitz  # PyMuPDF
import sqlite3
from llama_index.core import VectorStoreIndex, Document
from datetime import datetime
import uuid

DB_FILE = "pdf_qa_logs.db"

# Initialize the database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interactions (
                 id TEXT PRIMARY KEY,
                 timestamp TEXT,
                 pdf_name TEXT,
                 query TEXT,
                 response TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Extract text from PDF
def extract_text_from_pdf(pdf_bytes):
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Process PDF and create index
def process_pdf(pdf_bytes):
    extracted_text = extract_text_from_pdf(pdf_bytes)
    document = Document(text=extracted_text)
    index = VectorStoreIndex.from_documents([document])
    return index

# Log to SQLite
def log_interaction(pdf_name, query, response):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    interaction_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO interactions VALUES (?, ?, ?, ?, ?)",
              (interaction_id, timestamp, pdf_name, query, response))
    conn.commit()
    conn.close()

# Query the PDF
def query_pdf(pdf, query):
    if pdf is None:
        return "Please upload a PDF."
    if not query.strip():
        return "Please enter a valid query."

    try:
        pdf_name = pdf.name if hasattr(pdf, 'name') else "Uploaded PDF"
        index = process_pdf(pdf)  # Pass bytes directly, no .read()

        query_engine = index.as_query_engine()
        response = query_engine.query(query)

        log_interaction(pdf_name, query, response.response)

        return response.response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio app
with gr.Blocks() as app:
    pdf_upload = gr.File(label="Upload PDF", type="binary")
    query_input = gr.Textbox(label="Ask a question about the PDF")
    output = gr.Textbox(label="Answer")

    query_button = gr.Button("Submit")
    query_button.click(query_pdf, inputs=[pdf_upload, query_input], outputs=output)

if __name__ == "__main__":
    app.launch()
