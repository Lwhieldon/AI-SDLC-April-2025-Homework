import os
import gradio as gr
import fitz
import sqlite3
import anthropic  # Claude 3 SDK
from datetime import datetime
import uuid

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

DB_FILE = "pdf_qa_logs.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interactions (
                 id TEXT PRIMARY KEY,
                 timestamp TEXT,
                 pdf_name TEXT,
                 query TEXT,
                 prompt TEXT,
                 response TEXT)''')
    conn.commit()
    conn.close()

init_db()

def extract_text_chunks(pdf_bytes):
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc.load_page(page_num)
        text = page.get_text("text").strip()
        if text:
            chunks.append({"page": page_num + 1, "text": text})
    return chunks

def simple_keyword_ranking(chunks, query, top_k=5):
    ranked = sorted(
        chunks,
        key=lambda c: sum(query.lower().count(word.lower()) for word in c["text"].split()),
        reverse=True
    )
    return ranked[:top_k] if ranked else []

def build_prompt(chunks, query):
    context = "\n\n".join([f"Page {c['page']}:\n{c['text']}" for c in chunks])
    return f"""You are a helpful assistant answering questions about a PDF document. Use only the content provided below to answer the user's question. If the answer is not found, say \"Not found in PDF\".

--- PDF EXCERPTS ---
{context}
--- END EXCERPTS ---

Question: {query}
Answer:"""

def log_interaction(pdf_name, query, prompt, response):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    interaction_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?)", 
              (interaction_id, timestamp, pdf_name, query, prompt, response))
    conn.commit()
    conn.close()

def query_pdf(pdf, query):
    if pdf is None:
        return "Please upload a PDF."
    if not query.strip():
        return "Please enter a valid query."

    try:
        pdf_bytes = pdf.read() if hasattr(pdf, 'read') else pdf
        pdf_name = pdf.name if hasattr(pdf, 'name') else "Uploaded PDF"
        chunks = extract_text_chunks(pdf_bytes)
        top_chunks = simple_keyword_ranking(chunks, query)

        if not top_chunks:
            return "Sorry, no relevant content found."

        prompt = build_prompt(top_chunks, query)

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.content[0].text.strip()
        log_interaction(pdf_name, query, prompt, answer)
        return answer

    except Exception as e:
        return f"An error occurred: {str(e)}"

with gr.Blocks() as app:
    pdf_upload = gr.File(label="Upload PDF", type="binary")
    query_input = gr.Textbox(label="Ask a question about the PDF")
    output = gr.Textbox(label="Answer")

    query_button = gr.Button("Submit")
    query_button.click(query_pdf, inputs=[pdf_upload, query_input], outputs=output)

app.launch()