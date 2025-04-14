import os
import gradio as gr
import fitz
import sqlite3
import anthropic
from datetime import datetime
import uuid
import csv

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
DB_FILE = "pdf_qa_logs.db"
EXPORT_FILE = "interactions_export.csv"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interactions (
                 id TEXT PRIMARY KEY,
                 timestamp TEXT,
                 pdf_name TEXT,
                 query TEXT,
                 task_type TEXT,
                 temperature REAL,
                 top_p REAL,
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

def build_prompt(chunks, query, task_type):
    context = "\n\n".join([f"Page {c['page']}:\n{c['text']}" for c in chunks])

    if task_type == "extract":
        return f"""Extract structured data (like JSON) from the text below. Be precise and consistent.

--- PDF EXCERPTS ---
{context}
--- END EXCERPTS ---

Question: {query}
Output as JSON:"""
    else:  # email
        return f"""You're a recruiter writing personalized outreach emails based on the information below. Make it creative, professional, and warm.

--- PDF EXCERPTS ---
{context}
--- END EXCERPTS ---

Question: {query}
Email:"""

def log_interaction(pdf_name, query, task_type, temperature, top_p, prompt, response):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    interaction_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 
              (interaction_id, timestamp, pdf_name, query, task_type, temperature, top_p, prompt, response))
    conn.commit()
    conn.close()

def query_pdf(pdf, query, task_type, temperature, top_p):
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

        prompt = build_prompt(top_chunks, query, task_type)

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=temperature,
            top_p=top_p,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.content[0].text.strip()
        log_interaction(pdf_name, query, task_type, temperature, top_p, prompt, answer)
        return answer

    except Exception as e:
        return f"An error occurred: {str(e)}"

def export_interactions_to_csv():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM interactions")
    rows = c.fetchall()
    headers = [desc[0] for desc in c.description]

    with open(EXPORT_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    conn.close()
    return EXPORT_FILE

# -- UI --
with gr.Blocks() as app:
    gr.Markdown("### 📄 PDF QA Assistant with Temperature & Top-p Tuning")

    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", type="binary")
        task_type = gr.Radio(["extract", "email"], label="Task Type", value="extract")

    query_input = gr.Textbox(label="Enter your question or request")

    with gr.Row():
        temperature_input = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.2, step=0.1)
        top_p_input = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, value=0.1, step=0.1)

    output = gr.Textbox(label="LLM Output")

    query_button = gr.Button("Submit")
    query_button.click(
        query_pdf,
        inputs=[pdf_upload, query_input, task_type, temperature_input, top_p_input],
        outputs=output
    )

    gr.Markdown("### 📥 Export Logged Interactions")
    download_btn = gr.Button("Download CSV of Logs")
    csv_output = gr.File(label="Download your CSV here")
    download_btn.click(export_interactions_to_csv, outputs=csv_output)

app.launch()
