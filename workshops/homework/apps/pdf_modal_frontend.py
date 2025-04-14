import os
import gradio as gr
import fitz  # PyMuPDF
import anthropic
from datetime import datetime

client = anthropic.Anthropic()

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

def query_pdf(pdf, query, task_type, temperature, top_p):
    if pdf is None:
        return "Please upload a PDF."
    if not query.strip():
        return "Please enter a valid query."

    try:
        pdf_bytes = pdf.read() if hasattr(pdf, 'read') else pdf
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
        return answer

    except Exception as e:
        return f"An error occurred: {str(e)}"

# UI
with gr.Blocks() as app:
    gr.Markdown("# 📄 PDF Query & Email Assistant")
    gr.Markdown("### Step 1: Upload a PDF File!")
    gr.Markdown("### Step 2: Toggle Task Type to Extract Summary or Create Email.")
    gr.Markdown("Note: This app is for demo purposes. Please do not upload sensitive documents.")
    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", type="binary")
        task_type = gr.Radio(["extract", "email"], label="Task Type", value="extract")
    query_input = gr.Textbox(label="Enter your question or request")
    with gr.Row():
        temperature_input = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
        top_p_input = gr.Slider(label="Top-p", minimum=0.1, maximum=1.0, value=1.0, step=0.05)
    output = gr.Textbox(label="LLM Output", lines=10)
    query_button = gr.Button("Submit")
    query_button.click(query_pdf, 
                       inputs=[pdf_upload, query_input, task_type, temperature_input, top_p_input],
                       outputs=output)

if __name__ == "__main__":
    app.launch()
