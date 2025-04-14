import modal
import openai  # Import OpenAI to set the API key

# Define the Modal app
app = modal.App("pdf-query-app")

# Create a container image with necessary dependencies
image = (
    modal.Image.debian_slim()
    .pip_install("gradio", "pymupdf", "llama-index-core", "llama-index-embeddings-openai", "llama-index-llms-openai")
)

# Modal function to launch the Gradio app
@app.function(image=image)
def run_gradio():
    import gradio as gr
    from llama_index.core import VectorStoreIndex, Document
    import fitz  # PyMuPDF

    # Set OpenAI API key directly
    openai.api_key = "XXX"  # Replace with your actual API key

    # Extract text from each page of a PDF
    def extract_text_from_pdf(pdf_file):
        pdf_doc = fitz.open(stream=pdf_file, filetype="pdf")
        text = "".join([page.get_text("text") for page in pdf_doc])
        return text

    # Create an index from the extracted PDF text
    def process_pdf(pdf_file):
        extracted_text = extract_text_from_pdf(pdf_file)
        document = Document(text=extracted_text)
        index = VectorStoreIndex.from_documents([document])
        return index

    # Query the indexed PDF
    def query_pdf(pdf, query):
        index = process_pdf(pdf)
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return response.response

    # Gradio app interface setup
    with gr.Blocks() as demo:
        gr.Markdown("## PDF Q&A App")
        pdf_upload = gr.File(label="Upload PDF", type="binary")
        query_input = gr.Textbox(label="Ask a question about the PDF")
        output = gr.Textbox(label="Answer")

        query_button = gr.Button("Submit")
        query_button.click(query_pdf, inputs=[pdf_upload, query_input], outputs=output)

    # Launch the Gradio app and expose it for external access
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)