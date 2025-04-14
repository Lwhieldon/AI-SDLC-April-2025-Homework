from fastapi import FastAPI  # FastAPI for building the API layer
from gradio.routes import mount_gradio_app  # Allows embedding Gradio inside FastAPI
import modal  # Modal for serverless deployment

from app_frontend import app as blocks  # Import Gradio app (from app_frontend.py)

# Create a lightweight Modal image (Debian-based) with required dependencies
image = modal.Image.debian_slim().pip_install(
    "gradio<6",  # Use Gradio version below 6 (v6 may break compatibility)
    "pymupdf",  # PDF processing library
    "llama-index-core",  # Core LlamaIndex library for indexing/querying
    "llama-index-embeddings-openai",  # OpenAI embeddings for LlamaIndex
    "llama-index-llms-openai"  # OpenAI LLM integration for LlamaIndex
)

# Define the Modal app container
app = modal.App("pdf-query-app", image=image)


@app.function(
    concurrency_limit=1,  # Only one instance (Gradio uses local file storage, preventing multiple replicas)
    allow_concurrent_inputs=1000,  # Async handling for up to 1000 concurrent requests within a single instance
    secrets=[modal.Secret.from_name("openai-secret")]  # Fetch OpenAI API key from Modal secrets
)
@modal.asgi_app()  # Register this as an ASGI app (compatible with FastAPI)
def serve() -> FastAPI:
    """
    Main server function: 
    - Wraps Gradio inside FastAPI 
    - Deploys the API through Modal with a single instance for session consistency
    """
    api = FastAPI(docs=True)  # Enable Swagger documentation at /docs
    return mount_gradio_app(app=api, blocks=blocks, path="/")  # Mount Gradio app at root path


@app.local_entrypoint()
def main():
    """
    Local development entry point: 
    - Allows running the app locally for testing
    - Prints the type of Gradio app to confirm readiness
    """
    print(f"{type(blocks)} is ready to go!")