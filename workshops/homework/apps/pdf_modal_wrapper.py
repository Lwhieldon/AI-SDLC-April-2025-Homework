# file: pdf_modal_wrapper.py
from fastapi import FastAPI
from gradio.routes import mount_gradio_app
import modal
from pdf_modal_frontend import app as blocks

# Build a Modal image with required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "gradio<6",
        "python-dotenv",
        "anthropic",
        "PyMuPDF",
        "fastapi"
    )
)

# Modal app wrapper
app = modal.App("pdf-query-email-app", image=image)
image_with_source = image.add_local_python_source("pdf_modal_frontend")

@app.function(
    max_containers=1,
    secrets=[modal.Secret.from_name("anthropic-key")]
)
@modal.asgi_app()
def serve() -> FastAPI:
    from pdf_modal_frontend import app as blocks
    api = FastAPI()
    return mount_gradio_app(app=api, blocks=blocks, path="/")

@app.local_entrypoint()
def main():
    from pdf_modal_frontend import app as blocks
    print(f"{type(blocks)} ready for local development")