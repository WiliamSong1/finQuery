
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import chunk_pages

import tempfile
import os

app = FastAPI(title="Financial RAG API")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False
        ) as temp:
            temp_path = temp.name

            while chunk := await file.read(1024 * 1024):
                temp.write(chunk)

        pages = parse_pdf(temp_path)
        chunks = chunk_pages(pages)

        return {
            "filename": file.filename,
            "total_pages_with_text": len(pages),
            "total_chunks": len(chunks),
            "chunks": chunks
        }

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="Could not parse PDF"
        ) from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
