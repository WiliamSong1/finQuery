
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import chunk_pages
from pydantic import BaseModel
from app.ingestion.chunker import chunk_pages
from app.database.vector_store import store_chunks
from app.retrieval.hybrid_search import (
    hybrid_search,
    reranked_hybrid_search
)
import tempfile
import os
import hashlib
from pathlib import Path

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
        hasher = hashlib.sha256()

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False
        ) as temp:

            temp_path = temp.name

            while chunk := await file.read(1024 * 1024):
                hasher.update(chunk)
                temp.write(chunk)

        document_id = hasher.hexdigest()
        original_filename = Path(file.filename).name

        pages = parse_pdf(temp_path)

        for page in pages:
            page["document_id"] = document_id
            page["document"] = original_filename

        chunks = chunk_pages(pages)

        stored_count = store_chunks(chunks)

        return {
            "document_id": document_id,
            "filename": original_filename,
            "total_pages_with_text": len(pages),
            "total_chunks": len(chunks),
            "indexed_chunks": stored_count
        }

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="Could not process PDF"
        ) from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()


class SearchRequest(BaseModel):
    query: str
    limit: int = 5



from typing import Literal


class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    search_type: Literal[
        "dense",
        "bm25",
        "hybrid",
        "reranked"
    ] = "reranked"



@app.post("/search")
def search_documents(request: SearchRequest):

    if not request.query.strip() or not 1 <= request.limit <= 100:
        raise HTTPException(
            status_code=400,
            detail="Query must be nonempty and limit must be 1-100"
        )

    if request.search_type == "dense":
        from app.database.vector_store import search_chunks

        results = search_chunks(
            request.query,
            request.limit
        )

    elif request.search_type == "bm25":
        from app.retrieval.bm25_search import bm25_search

        results = bm25_search(
            request.query,
            request.limit
        )

    elif request.search_type == "hybrid":
        results = hybrid_search(
            request.query,
            request.limit
        )

    else:
        results = reranked_hybrid_search(
            request.query,
            request.limit
        )

    return {
        "query": request.query,
        "search_type": request.search_type,
        "results": results
    }


from app.database.vector_store import (
    get_client,
    COLLECTION_NAME,
    ensure_collection
)


@app.get("/stats")
def database_stats():
    ensure_collection()

    client = get_client()

    result = client.count(
        collection_name=COLLECTION_NAME,
        exact=True
    )

    return {
        "total_stored_chunks": result.count
    }



