
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.ingestion.parser import parse_pdf
from app.retrieval.bm25_search import invalidate_bm25_index
from contextlib import asynccontextmanager
from app.database.vector_store import get_client
from app.ingestion.chunker import chunk_pages
from pydantic import BaseModel
from app.ingestion.chunker import chunk_pages
from app.core.config import settings
from app.database.vector_store import store_chunks
from app.retrieval.hybrid_search import (
    hybrid_search,
    reranked_hybrid_search
)
from openai import RateLimitError, APIConnectionError
from app.tools.sec_comparison import compare_sec_financials
from app.generation.answer import generate_answer
import tempfile
import os
import logging
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)
import hashlib
from pathlib import Path
from app.agents.graph import run_agent
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FinQuery starting")

    yield

    logger.info("FinQuery shutting down")

    get_client.cache_info()

    if get_client.cache_info().currsize > 0:
        get_client().close()
        get_client.cache_clear()

    logger.info("Qdrant connection closed")

app = FastAPI(
    title="FinQuery API",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "healthy"}




@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):

    if (
        not file.filename
        or not file.filename.lower().endswith(".pdf")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    filename = file.filename.replace("\\", "/").split("/")[-1]
    max_bytes = settings.max_upload_mb * 1024 * 1024

    temp_path = None
    total_bytes = 0
    hasher = hashlib.sha256()

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False
        ) as temp:

            temp_path = temp.name

            while True:
                data = await file.read(1024 * 1024)

                if not data:
                    break

                total_bytes += len(data)

                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Maximum upload size is "
                            f"{settings.max_upload_mb} MB"
                        )
                    )

                # Verify the PDF signature.
                if total_bytes == len(data):
                    if not data.startswith(b"%PDF-"):
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid PDF file signature"
                        )

                hasher.update(data)
                temp.write(data)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

        document_id = hasher.hexdigest()

        try:
            pages = parse_pdf(temp_path)
        except Exception as exc:
            logger.warning(
                "PDF parsing failed for %s",
                filename,
                exc_info=True
            )
            raise HTTPException(
                status_code=422,
                detail="Could not parse PDF"
            ) from exc

        if not pages:
            raise HTTPException(
                status_code=422,
                detail="PDF contains no extractable text"
            )

        for page in pages:
            page["document_id"] = document_id
            page["document"] = filename

        chunks = chunk_pages(pages)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No searchable document chunks found"
            )

        stored_count = store_chunks(chunks)
        invalidate_bm25_index()
        return {
            "document_id": document_id,
            "filename": filename,
            "size_bytes": total_bytes,
            "total_pages_with_text": len(pages),
            "total_chunks": len(chunks),
            "indexed_chunks": stored_count
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Document ingestion failed")

        raise HTTPException(
            status_code=500,
            detail="Document processing failed"
        ) from exc

    finally:
        await file.close()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)



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



class AskRequest(BaseModel):
    query: str
    limit: int = 5


@app.post("/ask")
def ask_question(request: AskRequest):

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    if not 1 <= request.limit <= 10:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 10"
        )

    try:
        result = generate_answer(
            query=request.query,
            limit=request.limit
        )

        return {
            "query": request.query,
            **result
        }

    except RateLimitError as exc:
        logger.warning("LLM provider rate limit reached")

        raise HTTPException(
            status_code=503,
            detail={
                "error": "llm_rate_limited",
                "message": (
                    "The language model is temporarily "
                    "unavailable. Please retry later."
                )
            }
        ) from exc

    except APIConnectionError as exc:
        logger.error("LLM provider connection failed")

        raise HTTPException(
            status_code=503,
            detail={
                "error": "llm_unavailable",
                "message": (
                    "Unable to connect to the "
                    "language model provider."
                )
            }
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc

@app.post("/agent/ask")
def agent_ask(request: AskRequest):

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    if not 1 <= request.limit <= 10:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 10"
        )

    result = run_agent(
        query=request.query,
        limit=request.limit
    )

    return result

class FinancialCompareRequest(BaseModel):
    cik: str
    metric: str
    start_year: int
    end_year: int


@app.post("/financial/compare")
def financial_compare(request: FinancialCompareRequest):

    try:
        return compare_sec_financials(
            cik=request.cik,
            metric=request.metric,
            start_year=request.start_year,
            end_year=request.end_year
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

