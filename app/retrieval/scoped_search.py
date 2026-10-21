
from qdrant_client import models
from rank_bm25 import BM25Okapi

from app.agents.planner import document_year
from app.database.vector_store import (
    get_client,
    ensure_collection,
    COLLECTION_NAME
)
from app.retrieval.embeddings import embed_texts
from app.retrieval.bm25_search import (
    get_all_chunks,
    tokenize
)
from app.retrieval.hybrid_search import reciprocal_rank_fusion
from app.retrieval.reranker import rerank_chunks


def search_filing_year(
    query: str,
    year: int,
    limit: int = 3
) -> list[dict]:

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if not 1 <= limit <= 10:
        raise ValueError("Limit must be between 1 and 10")

    # Identify Apple documents for the requested year.
    scoped_chunks = [
        chunk
        for chunk in get_all_chunks()
        if document_year(chunk["document"]) == year
        and chunk["document"].lower().startswith("apple")
        and chunk.get("document_id")
    ]

    if not scoped_chunks:
        return []

    document_ids = list({
        chunk["document_id"]
        for chunk in scoped_chunks
    })

    ensure_collection()
    client = get_client()

    # Dense retrieval with metadata filtering.
    vector = embed_texts([query])[0]

    dense_points = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(
                        any=document_ids
                    )
                )
            ]
        ),
        limit=30,
        with_payload=True
    ).points

    dense_results = [
        {
            **point.payload,
            "score": point.score
        }
        for point in dense_points
    ]

    # BM25 over the same year-specific documents.
    corpus = [
        tokenize(chunk["text"])
        for chunk in scoped_chunks
    ]

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))

    ranked_indices = sorted(
        range(len(scoped_chunks)),
        key=lambda i: float(scores[i]),
        reverse=True
    )

    sparse_results = [
        {
            **scoped_chunks[i],
            "score": float(scores[i])
        }
        for i in ranked_indices[:30]
        if scores[i] > 0
    ]

    # Combine dense and sparse rankings.
    fused = reciprocal_rank_fusion([
        dense_results,
        sparse_results
    ])

    # Rerank against the original query.
    return rerank_chunks(
        query=query,
        chunks=fused[:30],
        limit=limit
    )
