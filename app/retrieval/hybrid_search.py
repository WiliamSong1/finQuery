
from collections import defaultdict
from app.retrieval.reranker import rerank_chunks
from app.database.vector_store import search_chunks
from app.retrieval.bm25_search import bm25_search


def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    k: int = 60
) -> list[dict]:

    fused_scores = defaultdict(float)
    documents = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            chunk_id = result["chunk_id"]

            fused_scores[chunk_id] += 1 / (k + rank)

            if chunk_id not in documents:
                documents[chunk_id] = result.copy()

    sorted_ids = sorted(
        fused_scores,
        key=lambda chunk_id: fused_scores[chunk_id],
        reverse=True
    )

    return [
        {
            **documents[chunk_id],
            "score": fused_scores[chunk_id]
        }
        for chunk_id in sorted_ids
    ]


def hybrid_search(query: str, limit: int = 5) -> list[dict]:

    if not query.strip() or not 1 <= limit <= 100:
        raise ValueError("Invalid query or result limit")

    candidate_limit = max(20, limit * 4)

    dense_results = search_chunks(
        query=query,
        limit=candidate_limit
    )

    sparse_results = bm25_search(
        query=query,
        limit=candidate_limit
    )

    fused_results = reciprocal_rank_fusion([
        dense_results,
        sparse_results
    ])

    def reranked_hybrid_search(
            query: str,
            limit: int = 5,
            candidate_limit: int = 20
    ) -> list[dict]:

        if not query.strip() or not 1 <= limit <= 100:
            raise ValueError("Invalid query or result limit")

        candidate_limit = max(candidate_limit, limit)

        # Step 1: Retrieve candidates using hybrid search
        candidates = hybrid_search(
            query=query,
            limit=candidate_limit
        )

        # Step 2: Rerank the retrieved candidates
        results = rerank_chunks(
            query=query,
            chunks=candidates,
            limit=limit
        )
        return results


    return fused_results[:limit]

def reranked_hybrid_search(
    query: str,
    limit: int = 5,
    candidate_limit: int = 20
) -> list[dict]:

    # Validate the query and result limit
    if not query.strip() or not 1 <= limit <= 100:
        raise ValueError("Invalid query or result limit")

    # Ensure we retrieve enough candidates
    candidate_limit = max(candidate_limit, limit)

    # Step 1: Retrieve candidates using hybrid search
    candidates = hybrid_search(
        query=query,
        limit=candidate_limit
    )

    # Step 2: Rerank candidates using the cross-encoder
    results = rerank_chunks(
        query=query,
        chunks=candidates,
        limit=limit
    )

    # Step 3: Return the highest-ranked results
    return results

