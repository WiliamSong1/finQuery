
from functools import lru_cache

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(MODEL_NAME)


def rerank_chunks(
    query: str,
    chunks: list[dict],
    limit: int = 5
) -> list[dict]:

    if not chunks or limit <= 0:
        return []

    if not query.strip():
        raise ValueError("Query cannot be empty")

    model = get_reranker()

    # Pair the question with every retrieved chunk
    pairs = [
        (query, chunk["text"])
        for chunk in chunks
    ]

    # Generate relevance scores
    scores = model.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False
    )

    reranked = []

    for chunk, score in zip(chunks, scores):
        result = chunk.copy()

        # Preserve original hybrid RRF score
        result["retrieval_score"] = result.pop(
            "score", None
        )

        result["rerank_score"] = float(score)

        reranked.append(result)

    # Sort highest relevance score first
    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return reranked[:limit]
