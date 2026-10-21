
import re

from rank_bm25 import BM25Okapi
from app.database.vector_store import (
    get_client,
    ensure_collection,
    COLLECTION_NAME
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+(?:[.-]\w+)*\b", text.lower())


def get_all_chunks() -> list[dict]:
    ensure_collection()
    client = get_client()

    chunks = []
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for record in records:
            if record.payload and record.payload.get("text"):
                chunks.append(record.payload)

        if offset is None:
            break

    return chunks


def bm25_search(query: str, limit: int = 10) -> list[dict]:
    chunks = get_all_chunks()

    if not chunks or not query.strip() or limit <= 0:
        return []

    corpus = [tokenize(chunk["text"]) for chunk in chunks]

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda i: float(scores[i]),
        reverse=True
    )

    results = []

    for index in ranked_indices:
        score = float(scores[index])

        if score <= 0:
            continue

        results.append({
            **chunks[index],
            "score": score
        })

        if len(results) >= limit:
            break

    return results
