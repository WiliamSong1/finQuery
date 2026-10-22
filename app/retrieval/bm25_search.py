
import re
from functools import lru_cache
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

@lru_cache(maxsize=1)
def get_bm25_index():
    """
    Build and cache the BM25 index for the
    currently stored document collection.
    """

    chunks = get_all_chunks()

    if not chunks:
        return None, []

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    # Avoid attempting to index empty documents.
    filtered = [
        (chunk, tokens)
        for chunk, tokens in zip(
            chunks, tokenized_corpus
        )
        if tokens
    ]

    if not filtered:
        return None, []

    indexed_chunks = [
        chunk for chunk, _ in filtered
    ]

    corpus = [
        tokens for _, tokens in filtered
    ]

    bm25 = BM25Okapi(corpus)

    print(
        f"[BM25] Index built with "
        f"{len(indexed_chunks)} chunks"
    )

    return bm25, indexed_chunks


def invalidate_bm25_index():
    """
    Force BM25 to rebuild after document ingestion.
    """
    get_bm25_index.cache_clear()



def bm25_search(
    query: str,
    limit: int = 5
) -> list[dict]:

    if not query.strip() or limit <= 0:
        return []

    bm25, chunks = get_bm25_index()

    if bm25 is None:
        return []

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
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

