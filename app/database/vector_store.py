
import uuid
from functools import lru_cache
from app.core.config import settings
from qdrant_client import QdrantClient, models
from app.retrieval.embeddings import embed_texts


COLLECTION_NAME = "financial_documents"
VECTOR_SIZE = 384



@lru_cache(maxsize=1)
def get_client():
    database_path = settings.qdrant_path.resolve()

    database_path.mkdir(
        parents=True,
        exist_ok=True
    )

    return QdrantClient(
        path=str(database_path)
    )



def ensure_collection():
    client = get_client()

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )


def store_chunks(chunks: list[dict]) -> int:
    if not chunks:
        return 0

    ensure_collection()
    client = get_client()

    embeddings = embed_texts(
        [chunk["text"] for chunk in chunks]
    )

    points = []

    for chunk, vector in zip(chunks, embeddings):
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk["chunk_id"]
            )
        )

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "document": chunk["document"],
                    "page": chunk["page"],
                    "start_offset": chunk["start_offset"],
                    "text": chunk["text"]
                }
            )
        )

    for start in range(0, len(points), 100):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start:start + 100],
            wait=True
        )

    return len(points)


def search_chunks(query: str, limit: int = 5) -> list[dict]:
    ensure_collection()
    client = get_client()

    if not query.strip() or limit <= 0:
        return []

    query_vector = embed_texts([query])[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit
    ).points

    return [
        {
            "score": result.score,
            **result.payload
        }
        for result in results
    ]
