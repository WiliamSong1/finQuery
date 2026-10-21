
import hashlib


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 200
) -> list[dict]:

    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        text = page["text"]
        document = page["document"]
        document_id = page["document_id"]
        page_number = page["page"]

        for start in range(0, len(text), step):
            content = text[start:start + chunk_size]

            if not content.strip():
                continue

            unique_key = (
                f"{document_id}:{page_number}:"
                f"{start}:{content}"
            )

            chunk_id = hashlib.sha256(
                unique_key.encode("utf-8")
            ).hexdigest()

            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "document": document,
                "page": page_number,
                "start_offset": start,
                "text": content
            })

            if start + chunk_size >= len(text):
                break

    return chunks
