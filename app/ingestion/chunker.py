
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
        page_number = page["page"]

        for start in range(0, len(text), step):
            content = text[start:start + chunk_size]

            if not content.strip():
                continue

            chunk_id = hashlib.sha256(
                f"{document}:{page_number}:{start}:{content}".encode()
            ).hexdigest()[:16]

            chunks.append({
                "chunk_id": chunk_id,
                "document": document,
                "page": page_number,
                "start_offset": start,
                "text": content
            })

            if start + chunk_size >= len(text):
                break

    return chunks
