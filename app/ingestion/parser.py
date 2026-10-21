
import fitz
from pathlib import Path


def parse_pdf(file_path: str) -> list[dict]:
    pages = []

    with fitz.open(file_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")

            if not text.strip():
                continue

            pages.append({
                "document": Path(file_path).name,
                "page": page_number,
                "text": text.strip()
            })

    return pages
