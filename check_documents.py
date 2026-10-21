
from collections import Counter

from app.retrieval.bm25_search import get_all_chunks
from app.agents.planner import document_year


chunks = get_all_chunks()

counts = Counter(
    (chunk["document"], document_year(chunk["document"]))
    for chunk in chunks
)

print("Indexed documents:")

for (filename, year), count in sorted(counts.items()):
    print(f"{filename} | Year: {year} | Chunks: {count}")
