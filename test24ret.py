
from app.retrieval.scoped_search import search_filing_year
from app.tools.financial_analysis import select_explanation


query = (
    "Apple 2024 compared with 2023 "
    "research development expense growth "
    "selling general administrative costs "
    "headcount primarily driven"
)

results = search_filing_year(
    query=query,
    year=2024,
    limit=10
)

print("Retrieved chunks:", len(results))

for chunk in results:
    print("\nDocument:", chunk["document"])
    print("Page:", chunk["page"])
    print("Score:", chunk["rerank_score"])

    explanation = select_explanation(chunk["text"])
    print("Explanation:", explanation or "None found")
