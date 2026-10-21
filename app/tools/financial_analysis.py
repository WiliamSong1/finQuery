
import re
from decimal import Decimal

from app.tools.sec_comparison import compare_sec_financials
from app.retrieval.scoped_search import search_filing_year



def select_explanation(text: str) -> str:
    normalized = " ".join(text.split())

    patterns = [
        r"The growth in R&D expense.*?(?:\.|$)",
        r"The increase in R&D expense.*?(?:\.|$)",
        r"The growth in selling, general and administrative expenses.*?(?:\.|$)",
        r"The increase in selling, general and administrative expenses.*?(?:\.|$)"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE
        )

        if match:
            excerpt = match.group().strip()

            if len(excerpt) >= 40:
                return excerpt

    return ""



def analyze_operating_expenses(
    start_year: int,
    end_year: int
) -> dict:

    # Step 1: Get authoritative financial values.
    comparison = compare_sec_financials(
        cik="0000320193",
        metric="operating_expenses",
        start_year=start_year,
        end_year=end_year
    )

    million = Decimal("1000000")

    start = Decimal(comparison["start_value"]) / million
    end = Decimal(comparison["end_value"]) / million
    change = Decimal(comparison["absolute_change"]) / million

    pct = comparison["percentage_change"]

    if change > 0:
        direction = "increased"
    elif change < 0:
        direction = "decreased"
    else:
        direction = "remained unchanged"

    answer = (
        f"Apple's operating expenses {direction} "
        f"from ${start:,.0f} million in {start_year} [1] "
        f"to ${end:,.0f} million in {end_year} [2]. "
        f"The change was ${abs(change):,.0f} million."
    )

    if pct is not None:
        answer += (
            f" This represents a "
            f"{abs(Decimal(pct)):.2f}% "
            f"{'increase' if change > 0 else 'decrease' if change < 0 else 'change'}."
        )

    sources = [
        {
            "citation_id": str(i),
            "source_type": "sec_xbrl",
            **source
        }
        for i, source in enumerate(
            comparison["sources"],
            start=1
        )
    ]

    # Step 2: Retrieve relevant management discussion
    # for each adjacent-year comparison.
    commentary = []
    missing_years = []

    for year in range(start_year + 1, end_year + 1):
        query = (
            f"Apple fiscal {year} compared with {year - 1} "
            "operating expenses research and development "
            "selling general administrative expense growth "
            "primarily driven by increases in costs"
        )

        chunks = search_filing_year(
            query=query,
            year=year,
            limit=5
        )

        selected = None
        excerpt = ""

        for chunk in chunks:
            sentence = select_explanation(chunk["text"])

            if sentence:
                selected = chunk
                excerpt = sentence
                break

        if selected is None:
            missing_years.append(year)
            continue

        citation_id = str(len(sources) + 1)

        sources.append({
            "citation_id": citation_id,
            "source_type": "pdf",
            "document": selected["document"],
            "document_id": selected.get("document_id"),
            "chunk_id": selected["chunk_id"],
            "page": selected["page"],
            "fiscal_year": year
        })

        commentary.append({
            "year": year,
            "compared_to": year - 1,
            "excerpt": excerpt,
            "citation_id": citation_id
        })

    # Step 3: Append source quotations.
    if commentary:
        answer += "\n\nManagement discussion excerpts:"

        for item in commentary:
            answer += (
                f"\n\n{item['year']} compared with "
                f"{item['compared_to']}: "
                f"\"{item['excerpt']}\" "
                f"[{item['citation_id']}]"
            )

    status = (
        "answered"
        if not missing_years
        else "partial_evidence"
    )

    return {
        "answer": answer,
        "status": status,
        "answer_type": "financial_analysis",
        "calculation": comparison,
        "commentary": commentary,
        "missing_commentary_years": missing_years,
        "sources": sources
    }
