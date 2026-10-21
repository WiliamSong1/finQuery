
import re


def extract_years(query: str) -> list[int]:
    years = [
        int(year)
        for year in re.findall(r"\b20\d{2}\b", query)
    ]

    years = sorted(set(years))

    if not years:
        return []

    # Expand explicit year ranges
    range_requested = bool(
        re.search(
            r"\b(between|from|through)\b",
            query,
            re.IGNORECASE
        )
    )

    if range_requested and len(years) == 2:
        start, end = years
        if end - start <= 4:
            return list(range(start, end + 1))

    return years


def plan_query(query: str) -> dict:
    years = extract_years(query)

    if len(years) > 5:
        raise ValueError(
            "This version supports up to five reporting years"
        )

    subqueries = []

    if years:
        for year in years:
            subqueries.append({
                "year": year,
                "query": (
                    f"{query}\n"
                    f"Retrieve financial information for "
                    f"fiscal year {year}."
                )
            })
    else:
        subqueries.append({
            "year": None,
            "query": query
        })

    return {
        "required_years": years,
        "subqueries": subqueries
    }


def document_year(document_name: str):
    matches = re.findall(
        r"20\d{2}",
        document_name
    )

    if not matches:
        return None

    return int(matches[-1])
