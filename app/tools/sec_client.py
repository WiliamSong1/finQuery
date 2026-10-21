
import os
from datetime import date
from functools import lru_cache

import requests
from dotenv import load_dotenv


load_dotenv()

SEC_BASE = "https://data.sec.gov/api/xbrl/companyfacts"


@lru_cache(maxsize=16)
def get_company_facts(cik: str) -> dict:
    user_agent = os.getenv("SEC_USER_AGENT")

    if not user_agent:
        raise ValueError("SEC_USER_AGENT is missing")

    if not cik.isdigit() or len(cik) > 10:
        raise ValueError("Invalid CIK")

    padded_cik = cik.zfill(10)
    url = f"{SEC_BASE}/CIK{padded_cik}.json"

    response = requests.get(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json"
        },
        timeout=30
    )

    response.raise_for_status()
    return response.json()


def get_annual_metric(
    cik: str,
    metric: str,
    year: int
) -> dict:

    facts = get_company_facts(cik)

    concepts = facts["facts"].get("us-gaap", {})

    if metric not in concepts:
        raise ValueError(
            f"Metric '{metric}' is not available"
        )

    observations = (
        concepts[metric]
        .get("units", {})
        .get("USD", [])
    )

    candidates = []

    for item in observations:
        if item.get("form") != "10-K":
            continue

        if item.get("fp") != "FY":
            continue

        if "start" not in item or "end" not in item:
            continue

        start = date.fromisoformat(item["start"])
        end = date.fromisoformat(item["end"])

        duration = (end - start).days

        # For Apple, the fiscal year ends in September.
        if end.year != year:
            continue

        # Exclude quarterly and year-to-date figures.
        if not 330 <= duration <= 400:
            continue

        candidates.append(item)

    if not candidates:
        raise ValueError(
            f"No annual observation for {metric} in {year}"
        )

    # Prefer the original annual filing for that fiscal year.
    candidates.sort(
        key=lambda item: (
            item.get("fy") != year,
            item.get("filed", "9999-99-99")
        )
    )

    selected = candidates[0]

    return {
        "company": facts["entityName"],
        "cik": str(facts["cik"]),
        "metric": metric,
        "fiscal_year": year,
        "value": selected["val"],
        "unit": "USD",
        "period_start": selected["start"],
        "period_end": selected["end"],
        "filing_accession": selected["accn"],
        "filed": selected["filed"],
        "source": (
            f"{SEC_BASE}/CIK{str(facts['cik']).zfill(10)}.json"
        )
    }
