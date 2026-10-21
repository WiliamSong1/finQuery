
import re
from decimal import Decimal

from app.tools.financial_records import FinancialRecord


METRIC_ALIASES = {
    "total operating expenses": "operating_expenses",
    "operating expenses": "operating_expenses",
    "total net sales": "revenue",
    "total net revenue": "revenue",
    "operating income": "operating_income",
    "net income": "net_income"
}

VALUE_PATTERN = re.compile(
    r"^\s*(20\d{2})\s*:\s*"
    r"(\$?\(?-?\d[\d,]*(?:\.\d+)?\)?)\s*$"
)


def parse_amount(raw: str) -> Decimal:
    cleaned = raw.replace("$", "").replace(",", "").strip()

    negative = cleaned.startswith("(") and cleaned.endswith(")")

    if negative:
        cleaned = cleaned[1:-1]

    amount = Decimal(cleaned)

    return -amount if negative else amount


def extract_records(
    text: str,
    company: str,
    document_id: str,
    document: str,
    page: int,
    chunk_id: str
) -> list[FinancialRecord]:

    lines = text.splitlines()
    records = []

    active_metric = None
    active_unit = None

    for line in lines:
        normalized = line.lower().strip()

        # Look for an explicitly labeled metric and unit.
        metric_found = None

        for label, metric in METRIC_ALIASES.items():
            if normalized.startswith(label):
                metric_found = metric
                break

        if metric_found is not None:
            unit_match = re.search(
                r"\(\s*(USD millions|USD billions)\s*\)",
                line,
                re.IGNORECASE
            )

            active_metric = metric_found
            active_unit = (
                unit_match.group(1).lower()
                if unit_match
                else None
            )
            continue

        match = VALUE_PATTERN.match(line)

        if match and active_metric and active_unit:
            year = int(match.group(1))
            value = parse_amount(match.group(2))

            records.append(
                FinancialRecord(
                    document_id=document_id,
                    document=document,
                    page=page,
                    company=company,
                    metric=active_metric,
                    fiscal_year=year,
                    value=value,
                    unit=active_unit,
                    evidence=line.strip(),
                    chunk_id=chunk_id
                )
            )

        elif line.strip():
            # Do not carry a metric across unrelated text.
            active_metric = None
            active_unit = None

    return records
