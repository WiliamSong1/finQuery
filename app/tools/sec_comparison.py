
from decimal import Decimal

from app.tools.sec_client import get_annual_metric


SUPPORTED_METRICS = {
    "operating_expenses": "OperatingExpenses",
    "operating_income": "OperatingIncomeLoss",
    "net_income": "NetIncomeLoss"
}


def compare_sec_financials(
    cik: str,
    metric: str,
    start_year: int,
    end_year: int
) -> dict:

    if metric not in SUPPORTED_METRICS:
        raise ValueError("Unsupported financial metric")

    if start_year >= end_year:
        raise ValueError("Invalid year range")

    sec_tag = SUPPORTED_METRICS[metric]

    start = get_annual_metric(
        cik, sec_tag, start_year
    )

    end = get_annual_metric(
        cik, sec_tag, end_year
    )

    start_value = Decimal(str(start["value"]))
    end_value = Decimal(str(end["value"]))

    change = end_value - start_value

    percentage_change = None

    if start_value != 0:
        percentage_change = (
            change / abs(start_value)
        ) * Decimal("100")

        percentage_change = percentage_change.quantize(
            Decimal("0.01")
        )

    return {
        "company": start["company"],
        "metric": metric,
        "start_year": start_year,
        "end_year": end_year,
        "start_value": str(start_value),
        "end_value": str(end_value),
        "absolute_change": str(change),
        "percentage_change": (
            str(percentage_change)
            if percentage_change is not None
            else None
        ),
        "unit": "USD",
        "sources": [start, end]
    }
