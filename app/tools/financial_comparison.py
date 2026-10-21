
from decimal import Decimal

from app.tools.financial_records import (
    FinancialRecord,
    FinancialComparison
)


def compare_financial_records(
    start: FinancialRecord,
    end: FinancialRecord
) -> FinancialComparison:

    if start.company != end.company:
        raise ValueError("Companies do not match")

    if start.metric != end.metric:
        raise ValueError("Financial metrics do not match")

    if start.unit != end.unit:
        raise ValueError("Financial units do not match")

    if start.fiscal_year >= end.fiscal_year:
        raise ValueError("Invalid year order")

    # Ensure source evidence exists.
    if not start.evidence or not end.evidence:
        raise ValueError("Missing supporting evidence")

    absolute_change = end.value - start.value

    percentage_change = None

    if start.value != 0:
        percentage_change = (
            absolute_change / abs(start.value)
        ) * Decimal("100")

        percentage_change = percentage_change.quantize(
            Decimal("0.01")
        )

    return FinancialComparison(
        company=start.company,
        metric=start.metric,
        start_year=start.fiscal_year,
        end_year=end.fiscal_year,
        start_value=start.value,
        end_value=end.value,
        absolute_change=absolute_change,
        percentage_change=percentage_change,
        unit=start.unit,
        sources=[start, end]
    )
