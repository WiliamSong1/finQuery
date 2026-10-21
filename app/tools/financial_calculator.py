
from decimal import Decimal, InvalidOperation


def calculate_change(
    start_value: float,
    end_value: float,
    start_year: int,
    end_year: int
) -> dict:

    if start_year >= end_year:
        raise ValueError(
            "Start year must be earlier than end year"
        )

    try:
        start = Decimal(str(start_value))
        end = Decimal(str(end_value))
    except InvalidOperation:
        raise ValueError("Invalid financial values")

    if not start.is_finite() or not end.is_finite():
        raise ValueError("Values must be finite")

    absolute_change = end - start

    percentage_change = None

    if start != 0:
        percentage_change = (
            absolute_change / abs(start)
        ) * 100

    return {
        "start_year": start_year,
        "end_year": end_year,
        "start_value": float(start),
        "end_value": float(end),
        "absolute_change": float(absolute_change),
        "percentage_change": (
            round(float(percentage_change), 2)
            if percentage_change is not None
            else None
        ),
        "direction": (
            "increased"
            if absolute_change > 0
            else "decreased"
            if absolute_change < 0
            else "unchanged"
        )
    }
