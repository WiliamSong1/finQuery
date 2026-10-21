
import re
from decimal import Decimal, InvalidOperation


def parse_financial_value(value: str):
    """
    Convert financial strings into numeric values.

    Examples:
        "$54,847" -> 54847
        "(1,250)" -> -1250
        "12.5%" -> 12.5
    """

    cleaned = value.strip().replace(",", "").replace("$", "")

    negative = (
        cleaned.startswith("(")
        and cleaned.endswith(")")
    )

    if negative:
        cleaned = cleaned[1:-1]

    cleaned = cleaned.replace("%", "").strip()

    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None

    if not number.is_finite():
        return None

    if negative:
        number = -number

    return float(number)


def extract_financial_values(text: str) -> list[dict]:
    """
    Extract monetary figures and their surrounding context.

    This identifies candidates, not verified financial facts.
    """

    pattern = r"\$?\(?-?\d[\d,]*(?:\.\d+)?\)?%?"

    results = []

    for match in re.finditer(pattern, text):
        raw_value = match.group()

        # Avoid standalone small numbers
        if len(raw_value.replace(",", "")) < 3:
            continue

        numeric_value = parse_financial_value(raw_value)

        if numeric_value is None:
            continue

        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)

        results.append({
            "raw_value": raw_value,
            "numeric_value": numeric_value,
            "context": text[start:end].strip(),
            "start_offset": match.start()
        })

    return results
