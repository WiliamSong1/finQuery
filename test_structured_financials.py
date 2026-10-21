
from app.tools.structured_extractor import extract_records
from app.tools.financial_comparison import compare_financial_records
from decimal import Decimal

sample_text = """
Total operating expenses (USD millions)
2023: $54,847
2024: $57,467
2025: $62,151
"""

records = extract_records(
    text=sample_text,
    company="Apple",
    document_id="test-document-001",
    document="sample_financial_data.pdf",
    page=27,
    chunk_id="test-chunk-001"
)

print("EXTRACTED RECORDS")

for record in records:
    print(record.model_dump(mode="json"))

assert len(records) == 3

records_by_year = {
    record.fiscal_year: record
    for record in records
}

comparison = compare_financial_records(
    start=records_by_year[2023],
    end=records_by_year[2025]
)

print("\nFINANCIAL COMPARISON")
print(comparison.model_dump_json(indent=2))

assert comparison.absolute_change == Decimal("7304")
assert comparison.percentage_change == Decimal("13.32")
assert len(comparison.sources) == 2

print("\nAll structured financial tests passed!")
