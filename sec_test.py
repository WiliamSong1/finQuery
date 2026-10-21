
from pprint import pprint

from app.tools.sec_comparison import compare_sec_financials


result = compare_sec_financials(
    cik="0000320193",
    metric="operating_expenses",
    start_year=2023,
    end_year=2025
)

pprint(result)

assert result["start_year"] == 2023
assert result["end_year"] == 2025
assert len(result["sources"]) == 2

print("SEC financial comparison test passed!")
