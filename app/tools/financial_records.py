
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, Field


class FinancialRecord(BaseModel):
    document_id: str
    document: str
    page: int = Field(..., ge=1)

    company: str
    metric: str
    fiscal_year: int = Field(..., ge=1900, le=2100)

    value: Decimal
    unit: str

    evidence: str
    chunk_id: str


class FinancialComparison(BaseModel):
    company: str
    metric: str
    start_year: int
    end_year: int

    start_value: Decimal
    end_value: Decimal

    absolute_change: Decimal
    percentage_change: Optional[Decimal] = None

    unit: str
    sources: List[FinancialRecord]
