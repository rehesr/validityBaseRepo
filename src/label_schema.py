from typing import List, Optional

from pydantic import BaseModel, Field


class Holding(BaseModel):
    name: str
    yf_ticker: Optional[str] = None
    weight: float
    manager: Optional[str] = None
    fund: Optional[str] = None
    rationale: Optional[str] = None


class Exclusion(BaseModel):
    name: str
    code: Optional[str] = None
    reason: str


class HoldingsExtraction(BaseModel):
    holdings: List[Holding] = Field(default_factory=list)
    exclusions: List[Exclusion] = Field(default_factory=list)
