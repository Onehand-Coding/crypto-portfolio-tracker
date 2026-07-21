from pydantic import BaseModel, Field


class CapitalFlowRow(BaseModel):
    source: str
    type: str
    direction: str = Field(description="'in' or 'out'")
    quantity: float
    price_usd: float
    value_usd: float
    provenance: str = Field(
        description="'computed', 'usdt_peg_fallback', or 'failed_lookup'"
    )
    is_suspect: bool = Field(
        description="True when the USD value may not reflect the real rate"
    )


class CapitalFlowResponse(BaseModel):
    rows: list[CapitalFlowRow]
    total_in_usd: float
    total_out_usd: float
    net_invested_usd: float
    suspect_count: int
