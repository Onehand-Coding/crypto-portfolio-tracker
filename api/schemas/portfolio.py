from typing import Optional

from pydantic import BaseModel, Field

from api.schemas.system import Environment, Staleness


class AccountingBasis(BaseModel):
    """One of the two accounting models. Both are correct; they answer
    different questions, and the UI must never present them as equal."""

    label: str = Field(description="'NET INVESTED BASIS' or 'FIFO BASIS'")
    question: str = Field(description="The plain question this basis answers")
    basis_usd: float = Field(description="Denominator: net in, or cost basis")
    pl_usd: float
    pl_percent: float


class Holding(BaseModel):
    symbol: str
    total_quantity: float
    spot_quantity: Optional[float] = None
    earn_quantity: Optional[float] = None
    current_price: Optional[float] = None
    value_usd: Optional[float] = None
    average_cost_basis: Optional[float] = None
    cost_basis_total: Optional[float] = None
    unrealized_pl_usd: Optional[float] = None
    unrealized_pl_percent: Optional[float] = None
    is_core: bool = False


class CockpitResponse(BaseModel):
    total_value_usd: float
    net_invested: AccountingBasis
    fifo: AccountingBasis
    holdings: list[Holding]
    staleness: Staleness
    environment: Environment
    has_data: bool = Field(
        description="False when no sync has ever run; the UI renders an "
                    "explicit empty state rather than zeros"
    )
