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
    pl_percent: Optional[float] = Field(
        None,
        description="None when basis_usd is zero -- the percentage is undefined, "
                    "not zero. Rendering 0% there would read as 'unchanged' while "
                    "the portfolio is actually up.",
    )


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
    price_unavailable: bool = Field(
        False,
        description="True when the price lookup failed, so value_usd is unknown "
                    "rather than zero. The UI must not collapse these into dust: "
                    "a real position would silently vanish.",
    )


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
    unpriced_count: int = Field(
        0,
        description="Holdings whose price could not be fetched. Non-zero means "
                    "total_value_usd is understated by an unknown amount and "
                    "must be presented with that caveat.",
    )
