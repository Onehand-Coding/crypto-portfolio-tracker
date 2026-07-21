from typing import Optional

from pydantic import BaseModel, Field

from api.schemas.system import Staleness


class WalletBalance(BaseModel):
    symbol: str
    quantity: float
    value_usd: Optional[float] = Field(
        None, description="None when the balance could not be priced -- not zero."
    )


class WalletsResponse(BaseModel):
    has_data: bool
    spot_earn_value_usd: float
    futures_value_usd: float
    funding_value_usd: float
    total_value_usd: float
    spot_holdings: list[WalletBalance]
    futures_balances: list[WalletBalance]
    funding_balances: list[WalletBalance]
    staleness: Staleness
