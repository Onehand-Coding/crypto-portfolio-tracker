"""
Shared models and data structures for the crypto portfolio tracker.
"""

from enum import Enum
from typing import Dict, Any, List
from dataclasses import dataclass, field


class TransactionType(Enum):
    """Enum for transaction types to ensure consistency."""
    BUY = "BUY"
    SELL = "SELL"
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    P2P_BUY = "P2P_BUY"
    CONVERT = "CONVERT"
    EARN_REWARD = "EARN_REWARD"
    EARN_SUBSCRIPTION = "EARN_SUBSCRIPTION"
    EARN_REDEMPTION = "EARN_REDEMPTION"
    DIVIDEND = "DIVIDEND"
    STAKING_SUBSCRIBE = "STAKING_SUBSCRIBE"
    STAKING_REDEMPTION = "STAKING_REDEMPTION"
    STAKING_INTEREST = "STAKING_INTEREST"
    TRANSFER = "TRANSFER"
    TRANSFER_IN = "TRANSFER_IN"
    TRANSFER_OUT = "TRANSFER_OUT"

    @classmethod
    def values(cls) -> List[str]:
        return [item.value for item in cls]


@dataclass
class TradeResult:
    """Result of a trade operation."""

    success: bool
    messages: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)


class ExecutionMode(Enum):
    """Execution modes for rebalancing trades."""

    AUTO = "auto"
    BULK = "bulk"
    INTERACTIVE = "interactive"
    CONFIRM = "confirm"
