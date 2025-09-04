"""
Shared models and data structures for the crypto portfolio tracker.
"""

from enum import Enum
from typing import Dict, Any, List
from dataclasses import dataclass, field


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
