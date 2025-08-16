from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class Signal:
    side: str  # 'buy' | 'sell' | 'flat'
    strength: float = 1.0

class Strategy(ABC):
    name: str

    @abstractmethod
    def on_bar(self, bar: dict[str, Any]) -> Signal | None:
        ...
