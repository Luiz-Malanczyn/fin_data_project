from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.schemas import InvestmentHistoryRecord


class BaseLoader(ABC):
    @abstractmethod
    def load(self, records: list[InvestmentHistoryRecord]) -> int:
        """Write the records to the final destination. Returns the number of rows written."""
        ...
