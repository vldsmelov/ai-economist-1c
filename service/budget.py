"""Domain logic for managing budgets and analysing purchase tables."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple


@dataclass
class BudgetCategory:
    """Represents a spending category with an available limit and keywords."""

    name: str
    limit: float
    keywords: List[str] = field(default_factory=list)

    def normalised_name(self) -> str:
        return self.name.strip().lower()

    def normalised_keywords(self) -> List[str]:
        return [keyword.strip().lower() for keyword in self.keywords if keyword.strip()]


@dataclass
class PurchaseItem:
    description: str
    amount: float
    category_hint: Optional[str] = None

    def normalised_description(self) -> str:
        return self.description.lower()

    def normalised_hint(self) -> Optional[str]:
        return self.category_hint.lower() if self.category_hint else None


@dataclass
class PurchaseAllocation:
    description: str
    amount: float
    category: Optional[str]
    matched_keyword: Optional[str]


@dataclass
class CategorySummary:
    category: str
    required: float
    available: float

    @property
    def remaining(self) -> float:
        return self.available - self.required

    @property
    def is_enough(self) -> bool:
        return self.required <= self.available


@dataclass
class AnalysisResult:
    allocations: List[PurchaseAllocation]
    summaries: List[CategorySummary]

    def recognised_categories(self) -> List[str]:
        return [summary.category for summary in self.summaries if summary.required > 0]

    def enough(self) -> List[CategorySummary]:
        return [summary for summary in self.summaries if summary.is_enough and summary.required > 0]

    def not_enough(self) -> List[CategorySummary]:
        return [summary for summary in self.summaries if not summary.is_enough and summary.required > 0]

    def uncategorised(self) -> List[PurchaseAllocation]:
        return [allocation for allocation in self.allocations if allocation.category is None]


class BudgetManager:
    """In-memory manager that keeps the uploaded budget and performs analysis."""

    def __init__(self) -> None:
        self._categories: dict[str, BudgetCategory] = {}

    def load_budget(self, categories: Iterable[BudgetCategory]) -> None:
        combined: dict[str, BudgetCategory] = {}
        for category in categories:
            normalised_name = category.normalised_name()
            if not normalised_name:
                continue
            cleaned_keywords = [keyword.strip() for keyword in category.keywords if keyword and keyword.strip()]
            existing = combined.get(normalised_name)
            if existing is None:
                combined[normalised_name] = BudgetCategory(
                    name=category.name.strip(),
                    limit=max(category.limit, 0.0),
                    keywords=list(dict.fromkeys(cleaned_keywords)),
                )
            else:
                existing.limit = max(existing.limit + max(category.limit, 0.0), 0.0)
                existing_keywords_lower = {keyword.lower(): keyword for keyword in existing.keywords}
                for keyword in cleaned_keywords:
                    lower_keyword = keyword.lower()
                    if lower_keyword not in existing_keywords_lower:
                        existing.keywords.append(keyword)
                        existing_keywords_lower[lower_keyword] = keyword
        self._categories = combined

    def reset(self) -> None:
        self._categories = {}

    def categories(self) -> List[BudgetCategory]:
        return list(self._categories.values())

    def analyse(self, purchases: Iterable[PurchaseItem]) -> AnalysisResult:
        if not self._categories:
            raise ValueError("Budget is not loaded")

        totals: dict[str, float] = {name: 0.0 for name in self._categories}
        allocations: List[PurchaseAllocation] = []
        for purchase in purchases:
            category, keyword = self._match_category(purchase)
            if category is not None:
                totals[category.normalised_name()] += max(purchase.amount, 0.0)
                allocations.append(
                    PurchaseAllocation(
                        description=purchase.description,
                        amount=purchase.amount,
                        category=category.name,
                        matched_keyword=keyword,
                    )
                )
            else:
                allocations.append(
                    PurchaseAllocation(
                        description=purchase.description,
                        amount=purchase.amount,
                        category=None,
                        matched_keyword=None,
                    )
                )

        summaries = [
            CategorySummary(
                category=category.name,
                required=totals.get(category.normalised_name(), 0.0),
                available=category.limit,
            )
            for category in self._categories.values()
        ]
        return AnalysisResult(allocations=allocations, summaries=summaries)

    def _match_category(self, purchase: PurchaseItem) -> Tuple[Optional[BudgetCategory], Optional[str]]:
        if purchase.category_hint:
            hinted = self._categories.get(purchase.normalised_hint() or "")
            if hinted is not None:
                return hinted, "category_hint"

        description = purchase.normalised_description()
        best_match: Optional[BudgetCategory] = None
        matched_keyword: Optional[str] = None
        best_score = -1

        for category in self._categories.values():
            for keyword in category.keywords:
                normalised_keyword = keyword.lower()
                if normalised_keyword and normalised_keyword in description:
                    score = len(normalised_keyword)
                    if score > best_score:
                        best_score = score
                        best_match = category
                        matched_keyword = keyword

        if best_match is not None:
            return best_match, matched_keyword

        for category in self._categories.values():
            category_name = category.normalised_name()
            if category_name and category_name in description:
                return category, category.name

        return None, None


budget_manager = BudgetManager()
"""Module level manager used by the API layer."""
