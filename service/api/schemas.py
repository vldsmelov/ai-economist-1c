"""Pydantic schemas for the AI Economist budgeting API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class BudgetRow(BaseModel):
    category: str = Field(..., min_length=1, description="Название категории бюджета")
    limit: float = Field(..., ge=0, description="Доступный лимит по категории")
    keywords: List[str] = Field(default_factory=list, description="Ключевые слова для категоризации")

    @field_validator("keywords", mode="before")
    @classmethod
    def _normalise_keywords(cls, value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [keyword.strip() for keyword in value.split(",") if keyword.strip()]
        if isinstance(value, list):
            normalised: List[str] = []
            for keyword in value:
                if isinstance(keyword, str):
                    trimmed = keyword.strip()
                    if trimmed:
                        normalised.append(trimmed)
            return normalised
        raise TypeError("keywords must be a string or a list of strings")


class BudgetUploadRequest(BaseModel):
    rows: List[BudgetRow] = Field(..., min_length=1, description="Строки бюджетной таблицы")


class BudgetCategoryResponse(BaseModel):
    category: str
    limit: float
    keywords: List[str]


class BudgetResponse(BaseModel):
    categories: List[BudgetCategoryResponse]
    total_limit: float


class PurchaseRow(BaseModel):
    description: str = Field(..., min_length=1, description="Описание закупки")
    amount: float = Field(..., ge=0, description="Стоимость закупки")
    category_hint: Optional[str] = Field(default=None, description="Подсказка по категории, если известна")


class PurchaseTableRequest(BaseModel):
    purchases: List[PurchaseRow] = Field(..., min_length=1, description="Таблица с закупками")


class PurchaseAllocationResponse(BaseModel):
    description: str
    amount: float
    category: Optional[str]
    matched_keyword: Optional[str]


class CategorySummaryResponse(BaseModel):
    category: str
    required: float
    available: float
    remaining: float
    is_enough: bool


class PurchaseAnalysisResponse(BaseModel):
    recognised_categories: List[str]
    enough: List[CategorySummaryResponse]
    not_enough: List[CategorySummaryResponse]
    budget_summary: List[CategorySummaryResponse]
    uncategorised_purchases: List[PurchaseAllocationResponse]
    allocations: List[PurchaseAllocationResponse]
