"""FastAPI application exposing budgeting capabilities for the economist service."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ..budget import BudgetCategory, CategorySummary, PurchaseItem, budget_manager
from .schemas import (
    BudgetCategoryResponse,
    BudgetResponse,
    BudgetUploadRequest,
    CategorySummaryResponse,
    PurchaseAnalysisResponse,
    PurchaseAllocationResponse,
    PurchaseTableRequest,
)

app = FastAPI(title="AI Economist Budgeting Service", version="1.0.0")


def _build_budget_response() -> BudgetResponse:
    category_objects = budget_manager.categories()
    categories = [
        BudgetCategoryResponse(
            category=category.name,
            limit=category.limit,
            keywords=category.keywords,
        )
        for category in category_objects
    ]
    total_limit = sum(category.limit for category in category_objects)
    return BudgetResponse(categories=categories, total_limit=total_limit)


def _summary_to_schema(summary: CategorySummary) -> CategorySummaryResponse:
    return CategorySummaryResponse(
        category=summary.category,
        required=summary.required,
        available=summary.available,
        remaining=summary.remaining,
        is_enough=summary.is_enough,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/budget", response_model=BudgetResponse)
def upload_budget(request: BudgetUploadRequest) -> BudgetResponse:
    categories = [
        BudgetCategory(name=row.category, limit=row.limit, keywords=row.keywords)
        for row in request.rows
    ]
    budget_manager.load_budget(categories)
    return _build_budget_response()


@app.get("/budget", response_model=BudgetResponse)
def get_budget() -> BudgetResponse:
    if not budget_manager.categories():
        raise HTTPException(status_code=404, detail="Бюджет не загружен")
    return _build_budget_response()


@app.post("/purchases/analyze", response_model=PurchaseAnalysisResponse)
def analyse_purchases(request: PurchaseTableRequest) -> PurchaseAnalysisResponse:
    purchases = [
        PurchaseItem(
            description=row.description,
            amount=row.amount,
            category_hint=row.category_hint,
        )
        for row in request.purchases
    ]
    try:
        result = budget_manager.analyse(purchases)
    except ValueError as exc:  # Budget not loaded
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    budget_summary = [_summary_to_schema(summary) for summary in result.summaries]
    enough = [_summary_to_schema(summary) for summary in result.enough()]
    not_enough = [_summary_to_schema(summary) for summary in result.not_enough()]
    allocations = [
        PurchaseAllocationResponse(
            description=allocation.description,
            amount=allocation.amount,
            category=allocation.category,
            matched_keyword=allocation.matched_keyword,
        )
        for allocation in result.allocations
    ]
    uncategorised = [
        PurchaseAllocationResponse(
            description=allocation.description,
            amount=allocation.amount,
            category=None,
            matched_keyword=None,
        )
        for allocation in result.uncategorised()
    ]

    return PurchaseAnalysisResponse(
        recognised_categories=result.recognised_categories(),
        enough=enough,
        not_enough=not_enough,
        budget_summary=budget_summary,
        uncategorised_purchases=uncategorised,
        allocations=allocations,
    )
