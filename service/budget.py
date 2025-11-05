"""Domain logic for managing budgets and analysing purchase tables."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple


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


def _normalise_number(value: str) -> float:
    cleaned = (
        value.replace("\ufeff", "")
        .replace("\xa0", " ")
        .replace("\u202f", " ")
        .replace("\u00ad", "")
        .replace("'", "")
        .strip()
    )
    if not cleaned:
        return 0.0
    cleaned = cleaned.replace(" ", "").replace("−", "-").replace(",", ".")
    if cleaned.endswith("-"):
        cleaned = "-" + cleaned[:-1]
    try:
        return float(Decimal(cleaned))
    except InvalidOperation as exc:  # pragma: no cover - defensive
        raise ValueError(f"Не удалось распознать числовое значение: {value}") from exc


def parse_budget_csv(content: bytes, *, call_llm: Callable[[str], str]) -> List[BudgetCategory]:
    """Use an LLM to extract budget categories and available limits from CSV data."""

    if not content:
        raise ValueError("Файл бюджета пуст")

    text = _decode_budget_content(content)
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Файл бюджета пуст")

    prompt = (
        "Тебе передан бюджет в формате CSV. Найди все строки с товарными категориями и "
        "их доступными лимитами. Верни только валидный JSON-объект без пояснений. Структура "
        "JSON — словарь, где ключом является точное имя категории из таблицы, а значением "
        "строка с числом из столбца 'Доступно' (сохрани исходное форматирование чисел). "
        "Игнорируй строки с итогами и служебную информацию.\n\n"
        f"CSV:\n{cleaned_text}"
    )

    llm_response = call_llm(prompt)

    limits = _parse_llm_budget_response(llm_response)
    if not limits:
        raise ValueError("LLM не вернул данные о категориях бюджета")

    keywords_map = _extract_keywords_from_csv(text)

    categories: List[BudgetCategory] = []
    for name, value in limits.items():
        category_name = name.strip()
        if not category_name:
            continue
        if "итого" in category_name.lower():
            continue
        try:
            limit_value = _normalise_number(str(value))
        except ValueError as exc:
            raise ValueError(
                f"Не удалось обработать значение '{value}' для категории '{category_name}'"
            ) from exc

        normalised_name = category_name.lower()
        keywords = keywords_map.get(normalised_name, [])

        categories.append(
            BudgetCategory(name=category_name, limit=limit_value, keywords=list(keywords))
        )

    if not categories:
        raise ValueError("Не удалось распознать категории бюджета")

    return categories


def _decode_budget_content(content: bytes) -> str:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(
        "Не удалось декодировать файл бюджета" + (f": {last_error}" if last_error else "")
    )


def _parse_llm_budget_response(response: str) -> dict[str, object]:
    try:
        data = json.loads(response)
    except json.JSONDecodeError as exc:  # pragma: no cover - depends on LLM behaviour
        raise ValueError("LLM вернул некорректный JSON") from exc

    extracted: dict[str, object] = {}

    if isinstance(data, dict):
        for key in ("categories", "категории"):
            collection = data.get(key)
            if isinstance(collection, list):
                extracted.update(_extract_from_list(collection))

        for key, value in data.items():
            if not isinstance(key, str):
                continue
            if key.lower() in {"categories", "категории"}:
                continue
            limit_value = _extract_limit_value(value)
            if limit_value is not None:
                extracted[key] = limit_value

    elif isinstance(data, list):
        extracted.update(_extract_from_list(data))

    return extracted


def _extract_from_list(items: Iterable[object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name: Optional[str] = None
        for key in ("name", "category", "категория"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                name = value.strip()
                break
        if not name:
            continue
        limit_value = _extract_limit_value(item)
        if limit_value is not None:
            result[name] = limit_value
    return result


def _extract_limit_value(data: object) -> Optional[object]:
    if isinstance(data, (int, float, str)):
        return data
    if isinstance(data, dict):
        for key in (
            "available",
            "available_limit",
            "availableLimit",
            "limit",
            "доступно",
            "доступный лимит",
            "лимит",
        ):
            if key in data:
                value = data[key]
                if isinstance(value, (int, float, str)):
                    return value
    return None


def _extract_keywords_from_csv(text: str) -> dict[str, List[str]]:
    sample = text[:1024]
    delimiter = ";" if sample.count(";") >= sample.count(",") else ","
    reader = csv.reader(StringIO(text), delimiter=delimiter)

    header: Optional[List[str]] = None
    for row in reader:
        normalised = [cell.strip().lower() for cell in row]
        if any("катег" in cell for cell in normalised):
            header = row
            break

    if header is None:
        return {}

    normalised_header = [cell.strip().lower() for cell in header]
    try:
        category_index = next(i for i, cell in enumerate(normalised_header) if "катег" in cell)
    except StopIteration:  # pragma: no cover - guarded by earlier check
        return {}

    keyword_index = next(
        (i for i, cell in enumerate(normalised_header) if "ключ" in cell),
        None,
    )

    if keyword_index is None:
        return {}

    keywords: dict[str, List[str]] = {}
    for row in reader:
        if len(row) <= max(category_index, keyword_index):
            continue
        category_name = row[category_index].strip()
        if not category_name:
            continue
        if "итого" in category_name.lower():
            continue
        raw_keywords = row[keyword_index].strip() if keyword_index < len(row) else ""
        if not raw_keywords:
            continue
        split_keywords = [keyword.strip() for keyword in raw_keywords.split(",") if keyword.strip()]
        if split_keywords:
            keywords[category_name.lower()] = split_keywords

    return keywords


class BudgetManager:
    """Manager that keeps the uploaded budget and performs analysis."""

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._categories: dict[str, BudgetCategory] = {}
        self._storage_path = storage_path
        if self._storage_path:
            self._load_from_storage()

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
        self._replace_categories(combined, persist=True)

    def reset(self) -> None:
        self._categories = {}
        if self._storage_path:
            try:
                self._storage_path.unlink()
            except FileNotFoundError:
                pass

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


    def _replace_categories(self, categories: dict[str, BudgetCategory], *, persist: bool) -> None:
        self._categories = categories
        if persist and self._storage_path:
            self._save_to_storage()

    def _save_to_storage(self) -> None:
        if not self._storage_path:
            return
        data = [
            {"name": category.name, "limit": category.limit, "keywords": category.keywords}
            for category in self._categories.values()
        ]
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._storage_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def _load_from_storage(self) -> None:
        if not self._storage_path or not self._storage_path.exists():
            return
        try:
            with self._storage_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):  # pragma: no cover - best effort recovery
            return

        categories: List[BudgetCategory] = []
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name", "")).strip()
                if not name:
                    continue
                limit_raw = entry.get("limit", 0.0)
                try:
                    limit = float(limit_raw)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    continue
                raw_keywords = entry.get("keywords")
                if isinstance(raw_keywords, list):
                    keywords = [str(keyword) for keyword in raw_keywords if isinstance(keyword, str) and keyword.strip()]
                else:
                    keywords = []
                categories.append(BudgetCategory(name=name, limit=limit, keywords=keywords))

        combined: dict[str, BudgetCategory] = {}
        for category in categories:
            normalised_name = category.normalised_name()
            if not normalised_name:
                continue
            combined[normalised_name] = category
        if combined:
            self._replace_categories(combined, persist=False)


budget_storage_path = Path(__file__).resolve().parent / "data" / "budget.json"
budget_manager = BudgetManager(storage_path=budget_storage_path)
"""Module level manager used by the API layer."""
