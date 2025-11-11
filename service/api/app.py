"""FastAPI application exposing budgeting capabilities for the economist service."""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple
from urllib import error, request

from fastapi import FastAPI, HTTPException, Request

from ..budget import (
    CategorySummary,
    PurchaseItem,
    _normalise_number,
    budget_manager,
    parse_budget_csv,
)
from .schemas import (
    BudgetCategoryResponse,
    BudgetResponse,
    CategorySummaryResponse,
    PurchaseAnalysisResponse,
    PurchaseAllocationResponse,
    PurchaseTableRequest,
    SpecificationExtractRequest,
    SpecificationExtractResponse,
)

app = FastAPI(title="AI Economist Budgeting Service", version="1.0.0")

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "krith/qwen2.5-32b-instruct:IQ4_XS"
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST.rstrip('/')}/api/chat"


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


def _call_llm(prompt: str) -> str:
    """Send a prompt to the configured Ollama LLM service and return the reply."""

    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}

    encoded_payload = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        OLLAMA_GENERATE_URL,
        data=encoded_payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=300.0) as http_response:
            body = http_response.read().decode("utf-8")
            data = json.loads(body)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(
            status_code=502,
            detail=f"LLM service returned error: {detail or exc.reason}",
        ) from exc
    except error.URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach LLM service: {exc.reason}",
        ) from exc

    llm_response = data.get("response") or data.get("message")
    if not llm_response:
        raise HTTPException(
            status_code=502,
            detail="LLM service did not return a response",
        )

    return llm_response


def _strip_code_fence(text: str) -> str:
    """Remove optional Markdown code fences from an LLM response."""

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    # Remove the opening fence and optional language hint.
    stripped = stripped[3:].lstrip()
    if stripped.lower().startswith("json"):
        stripped = stripped[4:].lstrip()

    closing_index = stripped.rfind("```")
    if closing_index != -1:
        stripped = stripped[:closing_index]

    return stripped.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from arbitrary LLM output."""

    cleaned = _strip_code_fence(text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return parsed

    decoder = json.JSONDecoder()
    idx = 0
    length = len(cleaned)

    while idx < length:
        brace_index = cleaned.find("{", idx)
        if brace_index == -1:
            break

        try:
            parsed_fragment, offset = decoder.raw_decode(cleaned[brace_index:])
        except json.JSONDecodeError:
            idx = brace_index + 1
            continue

        if isinstance(parsed_fragment, dict):
            return parsed_fragment

        idx = brace_index + offset

    raise ValueError("Ответ LLM не содержит корректный JSON-объект")


def _compile_anchor_regex(anchor: str) -> re.Pattern[str]:
    tokens = re.split(r"\s+", anchor.strip())
    escaped_tokens = [re.escape(token) for token in tokens if token]
    pattern = r"\\s*".join(escaped_tokens)
    return re.compile(pattern, flags=re.DOTALL)


def _find_span_by_anchors(
    text: str, begin_anchor: str, end_anchor: str
) -> Optional[Tuple[int, int]]:
    if not begin_anchor or not end_anchor:
        return None

    begin_re = _compile_anchor_regex(begin_anchor)
    end_re = _compile_anchor_regex(end_anchor)

    begin_match = begin_re.search(text)
    if not begin_match:
        return None

    end_match = end_re.search(text, begin_match.end())
    if not end_match:
        return None

    return begin_match.start(), end_match.end()


def _ollama_find_specification_anchors(
    text: str,
) -> Optional[Tuple[str, str, str]]:
    system_prompt = (
        "Ты — аккуратный извлекатель юридических разделов. Тебе передают полный текст "
        "договора на русском. Нужно найти раздел со Спецификацией (он начинается заголовком\n"
        "'Приложение № 1' и содержит таблицу с позициями/ценами, строку 'ИТОГО' и строку\n"
        "'В том числе НДС'). Твоя задача — вернуть ДВА коротких якоря из исходного текста:\n"
        "begin_anchor — первые ~40–160 символов раздела, начиная строго с заголовка;\n"
        "end_anchor — последние ~40–160 символов раздела, заканчивая на строке с НДС.\n"
        "Важно: копируй якоря ПО СИМВОЛАМ как в исходнике (без переформатирования), но они могут быть короче раздела.\n"
        "Верни только JSON без пояснений."
    )

    user_prompt = {
        "instruction": (
            "Найди раздел Спецификации (Приложение № 1) и верни JSON: {\n"
            '  "begin_anchor": string,\n'
            '  "end_anchor": string,\n'
            '  "reason": string\n'
            "}.\n"
            "Требования:\n"
            "- begin_anchor должен начинаться ровно с заголовка 'Приложение' данной Спецификации;\n"
            "- end_anchor должен оканчиваться на строке с фразой 'В том числе НДС' и суммой;\n"
            "- длина каждого якоря 40..160 символов;\n"
            "- никакого текста вне JSON.\n"
        ),
        "text": text,
    }

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1},
    }

    encoded_payload = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        OLLAMA_CHAT_URL,
        data=encoded_payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=300.0) as http_response:
            body = http_response.read().decode("utf-8")
            data = json.loads(body)
    except error.HTTPError:
        return None
    except error.URLError:
        return None

    message = data.get("message") or {}
    content = message.get("content")
    if not content:
        return None

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None

    begin_anchor = str(parsed.get("begin_anchor", "")).strip()
    end_anchor = str(parsed.get("end_anchor", "")).strip()
    reason = str(parsed.get("reason", "")).strip()

    if 40 <= len(begin_anchor) <= 400 and 40 <= len(end_anchor) <= 400:
        return begin_anchor, end_anchor, reason

    return None


def _regex_fallback(
    text: str,
) -> Optional[Tuple[int, int, str, str, str]]:
    start_re = re.compile(r"Приложение\s*№\s*1", re.IGNORECASE | re.DOTALL)
    end_re = re.compile(r"В\s*том\s*числе\s*НДС[^\n]*?(?:\n.+)?", re.IGNORECASE)

    start_match = start_re.search(text)
    if not start_match:
        return None

    end_match = end_re.search(text, start_match.start())
    if not end_match:
        return None

    start = start_match.start()
    end = end_match.end()
    begin_anchor = text[start : min(len(text), start + 140)]
    end_anchor = text[max(start, end - 140) : end]

    return start, end, begin_anchor, end_anchor, "regex_fallback"


@app.get("/llmtest")
def llm_test() -> dict[str, str]:
    """Send a test prompt to the configured Ollama LLM service."""

    prompt = "Представься"
    llm_response = _call_llm(prompt)

    return {"prompt": prompt, "response": llm_response}


def _extract_boundary(content_type: str) -> str | None:
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part.split("=", 1)[1].strip()
            if boundary.startswith("\"") and boundary.endswith("\""):
                boundary = boundary[1:-1]
            return boundary
    return None


def _parse_multipart_file(body: bytes, boundary: str) -> bytes:
    delimiter = f"--{boundary}".encode()
    sections = body.split(delimiter)

    for section in sections:
        section = section.strip(b"\r\n")
        if not section or section in {b"--", b"--\r\n"}:
            continue

        if section.endswith(b"--"):
            section = section[:-2]

        header_sep = b"\r\n\r\n"
        if header_sep not in section:
            header_sep = b"\n\n"

        _, _, data = section.partition(header_sep)
        if not data:
            continue

        return data.rstrip(b"\r\n")

    raise HTTPException(status_code=400, detail="Не удалось прочитать файл из формы")


@app.post(
    "/purchases/specification",
    response_model=SpecificationExtractResponse,
)
def extract_specification(
    request: SpecificationExtractRequest,
) -> SpecificationExtractResponse:
    text = request.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Документ пуст или не содержит данных")

    notes: list[str] = []

    llm_result = _ollama_find_specification_anchors(text)
    if llm_result:
        begin_anchor, end_anchor, reason = llm_result
        if reason:
            notes.append(reason)
        span = _find_span_by_anchors(text, begin_anchor, end_anchor)
        if span:
            start, end = span
            return SpecificationExtractResponse(
                content=text[start:end],
                start=start,
                end=end,
                begin_anchor=begin_anchor,
                end_anchor=end_anchor,
                method="llm",
                notes="; ".join(notes) if notes else None,
            )
        notes.append("Не удалось сопоставить якоря с исходным текстом")

    fallback = _regex_fallback(text)
    if fallback:
        start, end, begin_anchor, end_anchor, method = fallback
        return SpecificationExtractResponse(
            content=text[start:end],
            start=start,
            end=end,
            begin_anchor=begin_anchor,
            end_anchor=end_anchor,
            method=method,
            notes="; ".join(notes) if notes else None,
        )

    raise HTTPException(status_code=404, detail="Не удалось найти спецификацию в документе")


@app.post("/llm/purchases")
async def llm_analyse_purchases(request: Request) -> dict[str, object]:
    """Analyse a purchases file with the LLM and return the JSON result."""

    body = await request.body()

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type.lower():
        boundary = _extract_boundary(content_type)
        if not boundary:
            raise HTTPException(status_code=400, detail="Граница multipart-запроса не найдена")
        file_content = _parse_multipart_file(body, boundary)
    else:
        file_content = body

    if not file_content:
        raise HTTPException(status_code=400, detail="Файл пуст или не содержит данных")

    try:
        content = file_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Файл должен быть в кодировке UTF-8") from exc

    cleaned_content = content.strip()
    if not cleaned_content:
        raise HTTPException(status_code=400, detail="Файл пуст или не содержит данных")

    budget_categories = budget_manager.categories()
    categories_payload = [
        {
            "category": category.name,
            "limit": category.limit,
            "keywords": category.keywords,
        }
        for category in budget_categories
    ]
    categories_json = json.dumps(categories_payload, ensure_ascii=False, indent=2)
    if categories_payload:
        categories_instruction = (
            "Поле \"категория\" заполни точным названием одной из категорий бюджета из списка "
            "ниже (см. значение ключа \"category\"). Подбирай категорию на основе названия товара "
            "и ключевых слов. Если подходящую категорию определить нельзя, используй значение null."
        )
    else:
        categories_instruction = (
            "Категории бюджета не загружены. Для поля \"категория\" используй значение null."
        )

    prompt = (
        "Тебе передан полный текст договора. Найди в нём раздел со спецификацией закупок, извлеки "
        "таблицу с позициями товаров и только её анализируй. Проигнорируй остальные части документа. "
        "Ответ должен содержать только один валидный JSON-объект без дополнительных комментариев, текста "
        "и Markdown-оформления. Каждый ключ — точное наименование товара из таблицы (без порядковых номеров), "
        "а значение — объект с полями \"цена\", \"количество\", \"сумма\", \"категория\". Значения полей \"цена\", "
        "\"количество\" и \"сумма\" должны быть строками и полностью совпадать с исходными данными. "
        "Игнорируй строки с итогами, НДС и прочей служебной информацией. Не заключай ответ в тройные кавычки "
        "или иные разделители. "
        + categories_instruction
        + "\n\nКатегории бюджета (JSON):\n"
        + categories_json
        + "\n\Текст договора:\n"
        + cleaned_content
    )

    llm_response = _call_llm(prompt)

    try:
        parsed_response = _extract_json_object(llm_response)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    known_categories = {
        category.normalised_name(): category.name for category in budget_categories
    }
    category_limits_by_name = {category.name: category.limit for category in budget_categories}
    category_limits_by_normalised = {
        category.normalised_name(): category.limit for category in budget_categories
    }
    validated_response: dict[str, dict[str, object]] = {}
    for product_name, data in parsed_response.items():
        if not isinstance(product_name, str) or not isinstance(data, dict):
            raise HTTPException(
                status_code=502,
                detail="Ответ LLM содержит некорректную структуру данных",
            )

        try:
            price = data["цена"]
            quantity = data["количество"]
            total = data["сумма"]
        except KeyError as exc:
            raise HTTPException(
                status_code=502,
                detail="Ответ LLM содержит неполные данные о покупках",
            ) from exc

        if not all(isinstance(value, str) and value for value in (price, quantity, total)):
            raise HTTPException(
                status_code=502,
                detail="Ответ LLM содержит значения в неверном формате",
            )

        category_value = data.get("категория")
        if category_value is not None and not isinstance(category_value, str):
            raise HTTPException(
                status_code=502,
                detail="Ответ LLM содержит некорректное значение категории",
            )

        category_name: str | None = None
        if isinstance(category_value, str):
            stripped_category = category_value.strip()
            if stripped_category:
                category_name = known_categories.get(
                    stripped_category.lower(), stripped_category
                )

        if category_name is None:
            auto_category, _ = budget_manager.categorise(product_name)
            category_name = auto_category

        validated_response[product_name] = {
            "цена": price,
            "количество": quantity,
            "сумма": total,
            "категория": category_name,
        }

    if not validated_response:
        raise HTTPException(
            status_code=502,
            detail="Ответ LLM не содержит данных о покупках",
        )

    def format_number(value: float | None) -> str:
        if value is None:
            return ""
        rounded = round(value)
        if abs(value - rounded) < 1e-6:
            return str(int(rounded))
        return (f"{value:.2f}").rstrip("0").rstrip(".")

    summary: dict[str, dict[str, object]] = {}
    totals: dict[str, float] = {}

    for product_name, data in validated_response.items():
        category_name = data["категория"]
        category_key = category_name if category_name is not None else "null"
        summary.setdefault(
            category_key,
            {
                "товары": [],
                "доступный_бюджет": "",
                "необходимая_сумма": 0.0,
                "достаточно": "Неизвестно",
            },
        )
        summary[category_key]["товары"].append(product_name)
        totals[category_key] = totals.get(category_key, 0.0) + _normalise_number(
            data["сумма"]
        )

    for category_key, details in summary.items():
        required_total = totals.get(category_key, 0.0)
        category_name = None if category_key == "null" else category_key
        available_limit: float | None = None
        if category_name is not None:
            available_limit = category_limits_by_name.get(category_name)
            if available_limit is None:
                available_limit = category_limits_by_normalised.get(
                    category_name.strip().lower()
                )

        details["необходимая_сумма"] = format_number(required_total)
        details["доступный_бюджет"] = format_number(available_limit)

        if category_name is None or available_limit is None:
            details["достаточно"] = "Неизвестно"
        else:
            details["достаточно"] = (
                "Да" if required_total <= available_limit else "Нет"
            )

    return summary


@app.post("/budget", response_model=BudgetResponse)
async def upload_budget(request: Request) -> BudgetResponse:
    body = await request.body()

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type.lower():
        boundary = _extract_boundary(content_type)
        if not boundary:
            raise HTTPException(status_code=400, detail="Граница multipart-запроса не найдена")
        file_content = _parse_multipart_file(body, boundary)
    else:
        file_content = body

    if not file_content:
        raise HTTPException(status_code=400, detail="Файл пуст или не содержит данных")

    try:
        categories = parse_budget_csv(file_content, call_llm=_call_llm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
