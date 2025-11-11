"""FastAPI application exposing budgeting capabilities for the economist service."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple
from urllib import error, request

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from ..budget import (
    BudgetCategory,
    PurchaseItem,
    _normalise_number,
    budget_manager,
    parse_budget_csv,
)
from .schemas import (
    BudgetCategoryResponse,
    BudgetResponse,
    PurchaseAnalysisResponse,
    PurchaseTableRequest,
    SpecificationExtractResponse,
)


@dataclass(slots=True)
class _SpecificationSection:
    content: str
    start: int
    end: int
    begin_anchor: str
    end_anchor: str
    method: str
    notes: list[str]

app = FastAPI(title="AI Economist Budgeting Service", version="1.0.0")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "krith/qwen2.5-32b-instruct:IQ4_XS")
_DEFAULT_OLLAMA_BASE_URLS: Tuple[str, ...] = (
    "http://localhost:11434",
    "http://127.0.0.1:11434",
)
OLLAMA_TIMEOUT = 300.0

UNCATEGORISED_CATEGORY = "категория_не_определена"
UNKNOWN_VALUE = "неопределено"


def _normalise_base_url(url: str) -> str:
    return url.rstrip("/")


def _ollama_base_urls() -> Tuple[str, ...]:
    env_hosts = os.getenv("OLLAMA_HOST", "")
    if env_hosts:
        parts = re.split(r"[,\s]+", env_hosts)
        normalised = tuple(
            _normalise_base_url(part.strip())
            for part in parts
            if part and part.strip()
        )
        if normalised:
            return normalised

    return _DEFAULT_OLLAMA_BASE_URLS


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _iter_ollama_urls(path: str) -> Iterable[str]:
    normalized_path = path if path.startswith("/") else f"/{path}"
    for base in _ollama_base_urls():
        yield f"{base.rstrip('/')}{normalized_path}"


def _ollama_request(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    last_error: error.URLError | None = None
    encoded_payload = json.dumps(payload).encode("utf-8")

    for url in _iter_ollama_urls(path):
        http_request = request.Request(
            url,
            data=encoded_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=OLLAMA_TIMEOUT) as http_response:
                body = http_response.read().decode("utf-8")
                return json.loads(body)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore").strip()
            raise HTTPException(
                status_code=502,
                detail=(
                    f"LLM service returned error ({exc.code}): "
                    f"{detail or exc.reason}"
                ),
            ) from exc
        except error.URLError as exc:
            last_error = exc
            continue

    if last_error is not None:
        reason = last_error.reason
        detail = str(reason) if reason is not None else "connection error"
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach LLM service: {detail}",
        ) from last_error

    raise HTTPException(status_code=502, detail="LLM service is unavailable")


def _call_llm(prompt: str) -> str:
    """Send a prompt to the configured Ollama LLM service and return the reply."""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    data = _ollama_request("/api/chat", payload)

    message = data.get("message") or {}
    llm_response = message.get("content")
    if not llm_response:
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


def _llm_extract_purchases_table(
    document_text: str,
    *,
    context_description: str = "полный текст договора",
) -> tuple[dict[str, dict[str, object]], list]:
    cleaned_content = document_text.strip()
    if not cleaned_content:
        raise HTTPException(status_code=400, detail="Файл пуст или не содержит данных")

    budget_categories = list(budget_manager.categories())
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
        f"Тебе передан {context_description}. Найди в нём раздел со спецификацией закупок, извлеки "
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
        + "\n\nТекст для анализа:\n"
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

    return validated_response, budget_categories


def _summarise_category_records(
    records: Iterable[tuple[str, float, str | None]],
    budget_categories: Iterable[BudgetCategory],
) -> dict[str, dict[str, object]]:
    category_list = list(budget_categories)
    category_limits_by_name = {category.name: category.limit for category in category_list}
    category_limits_by_normalised = {
        category.normalised_name(): category.limit for category in category_list
    }

    summary: dict[str, dict[str, object]] = {}

    for description, amount, category_name in records:
        key = category_name if category_name else UNCATEGORISED_CATEGORY
        bucket = summary.setdefault(
            key,
            {
                "товары": [],
                "доступный_бюджет": None,
                "необходимая_сумма": 0.0,
                "достаточно": UNKNOWN_VALUE,
            },
        )
        bucket["товары"].append(description)
        bucket["необходимая_сумма"] = bucket.get("необходимая_сумма", 0.0) + max(amount, 0.0)

    for key, details in summary.items():
        required_total = round(float(details.get("необходимая_сумма", 0.0)), 2)
        details["необходимая_сумма"] = required_total

        if key == UNCATEGORISED_CATEGORY:
            details["доступный_бюджет"] = UNKNOWN_VALUE
            details["достаточно"] = UNKNOWN_VALUE
            continue

        available_limit = category_limits_by_name.get(key)
        if available_limit is None:
            normalised_key = key.strip().lower()
            available_limit = category_limits_by_normalised.get(normalised_key)

        if available_limit is None:
            details["доступный_бюджет"] = UNKNOWN_VALUE
            details["достаточно"] = UNKNOWN_VALUE
        else:
            available_limit = round(float(available_limit), 2)
            details["доступный_бюджет"] = available_limit
            details["достаточно"] = "Да" if required_total <= available_limit + 1e-6 else "Нет"

    return summary


def _analyse_specification_text(content: str) -> PurchaseAnalysisResponse:
    purchases_data, budget_categories = _llm_extract_purchases_table(
        content,
        context_description="фрагмент спецификации с перечнем закупок",
    )

    records = [
        (
            description,
            _normalise_number(details["сумма"]),
            details.get("категория") or None,
        )
        for description, details in purchases_data.items()
    ]

    summary = _summarise_category_records(records, budget_categories)
    return PurchaseAnalysisResponse.model_validate(summary)


def _extract_specification_section(
    text: str, initial_notes: list[str]
) -> _SpecificationSection:
    notes = list(initial_notes)

    tail_limit = 3000
    if len(text) > tail_limit:
        analyzed_text = text[-tail_limit:]
        offset = len(text) - tail_limit
        notes.append("Анализируем последние 3000 символов документа")
    else:
        analyzed_text = text
        offset = 0

    llm_result = _ollama_find_specification_anchors(analyzed_text)
    if llm_result:
        begin_anchor, end_anchor, reason = llm_result
        if reason:
            notes.append(reason)
        span = _find_span_by_anchors(analyzed_text, begin_anchor, end_anchor)
        if span:
            start_relative, end_relative = span
            start = start_relative + offset
            end = end_relative + offset
            return _SpecificationSection(
                content=text[start:end],
                start=start,
                end=end,
                begin_anchor=begin_anchor,
                end_anchor=end_anchor,
                method="llm",
                notes=notes,
            )
        notes.append("Не удалось сопоставить якоря с исходным текстом")

    fallback = _regex_fallback(analyzed_text)
    if fallback:
        start_relative, end_relative, begin_anchor, end_anchor, method = fallback
        start = start_relative + offset
        end = end_relative + offset
        begin_anchor = text[start : min(end, start + len(begin_anchor))]
        end_anchor_start = max(start, end - len(end_anchor))
        end_anchor = text[end_anchor_start:end]
        return _SpecificationSection(
            content=text[start:end],
            start=start,
            end=end,
            begin_anchor=begin_anchor,
            end_anchor=end_anchor,
            method=method,
            notes=notes,
        )

    raise HTTPException(status_code=404, detail="Не удалось найти спецификацию в документе")


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

    try:
        data = _ollama_request("/api/chat", payload)
    except HTTPException:
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


def _decode_request_bytes(data: bytes) -> str:
    if not data:
        return ""

    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _read_text_from_payload(body: bytes, content_type: str) -> tuple[str, list[str]]:
    if not body:
        raise HTTPException(status_code=400, detail="Документ пуст или не содержит данных")

    lowered_content_type = content_type.lower()
    notes: list[str] = []

    if "multipart/form-data" in lowered_content_type:
        boundary = _extract_boundary(content_type)
        if not boundary:
            raise HTTPException(status_code=400, detail="Не удалось определить границы multipart-формы")
        file_bytes = _parse_multipart_file(body, boundary)
        text = _decode_request_bytes(file_bytes)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Загруженный файл пуст")
        notes.append("Текст получен из загруженного файла")
        return text, notes

    decoded_body = _decode_request_bytes(body)

    if "application/json" in lowered_content_type or not lowered_content_type:
        try:
            payload = json.loads(decoded_body)
        except json.JSONDecodeError:
            if "application/json" in lowered_content_type:
                raise HTTPException(status_code=400, detail="Некорректный JSON в теле запроса") from None
        else:
            text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise HTTPException(status_code=400, detail="Поле 'text' отсутствует или пустое")
            return text, notes

    if decoded_body.strip():
        notes.append("Текст получен напрямую из тела запроса")
        return decoded_body, notes

    raise HTTPException(status_code=400, detail="Документ пуст или не содержит данных")


async def _read_text_from_request(request: Request) -> tuple[str, list[str]]:
    content_type = request.headers.get("content-type", "")
    body = await request.body()
    return _read_text_from_payload(body, content_type)


@app.post(
    "/purchases/specification",
    response_model=SpecificationExtractResponse,
)
async def extract_specification(request: Request) -> SpecificationExtractResponse:
    text, initial_notes = await _read_text_from_request(request)
    section = _extract_specification_section(text, initial_notes)
    analysis = _analyse_specification_text(section.content)

    response_notes = list(section.notes)
    response_notes.append("Спецификация проанализирована относительно бюджета")

    return SpecificationExtractResponse(
        content=section.content,
        start=section.start,
        end=section.end,
        begin_anchor=section.begin_anchor,
        end_anchor=section.end_anchor,
        method=section.method,
        notes="; ".join(response_notes) if response_notes else None,
        analysis=analysis,
    )


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

    validated_response, budget_categories = _llm_extract_purchases_table(cleaned_content)

    records = [
        (
            product_name,
            _normalise_number(data["сумма"]),
            data.get("категория") or None,
        )
        for product_name, data in validated_response.items()
    ]

    summary = _summarise_category_records(records, budget_categories)
    return PurchaseAnalysisResponse.model_validate(summary).model_dump(mode="json")


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


def _analyse_purchase_items(
    purchases: Iterable[PurchaseItem],
) -> PurchaseAnalysisResponse:
    purchase_list = list(purchases)
    budget_categories = budget_manager.categories()

    if not budget_categories:
        raise HTTPException(status_code=400, detail="Budget is not loaded")

    records: list[tuple[str, float, str | None]] = []
    for purchase in purchase_list:
        category_name, _ = budget_manager.categorise(
            purchase.description,
            category_hint=purchase.category_hint,
        )
        records.append((purchase.description, max(purchase.amount, 0.0), category_name))

    summary = _summarise_category_records(records, budget_categories)
    return PurchaseAnalysisResponse.model_validate(summary)


def analyse_purchases(request: PurchaseTableRequest) -> PurchaseAnalysisResponse:
    purchases = [
        PurchaseItem(
            description=row.description,
            amount=row.amount,
            category_hint=row.category_hint,
        )
        for row in request.purchases
    ]

    return _analyse_purchase_items(purchases)


@app.post("/purchases/analyze", response_model=PurchaseAnalysisResponse)
async def analyse_purchases_endpoint(request: Request) -> PurchaseAnalysisResponse:
    content_type = request.headers.get("content-type", "")
    lowered_content_type = content_type.lower()
    body = await request.body()

    if "application/json" in lowered_content_type:
        decoded_body = _decode_request_bytes(body)
        try:
            payload = json.loads(decoded_body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Некорректный JSON в теле запроса") from None

        if isinstance(payload, dict) and "purchases" in payload:
            try:
                table = PurchaseTableRequest.model_validate(payload)
            except ValidationError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=json.loads(exc.json()),
                ) from exc
            return analyse_purchases(table)

    text, notes = _read_text_from_payload(body, content_type)
    section = _extract_specification_section(text, notes)
    return _analyse_specification_text(section.content)
