"""Tests for the /purchases/specification endpoint."""

from __future__ import annotations

import importlib
import json
from typing import Any, Generator

import pytest
from fastapi import HTTPException
from starlette.requests import Request

app_module = importlib.import_module("service.api.app")
SpecificationExtractResponse = app_module.SpecificationExtractResponse
PurchaseAnalysisResponse = app_module.PurchaseAnalysisResponse
extract_specification = app_module.extract_specification
analyse_purchases_endpoint = app_module.analyse_purchases_endpoint


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Disable LLM usage in tests to keep them deterministic."""

    def _fake_llm(_: str) -> tuple[str, str, str] | None:  # noqa: ANN401 - FastAPI signature
        return None

    monkeypatch.setattr(app_module, "_ollama_find_specification_anchors", _fake_llm)
    yield


@pytest.fixture
def analysis_stub() -> PurchaseAnalysisResponse:
    return PurchaseAnalysisResponse(
        recognised_categories=[],
        enough=[],
        not_enough=[],
        budget_summary=[],
        uncategorised_purchases=[],
        allocations=[],
    )


@pytest.fixture(autouse=True)
def _patch_spec_analysis(
    monkeypatch: pytest.MonkeyPatch,
    analysis_stub: PurchaseAnalysisResponse,
) -> None:
    monkeypatch.setattr(
        app_module,
        "_analyse_specification_text",
        lambda content: analysis_stub,
    )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


_SPEC_TEXT = """﻿Договор поставки\n\nПриложение № 1\nк Договору\n\nНаименование и характеристика Товара\nКол-во\nЕд.\nизм.\nЦена за 1 ед., с НДС 20%, руб.\nСумма,\nс НДС 20%, руб.\nСтрана завода-изготовителя\n1\n17.3\" Ноутбук ARDOR Gaming RAGE R17-I7ND405 черный\n6\nшт.\n152 099,00\n912 594,00\nКитай\nИТОГО:\n912 594,00\n\nВ том числе НДС 20%:\n152 099,00\n""".strip()


def _build_request(body: bytes, content_type: str) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "headers": [(b"content-type", content_type.encode("latin-1"))],
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


@pytest.mark.anyio
async def test_specification_accepts_json_body(
    analysis_stub: PurchaseAnalysisResponse,
) -> None:
    request = _build_request(
        json.dumps({"text": _SPEC_TEXT}).encode("utf-8"),
        "application/json",
    )

    response = await extract_specification(request)

    assert isinstance(response, SpecificationExtractResponse)
    assert response.method == "regex_fallback"
    assert "Приложение № 1" in response.content
    assert "В том числе НДС" in response.content
    assert response.analysis == analysis_stub


@pytest.mark.anyio
async def test_specification_accepts_file_upload(
    analysis_stub: PurchaseAnalysisResponse,
) -> None:
    file_payload = (
        b"--boundary\r\n"
        b"Content-Disposition: form-data; name=\"file\"; filename=\"spec.txt\"\r\n"
        b"Content-Type: text/plain\r\n\r\n"
        + _SPEC_TEXT.encode("utf-8-sig")
        + b"\r\n--boundary--\r\n"
    )
    request = _build_request(
        file_payload,
        "multipart/form-data; boundary=boundary",
    )

    response = await extract_specification(request)

    assert isinstance(response, SpecificationExtractResponse)
    assert response.method == "regex_fallback"
    assert response.notes is not None
    assert "Текст получен из загруженного файла" in response.notes
    assert response.analysis == analysis_stub


@pytest.mark.anyio
async def test_specification_rejects_empty_payload() -> None:
    request = _build_request(b"", "text/plain")

    with pytest.raises(HTTPException) as excinfo:
        await extract_specification(request)

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "Документ пуст или не содержит данных"


@pytest.mark.anyio
async def test_analyse_endpoint_processes_uploaded_file(
    monkeypatch: pytest.MonkeyPatch,
    analysis_stub: PurchaseAnalysisResponse,
) -> None:
    captured: dict[str, str] = {}

    def _capture(content: str) -> PurchaseAnalysisResponse:
        captured["content"] = content
        return analysis_stub

    monkeypatch.setattr(app_module, "_analyse_specification_text", _capture)

    file_payload = (
        b"--boundary\r\n"
        b"Content-Disposition: form-data; name=\"file\"; filename=\"spec.txt\"\r\n"
        b"Content-Type: text/plain\r\n\r\n"
        + _SPEC_TEXT.encode("utf-8-sig")
        + b"\r\n--boundary--\r\n"
    )
    request = _build_request(
        file_payload,
        "multipart/form-data; boundary=boundary",
    )

    response = await analyse_purchases_endpoint(request)

    assert response == analysis_stub
    assert "Приложение № 1" in captured["content"]
    assert "В том числе НДС" in captured["content"]
