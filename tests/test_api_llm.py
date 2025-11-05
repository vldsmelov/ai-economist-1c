"""Tests for the /llmtest endpoint."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import pytest
from fastapi import HTTPException

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib

app_module = importlib.import_module("service.api.app")
from service.api.app import llm_analyse_purchases, llm_test

from urllib import error


def test_llmtest_returns_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_: Any, **__: Any):  # noqa: ANN401 - signature required
        class _FakeResponse:
            def __init__(self) -> None:
                self._data = json.dumps({"response": "Я виртуальный ассистент"}).encode("utf-8")

            def read(self) -> bytes:
                return self._data

        class _ContextManager:
            def __enter__(self) -> _FakeResponse:
                return _FakeResponse()

            def __exit__(self, *_: object) -> None:
                return None

        return _ContextManager()

    monkeypatch.setattr(app_module.request, "urlopen", fake_urlopen)

    result = llm_test()

    assert result == {
        "prompt": "Представься",
        "response": "Я виртуальный ассистент",
    }


def test_llmtest_handles_request_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_: Any, **__: Any) -> None:  # noqa: ANN401 - signature required
        raise error.URLError("boom")

    monkeypatch.setattr(app_module.request, "urlopen", fake_urlopen)

    with pytest.raises(HTTPException) as excinfo:
        llm_test()

    assert "Failed to reach LLM service" in str(excinfo.value)


def test_llm_analyse_purchases_returns_json(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        '17.3" Ноутбук ARDOR Gaming RAGE R17-I7ND405 черный': {
            "цена": "152 099,00",
            "количество": "6",
            "сумма": "912 594,00",
        },
        "Холодильник ACELINE B16AMG белый": {
            "цена": "24 699,00",
            "количество": "1",
            "сумма": "24 699,00",
        },
    }

    def fake_urlopen(*_: Any, **__: Any):  # noqa: ANN401 - signature required
        class _FakeResponse:
            def __init__(self) -> None:
                self._data = json.dumps({"response": json.dumps(payload)}).encode("utf-8")

            def read(self) -> bytes:
                return self._data

        class _ContextManager:
            def __enter__(self) -> _FakeResponse:
                return _FakeResponse()

            def __exit__(self, *_: object) -> None:
                return None

        return _ContextManager()

    monkeypatch.setattr(app_module.request, "urlopen", fake_urlopen)

    file_payload = """
Наименование и характеристика Товара
Кол-во
Ед.
изм.
Цена за 1 ед., с НДС 20%, руб.
Сумма,
с НДС 20%, руб.
Страна завода-изготовителя
1
17.3" Ноутбук ARDOR Gaming RAGE R17-I7ND405 черный
6
шт.
152 099,00
912 594,00
Китай
2
Холодильник ACELINE B16AMG белый
1
шт.
24 699,00
24 699,00
ИТОГО:
1 215 616,00

В том числе НДС 20%:
202 602,67
""".encode("utf-8")

    result = llm_analyse_purchases(file_payload)

    assert result == payload


def test_llm_analyse_purchases_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_: Any, **__: Any):  # noqa: ANN401 - signature required
        class _FakeResponse:
            def __init__(self) -> None:
                self._data = json.dumps({"response": "not-json"}).encode("utf-8")

            def read(self) -> bytes:
                return self._data

        class _ContextManager:
            def __enter__(self) -> _FakeResponse:
                return _FakeResponse()

            def __exit__(self, *_: object) -> None:
                return None

        return _ContextManager()

    monkeypatch.setattr(app_module.request, "urlopen", fake_urlopen)

    with pytest.raises(HTTPException) as excinfo:
        llm_analyse_purchases("товар 1, 2 шт".encode("utf-8"))

    assert "invalid JSON" in str(excinfo.value)


def test_llm_analyse_purchases_empty_file() -> None:
    with pytest.raises(HTTPException) as excinfo:
        llm_analyse_purchases(b"   ")

    assert "Файл пуст" in str(excinfo.value)


def test_llm_analyse_purchases_missing_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"товар": {"цена": "100", "количество": "1"}}

    def fake_urlopen(*_: Any, **__: Any):  # noqa: ANN401 - signature required
        class _FakeResponse:
            def __init__(self) -> None:
                self._data = json.dumps({"response": json.dumps(payload)}).encode("utf-8")

            def read(self) -> bytes:
                return self._data

        class _ContextManager:
            def __enter__(self) -> _FakeResponse:
                return _FakeResponse()

            def __exit__(self, *_: object) -> None:
                return None

        return _ContextManager()

    monkeypatch.setattr(app_module.request, "urlopen", fake_urlopen)

    with pytest.raises(HTTPException) as excinfo:
        llm_analyse_purchases("товар, 1 шт".encode("utf-8"))

    assert "incomplete data" in str(excinfo.value)
