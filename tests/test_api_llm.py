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
from service.api.app import llm_test

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
