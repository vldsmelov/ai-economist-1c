import json
import os
import sys

import pytest

try:  # pragma: no cover - optional dependency in tests
    from fastapi.testclient import TestClient
except (ModuleNotFoundError, RuntimeError):  # pragma: no cover - skip tests if dependency missing
    TestClient = None

pytestmark = pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependencies are not installed")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from service.api.app import app
from service.budget import budget_manager


@pytest.fixture(autouse=True)
def reset_budget_manager():
    budget_manager.reset()
    yield
    budget_manager.reset()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _mock_budget_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    response = {
        "Электроника": "2 000 000,00",
        "Крупная бытовая техника": "5 215 000,00",
        "Оргтехника": "600 000,00",
        "Материалы": "103 398,25",
    }

    def _fake_call_llm(prompt: str) -> str:  # pragma: no cover - executed in tests only
        return json.dumps(response, ensure_ascii=False)

    monkeypatch.setattr("service.api.app._call_llm", _fake_call_llm)


def _build_budget_csv() -> str:
    return "\n".join(
        [
            "Период планирования;2025",
            "Организация;ООО \"Пример\"",
            "Счет бюджета;Основной",
            "Товарная категория;Использовано;Корректировка;Согласованный лимит;Зарезервировано;Фактическая оплата;Плановая оплата;Доступно;Ключевые слова",
            "Электроника;0;0;2 000 000,00;0;0;0;2 000 000,00;электроника, ноутбук",
            "Крупная бытовая техника;0;0;5 215 000,00;0;0;0;5 215 000,00;холодильник",
            "Оргтехника;0;0;600 000,00;0;0;0;600 000,00;оргтехника, принтер",
            "Материалы;0;0;103 398,25;0;0;0;103 398,25;",
            "Итого;0;0;7 918 398,25;0;0;0;7 918 398,25;",
        ]
    )


def test_upload_and_get_budget(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    _mock_budget_llm(monkeypatch)
    csv_content = _build_budget_csv()

    response = client.post(
        "/budget",
        files={"file": ("budget.csv", csv_content.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_limit"] == pytest.approx(7_918_398.25)
    assert len(data["categories"]) == 4

    categories = {row["category"]: row for row in data["categories"]}
    assert categories["Электроника"]["limit"] == pytest.approx(2_000_000)
    assert categories["Электроника"]["keywords"] == ["электроника", "ноутбук"]
    assert categories["Материалы"]["limit"] == pytest.approx(103_398.25)
    assert categories["Материалы"]["keywords"] == []

    response_get = client.get("/budget")
    assert response_get.status_code == 200
    data_get = response_get.json()
    assert data_get == data


def test_analyse_purchases(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    _mock_budget_llm(monkeypatch)
    csv_content = _build_budget_csv()
    client.post(
        "/budget",
        files={"file": ("budget.csv", csv_content.encode("utf-8"), "text/csv")},
    )

    purchases_payload = {
        "purchases": [
            {"description": "Закупка электроники и ноутбуков", "amount": 1_200_000},
            {"description": "Поставка холодильников для офиса", "amount": 500_300},
            {"description": "Обслуживание и закупка оргтехники", "amount": 750_000},
            {"description": "Офисная мебель", "amount": 25_000},
        ]
    }

    response = client.post("/purchases/analyze", json=purchases_payload)
    assert response.status_code == 200
    data = response.json()

    assert set(data.keys()) == {
        "Электроника",
        "Крупная бытовая техника",
        "Оргтехника",
        "категория_не_определена",
    }

    electronics = data["Электроника"]
    assert electronics["доступный_бюджет"] == pytest.approx(2_000_000)
    assert electronics["необходимая_сумма"] == pytest.approx(1_200_000)
    assert electronics["достаточно"] == "Да"
    assert "Закупка электроники и ноутбуков" in electronics["товары"]

    appliances = data["Крупная бытовая техника"]
    assert appliances["доступный_бюджет"] == pytest.approx(5_215_000)
    assert appliances["необходимая_сумма"] == pytest.approx(500_300)
    assert appliances["достаточно"] == "Да"
    assert "Поставка холодильников для офиса" in appliances["товары"]

    orgtech = data["Оргтехника"]
    assert orgtech["доступный_бюджет"] == pytest.approx(600_000)
    assert orgtech["необходимая_сумма"] == pytest.approx(750_000)
    assert orgtech["достаточно"] == "Нет"
    assert "Обслуживание и закупка оргтехники" in orgtech["товары"]

    uncategorised = data["категория_не_определена"]
    assert uncategorised["доступный_бюджет"] == "неопределено"
    assert uncategorised["достаточно"] == "неопределено"
    assert uncategorised["необходимая_сумма"] == pytest.approx(25_000)
    assert "Офисная мебель" in uncategorised["товары"]


def test_analyse_without_budget_returns_error(client: TestClient):
    purchases_payload = {
        "purchases": [
            {"description": "Покупка принтера", "amount": 100_000},
        ]
    }

    response = client.post("/purchases/analyze", json=purchases_payload)
    assert response.status_code == 400
    assert "Budget is not loaded" in response.json()["detail"]
