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


def test_upload_and_get_budget(client: TestClient):
    payload = {
        "rows": [
            {"category": "Электроника", "limit": 2_000_000, "keywords": ["электроника", "ноутбук"]},
            {"category": "Крупная бытовая техника", "limit": 5_215_000, "keywords": ["холодильник"]},
        ]
    }

    response = client.post("/budget", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_limit"] == pytest.approx(7_215_000)
    assert len(data["categories"]) == 2

    response_get = client.get("/budget")
    assert response_get.status_code == 200
    data_get = response_get.json()
    assert data_get == data


def test_analyse_purchases(client: TestClient):
    budget_payload = {
        "rows": [
            {"category": "Электроника", "limit": 2_000_000, "keywords": ["электроника", "ноутбук"]},
            {"category": "Крупная бытовая техника", "limit": 5_215_000, "keywords": ["холодильник", "бытовая техника"]},
            {"category": "Оргтехника", "limit": 600_000, "keywords": ["оргтехника", "принтер"]},
        ]
    }
    client.post("/budget", json=budget_payload)

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

    recognised = data["recognised_categories"]
    assert "Электроника" in recognised
    assert "Крупная бытовая техника" in recognised
    assert "Оргтехника" in recognised

    not_enough = {item["category"]: item for item in data["not_enough"]}
    assert not_enough["Оргтехника"]["is_enough"] is False
    assert not_enough["Оргтехника"]["required"] == pytest.approx(750_000)
    assert not_enough["Оргтехника"]["available"] == pytest.approx(600_000)

    enough = {item["category"]: item for item in data["enough"]}
    assert enough["Электроника"]["is_enough"] is True
    assert enough["Крупная бытовая техника"]["is_enough"] is True

    uncategorised = data["uncategorised_purchases"]
    assert any(item["description"] == "Офисная мебель" for item in uncategorised)

    assert len(data["budget_summary"]) == 3


def test_analyse_without_budget_returns_error(client: TestClient):
    purchases_payload = {
        "purchases": [
            {"description": "Покупка принтера", "amount": 100_000},
        ]
    }

    response = client.post("/purchases/analyze", json=purchases_payload)
    assert response.status_code == 400
    assert "Budget is not loaded" in response.json()["detail"]
