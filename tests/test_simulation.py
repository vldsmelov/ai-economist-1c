import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from service.simulation import Simulation, SimulationConfig

try:
    from fastapi.testclient import TestClient
    from service.api.app import app
except ModuleNotFoundError:  # pragma: no cover - optional dependency for tests
    TestClient = None
    app = None


def test_episode_runs_and_produces_summary():
    config = SimulationConfig(steps=5, num_agents=2, width=4, height=4, seed=42)
    simulation = Simulation(config=config)
    episode = simulation.run_episode()

    summary = episode.summary()
    assert "mean_wealth" in summary
    assert len(episode.steps) == 5
    assert len(episode.final_wealth) == 2


@pytest.mark.skipif(TestClient is None, reason="FastAPI is not installed")
def test_test_endpoint_returns_episode():
    client = TestClient(app)

    response = client.get("/simulate/test")

    assert response.status_code == 200
    payload = response.json()
    assert "summary" in payload
    assert payload["summary"].get("mean_wealth") is not None
