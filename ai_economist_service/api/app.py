"""FastAPI application exposing the AI Economist simulation."""
from __future__ import annotations

from fastapi import FastAPI

from ..simulation import EpisodeResult, Simulation, SimulationConfig
from .schemas import SimulationRequest, SimulationResponse, StepState

app = FastAPI(title="AI Economist Service", version="0.1.0")


def build_simulation(request: SimulationRequest) -> Simulation:
    config = SimulationConfig()
    if request.width is not None:
        config.width = request.width
    if request.height is not None:
        config.height = request.height
    if request.num_agents is not None:
        config.num_agents = request.num_agents
    if request.resource_density is not None:
        config.resource_density = request.resource_density
    if request.resource_amount is not None:
        config.resource_amount = request.resource_amount
    if request.seed is not None:
        config.seed = request.seed
    if request.steps is not None:
        config.steps = request.steps
    return Simulation(config=config)


def format_episode(result: EpisodeResult) -> SimulationResponse:
    return SimulationResponse(
        summary=result.summary(),
        steps=[
            StepState(
                step_index=step.step_index,
                actions=step.actions,
                collected=step.collected,
                taxes=step.taxes,
                wealth=step.wealth,
            )
            for step in result.steps
        ],
        final_wealth=result.final_wealth,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/simulate", response_model=SimulationResponse)
def simulate(request: SimulationRequest) -> SimulationResponse:
    simulation = build_simulation(request)
    episode = simulation.run_episode()
    return format_episode(episode)
