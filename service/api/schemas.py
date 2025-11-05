"""Pydantic schemas for the AI Economist API."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    steps: Optional[int] = Field(default=None, ge=1, description="Number of steps to run")
    width: Optional[int] = Field(default=None, ge=2)
    height: Optional[int] = Field(default=None, ge=2)
    num_agents: Optional[int] = Field(default=None, ge=1)
    resource_density: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    resource_amount: Optional[float] = Field(default=None, gt=0.0)
    seed: Optional[int] = Field(default=None, ge=0)


class StepState(BaseModel):
    step_index: int
    actions: Dict[str, List[int]]
    collected: Dict[str, float]
    taxes: Dict[str, float]
    wealth: Dict[str, float]


class SimulationResponse(BaseModel):
    summary: Dict[str, float]
    steps: List[StepState]
    final_wealth: Dict[str, float]
